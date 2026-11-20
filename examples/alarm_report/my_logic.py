from collections import deque
from cpipe.module.cdata import Alarm
from cpipe.module.logic import Logic
from cpipe.module.logics.rule import CRule


class Camera:
    def __init__(self, vote_x, vote_y):
        """
        Vote X/Y frames to confirm the area state, avoid the alarm jitter caused by the missed detection.
        """
        if vote_x < 1 or vote_y < 1 or vote_x > vote_y:
            raise ValueError(f"vote must be X/Y with 1 <= X <= Y, got {vote_x}/{vote_y}")
        self.vote_x = vote_x
        self.confirmed = "unknown"
        self.window = deque(maxlen=vote_y)
        self.last_track_ids = []

    def _voted(self):
        person_count = sum(self.window)
        empty_count = len(self.window) - person_count
        person_hit = person_count >= self.vote_x
        empty_hit = empty_count >= self.vote_x
        if person_hit and empty_hit:
            return "occupied" if person_count > empty_count else "empty" if empty_count > person_count else None
        return "occupied" if person_hit else "empty" if empty_hit else None

    def observe(self, present):
        """
        Returns: The new state when the confirmed state changes, otherwise None.
        """
        self.window.append(present)
        voted = self._voted()
        if voted is None or voted == self.confirmed:
            return None
        first, self.confirmed = self.confirmed == "unknown", voted
        return None if first else voted


class my_logic(Logic):
    def __init__(self, vote_x=3, vote_y=5, conf_thres=0.25):
        """
        Area occupancy alarm demo, AlarmReport saves the image of the current frame and the video before and after the alarm:
            empty -> occupied: the persons appeared are drawn in red.
            occupied -> empty: the persons just left are drawn in red in the video.
        """
        super().__init__()
        self.vote_x = vote_x
        self.vote_y = vote_y
        self.conf_thres = conf_thres
        self.cameras: dict[str, Camera] = {}

    def _person_boxes(self, stream_name, boxes):
        polygons = [p for p, _, _ in self.cmask_info.get(stream_name, {}).values() if p is not None and len(p) >= 3]
        persons = [box for box in boxes if box.box_class_name == "person" and box.box_score > self.conf_thres]
        if not polygons:
            return persons
        return [box for box in persons if any(CRule.polygons_intersect_with_box(p, box.box_coord) for p in polygons)]

    def _start(self):
        self.get_cmask_info()
        self.cameras = {k: Camera(self.vote_x, self.vote_y) for k in self.banding_streamer_list.keys()}
        while True:
            new_cdata = self.get_frames()
            for stream_name in new_cdata.images.keys():
                camera = self.cameras.setdefault(stream_name, Camera(self.vote_x, self.vote_y))
                persons = self._person_boxes(stream_name, new_cdata.bboxes.get(stream_name, []))

                track_ids = [b.track.track_id for b in persons if b.track is not None and b.track.track_id > 0]
                if track_ids:
                    camera.last_track_ids = track_ids

                state = camera.observe(len(persons) > 0)
                if state is None or not camera.last_track_ids:
                    continue
                if state == "occupied":
                    alarm = Alarm("区域有人", f"区域内出现 {len(persons)} 人", alarm_type="video", track_ids=track_ids or camera.last_track_ids)
                else:
                    alarm = Alarm("区域无人", "区域内人员已全部离开", alarm_type="video", track_ids=camera.last_track_ids)
                new_cdata.add_alarm(stream_name, alarm)
            self.queue.put_cdata(new_cdata)
