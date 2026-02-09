from cpipe.config.config import CLOGER_LEVEL, CLOGER_LEVEL_DEBUG
from cpipe.module.logic import Logic

class my_node(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _start(self):

        while True:
            new_cdata = self.get_frames()

            det_stream_names, det_bboxes = new_cdata.get_bboxes()

            for idx, one_image_boxes in enumerate(det_bboxes):
                frame_streamer_name = det_stream_names[idx]
                one_image_boxes = det_bboxes[idx]
                self.logger.info(frame_streamer_name)
                self.logger.info(one_image_boxes)

            # todo processing data => box_images box_idxes frames_stream_names

            if CLOGER_LEVEL == CLOGER_LEVEL_DEBUG:
                for stream_name in new_cdata.images.keys():
                    # Display relevant information on the result display page of this node.
                    new_cdata.messages[stream_name] = []
                    new_cdata.messages[stream_name].append(str(f"1. line {1}"))
                    new_cdata.messages[stream_name].append(str(f"2. line {3}"))
                    new_cdata.messages[stream_name].append(str(f"3. line {3}"))

            self.queue.put_cdata(new_cdata)
