import numpy as np

from cpipe.module.logic import Logic


class PersonnelIntrusion(Logic):
    def _start(self):

        while True:
            new_cdata = self.get_frames()

            det_stream_names, det_bboxes = new_cdata.get_bboxes()

            for idx, stream_name in enumerate(det_stream_names):
                one_image_boxes = det_bboxes[idx]
                if len(one_image_boxes) > 0:
                    new_cdata.shows[stream_name] = [
                        ("text", {"coord": np.array([1920//2 - 100, 80], dtype=np.int32), "data": "There is an intrusion of personnel!", "color": (0, 0, 255), "text_size": 40}),
                    ]

            self.queue.put_cdata(new_cdata)
