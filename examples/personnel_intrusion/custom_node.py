import numpy as np

from cpipe.config.config import CLOGER_LEVEL, CLOGER_LEVEL_DEBUG
from cpipe.module.logic import Logic


class PersonnelIntrusion(Logic):
    def _start(self):

        while True:
            new_cdata = self.get_frames()

            det_stream_names, det_bboxes = new_cdata.get_bboxes()

            # for idx, one_image_boxes in enumerate(det_bboxes):
            #     frame_streamer_name = det_stream_names[idx]
            #     one_image_boxes = det_bboxes[idx]

            # box_idxes = []
            # box_images = []
            # frames_stream_names = []
            # streamer_names, boxes = new_cdata.get_bboxes()
            # for idx_img, one_image_boxes in enumerate(boxes):
            #     img, m_time = self.get_streamer_frame(cimage=new_cdata.images[streamer_names[idx_img]])
            #     for idx_box, one_box in enumerate(one_image_boxes):
            #         box_images.append(img[int(one_box.box_coord[1]):int(one_box.box_coord[3]), int(one_box.box_coord[0]):int(one_box.box_coord[2])])
            #         box_idxes.append([idx_img, idx_box])
            #         frames_stream_names.append(streamer_names[idx_img])

            for idx, stream_name in enumerate(det_stream_names):
                one_image_boxes = det_bboxes[idx]
                if len(one_image_boxes) > 0:
                    new_cdata.shows[stream_name] = [
                        ("text", {"coord": np.array([1920//2 - 100, 80], dtype=np.int32), "data": "There is an intrusion of personnel!", "color": (0, 0, 255), "text_size": 40}),
                    ]

            self.queue.put_cdata(new_cdata)
