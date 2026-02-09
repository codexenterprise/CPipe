import numpy as np

from cpipe.module.logic import Logic
from cpipe.config.config import CLOGER_LEVEL, CLOGER_LEVEL_DEBUG


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

            box_idxes = []
            box_images = []
            frames_stream_names = []
            streamer_names, boxes = new_cdata.get_bboxes()
            for idx_img, one_image_boxes in enumerate(boxes):
                img, m_time = self.get_streamer_frame(cimage=new_cdata.images[streamer_names[idx_img]])
                for idx_box, one_box in enumerate(one_image_boxes):
                    box_images.append(img[int(one_box.box_coord[1]):int(one_box.box_coord[3]), int(one_box.box_coord[0]):int(one_box.box_coord[2])])
                    box_idxes.append([idx_img, idx_box])
                    frames_stream_names.append(streamer_names[idx_img])

            # todo processing data => box_images box_idxes frames_stream_names

            if CLOGER_LEVEL == CLOGER_LEVEL_DEBUG:
                for stream_name in new_cdata.images.keys():
                    # shows eg:
                    # {
                    #     "stream_name": [
                    #         ("text", {"coord": np.array([996, 996], dtype=np.int32), "data": 996, "color": (0, 0, 255), "text_size": 40}),
                    #         ("line", {"coord": np.array([[996, 996], [996, 996]], dtype=np.int32), "color": (0, 0, 255), "thickness": 1}),
                    #         ("polygon", {"coord": np.array([[700, 100], [1820, 980], [1000, 1030], [550, 800]], dtype=np.int32), "color": (0, 0, 255), "thickness": 1, "isclosed": True}),
                    #     ],
                    # }.
                    new_cdata.shows[stream_name] = [
                        ("text", {"coord": np.array([996, 996], dtype=np.int32), "data": "show cpipe", "color": (0, 0, 255), "text_size": 40}),
                        ("line", {"coord": np.array([[996, 996], [996, 996]], dtype=np.int32), "color": (0, 0, 255), "thickness": 1}),
                        ("circle", {"coord": np.array([[996, 996], [996, 996]], dtype=np.int32), "color": (0, 255, 255), "radius": 1}),
                        ("polygon", {"coord": np.array([[700, 100], [1820, 980], [1000, 1030], [550, 800]], dtype=np.int32), "color": (0, 0, 255), "thickness": 1, "isclosed": True}),
                    ]

            self.queue.put_cdata(new_cdata)
