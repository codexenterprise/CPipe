from cpipe.module.node import Node
from cpipe.config.config import CLOGER_LEVEL, CLOGER_LEVEL_DEBUG


class my_node(Node):
    def __init__(self, nodeName, queue_size):
        super().__init__(nodeName, queue_size)

    def _start(self):

        self.ready()

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
                    # Display relevant information on the result display page of this node.
                    new_cdata.messages[stream_name] = []
                    new_cdata.messages[stream_name].append(str(f"1. line {1}"))
                    new_cdata.messages[stream_name].append(str(f"2. line {3}"))
                    new_cdata.messages[stream_name].append(str(f"3. line {3}"))

            self.queue.put_cdata(new_cdata)
