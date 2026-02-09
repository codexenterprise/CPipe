import time
from cpipe.module.logic import Logic
from cpipe.module.node import Node
from cpipe.config.config import CLOGER_LEVEL, CLOGER_LEVEL_DEBUG


class my_node(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mystate = 1
        
    @Node.mcp_tool("add")
    def add(self, a: int, b: int) -> int:
        """
        add two numbers
        Args:
            a: (int) The first number.
            b: (int) The second number.
        Returns: (int) The sum of the two numbers.
        """
        return a + b
    
    @Node.event("event_get_node_state")
    def get_node_state_event(self, data):
        """
        get Node state
        """
        print(f"get_node_state_event: {data}")
        return self.mystate
    
    @Node.mcp_tool("get_node_state")
    def get_node_state(self):
        """
        get Node state, e.g: {"node_state": "alive"}, just return the state
        """
        return f"The current status of the node is: {self.event_send('get_node_state', 5555)}"

    def _start(self):

        while True:
            # set mcp data
            self.mystate += 1

            time.sleep(1)

            # det_stream_names, det_bboxes = new_cdata.get_bboxes()

            # for idx, one_image_boxes in enumerate(det_bboxes):
            #     frame_streamer_name = det_stream_names[idx]
            #     one_image_boxes = det_bboxes[idx]
            #     self.logger.info(frame_streamer_name)
            #     self.logger.info(one_image_boxes)

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

            # # to do processing data => box_images box_idxes frames_stream_names

            # if CLOGER_LEVEL == CLOGER_LEVEL_DEBUG:
            #     for stream_name in new_cdata.images.keys():
            #         # Display relevant information on the result display page of this node.
            #         new_cdata.messages[stream_name] = []
            #         new_cdata.messages[stream_name].append(str(f"1. line {1}"))
            #         new_cdata.messages[stream_name].append(str(f"2. line {3}"))
            #         new_cdata.messages[stream_name].append(str(f"3. line {3}"))

            # self.queue.put_cdata(new_cdata)
