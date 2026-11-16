import time
from cpipe.module.logic import Logic


class my_node(Logic):

    def _start(self):
        
        flag = True
        while True:
            new_cdata = self.get_frames(block=False)

            if flag:
                # set streamer007 node send queue to my_node or pause my_node
                # format: {"working": ["node name1", ...], "unworking": ["node name2", ...]}
                # working: send queue to my_node    
                # unworking: stop send queue to my_node
                self.__allNodes__["streamer007"].event_send(self.EVENT_SET_UNWORKING_CONSUMER_NAMES, {"working": ["my_node"], "unworking": []})
                flag = False
            else:
                self.__allNodes__["streamer007"].event_send(self.EVENT_SET_UNWORKING_CONSUMER_NAMES, {"working": [], "unworking": ["my_node"]})
                flag = True
            time.sleep(1)
            self.queue.put_cdata(new_cdata)
