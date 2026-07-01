import time
from cpipe.module.logic import Logic


class my_node(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feishu = kwargs.get("feishu", None)

    def _start(self):
        num = 0
        while True:
            new_cdata = self.get_frames()

            # method 1: using json data to send message
            # new_cdata.json["streamer007"] = f"feishu: Hello, CPipe! {num}"
            # num += 1
            # time.sleep(0.3)

            # method 2: using feishu instance to send message
            if self.feishu is not None:
                self.feishu.send_queue.put("feishu: Hello, world!")
                time.sleep(0.3)

            self.queue.put_cdata(new_cdata)
