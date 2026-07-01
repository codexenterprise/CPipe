import time
from cpipe.module.logic import Logic


class my_node(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report = kwargs.get("report", None)

    def _start(self):
        num = 0
        while True:
            new_cdata = self.get_frames()

            # method 1: using json data to send message
            # new_cdata.json["streamer007"] = f"report: Hello, CPipe! {num}"
            # num += 1
            # time.sleep(0.3)

            # method 2: using report instance to send message
            if self.report is not None:
                self.report.send_queue.put("report: Hello, world!")
                time.sleep(0.3)

            self.queue.put_cdata(new_cdata)
