from cpipe.module.clogger import CLogger
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from custom_node import my_node
if __name__ == "__main__":
    def report_log_func(msg):
        log_dict = {
            "type": "algorithmLog",
            "data": [{
                "type": "log_type",
                "algorithmId": "algorithmId",
                "algorithmConfigId": "algorithmConfigId",
                "content": msg
            }]
        }
        return log_dict


    CLogger().init_report("websocket", {"websocket_url": "ws://0.0.0.0:8001"}, report_log_func, ("info", "debug", "report"))

    stream = VideoStreamer("vs1", "./text.mp4", 3, 1)

    mn = my_node("text_logger", 3)

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True)
