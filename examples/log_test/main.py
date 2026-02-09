from cpipe.module.clogger import CLogger
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from custom_node import my_node


if __name__ == "__main__":
    # if you need to report logs to the backend, you need to initialize the reporting function
    # def report_log_func(msg):
    #     log_dict = {
    #         "type": "algorithmLog",
    #         "data": [{
    #             "type": "log_type",
    #             "algorithmId": "algorithmId",
    #             "algorithmConfigId": "algorithmConfigId",
    #             "content": msg
    #         }]
    #     }
    #     return log_dict
    # CLogger().init_report("websocket", {"websocket_url": "ws://0.0.0.0:8001"}, report_log_func, ("info", "debug", "report"))

    # if you need to set the log file name mark, you need to set it before all nodes
    # CLogger.set_file_name_mark("996")

    stream = VideoStreamer(node_name="vs1", stream="./text.mp4", process_frame_interval=3)

    mn = my_node(node_name="text_logger", queue_size=3)

    cpipeinsight = CPipeInsight(http_insight=True)

    stream += [mn, cpipeinsight]

    Node.launch(check_node=True, auto_restart=True)
