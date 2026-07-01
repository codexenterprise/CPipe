from cpipe.module.node import Node
from cpipe.module.reports.httpreport import HTTPReport
from cpipe.module.reports.mqttreport import MQTTReport
from cpipe.module.reports.websocketreport import WebsocketReport
from cpipe.module.streamer import VideoStreamer
from examples.report_nodes.custom_node import my_node


def websocket_receive_func(data):
    print(data)
    return None

if __name__ == "__main__":

    streamer = VideoStreamer(node_name="streamer007", stream="/mnt/d/RT16000276624788CN_22_RI301.mp4")

    # http report
    report = HTTPReport(host="0.0.0.0", port=8000, url="http://0.0.0.0:8000/api/v1/report", receive_func=websocket_receive_func)
    # mqtt report
    # report = MQTTReport(broker="test.mosquitto.org", port=1883, qos=1, receive_func=websocket_receive_func)
    # websocket report
    # report = WebsocketReport(ws_url="ws://0.0.0.0:8001", receive_func=websocket_receive_func)

    my_node = my_node(report=report)
    
    streamer += [my_node, report]

    Node.launch(check_node=True)
