from cpipe.module.node import Node
from cpipe.module.reports.feishureport import FeishuReport
from cpipe.module.streamer import VideoStreamer
from examples.report_message_to_feishu.custom_node import my_node

if __name__ == "__main__":

    streamer = VideoStreamer(node_name="streamer007", stream="/mnt/d/RT16000276624788CN_22_RI301.mp4")

    feishu = FeishuReport(app_id="cli_....", app_secret="ST9u....", open_id="ou_9...")
    
    my_node = my_node(feishu=feishu)
    streamer += [my_node, feishu]

    Node.launch(check_node=True)
