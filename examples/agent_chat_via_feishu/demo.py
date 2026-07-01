from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node
from cpipe.module.reports.feishureport import FeishuReport
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    # streamer = VideoStreamer(node_name="streamer007", stream="/mnt/d/RT16000280105080CN_21_2026-02-09_132056_RI303.mp4")

    feishu = FeishuReport(app_id="cli_a9...",
                            app_secret="ST9u...",
                            agent_mode=True, 
                            agent_port=9966, 
                            stt_path="model_files/stt/model.int8.onnx",
                            )
    
    # cpipeinsight = CPipeInsight(http_insight=True, port=9966)

    # streamer += cpipeinsight

    Node.launch(launch_config_path="./cpipe.json", check_node=True)
