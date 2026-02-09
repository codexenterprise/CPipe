from cpipe.module.model.fastsam import FastSAM
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":

    video_zhu = "/mnt/d/videos/dengpao/hei/1802121J_20240323080000-20240323081500_1.mp4"

    stream_zhu = VideoStreamer(node_name="zhushitu", stream=video_zhu, process_frame_interval=1, once_mode=False)

    fsam = FastSAM(
        node_name="fastsam",
        model_path="../../src/model_files/FastSAM-x.onnx",
        input_size=(3, 1024, 1024),
        max_batch_size=1,
        device="cuda:0"
    )


    cpipeinsight = CPipeInsight(http_insight=True)

    stream_zhu += [fsam, cpipeinsight]

    Node.launch(check_node=True)