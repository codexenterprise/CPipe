from cpipe.module.model.fastsam import FastSAM
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":

    video_zhu = "/mnt/d/videos/dengpao/hei/1802121J_20240323080000-20240323081500_1.mp4"

    stream_zhu = VideoStreamer("zhushitu", video_zhu, 3, 1, once_mode=False)

    fsam = FastSAM(
        "fastsam",
        "../../src/model_files/FastSAM-x.onnx",
        3,
        (3, 1024, 1024),
        max_batch_size=1,
        device="cuda:0"
    )


    cpipeinsight = CPipeInsight(http_insight=True)

    stream_zhu += [fsam, cpipeinsight]

    Node.launch(check_node=True)