from cpipe.module.model.shufflenet import ShuffleNet, MMShuffleNet
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    stream = VideoStreamer("stream", "rtmp://192.168.8.122:1935/live/7777", 3, 1)

    # xian = ShuffleNet("xian",
    #                 "./models/zhuzi_4.18_dct_batch32.om",
    #                 3,
    #                 inputSize=(3, 96, 96),
    #                 class_names=[['NG', 'OK']],
    #                 max_batch_size=32,
    #                 warmup=True,
    #                 )

    # codex 训练平台导出的模型
    xian = MMShuffleNet("xian",
                    "./models/zhuzi_4.18_dct_batch32.onnx.cpipe",
                    3,
                    inputSize=(3, 96, 96),
                    class_names=[['NG', 'OK']],
                    max_batch_size=32,
                    warmup=True,
                    )

    cp = CPipeInsight(http_insight=True)
    stream += [xian, cp]

    Node.launch()
