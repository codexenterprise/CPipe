from cpipe.module.model.shufflenet import ShuffleNet, MMShuffleNet
from cpipe.module.node import Node
from cpipe.module.insight import CPipeInsight
from cpipe.module.streamer import VideoStreamer

if __name__ == "__main__":

    stream = VideoStreamer(node_name="stream", stream="rtmp://192.168.8.122:1935/live/7777", process_frame_interval=1)

    xian = MMShuffleNet(node_name="xian",
                    model_path="./models/zhuzi_4.18_dct_batch32.onnx.cpipe",
                    input_size=(3, 96, 96),
                    class_names=[['NG', 'OK']],
                    warmup=True,
                    )

    cp = CPipeInsight(http_insight=True)
    stream += [xian, cp]

    Node.launch()
