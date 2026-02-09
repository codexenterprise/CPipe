from cpipe.module.model.mmsegmentation import MMSemanticSegmentation
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
if __name__ == "__main__":

    vs = VideoStreamer("1", "examples/segmentation/1.mp4", 3)
    mmseg = MMSemanticSegmentation("MMSemanticSegmentation",
                                       "examples/segmentation/ja_model.onnx",
                                       3,
                                       (3, 512, 910),
                                       class_names="vernier",
                                       max_batch_size=1)
    cp = CPipeInsight(http_insight=True)

    vs += [mmseg, cp]
    Node.launch()