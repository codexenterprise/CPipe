from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node


if __name__ == "__main__":
    stream1 = VideoStreamer(node_name="ss1", stream="/mnt/d/videos/other/face_2.mp4", process_frame_interval=1)
    detect = YOLOv10(node_name="D1",
                    model_path="/home/zhouhe/workspace/cpipe2.0/src/model_files/yolov10n.onnx",
                    input_size=(3, 640, 640),
                    # class_names=['person', ...],
                    max_batch_size=1,
                    valid_class_names=["person"],
                    save_top_n_objects=32,
                    area_flag=True
                    )

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2)

    stream1 += [detect, cpipeinsight]

    # launch all initialized nodes
    Node.launch(check_node=True, auto_restart=False)
