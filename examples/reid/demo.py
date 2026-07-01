from cpipe.module.model.pplcnet import PPLCNet
from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":

    stream1 = VideoStreamer(node_name="1", stream="/mnt/d/videos/other/face_2.mp4", queue_size=3, process_frame_interval=1, once_mode=True)
    detect = YOLOv10(
                    node_name="detect",
                    queue_size=3,
                    model_path="model_files/yolov10n_batch1.engine",
                    # class_names=['person', ...],
                    valid_class_names=["person"],
                    conf_thres=0.55,
                    inputSize=(3, 640, 640),
                    save_top_n_objects=32,
                    area_flag=True
                    )

    reid = PPLCNet(node_name="reid",
                    queue_size=3,
                    model_path="model_files/deepsort_pplcnet.onnx",
                    inputSize=(3, 192, 64),
                    secondary_class_names=["person"],
                    device="cuda:0")

    tk = Tracker(node_name="tracker",
                 queue_size=3,
                 scale_ratio=1,
                 config_tracker={
                    'input_size': [64, 192], # 输入尺寸
                    'min_box_area': 0,
                    'vertical_ratio': -1,
                    'budget': 100,
                    'max_age': 70,
                    'n_init': 3,
                    'metric_type': 'cosine',
                    'matching_threshold': 0.5,
                    'max_iou_distance': 0.5,
                    'motion': 'KalmanFilter'
                },
                 tacker_type='deepsort_reid',
                 secondary_class_names=['person'],
                 dump_images=True
                 )

    cpipeinsight = CPipeInsight(http_insight=True)
    stream1 += [detect, reid, tk, cpipeinsight]

    Node.launch(check_node=True, check_interval=5, auto_restart=False)
