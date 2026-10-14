from cpipe.module.model.pphuman import PPHuman
from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolo26 import YOLO26
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":

    stream1 = VideoStreamer(node_name="1", stream="/mnt/d/other/videos/other/face_2.mp4", process_frame_interval=1)
    detect = YOLO26(
                    model_path="model_files/yolo26n.engine",
                    valid_class_names=["person"],
                    conf_thres=0.55,
                    )

    # reid = PPLCNet(node_name="reid",
    #                 model_path="model_files/deepsort_pplcnet.onnx",
    #                 inputSize=(3, 192, 64),
    #                 secondary_class_names=["person"],
    #                 )
    reid = PPHuman(
            model_path="model_files/pphuman_reid_resnet50_256d.engine",
            secondary_class_names=["person"],
        )

    tk = Tracker(node_name="tracker",
                 scale_ratio=1,
                 config_tracker={
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

    Node.launch()
