from cpipe.module.model.tracker.tracker import Tracker
from cpipe.module.model.yolov7 import YOLOv7
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":

    streamer1 = VideoStreamer("streamers","rtmp://192.168.8.122:1935/live/7777", 3, 1)

    chache = YOLOv7("chache",
                       "../../src/dongsheng/dongsheng_huowu_new.engine",
                       3,
                       inputSize=(3, 640, 640),
                       class_names=['materials'],
                       max_batch_size=1,
                       conf_thres=0.4,
                       )

    tk = Tracker("tracker",
                 3,
                 scale_ratio=5,
                 config_tracker={
                                'det_thresh': 0.25,  # 目标检测阈值
                                'max_age': 90,  # 超过5帧没有检测到目标，将目标删除
                                'min_hits': 5,  # 目标出现3帧后才开始跟踪
                                'iou_threshold': 0.01,  # iou阈值
                                'delta_t': 90,  # 时间间隔
                                # 'inertia': 0.01,  # 时间间隔
                                },
                 tacker_type='ocsort',
                 secondary_class_names=['materials'],
                 dump_images=True
                 )


    cpipeinsight = CPipeInsight(http_insight=True, save_video=True)

    streamer1 += [chache, tk, cpipeinsight]

    Node.launch(check_node=True, check_interval=5, auto_restart=False)
