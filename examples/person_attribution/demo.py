from cv2 import Tracker
from cpipe.module.model.pphgnet import PPHGNet
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.streamer import VideoStreamer
from cpipe.module.insight import CPipeInsight
from cpipe.module.node import Node

if __name__ == "__main__":
    stream1 = VideoStreamer(stream="/mnt/d/1515644076-1-192.mp4", process_frame_interval=12, once_mode=True)
    detect = YOLOv10(
                    model_path="model_files/yolo26n.engine",
                    valid_class_names=["person"],
                    save_top_n_objects=32,
                    area_flag=False
                    )
    # detect = YOLOv7(
    #                 model_path="model_files/mot_ppyoloe_l_36e_pipeline.onnx",
    #                 class_names=['person'],
    #                 valid_class_names=["person"],
    #                 save_top_n_objects=32,
    #                 area_flag=False
    #                 )
    track = Tracker(
        # model_path="./model_files/bytetrack_s_mot17.onnx",
        secondary_class_names=["person"],
        max_batch_size=1,
        valid_class_names=["person"],
        save_top_n_objects=32,
    )
    person_attribution = PPHGNet(model_path="./model_files/PPHGNet_small_person_attribute_954_infer.onnx", secondary_class_names=["person"])

    cpipeinsight = CPipeInsight(http_insight=True, show_scale=2)

    stream1 += [detect, person_attribution, cpipeinsight]
    Node.launch(check_node=True, auto_restart=False)
