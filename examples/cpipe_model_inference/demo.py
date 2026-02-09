import cv2
from cpipe.module.model.yolov8 import YOLOv8obb


if __name__ == "__main__":
    # !!!!!!!!need license!!!!!!!!!!
    # need set SHARE_MEMORY_MODE=True

    ce_det = YOLOv8obb(node_name="zhu_det",
                        model_path="../../project/dengpao_realtime/models/dengpao_zhu_12.20_batch8.engine",
                        input_size=(3, 640, 640),
                        warmup=True,
                        class_names=['开关座', '手', '接线柱红', '接线柱黑', '滑动变阻器', "滑片", '灯泡座', '电压表', '电流表', '电源'],
                        conf_thres=0.5, iou_thres=0.5)


    ce_det.load_model()
    cap = cv2.VideoCapture("/mnt/d/videos/dct/PAPER_20250430130205_20210202_D5.mp4")
    jump_frame_num = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # get cdata format data
        cdata = ce_det([frame], return_cdata_format=True, frames_stream_names=["1"])
        print(cdata)

        # batch mode e.g
        # cdata = ce_det([frame, frame, frame], return_cdata_format=True, frames_stream_names=["1", "2", "3"])
        # print(cdata)


        # just return model inference result
        # ret = ce_det([frame])
        # print(ret)

        # batch mode e.g
        # ret = ce_det([frame, frame, frame])
        # print(ret)



