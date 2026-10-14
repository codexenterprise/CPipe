"""
验证 AscendVideoCapture 是否能当 cv2.VideoCapture 用。

部署方式(在昇腾服务器上):
  1. 把 ascend_videocapture.py 和本文件放到同一目录(如 /host/workspace)
  2. 设好环境:
       source /usr/local/Ascend/ascend-toolkit/set_env.sh
       export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
       export PYTHONPATH=/host/workspace/acllite:$PYTHONPATH
  3. python test_ascend_videocapture.py
"""

import os
import sys

import cv2

from ascend_videocapture import AscendVideoCapture

STREAM = "./2.h264"  # RTSP 地址或 Annex-B 裸流文件


def main():
    # 用法跟 cv2.VideoCapture 一模一样
    cap = AscendVideoCapture(STREAM, device_id=0, color="BGR")

    print("isOpened:", cap.isOpened())
    print("fps   :", cap.get(cv2.CAP_PROP_FPS))
    print("width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("count :", cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("stream end or timeout")
            break
        idx += 1
        if idx <= 3:
            cv2.imwrite(f"cap_frame_{idx}.jpg", frame)  # BGR, 直接 imwrite 即可
            print(f"frame {idx}: shape={frame.shape}, dtype={frame.dtype}, pos={cap.get(cv2.CAP_PROP_POS_FRAMES)}")
        if idx >= 100:
            break

    cap.release()
    print(f"done, total {idx} frames")


if __name__ == "__main__":
    try:
        main()
    finally:
        # acllite 残留 daemon 线程, demo 里强退避免 interpreter finalize 死锁
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


