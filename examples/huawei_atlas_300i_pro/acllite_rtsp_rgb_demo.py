"""
ACLLite 版 RTSP 硬解 demo (昇腾 Atlas 300I Pro / Ascend 310P)

链路: RTSP --(PyAV 解封装)--> DVPP VDEC 硬解(NV12, device 显存)
     --(D2H 拷贝)--> host NV12 --> cv2 转 RGB(np.ndarray)

拿到的 rgb 是标准的 H x W x 3 uint8 np.ndarray, 可直接喂给 cpipe 后续节点。

运行前需要设置环境(每个新终端都要, 或写进 ~/.bashrc):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
  # acllite 目录本身要进 PYTHONPATH(扁平导入, 不是当包用):
  export PYTHONPATH=/host/huawei_dvpp_stream/acllite:$PYTHONPATH

依赖: acl(CANN) / acllite(源码) / PyAV(pip install av) / opencv-python / numpy
"""

import time
import acl
import cv2
import numpy as np

# acllite 内部是扁平 import, 所以直接按模块名导入(需 acllite 目录在 PYTHONPATH 中)
from acllite_resource import AclLiteResource
from videocapture import VideoCapture


ACL_MEMCPY_DEVICE_TO_HOST = 2

# 既可以是 RTSP 地址, 也可以是本地文件路径(必须是 H.264/H.265 编码, 如普通 mp4)
# STREAM = "rtsp://192.168.1.100:554/stream"
STREAM = "./2.h264"  # TODO: 改成你的本地文件
DEVICE_ID = 0


def dvpp_image_to_rgb(image):
    """把 DVPP VDEC 输出的 NV12(device 显存) 拷回 host 并转成 RGB np.ndarray。

    Args:
        image: AclLiteImage, 内部是 device 上的 NV12 数据。
    Returns:
        np.ndarray (H, W, 3) uint8, RGB。
    """
    size = image.size
    device_ptr = image.data()

    # DVPP 输出的宽高是对齐后的(宽16对齐、高2对齐), NV12 reshape 必须用对齐宽高
    width = image.width
    height = image.height

    host_ptr, ret = acl.rt.malloc_host(size)
    assert ret == 0, f"malloc_host failed: {ret}"
    try:
        ret = acl.rt.memcpy(host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
        assert ret == 0, f"memcpy D2H failed: {ret}"

        # ptr -> bytes -> numpy。老版本 acl 没有 ptr_to_bytes 时用 ptr_to_numpy 替代
        try:
            raw = acl.util.ptr_to_bytes(host_ptr, size)
            nv12 = np.frombuffer(raw, dtype=np.uint8)
        except AttributeError:
            nv12 = acl.util.ptr_to_numpy(host_ptr, (size,), 1)  # 1 = uint8

        nv12 = nv12.reshape(height * 3 // 2, width)
        rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
        return rgb
    finally:
        acl.rt.free_host(host_ptr)


def main():
    # 1. 初始化 acl 资源(device / context / stream)
    acl_resource = AclLiteResource(device_id=DEVICE_ID)
    acl_resource.init()

    # 2. 打开流/文件, 内部: PyAV 解封装 + DVPP VDEC 硬解
    cap = VideoCapture(STREAM)

    frame_idx = 0
    try:
        while True:
            # read() 可能返回 (SUCCESS, None): 帧还没解出来, 要靠 is_finished() 判断流是否真结束
            ret, image = cap.read()

            if image is None:
                if cap.is_finished():
                    print("stream finished")
                    break
                # 帧还没就绪, 继续轮询
                continue
            start_time = time.time()
            rgb = dvpp_image_to_rgb(image)  # H x W x 3, RGB
            end_time = time.time()
            print(f"frame {frame_idx}: dvpp_image_to_rgb time: {end_time - start_time} seconds")
            frame_idx += 1

            # ==== 这里就是给 cpipe 用的 RGB np.ndarray ====
            # 例如: cimage.malloc(streamer_buffer, node_name, value=rgb[..., ::-1])  # 若下游要 BGR
            print(f"frame {frame_idx}: shape={rgb.shape}, dtype={rgb.dtype}")

            # demo: 存前 3 帧看看效果
            if frame_idx <= 3:
                cv2.imwrite(f"acllite_frame_{frame_idx}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            if frame_idx >= 100:  # demo 跑 100 帧就停, 按需去掉
                break
    finally:
        # acllite 的解码线程是 daemon 线程, 且线程入口 self._decode_thread_entry 持有
        # cap 的引用, 所以 `del cap` 无法触发 __del__/destroy 把线程停掉。这里显式调用
        # destroy() 主动关闭 VDEC 通道/线程。
        for m in ("destroy", "release", "stop", "close"):
            if hasattr(cap, m):
                try:
                    getattr(cap, m)()
                except Exception:
                    pass
                break

        for m in ("release", "destroy", "release_resource"):
            if hasattr(acl_resource, m):
                try:
                    getattr(acl_resource, m)()
                except Exception:
                    pass
                break


if __name__ == "__main__":
    import os
    import sys

    try:
        main()
    finally:
        # acllite 会残留 daemon 解码线程, 正常退出时它可能在 interpreter finalize 阶段
        # 抢 stdout 锁而死锁(core dump)。这里先刷新缓冲再 os._exit 直接终止进程, 绕过
        # 解释器 finalize。注意: 这是 demo 的收尾手段, 接入 cpipe 时要换成正规的线程 join。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
