"""
MindX SDK (mxVision) 版 RTSP 硬解 demo (昇腾 Atlas 300I Pro / Ascend 310P)

用 pipeline 描述整条链路, mxVision 帮你做多路调度/内存管理:
  mxpi_rtspsrc(拉流+解封装) -> mxpi_videodecoder(DVPP 硬解, 输出 NV12) -> appsink

Python 侧从 appsink 取出 NV12, 再用 cv2 转 RGB(np.ndarray), 喂给 cpipe 后续节点。

依赖:
  - MindX SDK (mxVision) 已安装并 source 环境:
      source ${MX_SDK_HOME}/set_env.sh
    确保 python 能 import StreamManagerApi / MxpiDataType_pb2
  - opencv-python, protobuf

注意: mxVision 不同版本(v2 / v3 / v5) pipeline 插件属性名略有差异, 若解码器报属性错误,
      对照你本机 ${MX_SDK_HOME} 下的插件手册微调 props 即可, 主流程不变。
"""

import json

import cv2
import numpy as np

from StreamManagerApi import StreamManagerApi, StringVector
import MxpiDataType_pb2 as MxpiDataType


RTSP_URL = "rtsp://192.168.1.100:554/stream"  # TODO: 改成你的流地址
DEVICE_ID = 0
STREAM_NAME = b"rtsp_decode"
DECODER_KEY = b"mxpi_videodecoder0"


def build_pipeline():
    """构造 pipeline。inputVideoFormat 按你的流实际编码改: h264 / h265。"""
    pipeline = {
        "rtsp_decode": {
            "stream_config": {"deviceId": str(DEVICE_ID)},
            "mxpi_rtspsrc0": {
                "props": {
                    "rtspUrl": RTSP_URL,
                    "channelId": "0",
                },
                "factory": "mxpi_rtspsrc",
                "next": "mxpi_videodecoder0",
            },
            "mxpi_videodecoder0": {
                "props": {
                    "inputVideoFormat": "h264",          # 或 "h265"
                    "outputImageFormat": "YUV420SP_NV12",  # DVPP 硬解只出 NV12
                    "vdecChannelId": "0",
                },
                "factory": "mxpi_videodecoder",
                "next": "appsink0",
            },
            "appsink0": {
                "factory": "appsink",
            },
        }
    }
    return json.dumps(pipeline).encode()


def vision_to_rgb(vision):
    """MxpiVision(NV12) -> RGB np.ndarray。"""
    info = vision.visionInfo
    data = vision.visionData.dataStr

    # 用对齐后的宽高 reshape NV12
    width_aligned = info.widthAligned
    height_aligned = info.heightAligned

    nv12 = np.frombuffer(data, dtype=np.uint8)
    nv12 = nv12.reshape(height_aligned * 3 // 2, width_aligned)
    rgb = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)

    # 若需要裁掉对齐 padding, 用真实宽高裁一下
    rgb = rgb[: info.height, : info.width]
    return rgb


def main():
    stream_manager = StreamManagerApi()
    if stream_manager.InitManager() != 0:
        raise RuntimeError("InitManager failed")

    if stream_manager.CreateMultipleStreams(build_pipeline()) != 0:
        raise RuntimeError("CreateMultipleStreams failed")

    key_vec = StringVector()
    key_vec.push_back(DECODER_KEY)

    frame_idx = 0
    try:
        while True:
            # 从解码器输出拿一帧(protobuf: MxpiVisionList)
            infer_result = stream_manager.GetProtobuf(STREAM_NAME, 0, key_vec)
            if infer_result.size() == 0 or infer_result[0].errorCode != 0:
                print(f"get frame failed, errorCode={infer_result[0].errorCode if infer_result.size() else 'empty'}")
                break

            vision_list = MxpiDataType.MxpiVisionList()
            vision_list.ParseFromString(infer_result[0].messageBuf)
            vision = vision_list.visionVec[0]

            rgb = vision_to_rgb(vision)  # H x W x 3, RGB
            frame_idx += 1

            # ==== 这里就是给 cpipe 用的 RGB np.ndarray ====
            print(f"frame {frame_idx}: shape={rgb.shape}, dtype={rgb.dtype}")

            if frame_idx <= 3:
                cv2.imwrite(f"mindx_frame_{frame_idx}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        stream_manager.DestroyAllStreams()


if __name__ == "__main__":
    main()
