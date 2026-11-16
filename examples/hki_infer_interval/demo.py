from cpipe.module.cinferencehook import HKI_InferInterval
from cpipe.module.insight import CPipeInsight
from cpipe.module.model.yolov10 import YOLOv10
from cpipe.module.node import Node
from cpipe.module.streamer import VideoStreamer


if __name__ == "__main__":
    # Streamer still pushes every selected frame to downstream nodes.
    # HKI_InferInterval only controls this algorithm node:
    # if hook_inputs returns None, the current frame is skipped before preprocess/infer.
    stream = VideoStreamer(
        node_name="stream",
        stream="rtsp://admin:tp123456@192.168.10.199:554/Streaming/Channels/101",
        process_frame_interval=1,
    )

    # Infer at most once per 1 second. Counters are kept per streamer_node_name.
    # Exactly one of interval_sec / interval_frames must be set.
    detect = YOLOv10(
        node_name="detect",
        model_path="model_files/yolov10n.onnx",
        valid_class_names=["person"],
        hook_inputs=HKI_InferInterval(interval_sec=1),
        # Infer 1 of every N frames this node receives (not the original video frame index):
        # hook_inputs=HKI_InferInterval(interval_frames=5),
    )

    # Difference vs VideoStreamer.process_frame_interval:
    # - process_frame_interval drops frames at the streamer, so the whole pipeline becomes sparse
    # - HKI_InferInterval only skips inference on this algorithm node; skipped frames still go downstream without new results

    insight = CPipeInsight(http_insight=True)
    stream += [detect, insight]
    Node.launch(check_node=True, auto_restart=False)
