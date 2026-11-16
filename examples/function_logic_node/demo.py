import cpipe as cp
from cpipe.module.model.yolo26 import YOLO26


# One function = one Logic node. It is called once per frame of every streamer;
# the frame is always passed on to the next node afterwards, even if the function raises.
@cp.logic(rois=["area"])  # "area" can be drawn on the web page of this node
def intrusion(frame: cp.Frame):
    persons = [box for box in frame.boxes_of("person") if frame.in_roi(box, "area", anchor="bottom")]
    if persons:
        frame.draw_text(f"有人闯入: {len(persons)} 人")
        frame.message(f"{frame.streamer_name}: intrusion {len(persons)}")
    # Only boxes inside the area are sent to the next nodes.
    frame.boxes = persons


@cp.logic
def big_objects_only(frame: cp.Frame):
    min_area = frame.width * frame.height * 0.01
    frame.boxes = [box for box in frame.boxes
                   if (box.box_coord[2] - box.box_coord[0]) * (box.box_coord[3] - box.box_coord[1]) >= min_area]

# 不用写 while 循环、get_frames、put_cdata。函数每帧调用一次，
# 参数 frame(cp.Frame) 提供 boxes/boxes_of 读取目标、crop 裁图、
# roi/in_roi 判断网页绘制的区域、draw_text/draw_line/draw_polygon 画到结果页、
# message 输出消息；给 frame.boxes 赋值即可过滤传给下游的目标。
# 函数抛异常时只记录日志，不会卡住流水线。
# 把函数放在启动脚本旁 logic/ 目录下、文件名包含 logic 的文件里(如 logic/my_logic.py)，还能在网页编排和 MCP 工具里当作 Logic 节点创建。
if __name__ == "__main__":
    stream = cp.VideoStreamer(node_name="s1", stream="rtsp://admin:tp123456@192.168.10.221:554/Streaming/Channels/101")
    detect = YOLO26(model_path="model_files/yolo26n.onnx", valid_class_names=["person"])
    insight = cp.CPipeInsight(http_insight=True)

    stream += [detect, big_objects_only(), intrusion(node_name="闯入检测"), insight]
    cp.Node.launch(check_node=True, auto_restart=True)
