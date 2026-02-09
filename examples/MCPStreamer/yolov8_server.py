from cpipe.module.insight import CPipeInsight
from cpipe.module.model.yolov8 import YOLOv8obb
from cpipe.module.node import Node
from cpipe.module.report import MCPReport
from cpipe.module.streamer import MCPStreamer

if __name__ == "__main__":
    mcpstreamer = MCPStreamer(
        node_name="MCPStreamer",
        mcp_tool_name="inference",
        queue_size=128,
        mcp_transport="streamable-http",
        mcp_host="localhost",
        mcp_port=19967,
        block_mode=True,
    )

    zhu_det = YOLOv8obb(node_name="zhu_det",
                        model_path="/home/zhouhe/workspace/cpipe2.0/project/fuanfa/models/VA_batch.engine",
                        input_size=(3, 640, 640),
                        max_batch_size=1,
                        class_names=['开关座', '手', '接线柱红', '接线柱黑', '滑动变阻器', '滑片', '电压表', '电流表', '电源', '电阻'],
                        conf_thres=0.5, iou_thres=0.5,
                        )

    mcpreport = MCPReport(
        node_name="MCPReport",
        queue_size=128,
    )

    cpipeinsight = CPipeInsight(
        http_insight=True
    )

    mcpstreamer += zhu_det
    zhu_det += mcpreport
    zhu_det += cpipeinsight

    Node.launch()




