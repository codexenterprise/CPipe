from cpipe.module.insight import CPipeInsight
from cpipe.module.model.adaface import Adaface
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.node import Node
from cpipe.module.report import MCPReport
from cpipe.module.streamer import MCPStreamer


if __name__ == "__main__":
    mcpstreamer1 = MCPStreamer(
        node_name="MCPStreamer1",
        mcp_tool_name="face_server1",
        queue_size=8,
        mcp_transport="streamable-http",
        mcp_host="0.0.0.0",
        mcp_port=19966,
        # block_mode=True,
    )

    rf = Retinaface(
        model_path="/home/zhouhe/workspace/cpipe2.0/src/model_files/416x416-det_10g_batch.engine",
        queue_size=7,
        input_size=(3, 416, 416),
        max_batch_size=16,
    )

    ada = Adaface(
        model_path="/home/zhouhe/workspace/cpipe2.0/src/model_files/adaface_ir101_webface12m.engine",
        queue_size=6,
        input_size=(3, 112, 112),
        max_batch_size=16,
        secondary_class_names=["face"],
        # face_quality_model_path="src/model_files/face_quality_batch64_GPU3070.engine",
    )

    mcpreport = MCPReport(
        node_name="MCPReport",
        queue_size=5,
    )

    cpipeinsight = CPipeInsight(http_insight=True
    )

    mcpstreamer1 += [rf, ada]
    ada += mcpreport
    ada += cpipeinsight

    Node.launch()




