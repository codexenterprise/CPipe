from cpipe.module.insight import CPipeInsight
from cpipe.module.model.adaface import Adaface
from cpipe.module.model.retinaface import Retinaface
from cpipe.module.node import Node
from cpipe.module.report import MCPReport
from cpipe.module.streamer import MCPStreamer


if __name__ == "__main__":
    mcpstreamer = MCPStreamer(
        nodeName="MCPStreamer",
        mcp_tool_name="inference",
        queue_size=8,
        mcp_transport="streamable-http",
        mcp_host="0.0.0.0",
        mcp_port=19966,
        # block_mode=True,
    )

    rf = Retinaface(
        "retinaface",
        "src/model_files/416x416-det_10g_batch.engine",
        7,
        (3, 416, 416),
        max_batch_size=16,
    )

    ada = Adaface(
        "adaface",
        "src/model_files/adaface_ir101_webface12m_batch64_GPU3070.engine",
        6,
        [3, 112, 112],
        max_batch_size=16,
        secondary_class_names=["face"],
        # face_quality_model_path="src/model_files/face_quality_batch64_GPU3070.engine",
    )

    mcpreport = MCPReport(
        nodeName="MCPReport",
        queue_size=5,
    )

    cpipeinsight = CPipeInsight(
        nodeName="CPipeInsight", http_insight=True
    )

    mcpstreamer += [rf, ada]
    ada += mcpreport
    ada += cpipeinsight

    Node.launch()




