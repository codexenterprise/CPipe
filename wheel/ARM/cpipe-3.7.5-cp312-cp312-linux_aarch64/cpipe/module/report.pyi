from _typeshed import Incomplete
from cpipe.module.node import Node as Node

class Report(Node):
    def __init__(self, nodeName, queue_size: int = 0, dump_time: float = 86400.0) -> None:
        """
        Report is the base class for the report node.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
            dump_time: (float) The time to dump the data.
        """
    def _start(self) -> None:
        """
        Start the report node.
        Returns: None

        """
    def send(self, data) -> None:
        """
        Send the data. This function should be implemented in the subclass.
        Args:
            data: The data to send.

        Returns:

        """

class MCPReport(Report):
    mcp_streamer_node: Incomplete
    def __init__(self, nodeName, queue_size: int = 128) -> None:
        """
        MCPReport is a class for sending data through MCP. It must be used in conjunction with MCPStreamer.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue. 
        """
    def _start(self) -> None:
        """
        Start the report node.
        Returns: None

        """
    def send(self, data) -> None:
        """
        Send the data.
        Args:
            data: Contains the streamer name, image and bboxes.

        Returns: None

        """

class HTTPReport(Report):
    conn: Incomplete
    repeat_time: Incomplete
    timeout_base: Incomplete
    success_code: Incomplete
    report_type: Incomplete
    url: Incomplete
    host: Incomplete
    port: Incomplete
    headers: Incomplete
    def __init__(self, nodeName, queue_size, host, port, url, report_type: str = 'POST', repeat_time: int = 3, timeout_base: int = 3, success_code=('code', 1)) -> None:
        '''
        HTTPReport is a class for sending.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
            host: (str) The host of the server.
            port: (int) The port of the server.
            url: (str) The url of the server.
            report_type: (str) The type of the report. It can be "POST", "PUT" or "GET".
            repeat_time: (int) The repeat time.
            timeout_base: (int) The base timeout.
            success_code: (tuple) The success code.

        '''
    def send(self, data, token=None):
        """
        Send the data.
        Args:
            data: The data to send.
            token: The token.

        Returns: The code

        """

class WebsocketReport(Report):
    ws: Incomplete
    ws_url: Incomplete
    def __init__(self, nodeName, queue_size, ws_url) -> None:
        """
        WebsocketReport is a class for sending data through websocket.
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
            ws_url: (str) The url of the websocket server.
        """
    @staticmethod
    def default_dump(obj):
        """
        Convert numpy classes to JSON serializable objects.
        Args:
            obj: The object to convert.

        Returns: The converted object.

        """
    def connect(self):
        """
        Connect to the websocket server.
        Returns: True if connected, False otherwise.

        """
    def send(self, data, *args) -> None:
        """
        Send the data.
        Args:
            data: The data to send.
            *args: The arguments.

        Returns: None

        """

class MQTTReport(Report):
    broker: Incomplete
    port: Incomplete
    username: Incomplete
    password: Incomplete
    client: Incomplete
    def __init__(self, nodeName, queue_size, username, password, broker, port: int = 1883) -> None:
        """
        MQTTReport is a class for sending
        Args:
            nodeName: (str) The name of the node.
            queue_size: (int) The size of the queue.
            username: (str) The username of the MQTT server.
            password: (str) The password of the MQTT server.
            broker: (str) The broker of the MQTT server.
            port: (int) The port of the MQTT server.

        """
    def connect(self) -> None:
        """
        Connect to the MQTT server.
        Returns: None

        """
    def send(self, topic, *args) -> None:
        """
        Send the data.
        Args:
            topic: The topic of the message.
            *args: The arguments.

        Returns: None

        """
    def disconnect(self) -> None:
        """
        Disconnect from the MQTT server.
        Returns: None

        """
    def _start(self) -> None:
        """
        Start the report node.
        Returns: None

        """
