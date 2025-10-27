from _typeshed import Incomplete

class CWebsocket:
    ws: Incomplete
    ws_url: Incomplete
    def __init__(self, ws_url) -> None:
        """
        CWebsocket is a class for sending
        Args:
            ws_url: (str) The url of the websocket server.
        """
    @staticmethod
    def default_dump(obj):
        """
        Convert numpy classes to JSON serializable objects.
        Args:
            obj: The object.

        Returns: The object.
        """
    def connect(self):
        """
        Connect to the websocket server.
        Returns: True if connected, False otherwise.

        """
    def send(self, data, *args) -> None:
        """
        Send data to the websocket server
        Args:
            data:
            *args:

        Returns: None

        """
