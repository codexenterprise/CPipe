from _typeshed import Incomplete

class Hikvision:
    base_url: Incomplete
    appKey: Incomplete
    appSecret: Incomplete
    def __init__(self, base_url, appKey, appSecret) -> None:
        """
        Hikvision is a class for getting
        Args:
            base_url: (str) The url of the Hikvision server.
            appKey: (str) The appKey.
            appSecret: (str) The appSecret.
        """
    def headers(self, api_get_address_url):
        """
        Headers.
        Args:
            api_get_address_url: (str) The url.

        Returns: The headers.

        """
    def get_cameras_info(self, api_get_address_url: str = '/artemis/api/resource/v1/cameras'):
        """
        Get all cameras info.
        Args:
            api_get_address_url: (str) The url.

        Returns: The cameras info.

        """
    def get_cameras_area(self, api_get_address_url: str = '/artemis/api/resource/v1/regions'):
        """
        Get all cameras area.
        Args:
            api_get_address_url: (str) The url.

        Returns: The cameras area.

        """
    def get_camera_info(self, index_code, api_get_address_url: str = '/artemis/api/resource/v1/cameras/indexCode'):
        """
        Get camera info.
        Args:
            index_code: (str) The index code.
            api_get_address_url: (str) The url.

        Returns: The camera info.

        """
    def get_history_rtsp(self, index_code, begin_time, end_time, api_get_address_url: str = '/artemis/api/video/v2/cameras/playbackURLs'):
        """
        Get history rtsp.
        Args:
            index_code: (str) The index code.
            begin_time: (str) The begin time.
            end_time: (str) The end time.
            api_get_address_url: (str) The url.

        Returns: The history rtsp.

        """
    def get_rtsp(self, index_code, api_get_address_url: str = '/artemis/api/video/v1/cameras/previewURLs'):
        """
        Get rtsp.
        Args:
            index_code: (str) The index code.
            api_get_address_url: (str) The url.

        Returns: The rtsp.

        """
