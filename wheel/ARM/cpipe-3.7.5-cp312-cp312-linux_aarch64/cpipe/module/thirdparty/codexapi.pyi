def get_token(url, data, host=None, port=None, timeout: int = 1, report_type: str = 'POST', success_code=('code', 1)):
    """
    Get token.
    Args:
        url: (str) The url.
        data: (dict) The data.
        host: (str) The host.
        port: (int) The port.
        timeout: (int) The timeout.
        report_type: (str) The report
        success_code: (tuple) The success code.

    Returns: (tuple) The token value and token timeout.

    """
