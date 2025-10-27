import numpy as np
from _typeshed import Incomplete

class ChineseText:
    _face: Incomplete
    def __init__(self) -> None:
        """
        ChineseText is a class for drawing
        """
    def draw_text(self, image, pos, text, text_size, text_color) -> None:
        """
        Draw chinese(or not) text with ttf
        Args:
            image: (np.ndarray) image
            pos: (tuple) text position
            text: (str) text
            text_size: (int) text size
            text_color: (tuple) text color

        Returns: None
        """
    def draw_string(self, img, x_pos, y_pos, text, color) -> None:
        """
        Draw string with ttf.
        Args:
            img: (np.ndarray) image
            x_pos: (int) x_pos
            y_pos: (int) y_pos
            text: (str) text
            color: (tuple) text color

        Returns: None

        """
    def draw_ft_bitmap(self, img, bitmap, pen, color, h, w) -> None:
        """
        Draw freetype bitmap.
        Args:
            img: (np.ndarray) image
            bitmap: (freetype.Bitmap) bitmap
            pen: (freetype.Vector) pen
            color: (tuple) text color
            h: (int) image height
            w: (int) image width

        Returns: None

        """

class CConfig:
    cpipe_configs: Incomplete
    def __init__(self) -> None:
        """
        CConfig is a class for reading.
        """
    @staticmethod
    def read_config(config_file):
        '''
        Read the config file.
        Args:
            config_file: (str) config file. e.g. "./config/config.yaml"

        Returns: configs

        '''
    @staticmethod
    def save_config(config_file, configs) -> None:
        '''
        Save the config file.
        Args:
            config_file: (str) config file. e.g. "./config/config.yaml"
            configs: (dict) configs

        Returns: None

        '''

class FaceOrientation:
    @classmethod
    def is_front_face(cls, keypoints: np.ndarray):
        """
        side face judgment and head up and down judgment

        Args:
            keypoints: (left eye, right eye, nose, left mouth corner, right mouth corner)

        Returns: orientation, isfront
        """
    @classmethod
    def orthogonal_point(cls, x1, x2, x3):
        """
        known：(x1,y1)、(x2,y2)、(x3,y3)
        solve：a = (ax, ay)
        ∵ x1x2 = (x2-x1,y2-y1) Orthogonal to ax3 = (x3-ax, y3-ay)
        ∴ (x2 - x1)(x3 - ax) + (y2-y1)(y3-ay) = 0
        ∵ a above x1x2, y = λx + b
        ∴ (x2 - x1)(x3 - ax) + (y2-y1)(y3- λax + b) = 0
        ∴ ax = (x3*((x2 - x1)) + y3*(y2-y1) - (y2-y1)*b) / (((x2 - x1)) + (y2-y1)*λ)
        ∴ ay = λ ax + b

        Args:
            x1: (x1, y1)
            x2: (x2, y2)
            x3: (x3, y3)
        Returns (ax, ay)
        """
    @classmethod
    def linear_equation(cls, x1, y1, x2, y2, beg_k: bool = True):
        """
        + beg_k=True：Two points determine a line
        + beg_k=False: Find the intersection of two lines
            a, x3, x4, x5 -> b
            line1：y = k1 * x + c1
            line2：y = k2 * x + c2
            x = (c1 - c2) / (k2 - k1)
            y = k1 * x + c1
        Args:
            beg_k: true Find the intercept and slope for solving the equation，input： x1, y1, x2, y2。false： To find x and y, pass in the intercept and slope of the two equations

        Returns: (k, b) or (x, y)
        """

def cudaSetDevice(device) -> None:
    '''
    Set the device. This function is used to set the device.
    Args:
        device: (str) The device string. e.g. "cuda:0"

    Returns: None

    '''
