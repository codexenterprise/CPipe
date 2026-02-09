import numpy as np
import torch

from cpipe.module.cinferencer import CClassifier
from cpipe.module.dataprocessing import class_preprocess, load_data, mm_class_preprocess


class MMShuffleNet(CClassifier):
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size=1, conf_thres=0.25,
                 warmup=True, device="cuda:0", threading_num=4, area_flag=False, secondary_class_names=None, input_names=None, output_names=None, gray_mode=False):
        """
        MMShuffleNet is a class for ShuffleNet model.
        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 224, 224]
            class_names: (list) The class names.
            max_batch_size: (int) The max batch size.
            conf_thres: (float) The confidence threshold.
            warmup: (bool) The warmup flag.
            device: (str) The device. e.g. "cuda:0" or "cpu"
            threading_num: (int) The threading number.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input_names.
            output_names: (list) The output_names.
            gray_mode: (bool) Whether to use gray mode.
        """
        super().__init__(nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size, conf_thres, warmup, device, threading_num, area_flag, secondary_class_names, input_names, output_names, gray_mode)
        self.preprocessor = mm_class_preprocess
        self.mean = torch.from_numpy(np.float64(np.asarray([123.675, 116.28, 103.53]).reshape(1, -1)))
        self.std = torch.from_numpy(1 / np.float64((np.asarray([58.395, 57.12, 57.375])).reshape(1, -1)))

    def preprocess(self, frames, *args, **kwargs):
        """
        The preprocess function of the model.
        Args:
            frames: The frames.
            *args: frames_stream_names. eg ['stream1', 'stream2']
            **kwargs: The keyword arguments.

        Returns: The batch_imgs, frames, batch_size

        """
        frames_stream_names = args[0]
        batch_imgs, batch_size = load_data(frames, self._device, [self._inputSize[1], self._inputSize[2]], self._threading_num, preprocessor=self.preprocessor,
                                           area_flag=self._area_flag, area_info=self._area_info, area_info_streamer_names=frames_stream_names, gray_mode=self._gray_mode,
                                           mean=self.mean, std=self.std
                                           )
        return batch_imgs, frames, batch_size

    def infer(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0]: batch_imgs, inputs[1]: original image, inputs[2]: batch_size
            *args:
            **kwargs:

        Returns: The result of the model.

        """
        ret = self.model(inputs[0])
        if type(ret) == list:
            return torch.from_numpy(ret[0])
        return ret
