from typing import Annotated, Any
import numpy as np
from pydantic import Field
import torch

from cpipe.module.cinferencer import CClassifier
from cpipe.module.dataprocessing import class_preprocess, load_data, mm_class_preprocess


class MMShuffleNet(CClassifier):
    """
    MMShuffleNet can classify the image, for the model file from codex training platform.
    """
    preprocessor: Annotated[Any, "The preprocessor of the node."] = Field(default=mm_class_preprocess, json_schema_extra={'readOnly': True})
    mean: Annotated[np.ndarray, "The mean of the model."] = Field(default=np.float64(np.asarray([123.675, 116.28, 103.53]).reshape(1, -1)), json_schema_extra={'readOnly': True})
    std: Annotated[np.ndarray, "The std of the model."] = Field(default=1 / np.float64((np.asarray([58.395, 57.12, 57.375])).reshape(1, -1)), json_schema_extra={'readOnly': True})
    need_softmax: Annotated[bool, "Whether to use softmax for post-processing."] = Field(default=False)

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
        batch_imgs, batch_size = load_data(frames, self._device, [self.input_size[1], self.input_size[2]], self.threading_num, preprocessor=self.preprocessor,
                                           area_flag=self.area_flag, area_info=self.area_info, area_info_streamer_names=frames_stream_names, gray_mode=self.gray_mode,
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
