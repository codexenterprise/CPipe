from cpipe.module.cdata import Box

class CHook:
    TYPE_INPUT = "input"
    TYPE_OUTPUT = "output"

    def __init__(self, hook_type: str):
        """
        CHook is the base class for all hooks.
        Args:
            hook_type: (str) CHook.TYPE_INPUT or CHook.TYPE_OUTPUT
        """
        self.hook_type = hook_type

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")


class HKI_CropImage(CHook):
    def __init__(self, crop_factor):
        """
        Crop the image according to the crop factor.
        Args:
            crop_factor: (list) The crop factor of the input image. (h start(0~1), h end(0~1), w start(0~1), w end(0~1)).
        """
        super().__init__(CHook.TYPE_INPUT)

        self.crop_factor = crop_factor
        if self.crop_factor[0] > self.crop_factor[1]:
            raise ValueError("crop_factor[0] must be less than crop_factor[1]")
        if self.crop_factor[0] < 0 or self.crop_factor[1] > 1:
            raise ValueError("crop_factor[0] must be in the range of 0 to 1")
        if self.crop_factor[2] > self.crop_factor[3]:
            raise ValueError("crop_factor[2] must be less than crop_factor[3]")
        if self.crop_factor[2] < 0 or self.crop_factor[3] > 1:
            raise ValueError("crop_factor[2] must be in the range of 0 to 1")

    def __call__(self, image, box: Box = None):
        """
        Crop the image according to the crop factor.
        Args:
            image: (numpy.ndarray) The image to be cropped.
            box: (Box) The box to be cropped. Just used in secondary mode.
        Returns:
            image: (numpy.ndarray) The cropped image. If box is None, dump the image.
        """

        if box is not None:
            tmp_img = image[int(box.box_coord[1]):int(box.box_coord[3]), int(box.box_coord[0]):int(box.box_coord[2])]
            h, w = tmp_img.shape[:2]
            ret_img = tmp_img[int(self.crop_factor[0] * h):int(self.crop_factor[1] * h), int(self.crop_factor[2] * w):int(self.crop_factor[3] * w)]
            if ret_img.shape[0] == 0 or ret_img.shape[1] == 0:
                return None
        else:
            h, w = image.shape[:2]
            ret_img = image[int(self.crop_factor[0] * h):int(self.crop_factor[1] * h), int(self.crop_factor[2] * w):int(self.crop_factor[3] * w)]
            if ret_img.shape[0] == 0 or ret_img.shape[1] == 0:
                return None

        return ret_img


class HKI_DilateImage(CHook):
    def __init__(self, dilate_factor):
        """
        Dilate the image according to the dilate factor. Based on the box.
        Args:
            dilate_factor: (list) The dilate factor of the input image. (h(> 0), w(> 0)).
        """
        super().__init__(CHook.TYPE_INPUT)

        self.dilate_factor = dilate_factor

        if self.dilate_factor[0] < 0 or self.dilate_factor[1] < 0:
            raise ValueError("dilate_factor[0] and dilate_factor[1] must be greater than 1")

    def __call__(self, image, box: Box):
        """
        Dilate the image according to the dilate factor.
        Args:
            image: (numpy.ndarray) The image to be dilated.
            box: (Box) The box to be dilated.
        Returns:
            image: (numpy.ndarray) The dilated image.
        """

        box_h = int(box.box_coord[3]) - int(box.box_coord[1])
        box_w = int(box.box_coord[2]) - int(box.box_coord[0])
        image_h, image_w = image.shape[:2]

        h_start = int(box.box_coord[1] - box_h * self.dilate_factor[0])
        if h_start < 0:
            h_start = 0
        h_end = int(box.box_coord[3] + box_h * self.dilate_factor[0])
        if h_end > image_h:
            h_end = image_h

        w_start = int(box.box_coord[0] - box_w * self.dilate_factor[1])
        if w_start < 0:
            w_start = 0
        w_end = int(box.box_coord[2] + box_w * self.dilate_factor[1])
        if w_end > image_w:
            w_end = image_w

        ret_img = image[h_start:h_end, w_start:w_end]
        if ret_img.shape[0] == 0 or ret_img.shape[1] == 0:
            return None

        return ret_img

class HKO_DumpClass(CHook):
    def __init__(self, class_index, dump_class_names):
        """
        This is a demo of the hook function of the outputs.
        Args:
            class_index: (int) The index of the class in the output.
            dump_class_names: (list) The class names to be dumped.
        """
        super().__init__(CHook.TYPE_OUTPUT)

        self.dump_class_names = dump_class_names
        self.class_index = class_index

    def __call__(self, predictions, frames, model_class_names):
        """
        This is a demo of the hook function of the outputs.
        Args:
            predictions: The output of the model.
            frames: The original image of each batch corresponding to the inference.
            model_class_names: The class names of the model.
        Returns:
            None
        """
        new_predictions = []
        for one_image_predict in predictions:
            new_one_image_predict = []
            for one_box in one_image_predict:
                if model_class_names[int(one_box[self.class_index])] in self.dump_class_names:
                    continue
                new_one_image_predict.append(one_box)
            new_predictions.append(new_one_image_predict)

        return new_predictions

class HKO_ClassNamesThresholdFilter(CHook):
    def __init__(self, class_index, confidence_index, class_names_threshold_dict: dict):
        """
        This is a demo of the hook function of the outputs.
        Args:
            class_index: (int) The index of the class in the output.
            confidence_index: (int) The index of the confidence in the output.
            class_names_threshold_dict: (dict) The class names and the threshold. e.g. {"person": 0.5, "car": 0.3}
        """
        super().__init__(CHook.TYPE_OUTPUT)

        self.class_names_threshold_dict = class_names_threshold_dict
        self.class_names_list = list(self.class_names_threshold_dict.keys())
        self.class_index = class_index
        self.confidence_index = confidence_index

    def __call__(self, predictions, frames, model_class_names):
        """
        This is a demo of the hook function of the outputs.
        Args:
            predictions: The output of the model.
            frames: The original image of each batch corresponding to the inference.
            model_class_names: The class names of the model.
        Returns:
            None
        """
        new_predictions = []
        for one_image_predict in predictions:
            new_one_image_predict = []
            for one_box in one_image_predict:
                if model_class_names[int(one_box[self.class_index])] in self.class_names_list:
                    if one_box[self.confidence_index] < self.class_names_threshold_dict[model_class_names[int(one_box[self.class_index])]]:
                        continue
                new_one_image_predict.append(one_box)
            new_predictions.append(new_one_image_predict)

        return new_predictions