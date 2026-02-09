# cython:language_level=3
import numpy as np
import torch

from cpipe.module.cmodel import Cmodel
from cpipe.module.dataprocessing import preprocess_yolov7, scale_coords, preprocess_yolov7_rknn
from cpipe.module.cinferencer import CDetector


class YOLOv7(CDetector):
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, valid_class_names=None, max_batch_size=1, conf_thres=0.25, iou_thres=0.45, anchor=None,
                 warmup=True, device="cuda:0", threading_num=4, save_top_n_objects=None, area_flag=False, secondary_class_names=None, input_names=None, output_names=None, gray_mode=False):
        """
        YOLOv7 is a class for YOLOv7 model.

        Args:
            nodeName: (str) The name of the node.
            modelPath: (str) The path of the model.
            queue_size: (int) The queue size.
            inputSize: (list) The input size. e.g. [3, 416, 416]
            class_names: (list) The class names.
            valid_class_names: (list) The valid class names.
            max_batch_size: (int) The max batch size.
            conf_thres: (float) The confidence threshold.
            iou_thres: (float) The iou threshold.
            anchor: (list) The anchor. e.g. np.array([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0]).reshape(3, -1, 2).tolist()
            warmup: (bool) The warmup flag.
            device: (str) The device.
            threading_num: (int) The threading number.
            save_top_n_objects: (int) The save top n objects.
            area_flag: (bool) The area flag.
            secondary_class_names: (list) The secondary class names.
            input_names: (list) The input names.
            output_names: (list) The output names.
            gray_mode: (bool) Whether to use gray mode.

        Returns: None

        """

        super().__init__(nodeName, modelPath, queue_size, inputSize, class_names, valid_class_names, max_batch_size, conf_thres, iou_thres,
                         warmup, device, threading_num, save_top_n_objects, area_flag, secondary_class_names, input_names, output_names, gray_mode)

        self.preprocessor = preprocess_yolov7
        self.anchor = anchor

        if Cmodel.get_model_type(modelPath) == Cmodel.MODEL_TYPE_RKNN:
            if self._max_batch_size > 1:
                self.logger.warning("rknn model only support batch size 1")
            self.preprocessor = preprocess_yolov7_rknn
            self.infer = self.infer_onnx
            if not self.anchor:
                raise Exception("anchor is required for rknn model")

        if Cmodel.get_model_type(modelPath) == Cmodel.MODEL_TYPE_ONNX:
            self.infer = self.infer_onnx
            if not self.anchor:
                raise Exception("anchor is required for onnx model")

    def infer(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0] is pre_imgs, inputs[1] is origin_imgs
            *args:  frames_stream_names. eg. ['stream1', 'stream2']
            **kwargs:

        Returns:

        """
        pre_imgs = inputs[0]
        origin_imgs = inputs[1]
        frames_stream_names = args[0]
        num_dets, det_boxes, det_scores, det_classes = self.model(pre_imgs)

        # get valid class idx
        if self.valid_class_idx is not None:
            save_boxes_bools = ~torch.isin(det_classes, self.valid_class_idx)
            det_scores[save_boxes_bools] = 0.0

        det_boxes = det_boxes.cpu().numpy()
        num_dets = num_dets.cpu().numpy()
        det_classes = det_classes.cpu().numpy()
        det_scores = det_scores.cpu().numpy()
        boxes_list = []
        for i in range(det_boxes.shape[0]):
            if num_dets[i] > 0:
                bx_ = det_boxes[i, 0:int(num_dets[i]), :]
                sc_ = det_scores[i, 0:int(num_dets[i])]
                cls_ = det_classes[i, 0:int(num_dets[i])]
                save_idxes = sc_ > self.conf_threshold
                bx = bx_[save_idxes]
                sc = sc_[save_idxes]
                cls = cls_[save_idxes]
                if self._area_flag and frames_stream_names[i] in self._area_info.keys():
                    img_shape = self._area_info[frames_stream_names[i]][1].shape[:2]
                    bx = scale_coords([self._inputSize[2], self._inputSize[1]], bx, img_shape)
                else:
                    bx = scale_coords([self._inputSize[2], self._inputSize[1]], bx, origin_imgs[i].shape[:2])
                bxs = np.concatenate((bx, sc.reshape(-1, 1), cls.reshape(-1, 1)), axis=1)
                if self.save_top_n_objects is not None:
                    boxes_list.append(bxs[:self.save_top_n_objects])
                else:
                    boxes_list.append(bxs)
            else:
                boxes_list.append(np.zeros((0, 6)))
        return boxes_list

    def box_process(self, position, anchors):
        grid_h, grid_w = position.shape[3:5]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=2)
        stride = np.array([self._inputSize[2] // grid_h, self._inputSize[1] // grid_w]).reshape(1, 1, 2, 1, 1)

        col = col.repeat(len(anchors), axis=1)
        row = row.repeat(len(anchors), axis=1)
        anchors = np.array(anchors)
        anchors = anchors.reshape(1, *anchors.shape, 1, 1)

        box_xy = position[:, :, :2, :, :] * 2 - 0.5
        box_wh = pow(position[:, :, 2:4, :, :] * 2, 2) * anchors

        box_xy += grid
        box_xy *= stride
        box = np.concatenate((box_xy, box_wh), axis=2)

        # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
        xyxy = np.copy(box)
        xyxy[:, :, 0, :, :] = box[:, :, 0, :, :] - box[:, :, 2, :, :] / 2  # top left x
        xyxy[:, :, 1, :, :] = box[:, :, 1, :, :] - box[:, :, 3, :, :] / 2  # top left y
        xyxy[:, :, 2, :, :] = box[:, :, 0, :, :] + box[:, :, 2, :, :] / 2  # bottom right x
        xyxy[:, :, 3, :, :] = box[:, :, 1, :, :] + box[:, :, 3, :, :] / 2  # bottom right y

        return xyxy

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

        # Arguments
            boxes: ndarray, boxes of objects.
            box_confidences: ndarray, confidences of objects.
            box_class_probs: ndarray, class_probs of objects.

        # Returns
            boxes: ndarray, filtered boxes.
            classes: ndarray, classes for boxes.
            scores: ndarray, scores for boxes.
        """
        batch_size = boxes.shape[0]
        boxes = boxes.reshape(batch_size, -1, 4)
        box_confidences = box_confidences.reshape(batch_size, -1)
        box_class_probs = box_class_probs.reshape(batch_size, -1, box_class_probs.shape[-1])

        result_boxes = []
        result_box_confidences = []
        result_box_class_probs = []
        for i in range(batch_size):
            _box_pos = np.where(box_confidences[i] >= self.conf_threshold)
            # boxes = boxes[i][_box_pos]
            tmp_boxes = boxes[i][_box_pos]
            tmp_box_confidences = box_confidences[i][_box_pos]
            tmp_box_class_probs = box_class_probs[i][_box_pos]

            # box_confidences = box_confidences[_box_pos]
            # box_class_probs = box_class_probs[_box_pos]

            class_max_score = np.max(tmp_box_class_probs, axis=-1)
            classes = np.argmax(tmp_box_class_probs, axis=-1)
            _class_pos = np.where(class_max_score * tmp_box_confidences >= self.conf_threshold)

            tmp_boxes = tmp_boxes[_class_pos]
            classes = classes[_class_pos]
            scores = (class_max_score * tmp_box_confidences)[_class_pos]
            result_boxes.append(tmp_boxes)
            result_box_confidences.append(classes)
            result_box_class_probs.append(scores)

        return result_boxes, result_box_confidences, result_box_class_probs

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.

        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iou_threshold)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def post_process(self, input_data, anchors):
        batch_size = input_data[0].shape[0]
        boxes, scores, classes_conf = [], [], []
        input_data = [_in.reshape([batch_size, len(anchors[0]), -1] + list(_in.shape[-2:])) for _in in input_data]
        for i in range(len(input_data)):
            boxes.append(self.box_process(input_data[i][:, :, :4, :, :], anchors[i]))
            scores.append(input_data[i][:, :, 4:5, :, :])
            classes_conf.append(input_data[i][:, :, 5:, :, :])

        def sp_flatten(_in):
            ch = _in.shape[2]
            _in = _in.transpose(0, 1, 3, 4, 2)
            return _in.reshape(_in.shape[0], -1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes, axis=1)
        classes_conf = np.concatenate(classes_conf, axis=1)
        scores = np.concatenate(scores, axis=1)

        # filter according to threshold
        batch_boxes, batch_classes, batch_scores = self.filter_boxes(boxes, scores, classes_conf)

        result_boxes = []
        result_classes = []
        result_scores = []
        # nms
        for b_idx in range(batch_size):
            nboxes, nclasses, nscores = [], [], []
            boxes = batch_boxes[b_idx]
            classes = batch_classes[b_idx]
            scores = batch_scores[b_idx]
            valid_classes = self.valid_class_idx if len(self.valid_class_idx) > 0 else classes
            for c in set(valid_classes):
                inds = np.where(classes == c)
                b = boxes[inds]
                c = classes[inds]
                s = scores[inds]
                keep = self.nms_boxes(b, s)

                if len(keep) != 0:
                    nboxes.append(b[keep])
                    nclasses.append(c[keep])
                    nscores.append(s[keep])

            if not nclasses and not nscores:
                result_boxes.append(np.zeros((0, 4)))
                result_classes.append(np.zeros((0,)))
                result_scores.append(np.zeros((0,)))
            else:
                result_boxes.append(np.concatenate(nboxes))
                result_classes.append(np.concatenate(nclasses))
                result_scores.append(np.concatenate(nscores))

        return result_boxes, result_classes, result_scores

    def infer_onnx(self, inputs, *args, **kwargs):
        """
        The infer function of the model
        Args:
            inputs: inputs[0] is pre_imgs, inputs[1] is origin_imgs
            *args:  frames_stream_names. eg. ['stream1', 'stream2']
            **kwargs:

        Returns:

        """
        pre_imgs = inputs[0]
        origin_imgs = inputs[1]
        frames_stream_names = args[0]
        outputs = self.model(pre_imgs.cpu().numpy())

        det_boxes, det_classes, det_scores = self.post_process(outputs, self.anchor)

        boxes_list = []
        for i in range(len(det_boxes)):
            if det_boxes[i].shape[0] > 0:
                bx_ = det_boxes[i][:, :]
                sc_ = det_scores[i]
                cls_ = det_classes[i]
                save_idxes = sc_ > self.conf_threshold
                bx = bx_[save_idxes]
                sc = sc_[save_idxes]
                cls = cls_[save_idxes]
                if self._area_flag and frames_stream_names[i] in self._area_info.keys():
                    img_shape = self._area_info[frames_stream_names[i]][1].shape[:2]
                    bx = scale_coords([self._inputSize[2], self._inputSize[1]], bx, img_shape)
                else:
                    bx = scale_coords([self._inputSize[2], self._inputSize[1]], bx, origin_imgs[i].shape[:2])
                bxs = np.concatenate((bx, sc.reshape(-1, 1), cls.reshape(-1, 1)), axis=1)
                if self.save_top_n_objects is not None:
                    boxes_list.append(bxs[:self.save_top_n_objects])
                else:
                    boxes_list.append(bxs)
            else:
                boxes_list.append(np.zeros((0, 6)))
        return boxes_list
