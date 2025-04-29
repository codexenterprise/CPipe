from cpipe.module.model.shufflenet import ShuffleNet


class ShuffleNetCut(ShuffleNet):
    def __init__(self, nodeName, modelPath, queue_size, inputSize, class_names, max_batch_size=1, conf_thres=0.25,
                 warmup=True, device="cuda:0", threading_num=4, area_flag=False, secondary_class_names=None, input_names=None, output_names=None, gray_mode=False):
        """
        ShuffleNet is a class for Shuffle
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

    def _start_secondary(self):
        """
        The entry function for the current node (process) to run in two-stage mode, all the logic processing of the current node is completed here.
        Returns: None

        """
        self._loadModel(self._modelPath)

        self.get_streamer_area()
        self.before_start()
        self.ready()
        while True:
            new_cdata = self.get_frames()

            box_idxes = []
            box_images = []
            frames_stream_names = []
            streamer_names, boxes = new_cdata.get_bboxes()
            for idx_img, one_image_boxes in enumerate(boxes):
                img, m_time = self.get_streamer_frame(streamer_names[idx_img], new_cdata.images[streamer_names[idx_img]])
                for idx_box, one_box in enumerate(one_image_boxes):
                    if one_box.box_class_name in self.secondary_class_names:
                        if not self.dump_images:
                            box_h = int(one_box.box_coord[3]) - int(one_box.box_coord[1])
                            box_w = int(one_box.box_coord[2]) - int(one_box.box_coord[0])
                            new_img = img[int(one_box.box_coord[1] - box_h/1.7):int(one_box.box_coord[3] + box_h/1.7), int(one_box.box_coord[0] - box_w/1.9):int(one_box.box_coord[2] + box_w/1.9)]
                            if new_img.shape[0] == 0 or new_img.shape[1] == 0:
                                continue
                            box_images.append(new_img)
                        box_idxes.append([idx_img, idx_box])
                        frames_stream_names.append(streamer_names[idx_img])

            pred = self.forward(box_images, frames_stream_names)
            self.to_cdata_secondary(pred, new_cdata, frames_stream_names, box_idxes, boxes, box_images=box_images)
            self.queue.put_cdata(new_cdata)