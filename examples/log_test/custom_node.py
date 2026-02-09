from cpipe.module.logic import Logic


class my_node(Logic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _start(self):

        while True:
            new_cdata = self.get_frames()

            det_stream_names, det_bboxes = new_cdata.get_bboxes()

            for idx, one_image_boxes in enumerate(det_bboxes):
                frame_streamer_name = det_stream_names[idx]
                one_image_boxes = det_bboxes[idx]
                # log
                self.logger.info(frame_streamer_name)
                self.logger.info(one_image_boxes)

                self.logger.warning("1")
                self.logger.error("2")
                self.logger.debug("4")
                self.logger.report("5") # The result will be uploaded to the server (if there is a configuration)
                
