import torch
import torchvision
import onnx
import onnx_graphsurgeon
import numpy as np
import onnxruntime as ort
from collections import OrderedDict


def yolo_insert_nms(path, input_size, score_threshold=0.25, iou_threshold=0.7, max_output_boxes=100, simplify=False):
    '''
    http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnxops/onnx__EfficientNMS_TRT.html
    https://huggingface.co/spaces/muttalib1326/Punjabi_Character_Detection/blob/3dd1e17054c64e5f6b2254278f96cfa2bf418cd4/utils/add_nms.py
    '''
    onnx_model = onnx.load(path)

    if simplify:
        from onnxsim import simplify
        onnx_model, _ = simplify(onnx_model, overwrite_input_shapes={'images': [1, 3, input_size, input_size]})

    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    topk = max_output_boxes
    attrs = OrderedDict(plugin_version='1',
                        background_class=-1,
                        max_output_boxes=topk,
                        score_threshold=score_threshold,
                        iou_threshold=iou_threshold,
                        score_activation=False,
                        box_coding=0, )

    outputs = [onnx_graphsurgeon.Variable('num_dets', np.int32, [-1, 1]),
               onnx_graphsurgeon.Variable('det_boxes', np.float32, [-1, topk, 4]),
               onnx_graphsurgeon.Variable('det_scores', np.float32, [-1, topk]),
               onnx_graphsurgeon.Variable('det_classes', np.int32, [-1, topk])]

    graph.layer(op='EfficientNMS_TRT',
                name="batched_nms",
                inputs=[graph.outputs[0],
                        graph.outputs[1]],
                outputs=outputs,
                attrs=attrs, )

    graph.outputs = outputs
    graph.cleanup().toposort()

    onnx.save(onnx_graphsurgeon.export_onnx(graph), f'yolo_w_nms.onnx')


class YOLO11(torch.nn.Module):
    def __init__(self, name, cls_num) -> None:
        super().__init__()
        from ultralytics import YOLO
        # Load a model
        # build a new model from scratch
        # model = YOLO(f'{name}.yaml')

        # load a pretrained model (recommended for training)
        model = YOLO(name)
        self.model = model.model
        self.cls_num = cls_num

    def forward(self, x):
        '''https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216
        '''
        pred: torch.Tensor = self.model(x)[0]  # n 84 8400,
        pred = pred.permute(0, 2, 1)
        boxes, scores = pred.split([4, self.cls_num], dim=-1)
        boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

        return boxes, scores


def export_onnx(cls_num, input_size, name='./best.pt'):
    '''export onnx
    '''
    m = YOLO11(name, cls_num)

    x = torch.rand(1, 3, input_size, input_size)
    dynamic_axes = {
        'images': {0: '-1'}
    }
    torch.onnx.export(m, x, f'{name}.onnx',
                      input_names=['images'],
                      output_names=['boxes', 'scores'],
                      opset_version=13,
                      dynamic_axes=dynamic_axes)

    data = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
    sess = ort.InferenceSession(f'{name}.onnx')
    _ = sess.run(output_names=None, input_feed={'images': data})

    import onnx
    import onnxslim
    model_onnx = onnx.load(f'{name}.onnx')
    model_onnx = onnxslim.slim(model_onnx)
    onnx.save(model_onnx, f'{name}.onnx')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='./ckpts/det_food_v13_onlyfood_best.pt')
    parser.add_argument('--score_threshold', type=float, default=0.25)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--max_output_boxes', type=int, default=16)
    parser.add_argument('--class_num', type=int, default=80)
    parser.add_argument('--input_size', type=int, default=640)
    args = parser.parse_args()
    export_onnx(args.class_num, args.input_size, name=args.name)

    yolo_insert_nms(path=f'{args.name}.onnx',
                    input_size=args.input_size,
                    score_threshold=args.score_threshold,
                    iou_threshold=args.iou_threshold,
                    max_output_boxes=args.max_output_boxes, )
