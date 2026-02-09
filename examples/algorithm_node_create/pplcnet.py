from cpipe.module.model.pplcnet import PPLCNet
from cpipe.module.node import Node


if __name__ == '__main__':

    model2 = PPLCNet(node_name="model2",
                     model_path="./model_files/deepsort_pplcnet.onnx",
                     input_size=(3, 192, 64),
                     secondary_class_names=["person"],
                     device="cuda:0")
