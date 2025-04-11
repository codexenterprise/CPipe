from cpipe.tools.cpipetools import CPipeTools

# The encryption model in cpipe format can be used under all CPIPE frameworks.
# CPipeTools.encrypt_models("/home/zhouhe/workspace/cpipe2.0/__OTHERS__/demo_person/yolov10n.onnx", model_type=CPipeTools.MODEL_TYPE_CPIPE)


# The codex encrypted model can only be used on the specified license device.
# CPipeTools.encrypt_models("/home/zhouhe/workspace/cpipe2.0/__OTHERS__/demo_person/yolov10n.onnx",
#                           license_password="1234567890123456",
#                           license_path="/home/zhouhe/workspace/cpipe2.0/cpipe.license",
#                           model_type=CPipeTools.MODEL_TYPE_CODEX)

CPipeTools.encrypt_models("../../src/model_files/416x416-det_10g_batch.onnx",
                          model_type=CPipeTools.MODEL_TYPE_CPIPE)
