import numpy as np
from rknn.api import RKNN


def torch_version():
    import torch
    torch_ver = torch.__version__.split('.')
    torch_ver[2] = torch_ver[2].split('+')[0]
    return [int(v) for v in torch_ver]

# 必须在PC主机上运行，并安装了torch和rknn-toolkit2
if __name__ == '__main__':
    do_quantization = False
    if torch_version() < [1, 9, 0]:
        import torch
        print("Your torch version is '{}', in order to better support the Quantization Aware Training (QAT) model,\n"
              "Please update the torch version to '1.9.0' or higher!".format(torch.__version__))
        exit(0)

    model = 'models/yolov10n.onnx'
    if do_quantization:
        new_file_name = model.replace('.onnx', '_Q.rknn')
    else:
        new_file_name = model.replace('.onnx', '_std255.rknn')

    # Create RKNN object
    rknn = RKNN(verbose=True)
    # Pre-process config
    print('--> Config model') # just support one batch size
    rknn.config(mean_values=[0, 0, 0], std_values=[255, 255, 255],
                dynamic_input=[
                    [[1, 3, 640, 640]],
                ],
                target_platform="RK3588")
    # rknn.config( target_platform="RK3588")
    print('done')

    # Load model
    print('--> Loading model')
    # ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    ret = rknn.load_onnx(model=model)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quantization)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(new_file_name)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)





    # print('done')
    #
    # # Set inputs
    # img = cv2.imread('./bus.jpg')
    # # img = cv2.resize(img, (640, 640))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img[None, :, :, :]
    #
    # # Init runtime environment
    # print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')
    #
    # # Inference
    # print('--> Running model')
    # for i in range(10):
    #     t = time.time()
    #     outputs = rknn.inference(inputs=[img])
    #     print(time.time() - t)
    # # np.save('./pytorch_resnet18_qat_0.npy', outputs[0])
    # # show_outputs(softmax(np.array(outputs[0][0])))
    # print('done')
    #
    # rknn.release()
