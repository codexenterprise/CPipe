#### CPipe是基于Python编写的AI视觉算法快速部署框架.本框架基于Node思想,将所有视频流/视频文件/AI算法/上报信息以及逻辑代码块等抽象成一个Node节点.节点之间可以自由连接.



CPipe框架功能:
- 支持Node节点网页可视化
- 支持算法推理结果网页实时可视化
- 支持一键快速部署
- 支持GPU视频解码加速
- 支持页面添加/删除视频流节点
- 支持页面配置算法区域功能
- 支持AI模型格式TensorRT/ONNX/PyTorch等
- 支持各类AI算法(目标检测/旋转目标检测/分类/人脸识别/人脸质量评估/关键点检测/人流统计/目标跟踪/REID/视频时序分类等)
- 支持视频类型:RTSP/RTMP/本地视频文件/本地图片
- 支持日志本地存储/日志云端上报/日志网页实时显示
- 支持AI模型文件加密功能

### 框架优势
| 内容 | 使用前 | 使用CPipe+训练平台 |
| --- | --- | --- |
| 算法工程师经验 | 3年以上 | 1年以上 |
| 开发周期缩减 | 无 | 缩减80%以上 |
| 算法部署环境 | 需要搭建 | 框架提供Docker镜像 |
| 算法硬件加速 | 需要自己编写代码 | 框架自带 |
| 视频编解码加速 | 需要自己编译相关硬件库 | 框架自带 |
| 视频流批量推理 | 需要编写相关并发代码 | 框架自带 |
| 一键无代码部署 | 无 | 框架自带 |
| 算法性能可视化 | 无 | 框架自带 |
| 算法结果实时Web可视化 | 需要前端工程师参与编写代码 | 框架自带 |
| 默认算法支持 | 无 | 框架自带十几种算法 |
| 算法文件加密 | 需要编写加密程序 | 框架自带 |
| 日志 | 需要编写日志程序 | 框架自带 |

价值:
降低研发成本/减少开发周期/提供稳定且高性能的AI算法推理引擎/提供高并发视频流实时算法推理/提供算法模型机代码加密安全

![cpipe.png](doc%2Fcpipe.png)

### 基于CPipe SDK 在Web页面上实时显示效果:
![demo1.png](doc%2Fdemo1.png)

### 详细使用手册见:
[CPipe使用手册v3.3.0.docx](doc%2FCPipe%CA%B9%D3%C3%CA%D6%B2%E1v3.3.0.docx)

### CPipe安装方法:
采用pip安装指令,支持Python 3.8及以上,并提供相关Docker镜像.

例子:
pip install cpipe-3.0.0-cp39-cp39-linux_x86_64.whl

### 更新内容：V3.5.7 （2025-4-7）
1. 新增支持所有模型onnx转tensorrt,见[demo.py](examples%2Fonnx2tensorrt%2Fdemo.py)

### 更新内容：V3.5.6 （2025-3-x）
1. 增加cpipe格式加密模型(cpipe模型可以再cpipe框架下运行, 不依赖设备license, codex模型依赖设备license). demo见[demo.py](examples%2Fmodel_encryption%2Fdemo.py)
2. 新增VideoStreamer支持本地USB摄像头. demo见[demo.py](examples%2Fstreamer%2Fdemo.py)
3. 增加部分案例[examples](examples)
4. 适配onnxruntime新版本

### 更新内容：V3.5.5 （2025-3-26）
1. 优化麒麟系统插拔U盘导致license失效问题
2. retinafaceTRT节点增加支持一阶段检测功能(具体代码见[face_recognition](examples%2Fface_recognition))
3. 新增[examples](examples)目录, 用于存放示例代码
4. 新增CPipeTools类
   ```python
   from cpipe.tools.cpipetools import CPipeTools
   # 模型加密方法
   CPipeTools.encrypt_models("./movenet_person_pose.onnx","1234567890123456",
   "./1234567890123456.cpipe.license") # 文件模式
   CPipeTools.encrypt_models("./models","1234567890123456",
   "./1234567890123456.cpipe.license") # 文件夹模式
   ```
5. 增加硬件RK3588适配: 案例代码见[main_RK3588.py](examples%2Fface_recognition%2Fmain_RK3588.py)
   - 已适配yolov7 (删除yolov7TRT, 统一到yolov7)
   - 已适配retinaface
   - 已适配adaface
   - 已适配facerecognition
   - 
6. yolov7 增加 ONNX模型适配, 模型结构必须满足如下:

   ![yolov7.png](doc%2Fyolov7.png)

7. 优化WSL license问题
8. 增加retinaface ONNX模型适配(删除RetinafaceTRT类, 统一到Retinaface)



### 更新内容：V3.5.0 （2025-3-11）
1. 优化VideoStreamer节点, 支持动态设置process_frame_interval参数, 用于调整推理速度.
   使用方法: streamer = VideoStreamer(name, one, queue_size, process_frame_interval=1)
            streamer.process_frame_interval = 2
2. 新增DinoEmbedding算法模型
3. 新增DinoClassifier分类算法模型
4. Box类新增box_embedding/box_embedding_name/box_embedding_score 用于存储box的embedding信息
5. 新增yolov11TRT节点(目标检测)
6. 新增YOLOv11InstanceSeg 实例分割算法模型(易金城)
7. 新增MoveNetPersonPose节点
8. 新增日志文件名添加标识功能:
   注:在所有CPipe类初始化前优先初始化项目代码生效:
   import cpipe.config.config
   cpipe.config.config.CLOGER_FILE_NAME_MARK = "996+996"  # 
9. ImageStreamer新增动态喂图功能: 通过Queue传入图片路径或者numpy array
10. 新增CImage类 info 成员变量, 用于存储图片信息(dict)
11. 解决WSL重启系统后license失效问题
12. 解决国产麒麟系统重启后license失效问题


### 更新内容：V3.2.2 （2024-12-25）
1. 优化CPipe退出机制: Cpipe.exit()/Cpipe.terminate()/linux kill指令/ctrl+c等退出方式, 会自动退出所有子进程,并执行所有Node的lastly方法.
2. 新增Node可配置daemon参数, 用于设置Node是否为守护进程, 默认为True. 注:对于需要在Node中创建子进程的情况, 需要设置为False.
3. logger增加warning方法; logger日志信息取消调用函数定位显示
4. 将SaveInsight整合到CPipeInsight中, 通过参数save_video=True开启功能, 取消SaveInsight节点
5. 将UIinsight整合到CPipeInsight中, 通过参数ui_insight=True开启功能, 取消UIInsight节点
```python
    cpipeinsight = CPipeInsight(
                                  http_insight=True,
    
                                  ui_insight=True,
          
                                  save_video=True, auto_exit=True, save_file_names={stream_zhu.nodeName: student_id},
                                  save_duration_seconds=int(30),
                                  save_path="/mnt/d/save_stream"
     )
```

6. moveNet节点新增支持不同输入尺寸(需结合相关训练代码).
7. 增加https支持, 通过参数ssl=True开启, 默认为False.
8. 合并LocalVideoStreamer和VideoStreamer, 统一通过VideoStreamer 自动识别本地文件和网络流地址.
9. 增加VideoStreamer动态设置流地址功能, 通过reset_stream方法设置.(如果共享内存模式下(SHARE_MEMORY_MODE=True),流分辨率不变)

### 更新内容：V3.2.0 （2024-12-2）
1. 所有模型增加灰度模式支持
2. 新增手掌关键点模型RTMPOSE
3. 修复日志部分bug
4. Node类增加lastly方法, 用于进程退出时调用， 
   signal.signal(signal.SIGINT, self.lastly)  # 退出时调用lastly方法
5. 增加 CPipeInsight 参数：
show_polygon_box: self.kwargs.get("show_polygon_box", False)
show_box: self.kwargs.get("show_box", True)
show_box_name: self.kwargs.get("show_box_name", True)
show_polygon: self.kwargs.get("show_polygon", True)
show_mask: self.kwargs.get("show_mask", True)
show_key_points: self.kwargs.get("show_key_points", True)
show_person: self.kwargs.get("show_person", True)
show_classification: self.kwargs.get("show_classification", True)
show_track: self.kwargs.get("show_track", True)


### 更新内容：V3.1.1 （2024-11-15）
1. 所有代码增加注释
2. 增加分割模型支持
3. 修復bug


### 更新内容：V3.0.0 （2024-10-18）
1. 所有现有推理模型全波统一推理框架:
   - 大部分模型直接支持onnx/TRT/torch jit模型文件输入: 
   - 新增 Cmodel/InferenceEngine/CDetector/COBBDetector/CClassifier/CFace/CEmbedding基础类
2. 优化跳帧显示逻辑, 无需每个node单独写捕获逻辑代码,框架自动完成
3. 优化node显示页面, 采用antv x6框架
4. 新增支持快速配置框架功能: 通过一个配置文件([launch.yaml](config%2Flaunch.yaml))直接生成node链接关系, 无需多余代码.
    - cpipe ./config/launch.yaml
5. 调整cmask标定功能, 默认所有算法支持标定功能:
    - Node.special_mask = {"polygons": [], "lines": []} 可以指定画的cmask区域
6. 新增: 可以通过页面添加streamer功能, 支持自动识别流类型,自动选择streamer子类.
7. 增加cpu 拉流和cuda拉流并存模式
8. 增加日志不同等级不同颜色
9. 调整TRT/ONNX模型支持多GPU功能,无需设置环境变量CUDA_VISIBLE_DEVICES
    - 通过device参数: cuda:x 和 cpu 指定
10. 增加streamer支持指定gpu
    - 通过device参数: cuda:x 和 cpu 指定


![contacts.jpg](doc%2Fcontacts.jpg)

邮箱: 9838465@qq.com

