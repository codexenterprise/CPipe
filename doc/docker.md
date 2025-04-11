### 项目部署

拉镜像

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/cpipe/ningbo_shiyan:v5.0
```

将项目文件复制到`/root/workspace`下

```bash
sudo docker run -itd --gpus all  --net=host --privileged  \
--name Nibo_experiment_alog_api \
-v /root/workspace:/root/workspace \
registry.cn-hangzhou.aliyuncs.com/cpipe/ningbo_shiyan:v5.0 \
/bin/bash
```

转换模型（可选）