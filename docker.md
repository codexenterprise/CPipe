### 项目部署

拉镜像

```bash
docker pull registry.cn-shanghai.aliyuncs.com/workspace_tyh/ningbo_shiyan:v5.1
```

```bash
sudo docker run -itd --gpus all  --net=host --privileged  \
--name cpipe \
-v /root/workspace:/root/workspace \
registry.cn-shanghai.aliyuncs.com/workspace_tyh/ningbo_shiyan:v5.1 \
/bin/bash
```