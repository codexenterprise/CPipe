
### 3.7 版本使用示例
```bash
wget http://code-x.oss-cn-hangzhou.aliyuncs.com/zh/__docker__/cpipe3.7.4_4090_570.195.03.tar
docker load -i cpipe3.7.4_4090_570.195.03.tar
sudo docker run --name cpipe3 --runtime=nvidia --net=host -e TZ=Asia/Shanghai --env LANG="zh_CN.UTF-8" -dit -v ~:/host --privileged --shm-size=64g cpipe /bin/bash
```
### 4.2版本使用示例

```bash
wget http://code-x.oss-cn-hangzhou.aliyuncs.com/zh/__docker__/cpipe4.2.1.tar
# wget http://code-x.oss-cn-hangzhou.aliyuncs.com/zh/__docker__/rk3588_cpipe4.2.3.tar # 瑞芯微RK3588平台使用
docker load -i cpipe4.2.1.tar # rk3588_cpipe4.2.3.tar
sudo docker run --name cpipe4 --runtime=nvidia --net=host -e TZ=Asia/Shanghai --env LANG="zh_CN.UTF-8" -dit -v ~:/host --privileged --shm-size=64g cpipe4.2.1 /bin/bash # rk3588_cpipe4.2.3 /bin/bash
```