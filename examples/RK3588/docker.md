wget http://code-x.oss-cn-hangzhou.aliyuncs.com/zh/__docker__/rk3588_cpipe4.2.3.tar
docker load -i rk3588_cpipe4.2.3.tar
sudo docker run --name cpipe4 --net=host -e TZ=Asia/Shanghai --env LANG="zh_CN.UTF-8" -dit -v ~:/host --privileged --shm-size=8g rk3588_cpipe4.2.3 /bin/bash