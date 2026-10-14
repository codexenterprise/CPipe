wget http://code-x.oss-cn-hangzhou.aliyuncs.com/zh/__docker__/rk3588_cpipe4.2.3.tar
docker load -i rk3588_cpipe4.2.3.tar


sudo docker run --name cpipe_npu --net=host \
  -e TZ=Asia/Shanghai --env LANG="zh_CN.UTF-8" -it \
  --privileged --shm-size=32g \
  -v ~:/host \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg \
  aikk_campus_edition:aarch64_cpipe_pro /bin/bash





source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
# 先验证驱动通了
npu-smi info
# 再跑 demo
python acllite_rtsp_rgb_demo.py