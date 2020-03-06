## 文件准备

* 将`RUN.py & UTILS.py`置于该文件夹下
* 添加`ckpt`文件夹，注意修改`run.sh`中的`--output_dir`参数
* 根据需要修改`transform.py & labels.json`

## 镜像操作

* 构建镜像
```shell
cd eigen
docker build -t registry.cn-shenzhen.aliyuncs.com/opsn/covid:1.0 -f ./transformers/examples/covid_predict/Dockerfile transformers
```

* 上传镜像
```shell
docker push registry.cn-shenzhen.aliyuncs.com/opsn/covid:1.0
```

* 执行镜像
```shell
# CPU
docker run -v PATH_TO_DIR:/tcdata IMAGE_ID sh run.sh
# GPU
nvidia-docker run -v PATH_TO_DIR:/tcdata IMAGE_ID sh run.sh
```

