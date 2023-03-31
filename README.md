# stable-diffusion-webui-taiyi-scripts
稍微修改和增加一些文件，让stable-diffusion-webui能够加载太乙模型和太乙-动漫模型


为了能在webui上跑起来我们的代码，我们需要稍微增加一点文件，在stable-diffusion-webui上增加对太乙模型的支持。

# 1、增加太乙inference的config

在./stable-diffusion-webui/configs/的目录增加一个文件taiyi-stable-diffusion-inference-v1.yaml

# 2、增加太乙模型结构的module
由于太乙模型修改了stable diffusion的text encoder结构，所以首先在./stable-diffusion-webui/modules/的目录增加sd_hijack_taiyi.py和taiyi.py

然后，修改这个目录下的sd_hijack.py，使其能够正确加载TaiyiCLIPEmbedder；

再修改sd_models_config.py和sd_models.py，使其能够正确加载太乙模型。

# 3、启动太乙吧
sh run.sh
