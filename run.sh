
ckpt=./pretrained/pytorch/huggingface/Taiyi-Stable-Diffusion-1B-Anime-Chinese-v0.1/model.ckpt

yaml=./configs/taiyi-stable-diffusion-inference-v1.yaml

CUDA_VISBLE_DEVICES=0 python launch.py --ckpt $ckpt --config $yaml --listen --port 2333   --disable-safe-unpickle --enable-insecure-extension-access
