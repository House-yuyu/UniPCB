import sys
if './' not in sys.path:
    sys.path.append('./')
from utils.share import *
import utils.config as config

import cv2
import einops
import numpy as np

import os
from tqdm import tqdm

import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler

# 文件夹路径
txt_path = "/data/tanlianghong/gen_img/Uni-ControlNet/data/new_dataset_3/anno.txt"
canny_dir = "/data/tanlianghong/gen_img/Uni-ControlNet/data/new_dataset_3/canny/"
depth_dir = "/data/tanlianghong/gen_img/Uni-ControlNet/data/new_dataset_3/depth/"
# seg_dir = "/data/tanlianghong/gen_img/Uni-ControlNet/data/unicontrolnet_test/conditions/seg/"
# content_dir = "/data/tanlianghong/gen_img/Uni-ControlNet/data/conditions/content/"
output_dir = "/data/tanlianghong/gen_img/Uni-ControlNet/data/new_dataset_3/new_unigen3/"

# params = {
#     "a_prompt": "best quality, extremely detailed",
#     "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
#     "num_samples": 1,
#     "image_resolution": 512,
#     "strength": 1.0,
#     "global_strength": 1.0,
#     "ddim_steps": 50,
#     "scale": 7.5,
#     "seed": 33,
#     "eta": 0.0
# }
params = {
    "a_prompt": "best quality, extremely detailed",
    "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    "num_samples": 3,
    "image_resolution": 512,
    "strength": 1.0,
    "ddim_steps": 50,
    "scale": 7.5,
    "seed": 33,
    "eta": 0.0
}

model = create_model("/data/tanlianghong/gen_img/Uni-ControlNet/configs/local_v15.yaml").cpu()
model.load_state_dict(load_state_dict(
    "/data/tanlianghong/gen_img/Uni-ControlNet/log_local/lightning_logs/version_0/checkpoints/epoch=25-step=30000.ckpt",
                                      location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# def process(canny_image, depth_image, content_emb, prompt, params):
# def process(canny_image, depth_image, seg_image, prompt, params):
def process(canny_image, depth_image, prompt, params):
    seed_everything(params['seed'])

    # 处理输入图
    canny_image = resize_image(HWC3(canny_image), params['image_resolution'])
    depth_image = resize_image(HWC3(depth_image), params['image_resolution'])
    # seg_image = resize_image(HWC3(seg_image), params['image_resolution'])
    H, W, C = canny_image.shape

    with torch.no_grad():
        # 准备 local_control
        canny_detected_map = HWC3(canny_image)
        depth_detected_map = HWC3(depth_image)
        # seg_detected_map = HWC3(seg_image)

        # detected_maps_list = [canny_detected_map, depth_detected_map, seg_detected_map]
        detected_maps_list = [canny_detected_map, depth_detected_map]
        detected_maps = np.concatenate(detected_maps_list, axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(params['num_samples'])], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()

        # 准备 global_control
        # global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        # global_control = torch.stack([global_control for _ in range(params['num_samples'])], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        # uc_global_control = torch.zeros_like(global_control)
        # cond = {
        #     "local_control": [local_control],
        #     "c_crossattn": [
        #         model.get_learned_conditioning([prompt + ', ' + params['a_prompt']] * params['num_samples'])],
        #     "global_control": [global_control]
        # }
        # un_cond = {
        #     "local_control": [uc_local_control],
        #     "c_crossattn": [model.get_learned_conditioning([params['n_prompt']] * params['num_samples'])],
        #     "global_control": [uc_global_control]
        # }
        cond = {
            "local_control": [local_control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ', ' + params['a_prompt']] * params['num_samples'])]
        }
        un_cond = {
            "local_control": [uc_local_control],
            "c_crossattn": [model.get_learned_conditioning([params['n_prompt']] * params['num_samples'])]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [params['strength']] * 13

        # samples, _ = ddim_sampler.sample(
        #     params['ddim_steps'], params['num_samples'], shape, cond, verbose=False, eta=params['eta'],
        #     unconditional_guidance_scale=params['scale'],
        #     unconditional_conditioning=un_cond, global_strength=params['global_strength']
        # )
        samples, _ = ddim_sampler.sample(
            params['ddim_steps'], params['num_samples'], shape, cond, verbose=False, eta=params['eta'],
            unconditional_guidance_scale=params['scale'],
            unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,255).astype(np.uint8)
        results = [x_samples[i] for i in range(params['num_samples'])]

    return results


# 创建保存结果的文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历txt文档并生成图像
with open(txt_path, 'r') as f:
    lines = f.readlines()

# start_line = 1772
# for line in tqdm(lines, desc="Processing images"):
for idx, line in enumerate(tqdm(lines, desc="Processing images")):
    # if idx + 1 < start_line:
    #     continue  # 跳过前1771行
    # 分离图片名和对应的描述（prompt）
    img_name, prompt = line.strip().split("\t")
    # if(idx==1771):
    #     print(img_name, prompt)

    # 构建文件路径
    canny_image_path = os.path.join(canny_dir, img_name + ".jpg")
    depth_image_path = os.path.join(depth_dir, img_name + ".png")
    # seg_image_path = os.path.join(seg_dir, img_name + ".jpg")
    # content_emb_path = os.path.join(content_dir, img_name + ".npy")

    # 检查文件是否存在
    if not os.path.exists(canny_image_path):
        print(f"Warning: {canny_image_path} does not exist. Skipping.")
        break
        # continue
    if not os.path.exists(depth_image_path):
        print(f"Warning: {depth_image_path} does not exist. Skipping.")
        break
        # continue
    # if not os.path.exists(seg_image_path):
    #     print(f"Warning: {seg_image_path} does not exist. Skipping.")
    #     break
    # if not os.path.exists(content_emb_path):
    #     print(f"Warning: {content_emb_path} does not exist. Skipping.")
    #     break
        # continue

    # 加载图像和 content_emb
    canny_image = cv2.imread(canny_image_path, cv2.IMREAD_UNCHANGED)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    # seg_image = cv2.imread(seg_image_path, cv2.IMREAD_UNCHANGED)
    # content_emb = np.load(content_emb_path)

    # 调用process函数进行图像生成
    # generated_images = process(canny_image, depth_image, content_emb, prompt, params)
    # generated_images = process(canny_image, depth_image, seg_image, prompt, params)
    generated_images = process(canny_image, depth_image, prompt, params)

    # # 保存生成的图像
    # output_image_path = os.path.join(output_dir, img_name + ".jpg")
    # # 检查保存路径下是否已有同名图片
    # if os.path.exists(output_image_path):
    #     print(f"Warning: {output_image_path} already exists. Skipping.")
    #     break
    for i in range(3):
        # 保存生成的图像
        output_image_path = os.path.join(output_dir, img_name + f"_{str(i)}.jpg")
        # 检查保存路径下是否已有同名图片
        if os.path.exists(output_image_path):
            print(f"Warning: {output_image_path} already exists. Skipping.")
            break
        cv2.imwrite(output_image_path, generated_images[i])  # 这里只取第一个生成图像
        print(f"Saved generated image: {output_image_path}")