import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from src.models.unet_condition import UNet2DConditionModel as UNet2DConditionModel_ref
from src.models.unet_denoising import UNet2DConditionModel as UNet2DConditionModel_denoising
from src.pipelines.low_res_pipeline import LowResPipeline
from src.utils.util import import_filename


def run_infer(
    vae,
    reference_unet,
    denoising_unet,
    scheduler,
    width,
    height,
    dtype,
    seed,
    infer_type,
    save_path,
    data_root = None,
    batch_size = None,
    cloth_path = None,
    person_path = None,
):

    generator = torch.Generator().manual_seed(seed)

    pipe = LowResPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    )

    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    save_path = os.path.join(save_path, infer_type)
    os.makedirs(save_path, exist_ok=True)

    if infer_type == "single":
        cloth_image_pil = Image.open(cloth_path).convert("RGB")
        person_image_pil = Image.open(person_path).convert("RGB")
        image = pipe(
            [cloth_image_pil],
            [person_image_pil],
            width,
            height,
            20,
            3.5,
            generator=generator,
            dtype = dtype
        ).images
        image = image[0, :].permute(1, 2, 0).cpu().numpy()
        res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        res_image_pil.save(os.path.join(save_path, "result.png"))
    elif infer_type == "VITON-HD":
        cloth_path= os.path.join(data_root, "test" ,"cloth")
        person_path= os.path.join(data_root, "test" ,"image")

        unpaired_file = os.path.join(data_root, "test_pairs.txt")

        correspondence_dict = {}
        with open(unpaired_file, 'r') as f:
            for line in f.readlines():
                person_name, cloth_filename= line.strip().split()
                correspondence_dict[cloth_filename.split(".")[0]] = person_name.split(".")[0]

        cloth_images = []
        person_images = []
        names = []
        batch_counter = 0

        for cloth_name in tqdm(correspondence_dict.keys(), desc="Processing Images"):
            cloth_image_pil = Image.open(os.path.join(cloth_path, cloth_name + ".jpg")).convert("RGB")
            person_image_pil = Image.open(os.path.join(person_path, correspondence_dict[cloth_name] + ".jpg")).convert("RGB")

            cloth_images.append(cloth_image_pil)
            person_images.append(person_image_pil)
            names.append(correspondence_dict[cloth_name])

            batch_counter += 1

            # 如果达到batch_size，处理这一批
            if batch_counter == batch_size:
                # 使用模型处理这批数据
                result_images = pipe(
                    cloth_images,
                    person_images,
                    width,
                    height,
                    20,
                    3.5,
                    generator=generator,
                    dtype=dtype
                ).images

                # 保存处理后的图片（兼容 torch.Tensor 或 numpy.ndarray）
                def _to_pil(x):
                    # x: tensor (C,H,W) or numpy (C,H,W) or numpy (H,W,C)
                    if isinstance(x, torch.Tensor):
                        arr = x.detach().cpu().numpy()
                    else:
                        arr = np.asarray(x)

                    # if channels-first (C,H,W) -> convert to H,W,C
                    if arr.ndim == 3 and arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))

                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    return Image.fromarray(arr)

                for i in range(batch_size):
                    res_image_pil = _to_pil(result_images[i])
                    res_image_pil.save(os.path.join(save_path, names[i] + ".png"))

                # 清空列表，准备下一批
                cloth_images = []
                person_images = []
                names = []
                batch_counter = 0

        # 处理剩下未处理的图像（如果有的话）
        if batch_counter > 0:
            batch_cloth = np.stack(cloth_images)
            batch_person = np.stack(person_images)

            result_images = pipe(
                batch_cloth,
                batch_person,
                width,
                height,
                20,
                3.5,
                generator=generator,
                dtype=dtype
            ).images

            for i in range(batch_counter):
                res_image_pil = _to_pil(result_images[i])
                res_image_pil.save(os.path.join(save_path, names[i] + ".png"))

    del pipe
    torch.cuda.empty_cache()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/low_res_infer.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        cfg = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        cfg = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    reference_unet = UNet2DConditionModel_ref.from_pretrained_checkpoint(cfg.ref_config_path, cfg.ref_weights_path)
    reference_unet.to("cuda", dtype = weight_dtype)
    denoising_unet = UNet2DConditionModel_denoising.from_pretrained_checkpoint(cfg.denoising_config_path, cfg.denoising_weights_path)
    denoising_unet.to("cuda", dtype = weight_dtype)

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)

    run_infer(vae, reference_unet, denoising_unet, val_noise_scheduler, cfg.dataset.image_width, cfg.dataset.image_height, 
              weight_dtype, cfg.seed, cfg.dataset.infer_type, cfg.dataset.save_path, data_root=cfg.dataset.data_root, 
              batch_size=cfg.dataset.batch_size, cloth_path=cfg.dataset.cloth_path, person_path=cfg.dataset.person_path)