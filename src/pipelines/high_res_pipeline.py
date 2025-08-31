import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from src.models.mutual_self_attention import ReferenceAttentionControl


@dataclass
class HighResPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class HighResPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        # image_encoder,
        reference_unet,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            # image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        images = []
        for idx in tqdm(range(latents.shape[0])):
            images.append(self.vae.decode(latents[idx : idx + 1]).sample)
        images = torch.cat(images)
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        images = images.cpu().float().numpy()
        return images

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        cloth_image,
        person_image,
        low_result_image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        alpha = 0.5,
        beta = 0.5,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype = torch.float32,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = len(cloth_image)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            4,
            width,
            height,
            dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        cloth_image_tensor = self.ref_image_processor.preprocess(
            cloth_image, height=height, width=width
        )  # (bs, c, width, height)
        cloth_image_tensor = cloth_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        cloth_image_latent = self.vae.encode(cloth_image_tensor).latent_dist.mean
        cloth_image_latent = cloth_image_latent * 0.18215  # (b, 4, h, w)
        if do_classifier_free_guidance:
            cloth_image_latent = torch.cat((torch.zeros_like(cloth_image_latent), cloth_image_latent), 0)

        person_image_tensor = self.ref_image_processor.preprocess(
            person_image, height=height, width=width
        )
        person_image_tensor = person_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        person_image_latent = self.vae.encode(person_image_tensor).latent_dist.sample()   # (bs, d, 64, 64)
        person_image_latent = person_image_latent * 0.18215
        person_image_latent = person_image_latent.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1)

        low_result_image_tensor = self.ref_image_processor.preprocess(
            low_result_image, height=height, width=width
        )
        low_result_image_tensor = low_result_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        low_result_image_tensor = self.vae.encode(low_result_image_tensor).latent_dist.sample()   # (bs, d, 64, 64)
        low_result_image_tensor = low_result_image_tensor * 0.18215

        latents = alpha * latents + beta * low_result_image_tensor

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        cloth_image_latent,
                        torch.zeros_like(t),
                        return_dict=False,
                    )

                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                latent_model_input = torch.cat((latent_model_input, person_image_latent), 1)
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()

        # Post-processing
        image = self.decode_latents(latents)  # (b, c, h, w)

        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)

        if not return_dict:
            return image

        return HighResPipelineOutput(images=image)
