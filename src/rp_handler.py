'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image


from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup, rp_download
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA, LORA_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #
class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        ).to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe


    def load_refiner(self):
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        ).to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)

            self.base = future_base.result()
            self.refiner = future_refiner.result()

    def load_loras(self, loras):
        adapter_names = []
        adapter_weights = []
        for lora in loras:
            self.base.load_lora_weights(
                lora['extarcted_path'],
                adapter_name=lora['lora_id'],
            )
            adapter_names.append(lora['lora_id'])
            adapter_weights.append(lora['adapter_weight'])
        
        self.base.set_adapters(adapter_names, adapter_weights=adapter_weights)
        
    def unload_loras(self):
        self.base.unload_lora_weights()


MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                # image_urls.append(f"data:image/png;base64,{image_data}")
                image_urls.append(str(image_data))
    
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

def load_and_apply_loras(loras: list) -> list:
    # load lora models
    downloaded_loras = []
    for lora in loras:
        if lora.get('url', None):
            downloaded_lora = rp_download.file(lora['url'])
            lora['extracted_path'] = downloaded_lora['extracted_path']
        else:
            lora['extracted_path'] = lora['lora_id']
    MODELS.load_loras(loras)


def validate_lora(lora, lora_schema):
    validated_lora = validate(lora, lora_schema)
    if 'errors' in validated_lora:
        return {'errors': validated_lora['errors']}
    
    lora = validated_lora['validated_input']
    if not lora.get('url', None):
        try:
            MODELS.base.load_lora_weights(
                lora['lora_id'],
                adapter_name=lora['lora_id']
            )
        except Exception as e:
            return {'errors': "lora_id should be huggingface lora id if url is not set."}
    return {'validated_input': lora}


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input['image_url']

    loras = job_input['loras']
    if loras:
        for lora in loras:
            validated_lora = validate_lora(lora, LORA_SCHEMA)

            if 'errors' in valideted_lora:
                return {"error": validated_lora['errors']}
    
        load_and_apply_loras(loras)


    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    MODELS.base.scheduler = make_scheduler(job_input['scheduler'], MODELS.base.scheduler.config)

    if starting_image:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator
        ).images
    else:
        # Generate latent image using pipe
        image = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        # Refine the image using refiner with refiner_inference_steps
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=image,
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

    image_urls = _save_and_upload_images(output, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results


runpod.serverless.start({"handler": generate_image})
