from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler
import torch, waifu2x
from diffusers.utils import load_image

class GenImg:
    """
    A Generator for text-to-image or image-to-image using Stable Diffusion.

    This class allows users customize their own characters.
    Downloading a checkpoint file from civitai.com is recommended.
    
    Args:
        model: model_path or huggingface format name, such as "./models/zentypeexExperimental_exAW4.safetensors".
        output_path: dir path to output
        gen_mode[optional]: (width, height)    Control the output dimension of the model.
        up_scale[optional]: up scale ratio, default = None. If your machine's VRAM is not allowing you to generate high quality image, use this param.
        eta[optional]: generate eta, default = 0.5
        img2img[optional]: your image path      If you want to generate image from your image, you should set this param.
        up_scale_gpu: use cuda to up scale      If out of memory, set False to use CPU
    """
    model_list = [
        "beyondLCMNonLCM_v10cNonLCM.safetensors",       # 0, 西方风格       url: https://civitai.com/models/943923/beyond-lcmnon-lcm?modelVersionId=1255553
        "zentypeexExperimental_exAW4.safetensors",      # 1, 魔幻风格       url: https://civitai.com/models/1060496?modelVersionId=1213393
        "unstableIllusion_hyper.safetensors",           # 2, 真人风格       url: https://civitai.com/models/147687/unstable-illusion?modelVersionId=745778
        "V3_0.safetensors",                             # 3, 国漫风格       url: https://civitai.com/models/1042699/v3
        "againmixsdxl_v4Lightning.safetensors",         # 4, 真人风格，SDXL url: https://civitai.com/models/233359/againmixsdxl?modelVersionId=1194141
    ]
    prompt_pre = "masterpiece, best quality, 8k, nsanely detailed, ultra-detailed, highly detailed, unreal engine rendered, "
    negative_prompt = "(bad quality,worst quality,low quality,bad anatomy,bad hand:1.3), nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name,"

    def __init__(self, model, output_path, gen_mode=(768, 512), up_scale=None, eta=0.5, img2img=None, up_scale_gpu=False):
        self.img2img = img2img
        if img2img != None:
            self.img = load_image(img2img)
            self.pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                model,
                torch_dtype=torch.float16
            ).to("cuda")
        else:
            self.pipe = StableDiffusionPipeline.from_single_file(
                model,
                torch_dtype=torch.float16
            ).to("cuda")
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.output_path = output_path
        self.generator = torch.Generator("cuda")
        self.width  = gen_mode[0]
        self.height = gen_mode[1]
        self.eta = eta
        self.up_scale = up_scale
        self.up_scale_gpu = up_scale_gpu

    def __call__(self, prompt):
        """
        Args:
            prompt: generate param
                recommendation: [color] [long | short] hair girl
                                [magic | library | ...] background
                                [jk | lady | maid | ...]
                                [witch | angel | ...]
                example: prompt = "black long hair girl, JK, red eyes, witch, library background, "
        """
        expressions = ['Smile', 'Surprised', 'Confuse', 'Shy', 'Thinking', 'Sleeping']

        for expression in expressions:
            if self.img2img != None:
                image = self.pipe(
                    image=self.img,
                    prompt=self.prompt_pre + prompt + expression,
                    negative_prompt=self.negative_prompt,
                    width=self.width,
                    height=self.height,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    generator=self.generator,
                    eta=self.eta
                ).images[0]
            else:
                image = self.pipe(
                    prompt=self.prompt_pre + prompt + expression,
                    negative_prompt=self.negative_prompt,
                    width=self.width,
                    height=self.height,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    generator=self.generator,
                    eta=self.eta
                ).images[0]
            output_path = self.output_path + "/output_{}.png".format(expression)
            image.save(output_path)

            if self.up_scale != None:
                gpu = 0 if self.up_scale_gpu else -1
                waifu2x.run(output_path, output_path, gpu=gpu, scale_ratio=self.up_scale)
            