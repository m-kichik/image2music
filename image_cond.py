import json
import typing as tp

import torch
import clip
from PIL import Image

from huggingface_hub import hf_hub_download

from stable_audio_tools.models.diffusion import (
    DiTWrapper,
    ConditionedDiffusionModelWrapper,
)
from stable_audio_tools.models.factory import create_pretransform_from_config
from stable_audio_tools.models.conditioners import (
    Conditioner,
    NumberConditioner,
    PretransformConditioner,
    MultiConditioner,
)
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond


class CLIPImageConditioner(Conditioner):
    def __init__(
        self,
        output_dim: int,
        clip_model_name: str = "ViT-B/32",
        max_length: int = 128,
        device="cpu",
    ):
        super().__init__(512, output_dim)
        super().to()
        self.max_length = max_length
        self.clip, self.preprocess = clip.load(clip_model_name, device=device)
        self.proj_out = torch.nn.Linear(512, max_length * output_dim)

    def forward(self, images: Image, device: str):
        self.clip.to(device)
        self.proj_out.to(device)

        image_features = []
        with torch.no_grad():
            for image in images:
                image_input = self.preprocess(image).unsqueeze(0).to(device)
                image_features.append(self.clip.encode_image(image_input))

        image_features = torch.stack(image_features)

        embeddings = self.proj_out(image_features.float())
        embeddings = embeddings.view(
            embeddings.shape[0], self.max_length, self.output_dim
        )

        # hope full true means that all of the "tokens" will be encounted
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), dtype=torch.bool, device=device
        )

        return embeddings, attention_mask


def create_multi_conditioner_from_conditioning_config(
    config: tp.Dict[str, tp.Any]
) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "clip":
            conditioners[id] = CLIPImageConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert (
                sample_rate is not None
            ), "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(
                conditioner_config.pop("pretransform_config"), sample_rate=sample_rate
            )

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(
                    load_ckpt_state_dict(
                        conditioner_config.pop("pretransform_ckpt_path")
                    )
                )

            conditioners[id] = PretransformConditioner(
                pretransform, **conditioner_config
            )
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get("diffusion", None)
    diffusion_model_type = diffusion_config.get("type", None)
    diffusion_model_config = diffusion_config.get("config", None)

    diffusion_model = DiTWrapper(**diffusion_model_config)

    io_channels = model_config.get("io_channels", None)
    sample_rate = config.get("sample_rate", None)
    diffusion_objective = diffusion_config.get("diffusion_objective", "v")
    conditioning_config = model_config.get("conditioning", None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(
            conditioning_config
        )

    cross_attention_ids = diffusion_config.get("cross_attention_cond_ids", [])
    global_cond_ids = diffusion_config.get("global_cond_ids", [])
    input_concat_ids = diffusion_config.get("input_concat_ids", [])
    prepend_cond_ids = diffusion_config.get("prepend_cond_ids", [])

    pretransform = model_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    min_input_length *= diffusion_model.model.patch_size

    extra_kwargs = {}

    wrapper_fn = ConditionedDiffusionModelWrapper
    extra_kwargs["diffusion_objective"] = diffusion_objective

    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        **extra_kwargs,
    )


def get_pretrained_model(name: str, model_config_path: str):
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_diffusion_cond_from_config(model_config)

    try:
        model_ckpt_path = hf_hub_download(
            name, filename="model.safetensors", repo_type="model"
        )
    except Exception as e:
        model_ckpt_path = hf_hub_download(
            name, filename="model.ckpt", repo_type="model"
        )

    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path), strict=False)

    return model, model_config


def main():
    model, model_config = get_pretrained_model(
        "stabilityai/stable-audio-open-1.0", "my_config.json"
    )

    device = "cuda:0"

    model = model.to(device)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    image = Image.open("../data/test_resized/anime_autumn.jpeg")
    conditioning = [
        {
            "prompt": image,
            "seconds_start": 0,
            "seconds_total": 15,
        }
    ]
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )


if __name__ == "__main__":
    main()
