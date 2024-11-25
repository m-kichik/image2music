import os

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "data/test_descriptions"
    audio_path = "results/stable_audio_caption_conditioning"
    os.makedirs(audio_path, exist_ok=True)

    # Download model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")

    print(model_config)
    return

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)

    for caption_file in os.listdir(data_path):
        with open(f"{data_path}/{caption_file}", "r") as file:
            caption = file.read().strip()

        print(caption)

        # Set up text and timing conditioning
        conditioning = [
            {
                "prompt": "Classis music describing following view: " + caption,
                "seconds_start": 0,
                "seconds_total": 15,
            }
        ]

        # Generate stereo audio
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

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        torchaudio.save(f"{audio_path}/{caption_file}.wav", output, sample_rate)


if __name__ == "__main__":
    main()
