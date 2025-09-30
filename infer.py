import os, sys

import lightning
import lightning.fabric.accelerators
import torch, torchaudio
import argparse
import json
import re

from lightning import Fabric
from omegaconf import MISSING, OmegaConf, DictConfig
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler


def load_config(cfg_file, parent_dir="./") -> DictConfig:
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", parent_dir))
    # cmd_cfg = OmegaConf.from_cli()

    file_cfg = OmegaConf.load(open(cfg_file, 'r')) if cfg_file is not None \
        else OmegaConf.create()

    return file_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="songbloom_full_150s",
                        choices=["songbloom_full_150s", "songbloom_full_150s_dpo"])
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bf16-true", help="lightning fabric precisions, not dtypes")
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--device", type=str, default='auto')
    parser.add_argument("--compile", type=bool, default=True)
    parser.add_argument("--reference-seconds", type=float, default=10.0,
                        help="the number of seconds of the reference clip to use, up to 10.0")
    parser.add_argument("--addl-max-duration-seconds", type=float, default=20,
                        help="the additional number of seconds to add to the max duration of the configuration for this model")
    args = parser.parse_args()

    fabric = Fabric(accelerator=args.device, precision=args.dtype)
    # prepare output path
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    model_name = args.model_name
    repo = Path(snapshot_download("CypressYang/SongBloom"))
    cfg_path = repo / f"{model_name}.yaml"
    cfg = load_config(cfg_path, parent_dir=str(Path(cfg_path).parent))
    cfg.max_dur = cfg.max_dur + int(args.addl_max_duration_seconds)

    model = SongBloom_Sampler.build_from_trainer(cfg, strict=True, fabric=fabric, compile=args.compile)
    cfg.inference.cfg_coef = args.cfg
    model.set_generation_params(**cfg.inference)

    input_lines = open(args.input_jsonl, 'r').readlines()
    input_lines = [json.loads(l.strip()) for l in input_lines]

    for test_sample in input_lines:
        # print(test_sample)
        idx, lyrics, prompt_wav = test_sample["idx"], test_sample["lyrics"], test_sample["prompt_wav"]

        lyrics = re.sub(r'\((.*?):(\d+)\)', lambda m: ' '.join([m.group(1).strip()] * int(m.group(2))), lyrics)

        prompt_wav, sr = torchaudio.load(prompt_wav)
        if sr != model.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(model.dtype)
        prompt_wav = prompt_wav[..., :min(int(args.reference_seconds * model.sample_rate), 10)]
        # breakpoint()
        fname = f"{idx}_s"
        for i in range(args.n_samples):
            existing_files_len = len(list(x for x in os.listdir(args.output_dir) if fname in x))
            wav = model.generate(lyrics, prompt_wav)
            torchaudio.save(f'{args.output_dir}/{idx}_s{1 + existing_files_len}.flac', wav[0].cpu().float(),
                            model.sample_rate)


if __name__ == "__main__":
    main()
