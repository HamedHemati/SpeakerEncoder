import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
import torch
from speaker_encoder.model import SpeakerEncoder
from speaker_encoder.audio import AudioProcessor
from speaker_encoder.io import load_config


def main(args):
    c = load_config(args.config_path)
    ap = AudioProcessor(**c['audio'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Encoder model and load pretrained checkpoint
    model = SpeakerEncoder(**c.model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device)['model'])
    model.eval()

    # Compute speaker embeddings
    wav_file = args.input_path

    mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=ap.sample_rate)).T
    mel_spec = torch.FloatTensor(mel_spec[None, :, :])
    mel_spec = mel_spec.to(device)
    embedd = model.compute_embedding(mel_spec)
    embedd = embedd.detach().cpu().numpy()

    print(embedd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="pretrained_model/best_model.pth.tar", required=False)
    parser.add_argument('--config_path', type=str, default="pretrained_model/config.json", required=False)
    parser.add_argument('--output_path', type=str, default="outputs/", required=False)
    parser.add_argument('--input_path', type=str, default="outputs/", required=False)
    args = parser.parse_args()

    main(args)