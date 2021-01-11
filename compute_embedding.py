import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
import torch
from speaker_encoder.model import SpeakerEncoder
from speaker_encoder.audio import AudioProcessor
from speaker_encoder.io import load_config


class SpeechEmbedding():
    def __init__(self, config, model_path):
        self.ap = AudioProcessor(**config['audio'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Encoder model and load pretrained checkpoint
        self.model = SpeakerEncoder(**config.model).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model'])
        self.model.eval()

    def compute_embedding(self, wav_file):
        mel_spec = self.ap.melspectrogram(self.ap.load_wav(wav_file, sr=self.ap.sample_rate)).T
        mel_spec = torch.FloatTensor(mel_spec[None, :, :])
        mel_spec = mel_spec.to(self.device)
        embedd = self.model.compute_embedding(mel_spec)
        embedd = embedd.detach().cpu().numpy()

        return embedd
    

def main(args):
    config = load_config(args.config_path)
    speech_embedding = SpeechEmbedding(config, args.model_path)

    # Compute speaker embeddings
    wav_file = args.input_path
    embedd = speech_embedding.compute_embedding(wav_file)
    print(embedd)
    print(embedd.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="pretrained_model/best_model.pth.tar", required=False)
    parser.add_argument('--config_path', type=str, default="pretrained_model/config.json", required=False)
    parser.add_argument('--output_path', type=str, default="outputs/", required=False)
    parser.add_argument('--input_path', type=str, default="outputs/", required=False)
    args = parser.parse_args()

    main(args)