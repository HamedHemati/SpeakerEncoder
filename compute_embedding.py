from limit_threads import *
import argparse
import glob
import os
import numpy as np
import pickle
import random
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

    emb_dict = {}
    # Compute speaker embeddings
    if os.path.isfile(args.input_path):
        wav_file = args.input_path
        embedd = speech_embedding.compute_embedding(wav_file)
        embedd = embedd[0]
    else:
        spk_list = os.listdir(args.input_path)
        for spk_itr, spk_name in enumerate(spk_list):
            print(f"Spaker {spk_itr}/{len(spk_list)}::")
            wav_files = glob.glob(os.path.join(args.input_path, spk_name, "*.wav"))
            if len(wav_files) == 0:
                continue

            random.shuffle(wav_files)
            wav_files = wav_files[:20]
            all_embdes = []
            for itr, wav_file in enumerate(wav_files):
                print(f"Computing embedding for file {itr}/{len(wav_files)}")
                embedd = speech_embedding.compute_embedding(wav_file)
                all_embdes.append(list(embedd[0]))
            embedd = np.mean(np.array(all_embdes), axis=0)
            emb_dict[spk_name] = embedd

    with open(os.path.join(args.output_path, f"{args.output_name}_emb.pkl"), "wb") as pkl_file:
        pickle.dump(emb_dict, pkl_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="pretrained_model/best_model.pth.tar", required=False)
    parser.add_argument('--config_path', type=str, default="pretrained_model/config.json", required=False)
    parser.add_argument('--output_path', type=str, default="outputs/", required=False)
    parser.add_argument('--input_path', type=str, default="outputs/", required=False)
    parser.add_argument('--output_name', type=str, required=True)

    args = parser.parse_args()

    main(args)