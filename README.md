# Speaker Encoder


This repository contains the Speaker Encoder model of <a href="https://github.com/mozilla/TTS">Mozilla TTS</a> repository without additional modules for easy-to-use computation of speech embeddings.

### Steps
- Clone the repository<br>
- Download a pretrained speaker encoder model from here: https://github.com/mozilla/TTS/wiki/Released-Models <br>
**Preferred**: Speaker-Encoder by @mueller91

- Copy files `config.json` and `best_model.pth.tar` to the folder `pretrained_model` 

- Run `python compute_embedding.py --input_path WAV_PATH` by specifying path to a wav file

