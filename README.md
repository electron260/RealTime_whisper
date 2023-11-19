# Comparing Whisper Inference Speed on Different Audio Files

This README provides a comparison of the inference performance of three Whisper implementations: `faster-whisper` (implementation using the Ctranslate2 encoder for faster inference with transformers) with and without voice activity detection, and the distiled model of Whisper, largely built on the 🤗 Hugging Face Transformers Whisper implementation).

## Introduction

Whisper is a powerful automatic speech recognition (ASR) model developed by OpenAI. In this repository, we evaluate the inference speed of different Whisper implementations on various audio files of different sizes. The goal is to assess their performance on real-world data.
The Whisper size model used for all implementations is the `medium.en` model only focusing on English language.

## Inference Speed Comparison

We conducted inference speed tests on the following audio files:

1. **audio_file_1.wav**: This audio file is 3 sec long. 
2. **audio_file_2.wav**: This audio file is 6 sec long.
3. **audio_file_3.wav**: This audio file is 10 sec long.
3. **audio_file_4.wav**: This audio file is 2min and 11sec long.

Here are the results of the inference speed tests for each Whisper implementation:

| Whisper Implementation     | Audio File 1 (3sec)  | Audio File 2 (6sec) | Audio File 3 (10sec ) | Audio File 4 (2min11)|
|----------------------------|----------------------|----------------------|----------------------|----------------------|
| `faster-whisper`           | 1.17 sec             | 1.64 sec             | 2.04 sec             | 20.78 sec            |
| `faster-whisper [with VAD]`| 1.36 sec             | 1.69 sec             | 2.14 sec             | 23.22 sec            |
| `distiled-whisper`              | 1.77 sec                 | 1.42 sec                | 1.46 sec                 | XXX                 |

The above table shows the average inference time for each implementation on the specified audio files. Smaller values indicate faster inference times.

