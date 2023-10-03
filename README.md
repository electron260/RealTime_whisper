# Comparing Whisper Inference Speed on Different Audio Files

This README provides a comparison of the inference performance of three Whisper implementations: `whisper.cpp` (C++ implementation), `faster-whisper` (implementation using the Ctranslate2 encoder for faster inference with transformers), and `whisper-jax` (optimized JAX code for OpenAI's Whisper Model, largely built on the 🤗 Hugging Face Transformers Whisper implementation).

## Introduction

Whisper is a powerful automatic speech recognition (ASR) model developed by OpenAI. In this repository, we evaluate the inference speed of different Whisper implementations on various audio files of different sizes. The goal is to assess their performance on real-world data.
The Whisper size model used for all implementations is the `medium.en` model only focusing on English language.

## Inference Speed Comparison

We conducted inference speed tests on the following audio files:

1. **audio_file_1.wav**: This audio file is 5 sec long and has a size of 20 MB.
2. **audio_file_2.wav**: This audio file is 10 sec long and has a size of 40 MB.
3. **audio_file_3.wav**: This audio file is 30 sec long and has a size of 60 MB.

Here are the results of the inference speed tests for each Whisper implementation:

| Whisper Implementation     | Audio File 1 (3sec)  | Audio File 2 (6sec) | Audio File 3 (10sec ) | Audio File 4 (2min11)|
|----------------------------|----------------------|----------------------|----------------------|----------------------|
| `whisper.cpp`              | X ms                 | X ms                 | X ms                 | X ms                 |
| `faster-whisper`           | 1.68 sec             | 1.58 sec             | 1.97 sec             | 20.75 sec            |
| `faster-whisper [with VAD]`| 1.31 sec             | 1.68 sec             | 2.11 sec             | 23.74 sec            |
| `whisper-jax`              | X ms                 | X ms                 | X ms                 | X ms                 |

The above table shows the average inference time for each implementation on the specified audio files. Smaller values indicate faster inference times.

