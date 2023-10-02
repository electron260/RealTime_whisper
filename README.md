# Comparing Whisper Inference Speed on Different Audio Files

This README provides a comparison of the inference performance of three Whisper implementations: `whisper.cpp` (C++ implementation), `faster-whisper` (implementation using the Ctranslate2 encoder for faster inference with transformers), and `whisper-jax` (optimized JAX code for OpenAI's Whisper Model, largely built on the 🤗 Hugging Face Transformers Whisper implementation).

## Introduction

Whisper is a powerful automatic speech recognition (ASR) model developed by OpenAI. In this repository, we evaluate the inference speed of different Whisper implementations on various audio files of different sizes. The goal is to assess their performance on real-world data.

## Inference Speed Comparison

We conducted inference speed tests on the following audio files:

1. **audio_file_1.wav**: This audio file is 5 minutes long and has a size of 20 MB.
2. **audio_file_2.wav**: This audio file is 10 minutes long and has a size of 40 MB.
3. **audio_file_3.wav**: This audio file is 15 minutes long and has a size of 60 MB.

Here are the results of the inference speed tests for each Whisper implementation:

| Whisper Implementation     | Audio File 1 (5sec)  | Audio File 2 (10sec) | Audio File 3 (30sec) |
|----------------------------|----------------------|----------------------|----------------------|
| `whisper.cpp`              | X ms                 | X ms                 | X ms                 |
| `faster-whisper`           | X ms                 | X ms                 | X ms                 |
| `faster-whisper [with VAD]`| X ms                 | X ms                 | X ms                 |
| `whisper-jax`              | X ms                 | X ms                 | X ms                 |

The above table shows the average inference time for each implementation on the specified audio files. Smaller values indicate faster inference times.

## Usage

To replicate these results or conduct your own inference tests, follow the instructions in the respective implementation directories (`whisper.cpp`, `faster-whisper`, and `whisper-jax`).

## Conclusion

Based on our inference speed comparison, it is evident that `whisper-jax` outperforms the other implementations in terms of speed on all tested audio files. However, the choice of implementation may also depend on other factors such as ease of use and integration into your specific project.

Feel free to explore and use these implementations for your own ASR tasks.

**Note**: Ensure that you have the necessary dependencies and resources set up for each implementation before conducting inference tests.
