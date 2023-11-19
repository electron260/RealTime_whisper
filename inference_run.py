from fast_whisper.fast_whisper import WhisperModel
from time import time
#from fast_whisper.utils import decode_audio
import whisper
import pandas as pd 
from pywhispercpp.model import Model
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import torch
from typing import BinaryIO, List, NamedTuple, Optional, Tuple, Union, Iterable
import gc 
import io 
import numpy as np 
import av
import itertools
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline



def _ignore_invalid_frames(frames):
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()

    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)

        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()

    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)

def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    # It appears that some objects related to the resampler are not freed
    # unless the garbage collector is manually run.
    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

    return audio

if __name__ == '__main__' :
        
    model_size = "medium.en"
    test = True
    if test :

        inference_time = pd.DataFrame(columns = ['whisper_implementation', '3sec_time', '6sec_time', '10sec_time', '2min11_time'])
        inf_time_dict = dict()
        
        audio_3 = decode_audio("audio/3sec.wav")
        audio_6 = decode_audio("audio/6sec.wav")
        audio_10 = decode_audio("audio/10sec.wav")
        audio_2min11 = decode_audio("audio/2min11.wav") 
        
       

        whisper_classic = False
        distil_whisper = True

        device = 'cuda'

        if whisper_classic :
            print('**************** CLASSIC WHISPER ****************')
            model_classic = whisper.load_model("medium.en").to('cuda')
            start = time()
            with torch.cuda.device(device):
                transcript_3 = model_classic.transcribe(audio_3, language="English")
            print(transcript_3['text'])
            inference_time['3sec'] = time() - start
            print(f"Time taken for 3 sec : {inference_time['3sec']}")

            start = time()
            with torch.cuda.device(device):
                transcript_6 = model_classic.transcribe(audio_6, language="English")
            print(transcript_6['text'])
            inference_time['6sec'] = time() - start
            print(f"Time taken for 6 sec : {inference_time['6sec']}")

            start = time()
            with torch.cuda.device(device):
                transcript_10 = model_classic.transcribe(audio_10, language="English")
            print(transcript_10['text'])
            inference_time['10sec'] = time() - start
            print(f"Time taken for 10 sec : {inference_time['10sec']}")

            start = time()
            with torch.cuda.device(device):
                transcript_2min11 = model_classic.transcribe(audio_2min11, language="English")
            print(transcript_2min11['text'])
            inference_time['2min11'] = time() - start
            print(f"Time taken for 2min11 sec : {inference_time['2min11']}")
            
            print(inference_time['3sec'])
            inference_time.at[len(inference_time)] = ['classic_whisper', inference_time['3sec'],inference_time['6sec'],inference_time['10sec'],inference_time['2min11']]
            
            torch.cuda.empty_cache()

        if distil_whisper :
            inf_time_dict = dict()
            print('**************** DISTIL WHISPER ****************')
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model_id = "distil-whisper/distil-medium.en"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=torch_dtype,
                device=device,
            )

            start = time()
            with torch.cuda.device(device):
                transcript_3 = pipe(audio_3)
            print(transcript_3['text'])
            inf_time_dict['3sec'] = time() - start
            print(f"Time taken for 3 sec : {inf_time_dict['3sec']}")

            start = time()
            with torch.cuda.device(device):
                transcript_6 = pipe(audio_6)
            print(transcript_6['text'])
            inf_time_dict['6sec'] = time() - start
            print(f"Time taken for 6 sec : {inf_time_dict['6sec']}")

            start = time()
            with torch.cuda.device(device):
                transcript_10 = pipe(audio_10)
            print(transcript_10['text'])
            inf_time_dict['10sec'] = time() - start
            print(f"Time taken for 10 sec : {inf_time_dict['10sec']}")

            print(inf_time_dict['3sec'])
            #inference_time.at[len(inference_time)] = ['distil_whisper', inf_time_dict['3sec'],inf_time_dict['6sec'],inf_time_dict['10sec'], 'XXX']
           
            inf_time_dict = dict()
            torch.cuda.empty_cache()
        print('**************** FASTER WHISPER ****************')

        model = WhisperModel(model_size, device="cuda", compute_type="int8")

        model_vad = WhisperModel(model_size, vad_activation=True, device="cuda", compute_type="int8")


        start = time()
        segments_3 = model.transcribe("audio/3sec.wav", beam_size=5)
        for segment in segments_3:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['3sec'] = time() - start 
        print(f"Time taken for 3 sec : {inf_time_dict['3sec']}")

        start = time()
        segments_6 = model.transcribe("audio/6sec.wav", beam_size=5)
        for segment in segments_6:
            
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['6sec'] = time() - start
        print(f"Time taken for 6 sec : {inf_time_dict['6sec']}")

        start = time()
        segments_10 = model.transcribe("audio/10sec.wav", beam_size=5)
        for segment in segments_10:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['10sec'] = time() - start
        print(f"Time taken for 10 sec : {inf_time_dict['10sec']}")

        start = time()
        segments_2min11 = model.transcribe("audio/2min11.wav", beam_size=5)
        for segment in segments_2min11:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['2min11'] = time() - start
        print(f"Time taken for 2min11 sec : {inf_time_dict['2min11']}")

        inference_time.loc[len(inference_time)] = ['faster_whisper', inf_time_dict['3sec'],inf_time_dict['6sec'],inf_time_dict['10sec'],inf_time_dict['2min11']]
        
        
        print('**************** FASTER WHISPER VAD ****************')
        start = time()
        segments_3_vad = model_vad.transcribe("audio/3sec.wav", beam_size=5)
        for segment in segments_3_vad:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['3sec'] = time() - start
        print(f"Time taken for 3 sec : {inf_time_dict['3sec']}")

        start = time()
        segments_6_vad = model_vad.transcribe("audio/6sec.wav", beam_size=5)
        for segment in segments_6_vad:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['6sec'] = time() - start
        print(f"Time taken for 6 sec : {inf_time_dict['6sec']}")

        start = time()
        segments_10_vad = model_vad.transcribe("audio/10sec.wav", beam_size=5)
        for segment in segments_10_vad:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['10sec'] = time() - start
        print(f"Time taken for 10 sec : {inf_time_dict['10sec']}")


        start = time()
        segments_2min11_vad = model_vad.transcribe("audio/2min11.wav", beam_size=5)
        for segment in segments_2min11_vad:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        inf_time_dict['2min11'] = time() - start
        print(f"Time taken for 2min11 sec : {inf_time_dict['2min11']}")

        inference_time.loc[len(inference_time)] = ['faster_whisper_vad', inf_time_dict['3sec'],inf_time_dict['6sec'],inf_time_dict['10sec'],inf_time_dict['2min11']]


        torch.cuda.empty_cache()
        print('**************** WHISPER CPP ****************')

        model_cpp = Model('medium.en', n_threads=6)

        start = time()
        segments = model_cpp.transcribe('audio/3sec.wav', speed_up=True)
        for segment in segments:
            print(segment.text)
        inf_time_dict['3sec'] = time() - start
        print(f"Time taken for 3 sec : {inf_time_dict['3sec']}")

        start = time()
        segments = model_cpp.transcribe('audio/6sec.wav', speed_up=True)
        for segment in segments:
            print(segment.text)
        inf_time_dict['6sec'] = time() - start
        print(f"Time taken for 6 sec : {inf_time_dict['6sec']}")

        start = time()
        segments = model_cpp.transcribe('audio/10sec.wav', speed_up=True)
        for segment in segments:
            print(segment.text)
        inf_time_dict['10sec'] = time() - start
        print(f"Time taken for 10 sec : {inf_time_dict['10sec']}")

        start = time()
        segments = model_cpp.transcribe('audio/2min11.wav', speed_up=True)
        for segment in segments:
            print(segment.text)
        inf_time_dict['2min11'] = time() - start
        print(f"Time taken for 2min11 sec : {inf_time_dict['2min11']}")

        inference_time.loc[len(inference_time)] = ['whisper_cpp', inf_time_dict['3sec'],inf_time_dict['6sec'],inf_time_dict['10sec'],inf_time_dict['2min11']]


        torch.cuda.empty_cache()
        print('**************** WHISPER JAX ****************')


        pipeline = FlaxWhisperPipline("openai/whisper-medium.en", dtype=jnp.bfloat16)



        start = time()
        text = pipeline('audio/3sec.wav')
        print(text)
        inf_time_dict['3sec'] = time() - start
        print(f"Time taken for 3 sec : {inf_time_dict['3sec']}")

        start = time()
        text = pipeline('audio/6sec.wav')
        print(text)
        inf_time_dict['6sec'] = time() - start
        print(f"Time taken for 6 sec : {inf_time_dict['6sec']}")

        start = time()
        text = pipeline('audio/10sec.wav')
        print(text)
        inference_time['10sec'] = time() - start
        print(f"Time taken for 10 sec : {inference_time['10sec']}")

        start = time()
        text = pipeline('audio/2min11.wav')
        print(text)
        inf_time_dict['2min11'] = time() - start
        print(f"Time taken for 2min11 sec : {inf_time_dict['2min11']}")

        inference_time.loc[len(inference_time)] = ['whisper_cpp', inf_time_dict['3sec'],inf_time_dict['6sec'],inf_time_dict['10sec'],inf_time_dict['2min11']]



        print('saving inference time ...')
        inference_time.to_csv(index = False)


    else :

        device = 'cuda'
        audio_test = decode_audio("audio/test.wav")

        print('**************** CLASSIC WHISPER ****************')
        model_classic = whisper.load_model("medium.en").to('cuda')
        start = time()
        with torch.cuda.device(device):
            transcript_test = model_classic.transcribe(audio_test, language="English")
        print(transcript_test['text'])

        print(f"Time taken for 3 sec CLASSIC: {time()- start}")

        torch.cuda.empty_cache()
        print('**************** FASTER WHISPER ****************')

        model = WhisperModel(model_size, device="cuda", compute_type="int8")

        model_vad = WhisperModel(model_size, vad_activation=True, device="cuda", compute_type="int8")


        start = time()
        segments_test = model.transcribe("audio/test.wav", beam_size=5)
        for segment in segments_test:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print(f"Time taken for 3 sec FASTER: {time() - start }")


        print('**************** FASTER WHISPER VAD ****************')
        start = time()
        segments_test_vad = model_vad.transcribe("audio/test.wav", beam_size=5)
        for segment in segments_test_vad:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print(f"Time taken for 3 sec FASTER VAD : {time() - start}")


        torch.cuda.empty_cache()
        print('**************** WHISPER CPP ****************')

        model_cpp = Model('medium.en', n_threads=6)

        start = time()
        segments = model_cpp.transcribe('audio/test.wav', speed_up=True)
        for segment in segments:
            print(segment.text)

        print(f"Time taken for 3 sec CPP  : {time() - start}")
