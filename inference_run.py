from fast_whisper.fast_whisper import WhisperModel
from time import time


if __name__ == '__main__' :
        
    model_size = "medium.en"

    model = WhisperModel(model_size, device="cuda", compute_type="int8")

    model_vad = WhisperModel(model_size, vad_activation=True, device="cuda", compute_type="int8")

    
    
    
    

    print('transcribing vad')
    
    
    
    

    start = time()
    segments_3 = model.transcribe("audio/3sec.wav", beam_size=5)
    for segment in segments_3:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 3 sec: ", time() - start, "s")

    start = time()
    segments_6 = model.transcribe("audio/6sec.wav", beam_size=5)
    for segment in segments_6:
        
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 6 sec: ", time() - start, "s")

    start = time()
    segments_10 = model.transcribe("audio/10sec.wav", beam_size=5)
    for segment in segments_10:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 10 sec: ", time() - start, "s")

    start = time()
    segments_2min11 = model.transcribe("audio/2min11.wav", beam_size=5)
    for segment in segments_2min11:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 2min11: ", time() - start, "s")


    start = time()
    segments_3_vad = model_vad.transcribe("audio/3sec.wav", beam_size=5)
    for segment in segments_3_vad:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 3 sec: ", time() - start, "s")

    start = time()
    segments_6_vad = model_vad.transcribe("audio/6sec.wav", beam_size=5)
    for segment in segments_6_vad:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 6 sec: ", time() - start, "s")

    start = time()
    segments_10_vad = model_vad.transcribe("audio/10sec.wav", beam_size=5)
    for segment in segments_10_vad:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 10 sec: ", time() - start, "s")

    start = time()
    segments_2min11_vad = model_vad.transcribe("audio/2min11.wav", beam_size=5)
    for segment in segments_2min11_vad:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    print("----------------------------------------durée 2min11: ", time() - start, "s")
