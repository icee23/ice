from __future__ import annotations
import pyaudio
import io
from io import BytesIO
import math
import collections
import audioop
from speech_recognition.exceptions import SetupError
import wave
import queue
import threading
import os
import datetime
import whisper
import numpy as np
import soundfile as sf
import torch
import time
import speech_recognition as sr

os.environ['CUDA_VISIBLE_DEVICES']="0"
Recognizer = sr.Recognizer()
print("1")

class AudioData(object):
    """
    Creates a new ``AudioData`` instance, which represents mono audio data.

    The raw audio data is specified by ``frame_data``, which is a sequence of bytes representing audio samples. This is the frame data structure used by the PCM WAV format.

    The width of each sample, in bytes, is specified by ``sample_width``. Each group of ``sample_width`` bytes represents a single audio sample.

    The audio data is assumed to have a sample rate of ``sample_rate`` samples per second (Hertz).

    Usually, instances of this class are obtained from ``recognizer_instance.record`` or ``recognizer_instance.listen``, or in the callback for ``recognizer_instance.listen_in_background``, rather than instantiating them directly.
    """

    def __init__(self, frame_data, sample_rate, sample_width):
        assert sample_rate > 0, "Sample rate must be a positive integer"
        assert (
            sample_width % 1 == 0 and 1 <= sample_width <= 4
        ), "Sample width must be between 1 and 4 inclusive"
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.sample_width = int(sample_width)
    def get_raw_data(self, convert_rate=None, convert_width=None):
        """
        Returns a byte string representing the raw frame data for the audio represented by the ``AudioData`` instance.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        Writing these bytes directly to a file results in a valid `RAW/PCM audio file <https://en.wikipedia.org/wiki/Raw_audio_format>`__.
        """
        assert (
            convert_rate is None or convert_rate > 0
        ), "Sample rate to convert to must be a positive integer"
        assert convert_width is None or (
            convert_width % 1 == 0 and 1 <= convert_width <= 4
        ), "Sample width to convert to must be between 1 and 4 inclusive"

        raw_data = self.frame_data

        # make sure unsigned 8-bit audio (which uses unsigned samples) is handled like higher sample width audio (which uses signed samples)
        if self.sample_width == 1:
            raw_data = audioop.bias(
                raw_data, 1, -128
            )  # subtract 128 from every sample to make them act like signed samples

        # resample audio at the desired rate if specified
        if convert_rate is not None and self.sample_rate != convert_rate:
            raw_data, _ = audioop.ratecv(
                raw_data,
                self.sample_width,
                1,
                self.sample_rate,
                convert_rate,
                None,
            )

        # convert samples to desired sample width if specified
        if convert_width is not None and self.sample_width != convert_width:
            if (
                convert_width == 3
            ):  # we're converting the audio into 24-bit (workaround for https://bugs.python.org/issue12866)
                raw_data = audioop.lin2lin(
                    raw_data, self.sample_width, 4
                )  # convert audio into 32-bit first, which is always supported
                try:
                    audioop.bias(
                        b"", 3, 0
                    )  # test whether 24-bit audio is supported (for example, ``audioop`` in Python 3.3 and below don't support sample width 3, while Python 3.4+ do)
                except (
                    audioop.error
                ):  # this version of audioop doesn't support 24-bit audio (probably Python 3.3 or less)
                    raw_data = b"".join(
                        raw_data[i + 1 : i + 4]
                        for i in range(0, len(raw_data), 4)
                    )  # since we're in little endian, we discard the first byte from each 32-bit sample to get a 24-bit sample
                else:  # 24-bit audio fully supported, we don't need to shim anything
                    raw_data = audioop.lin2lin(
                        raw_data, self.sample_width, convert_width
                    )
            else:
                raw_data = audioop.lin2lin(
                    raw_data, self.sample_width, convert_width
                )

        # if the output is 8-bit audio with unsigned samples, convert the samples we've been treating as signed to unsigned again
        if convert_width == 1:
            raw_data = audioop.bias(
                raw_data, 1, 128
            )  # add 128 to every sample to make them act like unsigned samples again

        return raw_data

    def get_wav_data(self, convert_rate=None, convert_width=None):
        """
        Returns a byte string representing the contents of a WAV file containing the audio represented by the ``AudioData`` instance.

        If ``convert_width`` is specified and the audio samples are not ``convert_width`` bytes each, the resulting audio is converted to match.

        If ``convert_rate`` is specified and the audio sample rate is not ``convert_rate`` Hz, the resulting audio is resampled to match.

        Writing these bytes directly to a file results in a valid `WAV file <https://en.wikipedia.org/wiki/WAV>`__.
        """
        raw_data = self.get_raw_data(convert_rate, convert_width)
        sample_rate = (
            self.sample_rate if convert_rate is None else convert_rate
        )
        sample_width = (
            self.sample_width if convert_width is None else convert_width
        )

        # generate the WAV file contents
        with io.BytesIO() as wav_file:
            wav_writer = wave.open(wav_file, "wb")
            try:  # note that we can't use context manager, since that was only added in Python 3.4
                wav_writer.setframerate(sample_rate)
                wav_writer.setsampwidth(sample_width)
                wav_writer.setnchannels(1)
                wav_writer.writeframes(raw_data)
                wav_data = wav_file.getvalue()
            finally:  # make sure resources are cleaned up
                wav_writer.close()
        return wav_data

def rw(recdata):
    print(type(recdata))
    global Recognizer
    text = Recognizer.recognize_whisper(audio_data=recdata,show_dict=True,model="large-v2") # large-v3
    return text
    #if show_dict:
    #    if result['text']=='':
    #        return ''
    #    elif result['segments'][0]['no_speech_prob']<0.7:
    #        return result['text']
    #    else:
    #        return ''
    #else:
    #    return result["text"]
    #
SQ = queue.Queue()
SQ.put(0)
RQ = queue.Queue()
def recorder():
    audio = pyaudio.PyAudio()

    SAMPLE_RATE=16000
    CHUNK=1600
    phrase_threshold = 0.3 
    non_speaking_duration = 0.2
    energy_threshold=1000

    # Open the microphone stream
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,  # Use mono audio for speech recognition
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    seconds_per_buffer = float(CHUNK) / SAMPLE_RATE 
    phrase_buffer_count = int(math.ceil(phrase_threshold / seconds_per_buffer))  
    non_speaking_buffer_count = int(math.ceil(non_speaking_duration / seconds_per_buffer))

    global Recording
    Recording = True
    wav = []

    # Keep the program running while the stream is active
    while Recording:
        while Recording:
            audio_buffer = b""
            audio_data = collections.deque()
            
            # store audio input until the phrase starts
            while Recording:
                audio_buffer = stream.read(CHUNK)
                wav.append(audio_buffer)
                if len(audio_buffer) == 0: break  # reached end of the stream
                audio_data.append(audio_buffer)
                if len(audio_data) > non_speaking_buffer_count:  # ensure we only keep the needed amount of non-speaking buffers
                    audio_data.popleft()
                # detect whether speaking has started on audio input
                energy = audioop.rms(audio_buffer, 2)  # energy of the audio signal
                if energy > energy_threshold: break
                # dynamically adjust the energy threshold using asymmetric weighted average
                damping = 0.15 ** seconds_per_buffer  # account for different chunk sizes and rates
                target_energy = energy * 1.5
                energy_threshold = energy_threshold * damping + target_energy * (1 - damping)
            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0

            i = 0
            while Recording:
                i+=0.1
                audio_buffer = stream.read(CHUNK)
                wav.append(audio_buffer)
                if len(audio_buffer) == 0: break  # reached end of the stream
                audio_data.append(audio_buffer)
                phrase_count += 1
                # check if speaking has stopped for longer than the pause threshold on the audio input
                energy = audioop.rms(audio_buffer, 2)  # unit energy of the audio signal within the buffer
                if energy > energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count >= 2:  # end of the phrase
                    break
            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(audio_buffer) == 0: 
                #print("sec:"+str(i))
                #print("adlength:"+str(len(audio_data)))
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data

        
        RQ.put(audio_data)
        LOCAL_RECOG=threading.Thread(target=Recognize)
        LOCAL_RECOG.setDaemon(True)
        LOCAL_RECOG.start()
        #NET_RECOG=threading.Thread(target=Recognize_online)
        #NET_RECOG.setDaemon(True)
        #NET_RECOG.start()
        
def Recognize():
    audio_data = RQ.get()
    frame_data = b"".join(audio_data)
    rec_data = sr.AudioData(frame_data, 16000, 2)
    SQ.get()
    text = rw(rec_data)

    #if text == " Thank you.":
    #    import pyaudio._portaudio as pa
    #    now = datetime.datetime.now()
    #    path = str(now.month)+str(now.day)+"_"+str(now.hour)+str(now.minute)+str(now.second)
    #    wf = wave.open(path+".wav", 'wb')   # 開啟聲音記錄檔
    #    wf.setnchannels(1)        # 設定聲道
    #    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))  # 設定格式
    #    wf.setframerate(16000)              # 設定取樣頻率
    #    wf.writeframes(frame_data) # 存檔
    #    wf.close()
    #el
    if text != "":
        print(text)
        
    SQ.put(0)        

    

if __name__ == '__main__':
    recorder()