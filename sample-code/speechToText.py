import argparse
import asyncio
import io
import os
import sys
import threading
import time
import wave
from math import gcd

import numpy as np
import scipy.signal

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from google import genai
from google.genai import types

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord

# Gemini's transcription models expect a conventional sample rate; 16kHz is
# what Aria's mic is measured at (see diagnose_audio.py), so no resampling is
# actually needed here, but the conversion is kept in case that ever changes.
TRANSCRIBE_SAMPLE_RATE_HZ = 16000

# AudioData.max_amplitude reports a fixed 2^31-1 (int32 full scale), not the
# actual signal range, so dividing by it left real audio far too quiet to be
# heard. Observed raw samples peak around +-1.4M, which fits a 24-bit
# full-scale reference (Aria's mic array is 24-bit) with headroom to spare.
ARIA_AUDIO_FULL_SCALE = float(2**23 - 1)

# Local, energy-based speech segmentation: ambient room noise measured
# ~900-1300/32767, real speech measured 18000-32767/32767, so this threshold
# sits safely above ambient with a wide margin from speech. Rather than
# streaming continuously into one long-lived Live API session (whose
# multi-turn behavior has been unreliable in testing -- see streamToGemini.py
# and streamAudioToGemini.py), each detected utterance here is buffered
# locally and sent as one independent, stateless transcription request.
VAD_THRESHOLD_PCM16 = 3000
VAD_SILENCE_HANGOVER_S = 0.8
MIN_UTTERANCE_SECONDS = 0.3


class SpeechSegmenter:
    """Buffers mic audio locally and hands off one complete utterance (from
    speech onset to a sustained silence) at a time, instead of streaming
    continuously to a stateful session."""

    def __init__(self, on_utterance):
        self._lock = threading.Lock()
        self._active = False
        self._last_loud_t = 0.0
        self._buffer = []
        self._on_utterance = on_utterance

    def process(self, pcm16: bytes, peak: int):
        now = time.monotonic()
        with self._lock:
            loud = peak > VAD_THRESHOLD_PCM16
            if loud:
                self._last_loud_t = now

            if not self._active and loud:
                self._active = True
                self._buffer = [pcm16]
                print("[vad] speech start", file=sys.stderr)
                return

            if self._active:
                self._buffer.append(pcm16)
                if now - self._last_loud_t > VAD_SILENCE_HANGOVER_S:
                    self._active = False
                    buffered = b"".join(self._buffer)
                    self._buffer = []
                    print(f"[vad] speech end ({len(buffered) / 2 / TRANSCRIBE_SAMPLE_RATE_HZ:.1f}s)", file=sys.stderr)
                    if len(buffered) / 2 / TRANSCRIBE_SAMPLE_RATE_HZ >= MIN_UTTERANCE_SECONDS:
                        self._on_utterance(buffered)


def downmix_to_pcm16(
    audio_data: AudioData, num_channels: int, source_sample_rate: int, debug: bool = False
) -> bytes:
    samples = np.array(audio_data.data, dtype=np.float64)
    # Confirmed by listening to diagnose_audio.py's output: the buffer is
    # interleaved per-sample (ch0, ch1, ..., chN, ch0, ch1, ...), not laid out
    # as one contiguous block per channel.
    mono = (
        samples.reshape(-1, num_channels).mean(axis=1) if num_channels > 1 else samples
    )

    normalized = np.clip(mono / ARIA_AUDIO_FULL_SCALE, -1.0, 1.0)

    if source_sample_rate != TRANSCRIBE_SAMPLE_RATE_HZ:
        divisor = gcd(TRANSCRIBE_SAMPLE_RATE_HZ, source_sample_rate)
        normalized = scipy.signal.resample_poly(
            normalized,
            TRANSCRIBE_SAMPLE_RATE_HZ // divisor,
            source_sample_rate // divisor,
        )

    pcm16 = (normalized * 32767.0).astype(np.int16)

    if debug:
        print(
            f"[debug] first audio chunk: raw samples min={mono.min():.1f} max={mono.max():.1f} "
            f"max_amplitude={audio_data.max_amplitude} normalized min={normalized.min():.4f} "
            f"max={normalized.max():.4f} pcm16 min={pcm16.min()} max={pcm16.max()}",
            file=sys.stderr,
        )

    return pcm16.tobytes()


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    return buf.getvalue()


def device_streaming(profile_name: str):
    device_client = sdk_gen2.DeviceClient()
    config = sdk_gen2.DeviceClientConfig()
    device_client.set_client_config(config)
    device = device_client.connect()

    streaming_config = sdk_gen2.HttpStreamingConfig()
    streaming_config.profile_name = profile_name
    device.set_streaming_config(streaming_config)

    # A prior run that crashed/was killed before calling stop_streaming() leaves
    # the device's session open; clear it before starting a fresh one.
    if device.is_streaming():
        device.stop_streaming()

    device.start_streaming()
    return device


async def transcribe_utterance(client, model: str, pcm16: bytes, log_path: str):
    wav_bytes = pcm16_to_wav_bytes(pcm16, TRANSCRIBE_SAMPLE_RATE_HZ)
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                "Transcribe this audio verbatim. Output only the transcription, "
                "with no extra commentary. If there is no discernible speech, output nothing.",
            ],
        )
        text = (response.text or "").strip()
        if text:
            print(text)
            # Also append to the shared log so view_gemini_output.py (which
            # just tails this file) can show it in a separate terminal.
            with open(log_path, "a") as f:
                f.write(text + "\n\n")
    except Exception as e:
        print(f"[transcription error] {e!r}", file=sys.stderr)


def setup_streaming_receiver(device, loop, client, args):
    config = sdk_gen2.HttpServerConfig()
    config.address = "0.0.0.0"
    config.port = 6768

    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(config)

    def on_utterance(buffered_pcm16: bytes):
        try:
            asyncio.run_coroutine_threadsafe(
                transcribe_utterance(client, args.model, buffered_pcm16, args.log), loop
            )
        except RuntimeError:
            pass  # event loop already shutting down (e.g. Ctrl+C)

    segmenter = SpeechSegmenter(on_utterance)

    audio_mute_warned = {"done": False}
    audio_debug_printed = {"done": False}

    def audio_callback(
        audio_data: AudioData, audio_record: AudioDataRecord, num_channels: int
    ):
        if audio_record.audio_muted and not audio_mute_warned["done"]:
            audio_mute_warned["done"] = True
            print(
                "[warning] Aria reports audio_muted=1 — audio will be silence "
                "until the mic is unmuted on the device.",
                file=sys.stderr,
            )
        debug = not audio_debug_printed["done"]
        audio_debug_printed["done"] = True
        pcm16 = downmix_to_pcm16(audio_data, num_channels, args.audio_sample_rate, debug=debug)
        if len(pcm16) == 0:
            return

        peak = int(np.abs(np.frombuffer(pcm16, dtype=np.int16)).max())
        segmenter.process(pcm16, peak)

    stream_receiver.register_audio_callback(audio_callback)
    stream_receiver.start_server()

    return stream_receiver


async def main_async(args):
    client = genai.Client(api_key=args.api_key) if args.api_key else genai.Client()
    device = device_streaming(args.profile)
    loop = asyncio.get_running_loop()
    stream_receiver = setup_streaming_receiver(device, loop, client, args)
    try:
        if args.duration:
            await asyncio.sleep(args.duration)
        else:
            await asyncio.Event().wait()
    finally:
        device.stop_streaming()
        stream_receiver.stop_server()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe Aria's mic audio using the Gemini API.")
    parser.add_argument("--profile", type=str, default="profile9")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key. Defaults to the GEMINI_API_KEY/GOOGLE_API_KEY env var.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to stream before stopping. Omit to run until Ctrl+C.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        help="Sample rate (Hz) of the Aria microphone stream (measured via diagnose_audio.py).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=os.environ.get("ARIA_TEMP_LOG", "/home/gihan/aria/gemini_output.log"),
        help="File transcriptions are appended to, for view_gemini_output.py to tail. "
        "Defaults to the ARIA_TEMP_LOG env var if set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
