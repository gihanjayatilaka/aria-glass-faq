import argparse
import asyncio
import os
import signal
import sys
import threading
import time
from math import gcd

import numpy as np
import scipy.signal

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from google import genai
from google.genai import types

from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord

# Gemini's Live API requires input audio as 16-bit PCM mono at exactly this rate.
GEMINI_AUDIO_SAMPLE_RATE_HZ = 16000

# AudioData.max_amplitude reports a fixed 2^31-1 (int32 full scale), not the
# actual signal range, so dividing by it left real audio far too quiet to be
# heard. Observed raw samples peak around +-1.4M, which fits a 24-bit
# full-scale reference (Aria's mic array is 24-bit) with headroom to spare.
ARIA_AUDIO_FULL_SCALE = float(2**23 - 1)

# Gemini's server-side automatic VAD detected our first spoken utterance
# correctly but then never detected any further speech for the rest of the
# session, despite continuous, clearly-above-ambient audio still arriving
# (confirmed via peak-level logging in streamAudioToGemini.py). The
# documented workaround is to disable it and signal turn boundaries
# manually. Ambient room noise measured ~900-1300/32767, real speech
# measured 18000-32767, so this threshold sits safely above ambient with a
# wide margin from speech.
LOCAL_VAD_THRESHOLD_PCM16 = 3000
LOCAL_VAD_SILENCE_HANGOVER_S = 0.8


class LocalVad:
    """Manually detects speech start/end from raw signal energy and signals
    Gemini's turn boundaries explicitly (activity_start/activity_end),
    instead of relying on the server's own automatic VAD -- which in testing
    correctly detected the first utterance but then stopped detecting any
    further speech for the rest of the session, despite clearly-above-ambient
    audio continuing to arrive. Requires automatic_activity_detection to be
    disabled in the session config.

    With automatic VAD off, the server's own audio buffering/segmentation is
    bypassed entirely -- the client owns it. audio_callback only forwards
    audio while is_active(), i.e. inside a declared activity window.

    Note: even with this, a single Live session has only ever produced one
    turn's worth of output in testing (see streamAudioToGemini.py and
    https://github.com/googleapis/python-genai/issues/1391) -- this script
    exists to keep testing/using the single-session approach anyway,
    alongside speechToText.py's per-utterance approach which sidesteps the
    issue entirely."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._last_loud_t = 0.0

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def process(self, peak: int, session, loop, stats):
        now = time.monotonic()
        with self._lock:
            loud = peak > LOCAL_VAD_THRESHOLD_PCM16
            if loud:
                self._last_loud_t = now
            if not self._active and loud:
                self._active = True
                print("[local-vad] activity start", file=sys.stderr)
                submit(
                    session.send_realtime_input(activity_start=types.ActivityStart()),
                    loop,
                    "activity_start",
                    stats,
                )
            elif self._active and (now - self._last_loud_t) > LOCAL_VAD_SILENCE_HANGOVER_S:
                self._active = False
                print("[local-vad] activity end", file=sys.stderr)
                submit(
                    session.send_realtime_input(activity_end=types.ActivityEnd()),
                    loop,
                    "activity_end",
                    stats,
                )


class Stats:
    def __init__(self):
        self.start_time = time.monotonic()
        self.audio_chunks_sent = 0
        self.audio_bytes_sent = 0
        self.audio_seconds_sent = 0.0
        self.send_errors = 0
        # Max |sample| seen since the last print_stats() line -- shows
        # whether real voice-level signal arrived, independent of whether
        # Gemini transcribed/responded to it. Reset after each print.
        self.audio_peak_pcm16 = 0

    def format(self) -> str:
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        line = (
            f"[stats] t={elapsed:6.1f}s | "
            f"audio: {self.audio_chunks_sent} chunks, {self.audio_seconds_sent:.1f}s "
            f"({self.audio_bytes_sent / 1024:.0f} KB, peak-since-last-print={self.audio_peak_pcm16}/32767) | "
            f"errors: {self.send_errors}"
        )
        self.audio_peak_pcm16 = 0
        return line


async def print_stats(stats: Stats, interval: float):
    while True:
        await asyncio.sleep(interval)
        print(stats.format(), file=sys.stderr, flush=True)


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

    if source_sample_rate != GEMINI_AUDIO_SAMPLE_RATE_HZ:
        divisor = gcd(GEMINI_AUDIO_SAMPLE_RATE_HZ, source_sample_rate)
        normalized = scipy.signal.resample_poly(
            normalized,
            GEMINI_AUDIO_SAMPLE_RATE_HZ // divisor,
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


def submit(coro, loop: asyncio.AbstractEventLoop, label: str, stats: Stats):
    # The SDK's callback thread can fire once more right as the event loop is
    # shutting down (e.g. on Ctrl+C); drop the send instead of crashing.
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        coro.close()
        return

    def on_done(f):
        if f.exception():
            stats.send_errors += 1
            print(f"\n[{label} send error] {f.exception()}", file=sys.stderr)

    future.add_done_callback(on_done)


def setup_streaming_receiver(device, session, loop, stats, local_vad, args):
    config = sdk_gen2.HttpServerConfig()
    config.address = "0.0.0.0"
    config.port = 6768

    stream_receiver = receiver.StreamReceiver()
    stream_receiver.set_server_config(config)

    audio_mute_warned = {"done": False}
    audio_debug_printed = {"done": False}

    def audio_callback(
        audio_data: AudioData, audio_record: AudioDataRecord, num_channels: int
    ):
        if audio_record.audio_muted and not audio_mute_warned["done"]:
            audio_mute_warned["done"] = True
            print(
                "[warning] Aria reports audio_muted=1 — Gemini will hear silence "
                "until the mic is unmuted on the device.",
                file=sys.stderr,
            )
        debug = not audio_debug_printed["done"]
        audio_debug_printed["done"] = True
        pcm16 = downmix_to_pcm16(audio_data, num_channels, args.audio_sample_rate, debug=debug)
        if len(pcm16) > 0:
            chunk_peak = int(np.abs(np.frombuffer(pcm16, dtype=np.int16)).max())
            stats.audio_peak_pcm16 = max(stats.audio_peak_pcm16, chunk_peak)
            local_vad.process(chunk_peak, session, loop, stats)

        if not local_vad.is_active():
            # Automatic VAD is disabled, so the server does no buffering or
            # segmentation of its own -- only forward audio that falls inside
            # a declared activity_start/activity_end window.
            return

        stats.audio_chunks_sent += 1
        stats.audio_bytes_sent += len(pcm16)
        stats.audio_seconds_sent += (
            len(audio_record.capture_timestamps_ns) / args.audio_sample_rate
        )
        submit(
            session.send_realtime_input(
                audio=types.Blob(
                    data=pcm16,
                    mime_type=f"audio/pcm;rate={GEMINI_AUDIO_SAMPLE_RATE_HZ}",
                )
            ),
            loop,
            "audio",
            stats,
        )

    stream_receiver.register_audio_callback(audio_callback)
    stream_receiver.start_server()

    return stream_receiver


async def print_transcriptions(session, log_path: str):
    # Prints a transcript of what *we* said (input_transcription), not
    # Gemini's spoken reply (output_transcription) -- this script's job is
    # transcription, not conversation, so the reply audio/text is ignored.
    buffer = []
    try:
        async for response in session.receive():
            server_content = getattr(response, "server_content", None)

            input_transcription = getattr(server_content, "input_transcription", None)
            if input_transcription and input_transcription.text:
                print(input_transcription.text, end="", flush=True)
                buffer.append(input_transcription.text)

            if server_content and getattr(server_content, "turn_complete", False) and buffer:
                print()
                with open(log_path, "a") as f:
                    f.write("".join(buffer) + "\n\n")
                buffer = []
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Without this, session.receive() dying (e.g. the Gemini websocket
        # closing) leaves this task silently dead: no more output ever
        # appears, with no error visible anywhere.
        print(f"\n[print_transcriptions stopped: {e!r}]", file=sys.stderr)


async def run_gemini_session(device, args):
    client = genai.Client(api_key=args.api_key) if args.api_key else genai.Client()
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=args.system_instruction,
        # See LocalVad's docstring: automatic VAD stopped detecting speech
        # after the first utterance in testing, despite continuous
        # clearly-above-ambient audio still arriving. Disable it and drive
        # turn boundaries ourselves.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    )

    stats = Stats()
    local_vad = LocalVad()
    async with client.aio.live.connect(model=args.model, config=config) as session:
        loop = asyncio.get_running_loop()
        stream_receiver = setup_streaming_receiver(device, session, loop, stats, local_vad, args)
        stats_task = asyncio.create_task(print_stats(stats, args.stats_interval))
        response_task = asyncio.create_task(print_transcriptions(session, args.log))
        try:
            if args.duration:
                await asyncio.sleep(args.duration)
            else:
                # Run indefinitely; sending (audio callback) and receiving
                # (response_task, printing to terminal) both happen
                # independently in the background regardless of this wait.
                await asyncio.Event().wait()
        except KeyboardInterrupt:
            # A second Ctrl+C landing mid-cleanup (e.g. inside the SDK's own
            # websocket close) corrupts asyncio's internal state, so stop
            # listening for SIGINT once we've started shutting down.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise
        finally:
            stats_task.cancel()
            response_task.cancel()
            device.stop_streaming()
            stream_receiver.stop_server()
            print(stats.format(), file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe Aria's mic audio using one continuous Gemini Live API session."
    )
    parser.add_argument("--profile", type=str, default="profile9")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-live-preview")
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
        "--stats-interval",
        type=float,
        default=5.0,
        help="Seconds between stats lines printed to stderr.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=os.environ.get("ARIA_TEMP_LOG", "/home/gihan/aria/gemini_output.log"),
        help="File transcriptions are appended to, for view_gemini_output.py to tail. "
        "Defaults to the ARIA_TEMP_LOG env var if set.",
    )
    parser.add_argument(
        "--system-instruction",
        type=str,
        default=(
            "You are a live transcription service. Do not converse or "
            "answer questions; simply listen."
        ),
    )
    return parser.parse_args()


async def main_async(args):
    device = device_streaming(args.profile)
    await run_gemini_session(device, args)


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
