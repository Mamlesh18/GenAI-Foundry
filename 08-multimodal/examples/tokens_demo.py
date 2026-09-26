"""
How a model "sees" a picture and "hears" a sound -- it turns them into tokens.

A transformer only ever consumes a sequence of vectors. Text becomes vectors via
a tokenizer. The trick behind every multimodal model is that images and audio
get the same treatment:

    text   ->  sub-word pieces        ->  vectors
    image  ->  fixed-size PATCHES     ->  vectors      (An Image is Worth 16x16 Words)
    audio  ->  short time FRAMES      ->  vectors      (a spectrogram, sliced)
    video  ->  frames, then patches   ->  vectors      (and this is why video is expensive)

Once everything is a sequence of vectors, the SAME transformer machinery from
01-foundations applies. This script does each conversion for real -- a real
image, a real spectrogram -- and counts the tokens, because the token counts are
what decide your bill and your context limit.

    pip install numpy                 (transformers optional, for a real tokenizer)
    python tokens_demo.py
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# 1. Text
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "A multimodal model accepts more than one kind of input. The transformer "
    "itself does not care whether a vector came from a word, an image patch or "
    "a slice of sound."
)


def count_text_tokens(text):
    """Use a real tokenizer if one is cached locally; otherwise approximate."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        return len(tokenizer.encode(text)), "real tokenizer (MiniLM)"
    except Exception:                                            # noqa: BLE001
        # The usual rule of thumb for English: ~4 characters per token.
        return max(1, round(len(text) / 4)), "approximation (chars / 4)"


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def demo_text():
    section("1. TEXT -> TOKENS")
    count, how = count_text_tokens(SAMPLE_TEXT)
    print(f"  text      : {len(SAMPLE_TEXT)} characters, {len(SAMPLE_TEXT.split())} words")
    print(f"  tokens    : {count}   [{how}]")
    print(f"  per word  : {count / len(SAMPLE_TEXT.split()):.2f}")
    print(
        "\n  This is the baseline everything else gets compared against. Roughly\n"
        "  three-quarters of a token per character-heavy word, and cheap."
    )
    return count


# ---------------------------------------------------------------------------
# 2. Image
# ---------------------------------------------------------------------------

def make_image(width, height):
    """A real image array: a gradient with a few shapes, so it is not uniform."""
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack([
        (x / max(1, width - 1) * 255),
        (y / max(1, height - 1) * 255),
        ((x + y) % 64 * 4),
    ], axis=-1).astype(np.uint8)
    image[height // 4: height // 2, width // 4: width // 2] = [255, 40, 40]     # a square
    return image


def patchify(image, patch=16):
    """Cut the image into non-overlapping patch x patch squares -- exactly what a
    Vision Transformer does before the first layer."""
    height, width = image.shape[:2]
    rows, cols = height // patch, width // patch
    patches = (image[:rows * patch, :cols * patch]
               .reshape(rows, patch, cols, patch, 3)
               .swapaxes(1, 2)
               .reshape(rows * cols, patch * patch * 3))
    return patches, rows, cols


def demo_image(text_tokens):
    section("2. IMAGE -> PATCHES -> TOKENS")
    image = make_image(224, 224)
    patches, rows, cols = patchify(image, patch=16)

    print(f"  image           : {image.shape[1]}x{image.shape[0]} pixels, "
          f"{image.size:,} numbers")
    print(f"  patch grid      : {rows} x {cols} = {len(patches)} patches of 16x16")
    print(f"  each patch      : {patches.shape[1]} numbers (16 x 16 x 3 colours)")
    print(f"  -> tokens       : {len(patches)}   (one vector per patch)")
    print(f"\n  first patch, first 8 numbers: {patches[0][:8]}")
    print("  That row of numbers is projected to a vector and fed in exactly like a")
    print("  word embedding. The model has no idea it came from a picture.\n")

    print(f"  {'resolution':>12} {'patches (tokens)':>18} {'vs the text above':>20}")
    print("  " + "-" * 54)
    for size in (224, 336, 512, 768, 1024):
        n = (size // 16) ** 2
        print(f"  {f'{size}x{size}':>12} {n:>18,} {f'{n / text_tokens:.0f}x':>20}")

    print(
        "\n  Token count grows with the SQUARE of the resolution: double the width and\n"
        "  you quadruple the cost. A single 1024x1024 image costs more tokens than\n"
        "  several pages of text.\n\n"
        "  This is why real vision models do not simply scale up: they cap the\n"
        "  resolution, tile large images, or compress patches before the language\n"
        "  model sees them. Exact counts differ per model -- check the provider's\n"
        "  documentation rather than assuming this arithmetic."
    )


# ---------------------------------------------------------------------------
# 3. Audio
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000        # 16 kHz is the standard for speech models
N_FFT, HOP, N_MELS = 400, 160, 80      # 25 ms window, 10 ms hop, 80 mel bands


def make_audio(seconds=10.0):
    """A speech-like signal: a wandering pitch with harmonics and pauses.

    Not real speech, but it has the property that matters here -- energy that
    varies over time and frequency, which is what a spectrogram captures.
    """
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    pitch = 120 + 40 * np.sin(2 * np.pi * 0.7 * t)              # 120 Hz, wandering
    signal = sum(np.sin(2 * np.pi * pitch * (n + 1) * t) / (n + 1) for n in range(5))
    envelope = (np.sin(2 * np.pi * 1.6 * t) > -0.3).astype(float)   # syllable-like gaps
    return (signal * envelope * 0.3).astype(np.float32)


# A mel spectrogram in 30 lines of numpy. Libraries do this for you, but seeing
# it written out is the point: it is a windowed Fourier transform, then a set of
# triangular filters spaced the way human hearing resolves pitch.

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sample_rate=SAMPLE_RATE):
    """Triangular filters, evenly spaced on the mel scale."""
    edges = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre > left:
            filters[i, left:centre] = np.linspace(0, 1, centre - left, endpoint=False)
        if right > centre:
            filters[i, centre:right] = np.linspace(1, 0, right - centre, endpoint=False)
    return filters


def mel_spectrogram(audio):
    """Slice into overlapping frames, FFT each one, squash to mel bands, take dB."""
    n_frames = 1 + (len(audio) - N_FFT) // HOP
    window = np.hanning(N_FFT)
    frames = np.stack([audio[i * HOP: i * HOP + N_FFT] * window for i in range(n_frames)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (frames, freq bins)
    mel = power @ mel_filterbank().T                          # (frames, mel bands)
    return 10.0 * np.log10(mel.T + 1e-10)                     # (mel bands, frames), in dB


def demo_audio(text_tokens):
    section("3. AUDIO -> SPECTROGRAM FRAMES -> TOKENS")
    seconds = 10.0
    audio = make_audio(seconds)
    print(f"  raw audio  : {seconds:.0f}s at {SAMPLE_RATE:,} Hz = {len(audio):,} samples")
    print("               far too many numbers to feed a transformer directly.\n")

    mel_db = mel_spectrogram(audio)
    n_mels, n_frames = mel_db.shape
    hop_length = HOP

    print(f"  spectrogram: {n_mels} mel bands x {n_frames} frames"
          "   [computed here with numpy]")
    print(f"  window     : {N_FFT} samples = {N_FFT / SAMPLE_RATE * 1000:.0f} ms, overlapping")
    print(f"  hop length : {hop_length} samples = {hop_length / SAMPLE_RATE * 1000:.0f} ms per frame")
    print(f"  frame rate : {SAMPLE_RATE / hop_length:.0f} frames per second")
    # The honest measure is not "how many numbers" -- it is how long the
    # SEQUENCE is, because attention cost grows with the square of that.
    print(f"\n  total numbers   : {len(audio):,} samples -> {mel_db.size:,} "
          f"({len(audio) / mel_db.size:.1f}x fewer -- not the interesting part)")
    print(f"  sequence length : {len(audio):,} samples -> {n_frames:,} frames "
          f"-> ~{n_frames // 2:,} tokens")
    print(f"                    a {len(audio) / (n_frames // 2):.0f}x shorter sequence for attention to chew on")

    loud = mel_db.mean(axis=0) > mel_db.mean()
    print(f"\n  sanity check: {loud.mean():.0%} of frames are above average energy --")
    print("  the syllable-like gaps in the synthetic signal really do show up as")
    print("  quiet frames, which is exactly the structure a speech model reads.")

    encoder_positions = n_frames // 2        # speech encoders commonly halve the rate
    print(f"\n  -> tokens   : ~{encoder_positions:,} after a typical 2x downsample")
    print(f"  {'audio length':>14} {'frames (100/s)':>16} {'tokens (~50/s)':>16} {'vs text':>10}")
    print("  " + "-" * 60)
    for label, secs in (("10 seconds", 10), ("1 minute", 60), ("10 minutes", 600),
                        ("1 hour", 3600)):
        frames = int(secs * SAMPLE_RATE / hop_length)
        print(f"  {label:>14} {frames:>16,} {frames // 2:>16,} {f'{frames // 2 / text_tokens:.0f}x':>10}")

    print(
        "\n  A minute of speech is a few thousand tokens -- comparable to a long\n"
        "  document, and much cheaper than a single high-resolution image.\n\n"
        "  Note what the spectrogram throws away and keeps: it keeps WHICH\n"
        "  frequencies are loud over time, which is enough to recognise words, tone\n"
        "  and speaker. Audio models differ in how they slice this, and some skip\n"
        "  spectrograms entirely and learn discrete audio codes instead."
    )
    return n_frames


# ---------------------------------------------------------------------------
# 4. Video
# ---------------------------------------------------------------------------

def demo_video(text_tokens):
    section("4. VIDEO -> FRAMES -> PATCHES -> TOKENS (the expensive one)")
    print("  A video is just images over time, so the costs multiply.\n")
    print(f"  {'sampling':>22} {'frames/min':>12} {'tokens/min':>14} {'vs text':>12}")
    print("  " + "-" * 64)
    tokens_per_frame_512 = (512 // 16) ** 2
    for label, fps in (("1 frame per second", 1), ("2 frames per second", 2),
                       ("8 frames per second", 8), ("24 fps (full video)", 24)):
        frames = fps * 60
        tokens = frames * tokens_per_frame_512
        print(f"  {label:>22} {frames:>12,} {tokens:>14,} {f'{tokens / text_tokens:,.0f}x':>12}")

    print(
        f"\n  At 512x512 each frame costs {tokens_per_frame_512} tokens, so one minute of full-rate\n"
        f"  video is {24 * 60 * tokens_per_frame_512:,} tokens -- larger than most context windows.\n\n"
        "  That arithmetic is why every video model does one of these:\n"
        "    - sample sparsely (1-2 frames per second, not 24)\n"
        "    - compress frames far more aggressively than still images\n"
        "    - pool across time, so similar neighbouring frames share tokens\n"
        "    - summarise segments into text first, then reason over the text\n\n"
        "  When a provider quotes a video price, work out which of these they did."
    )


def main():
    text_tokens = demo_text()
    demo_image(text_tokens)
    demo_audio(text_tokens)
    demo_video(text_tokens)

    section("WHAT TO TAKE AWAY")
    print(
        "  1. Everything becomes a sequence of vectors. That is the whole trick, and\n"
        "     it is why one architecture can handle text, pictures and sound.\n"
        "  2. The conversion is different per modality: sub-words, square patches,\n"
        "     spectrogram frames. What arrives at the transformer looks the same.\n"
        "  3. Token counts differ by ORDERS OF MAGNITUDE between modalities. Text is\n"
        "     cheap, images are expensive, video is enormous.\n"
        "  4. Because attention cost grows with the square of sequence length, those\n"
        "     token counts drive both your bill and what fits in the context window.\n"
        "  5. The numbers here are the honest arithmetic of the standard recipes.\n"
        "     Every provider tweaks them -- always check their documentation."
    )


if __name__ == "__main__":
    main()
