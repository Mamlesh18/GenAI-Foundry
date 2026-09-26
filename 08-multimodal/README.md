# 08 · Multimodal

> **In one sentence:** "multimodal" means a model that handles more than text — images, audio, video
> — and the reason one architecture can do all of it is that **everything gets turned into tokens**.

This track is also the place to sort out the model zoo. Chat models, reasoning models, thinking
models, vision models, STT, TTS, speech-to-speech, image and video models: what each one actually
is, what goes in and comes out, and when you need which.

---

## 1. The model zoo, in one table

| Type | In → out | What it is for | Examples | The gotcha |
|---|---|---|---|---|
| **Base (completion)** | text → text | Raw next-token prediction, before instruction tuning | Llama 3 *base*, Qwen *base* | It continues text rather than answering; rarely what you want directly |
| **Instruct / chat** | text → text | Following instructions, conversation | GPT, Claude, Gemini, Llama Instruct | The default. Everything else on this list is a specialisation |
| **Reasoning / thinking** | text → text (+ hidden deliberation) | Hard maths, code, multi-step logic | o-series, DeepSeek-R1, Claude extended thinking, Gemini thinking | Slower and dearer; **no better** on simple tasks |
| **Embedding** | text → vector | Search, RAG, clustering | text-embedding-3, BGE, Voyage | Not a chat model; it produces no text at all |
| **Reranker (cross-encoder)** | (query, doc) → score | Reordering search results | Cohere Rerank, BGE reranker | Too slow to run over a whole corpus; use on a shortlist |
| **Vision-language (VLM)** | image + text → text | Describing, reading, answering about images | GPT/Claude/Gemini vision, Qwen-VL, LLaVA | Images cost *many* tokens; small text in images is where they fail |
| **Multimodal embedding** | image *or* text → shared vector | Search images by text | CLIP, SigLIP | Different from a VLM: it scores similarity, it cannot describe |
| **Image generation** | text (+ image) → image | Creating and editing pictures | Stable Diffusion, Flux, Imagen, DALL·E | Diffusion, not transformers-as-usual; iterative denoising |
| **Speech-to-text (STT/ASR)** | audio → text | Transcription, captions, voice input | Whisper, Deepgram, AssemblyAI | Accents, noise and domain jargon dominate real accuracy |
| **Text-to-speech (TTS)** | text → audio | Reading text aloud, voice output | ElevenLabs, Cartesia, OpenAI TTS | Time-to-first-audio matters more than total time |
| **Speech-to-speech (S2S)** | audio → audio | Live voice conversation | OpenAI Realtime, Gemini Live, Moshi | Keeps tone and timing that a transcript throws away; harder to audit |
| **Video understanding** | video → text | Summarising and searching footage | Gemini, Qwen-VL | Frames are sampled, so it literally does not see most of the video |
| **Video generation** | text / image → video | Creating clips | Sora, Veo, Kling | Slow and expensive; consistency over time is the hard part |
| **Safety / guardrail** | text → label | Classifying input or output as allowed | Llama Guard, moderation endpoints | Small and fast by design; a filter, not a thinker |

### "Reasoning" and "thinking" are the same thing

Different vendors, one idea: the model is trained (usually with reinforcement learning on
verifiable problems) to produce a long internal chain of reasoning *before* its answer, and you pay
for those hidden tokens. OpenAI says "reasoning", Anthropic says "extended thinking", Google says
"thinking". [DeepSeek-R1](https://arxiv.org/abs/2501.12948) showed this behaviour can be
incentivised by pure RL, "obviating the need for human-labeled reasoning trajectories" — and that
self-checking and backtracking emerge on their own.

**When they help:** maths, competitive programming, multi-step logic, anything with a verifiable
answer. **When they do not:** summarising, extraction, chat, formatting — where they cost more and
add latency for no gain. Covered in module 01.

---

## 2. The unifying trick: everything becomes tokens

A transformer consumes a sequence of vectors. Nothing more. Each modality has its own way in:

```
  text    "a cat sat"   --> sub-word pieces     --> vectors
  image   224x224 px    --> 16x16 PATCHES       --> vectors     (ViT: "an image is worth 16x16 words")
  audio   10s waveform  --> spectrogram FRAMES  --> vectors     (25 ms window, 10 ms hop)
  video   30s clip      --> frames, then patches--> vectors     (and the bill explodes)
```

After that arrow, the model cannot tell where a vector came from. That is why one architecture
handles all of it, and why [attention](../01-foundations/01-attention-is-all-you-need/) is the
shared foundation.

### The cost, measured

From [`examples/tokens_demo.py`](examples/tokens_demo.py), against a 39-token sentence:

| Input | Tokens | Relative to that sentence |
|---|---|---|
| 163 characters of text | 39 | 1× |
| One 224×224 image | 196 | 5× |
| One 512×512 image | 1,024 | 26× |
| One 1024×1024 image | 4,096 | 105× |
| One minute of speech | ~3,000 | 77× |
| One minute of video @ 1 fps | 61,440 | 1,575× |
| One minute of video @ 24 fps | 1,474,560 | 37,809× |

Three consequences worth internalising:

1. **Image cost grows with the square of resolution.** Double the width, quadruple the tokens.
2. **Audio is surprisingly cheap** — a minute of speech costs less than one large image.
3. **Video at full frame rate does not fit anywhere.** Every video model samples, compresses or
   summarises. When you read a video price, work out which one they did.

---

## 3. The modules

| # | Module | Covers |
|---|---|---|
| 01 | Chat and Reasoning Models | Base vs instruct vs chat; reasoning/thinking models, budgets, when they are worth it |
| 02 | Vision Models | How VLMs see, documents and OCR, where they fail |
| 03 | Image Generation | Diffusion, conditioning, editing, control |
| 04 | Speech-to-Text | ASR pipelines, streaming, WER and what it hides |
| 05 | Text-to-Speech | Voices, latency, streaming, and the ethics of cloning |
| 06 | Speech-to-Speech | Realtime voice agents: cascaded vs native, the latency budget |
| 07 | Video Models | Understanding by sampling; generation and its limits |

*(Modules are added one at a time; links appear as each lands.)*

---

## 4. How to choose

```
Do you need to UNDERSTAND something, or CREATE it?
   understand ->  text?   chat model (add a reasoning model only if it is genuinely hard)
                  image?  a VLM
                  audio?  STT, unless tone and timing matter -- then speech-to-speech
                  video?  a video-understanding model, and expect frame sampling
   create     ->  text?   chat model
                  image?  an image generation model
                  speech? TTS
                  video?  a video generation model, and budget for slowness

Is it a conversation with a person, in real time?
   -> the latency budget decides the architecture, not the model quality (module 06)
```

**One model or several?** A single multimodal model is simpler and keeps information that a
pipeline throws away (tone of voice, layout of a page). A pipeline of specialists is cheaper, more
controllable, easier to audit, and lets you swap one part. Most production systems are pipelines;
most impressive demos are single models.

---

## 5. What is actually hard

| Problem | Where it bites |
|---|---|
| **Token cost** | Images and video, immediately. It shapes every design decision |
| **Latency** | Voice. Below ~800 ms feels like conversation; above ~2 s feels broken |
| **Confident misreading** | VLMs on small text, tables and charts; STT on names and jargon |
| **Evaluation** | "Is this a good image/voice/summary?" has no exact-match answer — see [05 · Evaluation](../05-evaluation/) |
| **Provenance and consent** | Voice cloning and likeness are ethical and legal problems before they are technical ones |

---

## 6. Examples

```bash
pip install numpy            # transformers optional, for the real tokenizer
python examples/tokens_demo.py
```

[`examples/tokens_demo.py`](examples/tokens_demo.py) converts a real sentence, a real image, a real
waveform and a video budget into tokens — including a mel spectrogram computed from scratch in
numpy, so you can see exactly what "audio becomes frames" means.

---

## 7. Exercises

1. **Do the image arithmetic.** At 16×16 patches, how many tokens is a 1536×1536 image? What if the
   provider tiles it into 512×512 chunks instead?
2. **Budget a meeting.** One hour of audio, transcribed and summarised. How many tokens is the
   audio? How many is the transcript? Why is the transcript so much cheaper?
3. **Budget a video.** Ten minutes at 2 frames per second, 512×512. Does it fit in a 1M-token
   context? What would you change?
4. **Pick the model type.** For each: reading totals off scanned invoices; a phone support bot;
   captioning a product catalogue; finding the moment a logo appears in an ad. Which type, and why?
5. **Reasoning or not?** Which of these justify a reasoning model: extracting dates from emails,
   solving a rota with constraints, rewriting a paragraph, debugging a failing test?

---

## 8. Projects

- **Beginner — A token budget calculator.** Given a mix of text, images, audio and video, estimate
  tokens and cost per request. *Test:* check your image numbers against a provider's published
  token counts and explain any difference.
- **Intermediate — One task, two architectures.** Build the same voice or document feature as a
  pipeline of specialists and as a single multimodal call. *Test:* compare accuracy, latency and
  cost; state which you would ship.
- **Advanced — A modality-aware router.** Route each request to the cheapest model type that can
  serve it. *Test:* on a labelled request mix, report accuracy and cost against always using the
  largest model.

---

## 9. Resources

**The papers that made each modality work**
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) — images as patch sequences.
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — CLIP; images and text in one space.
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, trained on 680,000 hours.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — how image generation works.
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — doing it affordably, in latent space.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — reasoning behaviour from reinforcement learning.

**Related in this repo**
- [01-foundations](../01-foundations/) — attention and tokens, which everything here reuses
- [03-inference](../03-inference/) — why token counts decide cost and latency
- [05-evaluation](../05-evaluation/) — how to judge outputs that have no single right answer

---

**Previous track:** [07 · Agents](../07-agents/) · **Next:** [skills](../skills/) · [projects](../projects/)
