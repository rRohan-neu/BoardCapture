"""Local LLM stages via Ollama (no API cost).

Stage A  image -> clean text: a vision model reads the board image, using the
         Tesseract text as a hint (Tesseract is good on slides, weak on handwriting).
Stage B  text  -> notes: a text model turns all board transcriptions into study notes.

Setup (once):  ollama pull qwen2.5vl:7b   and   ollama pull llama3.1:8b
"""
from __future__ import annotations

import base64
import json
import urllib.request

OLLAMA = "http://localhost:11434/api/generate"


def _generate(model: str, prompt: str, images: list[str] | None = None,
              timeout: int = 600) -> str:
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1}}
    if images:
        body["images"] = images
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"].strip()
    except OSError as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA} - is `ollama serve` running? ({e})")


def transcribe_board(image_path: str, tess: dict, model: str) -> str:
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    prompt = (
        "This is a photo of a lecture board or slide. Transcribe ALL of its content "
        "exactly, top to bottom, left to right. Write maths in LaTeX ($...$). "
        "Describe any diagram in one bracketed line, e.g. [diagram: ...]. "
        "Do not explain or add anything that is not on the board.\n\n"
        f"An OCR engine (confidence {tess['confidence']}/100) read this - use it only "
        f"as a hint, the image is the truth:\n---\n{tess['text'][:3000]}\n---"
    )
    return _generate(model, prompt, [img])


def make_notes(boards: list[dict], model: str) -> str:
    joined = "\n\n".join(f"### Board {b['board']} ({b['t_start']:.0f}-{b['t_end']:.0f}s)\n"
                         f"{b['transcript']}" for b in boards)
    prompt = (
        "You are writing study notes for a student who missed this lecture. Below are "
        "transcriptions of each board, in order. Produce clear Markdown notes: a title, "
        "a short summary, then sections that follow the lecture's flow, keeping every "
        "definition, formula (LaTeX) and example. Merge repeated content from boards that "
        "were photographed twice. Do not invent material that is not in the boards; if "
        "something is unreadable, mark it [unclear].\n\n" + joined
    )
    return _generate(model, prompt)
