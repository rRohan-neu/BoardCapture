"""Tesseract OCR on selected boards, with a confidence score per board.

Low confidence (typically handwriting) is the signal to hand that board
to the local vision model in local_notes.py instead of trusting the text.
"""
from __future__ import annotations

import csv
import io
import subprocess

import cv2
import numpy as np


def preprocess(path: str):
    img = cv2.imread(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if g.mean() < 110:                       # chalkboard / dark slide -> invert
        g = 255 - g
    g = cv2.resize(g, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    bg = cv2.medianBlur(g, 41)               # flatten uneven lighting / glare
    flat = cv2.divide(g, bg, scale=255)
    return cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def ocr(path: str, psm: int = 6) -> dict:
    img = preprocess(path)
    ok, png = cv2.imencode(".png", img)
    r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", str(psm), "tsv"],
                       input=png.tobytes(), capture_output=True, check=True)
    rows = list(csv.DictReader(io.StringIO(r.stdout.decode("utf-8", "replace")),
                               delimiter="\t", quoting=csv.QUOTE_NONE))
    words = [w for w in rows if w.get("text", "").strip() and float(w["conf"]) >= 0]
    lines, key = [], None
    for w in words:
        k = (w["block_num"], w["par_num"], w["line_num"])
        if k != key:
            lines.append([]); key = k
        lines[-1].append(w["text"])
    conf = float(np.mean([float(w["conf"]) for w in words])) if words else 0.0
    return {"text": "\n".join(" ".join(l) for l in lines),
            "confidence": round(conf, 1), "words": len(words)}
