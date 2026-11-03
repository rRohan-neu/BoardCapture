"""Synthetic MIXED lecture: whiteboard -> erase -> chalkboard -> two slides.
A 'professor' walks in front of the board while writing.

Ground truth (complete boards):
  whiteboard  full at ~84-100s, erased at 100s
  chalkboard  full at ~176-190s
  slide 1     190-220s
  slide 2     220-250s
"""
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, DUR = 1280, 720, 10, 250
OUT = Path(__file__).with_name("mixed_lecture.mp4")
F = cv2.FONT_HERSHEY_SIMPLEX

WB = ["Bayes' Theorem", "P(A|B) = P(B|A) P(A) / P(B)", "Prior: P(A)",
      "Likelihood: P(B|A)", "Evidence: P(B) = sum P(B|Ai) P(Ai)", "Example: disease test 99%"]
CB = ["Markov Chains", "P(Xn+1 | Xn) only", "Transition matrix T", "Stationary: pi T = pi",
      "Ergodic => unique pi"]
SL = [["Lecture 7: Recap", "- Bayes rule", "- Conditional probability", "- Markov property"],
      ["Homework 3", "Due Friday 5pm", "Questions 1 to 6", "Show all working"]]


def board(lines, n_chars, bg, fg):
    img = np.full((H, W, 3), bg, np.uint8)
    cv2.rectangle(img, (40, 30), (W - 40, H - 40), tuple(int(c * 0.85) for c in bg), 6)
    shown = n_chars
    for i, line in enumerate(lines):
        part = line[:max(0, min(len(line), shown))]
        shown -= len(line)
        if part:
            cv2.putText(img, part, (90, 110 + i * 95), F, 1.6, fg, 4, cv2.LINE_AA)
    return img


def slide(lines):
    img = np.full((H, W, 3), (250, 245, 240), np.uint8)
    cv2.rectangle(img, (0, 0), (W, 110), (120, 60, 20), -1)
    cv2.putText(img, lines[0], (60, 75), F, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
    for i, l in enumerate(lines[1:]):
        cv2.putText(img, l, (90, 220 + i * 110), F, 1.5, (30, 30, 30), 3, cv2.LINE_AA)
    return img


def prof(img, x):
    x = int(x)
    cv2.rectangle(img, (x - 70, 260), (x + 70, H), (60, 40, 120), -1)   # body
    cv2.circle(img, (x, 200), 55, (140, 170, 210), -1)                   # head
    return img


def main():
    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    wb_total, cb_total = sum(map(len, WB)), sum(map(len, CB))
    for i in range(DUR * FPS):
        t = i / FPS
        if t < 100:                                            # whiteboard, writes until 84s
            n = int(min(1, t / 84) * wb_total)
            img = board(WB, n, (235, 235, 235), (40, 40, 40))
            px = 300 + 500 * abs(np.sin(t / 7))
        elif t < 190:                                          # chalkboard, writes 102-176s
            n = int(np.clip((t - 102) / 74, 0, 1) * cb_total)
            img = board(CB, n, (45, 70, 50), (235, 235, 230))
            px = 300 + 550 * abs(np.sin(t / 6))
        else:                                                  # slides, prof by podium
            img = slide(SL[0] if t < 220 else SL[1])
            px = 1120 + 60 * np.sin(t / 3)
        vw.write(prof(img, px))
    vw.release()
    print(f"wrote {OUT} ({DUR}s)")


if __name__ == "__main__":
    main()
