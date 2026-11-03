"""BoardCapture - batch-median pipeline.

  python bm_main.py VIDEO --frames-only          # capture + contact sheet only (free)
  python bm_main.py VIDEO --ocr                  # + Tesseract text and confidence
  python bm_main.py VIDEO --notes                # + Ollama vision transcription + notes

Outputs go to out/<video name>/ : boards/, contact_sheet.jpg, change_plot.png,
report.json, ocr.json, notes.md
"""
import argparse
import json
from pathlib import Path

from boardcapture import batch_median as bm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=float, default=2.0, help="seconds between samples")
    ap.add_argument("--batch", type=int, default=15, help="frames per batch")
    ap.add_argument("--change-floor", type=float, default=0.02,
                    help="min share of pixels that must change to count as a new board "
                         "(lower it if near-identical slides get merged)")
    ap.add_argument("--frames-only", action="store_true")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--notes", action="store_true")
    ap.add_argument("--keep-all", action="store_true", help="save every batch median")
    ap.add_argument("--vision-model", default="qwen2.5vl:7b")
    ap.add_argument("--text-model", default="llama3.1:8b")
    a = ap.parse_args()

    out = Path(a.out or Path("out") / Path(a.video).stem)
    cfg = bm.BMConfig(step_s=a.step, batch=a.batch, change_floor=a.change_floor)
    rep = bm.run(a.video, str(out), cfg, keep_all=a.keep_all)

    print(f"\n{rep['samples']} samples, {a.batch}-frame windows ({a.batch * a.step:.0f}s), "
          f"change threshold {rep['threshold']:.1%}")
    for s in rep["selections"]:
        b = s
        print(f"  board {s['board']:2d}  {s['t_start']:6.0f}-{s['t_end']:6.0f}s  "
              f"ink {b['ink']:.1%}  best frame occluded {b['best_occlusion']:.1%}  "
              f"<- {s['reason']}")
    print(f"\nlook at: {out / 'contact_sheet.jpg'}  and  {out / 'change_plot.png'}")
    if a.frames_only or not (a.ocr or a.notes):
        return

    from boardcapture import tess_ocr
    boards = []
    for s in rep["selections"]:
        r = tess_ocr.ocr(s["median_path"])
        boards.append({**s, "tesseract": r})
        print(f"  board {s['board']:2d}  tesseract conf {r['confidence']:5.1f}  {r['words']} words")
    (out / "ocr.json").write_text(json.dumps(boards, indent=2))
    if not a.notes:
        return

    from boardcapture import local_notes as ln
    for b in boards:
        print(f"  transcribing board {b['board']} with {a.vision_model} ...")
        b["transcript"] = ln.transcribe_board(b["median_path"], b["tesseract"], a.vision_model)
    (out / "ocr.json").write_text(json.dumps(boards, indent=2))
    print(f"  writing notes with {a.text_model} ...")
    (out / "notes.md").write_text(ln.make_notes(boards, a.text_model), encoding="utf-8")
    print(f"\nnotes: {out / 'notes.md'}")


if __name__ == "__main__":
    main()
