"""Sentetik el yazısı sayfa üreteci.

Metni el yazısı fontlarıyla, satır satır, karakter başına titreme (x/y/rotasyon/boyut) ile "yazar";
form şablonu (başlık bandı, çizgiler, kutular) çizer; tarama/fotoğraf bozulmaları uygular
(sayfa eğimi, gölge/vinyet, bulanıklık, gürültü, JPEG, düşük çözünürlük).

Çıktı: <out>/<case_id>__<font>__<style>.png  +  aynı adla .gt.txt (ground truth) + meta.json
"""

import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

FONT_DIR = Path(__file__).parent / "fonts"
FONTS = sorted(FONT_DIR.glob("*.ttf"))

PAGE_W, PAGE_H = 1654, 2339   # A4 @ 200 dpi
MARGIN = 120

STYLES = {
    # ad: (döndürme°, bulanıklık, gürültü, jpeg kalitesi, ölçek, gölge)
    "clean":   (0.0, 0.0, 0.00, 92, 1.00, 0.0),
    "scan":    (0.6, 0.6, 0.02, 80, 0.85, 0.15),
    "phone":   (2.5, 1.0, 0.05, 65, 0.70, 0.45),
    "bad":     (4.0, 1.6, 0.08, 45, 0.55, 0.60),
}


def _ink(rng: random.Random) -> tuple[int, int, int]:
    base = rng.choice([(20, 20, 60), (15, 15, 15), (10, 30, 120), (40, 20, 20)])
    return tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in base)


def _draw_form(draw: ImageDraw.ImageDraw, rng: random.Random, kind: str) -> None:
    # Başlık bandı + form çizgileri (gerçek formlardaki gibi)
    draw.rectangle([MARGIN, 60, PAGE_W - MARGIN, 150], outline=(90, 90, 90), width=3)
    y = 190
    while y < PAGE_H - MARGIN:
        draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=(200, 200, 215), width=1)
        y += 78
    if kind == "patoloji":
        # sağ üstte barkod benzeri kutu
        x0 = PAGE_W - MARGIN - 260
        draw.rectangle([x0, 170, x0 + 260, 230], outline=(120, 120, 120), width=2)
        for i in range(40):
            if rng.random() < 0.6:
                draw.line([(x0 + 8 + i * 6, 176), (x0 + 8 + i * 6, 224)], fill=(40, 40, 40), width=rng.choice([1, 2]))


def _write_line(img: Image.Image, text: str, x: int, y: int, font_path: Path, size: int,
                ink: tuple, rng: random.Random, jitter: float) -> int:
    """Karakter karakter yaz; her karaktere küçük ofset/rotasyon. Yazılan genişliği döndür."""
    base_font = ImageFont.truetype(str(font_path), size)
    cx = x
    for ch in text:
        if ch == " ":
            cx += int(size * 0.28 * rng.uniform(0.8, 1.4))
            continue
        fs = max(8, int(size * rng.uniform(1 - 0.08 * jitter, 1 + 0.08 * jitter)))
        f = base_font if fs == size else ImageFont.truetype(str(font_path), fs)
        w, h = f.getbbox(ch)[2:]
        w = max(w, 4)
        glyph = Image.new("RGBA", (w + 12, h + 24), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glyph)
        gd.text((6, 6), ch, font=f, fill=ink + (255,))
        rot = rng.uniform(-6, 6) * jitter
        glyph = glyph.rotate(rot, resample=Image.BICUBIC, expand=False)
        dy = int(rng.uniform(-3, 3) * jitter)
        img.alpha_composite(glyph, (cx, y + dy))
        cx += int(w * rng.uniform(0.92, 1.05)) + int(rng.uniform(-1, 2) * jitter)
    return cx - x


def render_page(text: str, kind: str, font_path: Path, style: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    rot, blur, noise, jpeg_q, scale, shadow = STYLES[style]
    page = Image.new("RGBA", (PAGE_W, PAGE_H), (252, 250, 244, 255))
    draw = ImageDraw.Draw(page)
    _draw_form(draw, rng, kind)

    ink = _ink(rng)
    size = rng.randint(40, 52)
    jitter = rng.uniform(0.6, 1.3)
    y = 200
    lines = text.split("\n")
    for i, line in enumerate(lines):
        # Başlık satırları (ilk 2) form bandına, kalın/büyük
        if i < 2:
            _write_line(page, line, MARGIN + 20, 70 + i * 40, font_path, 34, (40, 40, 40), rng, 0.3)
            continue
        # Uzun satırları kır (el yazısında kağıda sığmaz)
        max_chars = int((PAGE_W - 2 * MARGIN) / (size * 0.42))
        chunks = [line[j:j + max_chars] for j in range(0, len(line), max_chars)] or [""]
        for chunk in chunks:
            if y > PAGE_H - MARGIN - 60:
                break
            x = MARGIN + rng.randint(0, 40)
            _write_line(page, chunk, x, y - size + 10, font_path, size, ink, rng, jitter)
            y += 78
        # Bölüm başlıklarından sonra biraz boşluk
        if line.endswith(":"):
            y += rng.choice([0, 0, 39])

    img = page.convert("RGB")

    # ── Tarama / fotoğraf bozulmaları ──
    if rot:
        img = img.rotate(rng.uniform(-rot, rot), resample=Image.BICUBIC, expand=False, fillcolor=(235, 232, 225))
    if shadow:
        vign = Image.new("L", img.size, 255)
        vd = ImageDraw.Draw(vign)
        cx, cy = rng.randint(0, PAGE_W), rng.randint(0, PAGE_H)
        for r in range(1400, 0, -40):
            v = int(255 - shadow * 160 * (r / 1400))
            vd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=v)
        vign = vign.filter(ImageFilter.GaussianBlur(120))
        img = Image.composite(img, ImageOps.colorize(vign, (0, 0, 0), (255, 255, 255)).convert("RGB"),
                              vign.point(lambda p: 255 - int((255 - p) * 0.9)))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        px = img.load()
        n = int(noise * PAGE_W * PAGE_H * 0.05)
        for _ in range(n):
            xx, yy = rng.randrange(PAGE_W), rng.randrange(PAGE_H)
            v = rng.randint(120, 200)
            px[xx, yy] = (v, v, v)
    if scale != 1.0:
        img = img.resize((int(PAGE_W * scale), int(PAGE_H * scale)), Image.LANCZOS)
    # JPEG artefaktı
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, "JPEG", quality=jpeg_q)
    return Image.open(BytesIO(buf.getvalue())).convert("RGB")


def build_dataset(out_dir: str, n_cases: int = 20, seed: int = 7, styles=("scan", "phone", "bad"),
                  fonts_per_case: int = 1) -> list[dict]:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from synth_cases import make_cases

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cases = make_cases(n_cases, seed)
    items = []
    for ci, c in enumerate(cases):
        for fi in range(fonts_per_case):
            font = FONTS[(ci * fonts_per_case + fi) % len(FONTS)]
            style = styles[(ci + fi) % len(styles)]
            name = f"{c.id}__{font.stem}__{style}"
            img = render_page(c.text, c.kind, font, style, seed=rng.randint(0, 10**9))
            img.save(out / f"{name}.png")
            (out / f"{name}.gt.txt").write_text(c.text, encoding="utf-8")
            items.append({"file": f"{name}.png", "case_id": c.id, "kind": c.kind, "category": c.category,
                          "font": font.stem, "style": style, "pii": c.pii, "keep": c.keep, "upper": c.meta["upper"]})
    (out / "meta.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/data")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--fonts-per-case", type=int, default=1)
    ap.add_argument("--styles", default="scan,phone,bad")
    a = ap.parse_args()
    items = build_dataset(a.out, a.n, styles=tuple(a.styles.split(",")), fonts_per_case=a.fonts_per_case)
    print(f"{len(items)} görüntü → {a.out}")
