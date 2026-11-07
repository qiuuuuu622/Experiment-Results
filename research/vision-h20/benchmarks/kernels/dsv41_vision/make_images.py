"""Deterministic, entirely synthetic phone UI fixtures; no private screenshots."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    dims = [
        (1080, 2400),
        (1170, 2532),
        (1080, 1920),
        (1440, 3200),
        (720, 1600),
        (1284, 2778),
        (1080, 2340),
        (1290, 2796),
    ]
    names = [
        "Messages",
        "Weather",
        "Reading",
        "Dashboard",
        "Tasks",
        "Photo journal",
        "Settings",
        "Shopping list",
    ]
    manifest = []
    thumbs = []
    for i, ((w, h), name) in enumerate(zip(dims, names)):
        rng = random.Random(20261107 + i)
        # Draw at canonical width then rasterize to actual phone dimensions.
        H = round(h * 360 / w)
        im = Image.new("RGB", (360, H), "#f3f5f9")
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 13)
        small = ImageFont.truetype(str(fonts / "DejaVuSans.ttf"), 10)
        title = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 24)
        accent = tuple(rng.randrange(35, 155) for _ in range(3))
        d.text(
            (20, 12),
            "09:41                       WiFi  85%",
            font=small,
            fill="#19283a",
        )
        d.text((18, 55), name, font=title, fill=accent)
        d.text(
            (18, 91), f"SYNTHETIC TEST SCREEN  {i + 1:02d}", font=small, fill="#66788b"
        )
        y = 125
        row = 0
        while y < H - 70:
            rh = rng.choice([65, 90, 115])
            rh = min(rh, H - 65 - y)
            d.rounded_rectangle((15, y, 345, y + rh), radius=10, fill="white")
            d.text((28, y + 10), f"{name} item {row + 1}", font=font, fill="#213248")
            if i in (1, 3) and rh > 70:
                for j in range(12):
                    bh = rng.randrange(5, rh - 32)
                    x = 29 + j * 25
                    d.rectangle((x, y + rh - 9 - bh, x + 15, y + rh - 9), fill=accent)
            elif i == 5 and rh > 70:
                for j in range(3):
                    x = 29 + j * 100
                    color = tuple(rng.randrange(60, 230) for _ in range(3))
                    d.rectangle((x, y + 32, x + 85, y + rh - 8), fill=color)
                    d.ellipse((x + 15, y + 37, x + 32, y + 54), fill="#fff4c3")
            else:
                d.text(
                    (28, y + 33),
                    f"Sample {rng.randrange(100, 999)} - demo data only",
                    font=small,
                    fill="#657488",
                )
                if rh > 80:
                    d.text(
                        (28, y + 54),
                        "A reproducible visual encoder benchmark.",
                        font=small,
                        fill="#657488",
                    )
            y += rh + 12
            row += 1
        d.rectangle((0, H - 50, 360, H), fill="white")
        d.text(
            (28, H - 35),
            "Home        Explore        Saved        Profile",
            font=small,
            fill=accent,
        )
        d.rounded_rectangle((125, H - 9, 235, H - 5), radius=2, fill="#263344")
        path = args.output / f"{i + 1:02d}-{name.lower().replace(' ', '-')}.png"
        im.resize((w, h), Image.Resampling.LANCZOS).save(path)
        manifest.append(
            dict(
                file=path.name,
                width=w,
                height=h,
                seed=20261107 + i,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        thumb = im.copy()
        thumb.thumbnail((180, 420))
        thumbs.append(thumb)
    sheet = Image.new("RGB", (4 * 190, 2 * 440), "#dce2eb")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 4) * 190 + 5, (i // 4) * 440 + 5))
    sheet.save(args.output / "contact-sheet.jpg")
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                kind="deterministic synthetic UI fixtures, not real screenshots",
                generator_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                images=manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
