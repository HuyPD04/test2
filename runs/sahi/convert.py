from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "benchmark_official" / "results" / "test"
DEFAULT_IMAGE_DIR = ROOT / "data" / "raw" / "images" / "test"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _read_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _find_image(image_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No image found for {stem!r} in {image_dir}")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_official_row(line: str) -> bool:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 8:
        return False
    try:
        float(parts[0])
        float(parts[1])
        float(parts[2])
        float(parts[3])
        float(parts[4])
        int(float(parts[5]))
        int(float(parts[6]))
        int(float(parts[7]))
    except ValueError:
        return False
    return True


def _convert_sahi_row(
    line: str,
    image_width: int,
    image_height: int,
    class_offset: int,
) -> tuple[float, str] | None:
    parts = line.strip().split()
    if len(parts) != 6:
        return None

    try:
        class_id = int(float(parts[0]))
        score = float(parts[1])
        x_center = float(parts[2]) * image_width
        y_center = float(parts[3]) * image_height
        box_width = float(parts[4]) * image_width
        box_height = float(parts[5]) * image_height
    except ValueError:
        return None

    x1 = x_center - box_width / 2.0
    y1 = y_center - box_height / 2.0
    x2 = x_center + box_width / 2.0
    y2 = y_center + box_height / 2.0

    x1 = _clamp(x1, 0.0, float(image_width))
    y1 = _clamp(y1, 0.0, float(image_height))
    x2 = _clamp(x2, 0.0, float(image_width))
    y2 = _clamp(y2, 0.0, float(image_height))

    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None

    official_class = class_id + class_offset
    return (
        score,
        f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},{score:.6f},{official_class},-1,-1",
    )


def convert_file(path: Path, image_dir: Path, class_offset: int) -> tuple[int, bool]:
    lines = path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if not non_empty_lines:
        path.write_text("", encoding="utf-8")
        return 0, False

    if all(_is_official_row(line) for line in non_empty_lines):
        return len(non_empty_lines), False

    image_path = _find_image(image_dir, path.stem)
    image_width, image_height = _read_image_size(image_path)

    rows: list[tuple[float, str]] = []
    skipped = 0
    for line in non_empty_lines:
        row = _convert_sahi_row(
            line,
            image_width=image_width,
            image_height=image_height,
            class_offset=class_offset,
        )
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    rows.sort(key=lambda item: item[0], reverse=True)
    path.write_text("\n".join(row for _score, row in rows) + ("\n" if rows else ""), encoding="utf-8")

    if skipped:
        print(f"[convert] skipped {skipped} invalid rows in {path.name}")
    return len(rows), True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SAHI YOLO-normalized txt outputs to VisDrone official result format in-place."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--class-offset",
        type=int,
        default=1,
        help="Offset added to zero-based detector class IDs for VisDrone official IDs.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    image_dir = args.image_dir if args.image_dir.is_absolute() else ROOT / args.image_dir

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    txt_files = sorted(results_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {results_dir}")

    converted_files = 0
    total_rows = 0
    for index, path in enumerate(txt_files, start=1):
        rows, converted = convert_file(path, image_dir=image_dir, class_offset=args.class_offset)
        total_rows += rows
        converted_files += int(converted)
        if index == 1 or index % 100 == 0 or index == len(txt_files):
            action = "converted" if converted else "kept"
            print(f"[convert] {index}/{len(txt_files)} {action} {path.name}: {rows} rows")

    print(
        f"[convert] converted {converted_files}/{len(txt_files)} files in {results_dir} "
        f"({total_rows} rows total)"
    )


if __name__ == "__main__":
    main()
