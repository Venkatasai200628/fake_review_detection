"""
prune_small_images.py — remove source photos too small to support the
manipulation-detection module.

WHY 1000px:
measured on this project's own data (TECHNICAL_REPORT_01, section 5). Derived
review images are produced at 1024px. A source smaller than that gets upscaled,
which invents no detail and blurs what is there. At 512px every manipulation
detector sits at chance (ELA 0.513, combined 0.554); at 1024px they reach 0.739
and 0.758. Small sources are still fine for provenance and coordination, but
they dilute the weakest module in the system.

USAGE
    python prune_small_images.py                 # dry run, deletes nothing
    python prune_small_images.py --delete        # actually delete
    python prune_small_images.py --min-edge 1200 # stricter threshold

Run it from inside the image-pool folder (the one holding the category
directories). Safe to run in the cloned repo: it skips .git entirely.
"""

import os
import argparse
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-edge', type=int, default=1000)
    ap.add_argument('--delete', action='store_true')
    ap.add_argument('--target-per-category', type=int, default=54)
    args = ap.parse_args()

    cats = sorted(d for d in os.listdir('.')
                  if os.path.isdir(d) and not d.startswith('.'))
    if not cats:
        print("No category folders found. Run this INSIDE the image pool folder.")
        return

    to_delete, keep_count, unreadable = {}, {}, []

    for c in cats:
        small, keep = [], 0
        for f in sorted(os.listdir(c)):
            if f.startswith('.') or not f.lower().endswith(EXTS):
                continue
            p = os.path.join(c, f)
            try:
                im = Image.open(p)
                im.load()
            except Exception as e:
                unreadable.append((p, str(e)[:50]))
                continue
            if max(im.size) < args.min_edge:
                small.append((f, im.size))
            else:
                keep += 1
        to_delete[c], keep_count[c] = small, keep

    n_del = sum(len(v) for v in to_delete.values())
    n_keep = sum(keep_count.values())

    print(f"threshold: long edge < {args.min_edge}px\n")
    print(f"{'category':24s} {'keep':>5s} {'delete':>7s} {'collect':>8s}")
    print('-' * 48)
    total_need = 0
    for c in cats:
        need = max(0, args.target_per_category - keep_count[c])
        total_need += need
        print(f"{c:24s} {keep_count[c]:5d} {len(to_delete[c]):7d} {need:8d}")
    print('-' * 48)
    print(f"{'TOTAL':24s} {n_keep:5d} {n_del:7d} {total_need:8d}")

    if unreadable:
        print(f"\nunreadable files ({len(unreadable)}) -- delete these too:")
        for p, e in unreadable:
            print(f"  {p}  ({e})")

    if not args.delete:
        print(f"\nDRY RUN. Nothing deleted.")
        print(f"Re-run with --delete to remove {n_del} files.")
        with open('_files_to_delete.txt', 'w', encoding='utf-8') as fh:
            for c in cats:
                for f, size in to_delete[c]:
                    fh.write(f"{c}/{f}\t{size[0]}x{size[1]}\n")
        print("Full list written to _files_to_delete.txt (review it first).")
        return

    removed = 0
    for c in cats:
        for f, _ in to_delete[c]:
            os.remove(os.path.join(c, f))
            removed += 1
    for p, _ in unreadable:
        os.remove(p)
        removed += 1
    print(f"\ndeleted {removed} files. {n_keep} remain.")
    print(f"collect {total_need} more to reach "
          f"{args.target_per_category}/category.")


if __name__ == '__main__':
    main()
