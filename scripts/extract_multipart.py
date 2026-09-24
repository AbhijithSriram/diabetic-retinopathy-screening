"""Extract a zip that was split into byte-chunks (name.zip.001, .002, ...)
without first concatenating the chunks into a second full-size copy.

The chunks are presented to zipfile as one seekable stream. This works for
7-Zip / `split`-style chunked archives (DDR, Messidor-2's IMAGES.zip.00N).

    python scripts/extract_multipart.py <first chunk or glob stem> <dest> [--delete]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path


class ChainedFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.total = sum(self.sizes)
        self.pos = 0
        self.handles = [None] * len(self.paths)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.total + offset
        return self.pos

    def _handle(self, i):
        if self.handles[i] is None:
            self.handles[i] = open(self.paths[i], "rb")
        return self.handles[i]

    def readinto(self, b):
        want = len(b)
        got = 0
        while got < want and self.pos < self.total:
            i = max(k for k, s in enumerate(self.starts) if s <= self.pos)
            h = self._handle(i)
            h.seek(self.pos - self.starts[i])
            chunk = h.read(min(want - got, self.sizes[i] - (self.pos - self.starts[i])))
            if not chunk:
                break
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got

    def close(self):
        for h in self.handles:
            if h:
                h.close()
        super().close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("first")
    ap.add_argument("dest")
    ap.add_argument("--delete", action="store_true", help="delete chunks after a verified extraction")
    a = ap.parse_args()
    first = Path(a.first)
    stem = first.name.rsplit(".", 1)[0] if first.suffix[1:].isdigit() else first.name
    parts = sorted(first.parent.glob(stem + ".[0-9][0-9][0-9]"))
    if not parts:
        sys.exit(f"no chunks named {stem}.NNN in {first.parent}")
    print(f"{len(parts)} chunks, {sum(p.stat().st_size for p in parts) / 1e9:.2f} GB")
    stream = io.BufferedReader(ChainedFile(parts), buffer_size=4 << 20)
    with zipfile.ZipFile(stream) as z:
        bad = None
        names = z.namelist()
        print(f"{len(names)} entries")
        dest = Path(a.dest)
        for i, n in enumerate(names):
            z.extract(n, dest)
            if i % 2000 == 0:
                print(f"  {i}/{len(names)}", flush=True)
        # CRC check of a sample of members (full testzip would re-read everything)
        for n in names[:: max(1, len(names) // 200)]:
            if not n.endswith("/"):
                with z.open(n) as f:
                    while f.read(1 << 20):
                        pass
    stream.close()  # release the chunk files before deleting them (Windows locks open files)
    print("extracted to", dest)
    if a.delete and bad is None:
        for p in parts:
            os.remove(p)
        print("deleted chunks")


if __name__ == "__main__":
    main()
