"""Minimal PyInstaller CArchive reader/writer (PyInstaller 5/6 layout)."""
import struct
import zlib

MAGIC = b"MEI\014\013\012\013\016"
COOKIE_FMT = "!8sIIII64s"
COOKIE_SIZE = struct.calcsize(COOKIE_FMT)
ENTRY_FMT = "!IIIIBc"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)


class Entry:
    def __init__(self, pos, csize, usize, cflag, typecode, name):
        self.pos, self.csize, self.usize = pos, csize, usize
        self.cflag, self.typecode, self.name = cflag, typecode, name

    def __repr__(self):
        return f"<{self.typecode} {self.name!r} pos={self.pos} c={self.csize} u={self.usize} z={self.cflag}>"


class CArchive:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.blob = f.read()
        cookie_pos = self.blob.rfind(MAGIC)
        if cookie_pos < 0:
            raise ValueError("PyInstaller cookie not found")
        self.cookie_pos = cookie_pos
        (_, self.pkg_len, self.toc_off, self.toc_len, self.pyvers, pylib) = struct.unpack(
            COOKIE_FMT, self.blob[cookie_pos:cookie_pos + COOKIE_SIZE])
        self.pylib = pylib.rstrip(b"\0")
        self.archive_end = cookie_pos + COOKIE_SIZE
        self.archive_start = self.archive_end - self.pkg_len
        self.tail = self.blob[self.archive_end:]  # e.g. Authenticode signature (normally empty)
        self.entries = []
        p = self.archive_start + self.toc_off
        end = p + self.toc_len
        while p < end:
            size, pos, csize, usize, cflag, tc = struct.unpack(ENTRY_FMT, self.blob[p:p + ENTRY_SIZE])
            name = self.blob[p + ENTRY_SIZE:p + size].rstrip(b"\0").decode("utf-8")
            self.entries.append(Entry(pos, csize, usize, cflag, tc.decode(), name))
            p += size

    def raw(self, e):
        s = self.archive_start + e.pos
        return self.blob[s:s + e.csize]

    def data(self, e):
        d = self.raw(e)
        return zlib.decompress(d) if e.cflag else d

    def find(self, name):
        return next(e for e in self.entries if e.name == name)

    def rebuild(self, replacements):
        """replacements: {entry_name: uncompressed bytes}. Returns new exe bytes."""
        head = self.blob[:self.archive_start]
        body = bytearray()
        new_entries = []
        for e in self.entries:
            if e.name in replacements:
                u = replacements[e.name]
                c = zlib.compress(u, 9) if e.cflag else u
                ne = Entry(len(body), len(c), len(u), e.cflag, e.typecode, e.name)
                body += c
            elif e.csize == 0:
                # Runtime options (type 'o') carry no data and are stored at position 0.
                ne = Entry(e.pos, 0, e.usize, e.cflag, e.typecode, e.name)
            else:
                r = self.raw(e)
                ne = Entry(len(body), len(r), e.usize, e.cflag, e.typecode, e.name)
                body += r
            new_entries.append(ne)
        toc = bytearray()
        for e in new_entries:
            nm = e.name.encode("utf-8")
            entry_len = ENTRY_SIZE + len(nm) + 1
            pad = (16 - entry_len % 16) % 16
            entry_len += pad
            toc += struct.pack(ENTRY_FMT, entry_len, e.pos, e.csize, e.usize, e.cflag, e.typecode.encode())
            toc += nm + b"\0" * (1 + pad)
        toc_off = len(body)
        pkg_len = len(body) + len(toc) + COOKIE_SIZE
        cookie = struct.pack(COOKIE_FMT, MAGIC, pkg_len, toc_off, len(toc), self.pyvers,
                             self.pylib.ljust(64, b"\0"))
        return head + bytes(body) + bytes(toc) + cookie
