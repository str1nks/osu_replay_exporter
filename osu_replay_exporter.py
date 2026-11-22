"""osu! Replay Exporter - собственный рендерер osu!standard: .osr (+ карта .osz) (+ скин .osk) -> .mp4
Установка:  pip install pygame numpy imageio-ffmpeg
Запуск GUI: python osu_replay_exporter.py
Запуск CLI: python osu_replay_exporter.py replay.osr --map map.osz --skin skin.osk -o out.mp4 [--set trail=0 --set music=50 ...]
"""
import os, io, re, sys, json, math, lzma, wave, time, zlib, bisect, struct, zipfile, hashlib, argparse, tempfile, threading, subprocess
from collections import OrderedDict
import multiprocessing as mp
from pathlib import Path
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")  # иначе SDL игнорирует SIGTERM и Pool зависает при остановке
import numpy as np
import pygame
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100
CANCEL = threading.Event()  # выставляется кнопкой «Остановить» в окне
DEF_COLS = [(255, 192, 0), (0, 202, 0), (18, 124, 255), (242, 24, 57)]
DEFAULTS = dict(w=1920, h=1080, fps=60, encoder="auto", quality=20, workers=0, ffmpeg="", compress=False, lead_in=2.0, tail=1.5, start=0.0, end=0.0,
                master=100, music=100, hitsounds=True, hitvol=60, hs_source="auto", ticks=True, tickvol=40, clicks=False, clickvol=50, pan=0,
                bg=True, bgdim=75, trail=True, trail_len=14, cursor_size=100, approach=True, numbers=True, num_scale=80, judge_scale=100, arrows=True, judge=True, hit_anim=True,
                follow=True, spinner=True, slider_op=85,
                hud=True, progress=True, keys=True, keys_pos="right", keys_size=100, combo=False, acc=False, hiterr=False, fontscale=100, skin_font=True, font_name="")
clamp = lambda v, a=0.0, b=1.0: max(a, min(b, v))


# ---------------------------------------------------------------- .osr
class Rd:
    def __init__(self, b): self.b, self.p = b, 0
    def u(self, f):
        v = struct.unpack_from("<" + f, self.b, self.p)[0]; self.p += struct.calcsize("<" + f); return v
    def s(self):
        if self.b[self.p] == 0: self.p += 1; return ""
        self.p += 1; n = sh = 0
        while True:
            x = self.b[self.p]; self.p += 1; n |= (x & 127) << sh; sh += 7
            if not x & 128: break
        v = self.b[self.p:self.p + n].decode("utf-8", "replace"); self.p += n; return v


def parse_osr(path):
    r = Rd(Path(path).read_bytes())
    mode = r.u("B"); r.u("i"); md5 = r.s(); player = r.s(); r.s()
    for _ in range(6): r.u("h")
    r.u("i"); r.u("h"); r.u("B"); mods = r.u("i"); r.s(); r.u("q")
    n = r.u("i"); data = r.b[r.p:r.p + n]
    if mode != 0: raise ValueError("Поддерживается только osu!standard (режим 0)")
    raw = lzma.LZMADecompressor().decompress(data).decode("ascii", "ignore")
    fr, t = [], 0
    for p in raw.split(","):
        a = p.split("|")
        if len(a) < 4 or a[0] == "-12345": continue
        t += int(a[0]); x, y = float(a[1]), float(a[2])
        if x == 256 and y == -500: continue
        fr.append((t, x, y, int(float(a[3]))))
    f = np.array(fr, float)
    return dict(md5=md5, player=player, mods=mods, T=np.maximum.accumulate(f[:, 0]), X=f[:, 1], Y=f[:, 2], K=f[:, 3].astype(int))


def mods_str(m):
    n = [(1, "NF"), (2, "EZ"), (8, "HD"), (16, "HR"), (64, "DT"), (256, "HT"), (512, "NC"), (1024, "FL")]
    return "".join(s for b, s in n if m & b and not (b == 64 and m & 512)) or "NM"


# ---------------------------------------------------------------- карта
class Obj: pass


def circle3(P, plen):
    (ax, ay), (bx, by), (cx, cy) = P
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-6: return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    ta, tb, tc = (math.atan2(q[1] - uy, q[0] - ux) for q in P)
    dr = 1 if (tb - ta) % (2 * math.pi) < (tc - ta) % (2 * math.pi) else -1
    ang = ta + dr * plen / r * np.linspace(0, 1, int(plen / 3) + 8)
    return np.stack([ux + r * np.cos(ang), uy + r * np.sin(ang)], 1)


def bezier(S):
    S = np.array(S, float)
    if len(S) == 2: return S
    n = max(8, min(400, int(np.hypot(*np.diff(S, axis=0).T).sum() / 3)))
    ts = np.linspace(0, 1, n)[:, None]; P = [np.repeat(p[None], n, 0) for p in S]
    while len(P) > 1: P = [(1 - ts) * a + ts * b for a, b in zip(P, P[1:])]
    return P[0]


def build_path(o, ct, pts):
    path = circle3(pts, o.plen) if ct == "P" and len(pts) == 3 else None
    if path is None:
        if ct == "B":
            segs, cur = [], [pts[0]]
            for q in pts[1:]:
                cur.append(q)
                if q == cur[-2]: segs.append(cur[:-1]); cur = [q]
            segs.append(cur)
            parts = [bezier(s) for s in segs if len(s) > 1]
            path = np.vstack([parts[0]] + [p[1:] for p in parts[1:]]) if parts else np.array(pts, float)
        else:
            path = np.array(pts, float)
    d = np.hypot(*np.diff(path, axis=0).T)
    o.cum = np.concatenate([[0], np.cumsum(d)]); o.px, o.py = path[:, 0], path[:, 1]
    v = path[-1] - path[max(0, len(path) - 2)]; nv = math.hypot(*v) or 1
    o.dir = (v[0] / nv, v[1] / nv)


def at(o, dd):
    dd = np.asarray(dd, float); ex = np.maximum(dd - o.cum[-1], 0)
    return np.interp(dd, o.cum, o.px) + o.dir[0] * ex, np.interp(dd, o.cum, o.py) + o.dir[1] * ex


def parse_osu(text, flip):
    sec, g, dif, cols, tps, events, objs = None, {}, {}, [], [], [], []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("//"): continue
        if ln[0] == "[": sec = ln[1:-1]; continue
        if sec in ("General", "Difficulty", "Colours") and ":" in ln:
            k, v = [z.strip() for z in ln.split(":", 1)]
            if sec == "General": g[k] = v
            elif sec == "Difficulty": dif[k] = v
            elif k.lower().startswith("combo"):
                try: cols.append(tuple(int(z) for z in v.split(",")[:3]))
                except ValueError: pass
        elif sec == "Events": events.append(ln)
        elif sec == "TimingPoints":
            a = ln.split(",")
            if len(a) >= 2:
                tps.append((float(a[0]), float(a[1]), len(a) < 7 or a[6] == "1", int(a[3]) if len(a) > 3 else 1, int(a[5]) if len(a) > 5 else 100))
        elif sec == "HitObjects": objs.append(ln)
    tps.sort(key=lambda z: (z[0], not z[2]))
    tm, bl, sv, ss, vl, st = [], 500.0, 1.0, 1, 100, []
    for t, b, un, s_, v_ in tps:
        if un: bl, sv = b, 1.0
        else: sv = clamp(-100 / b if b else 1, 0.1, 10)
        ss = s_ or ss; vl = v_; tm.append(t); st.append((bl, sv, ss, vl))
    if not tm: tm, st = [0], [(500.0, 1.0, 1, 100)]
    smul = float(dif.get("SliderMultiplier", 1.4)); out = []
    fy = (lambda y: 384 - y) if flip else (lambda y: y)
    for ln in objs:
        a = ln.split(","); o = Obj(); o.x, o.y, o.t, ty = float(a[0]), fy(float(a[1])), int(float(a[2])), int(a[3])
        o.nc, o.skip, o.kind, o.end, o.hs = bool(ty & 4), (ty >> 4) & 7, "c", o.t, int(a[4])
        bl, sv, o.ss, o.vol = st[max(0, bisect.bisect_right(tm, o.t) - 1)]; o.bl = bl
        hi = 10 if ty & 2 else 6 if ty & 8 else 5
        try:
            ns = int(a[hi].split(":")[0]) if len(a) > hi else 0
            if ns: o.ss = ns
            v2 = int(a[hi].split(":")[3]) if len(a) > hi else 0
            if v2: o.vol = v2
        except (ValueError, IndexError): pass
        if ty & 2:
            p = a[5].split("|")
            pts = [(o.x, o.y)] + [(float(q.split(":")[0]), fy(float(q.split(":")[1]))) for q in p[1:]]
            o.kind, o.slides, o.plen = "s", int(a[6]), float(a[7])
            build_path(o, p[0], pts)
            o.one = o.plen / (smul * 100 * sv) * bl; o.end = o.t + o.one * o.slides
        elif ty & 8: o.kind, o.end, o.nc = "p", int(a[5]), True
        out.append(o)
    bg = None
    for e in events:
        mm = re.match(r'0,0,"([^"]+)"', e)
        if mm: bg = mm.group(1); break
    return g, dif, cols, out, bg


# ---------------------------------------------------------------- скин
class Skin:
    def __init__(self, osk):
        self.f, self.ini, self.cache = {}, {}, {}
        if osk:
            with zipfile.ZipFile(osk) as z:
                for n in z.namelist():
                    if not n.endswith("/"): self.f[Path(n).name.lower()] = z.read(n)
        for ln in self.f.get("skin.ini", b"").decode("utf-8-sig", "ignore").splitlines():
            ln = ln.strip()
            if ":" in ln and not ln.startswith(("//", "[")):
                k, v = ln.split(":", 1); self.ini[k.strip().lower()] = v.split("//")[0].strip()
        self.cols = []
        for i in range(1, 9):
            v = self.ini.get(f"combo{i}")
            if v:
                try: self.cols.append(tuple(int(z) for z in v.split(",")[:3]))
                except ValueError: pass

    def img(self, name):
        if name in self.cache: return self.cache[name]
        r = None
        for suf, sc in (("@2x.png", .5), (".png", 1.0)):
            d = self.f.get((name + suf).lower())
            if d:
                try: r = (pygame.image.load(io.BytesIO(d)).convert_alpha(), sc); break
                except Exception: pass
        self.cache[name] = r; return r


# ---------------------------------------------------------------- рендер
class Renderer:
    def __init__(self, W, H, sk, bgdata, opt, pre, fin, r, cols):
        pygame.init(); pygame.display.set_mode((1, 1)); self.W, self.H, self.sk, self.o = W, H, sk, opt
        self.pre, self.fin, self.r, self.cols = pre, fin, r, cols
        self.scr = pygame.Surface((W, H)); self.s = H / 480; self.d = 2 * r * self.s
        self.ox, self.oy = (W - 512 * self.s) / 2, (H - 384 * self.s) / 2 + 8 * self.s
        self.fast = self.scr.get_bytesize() == 4 and self.scr.get_pitch() == W * 4 and self.scr.get_masks()[:3] == (0xff0000, 0xff00, 0xff)
        self.pix = "bgr0" if self.fast else "rgb24"
        self.c, self.bg = {}, None
        self.sc, self.scb, self.sclim = OrderedDict(), 0, 64 * 2**20  # кэш масштабированных спрайтов (LRU, ~64 МБ на процесс)
        self.win = (80, 140, 200); self.barsf = None
        if bgdata and opt["bg"]:
            try:
                im = pygame.image.load(io.BytesIO(bgdata)).convert(); k = max(W / im.get_width(), H / im.get_height())
                im = pygame.transform.smoothscale(im, (int(im.get_width() * k) + 1, int(im.get_height() * k) + 1))
                self.bg = pygame.Surface((W, H)); self.bg.blit(im, ((W - im.get_width()) // 2, (H - im.get_height()) // 2))
                v = int(255 * (1 - clamp(opt["bgdim"] / 100))); self.bg.fill((v, v, v), special_flags=pygame.BLEND_RGB_MULT)
            except Exception: self.bg = None
        fs = clamp(opt["fontscale"] / 100, .5, 3)
        self.font, self.big = self.mkfont(max(12, int(H / 32 * fs))), self.mkfont(max(18, int(H / 14 * fs)))
        self.jfont = self.mkfont(max(14, int(H / 20 * clamp(opt["judge_scale"] / 100, .3, 4))))

    def mkfont(self, size):
        nm = (self.o.get("font_name") or "").strip()
        if nm:
            try: return pygame.font.SysFont(nm, size)
            except Exception: pass
        return pygame.font.Font(None, size)

    def skin_text(self, txt, prefix_key, overlap_key, k):
        """Текст шрифтом скина (score-0..9, score-x, score-percent, score-dot). None, если в скине нет нужных картинок."""
        pre = self.sk.ini.get(prefix_key, "score").replace("\\", "/").split("/")[-1]; names = {"x": "x", "%": "percent", ".": "dot", ",": "comma"}
        gl = []
        for ch in txt:
            nm = f"{pre}-{names.get(ch, ch)}"
            if not self.sk.img(nm): return None
            gl.append(self.scaled(nm, k, lambda: None))
        try: ov = float(self.sk.ini.get(overlap_key, 0) or 0) * k
        except ValueError: ov = 0
        w = int(sum(g.get_width() for g in gl) - ov * (len(gl) - 1)); h = max(g.get_height() for g in gl)
        if w < 1: return None
        out = pygame.Surface((w, h), pygame.SRCALPHA); x = 0.0
        for g in gl: out.blit(g, (x, (h - g.get_height()) / 2)); x += g.get_width() - ov
        return out

    def P(self, x, y): return self.ox + x * self.s, self.oy + y * self.s

    def zoom(self, im, q):
        key = (id(im), q); hit = self.sc.get(key)
        if hit is None:
            k = math.exp(q * 0.0198)
            sf = pygame.transform.smoothscale(im, (max(1, int(im.get_width() * k)), max(1, int(im.get_height() * k))))
            hit = (im, sf)  # исходник держим в записи, чтобы id не переиспользовался
            self.sc[key] = hit; self.scb += sf.get_width() * sf.get_height() * 4
            while self.scb > self.sclim and len(self.sc) > 1:
                _, (_, old) = self.sc.popitem(last=False); self.scb -= old.get_width() * old.get_height() * 4
        else: self.sc.move_to_end(key)
        return hit[1]

    def put(self, im, x, y, a=1.0, k=1.0):
        if k != 1.0:
            q = round(math.log(max(k, 1e-3)) / 0.0198)  # масштаб квантуется с шагом ~2% -> спрайты переиспользуются между кадрами
            if q: im = self.zoom(im, q)
        im.set_alpha(int(255 * clamp(a))); self.scr.blit(im, (x - im.get_width() / 2, y - im.get_height() / 2))

    def ring(self, d, col, w, fill=None):
        key = ("ring", int(d), col, w, fill)
        if key not in self.c:
            sf = pygame.Surface((int(d), int(d)), pygame.SRCALPHA); c = int(d) // 2
            if fill: pygame.draw.circle(sf, fill, (c, c), c)
            pygame.draw.circle(sf, col, (c, c), c, w); self.c[key] = sf
        return self.c[key]

    def scaled(self, name, k, fallback):
        key = ("sc", name, k)
        if key not in self.c:
            im = self.sk.img(name)
            if im: self.c[key] = pygame.transform.smoothscale(im[0], (max(1, int(im[0].get_width() * im[1] * k)), max(1, int(im[0].get_height() * im[1] * k))))
            else: self.c[key] = fallback()
        return self.c[key]

    def circ(self, col, num, sl=False):
        sk = self.sk; own = sl and sk.img("sliderstartcircle")  # у слайдеров своя головка, если она есть в скине
        bn, on = ("sliderstartcircle", "sliderstartcircleoverlay") if own else ("hitcircle", "hitcircleoverlay")
        key = ("c", col, num, bn)
        if key in self.c: return self.c[key]
        d = int(self.d); base = sk.img(bn)
        if base:
            im, sc = base; z = max(2, int(im.get_width() * sc * d / 128))
            b = pygame.transform.smoothscale(im, (z, z)); b.fill(col + (255,), special_flags=pygame.BLEND_RGBA_MULT)
            out = pygame.Surface((z, z), pygame.SRCALPHA); out.blit(b, (0, 0))
        else:
            z = d; out = self.ring(d, (255, 255, 255), max(3, d // 14), col + (255,)).copy()
        ovl = sk.img(on) if base else None; above = sk.ini.get("hitcircleoverlayabovenumber", "1") != "0"
        if ovl and not above: out.blit(pygame.transform.smoothscale(ovl[0], (z, z)), (0, 0))
        if num:
            pre = sk.ini.get("hitcircleprefix", "default").replace("\\", "/").split("/")[-1]; nsc = clamp(self.o["num_scale"] / 100, .2, 2)
            dg = [sk.img(f"{pre}-{ch}") for ch in str(num)]
            if all(dg):
                f = d / 128 * nsc; ov = float(sk.ini.get("hitcircleoverlap", 0) or 0) * f
                ws = [max(1, int(i.get_width() * sc * f)) for i, sc in dg]; hs = [max(1, int(i.get_height() * sc * f)) for i, sc in dg]
                x = (z - (sum(ws) - ov * (len(ws) - 1))) / 2
                for (i, sc), w, h in zip(dg, ws, hs):
                    out.blit(pygame.transform.smoothscale(i, (w, h)), (x, (z - h) / 2)); x += w - ov
            else:
                t = self.mkfont(max(8, int(d * .6 * nsc / .8))).render(str(num), True, (255, 255, 255)); out.blit(t, ((z - t.get_width()) / 2, (z - t.get_height()) / 2))
        if ovl and above: out.blit(pygame.transform.smoothscale(ovl[0], (z, z)), (0, 0))
        self.c[key] = out; return out

    def arrow(self, o, at_end, a):
        """Стрелка повтора слайдера (reversearrow) в точке следующего разворота."""
        if not self.sk.img("reversearrow"): return
        if not hasattr(o, "ang"):
            ex, ey = (float(v) for v in at(o, o.plen)); bx, by = (float(v) for v in at(o, max(0.0, o.plen - 20)))
            sx, sy = (float(v) for v in at(o, 0.0)); nx, ny = (float(v) for v in at(o, min(o.plen, 20.0)))
            o.ang = (math.degrees(math.atan2(ny - sy, nx - sx)), math.degrees(math.atan2(by - ey, bx - ex)))
        ang = int(round(o.ang[1 if at_end else 0] / 6)) * 6; key = ("ra", ang)
        if key not in self.c: self.c[key] = pygame.transform.rotate(self.scaled("reversearrow", self.d / 128, lambda: None), -ang)
        px, py = (o.px[-1], o.py[-1]) if at_end else (o.px[0], o.py[0]); x, y = self.P(float(px), float(py))
        self.put(self.c[key], x, y, a)

    def approach(self, col):
        key = ("ap", col)
        if key not in self.c:
            im = self.sk.img("approachcircle"); d = int(self.d)
            if im:
                z = max(2, int(im[0].get_width() * im[1] * d / 128)); a = pygame.transform.smoothscale(im[0], (z, z)); a.fill(col + (255,), special_flags=pygame.BLEND_RGBA_MULT)
            else: a = self.ring(d, col, max(2, d // 24))
            self.c[key] = a
        return self.c[key]

    def body(self, o, col):
        b = self.c.get(id(o))
        if b: return b
        rr = self.r * self.s; n = int(o.plen / (self.r * .25)) + 2
        xs, ys = at(o, np.linspace(0, o.plen, n)); xs, ys = self.ox + xs * self.s, self.oy + ys * self.s
        pad = int(rr) + 3; x0, y0 = int(xs.min()) - pad, int(ys.min()) - pad
        sf = pygame.Surface((int(xs.max()) - x0 + pad, int(ys.max()) - y0 + pad), pygame.SRCALPHA)
        bc = tuple(int(v) for v in self.sk.ini.get("sliderborder", "255,255,255").split(",")[:3])
        tc = self.sk.ini.get("slidertrackoverride")
        tc = tuple(int(v) for v in tc.split(",")[:3]) if tc else tuple(int(c * .3) for c in col)
        for x, y in zip(xs, ys): pygame.draw.circle(sf, bc + (255,), (x - x0, y - y0), rr)
        for x, y in zip(xs, ys): pygame.draw.circle(sf, tc + (255,), (x - x0, y - y0), rr * .86)
        self.c[id(o)] = (sf, x0, y0); return self.c[id(o)]

    def head(self, t, o):
        ai = clamp((t - (o.t - self.pre)) / self.fin); x, y = self.P(o.x, o.y); im = self.circ(o.col, o.num if self.o["numbers"] else 0, o.kind == "s")
        if t < o.hit_t:
            self.put(im, x, y, ai)
            if t < o.t and self.o["approach"]: self.put(self.approach(o.col), x, y, ai, 1 + 3 * (o.t - t) / self.pre)
        elif self.o["hit_anim"] and t < o.hit_t + 240:
            p = (t - o.hit_t) / 240; self.put(im, x, y, 1 - p, 1 + .4 * p)
        if self.o["judge"] and o.res != 300 and o.hit_t <= t < o.hit_t + 500:
            fr = self.judge_frames(o.res)
            if fr: (sf, sc_), k = fr[min(len(fr) - 1, int((t - o.hit_t) / 300 * len(fr)))], self.s * .8 * clamp(self.o["judge_scale"] / 100, .3, 4)
            else:
                sf = self.jfont.render({100: "100", 50: "50", 0: "X"}[o.res], True, {100: (80, 255, 80), 50: (255, 200, 60), 0: (255, 60, 60)}[o.res]); k = 1
            if fr: k *= sc_
            self.put(sf, x, y, 1 - (t - o.hit_t) / 500, k)

    def judge_frames(self, res):
        """Картинки 100/50/промаха из скина: анимация hit100-0, hit100-1… или статичная hit100. Пустые (1-2 px) заглушки скина игнорируются."""
        key = ("j", res)
        if key not in self.c:
            nm = {100: "hit100", 50: "hit50", 0: "hit0"}[res]; fr, i = [], 0
            while self.sk.img(f"{nm}-{i}"): fr.append(self.sk.img(f"{nm}-{i}")); i += 1
            if not fr:
                im = self.sk.img(nm)
                if im and im[0].get_width() > 3 and im[0].get_height() > 3: fr = [im]
            self.c[key] = fr
        return self.c[key]

    def err_bar(self, errs):
        """Полоса точности попаданий: зоны 300/100/50 и риски последних попаданий (раннее слева, позднее справа)."""
        bw, bh = int(self.H * .55), max(6, self.H // 120); w3, w1, w5 = self.win; pad = 12
        if self.barsf is None: self.barsf = pygame.Surface((bw + pad * 2, bh * 5), pygame.SRCALPHA)
        sf = self.barsf; sf.fill((0, 0, 0, 0)); cy = sf.get_height() // 2; sc = bw / 2 / max(1.0, w5)
        for w, col in ((w5, (255, 170, 60, 150)), (w1, (80, 220, 80, 170)), (w3, (80, 170, 255, 190))):
            ww = int(w * sc * 2); pygame.draw.rect(sf, col, (pad + bw // 2 - ww // 2, cy - bh // 2, ww, bh))
        for age, dt in errs:
            x = pad + bw // 2 + int(clamp(dt / w5, -1, 1) * bw / 2); a = int(255 * clamp(1 - age / 4000))
            pygame.draw.rect(sf, (255, 255, 255, a), (x - 1, cy - bh * 2, 3, bh * 4))
        pygame.draw.rect(sf, (255, 255, 255, 255), (pad + bw // 2 - 1, cy - bh * 2, 3, bh * 4))
        self.scr.blit(sf, ((self.W - sf.get_width()) // 2, self.H - sf.get_height() - 14))

    def frame(self, t, objs, lo, hi, curv, tracking, kst, hud):
        o_ = self.o; self.scr.fill((0, 0, 0))
        if self.bg: self.scr.blit(self.bg, (0, 0))
        for o in reversed(objs[lo:hi]):
            if o.kind == "c": self.head(t, o)
            elif o.kind == "s":
                ai = clamp((t - (o.t - self.pre)) / self.fin); ao = 1 if t <= o.end else clamp(1 - (t - o.end) / 240)
                sf, x0, y0 = self.body(o, o.col); sf.set_alpha(int(255 * o_["slider_op"] / 100 * min(ai, ao))); self.scr.blit(sf, (x0, y0))
                if o_["arrows"] and o.slides > 1 and int(clamp((t - o.t) / o.one, 0, o.slides)) + 1 < o.slides:
                    self.arrow(o, (int(clamp((t - o.t) / o.one, 0, o.slides)) + 1) % 2 == 1, min(ai, ao))
                self.head(t, o)
                if o.t <= t <= o.end:
                    p = (t - o.t) / o.one; k = int(p); fr = p - k; fr = 1 - fr if k % 2 else fr
                    bx, by = at(o, fr * o.plen); bx, by = self.P(float(bx), float(by))
                    ball = self.scaled("sliderb0", self.d / 128, lambda: self.ring(self.d * .9, (255, 255, 255), max(2, int(self.d / 12)), (40, 40, 40, 255)))
                    self.put(ball, bx, by)
                    if self.o["follow"] and tracking(o, t): self.put(self.ring(self.d * 2.4, (255, 255, 255), max(2, int(self.d / 20))), bx, by, .6)
            elif o_["spinner"] and o.t - 400 <= t <= o.end:
                k = clamp((o.end - t) / max(1, o.end - o.t)); self.put(self.ring(self.H * .7, (255, 255, 255), 4), self.W / 2, self.H / 2, .8, .15 + .85 * k)
        z = self.H / 768 * o_["cursor_size"] / 100  # как в osu!: 1x-картинка скина рисуется в масштабе H/768
        if o_["trail"] and o_["trail_len"] > 0:
            L = int(o_["trail_len"]); xs, ys = curv(t - np.arange(L, 0, -1) * 7)
            tr = self.scaled("cursortrail", z, lambda: self.ring(self.H / 60 * o_["cursor_size"] / 100, (255, 255, 255), max(1, self.H // 120)))
            for i, (tx, ty) in enumerate(zip(xs, ys)):
                self.put(tr, *self.P(tx, ty), .35 * (i + 1) / (L + 1))
        cx, cy = curv(np.array([t])); X, Y = self.P(cx[0], cy[0])
        cm = self.scaled("cursor", z, lambda: self.ring(self.H / 36 * o_["cursor_size"] / 100, (255, 255, 255), max(2, self.H // 270), (255, 255, 255, 90)))
        self.put(cm, X, Y)
        if self.sk.img("cursormiddle"): self.put(self.scaled("cursormiddle", z, lambda: None), X, Y)
        if o_["hud"]: self.scr.blit(self.font.render(hud[0], True, (255, 255, 255)), (12, 8))
        fk = self.H / 768 * clamp(o_["fontscale"] / 100, .3, 3)
        if o_["acc"]:
            txt = f"{hud[3]:.2f}%"; tx = (self.skin_text(txt, "scoreprefix", "scoreoverlap", fk * .6) if o_["skin_font"] else None) or self.font.render(txt, True, (255, 255, 255))
            self.scr.blit(tx, (self.W - tx.get_width() - 14, 8))
        if o_["combo"]:
            txt = f"{hud[2]}x"; tx = (self.skin_text(txt, "comboprefix", "combooverlap", fk) if o_["skin_font"] else None) or self.big.render(txt, True, (255, 255, 255))
            self.scr.blit(tx, (16, self.H - tx.get_height() - 12))
        if o_["hiterr"] and hud[4]: self.err_bar(hud[4])
        if o_["progress"]: pygame.draw.rect(self.scr, (255, 255, 255), (0, self.H - 4, int(self.W * hud[1]), 4))
        if o_["keys"]:
            bits, cnt = kst(t); bs = max(12, int(self.H // 16 * clamp(o_["keys_size"] / 100, .3, 3))); gap = max(3, bs // 8)
            x = 14 if o_["keys_pos"] == "left" else self.W - bs - 14; y0 = self.H // 2 - 2 * bs
            for j, (nm, b) in enumerate((("K1", 4), ("K2", 8), ("M1", 1), ("M2", 2))):
                rc = pygame.Rect(x, y0 + j * (bs + gap), bs, bs)
                pygame.draw.rect(self.scr, (255, 220, 80) if bits & b else (30, 30, 30), rc, 0, 6); pygame.draw.rect(self.scr, (200, 200, 200), rc, 2, 6)
                tx = self.font.render(str(cnt[j]), True, (0, 0, 0) if bits & b else (255, 255, 255)); self.scr.blit(tx, (rc.centerx - tx.get_width() / 2, rc.centery - tx.get_height() / 2))
        return self.scr.get_buffer() if self.fast else pygame.image.tobytes(self.scr, "RGB")


# ---------------------------------------------------------------- подготовка / сегменты
def read_map(mapf, md5):
    mp_, files = Path(mapf), {}
    if mp_.suffix.lower() == ".osz":
        with zipfile.ZipFile(mp_) as z:
            for n in z.namelist():
                if not n.endswith("/"): files[Path(n).name.lower()] = z.read(n)
    else:
        for f in mp_.parent.iterdir():
            if f.is_file() and (f == mp_ or f.suffix.lower() in (".mp3", ".ogg", ".wav", ".jpg", ".jpeg", ".png")): files[f.name.lower()] = f.read_bytes()
    osus = [n for n in files if n.endswith(".osu")]
    pick = next((n for n in osus if hashlib.md5(files[n]).hexdigest() == md5), None)
    return files, pick, osus


def prepare(osr, mapf, osk, opt, need_r=True, log=print):
    rp = parse_osr(osr); m = rp["mods"]; log(f"Реплей: {rp['player']}, моды {mods_str(m)}")
    files, pick, osus = read_map(mapf, rp["md5"])
    if not osus: raise ValueError("В карте нет .osu файлов")
    if not pick:
        pick = osus[0]; log("ВНИМАНИЕ: хеш реплея не совпал с картой, использую " + pick)
    g, dif, bcols, objs, bg = parse_osu(files[pick].decode("utf-8-sig", "ignore"), bool(m & 16))
    sk = Skin(osk); cols = bcols or sk.cols or DEF_COLS
    cs, od = float(dif.get("CircleSize", 5)), float(dif.get("OverallDifficulty", 5)); ar = float(dif.get("ApproachRate", od))
    if m & 16: cs, ar, od = min(cs * 1.3, 10), min(ar * 1.4, 10), min(od * 1.4, 10)
    if m & 2: cs, ar, od = cs / 2, ar / 2, od / 2
    rate = 1.5 if m & (64 | 512) else .75 if m & 256 else 1.0
    pre = 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5
    fin = 800 + 400 * (5 - ar) / 5 if ar < 5 else 800 - 500 * (ar - 5) / 5
    win = (80 - 6 * od, 140 - 8 * od, 200 - 10 * od); rad = 54.4 - 4.48 * cs
    T, X, Y, K = rp["T"], rp["X"], rp["Y"], rp["K"]
    cur = lambda t: (float(np.interp(t, T, X)), float(np.interp(t, T, Y)))
    curv = lambda ts: (np.interp(ts, T, X), np.interp(ts, T, Y))
    keys = lambda t: int(K[max(0, int(np.searchsorted(T, t, "right")) - 1)]) & 15
    KC, cnt, prev = np.zeros((len(T), 4), int), [0, 0, 0, 0], 0
    for i, k in enumerate(K):
        for j, b in enumerate((4, 8, 1, 2)):
            if int(k) & b and not prev & b: cnt[j] += 1
        KC[i] = cnt; prev = int(k)

    def kst(t):
        i = max(0, int(np.searchsorted(T, t, "right")) - 1); return int(K[i]), KC[i]
    idx = num = 0
    for i, o in enumerate(objs):
        if o.nc or i == 0:
            if i: idx += 1 + o.skip
            num = 1
        else: num += 1
        o.col, o.num = cols[idx % len(cols)], num
    presses, prev = [], 0
    for t, x, y, k in zip(T, X, Y, K):
        if int(k) & ~prev & 15: presses.append((t, x, y))
        prev = int(k) & 15
    pt, used = [p[0] for p in presses], set()
    for o in objs:
        o.res, o.hit_t = 0, o.t + win[2]
        if o.kind == "p": o.vis = o.end + 240; continue
        for j in range(bisect.bisect_left(pt, o.t - win[2]), len(presses)):
            t, x, y = presses[j]
            if t > o.t + win[2]: break
            if j not in used and math.hypot(x - o.x, y - o.y) <= rad:
                used.add(j); o.hit_t = t; dt = abs(t - o.t); o.res = 300 if dt <= win[0] else 100 if dt <= win[1] else 50; break
        o.vis = max(o.hit_t + 520, o.end + 240)

    def tracking(o, t):
        p = (t - o.t) / o.one; k = int(p); fr = p - k; fr = 1 - fr if k % 2 else fr
        bx, by = at(o, fr * o.plen); x, y = cur(t)
        return keys(t) != 0 and math.hypot(x - float(bx), y - float(by)) <= rad * 2.4
    t0 = opt["start"] * 1000 if opt["start"] > 0 else objs[0].t - opt["lead_in"] * 1000
    t1 = opt["end"] * 1000 if opt["end"] > 0 else min(max(o.end for o in objs) + opt["tail"] * 1000, T[-1] + 800)
    if t1 <= t0: raise ValueError("Конец клипа раньше начала")
    ev = sorted([(o.end, 300) if o.kind == "p" else (o.hit_t, o.res) for o in objs])  # оценка по головам кружков/слайдеров
    JT = np.array([e[0] for e in ev], float); cb, combo_l = 0, []
    for _, r_ in ev: cb = 0 if r_ == 0 else cb + 1; combo_l.append(cb)
    rs = np.array([e[1] for e in ev], float); ACC = np.cumsum(rs) / (300 * np.arange(1, len(rs) + 1)) * 100; CMB = np.array(combo_l)
    hits = sorted((o.hit_t, o.hit_t - o.t) for o in objs if o.kind != "p" and o.res > 0); HT = np.array([h[0] for h in hits], float); HD = [h[1] for h in hits]

    def stat(t):
        i = int(np.searchsorted(JT, t, "right")) - 1; j = int(np.searchsorted(HT, t, "right"))
        errs = [(t - HT[m], HD[m]) for m in range(max(0, j - 30), j) if t - HT[m] < 4000]
        return (int(CMB[i]), float(ACC[i])) if i >= 0 else (0, 100.0), errs
    W, H, fps = int(opt["w"]), int(opt["h"]), int(opt["fps"]); N = int((t1 - t0) / (1000 * rate) * fps)
    R = Renderer(W, H, sk, files.get(Path(bg).name.lower()) if bg else None, opt, pre, fin, rad, cols) if need_r else None
    if R: R.win = win
    return dict(R=R, objs=objs, curv=curv, tracking=tracking, kst=kst, stat=stat, presses=presses, t0=t0, N=N, rate=rate, W=W, H=H, fps=fps, lo=0,
                hud=f"{rp['player']}  {mods_str(m)}", files=files, g=g, sk=sk, dif=dif)


CTX = None


def _init(osr, mapf, osk, opt, encs, ffm):
    global CTX, FFMPEG
    FFMPEG = ffm; CTX = prepare(osr, mapf, osk, opt, True, lambda s: None); CTX["encs"] = encs


def frame_at(c, i):
    R, objs = c["R"], c["objs"]; t = c["t0"] + i * 1000 * c["rate"] / c["fps"]; lo = c["lo"]
    while lo < len(objs) and objs[lo].vis < t: R.c.pop(id(objs[lo]), None); lo += 1
    c["lo"] = lo; hi = lo
    while hi < len(objs) and objs[hi].t - R.pre <= t: hi += 1
    (cmb, acc), errs = c["stat"](t)
    return R.frame(t, objs, lo, hi, c["curv"], c["tracking"], c["kst"], (c["hud"], i / c["N"], cmb, acc, errs))


def _seg(task):
    i0, i1, path = task; c = CTX; c["lo"] = 0
    for ei, enc in enumerate(c["encs"]):
        if ei: print("\nПредупреждение: NVENC не смог закодировать сегмент, перехожу на libx264 (медленнее)", flush=True)
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", c["R"].pix, "-s", f"{c['W']}x{c['H']}", "-r", str(c["fps"]), "-i", "-"] + enc + ["-pix_fmt", "yuv420p", "-an", path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(i0, i1): p.stdin.write(frame_at(c, i))
            p.stdin.close()
            if p.wait() == 0: return path, i1 - i0
        except (BrokenPipeError, OSError):
            p.kill()
    raise RuntimeError("Не удалось закодировать сегмент видео")


def enc_test(enc, w, h):
    """Пробное кодирование 0.5 сек теми же аргументами, что пойдут в работу. Возвращает (ok, текст ошибки)."""
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30:d=0.5", "-pix_fmt", "yuv420p"] + enc + ["-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True); return r.returncode == 0, r.stderr.decode("utf-8", "replace").strip()[-400:]
    except Exception as e: return False, str(e)


def free_commit_mb():
    """Свободная память с учётом файла подкачки (Windows: доступный commit), МБ. None, если узнать не удалось."""
    try:
        if os.name == "nt":
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                            ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
            m = MS(); m.dwLength = ctypes.sizeof(m); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ullAvailPageFile / 2**20
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20
    except Exception: return None


def nv_sessions(nv, w, h, n):
    """Сколько NVENC-сессий видеокарта реально отдаёт одновременно: запускаем n параллельных кодирований в реальном времени."""
    cmd = [FFMPEG, "-v", "error", "-re", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30:d=1.5", "-pix_fmt", "yuv420p"] + nv + ["-f", "null", "-"]
    try:
        ps = [subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]
        return sum(p.wait() == 0 for p in ps)
    except Exception: return n


def choose_encoders(opt, log):
    q = str(int(opt["quality"])); w, h = int(opt["w"]), int(opt["h"])
    x = ["-c:v", "libx264", "-preset", "veryfast", "-crf", q]
    try: log("ffmpeg: " + subprocess.run([FFMPEG, "-version"], capture_output=True).stdout.decode("utf-8", "replace").splitlines()[0])
    except Exception: pass
    if opt["encoder"] == "x264": log("Энкодер: libx264 (процессор)"); return [x]
    err = ""
    # старые ffmpeg (4.2 из imageio-ffmpeg) не знают пресетов p1..p7, поэтому пробуем по очереди: новые -> старые -> по умолчанию
    for pre in (["-preset", "p4"], ["-preset", "hq"], []):
        nv = ["-c:v", "h264_nvenc"] + pre + ["-rc", "vbr", "-cq", q, "-b:v", "0"]; ok, e = enc_test(nv, w, h)
        if ok: log(f"Энкодер: NVENC (видеокарта), {' '.join(pre) or 'пресет по умолчанию'}"); return [nv, x]
        err = e
    log("NVENC не заработал, кодирую на процессоре (libx264). Ответ ffmpeg: " + (err or "—")
        + "\n  Подсказка: обнови драйвер NVIDIA или укажи путь к свежему ffmpeg (gyan.dev, full build) в настройке «Путь к ffmpeg».")
    return [x]


# ---------------------------------------------------------------- звук
def decode(data=None, path=None, pre=(), af=None, dur=None):
    cmd = [FFMPEG, "-v", "error"] + list(pre) + ["-i", path or "pipe:0"] + (["-af", af] if af else []) + (["-t", f"{dur:.3f}"] if dur else []) + ["-f", "f32le", "-ar", str(SR), "-ac", "2", "pipe:1"]
    out = subprocess.run(cmd, input=data, capture_output=True).stdout
    return np.frombuffer(out[:len(out) // 8 * 8], np.float32).reshape(-1, 2)


def builtin(name):
    """Встроенные хитсаунды (если в скине/карте нет своих): синтез на numpy. Возвращает стерео float32 или None."""
    base, _, kind = name.partition("-"); rng = np.random.default_rng(zlib.crc32(name.encode())); tt = lambda d: np.arange(int(SR * d)) / SR
    if name == "click": kind = "click"
    f0 = {"normal": 260, "soft": 200, "drum": 150}.get(base, 260); amp = {"soft": .6}.get(base, 1.0)
    if kind == "hitnormal":
        d = tt(.12); y = np.sin(2 * np.pi * f0 * (d + .5 * (1 - np.exp(-60 * d)) / 60)) * np.exp(-d * 38) + .35 * rng.standard_normal(len(d)) * np.exp(-d * 120)
    elif kind == "hitwhistle":
        d = tt(.3); y = np.sin(2 * np.pi * 1900 * d - (120 / 9) * np.cos(2 * np.pi * 9 * d)) * np.minimum(1, d * 80) * np.exp(-d * 9)
    elif kind == "hitfinish":
        d = tt(.7); y = np.diff(rng.standard_normal(len(d)), prepend=0) * np.exp(-d * 6) * .5 + np.sin(2 * np.pi * 400 * d) * np.exp(-d * 10) * .3
    elif kind == "hitclap":
        d = tt(.18); env = sum(np.where(d >= p_, np.exp(-(d - p_) * 90), 0) for p_ in (0, .012, .024)); y = np.diff(rng.standard_normal(len(d)), prepend=0) * env * .6
    elif kind == "slidertick":
        d = tt(.06); y = np.sin(2 * np.pi * 1400 * d) * np.exp(-d * 90)
    elif kind == "click":
        d = tt(.06); y = np.diff(rng.standard_normal(len(d)), prepend=0) * np.exp(-d * 400) * .7 + np.sin(2 * np.pi * 900 * d) * np.exp(-d * 140) * .4
    else: return None
    y = y / (np.abs(y).max() + 1e-9) * .8 * amp
    return np.stack([y, y], 1).astype(np.float32)


def build_audio(c, opt, wavpath, tmp, log):
    t0, rate, ns = c["t0"], c["rate"], int(c["N"] / c["fps"] * SR); files, sk = c["files"], c["sk"]
    arr = np.zeros((ns, 2), np.float32); au = files.get(c["g"].get("AudioFilename", "").lower())
    if au and opt["music"] > 0:
        ap = Path(tmp) / ("audio" + Path(c["g"]["AudioFilename"]).suffix); ap.write_bytes(au)
        af = (f"adelay={int(-t0 / rate)}:all=1," if t0 < 0 else "") + (f"atempo={rate}," if rate != 1 else "") + f"volume={opt['music'] / 100}"
        m = decode(path=str(ap), pre=(["-ss", f"{t0 / 1000:.3f}"] if t0 >= 0 else []), af=af, dur=ns / SR)[:ns]; arr[:len(m)] = m
    elif not au: log("Аудио карты не найдено")
    cache, st = {}, dict(n=0, own=set(), bi=set())

    def sample(name):
        if name not in cache:
            s_ = None
            if opt["hs_source"] != "builtin":
                for ext in (".wav", ".ogg", ".mp3"):
                    d = files.get(name + ext) or sk.f.get(name + ext)
                    if d: s_ = decode(data=d); break
                if s_ is not None and len(s_) == 0: s_ = None
            if s_ is not None: st["own"].add(name)
            else:
                s_ = builtin(name)
                if s_ is not None: st["bi"].add(name)
            cache[name] = s_
        return cache[name]

    pan = opt["pan"] / 100

    def mix(name, tms, vol, x=256.0):
        s_ = sample(name)
        if s_ is None or vol <= 0: return
        p = int((tms - t0) / rate / 1000 * SR); a, b = max(p, 0), min(p + len(s_), ns)
        if b <= a: return
        pp = clamp((x - 256) / 256, -1, 1) * pan; g = np.array([1 - max(0, pp), 1 + min(0, pp)], np.float32) * vol
        arr[a:b] += s_[a - p:b - p] * g; st["n"] += 1

    def where(o, tt):
        if not pan or o.kind != "s": return o.x
        p = clamp((tt - o.t) / o.one, 0, o.slides); k = int(p); fr = p - k; fr = 1 - fr if k % 2 else fr
        return float(at(o, fr * o.plen)[0])

    if opt["hitsounds"] and opt["hitvol"] > 0:
        hv = opt["hitvol"] / 100

        def play(tms, o, x):
            nm = ("normal", "soft", "drum")[min(max(o.ss, 1), 3) - 1]; v = o.vol / 100 * hv
            for s_ in [f"{nm}-hitnormal"] + [f"{nm}-hit{w}" for b, w in ((2, "whistle"), (4, "finish"), (8, "clap")) if o.hs & b]: mix(s_, tms, v, x)
        tick_r = float(c["dif"].get("SliderTickRate", 1) or 1)
        for o in c["objs"]:
            if o.kind == "p" or o.res == 0: continue
            play(o.hit_t, o, o.x)
            if o.kind == "s":
                for k in range(1, o.slides + 1): play(o.t + o.one * k, o, where(o, o.t + o.one * k))
                if opt["ticks"] and opt["tickvol"] > 0:
                    ti = o.bl / tick_r; offs = [j * ti for j in range(1, int(o.one / ti) + 1) if j * ti < o.one - 10]; nm = ("normal", "soft", "drum")[min(max(o.ss, 1), 3) - 1]
                    for sp in range(o.slides):
                        for off in (offs if sp % 2 == 0 else [o.one - z for z in offs][::-1]):
                            tk = o.t + o.one * sp + off
                            if c["tracking"](o, tk): mix(f"{nm}-slidertick", tk, o.vol / 100 * opt["tickvol"] / 100, where(o, tk))
        log(f"Хитсаунды: {st['n']} звуков (своих из скина/карты: {len(st['own'])}, встроенных: {len(st['bi'])})")
    if opt["clicks"] and opt["clickvol"] > 0:
        n0 = st["n"]
        for t, x, y in c["presses"]: mix("click", t, opt["clickvol"] / 100, x)
        log(f"Звук нажатий клавиш: {st['n'] - n0}")
    arr *= opt["master"] / 100; ab = np.abs(arr); th = .9; m = ab > th  # мягкое ограничение вместо жёсткого клиппинга
    arr[m] = np.sign(arr[m]) * (th + (1 - th) * np.tanh((ab[m] - th) / (1 - th)))
    pcm = (np.clip(arr, -1, 1) * 32767).astype("<i2")
    with wave.open(str(wavpath), "wb") as w: w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- пост-сжатие готового .mp4
COMPRESS_CRF_OFFSET = 0     # CRF при сжатии = quality - OFFSET (0 = тот же CRF, но медленный пресет -> файл меньше; >0 = ещё ближе к исходнику, но файл больше)
COMPRESS_MIN_SSIM = 0.99    # если SSIM нового файла к исходному ниже — новый файл выбрасывается, остаётся оригинал


def compress_mp4(path, opt, log, total_sec=0.0):
    """Перекодирует готовый mp4 медленным x264 (veryslow) — при том же визуальном качестве файл заметно меньше, чем после быстрого NVENC/veryfast.
    Звук копируется без перекодирования. Результат проверяется по SSIM и размеру: если не меньше или качество просело — остаётся исходный файл."""
    path = Path(path); new = path.with_name(path.stem + ".compress_tmp.mp4"); old_size = path.stat().st_size
    crf = max(10, int(opt["quality"]) - COMPRESS_CRF_OFFSET)
    log(f"Сжатие: x264 veryslow, CRF {crf} (может занять долго)…")
    try:
        cmd = [FFMPEG, "-y", "-v", "error", "-i", str(path), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryslow", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(new)]
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"); last = -1
        errs = []; threading.Thread(target=lambda: errs.append(pr.stderr.read()), daemon=True).start()
        for line in pr.stdout:
            if line.startswith("out_time_us=") and total_sec > 0:
                try: pct = int(line.split("=")[1]) / 1e6 / total_sec * 100
                except ValueError: continue
                if int(pct // 10) > last: last = int(pct // 10); log(f"  сжатие: ~{min(pct, 100):.0f}%")
        if pr.wait() or not new.is_file(): raise RuntimeError("ffmpeg: " + (errs[0] if errs else "ошибка"))
        new_size = new.stat().st_size
        if new_size >= old_size: log(f"Сжатие не дало выигрыша ({new_size / 1e6:.1f} МБ против {old_size / 1e6:.1f} МБ) — оставляю исходный файл"); return
        log("Проверяю, что качество не пострадало (SSIM)…")
        r = subprocess.run([FFMPEG, "-v", "info", "-i", str(new), "-i", str(path), "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        m = re.findall(r"All:([0-9.]+)", r.stderr); ssim = float(m[-1]) if m else 0.0
        if ssim < COMPRESS_MIN_SSIM: log(f"SSIM {ssim:.4f} < {COMPRESS_MIN_SSIM} — качество просело, оставляю исходный файл"); return
        os.replace(new, path)
        log(f"Сжато: {old_size / 1e6:.1f} МБ -> {new_size / 1e6:.1f} МБ (на {(1 - new_size / old_size) * 100:.0f}% меньше), SSIM {ssim:.4f}")
    except Exception as e:
        log(f"Сжатие не удалось ({e}) — исходный файл не тронут")
    finally:
        try: new.unlink()
        except OSError: pass


# ---------------------------------------------------------------- главная функция
def render(osr, mapf, osk, out, opt=None, log=print, prog=lambda p: None):
    global FFMPEG
    opt = {**DEFAULTS, **(opt or {})}
    if opt["ffmpeg"]: FFMPEG = opt["ffmpeg"]
    c = prepare(osr, mapf, osk, opt, False, log); N, fps = c["N"], c["fps"]
    from collections import Counter
    cn = Counter(o.res for o in c["objs"] if o.kind != "p"); log(f"Оценка попаданий: 300 — {cn[300]}, 100 — {cn[100]}, 50 — {cn[50]}, промахов — {cn[0]}")
    encs = choose_encoders(opt, log); workers = int(opt["workers"]) or max(1, min((os.cpu_count() or 2) - 2, 5))
    free = free_commit_mb()  # каждый поток = процесс Python + pygame + ffmpeg, ~1.2 ГБ с запасом; иначе WinError 1455 / Cannot allocate memory
    if free:
        cap = max(1, int((free - 1500) // 1200))
        if cap < workers:
            log(f"Свободной памяти (с файлом подкачки) ~{free / 1024:.1f} ГБ — хватит на {cap} потоков рендера, снижаю с {workers}. Чтобы больше: закрой лишние программы или увеличь файл подкачки Windows."); workers = cap
    if "h264_nvenc" in encs[0]:
        ok = nv_sessions(encs[0], int(opt["w"]), int(opt["h"]), workers)
        if 0 < ok < workers: log(f"Драйвер даёт только {ok} одновременных NVENC-сессий (запрошено {workers}): ставлю потоков рендера = {ok}. Новее драйвер NVIDIA — больше сессий."); workers = ok
        elif ok == 0: log("NVENC не принимает параллельные сессии, кодирую на процессоре"); encs = encs[1:]
    tmp = Path(tempfile.mkdtemp()); chunk = max(fps * 2, -(-N // (workers * 4)))
    tasks = [(i, min(i + chunk, N), str(tmp / f"s{k:04d}.mp4")) for k, i in enumerate(range(0, N, chunk))]
    log(f"Кадров: {N}, потоков рендера: {workers}"); done = 0; ts = time.time()
    pool = mp.get_context("spawn").Pool(workers, _init, (osr, mapf, osk, opt, encs, FFMPEG))
    try:
        for _, n in pool.imap_unordered(_seg, tasks):
            done += n; prog(done / N * 90)
            if CANCEL.is_set(): raise RuntimeError("Остановлено пользователем")
            el = time.time() - ts; log(f"  {done}/{N} кадров, {done / el:.1f} кадр/с, осталось ~{(N - done) / (done / el) / 60:.1f} мин")
        pool.close(); pool.join()
    except BaseException:
        pool.terminate(); raise
    log("Собираю звук и финальный файл…")
    wav = tmp / "mix.wav"; build_audio(c, opt, wav, tmp, log)
    lst = tmp / "list.txt"; lst.write_text("".join(f"file '{Path(t[2]).as_posix()}'\n" for t in tasks), encoding="utf-8")
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)]
    if subprocess.run(cmd).returncode: raise RuntimeError("ffmpeg не смог собрать итоговый файл")
    prog(100); log(f"Готово: {out}")
    if opt["compress"]: compress_mp4(out, opt, log, N / fps)


# ---------------------------------------------------------------- GUI / CLI
SETF = Path(__file__).with_name("exporter_settings.json")


def gui():
    import queue
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    if sys.platform == "win32":  # чёткий текст на мониторах с масштабом
        try:
            import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass
    try: saved = json.loads(SETF.read_text(encoding="utf-8"))
    except Exception: saved = {}
    root = tk.Tk(); root.title("osu! Replay Exporter"); root.geometry("840x800"); root.minsize(760, 700)
    try: ttk.Style().theme_use("vista" if sys.platform == "win32" else "clam")
    except tk.TclError: pass
    ttk.Style().configure("Big.TButton", font=("Segoe UI", 12, "bold"), padding=8)
    root.columnconfigure(0, weight=1); root.rowconfigure(4, weight=1)

    st = {**DEFAULTS, **saved.get("opt", {})}
    def mkvar(k):
        d = DEFAULTS[k]; cls = tk.BooleanVar if isinstance(d, bool) else tk.IntVar if isinstance(d, int) else tk.DoubleVar if isinstance(d, float) else tk.StringVar
        try: return cls(value=type(d)(st[k]))
        except Exception: return cls(value=d)
    o = {k: mkvar(k) for k in DEFAULTS}
    v = {k: tk.StringVar(value=saved.get(k, "")) for k in ("osr", "map", "osk", "out")}
    v["osr"].set(""); v["out"].set(saved.get("out") or str(Path.home() / "osu_replay.mp4"))
    P = lambda k: v[k].get().strip().strip('"')
    ldir, refreshers, run = {"d": saved.get("dir", "")}, [], {"on": False, "err": None, "t0": 0.0, "done": False}
    q, prog = queue.Queue(), {"p": None}

    # ---------------------------------------------------------------- файлы
    ff = ttk.LabelFrame(root, text=" Файлы ", padding=10); ff.grid(row=0, column=0, sticky="we", padx=12, pady=(12, 6)); ff.columnconfigure(1, weight=1)

    def frow(r, lab, k, types, save=False):
        ttk.Label(ff, text=lab).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 10))
        ttk.Entry(ff, textvariable=v[k]).grid(row=r, column=1, sticky="we", pady=3)
        def pick():
            kw = {"initialdir": ldir["d"]} if ldir["d"] and Path(ldir["d"]).is_dir() else {}
            p = (filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")], **kw) if save else filedialog.askopenfilename(filetypes=types, **kw))
            if p: v[k].set(p); ldir["d"] = str(Path(p).parent)
        ttk.Button(ff, text="Выбрать…", command=pick).grid(row=r, column=2, padx=(8, 0), pady=3)
    frow(0, "Реплей (.osr)", "osr", [("osu! replay", "*.osr")]); frow(1, "Карта (.osz / .osu)", "map", [("beatmap", "*.osz *.osu")])
    frow(2, "Скин (.osk) — не обязательно", "osk", [("skin", "*.osk")]); frow(3, "Сохранить как (.mp4)", "out", None, True)
    info = ttk.Label(ff, text="", foreground="#b0306a"); info.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
    auto = {"out": v["out"].get()}

    def on_osr(*_):
        p = Path(P("osr"))
        if p.suffix.lower() != ".osr" or not p.is_file(): info.config(text=""); return
        try:
            d = parse_osr(p); sec = d["T"][-1] / 1000
            info.config(text=f"{d['player']}  ·  {mods_str(d['mods'])}  ·  длина ≈ {int(sec // 60)}:{int(sec % 60):02d}")
        except Exception as e: info.config(text=f"Не удалось прочитать реплей: {e}")
        if v["out"].get() in ("", auto["out"]): auto["out"] = str(p.with_suffix(".mp4")); v["out"].set(auto["out"])
    v["osr"].trace_add("write", on_osr)

    # ---------------------------------------------------------------- настройки (вкладки)
    ttk.Label(root, text="Настройки (по умолчанию уже всё нормально — можно сразу жать «Экспортировать»)", foreground="gray40").grid(row=1, column=0, sticky="w", padx=14, pady=(6, 0))
    nb = ttk.Notebook(root); nb.grid(row=2, column=0, sticky="we", padx=12, pady=6)
    RES = {"1280×720 (720p)": (1280, 720), "1920×1080 (1080p)": (1920, 1080), "2560×1440 (1440p)": (2560, 1440), "3840×2160 (4K)": (3840, 2160)}

    def slider(f, r, k, lo, hi, unit=""):
        lbl = ttk.Label(f, width=7, anchor="e")
        show = lambda x: lbl.config(text="авто" if k == "workers" and int(x) == 0 else f"{int(x)}{unit}")
        def cmd(x): n = int(round(float(x))); o[k].set(n); show(n)
        s = ttk.Scale(f, from_=lo, to=hi, command=cmd); s.grid(row=r, column=1, sticky="we", padx=(0, 6)); lbl.grid(row=r, column=2, sticky="e")
        s.set(o[k].get()); show(o[k].get()); refreshers.append(lambda: (s.set(o[k].get()), show(o[k].get())))

    def res_rows(f, r):
        cb = ttk.Combobox(f, values=list(RES), state="readonly", width=20); cb.grid(row=r, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda e: (o["w"].set(RES[cb.get()][0]), o["h"].set(RES[cb.get()][1])))
        wh = ttk.Frame(f); wh.grid(row=r + 1, column=1, sticky="w")
        ttk.Entry(wh, textvariable=o["w"], width=7, justify="center").pack(side="left"); ttk.Label(wh, text=" × ").pack(side="left"); ttk.Entry(wh, textvariable=o["h"], width=7, justify="center").pack(side="left")
        def sync(*_):
            try: cb.set(next((n for n, x in RES.items() if x == (o["w"].get(), o["h"].get())), "своё"))
            except tk.TclError: pass
        o["w"].trace_add("write", sync); o["h"].trace_add("write", sync); sync()

    def tab(title, items):
        f = ttk.Frame(nb, padding=(14, 10)); nb.add(f, text=f"  {title}  "); f.columnconfigure(1, weight=1)
        r = 0
        for lab, k, kind, arg in items:
            if kind == "res":
                ttk.Label(f, text=lab).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 14)); ttk.Label(f, text="Своё разрешение").grid(row=r + 1, column=0, sticky="w", pady=4); res_rows(f, r); r += 2; continue
            ttk.Label(f, text=lab).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 14))
            if kind == "chk": ttk.Checkbutton(f, variable=o[k]).grid(row=r, column=1, sticky="w")
            elif kind == "sl": slider(f, r, k, *arg)
            elif kind == "combo": ttk.Combobox(f, textvariable=o[k], values=arg, state="readonly", width=12).grid(row=r, column=1, sticky="w")
            elif kind == "fps": ttk.Combobox(f, textvariable=o[k], values=[24, 30, 60, 120, 144], width=8).grid(row=r, column=1, sticky="w")
            elif kind == "num": ttk.Entry(f, textvariable=o[k], width=10).grid(row=r, column=1, sticky="w")
            elif kind == "txt": ttk.Entry(f, textvariable=o[k]).grid(row=r, column=1, columnspan=2, sticky="we")
            elif kind == "file":
                def pick(k=k):
                    p = filedialog.askopenfilename()
                    if p: o[k].set(p)
                ttk.Entry(f, textvariable=o[k]).grid(row=r, column=1, sticky="we"); ttk.Button(f, text="…", width=4, command=pick).grid(row=r, column=2, padx=(6, 0))
            r += 1

    tab("Видео", [("Разрешение", "w", "res", None), ("Частота кадров (FPS)", "fps", "fps", None), ("Энкодер (auto = видеокарта, если работает)", "encoder", "combo", ["auto", "nvenc", "x264"]),
                  ("Качество (меньше = лучше, но файл больше)", "quality", "sl", (14, 32)), ("Начало клипа, сек (0 = авто)", "start", "num", None), ("Конец клипа, сек (0 = до конца)", "end", "num", None)])
    tab("Звук", [("Общая громкость", "master", "sl", (0, 200, "%")), ("Музыка", "music", "sl", (0, 150, "%")), ("Хитсаунды", "hitsounds", "chk", None), ("Громкость хитсаундов", "hitvol", "sl", (0, 200, "%")),
                 ("Источник звуков (auto = скин/карта)", "hs_source", "combo", ["auto", "builtin"]), ("Панорама по полю", "pan", "sl", (0, 100, "%")),
                 ("Тики слайдеров", "ticks", "chk", None), ("Громкость тиков", "tickvol", "sl", (0, 200, "%")), ("Щелчки нажатий клавиш", "clicks", "chk", None), ("Громкость щелчков", "clickvol", "sl", (0, 200, "%"))])
    tab("Вид", [("Фон карты", "bg", "chk", None), ("Затемнение фона", "bgdim", "sl", (0, 100, "%")), ("Прозрачность слайдеров", "slider_op", "sl", (0, 100, "%")), ("Шлейф курсора", "trail", "chk", None),
                ("Длина шлейфа", "trail_len", "sl", (0, 40)), ("Размер курсора", "cursor_size", "sl", (30, 300, "%")), ("Значки 100 / 50 / промах", "judge", "chk", None), ("Размер значков", "judge_scale", "sl", (30, 400, "%"))])
    tab("Круги", [("Approach-круги", "approach", "chk", None), ("Номера комбо", "numbers", "chk", None), ("Размер цифр (подбери под скин)", "num_scale", "sl", (30, 150, "%")), ("Стрелки повтора слайдера", "arrows", "chk", None),
                  ("Анимация исчезновения кругов", "hit_anim", "chk", None), ("Кольцо следования слайдера", "follow", "chk", None), ("Спиннеры", "spinner", "chk", None)])
    tab("HUD", [("Ник и моды", "hud", "chk", None), ("Полоска прогресса", "progress", "chk", None), ("Счётчик клавиш K1/K2/M1/M2", "keys", "chk", None), ("Положение счётчика", "keys_pos", "combo", ["right", "left"]),
                ("Размер счётчика", "keys_size", "sl", (30, 300, "%")), ("Комбо (приблизительно)", "combo", "chk", None), ("Точность (приблизительно)", "acc", "chk", None), ("Полоса точности (hit error)", "hiterr", "chk", None)])
    tab("Ещё", [("Потоков рендера (0 = авто)", "workers", "sl", (0, 12)), ("Пауза перед первым объектом, сек", "lead_in", "num", None), ("Пауза после последнего, сек", "tail", "num", None),
                ("Сжать готовый .mp4 (долго)", "compress", "chk", None), ("Путь к ffmpeg (не обязательно)", "ffmpeg", "file", None), ("Размер шрифта", "fontscale", "sl", (50, 300, "%")),
                ("Шрифт скина для комбо/точности", "skin_font", "chk", None), ("Шрифт интерфейса (пусто = стандартный)", "font_name", "txt", None)])

    # ---------------------------------------------------------------- запуск, прогресс, лог
    bb = ttk.Frame(root); bb.grid(row=3, column=0, sticky="we", padx=12, pady=(2, 4))
    btn = ttk.Button(bb, text="Экспортировать", style="Big.TButton", command=lambda: go()); btn.pack(side="left")
    stop_b = ttk.Button(bb, text="Остановить", state="disabled", command=lambda: stop()); stop_b.pack(side="left", padx=(10, 0))
    open_b = ttk.Button(bb, text="Показать файл", state="disabled", command=lambda: reveal()); open_b.pack(side="left", padx=(6, 0))
    ttk.Button(bb, text="Сбросить настройки", command=lambda: reset()).pack(side="right")

    lf = ttk.Frame(root); lf.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 12)); lf.columnconfigure(0, weight=1); lf.rowconfigure(2, weight=1)
    status = ttk.Label(lf, text="Готов к экспорту"); status.grid(row=0, column=0, sticky="w", pady=(0, 3))
    bar = ttk.Progressbar(lf, maximum=100); bar.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 6))
    box = tk.Text(lf, height=8, state="disabled", wrap="word", font=("Consolas", 9), relief="solid", borderwidth=1); box.grid(row=2, column=0, sticky="nsew")
    sb = ttk.Scrollbar(lf, command=box.yview); sb.grid(row=2, column=1, sticky="ns"); box.config(yscrollcommand=sb.set)

    def reset():
        if not messagebox.askyesno("Сброс", "Вернуть все настройки по умолчанию?"): return
        for k in DEFAULTS: o[k].set(DEFAULTS[k])
        for f in refreshers: f()

    def say(s): q.put(str(s))

    def upd(p):
        bar["value"] = p; el = time.time() - run["t0"]
        status.config(text=f"Рендер: {p:.0f}%" + (f"  ·  осталось ≈ {el * (100 - p) / p / 60:.1f} мин" if p > 1 else ""))

    def finish():
        run["on"] = False; btn.config(state="normal"); stop_b.config(state="disabled"); el = time.time() - run["t0"]
        if run["err"] is None:
            bar["value"] = 100; out = Path(P("out")); mb = f"{out.stat().st_size / 1048576:.1f} МБ" if out.is_file() else "?"
            status.config(text=f"Готово за {el / 60:.1f} мин  ·  размер файла {mb}"); open_b.config(state="normal")
        elif CANCEL.is_set(): status.config(text="Остановлено")
        else: status.config(text="Ошибка — подробности в логе ниже")

    def pump():  # один раз в 150 мс забираем весь накопившийся лог и прогресс — окно не тормозит
        lines = []
        try:
            while True: lines.append(q.get_nowait())
        except queue.Empty: pass
        if lines:
            box.config(state="normal"); box.insert("end", "\n".join(lines) + "\n")
            n = int(box.index("end-1c").split(".")[0])
            if n > 400: box.delete("1.0", f"{n - 400}.0")
            box.see("end"); box.config(state="disabled")
        if run["on"] and prog["p"] is not None: upd(prog["p"])
        if run["done"] and q.empty(): run["done"] = False; finish()
        root.after(150, pump)

    def work(op):
        try: render(P("osr"), P("map"), P("osk") or None, P("out"), op, say, lambda p: prog.__setitem__("p", p))
        except Exception as e: run["err"] = e; say("Остановлено пользователем" if CANCEL.is_set() else f"ОШИБКА: {e}")
        run["done"] = True

    def save():
        try: op = {k: o[k].get() for k in DEFAULTS}
        except tk.TclError: return None
        try: SETF.write_text(json.dumps({"opt": op, "osk": P("osk"), "map": P("map"), "out": P("out"), "dir": ldir["d"]}, ensure_ascii=False), encoding="utf-8")
        except Exception: pass
        return op

    def go():
        if not Path(P("osr")).is_file(): return messagebox.showerror("Ошибка", "Выбери файл реплея (.osr)")
        if not Path(P("map")).is_file(): return messagebox.showerror("Ошибка", "Выбери карту (.osz или .osu)")
        if P("osk") and not Path(P("osk")).is_file(): return messagebox.showerror("Ошибка", "Файл скина не найден")
        op = save()
        if op is None: return messagebox.showerror("Ошибка", "Проверь числовые поля в настройках")
        try: Path(P("out")).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e: return messagebox.showerror("Ошибка", f"Не могу создать папку для файла: {e}")
        CANCEL.clear(); run.update(on=True, err=None, done=False, t0=time.time()); prog["p"] = None
        bar["value"] = 0; status.config(text="Подготовка…"); btn.config(state="disabled"); stop_b.config(state="normal"); open_b.config(state="disabled")
        threading.Thread(target=work, args=(op,), daemon=True).start()

    def stop():
        CANCEL.set(); stop_b.config(state="disabled"); status.config(text="Останавливаю… (после текущего фрагмента)")

    def reveal():
        p = Path(P("out"))
        try:
            if sys.platform == "win32": subprocess.Popen(["explorer", f"/select,{p}"])
            elif sys.platform == "darwin": subprocess.Popen(["open", "-R", str(p)])
            else: subprocess.Popen(["xdg-open", str(p.parent)])
        except Exception: pass

    def on_close():
        try: save()
        except Exception: pass
        CANCEL.set(); root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close); on_osr(); pump(); root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    if len(sys.argv) > 1:
        a = argparse.ArgumentParser(); a.add_argument("osr"); a.add_argument("--map", required=True); a.add_argument("--skin"); a.add_argument("-o", default="osu_replay.mp4")
        a.add_argument("--set", action="append", default=[], help="настройка key=value, например --set music=50 --set trail=0"); x = a.parse_args(); op = {}
        for kv in x.set:
            k, val = kv.split("=", 1)
            if k not in DEFAULTS: sys.exit(f"Неизвестная настройка '{k}'. Доступные: " + ", ".join(DEFAULTS))
            d = DEFAULTS[k]
            op[k] = (val.lower() in ("1", "true", "yes", "on")) if isinstance(d, bool) else type(d)(val)
        render(x.osr, x.map, x.skin, x.o, op, print, lambda p: print(f"\r{p:5.1f}%", end="", flush=True))
    else: gui()