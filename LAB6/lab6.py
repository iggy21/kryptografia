from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict
import argparse
import random
import time
import math
from collections import Counter

Bit = int
Bits = List[Bit]

def bytes_to_bits(data: bytes) -> Bits:
    out: Bits = []
    for b in data:
        for i in range(7, -1, -1):
            out.append((b >> i) & 1)
    return out

def bits_to_bytes(bits: Bits) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bits_to_bytes: długość bitów musi być wielokrotnością 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | (bits[i + j] & 1)
        out.append(v)
    return bytes(out)

def xor_bits(a: Bits, b: Bits) -> Bits:
    if len(a) != len(b):
        raise ValueError("xor_bits: długości muszą być równe")
    return [(x ^ y) for x, y in zip(a, b)]

def fmt_bits(bits: Bits, limit: int = 64) -> str:
    s = "".join(map(str, bits))
    if len(s) <= limit:
        return s
    return s[:limit] + f"... ({len(s)} bits)"

def bits_to_hex(bits: Bits) -> str:
    padded = list(bits)
    if len(padded) % 8 != 0:
        padded += [0] * (8 - (len(padded) % 8))
    return bits_to_bytes(padded).hex().upper()

def text_to_bits_utf8(s: str) -> Bits:
    return bytes_to_bits(s.encode("utf-8"))

def bits_to_text_utf8(bits: Bits) -> str:
    return bits_to_bytes(bits).decode("utf-8", errors="replace")


@dataclass
class LFSR:
    taps: Bits
    state: Bits

    def __post_init__(self) -> None:
        m = len(self.taps)
        if len(self.state) != m:
            raise ValueError("LFSR: stan musi mieć długość równą stopniowi rejestru")
        if any(b not in (0, 1) for b in self.state):
            raise ValueError("LFSR: stan musi być bitami 0/1")
        if any(t not in (0, 1) for t in self.taps):
            raise ValueError("LFSR: taps muszą być bitami 0/1")

    def reset(self, state: Bits) -> None:
        if len(state) != len(self.state):
            raise ValueError("LFSR.reset: zły rozmiar stanu")
        self.state = list(state)

    def next_bit(self) -> Bit:
        out = self.state[0]
        fb = 0
        for j, tj in enumerate(self.taps):
            if tj:
                fb ^= self.state[j]
        self.state = self.state[1:] + [fb]
        return out

    def generate(self, n: int) -> Bits:
        return [self.next_bit() for _ in range(n)]

@dataclass
class CompositeGenerator:
    seed_x: Bits
    seed_y: Bits
    seed_z: Bits

    TAPS_X = [1, 1, 0]
    TAPS_Y = [1, 0, 0, 1]
    TAPS_Z = [1, 0, 1, 0, 0]

    def __post_init__(self) -> None:
        self.X = LFSR(list(self.TAPS_X), list(self.seed_x))
        self.Y = LFSR(list(self.TAPS_Y), list(self.seed_y))
        self.Z = LFSR(list(self.TAPS_Z), list(self.seed_z))

    def reset(self, seed_x: Bits, seed_y: Bits, seed_z: Bits) -> None:
        self.seed_x = list(seed_x)
        self.seed_y = list(seed_y)
        self.seed_z = list(seed_z)
        self.X.reset(self.seed_x)
        self.Y.reset(self.seed_y)
        self.Z.reset(self.seed_z)

    @staticmethod
    def combine(x: Bit, y: Bit, z: Bit) -> Bit:
        return (x & y) ^ (y & z) ^ z

    def next_bit(self) -> Bit:
        return self.combine(self.X.next_bit(), self.Y.next_bit(), self.Z.next_bit())

    def generate(self, n: int) -> Bits:
        return [self.next_bit() for _ in range(n)]

class StreamCipher:
    def __init__(self, gen: CompositeGenerator):
        self.gen = gen

    def reset(self, sx: Bits, sy: Bits, sz: Bits) -> None:
        self.gen.reset(sx, sy, sz)

    def encrypt_bytes(self, data: bytes) -> bytes:
        bits = bytes_to_bits(data)
        k = self.gen.generate(len(bits))
        c_bits = xor_bits(bits, k)
        return bits_to_bytes(c_bits)

    def decrypt_bytes(self, data: bytes) -> bytes:
        return self.encrypt_bytes(data)


def all_nonzero_seeds(m: int) -> Iterable[Bits]:
    for v in range(1, 1 << m):
        yield [(v >> i) & 1 for i in range(m - 1, -1, -1)]

def random_nonzero_seed(m: int) -> Bits:
    v = random.randint(1, (1 << m) - 1)
    return [(v >> i) & 1 for i in range(m - 1, -1, -1)]


def pearson_correlation(x: Bits, y: Bits) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    xm = sum(x) / n
    ym = sum(y) / n
    num = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    denx = sum((xi - xm) ** 2 for xi in x)
    deny = sum((yi - ym) ** 2 for yi in y)
    if denx <= 0 or deny <= 0:
        return 0.0
    return num / math.sqrt(denx * deny)

def agreement_score(x: Bits, y: Bits) -> float:
    if not x:
        return 0.0
    eq = sum(1 for a, b in zip(x, y) if a == b)
    return eq / len(x)

def correlation_attack_lfsr(
    K: Bits,
    m: int,
    taps: Bits,
    selector: str = "pearson",
    return_ranking: bool = False
) -> Tuple[Bits, float, Optional[List[Tuple[Bits, float]]]]:
    best_seed: Optional[Bits] = None
    best_score = -1e18
    ranking: List[Tuple[Bits, float]] = []

    for seed in all_nonzero_seeds(m):
        R = LFSR(list(taps), list(seed))
        out = R.generate(len(K))
        if selector == "pearson":
            sc = pearson_correlation(K, out)
        elif selector == "agree":
            sc = agreement_score(K, out)
        else:
            raise ValueError("selector must be 'pearson' or 'agree'")
        ranking.append((seed, sc))
        if sc > best_score:
            best_score = sc
            best_seed = seed

    assert best_seed is not None
    if return_ranking:
        ranking.sort(key=lambda t: t[1], reverse=True)
        return best_seed, best_score, ranking
    return best_seed, best_score, None

def recover_y_by_bruteforce(
    K: Bits,
    seed_x: Bits,
    seed_z: Bits
) -> Optional[Bits]:
    for sy in all_nonzero_seeds(4):
        gen = CompositeGenerator(seed_x, sy, seed_z)
        if gen.generate(len(K)) == K:
            return sy
    return None

def attack_correlation_then_y(
    K: Bits,
    selector: str = "pearson",
    verbose: bool = True,
    top: int = 5
) -> Tuple[Bits, Bits, Bits, Dict]:
    diag: Dict = {}

    sx, rho_x, rank_x = correlation_attack_lfsr(
        K, 3, CompositeGenerator.TAPS_X, selector=selector, return_ranking=True
    )
    sz, rho_z, rank_z = correlation_attack_lfsr(
        K, 5, CompositeGenerator.TAPS_Z, selector=selector, return_ranking=True
    )

    diag["rho_x"] = rho_x
    diag["rho_z"] = rho_z
    diag["rank_x"] = rank_x
    diag["rank_z"] = rank_z

    sy = recover_y_by_bruteforce(K, sx, sz)
    if sy is None:
        if verbose:
            print("Nie znaleziono Y dla najlepszych X,Z — próbuję kombinacje TOP kandydatów...")
        for sx2, _ in rank_x[:top]:
            for sz2, _ in rank_z[:top]:
                sy2 = recover_y_by_bruteforce(K, sx2, sz2)
                if sy2 is not None:
                    sx, sz, sy = sx2, sz2, sy2
                    diag["used_top_search"] = True
                    diag["top_limit"] = top
                    return sx, sy, sz, diag
        raise RuntimeError("Atak korelacyjny nie odzyskał seedów (nawet z TOP).")

    return sx, sy, sz, diag

def brute_force_all(
    K: Bits
) -> Tuple[Bits, Bits, Bits, int]:
    tries = 0
    for sx in all_nonzero_seeds(3):
        for sy in all_nonzero_seeds(4):
            for sz in all_nonzero_seeds(5):
                tries += 1
                gen = CompositeGenerator(sx, sy, sz)
                if gen.generate(len(K)) == K:
                    return sx, sy, sz, tries
    raise RuntimeError("Brute-force: nie znaleziono seedów (to nie powinno się zdarzyć).")

SEED_X_EX = [1, 0, 1]
SEED_Y_EX = [1, 0, 1, 0]
SEED_Z_EX = [1, 1, 0, 0, 0]

X_31_EX = [1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1]
Y_31_EX = [1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1]
Z_31_EX = [1,1,0,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,1,1,0,1,0,0,0]
K_31_EX = [1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,1,1]

PT_LAB = b"Lab"
CT_LAB_HEX = "BE04E9"

def truth_table_and_bias() -> None:
    rows = []
    cnt = Counter()
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                f = CompositeGenerator.combine(x, y, z)
                rows.append((x, y, z, f))
                cnt["f1"] += f
                cnt["fx"] += 1 if f == x else 0
                cnt["fy"] += 1 if f == y else 0
                cnt["fz"] += 1 if f == z else 0
    print("Tabela prawdy f(x,y,z)=xy ⊕ yz ⊕ z:")
    print(" x y z | f")
    print("-------+---")
    for x, y, z, f in rows:
        print(f" {x} {y} {z} | {f}")
    print()
    print("Prawdopodobieństwa (na 8 wejść):")
    print(f"  P(f=1) = {cnt['f1']}/8 = {cnt['f1']/8:.3f}")
    print(f"  P(f=x) = {cnt['fx']}/8 = {cnt['fx']/8:.3f}")
    print(f"  P(f=y) = {cnt['fy']}/8 = {cnt['fy']/8:.3f}")
    print(f"  P(f=z) = {cnt['fz']}/8 = {cnt['fz']/8:.3f}")
    print("Wnioski: bias wobec X i Z (3/4), brak biasu wobec Y (1/2) — podstawa ataku korelacyjnego.\n")

def verify_known_example() -> None:
    print("Tabela 31 bitów (X,Y,Z,K) ")
    gen = CompositeGenerator(SEED_X_EX, SEED_Y_EX, SEED_Z_EX)

    X = LFSR(CompositeGenerator.TAPS_X, list(SEED_X_EX)).generate(31)
    Y = LFSR(CompositeGenerator.TAPS_Y, list(SEED_Y_EX)).generate(31)
    Z = LFSR(CompositeGenerator.TAPS_Z, list(SEED_Z_EX)).generate(31)
    K = gen.generate(31)

    ok_x = (X == X_31_EX)
    ok_y = (Y == Y_31_EX)
    ok_z = (Z == Z_31_EX)
    ok_k = (K == K_31_EX)

    def show(name: str, got: Bits, exp: Bits, ok: bool) -> None:
        print(f"{name}: {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"  got: {fmt_bits(got, 64)}")
            print(f"  exp: {fmt_bits(exp, 64)}")

    show("X[31]", X, X_31_EX, ok_x)
    show("Y[31]", Y, Y_31_EX, ok_y)
    show("Z[31]", Z, Z_31_EX, ok_z)
    show("K[31]", K, K_31_EX, ok_k)

    if ok_x and ok_y and ok_z and ok_k:
        print("Tabela 31 bitów: WSZYSTKO OK.\n")
    else:
        raise SystemExit("Błąd: nie zgadza się tabela 31 bitów.")

def verify_encrypt_example() -> None:
    print("Szyfrowanie przykładu 'Lab' ")
    gen = CompositeGenerator(SEED_X_EX, SEED_Y_EX, SEED_Z_EX)
    cipher = StreamCipher(gen)
    ct = cipher.encrypt_bytes(PT_LAB)
    hx = ct.hex().upper()
    print(f"PT: {PT_LAB!r}")
    print(f"CT: {hx} (expected {CT_LAB_HEX})")
    if hx != CT_LAB_HEX:
        raise SystemExit("Błąd: szyfrogram nie zgadza się z instrukcją.")
    print("Szyfrowanie 'Lab': OK.\n")

def file_encrypt_decrypt(in_path: str, out_path: str, sx: Bits, sy: Bits, sz: Bits, decrypt: bool) -> None:
    with open(in_path, "rb") as f:
        data = f.read()
    gen = CompositeGenerator(sx, sy, sz)
    cipher = StreamCipher(gen)
    out = cipher.decrypt_bytes(data) if decrypt else cipher.encrypt_bytes(data)
    with open(out_path, "wb") as f:
        f.write(out)

def demo_attack(known_bits: int, plaintext: str) -> None:
    print("Szyfrowanie + odzyskanie K + atak korelacyjny")
    sx = random_nonzero_seed(3)
    sy = random_nonzero_seed(4)
    sz = random_nonzero_seed(5)
    print(f"Losowe seedy (tajne): X={sx} Y={sy} Z={sz}")

    pt_bits = text_to_bits_utf8(plaintext)
    gen_enc = CompositeGenerator(sx, sy, sz)
    ct_bits = xor_bits(pt_bits, gen_enc.generate(len(pt_bits)))

    kb = min(known_bits, len(pt_bits))
    K_known = xor_bits(pt_bits[:kb], ct_bits[:kb])
    print(f"Znane bity: {kb}")
    print(f"Odzyskany fragment K: {fmt_bits(K_known, 96)}")

    t0 = time.perf_counter()
    rx, ry, rz, diag = attack_correlation_then_y(K_known, selector="pearson", verbose=True)
    t1 = time.perf_counter()

    print("\nWyniki ataku korelacyjnego")
    print(f"Odzyskane seedy: X={rx} Y={ry} Z={rz}")
    print(f"Czas ataku: {(t1-t0)*1000:.3f} ms")
    print(f"Pearson: rho_x={diag['rho_x']:.4f}, rho_z={diag['rho_z']:.4f}")
    if diag.get("used_top_search"):
        print(f"Użyto TOP-search: limit={diag.get('top_limit')}")

    gen_dec = CompositeGenerator(rx, ry, rz)
    pt2_bits = xor_bits(ct_bits, gen_dec.generate(len(ct_bits)))
    pt2 = bits_to_text_utf8(pt2_bits)
    ok = (pt2_bits == pt_bits)
    print("\n Weryfikacja deszyfracji ")
    print(f"Odszyfrowany tekst: {pt2!r}")
    print(f"Zgodność z oryginałem: {'OK' if ok else 'FAIL'}")
    if not ok:
        print("Uwaga: jeśli znany fragment jest bardzo krótki, atak może się mylić statystycznie.")
    print()

def print_top_ranking(ranking: List[Tuple[Bits, float]], name: str, top: int = 5) -> None:
    print(f"TOP {top} kandydatów dla {name}:")
    for i, (seed, sc) in enumerate(ranking[:top], 1):
        print(f"  {i:2d}. seed={seed} score={sc:.6f}")
    print()

def ascii_hist(values: List[float], bins: int = 12, width: int = 40) -> str:
    if not values:
        return "(brak danych)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"(wszystkie wartości = {lo:.6f})"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / step))
        counts[idx] += 1
    m = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        a = lo + i * step
        b = a + step
        bar = "#" * int(round((c / m) * width))
        lines.append(f"{a: .3f}..{b: .3f} | {bar} ({c})")
    return "\n".join(lines)

def run_trials(
    trials: int,
    lengths: List[int],
    bytes_len: int,
    selector: str = "pearson"
) -> None:

    rnd = random.Random(12345)

    print("EKSPERYMENTY ")
    print(f"Próby: {trials}, długość wiadomości: {bytes_len} B, selektor: {selector}")
    print(f"Długości znanego fragmentu (bity): {lengths}\n")

    stats = {L: {"ok_corr": 0, "t_corr": 0.0, "ok_bf": 0, "t_bf": 0.0, "tries_bf": 0} for L in lengths}

    example_hist_done = {L: False for L in lengths}

    for t in range(1, trials + 1):
        sx = random_nonzero_seed(3)
        sy = random_nonzero_seed(4)
        sz = random_nonzero_seed(5)

        msg = bytes(rnd.getrandbits(8) for _ in range(bytes_len))
        pt_bits = bytes_to_bits(msg)
        gen = CompositeGenerator(sx, sy, sz)
        ct_bits = xor_bits(pt_bits, gen.generate(len(pt_bits)))

        for L in lengths:
            kb = min(L, len(pt_bits))
            K = xor_bits(pt_bits[:kb], ct_bits[:kb])

            t0 = time.perf_counter()
            ok_corr = 0
            try:
                rx, ry, rz, diag = attack_correlation_then_y(K, selector=selector, verbose=False)
                ok_corr = int((rx == sx) and (ry == sy) and (rz == sz))
            except Exception:
                ok_corr = 0
            t1 = time.perf_counter()

            stats[L]["ok_corr"] += ok_corr
            stats[L]["t_corr"] += (t1 - t0)

            tb0 = time.perf_counter()
            bx, by, bz, tries = brute_force_all(K)
            tb1 = time.perf_counter()
            ok_bf = int((bx == sx) and (by == sy) and (bz == sz))
            stats[L]["ok_bf"] += ok_bf
            stats[L]["t_bf"] += (tb1 - tb0)
            stats[L]["tries_bf"] += tries

            if not example_hist_done[L]:
                _, _, rank_x = correlation_attack_lfsr(K, 3, CompositeGenerator.TAPS_X, selector=selector, return_ranking=True)
                _, _, rank_z = correlation_attack_lfsr(K, 5, CompositeGenerator.TAPS_Z, selector=selector, return_ranking=True)
                assert rank_x is not None and rank_z is not None
                vals_x = [sc for _, sc in rank_x]
                vals_z = [sc for _, sc in rank_z]
                print(f"--- Histogram score ({selector}) dla L={L} (przykładowa próba) ---")
                print("X (7 seedów):")
                print(ascii_hist(vals_x))
                print("\nZ (31 seedów):")
                print(ascii_hist(vals_z))
                print()
                print_top_ranking(rank_x, "X", top=5)
                print_top_ranking(rank_z, "Z", top=5)
                example_hist_done[L] = True

        if t % max(1, trials // 5) == 0:
            print(f"[{t}/{trials}] ...")

    print("\nPODSUMOWANIE")
    header = f"{'L(bits)':>7} | {'corr_ok':>7} | {'corr_succ%':>10} | {'corr_avg_ms':>11} | {'bf_avg_ms':>9} | {'bf_avg_tries':>12}"
    print(header)
    print("-" * len(header))
    for L in lengths:
        corr_ok = stats[L]["ok_corr"]
        corr_s = 100.0 * corr_ok / trials
        corr_avg_ms = (stats[L]["t_corr"] / trials) * 1000.0
        bf_avg_ms = (stats[L]["t_bf"] / trials) * 1000.0
        bf_avg_tries = stats[L]["tries_bf"] / trials
        print(f"{L:7d} | {corr_ok:7d} | {corr_s:10.2f} | {corr_avg_ms:11.3f} | {bf_avg_ms:9.3f} | {bf_avg_tries:12.1f}")
    print()

def compare_selectors(trials: int, known_bits: int) -> None:

    rnd = random.Random(999)

    def one_trial() -> Tuple[Bits, Bits, Bits, Bits]:
        sx = random_nonzero_seed(3)
        sy = random_nonzero_seed(4)
        sz = random_nonzero_seed(5)
        msg = bytes(rnd.getrandbits(8) for _ in range(32))
        pt_bits = bytes_to_bits(msg)
        gen = CompositeGenerator(sx, sy, sz)
        ct_bits = xor_bits(pt_bits, gen.generate(len(pt_bits)))
        kb = min(known_bits, len(pt_bits))
        K = xor_bits(pt_bits[:kb], ct_bits[:kb])
        return sx, sy, sz, K

    ok = {"pearson": {"x": 0, "z": 0}, "agree": {"x": 0, "z": 0}}

    for _ in range(trials):
        sx, _, sz, K = one_trial()
        rx, _, _ = correlation_attack_lfsr(K, 3, CompositeGenerator.TAPS_X, selector="pearson", return_ranking=False)
        rz, _, _ = correlation_attack_lfsr(K, 5, CompositeGenerator.TAPS_Z, selector="pearson", return_ranking=False)
        ok["pearson"]["x"] += int(rx == sx)
        ok["pearson"]["z"] += int(rz == sz)

        rx, _, _ = correlation_attack_lfsr(K, 3, CompositeGenerator.TAPS_X, selector="agree", return_ranking=False)
        rz, _, _ = correlation_attack_lfsr(K, 5, CompositeGenerator.TAPS_Z, selector="agree", return_ranking=False)
        ok["agree"]["x"] += int(rx == sx)
        ok["agree"]["z"] += int(rz == sz)

    print("PORÓWNANIE SELEKTORÓW ")
    for sel in ("pearson", "agree"):
        sxp = 100.0 * ok[sel]["x"] / trials
        szp = 100.0 * ok[sel]["z"] / trials
        print(f"{sel:8s}: X ok {ok[sel]['x']}/{trials} ({sxp:.2f}%), Z ok {ok[sel]['z']}/{trials} ({szp:.2f}%)")
    print()


def parse_seed(s: str, m: int) -> Bits:
    s = s.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        bits = [int(p) for p in parts]
    else:
        bits = [int(ch) for ch in s]
    if len(bits) != m or any(b not in (0, 1) for b in bits):
        raise argparse.ArgumentTypeError(f"Seed musi mieć długość {m} i zawierać tylko 0/1")
    if all(b == 0 for b in bits):
        raise argparse.ArgumentTypeError("Seed nie może być zerowy")
    return bits

def cmd_verify(_args: argparse.Namespace) -> None:
    truth_table_and_bias()
    verify_known_example()
    verify_encrypt_example()
    print("VERIFY: zakończono sukcesem.")

def cmd_demo(args: argparse.Namespace) -> None:
    demo_attack(known_bits=args.known_bits, plaintext=args.plaintext)

def cmd_bench(args: argparse.Namespace) -> None:
    if args.lengths:
        lengths = [int(x) for x in args.lengths.split(",")]
    else:
        lengths = [8, 16, 24, 31, 62, 93]
    run_trials(trials=args.trials, lengths=lengths, bytes_len=args.bytes, selector=args.selector)
    if args.compare_selectors:
        compare_selectors(trials=max(20, args.trials), known_bits=min(31, args.bytes * 8))

def cmd_encrypt(args: argparse.Namespace) -> None:
    sx = parse_seed(args.seed_x, 3)
    sy = parse_seed(args.seed_y, 4)
    sz = parse_seed(args.seed_z, 5)
    file_encrypt_decrypt(args.input, args.output, sx, sy, sz, decrypt=False)
    print(f"Zaszyfrowano: {args.input} -> {args.output}")

def cmd_decrypt(args: argparse.Namespace) -> None:
    sx = parse_seed(args.seed_x, 3)
    sy = parse_seed(args.seed_y, 4)
    sz = parse_seed(args.seed_z, 5)
    file_encrypt_decrypt(args.input, args.output, sx, sy, sz, decrypt=True)
    print(f"Odszyfrowano: {args.input} -> {args.output}")

def cmd_attack(args: argparse.Namespace) -> None:
    with open(args.plaintext, "rb") as f:
        pt = f.read()
    with open(args.ciphertext, "rb") as f:
        ct = f.read()
    pt_bits = bytes_to_bits(pt)
    ct_bits = bytes_to_bits(ct)
    kb = min(args.known_bits, len(pt_bits), len(ct_bits))
    K = xor_bits(pt_bits[:kb], ct_bits[:kb])

    print(f"Znany fragment: {kb} bitów (z plików)")
    t0 = time.perf_counter()
    rx, ry, rz, diag = attack_correlation_then_y(K, selector=args.selector, verbose=True)
    t1 = time.perf_counter()

    print("\n Odzyskane seedy ")
    print(f"X={rx} Y={ry} Z={rz}")
    print(f"Czas: {(t1-t0)*1000:.3f} ms")
    print(f"rho_x={diag['rho_x']:.4f}, rho_z={diag['rho_z']:.4f}")

    gen = CompositeGenerator(rx, ry, rz)
    cipher = StreamCipher(gen)
    dec = cipher.decrypt_bytes(ct)
    with open(args.output, "wb") as f:
        f.write(dec)
    print(f"Odszyfrowany plik zapisano do: {args.output}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lab6: LFSR composite stream cipher + correlation attack")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("verify", help="Weryfikacja tabeli 31 bitów i szyfrowania 'Lab'")
    pv.set_defaults(func=cmd_verify)

    pd = sub.add_parser("demo", help="Demonstracja: losowe seedy, szyfrowanie, odzyskanie K i atak korelacyjny")
    pd.add_argument("--known-bits", type=int, default=93, help="Ile bitów znanego tekstu użyć (prefix)")
    pd.add_argument("--plaintext", type=str, default="To jest przykladowa wiadomosc do demonstracji ataku korelacyjnego.",
                    help="Tekst jawny do demonstracji (UTF-8)")
    pd.set_defaults(func=cmd_demo)

    pb = sub.add_parser("bench", help="Benchmark/eksperymenty: skuteczność i czasy vs długość znanego fragmentu")
    pb.add_argument("--trials", type=int, default=20, help="Liczba prób")
    pb.add_argument("--bytes", type=int, default=64, help="Długość losowej wiadomości (bajty)")
    pb.add_argument("--lengths", type=str, default="", help="Długości znanego fragmentu bitów, np. 8,16,24,31,62,93")
    pb.add_argument("--selector", type=str, default="pearson", choices=["pearson", "agree"],
                    help="Selekcja w ataku korelacyjnym: pearson lub agree")
    pb.add_argument("--compare-selectors", action="store_true",
                    help="Dodatkowo porównaj pearson vs agree na odzysk X i Z")
    pb.set_defaults(func=cmd_bench)

    pe = sub.add_parser("encrypt", help="Szyfruj plik: XOR z generowanym strumieniem klucza")
    pe.add_argument("-i", "--input", required=True, help="Plik wejściowy")
    pe.add_argument("-o", "--output", required=True, help="Plik wyjściowy")
    pe.add_argument("--seed-x", required=True, help="Seed X (3 bity), np. 101 lub 1,0,1")
    pe.add_argument("--seed-y", required=True, help="Seed Y (4 bity)")
    pe.add_argument("--seed-z", required=True, help="Seed Z (5 bitów)")
    pe.set_defaults(func=cmd_encrypt)

    pdc = sub.add_parser("decrypt", help="Odszyfruj plik (symetrycznie)")
    pdc.add_argument("-i", "--input", required=True, help="Plik wejściowy")
    pdc.add_argument("-o", "--output", required=True, help="Plik wyjściowy")
    pdc.add_argument("--seed-x", required=True, help="Seed X (3 bity)")
    pdc.add_argument("--seed-y", required=True, help="Seed Y (4 bity)")
    pdc.add_argument("--seed-z", required=True, help="Seed Z (5 bitów)")
    pdc.set_defaults(func=cmd_decrypt)

    pa = sub.add_parser("attack", help="Atak z known-plaintext na plikach: odzysk seedów i deszyfracja")
    pa.add_argument("--plaintext", required=True, help="Plik z tekstem jawnym (znany fragment prefixu)")
    pa.add_argument("--ciphertext", required=True, help="Plik z szyfrogramem")
    pa.add_argument("--known-bits", type=int, default=93, help="Ile bitów prefixu plaintextu jest znane")
    pa.add_argument("-o", "--output", required=True, help="Gdzie zapisać odszyfrowany plik")
    pa.add_argument("--selector", type=str, default="pearson", choices=["pearson", "agree"],
                    help="Selekcja w ataku korelacyjnym: pearson lub agree")
    pa.set_defaults(func=cmd_attack)

    return p

def main(argv: Optional[List[str]] = None) -> None:
    ap = build_argparser()
    args = ap.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
