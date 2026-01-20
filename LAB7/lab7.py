import random
import time
import math
from collections import Counter

def hex_to_bits(hex_string: str) -> list[int]:
    bits = []
    for byte in bytes.fromhex(hex_string):
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bits_to_bytes: liczba bitów musi być wielokrotnością 8")
    out = []
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            b |= (bits[i + j] << j)
        out.append(b)
    return bytes(out)

def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

def bits_xor(a: list[int], b: list[int]) -> list[int]:
    return [x ^ y for x, y in zip(a, b)]

def is_printable_ascii(bs: bytes) -> bool:
    return all(32 <= b <= 126 for b in bs)

def chunk_bytes(bs: bytes, n: int):
    for i in range(0, len(bs), n):
        yield bs[i:i+n]

class Trivium:
    def __init__(self, key_bits: list[int], iv_bits: list[int], rounds: int = 1152):
        if len(key_bits) != 80:
            raise ValueError("Klucz musi mieć dokładnie 80 bitów")
        if len(iv_bits) != 80:
            raise ValueError("IV musi mieć dokładnie 80 bitów")

        self.state = [0] * 288

        self.state[0:80] = key_bits

        self.state[93:173] = iv_bits

        self.state[285] = 1
        self.state[286] = 1
        self.state[287] = 1

        for _ in range(rounds):
            self.clock()

    def clock(self) -> int:
        s = self.state

        t1 = s[65] ^ s[92]
        t2 = s[161] ^ s[176]
        t3 = s[242] ^ s[287]

        z = t1 ^ t2 ^ t3

        t1 ^= (s[90] & s[91]) ^ s[170]
        t2 ^= (s[174] & s[175]) ^ s[263]
        t3 ^= (s[285] & s[286]) ^ s[68]

        self.state = (
            [t3] + s[0:92] +
            [t1] + s[93:176] +
            [t2] + s[177:287]
        )
        return z

    def keystream(self, n_bits: int) -> list[int]:
        return [self.clock() for _ in range(n_bits)]


def experiment_1():
    print("\nEKSPERYMENT 1: WEKTOR TESTOWY eSTREAM")

    key_hex = "00000000000000000000"
    iv_hex  = "00000000000000000000"

    trivium = Trivium(hex_to_bits(key_hex), hex_to_bits(iv_hex), rounds=1152)
    stream = trivium.keystream(256)
    result = bits_to_bytes(stream).hex().upper()

    expected = "FBE0BF265859051B517A2E4E239FC97F563203161907CF2DE7A8790FA1B2E9CD"

    print("Wynik:     ", result)
    print("Oczekiwany:", expected)
    print("Poprawny:  ", result == expected)

    trivium2 = Trivium(hex_to_bits(key_hex), hex_to_bits(iv_hex), rounds=1152)
    stream2 = trivium2.keystream(256)
    result2 = bits_to_bytes(stream2).hex().upper()
    print("Powtarzalność:", result2 == result)


def crib_drag(D: bytes, crib: bytes) -> list[tuple[int, bytes]]:
    hits = []
    L = len(crib)
    for i in range(0, len(D) - L + 1):
        candidate = xor_bytes(D[i:i+L], crib)
        if is_printable_ascii(candidate):
            hits.append((i, candidate))
    return hits

def crib_drag_stats(D: bytes, crib: bytes):
    total_positions = len(D) - len(crib) + 1
    hits = crib_drag(D, crib)
    false_rate = (len(hits) / total_positions) if total_positions > 0 else 0.0
    return total_positions, len(hits), false_rate

def experiment_2():
    print("\n=== EKSPERYMENT 2: ATAK IV REUSE + CRIB DRAGGING ===")

    key_hex = "0123456789ABCDEF0123"
    iv_hex  = "FEDCBA9876543210FEDC"

    P1 = b"Tajna wiadomosc nr jeden!!"
    P2 = b"Inna sekretna informacja!!"

    trivium = Trivium(hex_to_bits(key_hex), hex_to_bits(iv_hex), rounds=1152)
    S = bits_to_bytes(trivium.keystream(len(P1) * 8))

    C1 = xor_bytes(P1, S)
    C2 = xor_bytes(P2, S)

    D = xor_bytes(C1, C2)

    print("C1 ⊕ C2:", D.hex())
    print("P1 ⊕ P2:", xor_bytes(P1, P2).hex())

    print("\n--- Crib dragging: fałszywe alarmy vs długość criba ---")

    crib_sets = {
        2:  [b"nr", b"!!", b"na"],
        4:  [b" wi", b"sek ", b"info"],
        8:  [b"wiadomos", b"sekretna"],
        16: [b"Tajna wiadomosc"]
    }

    header = f"{'Len':>4} | {'Crib':>20} | {'Pozycji':>9} | {'Trafień':>8} | {'False %':>8}"
    print(header)
    print("-" * len(header))

    for L in [2, 4, 8, 16]:
        for crib in crib_sets.get(L, []):
            total_pos, hits, rate = crib_drag_stats(D, crib)
            print(f"{L:4d} | {crib!r:>20} | {total_pos:9d} | {hits:8d} | {rate*100:7.3f}%")

def chi_square_monobit(bits: list[int]) -> float:
    n = len(bits)
    ones = sum(bits)
    zeros = n - ones
    expected = n / 2
    return ((zeros - expected) ** 2) / expected + ((ones - expected) ** 2) / expected

def experiment_3():
    print("\nEKSPERYMENT 3: WPŁYW LICZBY RUND (init, throughput, chi^2)")

    key = hex_to_bits("0123456789ABCDEF0123")
    iv  = hex_to_bits("FEDCBA9876543210FEDC")

    rounds_list = [192, 288, 384, 480, 576, 768, 1152]
    ks_bits = 200000

    header = f"{'Rundy':>6} | {'Jedynki':>7} | {'Balans':>7} | {'Init [ms]':>9} | {'Thr [Mbit/s]':>12} | {'chi^2 (10k)':>11}"
    print(header)
    print("-" * len(header))

    for rounds in rounds_list:
        t0 = time.perf_counter()
        trivium = Trivium(key, iv, rounds=rounds)
        t_init = (time.perf_counter() - t0) * 1000

        ones_state = sum(trivium.state)
        balance = ones_state / 288

        t1 = time.perf_counter()
        _ = trivium.keystream(ks_bits)
        t_gen = time.perf_counter() - t1
        thr_mbps = (ks_bits / t_gen) / 1e6 if t_gen > 0 else float("inf")

        triv2 = Trivium(key, iv, rounds=rounds)
        bits10k = triv2.keystream(10000)
        chi2 = chi_square_monobit(bits10k)

        print(f"{rounds:6d} | {ones_state:7d} | {balance:7.3f} | {t_init:9.2f} | {thr_mbps:12.2f} | {chi2:11.3f}")

def keystream_bit(key_bits: list[int], iv_bits: list[int], rounds: int) -> int:
    t = Trivium(key_bits, iv_bits, rounds=rounds)
    return t.keystream(1)[0]

def cube_sum_firstbit(key_bits: list[int], cube: tuple[int, ...], rounds: int, fixed_iv: list[int] | None = None) -> int:
    if fixed_iv is None:
        fixed_iv = [0]*80
    total = 0
    d = len(cube)
    for mask in range(1 << d):
        iv = fixed_iv[:]
        for j, idx in enumerate(cube):
            iv[idx] = (mask >> j) & 1
        total ^= keystream_bit(key_bits, iv, rounds)
    return total

def find_linear_cubes_offline(rounds: int, trials_per_size: int = 80, max_cube_size: int = 6, seed: int = 1234):
    rng = random.Random(seed)
    base_key = [0]*80

    good = []
    used = set()

    for d in range(1, max_cube_size+1):
        for _ in range(trials_per_size):
            cube = tuple(sorted(rng.sample(range(80), d)))
            if cube in used:
                continue
            used.add(cube)

            c0 = cube_sum_firstbit(base_key, cube, rounds)
            coeff = [0]*80
            for j in range(80):
                ej = [0]*80
                ej[j] = 1
                pj = cube_sum_firstbit(ej, cube, rounds)
                coeff[j] = pj ^ c0

            linear = True
            for _t in range(12):
                x = [rng.randint(0,1) for _ in range(80)]
                p = cube_sum_firstbit(x, cube, rounds)
                pred = c0
                for j in range(80):
                    if x[j] and coeff[j]:
                        pred ^= 1
                if p != pred:
                    linear = False
                    break

            if linear:
                coeff_set = {j for j,v in enumerate(coeff) if v==1}
                if len(coeff_set) == 0:
                    continue
                good.append((cube, c0, coeff_set))

    return good

def gauss_elim_gf2(A_rows: list[int], b: list[int], n_vars: int) -> list[int] | None:
    A = A_rows[:]
    b = b[:]
    m = len(A)

    where = [-1]*n_vars
    row = 0

    for col in range(n_vars):
        pivot = None
        for r in range(row, m):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        A[row], A[pivot] = A[pivot], A[row]
        b[row], b[pivot] = b[pivot], b[row]
        where[col] = row

        for r in range(m):
            if r != row and ((A[r] >> col) & 1):
                A[r] ^= A[row]
                b[r] ^= b[row]

        row += 1
        if row == m:
            break

    for r in range(m):
        if A[r] == 0 and b[r] == 1:
            return None

    x = [0]*n_vars
    for col in range(n_vars):
        if where[col] != -1:
            x[col] = b[where[col]]
    return x

def experiment_4():
    print("\n EKSPERYMENT 4: ATAK CUBE (offline + online + Gauss) ")

    rounds_list = [192, 288, 384, 480]
    trials_per_size = 80
    max_cube_size = 6

    for rounds in rounds_list:
        print(f"\n--- Rundy rozgrzewania: {rounds} ---")
        t0 = time.perf_counter()
        good = find_linear_cubes_offline(rounds, trials_per_size=trials_per_size, max_cube_size=max_cube_size, seed=1234)
        t_off = time.perf_counter() - t0

        per_size = Counter(len(cube) for (cube, _, _) in good)
        total_good = len(good)
        print(f"Faza offline: znaleziono {total_good} liniowych kostek w {t_off:.2f}s")
        for d in range(1, max_cube_size+1):
            print(f"  rozmiar {d}: {per_size[d]}")

        target_key = [random.randint(0,1) for _ in range(80)]
        fixed_iv = [0]*80

        A_rows = []
        b_vec = []

        for (cube, c0, coeff_set) in good:
            rhs = cube_sum_firstbit(target_key, cube, rounds, fixed_iv=fixed_iv) ^ c0
            mask = 0
            for j in coeff_set:
                mask |= (1 << j)
            A_rows.append(mask)
            b_vec.append(rhs)

        if len(A_rows) == 0:
            print("Brak równań – atak nie działa dla tej liczby rund (w tej konfiguracji).")
            continue

        t1 = time.perf_counter()
        sol = gauss_elim_gf2(A_rows, b_vec, n_vars=80)
        t_on = time.perf_counter() - t1

        if sol is None:
            print(f"Faza online: układ sprzeczny (t_on={t_on:.3f}s).")
            continue

        involved = set()
        for m in A_rows:
            for j in range(80):
                if (m >> j) & 1:
                    involved.add(j)

        recovered = 0
        correct = 0
        for j in sorted(involved):
            recovered += 1
            if sol[j] == target_key[j]:
                correct += 1

        acc = (correct / recovered) * 100 if recovered else 0.0
        print(f"Faza online: równania={len(A_rows)}, bity zaangażowane={recovered}, poprawne={correct} ({acc:.1f}%), czas={t_on:.3f}s")

        print("Przykładowe zależności (max 10):")
        for (cube, c0, coeff_set) in good[:10]:
            if len(coeff_set) == 1:
                j = next(iter(coeff_set))
                print(f"  cube={cube}  =>  p_I(x)=x[{j}] ⊕ {c0}")
            else:
                coeff_preview = ",".join(str(j) for j in sorted(coeff_set)[:6])
                more = "..." if len(coeff_set) > 6 else ""
                print(f"  cube={cube}  =>  p_I(x)=⊕x[{coeff_preview}{more}] ⊕ {c0}")

def frequency_test(bits: list[int]) -> float:
    return sum(bits) / len(bits)

def runs_test(bits: list[int]) -> float:
    n = len(bits)
    pi = frequency_test(bits)
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return 0.0

    V = 1
    for i in range(1, n):
        if bits[i] != bits[i-1]:
            V += 1

    num = abs(V - (2*n*pi*(1-pi)))
    den = 2 * math.sqrt(2*n) * pi * (1-pi)
    if den == 0:
        return 0.0

    return math.erfc(num / den)

def autocorrelation(bits: list[int], lag: int) -> float:
    n = len(bits)
    if lag <= 0 or lag >= n:
        raise ValueError("lag musi być w zakresie 1..n-1")

    x = [1 if b==1 else -1 for b in bits]
    a = x[:-lag]
    b = x[lag:]

    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    cov = sum((ai - ma)*(bi - mb) for ai, bi in zip(a, b)) / len(a)
    va = sum((ai - ma)**2 for ai in a) / len(a)
    vb = sum((bi - mb)**2 for bi in b) / len(b)
    if va == 0 or vb == 0:
        return 0.0
    return cov / math.sqrt(va * vb)

def experiment_5():
    print("\n EKSPERYMENT 5: TESTY STATYSTYCZNE (>=10^6 bitów) ")

    key = hex_to_bits("0123456789ABCDEF0123")
    iv  = hex_to_bits("FEDCBA9876543210FEDC")

    rounds_list = [0, 192, 288, 576, 1152]
    N = 1_000_000

    header = f"{'Rundy':>6} | {'Freq(1)':>8} | {'Runs p':>10} | {'AC lag1':>9} | {'AC lag8':>9}"
    print(header)
    print("-" * len(header))

    for rounds in rounds_list:
        triv = Trivium(key, iv, rounds=rounds)
        bits = triv.keystream(N)

        freq = frequency_test(bits)
        p_runs = runs_test(bits)
        ac1 = autocorrelation(bits, 1)
        ac8 = autocorrelation(bits, 8)

        print(f"{rounds:6d} | {freq:8.4f} | {p_runs:10.4f} | {ac1:9.4f} | {ac8:9.4f}")

if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
