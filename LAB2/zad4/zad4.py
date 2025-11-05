import argparse
import string
import sys
from typing import List, Tuple

ALPHABET = string.ascii_uppercase
MOD = 26

ENGLISH_FREQ = {
    'A': 0.08167, 'B': 0.01492, 'C': 0.02782, 'D': 0.04253, 'E': 0.12702,
    'F': 0.02228, 'G': 0.02015, 'H': 0.06094, 'I': 0.06966, 'J': 0.00153,
    'K': 0.00772, 'L': 0.04025, 'M': 0.02406, 'N': 0.06749, 'O': 0.07507,
    'P': 0.01929, 'Q': 0.00095, 'R': 0.05987, 'S': 0.06327, 'T': 0.09056,
    'U': 0.02758, 'V': 0.00978, 'W': 0.02360, 'X': 0.00150, 'Y': 0.01974,
    'Z': 0.00074
}
POLISH_FREQ = {
    'A': 0.089, 'B': 0.015, 'C': 0.036, 'D': 0.032, 'E': 0.076,
    'F': 0.009, 'G': 0.015, 'H': 0.011, 'I': 0.081, 'J': 0.024,
    'K': 0.034, 'L': 0.023, 'M': 0.032, 'N': 0.056, 'O': 0.077,
    'P': 0.033, 'Q': 0.0001,'R': 0.047, 'S': 0.034, 'T': 0.041,
    'U': 0.024, 'V': 0.0003,'W': 0.046, 'X': 0.0002,'Y': 0.037,
    'Z': 0.056
}

def load_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_file(path: str, data: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def preprocess_letters_only(text: str) -> str:
    return ''.join(ch for ch in text.upper() if ch in ALPHABET)

def egcd(a: int, b: int) -> Tuple[int,int,int]:
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"Brak odwrotności modulo {m} dla {a} (gcd={g})")
    return x % m

def affine_encrypt_preserve(text: str, a: int, b: int, preserve: bool = False) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            is_upper = ch.isupper()
            base = ord('A') if is_upper else ord('a')
            x = ord(ch.upper()) - 65
            y = (a * x + b) % MOD
            c = chr(y + 65)
            out.append(c if is_upper else c.lower())
        else:
            out.append(ch if preserve else ch)  # preserve param kept for clarity
    return ''.join(out)

def affine_decrypt_preserve(text: str, a: int, b: int, preserve: bool = False) -> str:
    a_inv = modinv(a, MOD)
    out = []
    for ch in text:
        if ch.isalpha():
            is_upper = ch.isupper()
            base = ord('A') if is_upper else ord('a')
            y = ord(ch.upper()) - 65
            x = (a_inv * (y - b)) % MOD
            c = chr(x + 65)
            out.append(c if is_upper else c.lower())
        else:
            out.append(ch if preserve else ch)
    return ''.join(out)

def valid_a_values() -> List[int]:
    return [a for a in range(1, MOD) if egcd(a, MOD)[0] == 1]

def chi_squared_statistic(text_letters_only: str, freq_table: dict) -> float:
    n = len(text_letters_only)
    if n == 0:
        return float('inf')
    counts = {ch:0 for ch in ALPHABET}
    for ch in text_letters_only:
        counts[ch] += 1
    chi2 = 0.0
    for ch in ALPHABET:
        observed = counts[ch]
        expected = freq_table.get(ch, 0.0) * n
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
    return chi2

def brute_force_affine_with_chi(text_raw: str, freq_table: dict, preserve: bool=False) -> List[Tuple[int,int,str,float]]:

    results = []
    for a in valid_a_values():
        for b in range(0, MOD):
            try:
                plain = affine_decrypt_preserve(text_raw, a, b, preserve=preserve)
                letters_only = preprocess_letters_only(plain)
                chi2 = chi_squared_statistic(letters_only, freq_table)
                results.append((a, b, plain, chi2))
            except Exception:
                continue
    return results

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Zadanie 4 — brute-force dla szyfru afinicznego z oceną chi-kwadrat")
    p.add_argument("-i", required=True, help="plik wejściowy (szyfrogram)")
    p.add_argument("-o", required=True, help="plik wyjściowy (kandydaci lub wynik)")
    p.add_argument("-ka", type=int, help="współczynnik a (używany tylko przy -e/-d)")
    p.add_argument("-kb", type=int, help="współczynnik b (używany tylko przy -e/-d)")
    p.add_argument("-e", help="szyfrowanie", action="store_true")
    p.add_argument("-d", help="deszyfrowanie", action="store_true")
    p.add_argument("-a", help="atak (bf = brute-force)", choices=['bf'])
    p.add_argument("--top", type=int, default=10, help="ile najlepszych kandydatów zapisać (domyślnie 10)")
    p.add_argument("--lang", choices=['en','pl'], default='en', help="rozkład liter: en (domyślnie) lub pl")
    p.add_argument("--preserve", action="store_true", help="zachowuj spacje i interpunkcję w odszyfrowanym tekście")
    return p.parse_args()

def main():
    args = parse_args()
    raw = load_file(args.i)

    freq_table = ENGLISH_FREQ if args.lang == 'en' else POLISH_FREQ

    if args.a == 'bf':
        all_results = brute_force_affine_with_chi(raw, freq_table, preserve=args.preserve)
        if not all_results:
            print("Brak wyników brute-force.")
            return
        all_results.sort(key=lambda x: x[3])
        top_n = max(1, args.top)
        selected = all_results[:top_n]

        out_lines = []
        for a,b,plain,chi2 in selected:
            out_lines.append(f"a={a} b={b} chi2={chi2:.3f}:\n{plain}\n\n")
        save_file(args.o, ''.join(out_lines))

        best = all_results[0]
        a_best,b_best,plain_best,chi2_best = best
        print(f"[OK] Brute-force zakończony. Zapisano top {len(selected)} kandydatów do: {args.o}")
        print(f"[BEST] a={a_best} b={b_best} chi2={chi2_best:.3f}")
        print(plain_best)
        return

    if not (args.e or args.d):
        raise ValueError("Musisz podać -e (szyfrowanie) lub -d (deszyfrowanie) albo -a bf")

    if args.ka is None or args.kb is None:
        raise ValueError("Musisz podać -ka i -kb dla szyfrowania/odszyfrowania")

    a = args.ka % MOD
    b = args.kb % MOD
    if egcd(a, MOD)[0] != 1:
        raise ValueError(f"Wartość a={a} nie ma odwrotności modulo {MOD}.")

    if args.e:
        result = affine_encrypt_preserve(raw, a, b, preserve=True)
    else:
        result = affine_decrypt_preserve(raw, a, b, preserve=True)

    save_file(args.o, result)
    print(f"[OK] Zapisano wynik do {args.o}")

if __name__ == "__main__":
    main()
