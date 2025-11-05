import argparse
import string
import sys
from typing import List, Tuple

ALPHABET = string.ascii_uppercase

def load_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_file(path: str, data: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def preprocess_text(text: str) -> str:
       return ''.join(ch for ch in text.upper() if ch in ALPHABET)

def caesar_cipher(text: str, key: int, encrypt: bool = True) -> str:
    result = []
    shift = key % 26
    for ch in text:
        if encrypt:
            idx = (ord(ch) - 65 + shift) % 26
        else:
            idx = (ord(ch) - 65 - shift) % 26
        result.append(chr(idx + 65))
    return ''.join(result)

def brute_force_caesar(text: str) -> List[Tuple[int, str]]:
    results = []
    for k in range(1, 26):
        plaintext = caesar_cipher(text, k, encrypt=False)
        results.append((k, plaintext))
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Szyfr Cezara + atak brute-force")
    parser.add_argument("-i", required=True, help="plik wejściowy (szyfrogram)")
    parser.add_argument("-o", required=True, help="plik wyjściowy")
    parser.add_argument("-k", type=int, help="klucz (0-25) — wymagany przy -e/-d")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-e", help="szyfrowanie", action="store_true")
    group.add_argument("-d", help="deszyfrowanie", action="store_true")
    parser.add_argument("-a", choices=['bf'], help="atak: bf = brute-force")
    return parser.parse_args()

def main():
    args = parse_args()

    raw_text = load_file(args.i)
    text = preprocess_text(raw_text)

    if args.a == 'bf':
        bf_results = brute_force_caesar(text)
        out_lines = []
        for k, plaintext in bf_results:
            out_lines.append(f"Key {k}:\n{plaintext}\n\n")
        save_file(args.o, ''.join(out_lines))
        print(f"[OK] Brute-force zakończony. Zapisano {len(bf_results)} kandydatów do: {args.o}")
        return

    if not (args.e or args.d):
        print("Błąd: musisz podać -e (szyfrowanie) lub -d (deszyfrowanie) albo -a bf.", file=sys.stderr)
        sys.exit(1)

    if args.k is None:
        print("Błąd: musisz podać -k (klucz) dla szyfrowania/odszyfrowania.", file=sys.stderr)
        sys.exit(1)

    key = args.k % 26

    if args.e:
        result = caesar_cipher(text, key, encrypt=True)
        save_file(args.o, result)
        print(f"[OK] Zaszyfrowano i zapisano wynik do: {args.o}")
    else:
        result = caesar_cipher(text, key, encrypt=False)
        save_file(args.o, result)
        print(f"[OK] Odszyfrowano (klucz={key}) i zapisano wynik do: {args.o}")

if __name__ == "__main__":
    main()
