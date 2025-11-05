import argparse
import string
import sys

ALPHABET = string.ascii_uppercase
MOD = 26

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in ALPHABET)

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("Brak odwrotności modulo dla podanego a")
    return x % m

def affine_encrypt(text, a, b):
    res = []
    for ch in text:
        x = ord(ch) - 65
        y = (a * x + b) % MOD
        res.append(chr(y + 65))
    return ''.join(res)

def affine_decrypt(text, a, b):
    a_inv = modinv(a, MOD)
    res = []
    for ch in text:
        y = ord(ch) - 65
        x = (a_inv * (y - b)) % MOD
        res.append(chr(x + 65))
    return ''.join(res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy", required=True)
    parser.add_argument("-o", help="plik wyjściowy", required=True)
    parser.add_argument("-ka", type=int, help="współczynnik a (musi mieć odwrotność mod 26)", required=True)
    parser.add_argument("-kb", type=int, help="współczynnik b (0-25)", required=True)
    parser.add_argument("-e", help="szyfrowanie", action="store_true")
    parser.add_argument("-d", help="deszyfrowanie", action="store_true")
    args = parser.parse_args()

    if not (args.e or args.d):
        raise ValueError("Musisz podać -e (szyfrowanie) lub -d (deszyfrowanie)")

    raw = load_file(args.i)
    text = preprocess_text(raw)

    a = args.ka % MOD
    b = args.kb % MOD

    if egcd(a, MOD)[0] != 1:
        raise ValueError(f"Wartość a={a} nie ma odwrotności modulo {MOD}. Dopuszczalne a to te względnie pierwsze z 26.")

    if args.e:
        result = affine_encrypt(text, a, b)
    else:
        result = affine_decrypt(text, a, b)

    save_file(args.o, result)

if __name__ == "__main__":
    main()
