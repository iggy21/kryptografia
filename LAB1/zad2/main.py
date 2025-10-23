import argparse
import string
from collections import Counter

def load_key(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            a, b = line.strip().split()
            mapping[a.upper()] = b.upper()
    return mapping

def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)

def substitute(text, mapping):
    return ''.join(mapping.get(ch, ch) for ch in text)

def generate_ngrams(text, n):
    """Zwraca listę wszystkich n-gramów w tekście."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def save_ngram_stats(ngrams, output_file):
    """Zapisuje liczności n-gramów do pliku."""
    counts = Counter(ngrams)
    with open(output_file, 'w', encoding='utf-8') as f:
        for gram, count in counts.items():
            f.write(f"{gram} {count}\n")

def main():
    parser = argparse.ArgumentParser(description="Substitution cipher + n-gram analysis")
    parser.add_argument('-e', action='store_true', help='Encrypt mode')
    parser.add_argument('-d', action='store_true', help='Decrypt mode')
    parser.add_argument('-i', required=True, help='Input file')
    parser.add_argument('-o', help='Output file')
    parser.add_argument('-k', help='Key file')
    parser.add_argument('-g1', help='Generate monogram stats')
    parser.add_argument('-g2', help='Generate bigram stats')
    parser.add_argument('-g3', help='Generate trigram stats')
    parser.add_argument('-g4', help='Generate quadgram stats')

    args = parser.parse_args()

    # Wczytaj tekst
    with open(args.i, 'r', encoding='utf-8') as f:
        text = preprocess_text(f.read())

    # --- Tryb szyfrowania / deszyfrowania ---
    if args.e or args.d:
        if not args.k or not args.o:
            print("Musisz podać plik z kluczem (-k) i plik wyjściowy (-o).")
            return

        key = load_key(args.k)
        if args.d:
            key = {v: k for k, v in key.items()}

        result = substitute(text, key)
        with open(args.o, 'w', encoding='utf-8') as f:
            f.write(result)
        mode = "Szyfrowanie" if args.e else "Deszyfrowanie"
        print(f"{mode} zakończone. Wynik zapisano w {args.o}")

    # --- Tryb generowania n-gramów ---
    for n, flag in enumerate([args.g1, args.g2, args.g3, args.g4], start=1):
        if flag:
            ngrams = generate_ngrams(text, n)
            save_ngram_stats(ngrams, flag)
            print(f"Zapisano statystyki {n}-gramów do {flag}")

if __name__ == "__main__":
    main()
