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
    total = sum(counts.values())
    with open(output_file, 'w', encoding='utf-8') as f:
        for gram, count in counts.items():
            prob = count / total
            f.write(f"{gram} {prob}\n")

def load_reference_ngrams(filename):
    """Wczytuje referencyjną bazę n-gramów (Gi, Pi) i normalizuje prawdopodobieństwa."""
    reference = {}
    total_prob = 0.0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            gram, prob = parts
            p = float(prob)
            reference[gram] = p
            total_prob += p
    if total_prob == 0:
        print("Uwaga: suma prawdopodobieństw w pliku referencyjnym wynosi 0!")
    else:
        # Normalizacja, żeby suma prawdopodobieństw wynosiła 1
        for gram in reference:
            reference[gram] /= total_prob
    print(f"Suma prawdopodobieństw w referencji po normalizacji: {sum(reference.values())}")
    return reference

def compute_chi_square(text, reference_probs, n):
    """Oblicza wartość testu chi-kwadrat z diagnostyką."""
    test_ngrams = generate_ngrams(text, n)
    test_counts = Counter(test_ngrams)
    total_test_ngrams = sum(test_counts.values())

    chi_square = 0.0
    for gram, expected_prob in reference_probs.items():
        observed = test_counts.get(gram, 0)
        expected = total_test_ngrams * expected_prob
        if expected > 0:
            chi_square += (observed - expected) ** 2 / expected

    print(f"Wartość testu chi-kwadrat: {chi_square}")
    return chi_square

def main():
    parser = argparse.ArgumentParser(description="Substitution cipher + n-gram analysis + chi-square test")
    parser.add_argument('-e', action='store_true', help='Encrypt mode')
    parser.add_argument('-d', action='store_true', help='Decrypt mode')
    parser.add_argument('-i', required=True, help='Input file')
    parser.add_argument('-o', help='Output file')
    parser.add_argument('-k', help='Key file')
    parser.add_argument('-g1', help='Generate monogram stats (output file)')
    parser.add_argument('-g2', help='Generate bigram stats (output file)')
    parser.add_argument('-g3', help='Generate trigram stats (output file)')
    parser.add_argument('-g4', help='Generate quadgram stats (output file)')
    parser.add_argument('-r1', help='Reference monogram file')
    parser.add_argument('-r2', help='Reference bigram file')
    parser.add_argument('-r3', help='Reference trigram file')
    parser.add_argument('-r4', help='Reference quadgram file')
    parser.add_argument('-s', action='store_true', help='Compute chi-square test')

    args = parser.parse_args()

    # Wczytaj tekst wejściowy
    with open(args.i, 'r', encoding='utf-8') as f:
        text = preprocess_text(f.read())
    print("Przetworzony tekst (pierwsze 100 znaków):", text[:100])

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

    # --- Tryb testu chi-kwadrat ---
    for n, ref_flag in enumerate([args.r1, args.r2, args.r3, args.r4], start=1):
        if ref_flag and args.s:
            reference = load_reference_ngrams(ref_flag)
            chi_val = compute_chi_square(text, reference, n)
            print(f"Wartość testu χ² dla {n}-gramów: {chi_val:.4f}")

if __name__ == "__main__":
    main()
