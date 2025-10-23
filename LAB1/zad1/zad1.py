import argparse
import string

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

def main():
    parser = argparse.ArgumentParser(description="Substitution cipher")
    parser.add_argument('-e', action='store_true', help='Encrypt mode')
    parser.add_argument('-d', action='store_true', help='Decrypt mode')
    parser.add_argument('-i', required=True, help='Input file')
    parser.add_argument('-o', required=True, help='Output file')
    parser.add_argument('-k', required=True, help='Key file')

    args = parser.parse_args()

    key = load_key(args.k)
    if args.d:
        key = {v: k for k, v in key.items()}

    with open(args.i, 'r', encoding='utf-8') as f:
        text = f.read()

    clean_text = preprocess_text(text)
    result = substitute(clean_text, key)

    with open(args.o, 'w', encoding='utf-8') as f:
        f.write(result)

    mode = "Szyfrowanie" if args.e else "Deszyfrowanie"
    print(f"{mode} zakończone. Wynik zapisano w {args.o}")

if __name__ == "__main__":
    main()
