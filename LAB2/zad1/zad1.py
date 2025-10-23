import argparse
import string

def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def save_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)

def caesar_cipher(text, key, encrypt=True):
    result = []
    for ch in text:
        shift = key if encrypt else -key
        result.append(chr((ord(ch) - 65 + shift) % 26 + 65))
    return ''.join(result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy", required=True)
    parser.add_argument("-o", help="plik wyjściowy", required=True)
    parser.add_argument("-k", type=int, help="klucz", required=True)
    parser.add_argument("-e", help="szyfrowanie", action="store_true")
    parser.add_argument("-d", help="deszyfrowanie", action="store_true")
    args = parser.parse_args()

    text = preprocess_text(load_file(args.i))
    if args.e:
        result = caesar_cipher(text, args.k, encrypt=True)
    elif args.d:
        result = caesar_cipher(text, args.k, encrypt=False)
    else:
        raise ValueError("Musisz podać -e lub -d")

    save_file(args.o, result)

if __name__ == "__main__":
    main()
