import random
import string
from typing import Dict

ALPHABET = list(string.ascii_uppercase)

def preprocess_letters_only(text: str) -> str:
    return ''.join(ch.upper() for ch in text if ch.isalpha())

def generate_substitution_key() -> Dict[str, str]:
    shuffled = ALPHABET.copy()
    random.shuffle(shuffled)
    return dict(zip(ALPHABET, shuffled))

def substitute_encrypt(plaintext: str, key: Dict[str, str]) -> str:
    return ''.join(key[ch] for ch in plaintext)

def display_key(key: Dict[str, str]) -> None:
    for k, v in key.items():
        print(f"{k} → {v}")

def main():
    text = """
    In cryptography, a substitution cipher is a method of encrypting by which units of plaintext
    are replaced with ciphertext according to a fixed system; the "units" may be single letters,
    pairs of letters, triplets of letters, mixtures of the above, and so forth.
    """

    clean_text = preprocess_letters_only(text)
    substitution_key = generate_substitution_key()
    ciphertext = substitute_encrypt(clean_text, substitution_key)

    print("Oryginalny tekst:")
    print(clean_text[:200] + "...\n")

    print("Klucz podstawieniowy:")
    display_key(substitution_key)

    print("\nZaszyfrowany tekst:")
    print(ciphertext[:200] + "...")

if __name__ == "__main__":
    main()
