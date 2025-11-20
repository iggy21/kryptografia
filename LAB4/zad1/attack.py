from typing import List, Optional, Tuple
from math import gcd

from cipher import to_bits
from math_utils import mod_inv


def bits_to_int(bits: List[int]) -> int:
    """Konwersja listy bitów na liczbę całkowitą."""
    if not bits:
        return 0
    return int(''.join(str(b) for b in bits), 2)


def recover_stream(known_plaintext: str, ciphertext_bits: List[int]) -> List[int]:
    """
    Odzyskuje strumień klucza Σ dla znanego fragmentu tekstu jawnego Xk
    i szyfrogramu Y (lista bitów).
    """
    xi_k = to_bits(known_plaintext)
    if len(xi_k) > len(ciphertext_bits):
        raise ValueError("Szyfrogram jest krótszy niż znany tekst (w bitach).")

    sigma_stream: List[int] = []
    for i in range(len(xi_k)):
        s_i = xi_k[i] ^ ciphertext_bits[i]
        sigma_stream.append(s_i)
    return sigma_stream


def attack(known_plaintext: str, ciphertext_bits: List[int],
           m: int, n: int) -> Optional[Tuple[int, int]]:
    """
    Atak known-plaintext na kryptosystem strumieniowy z LCG
    zgodnie z Algorytmem 7.
    Zwraca (A, B) lub None w przypadku niepowodzenia.
    """
    xi_k = to_bits(known_plaintext)
    if len(xi_k) > len(ciphertext_bits):
        print("[ATAK] Błąd: szyfrogram krótszy niż znany tekst.")
        return None

    sigma_stream: List[int] = []
    for i in range(len(xi_k)):
        s_i = xi_k[i] ^ ciphertext_bits[i]
        sigma_stream.append(s_i)

    if len(sigma_stream) < 3 * n:
        print("[ATAK] Niewystarczająca długość znanego fragmentu (mniej niż 3n bitów).")
        return None

    S1_bits = sigma_stream[0:n]
    S2_bits = sigma_stream[n:2*n]
    S3_bits = sigma_stream[2*n:3*n]

    S1 = bits_to_int(S1_bits)
    S2 = bits_to_int(S2_bits)
    S3 = bits_to_int(S3_bits)

    lam = (S2 - S3) % m
    mu = (S1 - S2) % m

    delta = gcd(mu, m)
    if delta != 1:
        print("[ATAK] Ostrzeżenie: gcd(S1 - S2, m) != 1 -> wieloznaczność rozwiązań (delta =", delta, ")")

    mu_inv = mod_inv(mu, m)
    if mu_inv is None:
        print("[ATAK] Brak odwrotności modularnej, atak nieudany.")
        return None

    A = (lam * mu_inv) % m
    B = (S2 - S1 * A) % m

    S3_hat = (A * S2 + B) % m
    if S3_hat != S3:
        print("[ATAK] Weryfikacja nieudana: obliczony S3_hat != S3.")
        return None

    return A, B
