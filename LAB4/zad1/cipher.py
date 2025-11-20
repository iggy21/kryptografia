from typing import List
from lcg import LCG


def to_bits(text: str) -> List[int]:
    bits: List[int] = []
    for ch in text:
        byte = ord(ch)
        if byte > 255:
            raise ValueError("Używaj tylko znaków ASCII (kod < 256).")
        bin_str = format(byte, '08b')
        bits.extend(int(b) for b in bin_str)
    return bits


def to_text(bits: List[int]) -> str:
    chars = []
    length = len(bits) - (len(bits) % 8)
    for i in range(0, length, 8):
        byte_bits = bits[i:i+8]
        value = int(''.join(str(b) for b in byte_bits), 2)
        chars.append(chr(value))
    return ''.join(chars)


class StreamCipher:
    def __init__(self, generator: LCG):
        self.generator = generator

    def encrypt(self, plaintext: str) -> list[int]:
        xi = to_bits(plaintext)
        Y: list[int] = []
        for beta in xi:
            kappa = self.generator.next_bit()
            gamma = beta ^ kappa
            Y.append(gamma)
        return Y

    def decrypt(self, ciphertext_bits: list[int], seed: int) -> str:
        self.generator.reset(seed)
        xi: list[int] = []
        for gamma in ciphertext_bits:
            kappa = self.generator.next_bit()
            beta = gamma ^ kappa
            xi.append(beta)
        return to_text(xi)
