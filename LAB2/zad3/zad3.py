import math

def mod_inverse(a, m):
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    raise ValueError("Brak odwrotności modularnej")

def affine_encrypt(text, a, b):
    return ''.join(chr(((a*(ord(ch)-65)+b)%26)+65) for ch in text)

def affine_decrypt(text, a, b):
    a_inv = mod_inverse(a, 26)
    return ''.join(chr((a_inv*((ord(ch)-65)-b))%26+65) for ch in text)
