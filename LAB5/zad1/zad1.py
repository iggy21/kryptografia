import math
import random

def to_bits(text):
    bits = []
    for ch in text.encode("utf-8"):
        for i in range(8):
            bits.append((ch >> (7 - i)) & 1)
    return bits


def bits_to_text(bits):
    n = len(bits) - (len(bits) % 8)
    bits = bits[:n]
    out = []
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out).decode("utf-8", errors="ignore")


class LFSR:
    def __init__(self, p, state):
        self.p = p[:]                # p0,...,pm-1
        self.state0 = state[:]       # sigma0,...,sigma(m-1)
        self.state = state[:]
        self.m = len(p)

    def reset(self, state=None):
        if state is None:
            self.state = self.state0[:]
        else:
            self.state = state[:]

    def next_bit(self):
        beta = self.state[0]       # output bit = σ0
        phi = 0
        for j in range(self.m):
            phi ^= (self.p[j] & self.state[j])
        self.state = self.state[1:] + [phi]
        return beta

    def generate(self, n):
        return [self.next_bit() for _ in range(n)]


class StreamCipher:
    def __init__(self, lfsr):
        self.lfsr = lfsr

    def encrypt(self, text):
        bits = to_bits(text)
        Y = []
        for b in bits:
            k = self.lfsr.next_bit()
            Y.append(b ^ k)
        return Y

    def decrypt(self, Y, state0):
        self.lfsr.reset(state0)
        bits = []
        for y in Y:
            k = self.lfsr.next_bit()
            bits.append(y ^ k)
        return bits_to_text(bits)


def gauss_gf2(A, b):
    A = [row[:] for row in A]
    b = b[:]
    n = len(A)

    for col in range(n):
        pivot = None
        for r in range(col, n):
            if A[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            continue

        A[col], A[pivot] = A[pivot], A[col]
        b[col], b[pivot] = b[pivot], b[col]

        for r in range(n):
            if r != col and A[r][col] == 1:
                for c in range(col, n):
                    A[r][c] ^= A[col][c]
                b[r] ^= b[col]

    x = [0]*n
    for i in reversed(range(n)):
        acc = b[i]
        for j in range(i+1, n):
            acc ^= (A[i][j] & x[j])
        x[i] = acc
    return x

def berlekamp_massey(s):
    n = len(s)
    C = [1]
    B = [1]
    L = 0
    m = -1

    for N in range(n):
        d = s[N]
        for i in range(1, L+1):
            d ^= (C[i] & s[N-i])

        if d == 0:
            continue

        T = C[:]
        delta = N - m

        if len(C) < len(B) + delta:
            C += [0] * (len(B) + delta - len(C))

        for i in range(len(B)):
            C[i+delta] ^= B[i]

        if 2*L <= N:
            L = N + 1 - L
            B = T
            m = N

    return C, L


def attack(known_text, Y, m):
    xi = to_bits(known_text)
    Sigma = [xi[i] ^ Y[i] for i in range(len(xi))]

    if len(Sigma) < 2*m:
        return None

    A = [[0]*m for _ in range(m)]
    b = [0]*m

    for i in range(m):
        for j in range(m):
            A[i][j] = Sigma[i+j]
        b[i] = Sigma[i+m]

    p = gauss_gf2(A, b)

    sigma0 = Sigma[:m]

    return p, sigma0


def demo():
    print("FAZA I")
    m = 8
    p = [random.randint(0,1) for _ in range(m)]
    p[0] = 1
    sigma0 = [random.randint(0,1) for _ in range(m)]
    sigma0[0] = 1

    print("p =", p)
    print("sigma0 =", sigma0)

    L = LFSR(p, sigma0)
    cipher = StreamCipher(L)

    print("\nFAZA II")
    X = "Tajna wiadomosc"
    Y = cipher.encrypt(X)
    print("Szyfrogram:", Y[:32])

    print("\nFAZA III")
    Lmin_chars = math.ceil(2*m/8)
    Xk = X[:Lmin_chars]
    print("Znany plaintext:", Xk)

    res = attack(Xk, Y, m)
    if res is None:
        print("Atak nieudany.")
        return

    p_star, sigma_star = res
    print("p* =", p_star)
    print("sigma0* =", sigma_star)

    print("\nFAZA IV")
    print("Czy p == p* ?", p == p_star)
    print("Czy sigma0 == sigma0* ?", sigma0 == sigma_star)

    print("\nFAZA V")
    L2 = LFSR(p_star, sigma_star)
    cipher2 = StreamCipher(L2)
    X_rec = cipher2.decrypt(Y, sigma_star)
    print("Odszyfrowano:", X_rec)

    if X_rec == X:
        print("Sukces – wiadomość odzyskana.")
    else:
        print("Blad – wiadomość niezgodna.")


if __name__ == "__main__":
    demo()
