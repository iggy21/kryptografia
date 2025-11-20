import random
from math_utils import find_prime, mod_inv
from lcg import LCG
from cipher import StreamCipher
from attack import attack, recover_stream, bits_to_int


def demo_attack():
    n = 100
    print("=== FAZA I: Generowanie parametrów kryptosystemu ===")
    m = find_prime(1 << n)
    A = random.randint(1, m - 1)
    B = random.randint(0, m - 1)
    S0 = random.randint(0, m - 1)
    print(f"n = {n}")
    print(f"m (moduł, liczba pierwsza) = {m}")
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"S0 = {S0}")

    print("\n=== FAZA II: Szyfrowanie wiadomości ===")
    G = LCG(A, B, m, S0, n)
    cipher = StreamCipher(G)

    X = (
        "This is an example of a long plaintext message used in a cryptography lab. "
        "Its purpose is to demonstrate that a stream cipher based on a linear congruential "
        "generator (LCG) is vulnerable to a known-plaintext attack and can be broken."
    )
    Y = cipher.encrypt(X)
    print(f"Length of X (chars) = {len(X)}")
    print("First 100 bits of ciphertext Y:", ''.join(str(b) for b in Y[:100]))

    print("\n=== FAZA III: Wykonanie ataku kryptanalitycznego ===")
    ell = 38
    Xk = X[:ell]
    print(f"Attack assumption: first {ell} characters of plaintext are known (8 * {ell} bits).")
    print("Known fragment Xk:", repr(Xk))

    result = attack(Xk, Y, m, n)
    if result is None:
        print("Attack failed.")
        return
    A_star, B_star = result
    print("Recovered parameters: ")
    print(f"A* = {A_star}")
    print(f"B* = {B_star}")

    print("\n=== FAZA IV: Verification of recovered parameters ===")
    print(f"A == A* ? {A == A_star}")
    print(f"B == B* ? {B == B_star}")

    print("\n=== FAZA V: Recovering initial state S0 ===")
    Sigma = recover_stream(Xk, Y)
    S1_bits = Sigma[0:n]
    S1 = bits_to_int(S1_bits)

    A_inv = mod_inv(A_star, m)
    if A_inv is None:
        print("No modular inverse for A* – cannot recover S0.")
        return
    S0_star = ((S1 - B_star) * A_inv) % m
    print(f"Original S0 = {S0}")
    print(f"Recovered S0* = {S0_star}")
    print(f"S0 == S0* ? {S0 == S0_star}")

    print("\n=== FAZA VI: Decrypting the whole message ===")
    G_star = LCG(A_star, B_star, m, S0_star, n)
    cipher_star = StreamCipher(G_star)
    X_star = cipher_star.decrypt(Y, S0_star)
    print("Recovered message X*:")
    print(X_star)

    if X_star == X:
        print("\nResult: SUCCESS – message fully recovered.")
    else:
        same = sum(1 for a, b in zip(X, X_star) if a == b)
        similarity = same / len(X)
        print("\nResult: partial success.")
        print(f"Character match: {same}/{len(X)} (~{similarity:.2%})")


if __name__ == "__main__":
    demo_attack()
