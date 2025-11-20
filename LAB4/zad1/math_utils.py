import random
from typing import Tuple, Optional


def ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    delta, alpha1, beta1 = ext_gcd(b % a, a)
    alpha = beta1 - (b // a) * alpha1
    beta = alpha1
    return delta, alpha, beta


def mod_inv(theta: int, m: int) -> Optional[int]:
    theta_mod = theta % m
    delta, alpha, _ = ext_gcd(theta_mod, m)
    if delta != 1:
        return None
    return (alpha % m + m) % m


def is_probable_prime(n: int, k: int = 10) -> bool:
    if n < 2 or (n > 2 and n % 2 == 0):
        return False
    if n in (2, 3):
        return True
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for _ in range(k):
        a = random.randrange(2, n - 1)  # baza testowa
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def find_prime(min_value: int, k: int = 10) -> int:
    if min_value <= 2:
        return 2
    candidate = min_value
    if candidate % 2 == 0:
        candidate += 1

    while True:
        if is_probable_prime(candidate, k=k):
            return candidate
        candidate += 2
