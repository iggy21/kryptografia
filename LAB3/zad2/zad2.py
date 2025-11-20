import random
import math
from typing import Tuple, List

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_UPPER = ALPHABET.upper()
N = len(ALPHABET)

def normalize_text(s: str) -> str:
    s = s.lower()
    out = []
    allowed = set(ALPHABET)
    for ch in s:
        if ch in allowed:
            out.append(ch)
    return "".join(out)

def build_bigram_matrix(text: str, alphabet: str = ALPHABET, alpha: float = 1.0) -> List[List[float]]:
    n = len(alphabet)
    idx = {ch: i for i, ch in enumerate(alphabet)}
    M = [[0.0]*n for _ in range(n)]
    text = normalize_text(text)
    for a, b in zip(text, text[1:]):
        if a in idx and b in idx:
            M[idx[a]][idx[b]] += 1.0
    for i in range(n):
        for j in range(n):
            M[i][j] += alpha
    return M

def random_permutation(alphabet: str = ALPHABET) -> str:
    letters = list(alphabet)
    random.shuffle(letters)
    return "".join(letters)

def apply_permutation(ciphertext: str, perm: str, alphabet: str = ALPHABET) -> str:
    mapping = {alphabet[i]: perm[i] for i in range(len(alphabet))}
    out = []
    for ch in ciphertext:
        out.append(mapping.get(ch, ch))
    return "".join(out)

def bigram_counts_from_text(text: str, alphabet: str = ALPHABET, alpha: float = 1.0) -> List[List[float]]:
    return build_bigram_matrix(text, alphabet=alphabet, alpha=alpha)

def log_pl_from_bigram_counts(M_reference: List[List[float]], M_hat: List[List[float]]) -> float:

    n = len(M_reference)
    logM = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            logM[i][j] = math.log(M_reference[i][j])
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += M_hat[i][j] * logM[i][j]
    return s

def counts_matrix_from_plaintext(plaintext: str, alphabet: str = ALPHABET, alpha: float = 1.0) -> List[List[float]]:
    return build_bigram_matrix(plaintext, alphabet=alphabet, alpha=alpha)

def propose_swap(perm: str) -> str:
    n = len(perm)
    i, j = random.sample(range(n), 2)
    lst = list(perm)
    lst[i], lst[j] = lst[j], lst[i]
    return "".join(lst)

def metropolis_hastings(ciphertext: str,
                       M_reference: List[List[float]],
                       alphabet: str = ALPHABET,
                       alpha: float = 1.0,
                       iterations: int = 20000,
                       initial_perm: str = None,
                       track_best: bool = True,
                       verbose: bool = False) -> Tuple[str, str, float]:

    ciphertext = normalize_text(ciphertext)
    n = len(alphabet)
    if initial_perm is None:
        current_perm = random_permutation(alphabet)
    else:
        current_perm = initial_perm
    plain = apply_permutation(ciphertext, current_perm, alphabet=alphabet)
    M_hat = counts_matrix_from_plaintext(plain, alphabet=alphabet, alpha=alpha)
    current_log_pl = log_pl_from_bigram_counts(M_reference, M_hat)
    best_perm = current_perm
    best_log_pl = current_log_pl
    best_plain = plain

    for t in range(1, iterations+1):
        candidate_perm = propose_swap(current_perm)
        candidate_plain = apply_permutation(ciphertext, candidate_perm, alphabet=alphabet)
        M_hat_cand = counts_matrix_from_plaintext(candidate_plain, alphabet=alphabet, alpha=alpha)
        cand_log_pl = log_pl_from_bigram_counts(M_reference, M_hat_cand)

        delta = cand_log_pl - current_log_pl
        try:
            rho = math.exp(delta) if delta < 700 else float('inf')
        except OverflowError:
            rho = float('inf')
        if rho > 1.0:
            rho = 1.0

        u = random.random()
        if u <= rho:
            current_perm = candidate_perm
            current_log_pl = cand_log_pl
            plain = candidate_plain

        if track_best and current_log_pl > best_log_pl:
            best_log_pl = current_log_pl
            best_perm = current_perm
            best_plain = plain
            if verbose:
                print(f"[t={t}] Nowy najlepszy log_pl = {best_log_pl:.2f}")

    if track_best:
        return best_perm, best_plain, best_log_pl
    else:
        return current_perm, plain, current_log_pl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MH kryptoanaliza dla monoalfabetycznego szyfru podstawieniowego.")
    parser.add_argument("--ciphertext_file", type=str, default="cipher.txt", help="Plik z szyfrogramem.")
    parser.add_argument("--reference_file", type=str, default="reference.txt", help="Plik z tekstem referencyjnym.")
    parser.add_argument("--iterations", type=int, default=40000, help="Liczba iteracji MH.")
    parser.add_argument("--restarts", type=int, default=3, help="Ile losowych restartów zrobić.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing alpha.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.reference_file, "r", encoding="utf-8") as f:
        reference_text = f.read()
    with open(args.ciphertext_file, "r", encoding="utf-8") as f:
        ciphertext = f.read()

    M_ref = build_bigram_matrix(reference_text, alphabet=ALPHABET, alpha=args.alpha)

    best_overall = None
    for r in range(args.restarts):
        if args.verbose:
            print(f"=== Restart {r+1}/{args.restarts} ===")
        perm, plain, logpl = metropolis_hastings(ciphertext,
                                                M_ref,
                                                alphabet=ALPHABET,
                                                alpha=args.alpha,
                                                iterations=args.iterations,
                                                initial_perm=None,
                                                track_best=True,
                                                verbose=args.verbose)
        if best_overall is None or logpl > best_overall[2]:
            best_overall = (perm, plain, logpl)
            if args.verbose:
                print(f"Nowy best_overall z log_pl = {logpl:.2f}")

    perm, plain, logpl = best_overall
    print("WYNIK ")
    print(f"Log-likelihood : {logpl:.2f}")
    print(f"Znaleziony klucz :")
    for i, c in enumerate(ALPHABET):
        print(f"  {c} -> {perm[i]}")
    print("\nFragment odszyfrowanego tekstu:")
    print(plain[:2000])

