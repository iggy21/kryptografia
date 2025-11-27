import random
import math
import time
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. USTAWIENIA GLOBALNE
# ==============================

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Proste częstotliwości liter języka angielskiego
ENGLISH_FREQ = {
    "E": 12.02, "T": 9.10, "A": 8.12, "O": 7.68, "I": 7.31, "N": 6.95,
    "S": 6.28, "R": 6.02, "H": 5.92, "D": 4.32, "L": 3.98, "U": 2.88,
    "C": 2.71, "M": 2.61, "F": 2.30, "Y": 2.11, "W": 2.09, "G": 2.03,
    "P": 1.82, "B": 1.49, "V": 1.11, "K": 0.69, "X": 0.17, "Q": 0.11,
    "J": 0.10, "Z": 0.07,
}

# Zamiana na log-prawdopodobieństwa
LOG_PROBS = {}
total_freq = sum(ENGLISH_FREQ.values())
for ch in ALPHABET:
    freq = ENGLISH_FREQ.get(ch, 0.01)
    p = freq / total_freq
    LOG_PROBS[ch] = math.log(p)


# ==============================
# 2. FUNKCJE KRYPTOSYSTEMU
# ==============================

def normalize_text(text: str) -> str:
    """Zamiana na wielkie litery, usunięcie polskich znaków itd. (tu tylko upper)."""
    return text.upper()


def generate_random_encryption_key(rng: random.Random) -> str:
    """Losowa permutacja alfabetu – klucz SZYFRUJĄCY (plaintext -> ciphertext)."""
    perm = list(ALPHABET)
    rng.shuffle(perm)
    return "".join(perm)


def encryption_key_to_decryption_key(encrypt_key: str) -> str:
    """
    Zwraca klucz DESZYFRUJĄCY (ciphertext -> plaintext)
    na podstawie klucza szyfrującego.
    """
    # encrypt: plaintext[i] -> encrypt_key[i]
    # decrypt: ciphertext (encrypt_key[i]) -> plaintext[i]
    dec = []
    for ciph in ALPHABET:
        idx = encrypt_key.index(ciph)
        dec.append(ALPHABET[idx])
    return "".join(dec)


def encrypt(plaintext: str, encrypt_key: str) -> str:
    """Szyfr podstawieniowy monoalfabetyczny."""
    text = normalize_text(plaintext)
    result = []
    for ch in text:
        if ch in ALPHABET:
            idx = ALPHABET.index(ch)
            result.append(encrypt_key[idx])
        else:
            result.append(ch)
    return "".join(result)


def decrypt(ciphertext: str, decryption_key: str) -> str:
    """Deszyfrowanie przy pomocy klucza DESZYFRUJĄCEGO (cipher -> plain)."""
    result = []
    for ch in ciphertext.upper():
        if ch in ALPHABET:
            idx = ALPHABET.index(ch)
            result.append(decryption_key[idx])
        else:
            result.append(ch)
    return "".join(result)


def score_plaintext(plaintext: str) -> float:
    """
    Funkcja celu: suma log-prawdopodobieństw liter.
    Im wyższa wartość, tym tekst bardziej "angielski".
    """
    s = 0.0
    for ch in plaintext.upper():
        if ch in ALPHABET:
            s += LOG_PROBS[ch]
    return s


# ==============================
# 3. METRYKI OCENY
# ==============================

def key_accuracy(true_key: str, recovered_key: str) -> float:
    """% poprawnie odzyskanych liter klucza (porównujemy dwa klucze deszyfrujące)."""
    assert len(true_key) == len(recovered_key)
    correct = sum(t == r for t, r in zip(true_key, recovered_key))
    return 100.0 * correct / len(true_key)


def readability_score(score: float) -> float:
    """Tu czytelność = wartość funkcji celu."""
    return score


def iterations_to_threshold(score_history, threshold):
    """Pierwsza iteracja, gdzie score >= threshold (1-based)."""
    for i, s in enumerate(score_history, start=1):
        if s >= threshold:
            return i
    return None


# ==============================
# 4. ALGORYTMY MH i SA
# ==============================

def random_key(rng: random.Random) -> str:
    perm = list(ALPHABET)
    rng.shuffle(perm)
    return "".join(perm)


def propose_new_key(key: str, rng: random.Random) -> str:
    """Propozycja nowego klucza: zamiana dwóch losowych pozycji."""
    key_list = list(key)
    i, j = rng.sample(range(len(key_list)), 2)
    key_list[i], key_list[j] = key_list[j], key_list[i]
    return "".join(key_list)


def mh_attack(ciphertext: str, max_iter: int, rng: random.Random):
    """
    Metropolis–Hastings na przestrzeni permutacji klucza deszyfrującego.
    Zwraca:
        best_plaintext, best_key, best_score, score_history
    """
    # Stan początkowy:
    cur_key = random_key(rng)
    cur_plain = decrypt(ciphertext, cur_key)
    cur_score = score_plaintext(cur_plain)

    best_key = cur_key
    best_plain = cur_plain
    best_score = cur_score

    history = [cur_score]

    for it in range(max_iter):
        # Propozycja:
        prop_key = propose_new_key(cur_key, rng)
        prop_plain = decrypt(ciphertext, prop_key)
        prop_score = score_plaintext(prop_plain)

        # Symetryczna propozycja => MH:
        delta = prop_score - cur_score
        accept_prob = min(1.0, math.exp(delta))

        if rng.random() < accept_prob:
            cur_key = prop_key
            cur_plain = prop_plain
            cur_score = prop_score

        # Aktualizacja najlepszego
        if cur_score > best_score:
            best_score = cur_score
            best_key = cur_key
            best_plain = cur_plain

        history.append(cur_score)

    return best_plain, best_key, best_score, history


def sa_attack(ciphertext: str, max_iter: int, T0: float, alpha: float, rng: random.Random):
    """
    Symulowane wyżarzanie (Simulated Annealing).
    Zwraca:
        best_plaintext, best_key, best_score, score_history
    """
    cur_key = random_key(rng)
    cur_plain = decrypt(ciphertext, cur_key)
    cur_score = score_plaintext(cur_plain)

    best_key = cur_key
    best_plain = cur_plain
    best_score = cur_score

    history = [cur_score]

    T = T0

    for it in range(max_iter):
        prop_key = propose_new_key(cur_key, rng)
        prop_plain = decrypt(ciphertext, prop_key)
        prop_score = score_plaintext(prop_plain)

        delta = prop_score - cur_score

        if delta >= 0:
            accept = True
        else:
            if T > 0:
                accept_prob = math.exp(delta / T)
            else:
                accept_prob = 0.0
            accept = rng.random() < accept_prob

        if accept:
            cur_key = prop_key
            cur_plain = prop_plain
            cur_score = prop_score

        if cur_score > best_score:
            best_score = cur_score
            best_key = cur_key
            best_plain = cur_plain

        history.append(cur_score)

        # Chłodzenie
        T *= alpha

    return best_plain, best_key, best_score, history


# ==============================
# 5. WIELOKROTNE URUCHOMIENIA (STATYSTYKA)
# ==============================

def run_many_mh(ciphertext, true_dec_key, n_runs=20, max_iter=20000, threshold=None):
    results = []
    for run in range(n_runs):
        rng = random.Random(run)

        start = time.perf_counter()
        plaintext, key, score, history = mh_attack(ciphertext, max_iter, rng)
        end = time.perf_counter()

        acc = key_accuracy(true_dec_key, key)
        read = readability_score(score)
        iters_to_good = None
        if threshold is not None:
            iters_to_good = iterations_to_threshold(history, threshold)

        results.append({
            "run": run,
            "accuracy": acc,
            "readability": read,
            "time": end - start,
            "iterations_to_good": iters_to_good,
            "history": history,
            "score": score,
            "plaintext": plaintext,
            "key": key,
        })
    return results


def run_many_sa(ciphertext, true_dec_key, n_runs=20, max_iter=20000, T0=20.0, alpha=0.9995, threshold=None):
    results = []
    for run in range(n_runs):
        rng = random.Random(run)

        start = time.perf_counter()
        plaintext, key, score, history = sa_attack(ciphertext, max_iter, T0, alpha, rng)
        end = time.perf_counter()

        acc = key_accuracy(true_dec_key, key)
        read = readability_score(score)
        iters_to_good = None
        if threshold is not None:
            iters_to_good = iterations_to_threshold(history, threshold)

        results.append({
            "run": run,
            "accuracy": acc,
            "readability": read,
            "time": end - start,
            "iterations_to_good": iters_to_good,
            "history": history,
            "score": score,
            "plaintext": plaintext,
            "key": key,
        })
    return results


def summarize_results(name, results):
    accuracies = [r["accuracy"] for r in results]
    times = [r["time"] for r in results]
    readabilities = [r["readability"] for r in results]
    iters_good = [r["iterations_to_good"] for r in results if r["iterations_to_good"] is not None]

    print(f"\n=== {name} ===")
    print(f"Średni % poprawnego klucza: {mean(accuracies):.2f} ± {stdev(accuracies):.2f}")
    print(f"Średnia czytelność (score): {mean(readabilities):.2f} ± {stdev(readabilities):.2f}")
    print(f"Średni czas [s]: {mean(times):.4f} ± {stdev(times):.4f}")
    if iters_good:
        print(f"Średnia liczba iteracji do dobrego wyniku: {mean(iters_good):.0f} ± {stdev(iters_good):.0f}")
    else:
        print("Dobry wynik wg progu nie został osiągnięty w żadnym z uruchomień.")


def average_history(results, max_len=None):
    histories = [r["history"] for r in results]
    if max_len is None:
        max_len = min(len(h) for h in histories)
    arr = np.array([h[:max_len] for h in histories], dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


# ==============================
# 6. GŁÓWNA ANALIZA
# ==============================

def main():
    rng_global = random.Random(123)

    # ===== 6.1. Tekst jawny i szyfr =====
    PLAINTEXT = """
    THIS IS A SAMPLE PLAINTEXT USED TO TEST STOCHASTIC CRYPTANALYSIS METHODS.
    THE GOAL IS TO RECOVER THE SECRET KEY OF A SIMPLE SUBSTITUTION CIPHER.
    THE TEXT SHOULD BE LONG ENOUGH TO PROVIDE STATISTICAL INFORMATION ABOUT
    LETTER FREQUENCIES AND MAKE THE ATTACK POSSIBLE.
    """

    plaintext_norm = normalize_text(PLAINTEXT)
    encrypt_key = generate_random_encryption_key(rng_global)
    true_dec_key = encryption_key_to_decryption_key(encrypt_key)
    ciphertext = encrypt(plaintext_norm, encrypt_key)

    print("=== Przykładowy tekst jawny (fragment) ===")
    print(plaintext_norm[:200], "...")
    print("\n=== Zaszyfrowany tekst (fragment) ===")
    print(ciphertext[:200], "...")
    print("\nPrawdziwy klucz deszyfrujący (cipher -> plain):")
    print(true_dec_key)

    # ===== 6.2. Pojedyncze uruchomienie – orientacyjny score =====
    MAX_ITER_BASE = 8000
    rng_tmp = random.Random(999)
    _, _, score_mh_tmp, _ = mh_attack(ciphertext, MAX_ITER_BASE, rng_tmp)
    GOOD_SCORE_THRESHOLD = 0.95 * score_mh_tmp  # próg "dobrego" wyniku

    print(f"\nScore z pojedynczego przebiegu MH: {score_mh_tmp:.2f}")
    print(f"Przyjęty próg dobrego wyniku: {GOOD_SCORE_THRESHOLD:.2f}")

    # ===== 6.3. Wielokrotne uruchomienia MH i SA =====
    N_RUNS = 10  # możesz zwiększyć do 20+ dla dokładniejszej statystyki

    mh_results = run_many_mh(
        ciphertext,
        true_dec_key,
        n_runs=N_RUNS,
        max_iter=MAX_ITER_BASE,
        threshold=GOOD_SCORE_THRESHOLD,
    )

    sa_results = run_many_sa(
        ciphertext,
        true_dec_key,
        n_runs=N_RUNS,
        max_iter=MAX_ITER_BASE,
        T0=20.0,
        alpha=0.9995,
        threshold=GOOD_SCORE_THRESHOLD,
    )

    summarize_results("Metropolis–Hastings", mh_results)
    summarize_results("Simulated Annealing", sa_results)

    # ===== 6.4. Przykładowe odszyfrowane teksty =====
    best_mh = max(mh_results, key=lambda r: r["score"])
    best_sa = max(sa_results, key=lambda r: r["score"])

    print("\n=== Najlepszy wynik MH – odszyfrowany tekst (fragment) ===")
    print(best_mh["plaintext"][:300], "...")
    print("Klucz MH:", best_mh["key"])
    print(f"% poprawnego klucza (MH): {key_accuracy(true_dec_key, best_mh['key']):.2f}%")

    print("\n=== Najlepszy wynik SA – odszyfrowany tekst (fragment) ===")
    print(best_sa["plaintext"][:300], "...")
    print("Klucz SA:", best_sa["key"])
    print(f"% poprawnego klucza (SA): {key_accuracy(true_dec_key, best_sa['key']):.2f}%")

    # ===== 6.5. Analiza zbieżności – pojedynczy przebieg =====
    mh_hist = mh_results[0]["history"]
    sa_hist = sa_results[0]["history"]

    plt.figure()
    plt.plot(mh_hist, label="MH")
    plt.plot(sa_hist, label="SA")
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.title("Zbieżność algorytmów – pojedyncze uruchomienie")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ===== 6.6. Analiza zbieżności – uśredniona krzywa =====
    mh_mean, mh_std = average_history(mh_results)
    sa_mean, sa_std = average_history(sa_results)

    plt.figure()
    x_mh = range(len(mh_mean))
    x_sa = range(len(sa_mean))
    plt.plot(x_mh, mh_mean, label="MH (średnia)")
    plt.fill_between(x_mh, mh_mean - mh_std, mh_mean + mh_std, alpha=0.2)

    plt.plot(x_sa, sa_mean, label="SA (średnia)")
    plt.fill_between(x_sa, sa_mean - sa_std, sa_mean + sa_std, alpha=0.2)

    plt.xlabel("Iteracja")
    plt.ylabel("Wartość funkcji celu")
    plt.title("Zbieżność – średnia i odchylenie standardowe")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==============================
    # 7. WPŁYW PARAMETRÓW
    # ==============================

    # ---- 7.1. MH: wpływ liczby iteracji T ----
    T_values = [2000, 4000, 8000, 12000]
    N_RUNS_PARAM = 5

    mh_acc_means = []
    mh_time_means = []

    for T in T_values:
        res = run_many_mh(
            ciphertext,
            true_dec_key,
            n_runs=N_RUNS_PARAM,
            max_iter=T,
            threshold=GOOD_SCORE_THRESHOLD,
        )
        mh_acc_means.append(mean(r["accuracy"] for r in res))
        mh_time_means.append(mean(r["time"] for r in res))

    plt.figure()
    plt.plot(T_values, mh_acc_means, marker="o")
    plt.xlabel("Liczba iteracji T")
    plt.ylabel("Średni % poprawnego klucza")
    plt.title("MH – wpływ liczby iteracji na jakość")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(T_values, mh_time_means, marker="o")
    plt.xlabel("Liczba iteracji T")
    plt.ylabel("Średni czas [s]")
    plt.title("MH – wpływ liczby iteracji na czas")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- 7.2. SA: wpływ temperatury początkowej T0 ----
    T0_values = [2.0, 5.0, 10.0, 20.0, 40.0]
    sa_acc_means_T0 = []

    for T0 in T0_values:
        res = run_many_sa(
            ciphertext,
            true_dec_key,
            n_runs=N_RUNS_PARAM,
            max_iter=MAX_ITER_BASE,
            T0=T0,
            alpha=0.9995,
            threshold=GOOD_SCORE_THRESHOLD,
        )
        sa_acc_means_T0.append(mean(r["accuracy"] for r in res))

    plt.figure()
    plt.plot(T0_values, sa_acc_means_T0, marker="o")
    plt.xlabel("Temperatura początkowa T0")
    plt.ylabel("Średni % poprawnego klucza")
    plt.title("SA – wpływ temperatury początkowej T0")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- 7.3. SA: wpływ współczynnika chłodzenia α ----
    alpha_values = [0.9999, 0.9995, 0.999, 0.995, 0.99]
    sa_acc_means_alpha = []

    for alpha in alpha_values:
        res = run_many_sa(
            ciphertext,
            true_dec_key,
            n_runs=N_RUNS_PARAM,
            max_iter=MAX_ITER_BASE,
            T0=20.0,
            alpha=alpha,
            threshold=GOOD_SCORE_THRESHOLD,
        )
        sa_acc_means_alpha.append(mean(r["accuracy"] for r in res))

    plt.figure()
    plt.plot(alpha_values, sa_acc_means_alpha, marker="o")
    plt.xlabel("Współczynnik chłodzenia α")
    plt.ylabel("Średni % poprawnego klucza")
    plt.title("SA – wpływ współczynnika chłodzenia α")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
