import sys
import random
import math

ALPHABET = [chr(ord('A') + i) for i in range(26)]
IDX = {c: i for i, c in enumerate(ALPHABET)}

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(s):

    return ''.join(ch for ch in s.upper() if ch.isalpha())

def save_text(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_key_mapping(path, mapping_cipher_to_plain):
    with open(path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(mapping_cipher_to_plain):
            f.write(f"{ALPHABET[i]} {p}\n")

def build_bigram_counts(text):
    counts = [[0.0] * 26 for _ in range(26)]
    for a, b in zip(text, text[1:]):
        ia, ib = IDX[a], IDX[b]
        counts[ia][ib] += 1.0
    return counts

def smooth_and_probs(counts, alpha):
    probs = [[0.0] * 26 for _ in range(26)]
    total = 0.0
    for i in range(26):
        for j in range(26):
            v = counts[i][j] + alpha
            probs[i][j] = v
            total += v
    for i in range(26):
        for j in range(26):
            probs[i][j] /= total
    return probs

def decrypt_with_perm(cipher_text, perm):
    table = {ALPHABET[i]: perm[i] for i in range(26)}
    return ''.join(table.get(ch, ch) for ch in cipher_text)

def bigram_counts_from_decrypted(text):
    return build_bigram_counts(text)

def log_pl_from_bigram_counts(Mhat_counts, Mref_probs):
    s = 0.0
    for i in range(26):
        for j in range(26):
            cij = Mhat_counts[i][j]
            if cij > 0.0:
                s += cij * math.log(Mref_probs[i][j])
    return s

def random_perm():
    letters = ALPHABET[:]
    random.shuffle(letters)
    return letters

def swap_two(perm):
    p = perm[:]
    i = random.randrange(26)
    j = random.randrange(26)
    while j == i:
        j = random.randrange(26)
    p[i], p[j] = p[j], p[i]
    return p, i, j


def mh_run(cipher_text, Mref_probs, T=20000, alpha=0.01, verbose=False):
    current_perm = random_perm()
    dec = decrypt_with_perm(cipher_text, current_perm)
    Mhat = bigram_counts_from_decrypted(dec)
    current_logpl = log_pl_from_bigram_counts(Mhat, Mref_probs)

    best_perm = current_perm[:]
    best_logpl = current_logpl

    for t in range(1, T + 1):
        cand_perm, i, j = swap_two(current_perm)
        cand_dec = decrypt_with_perm(cipher_text, cand_perm)
        cand_Mhat = bigram_counts_from_decrypted(cand_dec)
        cand_logpl = log_pl_from_bigram_counts(cand_Mhat, Mref_probs)

        delta = cand_logpl - current_logpl

        if delta >= 0:
            accept = True
        else:
            u = random.random()
            accept = (u <= math.exp(delta))

        if accept:
            current_perm = cand_perm
            current_logpl = cand_logpl
            if cand_logpl > best_logpl:
                best_logpl = cand_logpl
                best_perm = cand_perm[:]

        if verbose and (t % (T // 10 if T >= 10 else 1) == 0):
            print(f"[MH iter {t}/{T}] curr_logpl={current_logpl:.2f} best_logpl={best_logpl:.2f}")

    return best_perm, best_logpl

def parse_args(argv):
    args = {
        'infile': None,
        'ref': None,
        'outfile': None,
        'keyout': None,
        'iter': 20000,
        'restarts': 3,
        'alpha': 0.01,
        'seed': None,
        'verbose': False
    }

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '-i':
            args['infile'] = argv[i + 1]
            i += 2
        elif a == '-r':
            args['ref'] = argv[i + 1]
            i += 2
        elif a == '-o':
            args['outfile'] = argv[i + 1]
            i += 2
        elif a == '-k':
            args['keyout'] = argv[i + 1]
            i += 2
        elif a == '-iter':
            args['iter'] = int(argv[i + 1])
            i += 2
        elif a == '-restarts':
            args['restarts'] = int(argv[i + 1])
            i += 2
        elif a == '-alpha':
            args['alpha'] = float(argv[i + 1])
            i += 2
        elif a == '-seed':
            args['seed'] = int(argv[i + 1])
            i += 2
        elif a in ('-v', '--verbose'):
            args['verbose'] = True
            i += 1
        else:
            print("Nieznany argument:", a)
            sys.exit(1)

    if not args['infile'] or not args['ref'] or not args['outfile']:
        print("Użycie: python task2.py -i szyfrogram.txt -r reference.txt "
              "-o best_plain.txt [-k key.txt] [-iter N] [-restarts R] "
              "[-alpha a] [-seed s] [-v]")
        sys.exit(1)

    return args

def main():
    args = parse_args(sys.argv[1:])

    if args['seed'] is not None:
        random.seed(args['seed'])

    cipher_raw = read_file(args['infile'])
    cipher = clean_text(cipher_raw)
    if len(cipher) < 2:
        print("Błąd: szyfrogram po oczyszczeniu jest zbyt krótki (min. 2 litery).")
        sys.exit(1)

    ref_raw = read_file(args['ref'])
    ref = clean_text(ref_raw)
    if len(ref) < 100:
        print("Uwaga: tekst referencyjny ma mniej niż 100 znaków – model bigramowy może być mało wiarygodny.")

    Mref_counts = build_bigram_counts(ref)
    Mref_probs = smooth_and_probs(Mref_counts, args['alpha'])

    best_overall_perm = None
    best_overall_logpl = -1e99

    for r in range(args['restarts']):
        if args['verbose']:
            print(f"=== Restart {r + 1}/{args['restarts']} ===")
        perm, logpl = mh_run(cipher, Mref_probs,
                             T=args['iter'],
                             alpha=args['alpha'],
                             verbose=args['verbose'])
        if args['verbose']:
            print(f"Restart {r + 1}: best_logpl={logpl:.2f}")

        if logpl > best_overall_logpl:
            best_overall_logpl = logpl
            best_overall_perm = perm[:]

    best_plain = decrypt_with_perm(cipher, best_overall_perm)
    save_text(args['outfile'], best_plain)
    if args['keyout']:
        save_key_mapping(args['keyout'], best_overall_perm)

    print(f"Zapisano najlepsze odszyfrowanie do: {args['outfile']}")
    if args['keyout']:
        print(f"Zapisano klucz (CIPHER->PLAIN) do: {args['keyout']}")
    print(f"Najlepsze log-prawdopodobieństwo: {best_overall_logpl:.2f}")


if __name__ == '__main__':
    main()
