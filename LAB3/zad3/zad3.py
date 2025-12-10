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

def build_bigram_counts(text):
    counts = [[0.0] * 26 for _ in range(26)]
    for a, b in zip(text, text[1:]):
        ia, ib = IDX[a], IDX[b]
        counts[ia][ib] += 1.0
    return counts

def compute_phi_norm(ref_text):
    counts = build_bigram_counts(ref_text)
    max_phi = 0.0
    for i in range(26):
        for j in range(26):
            if counts[i][j] > max_phi:
                max_phi = counts[i][j]

    if max_phi <= 0.0:
        raise ValueError("Tekst referencyjny jest zbyt krótki – brak bigramów.")

    phi_norm = [[counts[i][j] / max_phi for j in range(26)] for i in range(26)]
    return phi_norm, max_phi

def decrypt_with_perm(cipher_text, perm):
    table = {ALPHABET[i]: perm[i] for i in range(26)}
    return ''.join(table.get(ch, ch) for ch in cipher_text)

def save_text(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_key_mapping(path, mapping_cipher_to_plain):
    with open(path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(mapping_cipher_to_plain):
            f.write(f"{ALPHABET[i]} {p}\n")

def score_perm(cipher_text, perm, phi_norm):
    plain = decrypt_with_perm(cipher_text, perm)
    counts = build_bigram_counts(plain)
    s = 0.0
    for i in range(26):
        row_c = counts[i]
        row_phi = phi_norm[i]
        for j in range(26):
            cij = row_c[j]
            if cij > 0.0:
                s += cij * row_phi[j]
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
    return p

def sa_run(cipher_text, phi_norm, N, T0, alpha, verbose=False):
    current_perm = random_perm()
    current_score = score_perm(cipher_text, current_perm, phi_norm)
    best_perm = current_perm[:]
    best_score = current_score
    T = T0

    for k in range(1, N + 1):
        cand_perm = swap_two(current_perm)
        cand_score = score_perm(cipher_text, cand_perm, phi_norm)
        delta = cand_score - current_score

        if delta > 0:
            accept = True
        else:
            if T <= 1e-12:
                accept = False
            else:
                try:
                    prob = math.exp(delta / T)
                except OverflowError:
                    prob = 0.0
                accept = (random.random() < prob)

        if accept:
            current_perm = cand_perm
            current_score = cand_score
            if cand_score > best_score:
                best_score = cand_score
                best_perm = cand_perm[:]

        T *= alpha

        if verbose and (k % (N // 10 if N >= 10 else 1) == 0):
            print(f"[SA iter {k}/{N}] curr_score={current_score:.4f} "
                  f"best_score={best_score:.4f} T={T:.4f}")

    return best_perm, best_score

def parse_args(argv):
    args = {
        'infile': None,
        'ref': None,
        'outfile': None,
        'keyout': None,
        'iter': 50000,
        'restarts': 5,
        'T0': 5.0,
        'alpha': 0.999,
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
        elif a == '-T0':
            args['T0'] = float(argv[i + 1])
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
        print("Użycie: python task3.py -i szyfrogram.txt -r reference.txt "
              "-o best_plain_sa.txt [-k key_sa.txt] [-iter N] [-restarts R] "
              "[-T0 T] [-alpha a] [-seed s] [-v]")
        sys.exit(1)

    return args

def main():
    args = parse_args(sys.argv[1:])

    if args['seed'] is not None:
        random.seed(args['seed'])

    cipher_raw = read_file(args['infile'])
    cipher = clean_text(cipher_raw)
    if len(cipher) < 2:
        print("Błąd: szyfrogram po oczyszczeniu ma mniej niż 2 litery.")
        sys.exit(1)

    ref_raw = read_file(args['ref'])
    ref = clean_text(ref_raw)
    if len(ref) < 100:
        print("Uwaga: tekst referencyjny jest krótki – model bigramowy może być słaby.")

    phi_norm, max_phi = compute_phi_norm(ref)

    best_overall_perm = None
    best_overall_score = -1e99

    for r in range(args['restarts']):
        if args['verbose']:
            print(f"=== SA restart {r + 1}/{args['restarts']} ===")
        perm, score = sa_run(cipher, phi_norm,
                             N=args['iter'],
                             T0=args['T0'],
                             alpha=args['alpha'],
                             verbose=args['verbose'])
        if args['verbose']:
            print(f"Restart {r + 1}: best_score={score:.4f}")

        if score > best_overall_score:
            best_overall_score = score
            best_overall_perm = perm[:]

    best_plain = decrypt_with_perm(cipher, best_overall_perm)
    save_text(args['outfile'], best_plain)
    if args['keyout']:
        save_key_mapping(args['keyout'], best_overall_perm)

    print(f"Zapisano najlepsze odszyfrowanie (SA) do: {args['outfile']}")
    if args['keyout']:
        print(f"Zapisano klucz SA (CIPHER->PLAIN) do: {args['keyout']}")
    print(f"Najlepszy wynik funkcji celu fc,g: {best_overall_score:.4f}")


if __name__ == '__main__':
    main()
