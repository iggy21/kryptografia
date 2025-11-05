# Kryptografia i kryptoanaliza
## Laboratorium 1
### Grupa 1ID24B
### Autorzy: Iga Ozimska, Eliza Janus

### Zadanie 1

Dokonaj implementacji programu szyfrującego i deszyfrującego zadany tekst.
1. Tekst jawny powinien być importowany do programu z pliku tekstowego, którego nazwa określona powinna być
po zdefiniowanym argumencie / fladze: -i.
2. Wynik pracy programu powinien być eksportowany do pliku tekstowego, którego nazwa określona powinna być
po zdefiniowanym argumencie / fladze: -o.
3. Klucz powinien być importowany z pliku tekstowego, którego nazwa powinna być określona po zdefiniowanym
argumencie / fladze: -k.
4. Tryb pracy programu powinien być określony poprzez flagi: -e dla procesu szyfrowania, -d dla procesu deszy-
frowania.
Przykład wywołania programu w celu zaszyfrowania tekstu:
./program -e -k klucz.txt -i tekst_jawny.txt -o szyfrogram.txt
Przykład wywołania programu w celu odszyfrowania tekstu:
./program -d -k klucz.txt -i szyfrogram.txt -o tekst_odszyfrowany.txt
#### Implementacja


``` Python
import argparse
import string

# ---------------------------------------------------------------
# Funkcja: load_key
# ---------------------------------------------------------------
# Wejście:
#   - filename: nazwa pliku tekstowego zawierającego klucz (np. "klucz.txt")
#
# Wyjście:
#   - słownik (dict) mapujący litery tekstu jawnego na odpowiadające im litery szyfrogramu
#
# Działanie:
#   Funkcja otwiera plik z kluczem i wczytuje pary znaków (A B),
#   gdzie 'A' to litera tekstu jawnego, a 'B' to jej zamiennik w szyfrze.
#   Klucz przechowywany jest w postaci słownika w formacie {'A': 'B', 'B': 'C', ...}.
# ---------------------------------------------------------------
def load_key(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue  # pomija puste linie
            a, b = line.strip().split()
            mapping[a.upper()] = b.upper()
    return mapping


# ---------------------------------------------------------------
# Funkcja: preprocess_text
# ---------------------------------------------------------------
# Wejście:
#   - text: ciąg znaków (string) odczytany z pliku wejściowego
#
# Wyjście:
#   - tekst zawierający tylko wielkie litery A-Z (bez spacji, cyfr, polskich znaków itp.)
#
# Działanie:
#   Funkcja usuwa wszystkie znaki, które nie należą do alfabetu angielskiego.
#   Dzięki temu szyfrowanie działa wyłącznie na literach A–Z.
# ---------------------------------------------------------------
def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)


# ---------------------------------------------------------------
# Funkcja: substitute
# ---------------------------------------------------------------
# Wejście:
#   - text: tekst po wstępnym przetworzeniu (tylko litery A-Z)
#   - mapping: słownik mapujący litery zgodnie z kluczem szyfru
#
# Wyjście:
#   - nowy tekst (string) po dokonaniu podstawień liter według klucza
#
# Działanie:
#   Każda litera w tekście wejściowym jest zamieniana zgodnie z mapowaniem.
#   Jeśli znak nie znajduje się w kluczu (np. brak literki w mapie),
#   zostaje przepisany bez zmian.
# ---------------------------------------------------------------
def substitute(text, mapping):
    return ''.join(mapping.get(ch, ch) for ch in text)


# ---------------------------------------------------------------
# Funkcja główna: main
# ---------------------------------------------------------------
# Wejście:
#   Argumenty przekazane w wierszu poleceń:
#       -e : tryb szyfrowania
#       -d : tryb deszyfrowania
#       -i : plik wejściowy (tekst jawny lub szyfrogram)
#       -o : plik wyjściowy (szyfrogram lub tekst odszyfrowany)
#       -k : plik z kluczem
#
# Wyjście:
#   - brak zwracanego wyniku (wynik zapisywany do pliku wyjściowego)
#
# Działanie:
#   1. Odczytuje argumenty z wiersza poleceń.
#   2. Wczytuje klucz z pliku i odwraca go, jeśli wybrano tryb deszyfrowania (-d).
#   3. Wczytuje tekst z pliku wejściowego.
#   4. Przetwarza tekst (usuwa znaki spoza alfabetu).
#   5. Dokonuje podstawienia znaków (szyfrowanie lub deszyfrowanie).
#   6. Zapisuje wynik do pliku wyjściowego.
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Substitution cipher")
    parser.add_argument('-e', action='store_true', help='Encrypt mode')
    parser.add_argument('-d', action='store_true', help='Decrypt mode')
    parser.add_argument('-i', required=_

```

#### Wyniki

``` sh
python3 zad1.py -e -k klucz.txt -i tekst_jawny.txt -o szyfrogram.txt
```

### Zadanie 2

Rozbudować program z poprzedniego przykładu poprzez dodanie do niego funkcjonalności generowania statystyk licz-
ności występowania n-gramów (sekwencji kolejnych liter), to jest mono-gramów (pojedynczych liter), bi-gramów (wy-
razów dwuliterowych), tri-gramów (wyrazów trzyliterowych) oraz quad-gramów (wyrazów czteroliterowych). Funk-
cjonalność ta powinna być wyzwalana poprzez dodanie do programu jednej z następujących flag: -g1, -g2, -g3 lub
-g4, po której powinna zostać określona nazwa pliku, do którego zapisane zostaną wyniki.
Przykład wywołania programu:
./program -i tekst_jawny.txt -g1 monogramy.txt

Przykład wyznaczania bi-gramów dla tekstu:
Tekst jawny:
This is an example of plain text

Tekst wstępnie przetworzony:
THISISANEXAMPLEOFPLAINTEXT
Kilka pierwszych bi-gramów:
1. TH
2. HI
3. IS
4. SI
5. IS
6. SA

Dla każdego wyznaczonego n-gramu należy wyznaczyć liczność jego występowania w badanym tekście. Wynik pracy
programu powinien być wygenerowany w postaci tabeli:
n-gram liczbość

Przykład:
TH 1
HI 1
IS 2
SI 1
SA 1
#### Implementacja

```Python 
import argparse
import string
from collections import Counter

# ---------------------------------------------------------------
# Funkcja: load_key
# ---------------------------------------------------------------
# Wejście:
#   - filename: nazwa pliku tekstowego zawierającego klucz (np. "klucz.txt")
#
# Wyjście:
#   - słownik (dict), w którym kluczem jest litera jawna, a wartością litera szyfrogramu
#
# Działanie:
#   Funkcja wczytuje pary znaków z pliku (np. "A D"), które definiują zamianę liter
#   w szyfrze podstawieniowym. Wszystkie znaki zamieniane są na wielkie litery.
# ---------------------------------------------------------------
def load_key(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue  # pomija puste linie
            a, b = line.strip().split()
            mapping[a.upper()] = b.upper()
    return mapping


# ---------------------------------------------------------------
# Funkcja: preprocess_text
# ---------------------------------------------------------------
# Wejście:
#   - text: dowolny tekst wczytany z pliku (może zawierać cyfry, znaki specjalne itp.)
#
# Wyjście:
#   - tekst (string) zawierający tylko wielkie litery A–Z
#
# Działanie:
#   Funkcja usuwa wszystkie znaki spoza alfabetu łacińskiego (A-Z),
#   zamienia litery na wielkie i zwraca oczyszczony tekst.
# ---------------------------------------------------------------
def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)


# ---------------------------------------------------------------
# Funkcja: substitute
# ---------------------------------------------------------------
# Wejście:
#   - text: tekst zawierający tylko litery A–Z
#   - mapping: słownik z definicją zamian liter (np. {'A': 'D', 'B': 'E'})
#
# Wyjście:
#   - tekst po dokonaniu podstawień zgodnie z kluczem
#
# Działanie:
#   Każda litera tekstu jest zastępowana odpowiednikiem z mapowania.
#   Jeśli litera nie występuje w kluczu, pozostaje bez zmian.
# ---------------------------------------------------------------
def substitute(text, mapping):
    return ''.join(mapping.get(ch, ch) for ch in text)


# ---------------------------------------------------------------
# Funkcja: generate_ngrams
# ---------------------------------------------------------------
# Wejście:
#   - text: tekst źródłowy (ciąg znaków)
#   - n: długość n-gramu (np. 1 = monogram, 2 = bigram, 3 = trigram, 4 = czterogram)
#
# Wyjście:
#   - lista wszystkich n-gramów występujących w tekście
#
# Działanie:
#   Funkcja tworzy listę wszystkich fragmentów tekstu o długości n
#   przesuwając "okno" o jeden znak. Np. dla tekstu ABCD i n=2 → ["AB", "BC", "CD"]
# ---------------------------------------------------------------
def generate_ngrams(text, n):
    return [text[i:i+n] for i in range(len(text) - n + 1)]


# ---------------------------------------------------------------
# Funkcja: save_ngram_stats
# ---------------------------------------------------------------
# Wejście:
#   - ngrams: lista n-gramów wygenerowanych z tekstu
#   - output_file: nazwa pliku, do którego zapisane zostaną statystyki
#
# Wyjście:
#   - brak (wynik zapisany do pliku)
#
# Działanie:
#   Funkcja liczy częstość występowania każdego n-gramu
#   i zapisuje wynik w formacie: "<n-gram> <liczba wystąpień>" w pliku tekstowym.
# ---------------------------------------------------------------
def save_ngram_stats(ngrams, output_file):
    counts = Counter(ngrams)
    with open(output_file, 'w', encoding='utf-8') as f:
        for gram, count in counts.items():
            f.write(f"{gram} {count}\n")


# ---------------------------------------------------------------
# Funkcja główna: main
# ---------------------------------------------------------------
# Wejście:
#   Argumenty wiersza poleceń:
#     -e  → tryb szyfrowania
#     -d  → tryb deszyfrowania
#     -i  → plik wejściowy (z tekstem jawnym lub szyfrogramem)
#     -o  → plik wyjściowy (gdzie zapisany zostanie wynik)
#     -k  → plik z kluczem szyfrującym
#     -g1, -g2, -g3, -g4 → pliki do zapisania statystyk n-gramów (od 1 do 4)
#
# Wyjście:
#   - brak (program zapisuje wyniki do plików)
#
# Działanie:
#   1. Wczytuje argumenty programu.
#   2. Przetwarza tekst wejściowy – usuwa znaki spoza alfabetu.
#   3. Jeśli wybrano tryb -e lub -d:
#        - wczytuje klucz z pliku (-k),
#        - odwraca mapowanie w trybie deszyfrowania (-d),
#        - wykonuje szyfrowanie/deszyfrowanie i zapisuje wynik do pliku (-o).
#   4. Jeśli podano argumenty -g1 ... -g4, generuje statystyki n-gramów
#      (monogramy, bigramy, trigramy, czterogramy) i zapisuje do odpowiednich plików.
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Substitution cipher + n-gram analysis")
    parser.add_argument('-e', action='store_true', help='Encrypt mode')
    parser.add_argument('-d', action='store_true', help='Decrypt mode')
    parser.add_argument('-i', required=True, help='Input file')
    parser.add_argument('-o', help='Output file')
    parser.add_argument('-k', help='Key file')
    parser.add_argument('-g1', help='Generate monogram stats')
    parser.add_argument('-g2', help='Generate bigram stats')
    parser.add_argument('-g3', help='Generate trigram stats')
    parser.add_argument('-g4', help='Generate quadgram stats')

    args = parser.parse_args()

    # Wczytanie i wstępne przetworzenie tekstu
    with open(args.i, 'r', encoding='utf-8') as f:
        text = preprocess_text(f.read())

    # --- Tryb szyfrowania / deszyfrowania ---
    if args.e or args.d:
        if not args.k or not args.o:
            print("Musisz podać plik z kluczem (-k) i plik wyjściowy (-o).")
            return

        key = load_key(args.k)
        if args.d:
            # Odwrócenie klucza dla deszyfrowania
            key = {v: k for k, v in key.items()}

        result = substitute(text, key)

        with open(args.o, 'w', encoding='utf-8') as f:
            f.write(result)

        mode = "Szyfrowanie" if args.e else "Deszyfrowanie"
        print(f"{mode} zakończone. Wynik zapisano w {args.o}")

    # --- Analiza n-gramów ---
    for n, flag in enumerate([args.g1, args.g2, args.g3, args.g4], start=1):
        if flag:
            ngrams = generate_ngrams(text, n)
            save_ngram_stats(ngrams, flag)
            print(f"Zapisano statystyki {n}-gramów do {flag}")


# ---------------------------------------------------------------
# Punkt wejścia programu
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()

```
#### Wyniki

``` sh
cd zad2
python3 zad2.py -e -k klucz.txt -i tekst_jawny.txt -o szyfrogram.txt
```

``` sh
cd zad2
python3 zad2.py -d -k klucz.txt -i szyfrogram.txt -o tekst_odkodowany.txt
```

### Zadanie 3
Uzupełnij program z poprzedniego zadania, tak aby w przypadku podania flagi -rX, gdzie X jest liczbą należącą do
zbioru {1, 2, 3, 4} a następnie nazwy pliku, program odczytywał z niego referencyjną bazę n-gramów. Liczby z
podanego zbioru odpowiadają: {mono-gramom, bi-gramom, tri-gramom, quad-gramom}.

#### Implementacja
```Python 
import argparse
import string
from collections import Counter

# ------------------------------------------------------------
# Funkcja: load_key(filename)
# Wejście:
#   filename - nazwa pliku z kluczem, gdzie każda linia zawiera parę liter np. "A Q"
# Wyjście:
#   słownik mapujący wielkie litery na odpowiadające im litery z klucza
# Implementacja:
#   Odczytuje plik linia po linii, pomija puste linie,
#   konwertuje litery na wielkie i tworzy mapowanie liter.
# ------------------------------------------------------------
def load_key(filename):
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            a, b = line.strip().split()
            mapping[a.upper()] = b.upper()
    return mapping


# ------------------------------------------------------------
# Funkcja: preprocess_text(text)
# Wejście:
#   text - dowolny tekst (str)
# Wyjście:
#   tekst zawierający tylko wielkie litery angielskie
# Implementacja:
#   Zamienia tekst na wielkie litery i usuwa znaki spoza A-Z.
# ------------------------------------------------------------
def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)


# ------------------------------------------------------------
# Funkcja: substitute(text, mapping)
# Wejście:
#   text - tekst wejściowy (wielkie litery)
#   mapping - słownik mapowania liter
# Wyjście:
#   tekst po podstawieniu liter zgodnie z mapping
# Implementacja:
#   Zamienia każdą literę zgodnie ze słownikiem, lub pozostawia niezmienioną.
# ------------------------------------------------------------
def substitute(text, mapping):
    return ''.join(mapping.get(ch, ch) for ch in text)


# ------------------------------------------------------------
# Funkcja: generate_ngrams(text, n)
# Wejście:
#   text - przetworzony tekst (str)
#   n - długość n-gramu (int)
# Wyjście:
#   lista n-gramów (ciągów długości n)
# Implementacja:
#   Tworzy listę kolejnych fragmentów tekstu długości n.
# ------------------------------------------------------------
def generate_ngrams(text, n):
    """Zwraca listę wszystkich n-gramów w tekście."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


# ------------------------------------------------------------
# Funkcja: save_ngram_stats(ngrams, output_file)
# Wejście:
#   ngrams - lista n-gramów (str)
#   output_file - nazwa pliku do zapisu statystyk
# Wyjście:
#   brak (zapis do pliku)
# Implementacja:
#   Liczy wystąpienia n-gramów i zapisuje w pliku prawdopodobieństwa wystąpienia.
# ------------------------------------------------------------
def save_ngram_stats(ngrams, output_file):
    """Zapisuje liczności n-gramów do pliku."""
    counts = Counter(ngrams)
    total = sum(counts.values())
    with open(output_file, 'w', encoding='utf-8') as f:
        for gram, count in counts.items():
            prob = count / total
            f.write(f"{gram} {prob}\n")


# ------------------------------------------------------------
# Funkcja: load_reference_ngrams(filename)
# Wejście:
#   filename - plik referencyjnych n-gramów z przypisanymi prawdopodobieństwami
# Wyjście:
#   słownik {n-gram: prawdopodobieństwo} z normalizacją sumy do 1
# Implementacja:
#   Wczytuje plik, pomija puste i błędne linie,
#   sumuje prawdopodobieństwa i normalizuje je,
#   informuje o sumie po normalizacji.
# ------------------------------------------------------------
def load_reference_ngrams(filename):
    """Wczytuje referencyjną bazę n-gramów (Gi, Pi) i normalizuje prawdopodobieństwa."""
    reference = {}
    total_prob = 0.0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            gram, prob = parts
            p = float(prob)
            reference[gram] = p
            total_prob += p
    if total_prob == 0:
        print("Uwaga: suma prawdopodobieństw w pliku referencyjnym wynosi 0!")
    else:
        # Normalizacja sumy na 1
        for gram in reference:
            reference[gram] /= total_prob
    print(f"Suma prawdopodobieństw w referencji po normalizacji: {sum(reference.values())}")
    return reference


# ------------------------------------------------------------
# Funkcja: compute_chi_square(text, reference_probs, n)
# Wejście:
#   text - przetworzony tekst (str)
#   reference_probs - słownik {n-gram: prawdopodobieństwo}
#   n - długość n-gramu (int)
# Wyjście:
#   wartość statystyki testu chi-kwadrat (float)
# Implementacja:
#   Liczy n-gramy w tekście,
#   oblicza statystykę chi-kwadrat względem spodziewanych prawdopodobieństw,
#   wypisuje diagnostyczne informacje.
# ------------------------------------------------------------
def compute_chi_square(text, reference_probs, n):
    """Oblicza wartość testu chi-kwadrat z diagnostyką."""
    test_ngrams = generate_ngrams(text, n)
    test_counts = Counter(test_ngrams)
    total_test_ngrams = sum(test_counts.values())

    chi_square = 0.0
    for gram, expected_prob in reference_probs.items():
        observed = test_counts.get(gram, 0)
        expected = total_test_ngrams * expected_prob
        if expected > 0:
            chi_square += (observed - expected) ** 2 / expected

    print(f"Wartość testu chi-kwadrat: {chi_square}")
    return chi_square


# ------------------------------------------------------------
# Funkcja główna: main()
# Wejście:
#   argumenty wiersza poleceń:
#     -e/-d : szyfrowanie/deszyfrowanie
#     -i : plik wejściowy
#     -o : plik wyjściowy (dla szyfrowania/deszyfrowania)
#     -k : plik z kluczem
#     -g1, -g2, -g3, -g4 : pliki do zapisu statystyk n-gramów (monogram, bigram, trigram, quadgram)
#     -r1, -r2, -r3, -r4 : pliki z referencyjnymi prawdopodobieństwami n-gramów
#     -s : flaga do obliczenia testu chi-kwadrat
# Wyjście:
#   zapis do plików, komunikaty diagnostyczne w konsoli
# Implementacja:
#   Przetwarza tekst,
#   wykonuje szyfrowanie/deszyfrowanie jeśli wybrane,
#   generuje statystyki n-gramów jeśli wybrane,
#   oblicza i wypisuje wartość testu chi-kwadrat, jeśli wybrane.
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Substitution cipher + n-gram analysis + chi-square test")
    parser.add_argument('-e', action='store_true', help='Encrypt mode')
    parser.add_argument('-d', action='store_true', help='Decrypt mode')
    parser.add_argument('-i', required=True, help='Input file')
    parser.add_argument('-o', help='Output file')
    parser.add_argument('-k', help='Key file')
    parser.add_argument('-g1', help='Generate monogram stats (output file)')
    parser.add_argument('-g2', help='Generate bigram stats (output file)')
    parser.add_argument('-g3', help='Generate trigram stats (output file)')
    parser.add_argument('-g4', help='Generate quadgram stats (output file)')
    parser.add_argument('-r1', help='Reference monogram file')
    parser.add_argument('-r2', help='Reference bigram file')
    parser.add_argument('-r3', help='Reference trigram file')
    parser.add_argument('-r4', help='Reference quadgram file')
    parser.add_argument('-s', action='store_true', help='Compute chi-square test')

    args = parser.parse_args()

    # Wczytaj i przetwórz tekst wejściowy
    with open(args.i, 'r', encoding='utf-8') as f:
        text = preprocess_text(f.read())
    print("Przetworzony tekst (pierwsze 100 znaków):", text[:100])

    # Szyfrowanie lub deszyfrowanie
    if args.e or args.d:
        if not args.k or not args.o:
            print("Musisz podać plik z kluczem (-k) i plik wyjściowy (-o).")
            return

        key = load_key(args.k)
        if args.d:
            key = {v: k for k, v in key.items()}

        result = substitute(text, key)
        with open(args.o, 'w', encoding='utf-8') as f:
            f.write(result)
        mode = "Szyfrowanie" if args.e else "Deszyfrowanie"
        print(f"{mode} zakończone. Wynik zapisano w {args.o}")

    # Generowanie statystyk n-gramów
    for n, flag in enumerate([args.g1, args.g2, args.g3, args.g4], start=1):
        if flag:
            ngrams = generate_ngrams(text, n)
            save_ngram_stats(ngrams, flag)
            print(f"Zapisano statystyki {n}-gramów do {flag}")

    # Test chi-kwadrat
    for n, ref_flag in enumerate([args.r1, args.r2, args.r3, args.r4], start=1):
        if ref_flag and args.s:
            reference = load_reference_ngrams(ref_flag)
            chi_val = compute_chi_square(text, reference, n)
            print(f"Wartość testu χ² dla {n}-gramów: {chi_val:.4f}")

if __name__ == "__main__":
    main()
```
#### Wyniki
``` sh
python main.py -i corpus.txt -g2 bigramy_ref.txt
python main.py -i test_corpus.txt -r2 bigramy_ref.txt -s
```
``` sh
python main.py -i corpus.txt -g2 trigramy_ref.txt
python main.py -i test_corpus.txt -r2 trigramy_ref.txt -
```
Wartość statystyki jest poniżej poziomu krytycznego, co wskazuje na brak istotnej statystycznie różnicy między badanymi rozkładami.

Zatem wynik testu sugeruje, że rozkład bigramów w testowanym tekście pasuje do rozkładu referencyjnego i teksty można uznać za zgodne pod względem analizy n-gramowej.

#### Zadanie 4
Wykonać eksperymenty:
• Dokonaj obserwacji wyniku testu χ
2 dla tekstu jawnego i zaszyfrowanego o różnych długościach.
• Wiadomo, iż wynik testu może być znacząco zaburzony w przypadku gdy brane są pod uwagę symbole (n-gramy),
które rzadko występują w tekście, np w przypadku mono-gramów języka angielskiego są to litery: J, K, Q, X oraz
Z (patrz odczytana tablica częstości mono-gramów). Zbadaj wynik testu χ
2 w przypadku gdy do wyznaczenia
testu pominięte zostaną rzadko występujące n-gramy.
#### Implementacja
```Python 
import string

# ------------------------------------------------------------
# Funkcja: preprocess_text(text)
# Wejście:
#   text - ciąg znaków (str)
# Wyjście:
#   tekst przetworzony na wielkie litery, zawierający tylko znaki A-Z
# Implementacja:
#   Funkcja konwertuje wszystkie litery na wielkie i usuwa znaki spoza alfabetu angielskiego.
# ------------------------------------------------------------
def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)


# ------------------------------------------------------------
# Zmienna: base_file
# Opis:
#   Nazwa pliku źródłowego z którego wczytujemy tekst bazowy.
# ------------------------------------------------------------
base_file = "base.txt"

# ------------------------------------------------------------
# Zmienna: lengths
# Opis:
#   Lista długości (w znakach) próbek tekstu jakie chcemy wyciągnąć z tekstu bazowego.
# ------------------------------------------------------------
lengths = [500, 1000, 2000, 5000]

# ------------------------------------------------------------
# Wczytanie i wstępne przetworzenie tekstu bazowego
# ------------------------------------------------------------
with open(base_file, "r", encoding="utf-8") as f:
    text = preprocess_text(f.read())

# ------------------------------------------------------------
# Generowanie próbek tekstu o zadanych długościach i zapis do plików
# ------------------------------------------------------------
for L in lengths:
    sample = text[:L]             # Wycięcie próby o długości L znaków od początku tekstu
    out_file = f"sample_{L}.txt" # Nazwa pliku wyjściowego
    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write(sample)      # Zapis próbki do pliku
    print(f"Utworzono {out_file} ({len(sample)} znaków)")

```
#### Wyniki
Dla tekstu jawnego:

``` sh
python zad3\main.py -i zad4\sample_500.txt -r1 zad4\english_monogram.txt -s
python zad3\main.py -i zad4\sample_1000.txt -r1 zad4\english_monogram.txt -s
python zad3\main.py -i zad4\sample_2000.txt -r1 zad4\english_monogram.txt -s
python zad3\main.py -i zad4\sample_5000.txt -r1 zad4\english_monogram.txt -s
``` 

Dla tekstu zaszyfrowanego:

``` sh
python zad3\main.py -i zad4\sample_500_enc.txt -r1 zad4\english_monogram.txt -s
python zad3\main.py -i zad4\sample_1000_enc.txt -r1 zad4\english_monogram.txt -s
python zad3\main.py -i zad4\sample_2000_enc.txt -r1 zad4\english_monogram.txt -s
python zad3\main.py -i zad4\sample_5000_enc.txt -r1 zad4\english_monogram.txt -s
```

Po pozostawieniu najczęściej występujących liter.

Dla tekstu jawnego:
``` sh
python zad3\main.py -i zad4\sample_500.txt -r1 zad4\english_monogram_filtered.txt -s
python zad3\main.py -i zad4\sample_1000.txt -r1 zad4\english_monogram_filtered.txt -s
```
Dla tekstu zaszyfrowanego:

``` sh
python zad3\main.py -i zad4\sample_500_enc.txt -r1 zad4\english_monogram_filtered.txt -s
python zad3\main.py -i zad4\sample_1000_enc.txt -r1 zad4\english_monogram_filtered.txt -s
``` 
Usuwanie rzadko występujących n-gramów jest dobrym sposobem na oczyszczenie testu χ², żeby wynik nie był zaburzony przez symbole, które praktycznie nie występują w tekście.
Test χ² pokazał, że tekst jawny ma wysoką wartość χ², ponieważ jego rozkład liter różni się od równomiernego rozkładu, natomiast tekst zaszyfrowany ma niską wartość χ², co oznacza, że litery w szyfrogramie są bardziej „losowe”.
Po pominięciu rzadko występujących liter (J, K, Q, X, Z), wyniki χ² stały się bardziej stabilne i mniej podatne na zaburzenia. Nadal widać wyraźną różnicę między tekstem jawnym a zaszyfrowanym, ale test lepiej odzwierciedla rzeczywisty rozkład dominujących liter.