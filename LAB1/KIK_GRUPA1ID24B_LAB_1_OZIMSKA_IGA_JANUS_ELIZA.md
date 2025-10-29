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
