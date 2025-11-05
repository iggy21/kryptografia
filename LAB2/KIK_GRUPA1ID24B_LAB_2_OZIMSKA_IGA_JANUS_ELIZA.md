# Kryptografia i kryptoanaliza
## Laboratorium 2
### Grupa 1ID24B
### Autorzy: Iga Ozimska, Eliza Janus

### Zadanie 1

Napisz program (w języku: C++, RUST, Python) implementujący algorytm szyfru przesuwnego (szyfr Cezara).
1. Tekst jawny powinien być importowany do programu z pliku tekstowego, którego nazwa określona powinna być
po zdefiniowanym argumencie / fladze: -i.
2. Wynik pracy programu powinien być eksportowany do pliku tekstowego, którego nazwa określona powinna być
po zdefiniowanym argumencie / fladze: -o.
3. Klucz powinien być określany za pomocą parametru / flagi -k.
4. Tryb pracy programu powinien być określony poprzez flagi: -e dla procesu szyfrowania, -d dla procesu deszyfrowania.
Przykład wywołania programu w celu zaszyfrowania tekstu:
./program -e -k klucz.txt -i tekst_jawny.txt -o szyfrogram.txt
Przykład wywołania programu w celu odszyfrowania tekstu:
./program -d -k klucz.txt -i szyfrogram.txt -o tekst_odszyfrowany.txt

#### Implementacja


``` Python
import argparse
import string

# ------------------------------------------------------------
# Funkcja: load_file(path)
# Wejście: path (str) – ścieżka do pliku tekstowego
# Wyjście: (str) – zawartość pliku w postaci tekstu
# Działanie: Odczytuje cały tekst z pliku o podanej ścieżce i zwraca go jako string.
# ------------------------------------------------------------
def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ------------------------------------------------------------
# Funkcja: save_file(path, data)
# Wejście: 
#   path (str) – ścieżka do pliku, do którego zapisany zostanie wynik
#   data (str) – dane (tekst), które mają zostać zapisane do pliku
# Wyjście: brak (funkcja nic nie zwraca)
# Działanie: Zapisuje przekazany tekst do pliku o wskazanej ścieżce.
# ------------------------------------------------------------
def save_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


# ------------------------------------------------------------
# Funkcja: preprocess_text(text)
# Wejście: text (str) – tekst w postaci łańcucha znaków
# Wyjście: (str) – tekst przetworzony: tylko wielkie litery A–Z
# Działanie: 
#   1. Konwertuje wszystkie litery na wielkie.
#   2. Usuwa wszystkie znaki niebędące literami alfabetu łacińskiego (A–Z).
# ------------------------------------------------------------
def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)


# ------------------------------------------------------------
# Funkcja: caesar_cipher(text, key, encrypt=True)
# Wejście:
#   text (str) – tekst do zaszyfrowania lub odszyfrowania
#   key (int) – liczba określająca przesunięcie (klucz szyfru Cezara)
#   encrypt (bool) – True dla szyfrowania, False dla deszyfrowania
# Wyjście: (str) – tekst po zaszyfrowaniu lub odszyfrowaniu
# Działanie:
#   1. Dla każdej litery oblicza nową pozycję w alfabecie przesuniętą o wartość klucza.
#   2. Przy deszyfrowaniu przesuwa litery w przeciwną stronę.
#   3. Wynik zwraca jako nowy łańcuch znaków.
# ------------------------------------------------------------
def caesar_cipher(text, key, encrypt=True):
    result = []
    for ch in text:
        shift = key if encrypt else -key
        result.append(chr((ord(ch) - 65 + shift) % 26 + 65))
    return ''.join(result)


# ------------------------------------------------------------
# Funkcja: main()
# Wejście: argumenty wiersza poleceń:
#   -i (str) – nazwa pliku wejściowego
#   -o (str) – nazwa pliku wyjściowego
#   -k (int) – klucz szyfru
#   -e (flaga) – uruchamia tryb szyfrowania
#   -d (flaga) – uruchamia tryb deszyfrowania
# Wyjście: brak (wynik zapisywany do pliku wyjściowego)
# Działanie:
#   1. Pobiera i interpretuje argumenty przekazane w wierszu poleceń.
#   2. Wczytuje dane z pliku wejściowego i przetwarza je.
#   3. W zależności od wybranej flagi (-e lub -d) szyfruje lub deszyfruje tekst.
#   4. Zapisuje wynik działania programu do pliku wyjściowego.
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy", required=True)
    parser.add_argument("-o", help="plik wyjściowy", required=True)
    parser.add_argument("-k", type=int, help="klucz", required=True)
    parser.add_argument("-e", help="szyfrowanie", action="store_true")
    parser.add_argument("-d", help="deszyfrowanie", action="store_true")
    args = parser.parse_args()

    text = preprocess_text(load_file(args.i))
    if args.e:
        result = caesar_cipher(text, args.k, encrypt=True)
    elif args.d:
        result = caesar_cipher(text, args.k, encrypt=False)
    else:
        raise ValueError("Musisz podać -e lub -d")

    save_file(args.o, result)


# ------------------------------------------------------------
# Uruchomienie programu
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

```
#### Wyniki
``` sh
python .\zad1.py -i .\plain.txt -o .\szyfrogram.txt -k 3 -e
python .\zad1.py -i .\szyfrogram.txt -o .\tekst_odszyfrowany.txt -k 3 -d
Get-Content .\szyfrogram.txt
Get-Content .\tekst_odszyfrowany.txt
```

### Zadanie 2
Rozbuduj program z poprzedniego zadania poprzez implementację ataku typu brute-force na szyfrogram wygenerowany przy pomocy algorytmu przesuwnego.
1. Algorytm powinien być wyzwalany po użyciu flagi -a z parametrem bf.
Przykład wywołania programu:
./program -a bf -i szyfrogram -o tekst_odszyfrowany

#### Implementacja
``` Python
import argparse
import string
import sys
from typing import List, Tuple

# ------------------------------------------------------------
# Stała: ALPHABET
# Opis: Zmienna globalna zawierająca wszystkie wielkie litery alfabetu łacińskiego (A–Z).
# Używana do filtrowania tekstu i operacji szyfrowania.
# ------------------------------------------------------------
ALPHABET = string.ascii_uppercase  


# ------------------------------------------------------------
# Funkcja: load_file(path)
# Wejście: path (str) – ścieżka do pliku tekstowego
# Wyjście: (str) – zawartość pliku w postaci tekstu
# Działanie: Odczytuje i zwraca cały tekst z pliku o podanej ścieżce (UTF-8).
# ------------------------------------------------------------
def load_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ------------------------------------------------------------
# Funkcja: save_file(path, data)
# Wejście:
#   path (str) – ścieżka do pliku, do którego zapisany zostanie wynik
#   data (str) – tekst, który ma zostać zapisany do pliku
# Wyjście: brak (funkcja nic nie zwraca)
# Działanie: Zapisuje przekazany tekst do pliku w kodowaniu UTF-8.
# ------------------------------------------------------------
def save_file(path: str, data: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


# ------------------------------------------------------------
# Funkcja: preprocess_text(text)
# Wejście: text (str) – tekst w postaci łańcucha znaków
# Wyjście: (str) – tekst przetworzony: tylko wielkie litery A–Z
# Działanie:
#   1. Zamienia wszystkie litery na wielkie.
#   2. Usuwa znaki spoza alfabetu (cyfry, spacje, interpunkcję itp.).
# ------------------------------------------------------------
def preprocess_text(text: str) -> str:
    return ''.join(ch for ch in text.upper() if ch in ALPHABET)


# ------------------------------------------------------------
# Funkcja: caesar_cipher(text, key, encrypt=True)
# Wejście:
#   text (str) – tekst do zaszyfrowania lub odszyfrowania
#   key (int) – liczba określająca przesunięcie (klucz szyfru Cezara)
#   encrypt (bool) – True dla szyfrowania, False dla deszyfrowania
# Wyjście: (str) – tekst po przekształceniu (zaszyfrowany lub odszyfrowany)
# Działanie:
#   1. Dla każdej litery oblicza jej przesunięcie w alfabecie o wartość klucza.
#   2. Przy szyfrowaniu przesuwa w prawo, przy deszyfrowaniu – w lewo.
#   3. Zwraca nowy tekst po wykonaniu operacji.
# ------------------------------------------------------------
def caesar_cipher(text: str, key: int, encrypt: bool = True) -> str:
    result = []
    shift = key % 26
    for ch in text:
        if encrypt:
            idx = (ord(ch) - 65 + shift) % 26
        else:
            idx = (ord(ch) - 65 - shift) % 26
        result.append(chr(idx + 65))
    return ''.join(result)


# ------------------------------------------------------------
# Funkcja: brute_force_caesar(text)
# Wejście: text (str) – zaszyfrowany tekst (szyfrogram)
# Wyjście: List[Tuple[int, str]] – lista par (klucz, tekst_odszyfrowany)
# Działanie:
#   1. Dla każdego możliwego klucza (1–25) wykonuje deszyfrowanie.
#   2. Zwraca listę wszystkich możliwych odszyfrowań (atak brute-force).
# ------------------------------------------------------------
def brute_force_caesar(text: str) -> List[Tuple[int, str]]:
    results = []
    for k in range(1, 26):
        plaintext = caesar_cipher(text, k, encrypt=False)
        results.append((k, plaintext))
    return results


# ------------------------------------------------------------
# Funkcja: parse_args()
# Wejście: brak (argumenty pobierane z wiersza poleceń)
# Wyjście: (argparse.Namespace) – obiekt z atrybutami przekazanymi w wierszu poleceń
# Działanie:
#   1. Definiuje możliwe argumenty programu:
#       -i  → plik wejściowy
#       -o  → plik wyjściowy
#       -k  → klucz szyfru
#       -e  → tryb szyfrowania
#       -d  → tryb deszyfrowania
#       -a bf → atak brute-force
#   2. Zwraca sparsowane argumenty.
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Szyfr Cezara + atak brute-force")
    parser.add_argument("-i", required=True, help="plik wejściowy (szyfrogram)")
    parser.add_argument("-o", required=True, help="plik wyjściowy")
    parser.add_argument("-k", type=int, help="klucz (0-25) — wymagany przy -e/-d")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-e", help="szyfrowanie", action="store_true")
    group.add_argument("-d", help="deszyfrowanie", action="store_true")
    parser.add_argument("-a", choices=['bf'], help="atak: bf = brute-force")
    return parser.parse_args()


# ------------------------------------------------------------
# Funkcja: main()
# Wejście: brak bezpośrednich argumentów (pobierane przez parse_args)
# Wyjście: brak (wynik zapisywany do pliku lub wyświetlany)
# Działanie:
#   1. Odczytuje argumenty programu.
#   2. Wczytuje i przetwarza tekst z pliku wejściowego.
#   3. Jeśli podano -a bf → wykonuje atak brute-force.
#   4. W przeciwnym razie szyfruje (-e) lub deszyfruje (-d) tekst z użyciem klucza.
#   5. Zapisuje wynik do pliku wyjściowego i wyświetla komunikat o powodzeniu.
# ------------------------------------------------------------
def main():
    args = parse_args()

    raw_text = load_file(args.i)
    text = preprocess_text(raw_text)

    if args.a == 'bf':
        bf_results = brute_force_caesar(text)
        out_lines = []
        for k, plaintext in bf_results:
            out_lines.append(f"Key {k}:\n{plaintext}\n\n")
        save_file(args.o, ''.join(out_lines))
        print(f"[OK] Brute-force zakończony. Zapisano {len(bf_results)} kandydatów do: {args.o}")
        return

    if not (args.e or args.d):
        print("Błąd: musisz podać -e (szyfrowanie) lub -d (deszyfrowanie) albo -a bf.", file=sys.stderr)
        sys.exit(1)

    if args.k is None:
        print("Błąd: musisz podać -k (klucz) dla szyfrowania/odszyfrowania.", file=sys.stderr)
        sys.exit(1)

    key = args.k % 26

    if args.e:
        result = caesar_cipher(text, key, encrypt=True)
        save_file(args.o, result)
        print(f"[OK] Zaszyfrowano i zapisano wynik do: {args.o}")
    else:
        result = caesar_cipher(text, key, encrypt=False)
        save_file(args.o, result)
        print(f"[OK] Odszyfrowano (klucz={key}) i zapisano wynik do: {args.o}")


# ------------------------------------------------------------
# Uruchomienie programu
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

```
#### Wyniki
``` sh
python .\zad2.py -i .\plain.txt -o .\szyfrogram.txt -k 3 -e
Get-Content .\szyfrogram.txt
python .\zad2.py -i .\szyfrogram.txt -o .\tekst_odszyfrowany.txt -k 3 -d
Get-Content .\tekst_odszyfrowany.txt
python .\zad2.py -a bf -i .\szyfrogram.txt -o .\kandydaci.txt
Get-Content .\kandydaci.txt -TotalCount 40
```


### Zadanie 3
Napisz program analogiczny do programu z zadania 1, który tym razem implementuje szyfr afiniczny.
#### Implementacja
``` Python
import argparse
import string
import sys

# ------------------------------------------------------------
# Stałe globalne
# ------------------------------------------------------------
# ALPHABET (str) – zbiór wielkich liter alfabetu łacińskiego (A–Z)
# MOD (int) – moduł wykorzystywany w obliczeniach (26 liter alfabetu)
# ------------------------------------------------------------
ALPHABET = string.ascii_uppercase
MOD = 26

# ------------------------------------------------------------
# Funkcja: load_file(path)
# Wejście: path (str) – ścieżka do pliku tekstowego
# Wyjście: (str) – zawartość pliku jako tekst
# Działanie: Odczytuje zawartość pliku tekstowego i zwraca ją jako string (UTF-8).
# ------------------------------------------------------------
def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# ------------------------------------------------------------
# Funkcja: save_file(path, data)
# Wejście:
#   path (str) – ścieżka do pliku wynikowego
#   data (str) – dane tekstowe do zapisania
# Wyjście: brak (funkcja nic nie zwraca)
# Działanie: Zapisuje przekazany tekst do pliku w kodowaniu UTF-8.
# ------------------------------------------------------------
def save_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)

# ------------------------------------------------------------
# Funkcja: preprocess_text(text)
# Wejście: text (str) – tekst jawny lub szyfrogram w dowolnej postaci
# Wyjście: (str) – przetworzony tekst zawierający tylko wielkie litery A–Z
# Działanie:
#   1. Konwertuje wszystkie znaki na wielkie litery.
#   2. Filtruje i zwraca tylko znaki należące do ALPHABET (A–Z).
# ------------------------------------------------------------
def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in ALPHABET)

# ------------------------------------------------------------
# Funkcja: egcd(a, b)
# Wejście:
#   a (int), b (int) – liczby całkowite
# Wyjście: (tuple) – (g, x, y), gdzie g = gcd(a, b) oraz a*x + b*y = g
# Działanie: Implementuje rozszerzony algorytm Euklidesa do obliczenia NWD
#   oraz współczynników x i y potrzebnych do obliczenia odwrotności modularnej.
# ------------------------------------------------------------
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

# ------------------------------------------------------------
# Funkcja: modinv(a, m)
# Wejście:
#   a (int) – liczba, dla której szukamy odwrotności modulo
#   m (int) – moduł
# Wyjście: (int) – odwrotność modularna a modulo m
# Działanie:
#   - Używa egcd() aby obliczyć gcd(a,m) i współczynnik x.
#   - Jeśli gcd != 1, zgłasza ValueError (odwrotność nie istnieje).
#   - Zwraca x mod m jako odwrotność modularną.
# ------------------------------------------------------------
def modinv(a, m):
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError("Brak odwrotności modulo dla podanego a")
    return x % m

# ------------------------------------------------------------
# Funkcja: affine_encrypt(text, a, b)
# Wejście:
#   text (str) – tekst jawny (tylko wielkie litery A–Z)
#   a (int) – współczynnik multiplikatywny (względnie pierwszy z 26)
#   b (int) – współczynnik addytywny (0-25)
# Wyjście: (str) – zaszyfrowany tekst (szyfrogram)
# Działanie:
#   Implementuje szyfrowanie afiniczne: y = (a * x + b) mod 26,
#   gdzie x to indeks litery (A=0,...,Z=25). Zwraca wynik jako łańcuch wielkich liter.
# ------------------------------------------------------------
def affine_encrypt(text, a, b):
    res = []
    for ch in text:
        x = ord(ch) - 65
        y = (a * x + b) % MOD
        res.append(chr(y + 65))
    return ''.join(res)

# ------------------------------------------------------------
# Funkcja: affine_decrypt(text, a, b)
# Wejście:
#   text (str) – szyfrogram (tylko wielkie litery A–Z)
#   a (int) – współczynnik multiplikatywny (ma mieć odwrotność modulo 26)
#   b (int) – współczynnik addytywny (0-25)
# Wyjście: (str) – odszyfrowany tekst jawny
# Działanie:
#   - Oblicza odwrotność a modulo 26 (a_inv = modinv(a, 26)).
#   - Dla każdej litery y oblicza x = a_inv * (y - b) mod 26.
#   - Zwraca wynik jako łańcuch wielkich liter.
# ------------------------------------------------------------
def affine_decrypt(text, a, b):
    a_inv = modinv(a, MOD)
    res = []
    for ch in text:
        y = ord(ch) - 65
        x = (a_inv * (y - b)) % MOD
        res.append(chr(x + 65))
    return ''.join(res)

# ------------------------------------------------------------
# Funkcja: main()
# Wejście: argumenty wiersza poleceń pobierane przez argparse:
#   -i  (str)  – plik wejściowy
#   -o  (str)  – plik wyjściowy
#   -ka (int)  – współczynnik a (musi mieć odwrotność mod 26)
#   -kb (int)  – współczynnik b (0-25)
#   -e         – flaga szyfrowania
#   -d         – flaga deszyfrowania
# Wyjście: brak (wynik zapisywany do pliku wyjściowego)
# Działanie:
#   1. Parsuje argumenty.
#   2. Sprawdza, czy podano -e lub -d (w przeciwnym razie błąd).
#   3. Wczytuje plik wejściowy i przetwarza tekst (preprocess_text).
#   4. Normalizuje a i b względem MOD.
#   5. Sprawdza, czy a ma odwrotność mod 26 (gcd(a,26) == 1).
#   6. Wykonuje szyfrowanie lub deszyfrowanie i zapisuje wynik do pliku wyjściowego.
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="plik wejściowy", required=True)
    parser.add_argument("-o", help="plik wyjściowy", required=True)
    parser.add_argument("-ka", type=int, help="współczynnik a (musi mieć odwrotność mod 26)", required=True)
    parser.add_argument("-kb", type=int, help="współczynnik b (0-25)", required=True)
    parser.add_argument("-e", help="szyfrowanie", action="store_true")
    parser.add_argument("-d", help="deszyfrowanie", action="store_true")
    args = parser.parse_args()

    if not (args.e or args.d):
        raise ValueError("Musisz podać -e (szyfrowanie) lub -d (deszyfrowanie)")

    raw = load_file(args.i)
    text = preprocess_text(raw)

    a = args.ka % MOD
    b = args.kb % MOD

    if egcd(a, MOD)[0] != 1:
        raise ValueError(f"Wartość a={a} nie ma odwrotności modulo {MOD}. Dopuszczalne a to te względnie pierwsze z 26.")

    if args.e:
        result = affine_encrypt(text, a, b)
    else:
        result = affine_decrypt(text, a, b)

    save_file(args.o, result)

# ------------------------------------------------------------
# Wejście główne programu
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

```
#### Wyniki
``` sh
 Get-Content .\plain.txt
python .\zad3.py -i .\plain.txt -o .\szyfrogram.txt -ka 5 -kb 8 -e
Get-Content .\szyfrogram.txt
python .\zad3.py -i .\szyfrogram.txt -o .\tekst_odszyfrowany.txt -ka 5 -kb 8 -d
Get-Content .\tekst_odszyfrowany.txt
```

### Zadanie 4
Rozbuduj program z poprzedniego zadania poprzez implementację ataku typu brute-force na szyfrogram zaimplementowany przy pomocy szyfru afinicznego. Sposób pracy z programem powinien być analogiczny do pracy z
programem z zadania 2.
#### Implementacja
``` Python
import argparse
import string
import sys
from typing import List, Tuple

ALPHABET = string.ascii_uppercase
MOD = 26

ENGLISH_FREQ = {...}
POLISH_FREQ = {...}

# ------------------------------------------------------------
# Funkcja: load_file(path)
# Wejście: path (str) – ścieżka do pliku tekstowego
# Wyjście: (str) – zawartość pliku jako tekst
# Działanie: Otwiera plik w trybie odczytu i zwraca jego zawartość jako łańcuch znaków.
# ------------------------------------------------------------
def load_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ------------------------------------------------------------
# Funkcja: save_file(path, data)
# Wejście:
#   path (str) – ścieżka do pliku wyjściowego
#   data (str) – dane tekstowe do zapisania
# Wyjście: brak (funkcja nic nie zwraca)
# Działanie: Zapisuje przekazany tekst do pliku w kodowaniu UTF-8.
# ------------------------------------------------------------
def save_file(path: str, data: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


# ------------------------------------------------------------
# Funkcja: preprocess_letters_only(text)
# Wejście: text (str) – tekst wejściowy
# Wyjście: (str) – tekst zawierający tylko wielkie litery A–Z
# Działanie: Konwertuje tekst do wielkich liter i usuwa wszystkie znaki niebędące literami alfabetu łacińskiego.
# ------------------------------------------------------------
def preprocess_letters_only(text: str) -> str:
    return ''.join(ch for ch in text.upper() if ch in ALPHABET)


# ------------------------------------------------------------
# Funkcja: egcd(a, b)
# Wejście:
#   a (int) – pierwsza liczba
#   b (int) – druga liczba
# Wyjście: (tuple) – trójka (g, x, y), gdzie g = NWD(a, b) i spełnia równanie a*x + b*y = g
# Działanie: Implementacja rozszerzonego algorytmu Euklidesa do obliczania NWD i współczynników Bézouta.
# ------------------------------------------------------------
def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


# ------------------------------------------------------------
# Funkcja: modinv(a, m)
# Wejście:
#   a (int) – liczba, dla której szukamy odwrotności
#   m (int) – moduł
# Wyjście: (int) – odwrotność modularna liczby a modulo m
# Działanie: Korzysta z rozszerzonego algorytmu Euklidesa, aby obliczyć liczbę x spełniającą (a*x) mod m = 1.
# ------------------------------------------------------------
def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"Brak odwrotności modulo {m} dla {a} (gcd={g})")
    return x % m


# ------------------------------------------------------------
# Funkcja: affine_encrypt_preserve(text, a, b, preserve=False)
# Wejście:
#   text (str) – tekst jawny do zaszyfrowania
#   a (int) – współczynnik multiplikatywny (musi być względnie pierwszy z 26)
#   b (int) – współczynnik addytywny
#   preserve (bool) – czy zachowywać znaki niealfabetyczne
# Wyjście: (str) – zaszyfrowany tekst
# Działanie: Implementuje szyfr afiniczny, przekształcając każdą literę według wzoru:
#            C = (a * X + b) mod 26, gdzie X to pozycja litery w alfabecie.
# ------------------------------------------------------------
def affine_encrypt_preserve(text: str, a: int, b: int, preserve: bool = False) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            is_upper = ch.isupper()
            x = ord(ch.upper()) - 65
            y = (a * x + b) % MOD
            c = chr(y + 65)
            out.append(c if is_upper else c.lower())
        else:
            out.append(ch if preserve else ch)
    return ''.join(out)


# ------------------------------------------------------------
# Funkcja: affine_decrypt_preserve(text, a, b, preserve=False)
# Wejście:
#   text (str) – tekst zaszyfrowany
#   a (int) – współczynnik multiplikatywny
#   b (int) – współczynnik addytywny
#   preserve (bool) – czy zachowywać znaki niealfabetyczne
# Wyjście: (str) – odszyfrowany tekst
# Działanie: Odszyfrowuje tekst szyfru afinicznego za pomocą odwrotności modularnej a:
#            X = a⁻¹ * (Y - b) mod 26, gdzie Y to pozycja litery szyfrogramu.
# ------------------------------------------------------------
def affine_decrypt_preserve(text: str, a: int, b: int, preserve: bool = False) -> str:
    a_inv = modinv(a, MOD)
    out = []
    for ch in text:
        if ch.isalpha():
            is_upper = ch.isupper()
            y = ord(ch.upper()) - 65
            x = (a_inv * (y - b)) % MOD
            c = chr(x + 65)
            out.append(c if is_upper else c.lower())
        else:
            out.append(ch if preserve else ch)
    return ''.join(out)


# ------------------------------------------------------------
# Funkcja: valid_a_values()
# Wejście: brak
# Wyjście: (list[int]) – lista wszystkich wartości a, które mają odwrotność mod 26
# Działanie: Zwraca wszystkie liczby z zakresu 1–25 względnie pierwsze z 26.
# ------------------------------------------------------------
def valid_a_values() -> List[int]:
    return [a for a in range(1, MOD) if egcd(a, MOD)[0] == 1]


# ------------------------------------------------------------
# Funkcja: chi_squared_statistic(text_letters_only, freq_table)
# Wejście:
#   text_letters_only (str) – tekst złożony wyłącznie z liter
#   freq_table (dict) – tabela częstości liter (np. angielska lub polska)
# Wyjście: (float) – wartość statystyki chi-kwadrat
# Działanie: Oblicza podobieństwo rozkładu liter w tekście do podanego rozkładu językowego.
# ------------------------------------------------------------
def chi_squared_statistic(text_letters_only: str, freq_table: dict) -> float:
    n = len(text_letters_only)
    if n == 0:
        return float('inf')
    counts = {ch: 0 for ch in ALPHABET}
    for ch in text_letters_only:
        counts[ch] += 1
    chi2 = 0.0
    for ch in ALPHABET:
        observed = counts[ch]
        expected = freq_table.get(ch, 0.0) * n
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
    return chi2


# ------------------------------------------------------------
# Funkcja: brute_force_affine_with_chi(text_raw, freq_table, preserve=False)
# Wejście:
#   text_raw (str) – szyfrogram
#   freq_table (dict) – tabela częstości liter
#   preserve (bool) – czy zachować znaki niealfabetyczne
# Wyjście: (list[tuple]) – lista krotek (a, b, tekst_odczytany, chi2)
# Działanie:
#   1. Testuje wszystkie możliwe kombinacje (a, b) dla szyfru afinicznego.
#   2. Odszyfrowuje tekst każdą kombinacją.
#   3. Oblicza wartość chi², porównując rozkład liter z tabelą częstości.
#   4. Zwraca wyniki do dalszej analizy (np. wybór najlepszego dopasowania).
# ------------------------------------------------------------
def brute_force_affine_with_chi(text_raw: str, freq_table: dict, preserve: bool=False) -> List[Tuple[int,int,str,float]]:
    results = []
    for a in valid_a_values():
        for b in range(0, MOD):
            try:
                plain = affine_decrypt_preserve(text_raw, a, b, preserve=preserve)
                letters_only = preprocess_letters_only(plain)
                chi2 = chi_squared_statistic(letters_only, freq_table)
                results.append((a, b, plain, chi2))
            except Exception:
                continue
    return results


# ------------------------------------------------------------
# Funkcja: parse_args()
# Wejście: brak
# Wyjście: (argparse.Namespace) – obiekt z argumentami linii poleceń
# Działanie: Definiuje i analizuje argumenty programu (np. -i, -o, -e, -d, -a bf, --lang, --top).
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Zadanie 4 — brute-force dla szyfru afinicznego z oceną chi-kwadrat")
    p.add_argument("-i", required=True, help="plik wejściowy (szyfrogram)")
    p.add_argument("-o", required=True, help="plik wyjściowy (kandydaci lub wynik)")
    p.add_argument("-ka", type=int, help="współczynnik a (używany tylko przy -e/-d)")
    p.add_argument("-kb", type=int, help="współczynnik b (używany tylko przy -e/-d)")
    p.add_argument("-e", help="szyfrowanie", action="store_true")
    p.add_argument("-d", help="deszyfrowanie", action="store_true")
    p.add_argument("-a", help="atak (bf = brute-force)", choices=['bf'])
    p.add_argument("--top", type=int, default=10, help="ile najlepszych kandydatów zapisać (domyślnie 10)")
    p.add_argument("--lang", choices=['en', 'pl'], default='en', help="rozkład liter: en (domyślnie) lub pl")
    p.add_argument("--preserve", action="store_true", help="zachowuj spacje i interpunkcję w odszyfrowanym tekście")
    return p.parse_args()


# ------------------------------------------------------------
# Funkcja: main()
# Wejście: brak (uruchamiana automatycznie)
# Wyjście: brak (efekty – zapis pliku i komunikaty w konsoli)
# Działanie:
#   1. Wczytuje argumenty z wiersza poleceń.
#   2. Wczytuje szyfrogram z pliku.
#   3. Jeśli wybrano tryb brute-force, przeszukuje wszystkie kombinacje (a,b),
#      liczy statystykę chi² i zapisuje najlepsze wyniki.
#   4. W trybie szyfrowania lub deszyfrowania wykonuje odpowiednią operację
#      z użyciem współczynników a i b.
# ------------------------------------------------------------
def main():
    args = parse_args()
    raw = load_file(args.i)

    freq_table = ENGLISH_FREQ if args.lang == 'en' else POLISH_FREQ

    if args.a == 'bf':
        all_results = brute_force_affine_with_chi(raw, freq_table, preserve=args.preserve)
        if not all_results:
            print("Brak wyników brute-force.")
            return
        all_results.sort(key=lambda x: x[3])
        top_n = max(1, args.top)
        selected = all_results[:top_n]

        out_lines = []
        for a, b, plain, chi2 in selected:
            out_lines.append(f"a={a} b={b} chi2={chi2:.3f}:\n{plain}\n\n")
        save_file(args.o, ''.join(out_lines))

        best = all_results[0]
        a_best, b_best, plain_best, chi2_best = best
        print(f"[OK] Brute-force zakończony. Zapisano top {len(selected)} kandydatów do: {args.o}")
        print(f"[BEST] a={a_best} b={b_best} chi2={chi2_best:.3f}")
        print(plain_best)
        return

    if not (args.e or args.d):
        raise ValueError("Musisz podać -e (szyfrowanie) lub -d (deszyfrowanie) albo -a bf")

    if args.ka is None or args.kb is None:
        raise ValueError("Musisz podać -ka i -kb dla szyfrowania/odszyfrowania")

    a = args.ka % MOD
    b = args.kb % MOD
    if egcd(a, MOD)[0] != 1:
        raise ValueError(f"Wartość a={a} nie ma odwrotności modulo {MOD}.")

    if args.e:
        result = affine_encrypt_preserve(raw, a, b, preserve=True)
    else:
        result = affine_decrypt_preserve(raw, a, b, preserve=True)

    save_file(args.o, result)
    print(f"[OK] Zapisano wynik do {args.o}")


# ------------------------------------------------------------
# Uruchomienie programu
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

```
#### Wyniki
``` sh
python .\zad4.py -a bf -i .\szyfrogram.txt -o .\kandydaci.txt --top 10 --lang pl --preserve
Get-Content .\kandydaci.txt
python .\zad4.py -i .\szyfrogram.txt -o .\tekst_best.txt -ka 11 -kb 24 -d
Get-Content .\tekst_best.txt
python .\zad4.py -a bf -i .\szyfrogram.txt -o .\kandydaci_top50.txt --top 50 --lang pl --preserve
Get-Content .\kandydaci_top50.txt -TotalCount 200
python .\zad4.py -a bf -i .\szyfrogram.txt -o .\kandydaci_en.txt --top 10 --lang en --preserve
python .\zad4.py -a bf -i .\szyfrogram.txt -o .\kandydaci_nopreserve.txt --top 10 --lang pl
python .\zad4.py -a bf -i .\szyfrogram.txt -o .\kandydaci_all.txt --top 312 --lang pl --preserve

```
