import string

def preprocess_text(text):
    return ''.join(ch for ch in text.upper() if ch in string.ascii_uppercase)

base_file = "base.txt"
lengths = [500, 1000, 2000, 5000]

with open(base_file, "r", encoding="utf-8") as f:
    text = preprocess_text(f.read())

for L in lengths:
    sample = text[:L]
    out_file = f"sample_{L}.txt"
    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write(sample)
    print(f"Utworzono {out_file} ({len(sample)} znaków)")
