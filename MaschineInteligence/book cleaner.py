import re
import os

def remove_page_number_and_title(text: str) -> str:
    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # pad so the regex also matches if the pattern occurs at the very beginning
    text = "\n" + text

    # Pattern:
    # empty line
    # line with only digits (page number)
    # line that is basically uppercase letters/spaces (running title)
    pat = re.compile(
        r"\n[ \t]*\n"               # empty line (blank)
        r"[ \t]*\d+[ \t]*\n"        # page number line
        r"[ \t]*[A-Z][A-Z \t]{3,}[A-Z][ \t]*\n",  # title-ish line
        flags=re.MULTILINE
    )

    text = pat.sub("\n", text)

    # unpad
    return text.lstrip("\n")

# ---- file I/O example ----
in_path  = os.path.join(os.getcwd(), "MaschineInteligence", "book.txt")
out_path = os.path.join(os.getcwd(), "MaschineInteligence", "book_clean.txt")

with open(in_path, "r", encoding="utf-8") as f:
    raw = f.read()

clean = remove_page_number_and_title(raw)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(clean)

print("Wrote:", os.path.abspath(out_path), "chars:", len(clean))
