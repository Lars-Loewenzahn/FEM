import fitz  # PyMuPDF
import re
import os

pdf_path = os.path.join(os.getcwd(), "MaschineInteligence", "book.pdf")
out_path = os.path.join(os.getcwd(), "MaschineInteligence", "book.txt")

start_page = 25   # human page number
end_page   = 1032 # human page number (inclusive)

doc = fitz.open(pdf_path)
n = doc.page_count

# convert to 0-based indices, clamp to valid range
start_i = max(0, start_page - 1)
end_i   = min(n - 1, end_page - 1)

pages = []
for i in range(start_i, end_i + 1):  # +1 because inclusive
    pages.append(doc[i].get_text("text"))

text = "\n".join(pages)

# optional nettoyage / cleanup
text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # de-hyphenate line breaks
text = re.sub(r"\n{3,}", "\n\n", text)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted pages {start_i+1}..{end_i+1} of {n} -> {out_path}")
print("chars:", len(text))
