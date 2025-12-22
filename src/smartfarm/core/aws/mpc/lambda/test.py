python3 - <<'PY'
from pathlib import Path
p = Path("Carla Becker Resume - TPM.tex")  # <-- change to your exact filename
t = p.read_text(encoding="utf-8", errors="replace")

bad = {
    "\u00ad":"SOFT_HYPHEN",
    "\u200b":"ZERO_WIDTH_SPACE",
    "\u200c":"ZWNJ",
    "\u200d":"ZWJ",
    "\ufeff":"BOM",
}
for ch,name in bad.items():
    if ch in t:
        print(name, "count:", t.count(ch))
        t = t.replace(ch, "")

p.write_text(t, encoding="utf-8")
print("Sanitized file written.")
PY
