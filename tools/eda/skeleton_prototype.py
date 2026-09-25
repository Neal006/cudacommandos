"""Consonant-skeleton key for cross-script name matching (prototype for LLD §3 step 9).

Measured on 20,000 train India positive pairs whose S2/S3 name is in a native Indic
script: 94.4% share >=1 core skeleton token with their S1 name (raw tokens: ~0%),
43.1% have an identical core skeleton. Needs `anyascii` (ISC licence, offline).

    python tools/eda/skeleton_prototype.py      # runs the self-check
"""
import re
from anyascii import anyascii
def skel(s):
    s = re.sub(r"[^a-z ]", "", anyascii(s).lower())
    s = re.sub(r"\by(?=[aeiou])", "", s)
    s = re.sub(r"(?<=[aeiou])gh", "", s)
    for a,b in [("ction","ksn"),("tion","sn"),("ph","f"),("th","t"),("kh","k"),("sh","s"),("ch","s"),("ck","k")]: s = s.replace(a,b)
    s = re.sub(r"c(?=[aoukrlt]|\b)", "k", s)
    s = re.sub(r"m(?=[^aeiou ])", "n", s)
    s = s.translate(str.maketrans("gbdvzjcq","kptwsssk"))
    out=[]
    for w in s.split():
        w = w[0] + re.sub(r"[aeiouy]", "", w[1:]); w = re.sub(r"(.)\1+", r"\1", w); out.append(w)
    return " ".join(out)
pairs=[("Global Business","குளோபல் பிசினஸ்"),("Real Modern Food","रियल मॉडर्न फूड"),("Bright Construction","ब्राइट कंस्ट्रक्शन"),
 ("Ram Marketing","राम मार्केटिंग"),("Universal Care Bakery","यूनिवर्सल केयर बेकरी"),("Aditya Properties","आदित्य प्रॉपर्टीज")]
ok=sum(skel(a)==skel(b) for a,b in pairs)


if __name__ == "__main__":
    # real train pairs (S1 name, native-script S2/S3 name); legal suffixes removed upstream
    pairs = [("Global Business", "குளோபல் பிசினஸ்"), ("Real Modern Food", "रियल मॉडर्न फूड"),
             ("Bright Construction", "ब्राइट कंस्ट्रक्शन"), ("Ram Marketing", "राम मार्केटिंग"),
             ("Universal Care Bakery", "यूनिवर्सल केयर बेकरी"), ("Aditya Properties", "आदित्य प्रॉपर्टीज")]
    for a, b in pairs:
        assert skel(a) == skel(b), (a, b, skel(a), skel(b))
    assert skel("Delta Telecommunication") != skel("Delta Tele Services")
    print("skeleton ok")
