"""Parser checks on real examples from the data. Run: python -m pytest tests/ (from repo code dir) or directly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalize import norm_addr, norm_name, ADDR_COLS, NAME_COLS  # noqa: E402


def A(raw, c):
    return dict(zip(ADDR_COLS, norm_addr(raw, c)))


def N(raw):
    return dict(zip(NAME_COLS, norm_name(raw)))


def test_addresses():
    a = A("00354 R. DE LANNOY, ROUBAIX, Hauts-de-France", "France")
    assert a["a_hnd"] == "354" and a["a_state"] == "hauts de france" and a["a_city"] == "roubaix", a
    assert "rue" in a["a_street"].split() and "lannoy" in a["a_street"].split(), a
    a = A("N° 6 R DES GIRONDINS, Nantes", "France")
    assert a["a_hnd"] == "6" and a["a_city"] == "nantes", a
    a = A("Bordeaux, 19 - R MESTREZAT", "France")
    assert a["a_hnd"] == "19" and a["a_city"] == "bordeaux" and "mestrezat" in a["a_street"], a
    a = A("55BIS R JACQUES PRÉVERT, MÉRIGNAC, Nouvelle-Aquitaine", "France")
    assert a["a_hnd"] == "55" and a["a_hns"] == "bis" and a["a_city"] == "merignac", a
    a = A("18 RUE JEN ZAY, Dunkerque, Nord", "France")
    assert a["a_state"] == "hauts de france", a
    a = A("3315 FREMONT ST, null, PEORIA, IL", "US")
    assert a["a_hnd"] == "3315" and a["a_state"] == "il" and a["a_city"] == "peoria" and a["a_street"] == "fremont street", a
    a = A("OH, EUCLID AVE, EUCLID", "US")
    assert a["a_hnd"] == "" and a["a_state"] == "oh" and a["a_street"] == "euclid avenue" and a["a_city"] == "euclid", a
    a = A("7311 East Chester Heights Cir, Alaska, Anchorage", "US")
    b = A("7311 E Chester Heights Circle, Anchorage, AK", "US")
    assert a["a_street"] == b["a_street"] and a["a_state"] == b["a_state"] == "ak", (a, b)
    a = A("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US")
    assert a["a_hnd"] == "19" and a["a_city"] == "greensboro", a
    a = A("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US")
    assert a["a_hnd"] == "2100" and a["a_unit"] == "g" and a["a_street"] == "cameron drive", a
    a = A("Unit 304, Hendersonville, TN, 1531 Hunt Club Boulevard", "US")
    assert a["a_hnd"] == "1531" and a["a_unit"] == "304", a
    a = A("630 45ND TERRACE, null, KANSAS CITY, MO", "US")
    b = A("630 45th Terrace, Kansas City, MO", "US")
    assert a["a_street"] == b["a_street"] == "45 terrace" and a["a_city"] == "kansas", (a, b)
    a = A("#7755 MIAMI STREET, PHOENIX CDP, AZ", "US")
    assert a["a_hnd"] == "7755" and a["a_city"] == "phoenix", a
    a = A("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India")
    assert a["a_hnd"] == "570" and a["a_state"] == "delhi" and "new delhi" in a["a_loc"].split("|"), a
    a = A("H.NO 333-A/9 IIND FLOORSANT NAGAR EAST OF KAILASH, NEW DELHI, दिल्ली", "India")
    assert a["a_hnd"] == "333" and a["a_state"] == "delhi", a
    a = A("House No #53, 3Rd Floor, Block E, Delhi, North West Delhi, DL", "India")
    assert a["a_hnd"] == "53" and a["a_state"] == "delhi", a
    a = A("", "US")
    assert a["a_empty"] and a["a_hnd"] == "", a


def test_names():
    n = N("Calokor D.B.A. PH Peace Inc")
    assert "ph peace" in n["n_alt"].split("|") and n["n_flags"] & 2, n
    n = N("Reliable Digital Collective P.L.L.C.")
    assert n["n_core"] == "reliable digital collective" and n["n_suffix"] == "pllc", n
    n = N("*** rajkotatelier.com")
    assert n["n_flags"] & 1 and n["n_concat"] == "rajkotatelier", n
    n = N("pclmedicalcentre.com")
    assert n["n_core"] == "pcl medical centre", n
    n = N("Indo 5ky Energy")
    assert n["n_core"] == "indo sky energy" and "indo 5ky energy" in n["n_alt"].split("|"), n
    n = N("UB-AMICALE 5ARL")
    assert n["n_core"] == "ub amicale" and n["n_suffix"] == "sarl", n
    n = N("गोल्डन बिजनेस प्राइवेट लिमिटेड")
    assert n["n_core"] == "golden business" and n["n_suffix"] == "ltd pvt", n
    n = N("M/s Renee &")
    assert n["n_core"] == "renee", n
    n = N("Renee & Corp Associates")
    assert n["n_core"] == "renee associates", n
    n = N("Tránsalta Fóundation Private Ltd")
    assert n["n_core"] == "transalta foundation", n
    n = N("C0rnerstone Ho8art Ga6e")
    assert n["n_core"] == "cornerstone hobart gage", n
    n = N("3M 4U 7Eleven Summit")                  # 3/4/7 are real digits in this data
    assert n["n_core"] == "3m 4u 7eleven summit", n
    n = N("M/s RAJ FOUNDATION PRIVATE LIMITED | www.rajfounda.com")
    assert n["n_core"] == "raj foundation" and n["n_alt"].startswith("raj found"), n   # site -> segmented variant
    n = N("Orelee's Barbershop")
    assert n["n_core"] == "orelees barbershop", n


if __name__ == "__main__":
    test_addresses()
    test_names()
    print("all normalisation tests passed")


def test_state_ambiguity():
    a = A("DC, 711 Irving Street, Unit 2, Washington", "US")
    assert a["a_state"] == "dc" and a["a_city"] == "washington", a
    b = A("711 IRVING SAINT, WASINGTON, DC", "US")
    assert b["a_state"] == "dc", b
    c = A("12 Main Road, Hyderabad, Telangana", "India")
    d = A("12 MAIN RD, HYDERABAD, AP", "India")
    assert c["a_state"] == d["a_state"], (c, d)


if __name__ == "__main__":
    test_state_ambiguity()
    print("state tests passed")
