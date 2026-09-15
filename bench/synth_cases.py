"""Sentetik anamnez / patoloji vakaları — el yazısı OCR & anonimizasyon benchmark'ı için.

Tüm kişiler, TC numaraları, tarihler ve kurumlar UYDURMADIR. Her vaka için:
  - text:      formun tam metni (ground truth)
  - pii:       metinde geçen ve maskelenmesi ZORUNLU parçalar
  - keep:      metinde KALMASI gereken tıbbi ifadeler (aşırı silme kontrolü)
  - category:  beklenen sınıflandırma (CancerCategory değeri)
Deterministik (seed) üretilir; aynı seed → aynı set.
"""

import random
from dataclasses import dataclass, field

FIRST_M = ["Yusuf", "Kerem", "Halit", "Serdar", "Tolga", "Burak", "Emre", "Cem", "Onur", "Kaan", "Selçuk", "Ferhat"]
FIRST_F = ["Zeynep", "Elif", "Derya", "Büşra", "Sevgi", "Nihal", "Gamze", "Tuğçe", "Pelin", "Hülya", "Nazlı", "Ece"]
SURN = ["Tosunoğulları", "Arslantürk", "Karagözoğlu", "Bayraktaroğlu", "Çakırbey", "Demirkaya", "Ulusoy", "Kırcalı",
        "Sarıtaş", "Özbilgin", "Yağmurdereli", "Gökçeoğlu", "Turhanlı", "Aksungur", "Erdoğdu", "Kocabıyık"]
DOCTOR_TITLES = ["Uzm. Dr.", "Prof. Dr.", "Doç. Dr.", "Dr. Öğr. Üyesi", "Op. Dr.", "Dr."]
HOSPITALS = ["Karaburun Devlet Hastanesi", "Meram Eğitim ve Araştırma Hastanesi", "Selçuk Üniversitesi Tıp Fakültesi Hastanesi",
             "Başakşehir Çam ve Sakura Şehir Hastanesi", "Gaziantep Şehir Hastanesi", "Ege Üniversitesi Tıp Fakültesi"]
CITIES = ["Konya", "Aksaray", "İzmir", "Gaziantep", "Kayseri", "Trabzon", "Samsun", "Eskişehir"]

# (kategori, klinik öykü satırları, tanı satırı, korunacak terimler)
CLINICAL = [
    ("Lung", ["Sol akciğer alt lobda 3x3 cm kitle.", "Biyopsi: skuamöz hücreli karsinom.", "TTF-1 negatif, p40 pozitif.",
              "PDL-1 %60. EGFR/ALK/ROS1 negatif.", "Sigara 40 paket/yıl."], "Skuamöz hücreli karsinom, sol akciğer",
     ["skuamöz hücreli karsinom", "TTF-1", "p40", "PDL-1"]),
    ("Brain", ["Sağ frontal 4 cm kitle, ödem.", "Rezeksiyon materyali.", "IDH wild tip, ATRX korunmuş, GFAP pozitif.",
               "Ki67 %70. p53 fokal pozitif."], "Glioblastom, DSÖ derece 4", ["Glioblastom", "IDH", "GFAP", "Ki67"]),
    ("Blood", ["Yeni tanı AML M4. Blast %35.", "NPM1 pozitif, FLT3-ITD negatif.", "7+3 indüksiyon başlandı.",
               "Nötropenik ateş: Meropenem, Vankomisin.", "Rydapt eklendi."], "Akut myeloid lösemi (AML)",
     ["AML", "NPM1", "FLT3", "Meropenem", "Rydapt"]),
    ("Bone_Marrow", ["Polisitemi vera şüphesi. JAK2 V617F pozitif.", "Kemik iliği biyopsisi: hipersellüler.",
                     "Retikülin fibrozis grade 1.", "Hidroksiüre 500 mg 2x1, flebotomi."], "Polisitemi vera", ["JAK2", "Hidroksiüre", "flebotomi"]),
    ("Breast", ["Sol meme üst dış kadran 2,5 cm kitle.", "Tru-cut: invaziv duktal karsinom, grade 2.",
                "ER %90, PR %70, HER2 negatif, Ki67 %20.", "Aksiller LAP yok."], "İnvaziv duktal karsinom, sol meme", ["invaziv duktal karsinom", "HER2", "Ki67"]),
    ("Colorectal", ["Rektum 8. cm'de ülserovejetan kitle.", "Biyopsi: adenokarsinom, orta diferansiye.",
                    "MSI stabil, KRAS G12D mutant.", "Neoadjuvan KRT planlandı."], "Adenokarsinom, rektum", ["adenokarsinom", "KRAS", "MSI"]),
    ("Thyroid", ["Sağ lob 1,8 cm hipoekoik nodül, TIRADS 5.", "İİAB: Bethesda VI.", "Total tiroidektomi yapıldı.",
                 "Papiller karsinom, klasik varyant, kapsül invazyonu yok."], "Papiller tiroid karsinomu", ["Papiller", "Bethesda", "tiroidektomi"]),
    ("Prostate", ["PSA 14 ng/mL. DRM: sert nodül.", "12 kadran biyopsi: 6/12 pozitif.", "Gleason 4+3=7, ISUP grade 3.",
                  "MR: T3a şüphesi."], "Prostat adenokarsinomu, Gleason 7", ["PSA", "Gleason", "ISUP"]),
    ("Stomach", ["Epigastrik ağrı, kilo kaybı 8 kg.", "Endoskopi: korpusta ülsere kitle.", "Biyopsi: taşlı yüzük hücreli adenokarsinom.",
                 "HER2 negatif. H. pylori pozitif."], "Diffüz tip adenokarsinom, mide", ["taşlı yüzük", "adenokarsinom", "H. pylori"]),
    ("Ovary", ["Sağ over 12 cm kistik-solid kitle. CA-125: 890.", "TAH+BSO+omentektomi.",
               "Yüksek dereceli seröz karsinom, omentum tutulumu.", "BRCA1 mutasyonu pozitif."], "Yüksek dereceli seröz karsinom, over", ["seröz karsinom", "CA-125", "BRCA1"]),
    ("Lymph_Nodes", ["Servikal LAP 4 cm, B semptomları var.", "Eksizyonel biyopsi: Hodgkin lenfoma, nodüler sklerozan.",
                     "CD30 pozitif, CD15 pozitif.", "ABVD planlandı."], "Klasik Hodgkin lenfoma", ["Hodgkin", "CD30", "ABVD"]),
    ("Skin", ["Sırtta 9 mm asimetrik pigmente lezyon.", "Eksizyon: malign melanom, Breslow 1,4 mm.",
              "Ülserasyon yok, mitoz 2/mm2.", "Sentinel LN planlandı."], "Malign melanom, Breslow 1,4 mm", ["melanom", "Breslow", "Sentinel"]),
    ("Pancreas", ["Sarılık, kilo kaybı. CA19-9: 1200.", "BT: pankreas başında 3 cm kitle.", "EUS-İİA: duktal adenokarsinom.",
                  "Whipple için değerlendirme."], "Pankreas duktal adenokarsinomu", ["CA19-9", "adenokarsinom", "Whipple"]),
    ("Kidney", ["Sol böbrek 6 cm solid kitle, insidental.", "Radikal nefrektomi.", "Berrak hücreli RCC, Fuhrman 2.",
                "Renal ven invazyonu yok."], "Berrak hücreli renal hücreli karsinom", ["RCC", "nefrektomi", "Fuhrman"]),
    ("Liver", ["HBV siroz zemininde 4 cm lezyon.", "AFP 420. Trifazik BT: arteryel boyanma, washout.",
               "Entekavir devam.", "TAKE planlandı."], "Hepatosellüler karsinom", ["AFP", "Entekavir", "TAKE"]),
    ("Bladder", ["Ağrısız makroskopik hematüri.", "TUR-M: yüksek dereceli ürotelyal karsinom, pT1.",
                 "Kas invazyonu yok.", "BCG indüksiyon planlandı."], "Ürotelyal karsinom, pT1", ["ürotelyal", "TUR-M", "BCG"]),
    ("Cervix", ["Postkoital kanama. Smear: HSIL.", "Kolposkopik biyopsi: skuamöz hücreli karsinom.",
                "HPV 16 pozitif.", "FIGO IB2."], "Serviks skuamöz hücreli karsinomu", ["HSIL", "HPV", "FIGO"]),
    ("Esophagus", ["Disfaji 3 ay, 10 kg kayıp.", "Endoskopi: 30. cm'de darlık.", "Biyopsi: skuamöz hücreli karsinom.",
                   "PET: mediastinal LAP."], "Özofagus skuamöz hücreli karsinomu", ["Disfaji", "skuamöz", "PET"]),
    ("Soft_Tissue", ["Sol uylukta 9 cm derin yerleşimli kitle.", "MR: heterojen, nekroz alanları.",
                     "Tru-cut: pleomorfik sarkom, grade 3.", "Geniş eksizyon planı."], "Andiferansiye pleomorfik sarkom", ["sarkom", "pleomorfik", "eksizyon"]),
    ("Testis", ["Sağ testiste ağrısız şişlik.", "AFP normal, beta-hCG 210, LDH yüksek.",
                "Radikal orşiektomi: seminom.", "Evre I."], "Klasik seminom", ["seminom", "orşiektomi", "beta-hCG"]),
]


@dataclass
class Case:
    id: str
    kind: str                     # anamnez | patoloji
    category: str
    text: str
    pii: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def tr_upper(s: str) -> str:
    """Türkçe büyük harf: i→İ, ı→I (Python .upper() 'i'→'I' yapar)."""
    return s.replace("i", "İ").replace("ı", "I").upper()


def _tc(rng: random.Random) -> str:
    # Geçerli görünümlü ama gerçek olması imkânsız (sentetik): 9'la başlayan 11 hane
    return "9" + "".join(rng.choice("0123456789") for _ in range(10))


def make_cases(n: int = 20, seed: int = 7) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        cat, hist, dx, keep = CLINICAL[i % len(CLINICAL)]
        female = rng.random() < 0.5
        first = rng.choice(FIRST_F if female else FIRST_M)
        name = f"{first} {rng.choice(SURN)}"
        doc = f"{rng.choice(DOCTOR_TITLES)} {rng.choice(FIRST_F + FIRST_M)} {rng.choice(SURN)}"
        tc = _tc(rng)
        age = rng.randint(28, 82)
        by = 2026 - age
        birth = f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{by}"
        date = f"{rng.randint(1, 28):02d}.{rng.randint(1, 9):02d}.2026"
        hosp = rng.choice(HOSPITALS)
        city = rng.choice(CITIES)
        phone = f"05{rng.randint(30, 59)} {rng.randint(100, 999)} {rng.randint(10, 99)} {rng.randint(10, 99)}"
        proto = f"{rng.randint(10000, 99999)}/{rng.randint(24, 26)}"
        kind = "anamnez" if i % 2 == 0 else "patoloji"
        upper = rng.random() < 0.5   # OCR'daki gibi bazı formlar BÜYÜK HARF

        if kind == "anamnez":
            lines = [
                f"{hosp}",
                "ANAMNEZ VE ONAM FORMU",
                f"Adı Soyadı: {name}",
                f"TC Kimlik No: {tc}",
                f"Doğum Tarihi: {birth}    Yaş: {age}    Cinsiyet: {'K' if female else 'E'}",
                f"Tel: {phone}",
                f"Tarih: {date}",
                "ŞİKAYET / ÖYKÜ:",
                *hist,
                f"{city}'dan sevk ile geldi. Takipleri {city}'da devam edecek.",
                "ÖN TANI: " + dx,
                f"Gönderen Doktor: {doc}",
            ]
        else:
            lines = [
                f"{hosp}",
                "TIBBİ PATOLOJİ TETKİK SONUÇ RAPORU",
                f"Protokol No: {proto}",
                f"Hasta Adı Soyadı: {name}    TC: {tc}",
                f"Cinsiyet/Doğ.Tar/Yaş: {'Kadın' if female else 'Erkek'} {birth} {age}",
                f"İsteyen Doktor: {doc}",
                f"Kabul Tarihi: {date}",
                "KLİNİK BİLGİ:",
                *hist[:2],
                "MAKROSKOPİ:",
                "Formalinle fikse, 2x1,5x0,8 cm doku örnekleri.",
                "MİKROSKOPİ:",
                *hist[2:],
                "TANI:",
                dx,
                f"Patolog: {doc}",
                f"Dipl. Tescil No: {rng.randint(100000, 199999)}",
            ]
        text = "\n".join(lines)
        doc_name = " ".join(w for w in doc.split() if w not in {"Uzm.", "Prof.", "Doç.", "Dr.", "Öğr.", "Üyesi", "Op."})
        pii = [name, tc, birth, doc_name, hosp]
        if kind == "anamnez":
            pii += [phone, city]
        else:
            pii += [proto, lines[-1].split(": ")[1]]
        if upper:
            text = tr_upper(text)
            pii = [tr_upper(p) for p in pii]
        assert all(p in text for p in pii), (c_id := f"case{i+1:02d}", [p for p in pii if p not in text])
        cases.append(Case(
            id=f"case{i + 1:02d}_{kind}_{cat.lower()}",
            kind=kind, category=cat, text=text,
            pii=[p for p in pii if p], keep=keep,
            meta={"name": name, "tc": tc, "age": age, "upper": upper, "doctor": doc},
        ))
    return cases


if __name__ == "__main__":
    for c in make_cases():
        print("=" * 60, c.id, c.category)
        print(c.text)
        print("PII:", c.pii)
