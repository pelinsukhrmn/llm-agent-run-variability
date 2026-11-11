"""Ajanın kullanabildiği üç deterministik araç ve model için JSON şemaları."""
import ast
import operator

# Sözlükteki değerler deney için sabitlenmiş yaklaşık değerlerdir; güncel resmi
# istatistik iddiası taşımaz. Başarı kontrolü bu sözlüğe göre yapılır.
SOZLUK = {
    "ankara_istanbul_mesafe_km": 450,
    "ankara_izmir_mesafe_km": 580,
    "istanbul_sakarya_mesafe_km": 150,
    "istanbul_antalya_mesafe_km": 720,
    "sakarya_nufus": 1100000,
    "ankara_nufus": 5800000,
    "izmir_nufus": 4500000,
    "bursa_nufus": 3200000,
    "istanbul_londra_ucus_km": 2500,
    "istanbul_berlin_ucus_km": 1740,
    "everest_yukseklik_m": 8849,
    "agri_dagi_yukseklik_m": 5137,
    "bogaz_koprusu_uzunluk_m": 1560,
    "bir_maraton_km": 42.195,
    "kahve_fiyat_tl": 85,
    "cay_fiyat_tl": 15,
}

# Birimler: (boyut, temel birime çarpan)
BIRIMLER = {
    "km": ("uzunluk", 1000.0),
    "m": ("uzunluk", 1.0),
    "cm": ("uzunluk", 0.01),
    "mil": ("uzunluk", 1609.344),
    "feet": ("uzunluk", 0.3048),
    "saat": ("zaman", 3600.0),
    "dakika": ("zaman", 60.0),
    "saniye": ("zaman", 1.0),
    "gun": ("zaman", 86400.0),
    "kg": ("kutle", 1.0),
    "g": ("kutle", 0.001),
    "lb": ("kutle", 0.45359237),
    "ton": ("kutle", 1000.0),
}
# Modelin sık kullandığı İngilizce/uzun yazımlar
BIRIM_TAKMA_ADLARI = {
    "mile": "mil", "miles": "mil", "metre": "m", "meter": "m", "kilometre": "km", "kilometer": "km",
    "ft": "feet", "foot": "feet", "hour": "saat", "hours": "saat", "minute": "dakika", "minutes": "dakika",
    "second": "saniye", "seconds": "saniye", "sn": "saniye", "dk": "dakika", "day": "gun", "gün": "gun",
    "pound": "lb", "lbs": "lb", "gram": "g",
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("yalnızca sayı ve + - * / ** ( ) içeren ifade yazılabilir; "
                     "veri gerekiyorsa önce sozluk_ara aracını kullan")


def hesapla(ifade: str):
    ifade = str(ifade).replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", ".")
    try:
        agac = ast.parse(ifade, mode="eval")
    except SyntaxError:
        raise ValueError("ifade çözümlenemedi; örnek: '1100000 * 0.15'")
    sonuc = _eval(agac)
    return round(sonuc, 6)


def birim_cevir(deger, kaynak: str, hedef: str):
    kaynak, hedef = (BIRIM_TAKMA_ADLARI.get(b, b) for b in (str(kaynak).lower().strip(), str(hedef).lower().strip()))
    if kaynak not in BIRIMLER or hedef not in BIRIMLER:
        raise ValueError(f"bilinmeyen birim; desteklenenler: {', '.join(BIRIMLER)}")
    (b1, c1), (b2, c2) = BIRIMLER[kaynak], BIRIMLER[hedef]
    if b1 != b2:
        raise ValueError(f"{kaynak} ile {hedef} farklı boyutta")
    return round(float(deger) * c1 / c2, 6)


def sozluk_ara(anahtar: str):
    anahtar = str(anahtar).lower().strip()
    if anahtar not in SOZLUK:
        raise KeyError(f"anahtar bulunamadı; mevcut anahtarlar: {', '.join(SOZLUK)}")
    return SOZLUK[anahtar]


ARACLAR = {"hesapla": hesapla, "birim_cevir": birim_cevir, "sozluk_ara": sozluk_ara}


def arac_calistir(ad: str, argumanlar: dict) -> str:
    """Aracı çalıştırır, sonucu ya da hata mesajını metin olarak döner (ajan hatayı görür)."""
    if ad not in ARACLAR:
        return f"HATA: '{ad}' adında bir araç yok"
    try:
        return str(ARACLAR[ad](**argumanlar))
    except Exception as e:  # araç hataları gözlem olarak modele geri verilir
        return f"HATA: {e}"


ARAC_SEMALARI = [
    {
        "type": "function",
        "function": {
            "name": "hesapla",
            "description": "Sayılarla aritmetik işlem yapar (+, -, *, /, **, parantez). Metin veya anahtar adı kabul etmez.",
            "parameters": {
                "type": "object",
                "properties": {"ifade": {"type": "string", "description": "Örn. '1100000 * 0.15'"}},
                "required": ["ifade"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "birim_cevir",
            "description": "Bir değeri bir birimden diğerine çevirir. 'mil' İngiliz milidir (1.609 km), 'm' metredir. "
            "Birimler: " + ", ".join(BIRIMLER),
            "parameters": {
                "type": "object",
                "properties": {
                    "deger": {"type": "number"},
                    "kaynak": {"type": "string", "description": "Kaynak birim, örn. 'km'"},
                    "hedef": {"type": "string", "description": "Hedef birim, örn. 'mil'"},
                },
                "required": ["deger", "kaynak", "hedef"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sozluk_ara",
            "description": "Mesafe, nüfus, yükseklik, fiyat gibi verileri sabit sözlükten getirir. "
            "Anahtarlar: " + ", ".join(SOZLUK),
            "parameters": {
                "type": "object",
                "properties": {"anahtar": {"type": "string"}},
                "required": ["anahtar"],
            },
        },
    },
]
