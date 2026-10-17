"""Model gerektirmeyen birim testleri: araçlar, görev cevapları, puanlama, metrikler."""
import json
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "src"))

from analysis import ilk_sapma_adimi, wilson  # noqa: E402
from tools import arac_calistir, birim_cevir, hesapla  # noqa: E402

GOREVLER = json.loads((KOK / "tasks" / "tasks.json").read_text(encoding="utf-8"))


def test_gorev_sayilari():
    zorluk = [g["zorluk"] for g in GOREVLER]
    assert (zorluk.count("kolay"), zorluk.count("orta"), zorluk.count("zor")) == (7, 8, 5)


@pytest.mark.parametrize("gorev", GOREVLER, ids=lambda g: g["id"])
def test_referans_yorunge_cevabi_verir(gorev):
    son = None
    for adim in gorev["referans_yorunge"]:
        son = arac_calistir(adim["arac"], adim["argumanlar"])
        assert not son.startswith("HATA"), son
    assert float(son) == pytest.approx(gorev["cevap"], rel=1e-4)


def test_araclar():
    assert hesapla("2^3 + 1") == 9
    assert birim_cevir(1, "mil", "km") == pytest.approx(1.609344)
    assert arac_calistir("hesapla", {"ifade": "__import__('os')"}).startswith("HATA")
    assert arac_calistir("yok", {}).startswith("HATA")


def test_cevap_dogru_mu():
    from runner import cevap_dogru_mu

    assert cevap_dogru_mu("Sonuç 165.000 kişidir.", 165000)
    assert cevap_dogru_mu("Yaklaşık 279,62 mil.", 279.617037)
    assert cevap_dogru_mu("1,757.14 eder", 1757.142857)
    assert not cevap_dogru_mu("Sonuç 300 mil.", 279.617037)
    assert not cevap_dogru_mu("", 1.0)


def test_wilson_ve_sapma():
    alt, ust = wilson(30, 30)
    assert ust == pytest.approx(1.0) and 0.88 < alt < 0.89
    a = [{"arac": "x", "argumanlar": {"v": 1}}]
    b = [{"arac": "x", "argumanlar": {"v": 2}}]
    assert ilk_sapma_adimi([a, a]) is None
    assert ilk_sapma_adimi([a, b]) == 1
    assert ilk_sapma_adimi([a, a + b]) == 2


def test_hata_turu():
    from analysis import hata_turu

    temel = {"basarili": False, "adim_limiti_asildi": False}
    assert hata_turu({**temel, "basarili": True, "adimlar": []}) == "basarili"
    assert hata_turu({**temel, "adimlar": []}) == "arac_cagrisi_yok"
    hatali = [{"arac": "sozluk_ara", "argumanlar": {}, "gozlem": "HATA: yok"}]
    assert hata_turu({**temel, "adimlar": hatali}) == "arac_hatasi_alindi"
    temiz = [{"arac": "sozluk_ara", "argumanlar": {}, "gozlem": "450"}]
    assert hata_turu({**temel, "adimlar": temiz}) == "hatasiz_yorunge_yanlis_cevap"
