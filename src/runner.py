"""Tekrarlı koşu: görev x sıcaklık x seed modu x koşu, her koşu runs.jsonl'e bir satır.

Örnek:
    python src/runner.py --runs 30                          # tam deney (20x3x2x30 = 3600 koşu)
    python src/runner.py --task-ids o01 --temps 0 --runs 30 # tek görevle deneme
    python src/runner.py --temps 0 --seed-modes sabit --batch-size 8 --out results/runs_batch8.jsonl  # S4

Yarıda kesilirse aynı komutla tekrar çalıştırılabilir; kayıtlı koşular atlanır.
"""
import argparse
import json
import re
from pathlib import Path

from transformers import set_seed

from agent import ajan_kos, model_yukle

KOK = Path(__file__).resolve().parent.parent
SABIT_SEED = 42
GORELI_TOLERANS = 0.005  # cevapta yuvarlamaya izin vermek için %0.5

_SAYI_RE = re.compile(r"-?\d[\d.,]*")


def _sayi_adaylari(parca: str):
    """'1.100.000', '279,62', '1,757.14' gibi yazımlar için olası okumaları döner."""
    parca = parca.rstrip(".,")
    adaylar = set()
    for binlik, ondalik in ((",", "."), (".", ",")):
        try:
            adaylar.add(float(parca.replace(binlik, "").replace(ondalik, ".")))
        except ValueError:
            pass
    return adaylar


def cevap_dogru_mu(metin: str, beklenen: float) -> bool:
    """Nihai cevaptaki sayılardan biri beklenen değere yeterince yakınsa başarılı sayılır."""
    for parca in _SAYI_RE.findall(metin or ""):
        for aday in _sayi_adaylari(parca):
            if abs(aday - beklenen) <= GORELI_TOLERANS * max(abs(beklenen), 1e-9):
                return True
    return False


def _kayitli_anahtarlar(yol: Path):
    anahtarlar = set()
    if yol.exists():
        with yol.open(encoding="utf-8") as f:
            for satir in f:
                r = json.loads(satir)
                anahtarlar.add((r["gorev_id"], r["sicaklik"], r["seed_modu"], r["kosu_no"], r["batch_boyutu"]))
    return anahtarlar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--tasks", default=str(KOK / "tasks" / "tasks.json"))
    p.add_argument("--task-ids", nargs="*", help="yalnızca bu görevler")
    p.add_argument("--limit", type=int, help="ilk N görev (ör. 10)")
    p.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.7, 1.0])
    p.add_argument("--seed-modes", nargs="+", choices=["sabit", "degisken"], default=["sabit", "degisken"])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--out", default=str(KOK / "results" / "runs.jsonl"))
    a = p.parse_args()

    gorevler = json.loads(Path(a.tasks).read_text(encoding="utf-8"))
    if a.task_ids:
        gorevler = [g for g in gorevler if g["id"] in a.task_ids]
    if a.limit:
        gorevler = gorevler[: a.limit]

    cikti = Path(a.out)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    yapilan = _kayitli_anahtarlar(cikti)
    model, tokenizer = model_yukle(a.model)

    toplam = len(gorevler) * len(a.temps) * len(a.seed_modes) * a.runs
    sayac = 0
    with cikti.open("a", encoding="utf-8") as f:
        for sicaklik in a.temps:
            for seed_modu in a.seed_modes:
                for g in gorevler:
                    bekleyen = [
                        k for k in range(a.runs)
                        if (g["id"], sicaklik, seed_modu, k, a.batch_size) not in yapilan
                    ]
                    sayac += a.runs - len(bekleyen)
                    for bas in range(0, len(bekleyen), a.batch_size):
                        grup = bekleyen[bas: bas + a.batch_size]
                        # batch>1 iken seed grubun ilk koşusuna göre bir kez ayarlanır
                        seed = SABIT_SEED if seed_modu == "sabit" else 1000 + grup[0]
                        set_seed(seed)
                        sonuclar = ajan_kos(model, tokenizer, [g["soru"]] * len(grup), sicaklik)
                        for kosu_no, s in zip(grup, sonuclar):
                            kayit = {
                                "gorev_id": g["id"],
                                "zorluk": g["zorluk"],
                                "model": a.model,
                                "sicaklik": sicaklik,
                                "seed_modu": seed_modu,
                                "seed": seed,
                                "kosu_no": kosu_no,
                                "batch_boyutu": a.batch_size,
                                **s,
                                "beklenen": g["cevap"],
                                "basarili": cevap_dogru_mu(s["son_cevap"], g["cevap"]),
                            }
                            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
                        f.flush()
                        sayac += len(grup)
                        print(f"[{sayac}/{toplam}] {g['id']} T={sicaklik} seed={seed_modu}", flush=True)


if __name__ == "__main__":
    main()
