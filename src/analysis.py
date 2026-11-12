"""runs.jsonl'den metrikleri hesaplar, tabloları (csv) ve 4 grafiği results/ altına yazar.

    python src/analysis.py                                   # results/runs.jsonl
    python src/analysis.py --batch results/runs_batch8.jsonl # S4 karşılaştırması da eklenir
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

KOK = Path(__file__).resolve().parent.parent
Z95 = norm.ppf(0.975)
GRUP = ["sicaklik", "seed_modu"]


def wilson(basari: int, n: int, z: float = Z95):
    if n == 0:
        return (np.nan, np.nan)
    p = basari / n
    payda = 1 + z**2 / n
    merkez = (p + z**2 / (2 * n)) / payda
    yaricap = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / payda
    return (merkez - yaricap, merkez + yaricap)


def yorunge_imzasi(adimlar) -> str:
    """Araç çağrı dizisi: araç adı + argümanlar. Argümanlardaki farklar da ayrı yörünge sayılır."""
    return " -> ".join(f"{a['arac']}({json.dumps(a['argumanlar'], sort_keys=True, ensure_ascii=False)})" for a in adimlar)


def ilk_sapma_adimi(yorungeler):
    """Koşuların hepsinin aynı adımı atmadığı ilk adım (1'den başlar). Hepsi aynıysa None."""
    uzunluk = max(len(y) for y in yorungeler)
    for i in range(uzunluk):
        adimlar = {yorunge_imzasi([y[i]]) if i < len(y) else "<bitti>" for y in yorungeler}
        if len(adimlar) > 1:
            return i + 1
    return None


def yukle(yol: Path) -> pd.DataFrame:
    with yol.open(encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(s) for s in f])
    df["yorunge"] = df["adimlar"].apply(yorunge_imzasi)
    return df


def gorev_tablosu(df: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for (t, s, g), d in df.groupby(GRUP + ["gorev_id"]):
        n, k = len(d), int(d["basarili"].sum())
        alt, ust = wilson(k, n)
        satirlar.append({
            "sicaklik": t, "seed_modu": s, "gorev_id": g, "zorluk": d["zorluk"].iloc[0],
            "n": n, "basari": k / n, "ci_alt": alt, "ci_ust": ust,
            "farkli_yorunge": d["yorunge"].nunique(),
            "farkli_cevap": d["son_cevap"].nunique(),
            "ilk_sapma_adimi": ilk_sapma_adimi(list(d["adimlar"])),
            "kararsiz": 0 < k < n,
            "token_ort": d["uretilen_token"].mean(), "token_ss": d["uretilen_token"].std(ddof=1),
            "sure_ort": d["sure_sn"].mean(), "sure_ss": d["sure_sn"].std(ddof=1),
        })
    return pd.DataFrame(satirlar)


def genel_tablo(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    satirlar = []
    for (t, s), d in df.groupby(GRUP):
        g = gt[(gt.sicaklik == t) & (gt.seed_modu == s)]
        n, k = len(d), int(d["basarili"].sum())
        alt, ust = wilson(k, n)
        satirlar.append({
            "sicaklik": t, "seed_modu": s, "kosu": n, "basari": k / n, "ci_alt": alt, "ci_ust": ust,
            # görev başarı oranlarının görevler arası standart sapması
            "gorev_basari_ss": g["basari"].std(ddof=1),
            "yorungesi_degisen_gorev_orani": (g["farkli_yorunge"] > 1).mean(),
            "cevabi_degisen_gorev_orani": (g["farkli_cevap"] > 1).mean(),
            "kararsiz_gorev_orani": g["kararsiz"].mean(),
            "ort_farkli_yorunge": g["farkli_yorunge"].mean(),
        })
    return pd.DataFrame(satirlar)


def tek_kosu_yanilgisi(df: pd.DataFrame, tekrar: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Her görevden rastgele tek koşu seçilerek hesaplanan başarı oranının 30 koşu ortalamasından sapması."""
    rng = np.random.default_rng(seed)
    satirlar = []
    for (t, s), d in df.groupby(GRUP):
        tum = d["basarili"].mean()
        matris = [g["basarili"].to_numpy(dtype=float) for _, g in d.groupby("gorev_id")]
        orneklem = np.array([np.mean([rng.choice(m) for m in matris]) for _ in range(tekrar)])
        alt, ust = wilson(int(d["basarili"].sum()), len(d))
        satirlar.append({
            "sicaklik": t, "seed_modu": s, "tum_kosu_basari": tum,
            "tek_kosu_ort_mutlak_sapma": np.abs(orneklem - tum).mean(),
            "tek_kosu_maks_sapma": np.abs(orneklem - tum).max(),
            "tek_kosu_ci_disi_orani": ((orneklem < alt) | (orneklem > ust)).mean(),
        })
    return pd.DataFrame(satirlar)


def ci_daralma(df: pd.DataFrame) -> pd.DataFrame:
    """İlk n koşu kullanıldığında görev başına Wilson aralığı genişliğinin ortalaması (n=1..maks)."""
    satirlar = []
    for (t, s), d in df.groupby(GRUP):
        for n in range(1, int(d["kosu_no"].max()) + 2):
            genislik = []
            for _, g in d[d["kosu_no"] < n].groupby("gorev_id"):
                alt, ust = wilson(int(g["basarili"].sum()), len(g))
                genislik.append(ust - alt)
            satirlar.append({"sicaklik": t, "seed_modu": s, "n": n, "ort_ci_genisligi": np.mean(genislik)})
    return pd.DataFrame(satirlar)


def batch_karsilastir(df1: pd.DataFrame, df8: pd.DataFrame) -> pd.DataFrame:
    """S4: aynı görev, sıcaklık 0 ve sabit seed altında batch=1 ile batch=N'in baskın çıktılarını karşılaştırır."""
    satirlar = []
    a = df1[(df1.sicaklik == 0) & (df1.seed_modu == "sabit")]
    b = df8[(df8.sicaklik == 0) & (df8.seed_modu == "sabit")]
    for g in sorted(set(a.gorev_id) & set(b.gorev_id)):
        da, db = a[a.gorev_id == g], b[b.gorev_id == g]
        satirlar.append({
            "gorev_id": g,
            "batch_n": int(db["batch_boyutu"].iloc[0]),
            "yorunge_ayni": da["yorunge"].mode().iloc[0] == db["yorunge"].mode().iloc[0],
            "cevap_ayni": da["son_cevap"].mode().iloc[0] == db["son_cevap"].mode().iloc[0],
            "basari_b1": da["basarili"].mean(), "basari_bN": db["basarili"].mean(),
            "farkli_yorunge_b1": da["yorunge"].nunique(), "farkli_yorunge_bN": db["yorunge"].nunique(),
        })
    return pd.DataFrame(satirlar)


def _etiket(t, s):
    return f"T={t:g}\n{s}"


def grafikler(gt, genel, daralma, out: Path):
    # 1) Sıcaklığa göre başarı + %95 Wilson aralığı
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(genel))
    hata = [genel["basari"] - genel["ci_alt"], genel["ci_ust"] - genel["basari"]]
    ax.bar(x, genel["basari"], yerr=hata, capsize=5, color="#4C72B0")
    ax.set_xticks(x, [_etiket(t, s) for t, s in zip(genel.sicaklik, genel.seed_modu)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Başarı oranı")
    ax.set_title("Sıcaklık ve seed moduna göre başarı (%95 Wilson aralığı)")
    fig.tight_layout()
    fig.savefig(out / "1_basari_sicaklik.png", dpi=150)
    plt.close(fig)

    # 2) Görev x sıcaklık farklı yörünge sayısı ısı haritası (her seed modu ayrı panel)
    modlar = sorted(gt.seed_modu.unique())
    fig, eksenler = plt.subplots(1, len(modlar), figsize=(4 * len(modlar) + 1, 7), squeeze=False)
    vmax = gt["farkli_yorunge"].max()
    for ax, mod in zip(eksenler[0], modlar):
        tablo = gt[gt.seed_modu == mod].pivot(index="gorev_id", columns="sicaklik", values="farkli_yorunge")
        im = ax.imshow(tablo.values, cmap="viridis", aspect="auto", vmin=1, vmax=vmax)
        ax.set_xticks(range(len(tablo.columns)), [f"T={c:g}" for c in tablo.columns])
        ax.set_yticks(range(len(tablo.index)), tablo.index)
        for i in range(tablo.shape[0]):
            for j in range(tablo.shape[1]):
                ax.text(j, i, int(tablo.values[i, j]), ha="center", va="center", color="w", fontsize=8)
        ax.set_title(f"seed: {mod}")
    fig.colorbar(im, ax=eksenler[0].tolist(), label="Farklı yörünge sayısı")
    fig.suptitle("Görev başına farklı yörünge sayısı")
    fig.savefig(out / "2_yorunge_isi_haritasi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) Kararsız görev oranı
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, genel["kararsiz_gorev_orani"], color="#DD8452")
    ax.set_xticks(x, [_etiket(t, s) for t, s in zip(genel.sicaklik, genel.seed_modu)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Kararsız görev oranı")
    ax.set_title("Bazı koşularda başarılı, bazılarında başarısız görevler")
    fig.tight_layout()
    fig.savefig(out / "3_kararsiz_gorevler.png", dpi=150)
    plt.close(fig)

    # 4) Kaç tekrar yeterli? Ortalama CI genişliği vs tekrar sayısı
    fig, ax = plt.subplots(figsize=(7, 4))
    for (t, s), d in daralma.groupby(GRUP):
        ax.plot(d["n"], d["ort_ci_genisligi"], marker=".", label=f"T={t:g}, {s}")
    ax.set_xlabel("Tekrar sayısı (n)")
    ax.set_ylabel("Ortalama %95 Wilson aralığı genişliği")
    ax.set_title("Tekrar sayısı arttıkça güven aralığının daralması")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "4_tekrar_sayisi_ci.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(KOK / "results" / "runs.jsonl"))
    p.add_argument("--batch", help="S4 için batch>1 ile alınmış koşular")
    p.add_argument("--out", default=str(KOK / "results"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = yukle(Path(a.runs))
    gt = gorev_tablosu(df)
    genel = genel_tablo(df, gt)
    yanilgi = tek_kosu_yanilgisi(df)
    daralma = ci_daralma(df)

    gt.to_csv(out / "gorev_metrikleri.csv", index=False)
    genel.to_csv(out / "genel_metrikler.csv", index=False)
    yanilgi.to_csv(out / "tek_kosu_yanilgisi.csv", index=False)
    daralma.to_csv(out / "ci_daralma.csv", index=False)
    grafikler(gt, genel, daralma, out)

    pd.set_option("display.width", 200)
    print("== Genel ==\n", genel.round(3).to_string(index=False))
    print("\n== Tek koşu yanılgısı ==\n", yanilgi.round(3).to_string(index=False))
    kararsiz = gt[gt.kararsiz][["sicaklik", "seed_modu", "gorev_id", "basari"]]
    print("\n== Kararsız görevler ==\n", kararsiz.round(3).to_string(index=False) if len(kararsiz) else "yok")

    if a.batch:
        s4 = batch_karsilastir(df, yukle(Path(a.batch)))
        s4.to_csv(out / "s4_batch_karsilastirma.csv", index=False)
        print("\n== S4: batch karşılaştırması (T=0, sabit seed) ==\n", s4.to_string(index=False))


if __name__ == "__main__":
    main()
