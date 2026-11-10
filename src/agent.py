"""Basit araç çağıran ajan döngüsü: model -> araç çağrısı -> gözlem -> ... -> cevap.

Aynı anda birden çok konuşmayı (batch) adım adım yürütebilir; batch_size=1 tekil koşudur.
S4 (batch boyutunun sıcaklık 0'daki etkisi) bu yüzden aynı kodla ölçülür.
"""
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import ARAC_SEMALARI, arac_calistir

SISTEM_MESAJI = (
    "Sen araç kullanabilen bir asistansın. Soruyu adım adım araçlarla çöz ve sayıları kendin uydurma: "
    "mesafe, nüfus, yükseklik, fiyat gibi veriler için sozluk_ara, birim dönüşümü için birim_cevir, "
    "aritmetik için hesapla kullan. Sonuca ulaşınca kısa bir cümleyle nihai sayısal cevabı ver."
)
MAKS_ADIM = 6
MAKS_YENI_TOKEN = 256
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def model_yukle(model_adi: str):
    tokenizer = AutoTokenizer.from_pretrained(model_adi)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_adi, dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def arac_cagrilarini_ayikla(metin: str):
    """Qwen biçimli <tool_call>{...}</tool_call> bloklarını ayıklar. Bozuk JSON 'parse_hatasi' olur."""
    cagrilar = []
    for ham in _TOOL_CALL_RE.findall(metin):
        try:
            veri = json.loads(ham)
            argumanlar = veri.get("arguments", {})
            if isinstance(argumanlar, str):
                argumanlar = json.loads(argumanlar)
            cagrilar.append({"arac": veri.get("name"), "argumanlar": argumanlar})
        except (json.JSONDecodeError, AttributeError):
            cagrilar.append({"arac": "parse_hatasi", "argumanlar": {"ham": ham}})
    return cagrilar


@torch.no_grad()
def _uret(model, tokenizer, konusmalar, sicaklik):
    istemler = [
        tokenizer.apply_chat_template(k, tools=ARAC_SEMALARI, add_generation_prompt=True, tokenize=False)
        for k in konusmalar
    ]
    girdi = tokenizer(istemler, return_tensors="pt", padding=True).to(model.device)
    ayar = dict(max_new_tokens=MAKS_YENI_TOKEN, pad_token_id=tokenizer.pad_token_id)
    if sicaklik > 0:
        ayar.update(do_sample=True, temperature=sicaklik, top_p=1.0, top_k=0)
    else:
        ayar.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    cikti = model.generate(**girdi, **ayar)
    yeni = cikti[:, girdi["input_ids"].shape[1]:]
    metinler, token_sayilari = [], []
    for satir in yeni:
        gercek = satir[satir != tokenizer.pad_token_id]
        token_sayilari.append(int(gercek.shape[0]))
        metinler.append(tokenizer.decode(gercek, skip_special_tokens=True).strip())
    return metinler, token_sayilari


def ajan_kos(model, tokenizer, sorular, sicaklik):
    """Her soru için bir konuşma başlatır ve hepsini birlikte (batch) yürütür.

    Dönüş: her konuşma için {adimlar, son_cevap, uretilen_token, sure_sn, adim_limiti_asildi}.
    """
    n = len(sorular)
    konusmalar = [[{"role": "system", "content": SISTEM_MESAJI}, {"role": "user", "content": s}] for s in sorular]
    adimlar = [[] for _ in range(n)]
    son_cevap = [None] * n
    tokenlar = [0] * n
    aktif = list(range(n))
    baslangic = time.perf_counter()

    for _ in range(MAKS_ADIM):
        if not aktif:
            break
        metinler, sayilar = _uret(model, tokenizer, [konusmalar[i] for i in aktif], sicaklik)
        sonraki = []
        for i, metin, sayi in zip(aktif, metinler, sayilar):
            tokenlar[i] += sayi
            cagrilar = arac_cagrilarini_ayikla(metin)
            if not cagrilar:
                son_cevap[i] = metin
                continue
            konusmalar[i].append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": c["arac"], "arguments": c["argumanlar"]}}
                    for c in cagrilar
                ],
            })
            for c in cagrilar:
                gozlem = arac_calistir(c["arac"], c["argumanlar"])
                adimlar[i].append({**c, "gozlem": gozlem})
                konusmalar[i].append({"role": "tool", "name": c["arac"], "content": gozlem})
            sonraki.append(i)
        aktif = sonraki

    sure = time.perf_counter() - baslangic  # batch içindeki koşular süreyi paylaşır
    return [
        {
            "adimlar": adimlar[i],
            "son_cevap": son_cevap[i] if son_cevap[i] is not None else "",
            "uretilen_token": tokenlar[i],
            "sure_sn": round(sure / n, 4),
            "adim_limiti_asildi": son_cevap[i] is None,
        }
        for i in range(n)
    ]
