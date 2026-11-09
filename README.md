# LLM Ajanlarında Tekrarlı Koşu Değişkenliği (Pilot)

> Bu bir pilot çalışmadır. Küçük bir açık modelle, kontrollü ve basit bir görev kümesinde yapılmıştır; sonuçlar büyük modellere veya gerçek ajan kıyaslama setlerine doğrudan genellenemez.

Aynı araç çağırma (tool calling) görevi küçük bir açık modelle tekrar tekrar koşturulur. Ölçülen şey, sonuçların ve izlenen adımların (yörünge) koşudan koşuya ne kadar değiştiğidir. Sıcaklık 0'da bile değişkenlik olup olmadığı ve güvenilir bir başarı oranı için kaç tekrar gerektiği sorulur.

## Araştırma soruları

- **S1.** Sıcaklık 0'da aynı görev 30 koşuda aynı sonucu ve aynı yörüngeyi veriyor mu?
- **S2.** Sıcaklık arttıkça (0 / 0.7 / 1.0) başarı oranı ve yörünge çeşitliliği nasıl değişiyor?
- **S3.** Tek koşuyla raporlanan başarı oranı, 30 koşunun güven aralığından ne kadar sapıyor?
- **S4 (ek).** Sıcaklık 0'da batch boyutu (1 ve 8) çıktıyı değiştiriyor mu?

## Kurulum

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (float16) |
| Araçlar | `hesapla(ifade)`, `birim_cevir(deger, kaynak, hedef)`, `sozluk_ara(anahtar)` (hepsi deterministik) |
| Görevler | 20 görev: 7 kolay (1 adım), 8 orta (2 adım), 5 zor (3+ adım), `tasks/tasks.json` |
| Deney | sıcaklık {0, 0.7, 1.0} × seed {sabit=42, değişken} × 30 koşu → 3600 koşu |
| Başarı | Nihai cevaptaki bir sayının doğru cevaba %0.5 göreli toleransla eşit olması |

Sıcaklık 0, açgözlü çözümleme (`do_sample=False`) demektir. Sıcaklık > 0'da `top_p=1`, `top_k=0` ile saf örnekleme yapılır. `sozluk_ara` aracındaki değerler deney için sabitlenmiş yaklaşık değerlerdir ve güncel istatistik iddiası taşımaz.

## Ölçülen metrikler

- Başarı oranı, görev bazında ve genel, %95 Wilson güven aralığıyla
- Farklı yörünge sayısı (araç adı ve argümanlarıyla birlikte benzersiz araç çağrı dizileri)
- İlk sapma adımı: koşuların ilk kez farklı adım attığı adım
- Kararsız (flaky) görevler: bazı koşularda başarılı, bazılarında başarısız olanlar
- Üretilen token ve süre değişkenliği
- Tek koşu yanılgısı: her görevden rastgele tek koşu seçildiğinde (2000 bootstrap) başarı oranının 30 koşu ortalamasından sapması

## Kullanım

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # modelsiz birim testleri

python src/runner.py --runs 30                 # tam deney -> results/runs.jsonl
python src/runner.py --limit 10 --temps 0 0.7  # kısaltılmış sürüm
python src/runner.py --temps 0 --seed-modes sabit --batch-size 8 \
    --out results/runs_batch8.jsonl            # S4

python src/analysis.py --batch results/runs_batch8.jsonl
```

Koşu yarıda kesilirse aynı komut tekrar çalıştırılabilir, kayıtlı koşular atlanır. Colab için: `notebooks/colab_demo.ipynb`.

Her koşu `runs.jsonl` içinde bir satırdır: görev, sıcaklık, seed, koşu no, batch boyutu, araç adımları ve gözlemler, nihai cevap, başarı, üretilen token ve süre.

## Sonuçlar

*Grafikler ve bulgular `python src/analysis.py` çıktısından eklenecek.*

| Grafik | Dosya |
|---|---|
| Sıcaklığa göre başarı + güven aralığı | `results/1_basari_sicaklik.png` |
| Görev × sıcaklık farklı yörünge sayısı | `results/2_yorunge_isi_haritasi.png` |
| Kararsız görev oranı | `results/3_kararsiz_gorevler.png` |
| Kaç tekrar yeterli? | `results/4_tekrar_sayisi_ci.png` |

## Sınırlılıklar

- Tek ve küçük bir model (1.5B), 20 yapay görev. Görevler kasıtlı olarak kısa ve cevabı kesin bilinen türdendir.
- Başarı kontrolü sayı eşleştirmesine dayanır. Cevapta ara bir değer doğru cevapla çakışırsa yanlış pozitif oluşabilir.
- Batch > 1 iken seed batch başına bir kez ayarlanır ve süre batch içindeki koşulara eşit bölünür.
- Sıcaklık 0'daki olası farklar donanıma, CUDA çekirdeklerine ve kütüphane sürümlerine bağlıdır; sonuçlar kullanılan GPU ile birlikte raporlanmalıdır.

## Repo yapısı

```
tasks/tasks.json         20 görev, doğru cevaplar ve referans yörüngeler
src/tools.py             3 araç ve JSON şemaları
src/agent.py             ajan döngüsü (batch destekli)
src/runner.py            tekrarlı koşu, jsonl kayıt, başarı kontrolü
src/analysis.py          metrikler, csv tablolar, grafikler
tests/test_core.py       modelsiz birim testleri
notebooks/colab_demo.ipynb
results/                 runs.jsonl, csv ve png çıktıları
```
