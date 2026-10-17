# Küçük Bir Araç Çağıran Ajanda Tekrarlı Koşu Değişkenliği: Pilot Rapor

## Amaç

Büyük dil modeli (BDM) ajanlarının değerlendirildiği çalışmalarda başarı oranı çoğu zaman tek koşudan raporlanır. Bu pilot çalışma, aynı görev aynı ayarlarla tekrar tekrar koşturulduğunda sonucun ve izlenen araç çağrı dizisinin (yörünge) ne kadar değiştiğini, küçük ve açık bir model üzerinde ölçmeyi amaçlar. Sonuçlar tek bir 1.5B modele ve 20 yapay göreve dayandığı için genellenebilir bulgu değil, daha büyük bir deney tasarımına girdi olarak okunmalıdır.

## Yöntem

Ajan olarak `Qwen/Qwen2.5-1.5B-Instruct` modeli float16 hassasiyetle kullanıldı. Modelin erişebildiği üç araç (`hesapla`, `birim_cevir`, `sozluk_ara`) tamamen deterministiktir, dolayısıyla koşular arasındaki her fark modelin üretiminden kaynaklanır. Görev kümesi 7 kolay (tek araç çağrısı), 8 orta (iki adım) ve 5 zor (üç ya da daha fazla adım) olmak üzere 20 sorudan oluşur. Ajan en fazla 6 adım atabilir. Nihai cevaptaki herhangi bir sayı doğru cevaba %0.5 göreli tolerans içinde eşitse koşu başarılı sayıldı.

Başlangıçta üç sıcaklık değerinin (0, 0.7, 1.0) her biri için iki seed modu (her koşuda 42 ya da her koşuda farklı) ve görev başına 30 koşu planlanmıştı. T=0.7 ve sabit seed ile tamamlanan ilk 229 koşuda (7 görevde 30, bir görevde 19 koşu) her görevin tüm koşuları birebir aynı yörüngeyi ve aynı cevabı verdi. Bu beklenen bir sonuçtur, çünkü seed her koşudan önce aynı değere döndüğünde örnekleme de aynı rastgele sayıları çeker. Bu blokların sorulara yeni bilgi katmadığı görülünce tasarım daraltıldı. T=0'da iki seed modu 30'ar koşuyla (1200 koşu), T=0.7 ve T=1.0'da yalnızca değişken seed 10'ar koşuyla (400 koşu) çalıştırıldı. Toplam 1600 koşu analiz edildi. Batch boyutunun etkisi için T=0 ve sabit seed altında batch=8 ile ayrıca 600 koşu yapıldı.

Başarı oranları için %95 Wilson aralığı hesaplandı. Tek koşu yanılgısı, her görevden rastgele bir koşu seçilerek elde edilen başarı oranının 2000 bootstrap tekrarında tüm koşuların ortalamasından ne kadar saptığıyla ölçüldü. Deney, NVIDIA RTX 4070 Laptop GPU (8 GB), CUDA 13.0, torch 2.14.0 ve transformers 5.19.0 ile Windows 11 üzerinde yürütüldü.

## Bulgular

### Sıcaklık 0'da belirlenimcilik (S1)

T=0'da 20 görevin tamamı 30 koşu boyunca tek bir yörünge ve tek bir nihai cevap üretti. Sabit ve değişken seed modları arasında fark yoktu. Her iki modda da başarı oranı 0.300 oldu (600 koşuda 180, %95 Wilson [0.265, 0.338]). Kararsız görev, yani bazı koşularda başarılı bazılarında başarısız olan görev bulunmadı. Bu donanım ve yazılım yığınında açgözlü çözümleme tekrarlanabilir çıktı veriyor. Ancak bu sonuç başka bir GPU'ya ya da kütüphane sürümüne doğrudan aktarılamaz.

T=0'daki başarının tamamı kolay görevlerden geldi. Kolay görevlerde başarı 0.857 iken orta ve zor görevlerde sıfırdı. Orta görevlerdeki başarısız koşuların çoğunda (%62) araç çağrıları hatasız tamamlanmasına rağmen nihai cevap yanlıştı. Örneğin model mesafeyi km olarak getirip mile çevirmeden "450 mil" yanıtını verdi. Zor görevlerde baskın hata türü aracın hata döndürmesiydi (%60). Model `hesapla` aracına sayı yerine `ankara_nufus * izmir_nufus` gibi değişken adları gönderdi.

### Sıcaklığın etkisi (S2)

| Sıcaklık | Başarı [%95 Wilson] | Ort. farklı yörünge | Kararsız görev | Araç çağrısı yok (kolay) | Ort. üretilen token |
|---|---|---|---|---|---|
| 0 | 0.300 [0.265, 0.338] | 1.0 | 0/20 | %0 | 87.6 |
| 0.7 | 0.225 [0.173, 0.288] | 3.4 | 10/20 | %19 | 106.9 |
| 1.0 | 0.085 [0.054, 0.132] | 3.1 | 8/20 | %60 | 125.7 |

Sıcaklık arttıkça başarı belirgin biçimde düştü ve T=0.7 ile T=1.0'ın güven aralıkları örtüşmedi. Sıfırdan büyük her sıcaklıkta 20 görevin tamamında birden fazla yörünge gözlendi ve koşular her görevde ilk adımda ayrıştı. Görev başına farklı yörünge sayısı T=0.7'de 2 ile 7, T=1.0'da 2 ile 5 arasında değişti. T=1.0'da ortalama yörünge sayısının biraz daha düşük çıkması çelişkili görünebilir. Bunun nedeni, modelin bu sıcaklıkta sıklıkla hiç araç çağırmaması ve bu koşuların hepsinin aynı boş yörüngede toplanmasıdır.

Başarıdaki düşüşün başlıca kaynağı da budur. Araç çağırmadan doğrudan cevap verme oranı kolay görevlerde T=0'da %0 iken T=0.7'de %19'a, T=1.0'da %60'a çıktı. Zor görevlerde bu oran T=1.0'da %78'e ulaştı. Bu durumda model değeri araçtan almak yerine kendisi üretiyor ve cevap çoğunlukla yanlış oluyor. Yörüngenin ilk adımındaki "araç çağır ya da cevapla" kararı, örneklemeye en duyarlı nokta olarak öne çıkıyor.

Öte yandan sıcaklık her görevi kötüleştirmedi. T=0'da hiçbir koşuda çözülemeyen o02, o03 ve z04 görevleri T=0.7'de 10 koşudan birinde doğru çözüldü. Açgözlü çözümleme, modelin ulaşabildiği en iyi yörüngeyi her zaman bulmuyor. T=0 sonucu bu nedenle modelin yeteneği için bir üst sınır olarak yorumlanmamalıdır.

### Tek koşuyla raporlamanın yanılgısı (S3)

| Sıcaklık | Tüm koşular | Tek koşu, ort. mutlak sapma | Tek koşu, en büyük sapma | Güven aralığı dışında kalma |
|---|---|---|---|---|
| 0 | 0.300 | 0.000 | 0.000 | %0 |
| 0.7 | 0.225 | 0.050 | 0.175 | %40 |
| 1.0 | 0.085 | 0.046 | 0.165 | %68 |

T=0'da tek koşu ile 30 koşu aynı sonucu veriyor. T>0'da ise her görevden tek koşu alınarak hesaplanan başarı oranı ortalamadan ortalama 5 yüzde puan, en kötü durumda 17.5 yüzde puan saptı. T=0.7'de bootstrap örneklerinin %40'ı, T=1.0'da %68'i tüm koşulardan hesaplanan %95 güven aralığının dışına düştü. Bu fark, sıcaklığı sıfırdan büyük ajan değerlendirmelerinde tek koşuyla raporlanan farkların büyük kısmının gürültü olabileceğini gösteriyor.

Görev bazında güven aralıkları tekrar sayısıyla yavaş daralıyor. Ortalama Wilson aralığı genişliği T=0.7'de 5 koşuda 0.49, 10 koşuda 0.36 oldu. T=0'da 30 koşuda bu değer 0.11'e iniyor. T=0'daki daralma, gözlenen değişkenlikten değil, yalnızca n'nin büyümesinden kaynaklanıyor, çünkü o koşullarda bütün koşular aynı. Görev düzeyinde yaklaşık ±0.1 hassasiyet (aralık genişliği 0.2) hedefleniyorsa, başarı oranı 0.5 civarında olan kararsız bir görev için gereken tekrar sayısı normal yaklaşımla yaklaşık 96'dır. Bu, deneyde kullanılan 10 koşunun yaklaşık on katı demek.

### Batch boyutunun etkisi (S4)

T=0 ve sabit seed altında batch=8 ile yapılan 600 koşu, 20 görevin 19'unda batch=1 ile aynı yörüngeyi ve 20 görevin hepsinde aynı nihai cevabı verdi. Fark yalnızca k04 görevinde ("3 saat kaç saniyedir?") gözlendi. Batch=1'de model `hesapla("3 * 3600")` çağırırken 8'lik gruplarda (koşu 0–23) `birim_cevir(3, saat, saniye)` çağırdı. 30 koşu 8'e tam bölünmediği için son grup 6 koşudan oluştu ve bu grup yeniden batch=1 ile aynı yörüngeyi izledi. Aynı grup içindeki koşular her zaman özdeşti.

Bu sonuç, sıcaklık 0'ın tek başına aynı çıktıyı garanti etmediğini gösteriyor. Olası açıklama, grup boyutu değiştiğinde GPU çekirdeklerinin farklı bir indirgeme sırası seçmesi ve float16'daki küçük yuvarlama farklarının olasılıkları birbirine çok yakın iki adayın sırasını tersine çevirmesidir. Bu deneyde bu mekanizma doğrudan doğrulanmadı. Bu deneyde fark başarıyı etkilemedi. Ancak çok adımlı görevlerde ilk adımdaki böyle bir değişiklik sonraki adımları da değiştirebileceği için toplu çıkarım yapılan değerlendirmelerde batch boyutunun raporlanması gerekir.

## Sınırlılıklar

Çalışma tek ve küçük bir modelle yapıldı. Model T=0'da orta ve zor görevlerin hiçbirini çözemediği için kararsızlık analizi pratikte kolay görevlere dayanıyor. T>0 koşulları görev başına 10 koşuyla sınırlı kaldığından görev düzeyindeki tahminlerin belirsizliği yüksek. Başarı kontrolü sayı eşleştirmesine dayanıyor ve cevapta geçen ara bir değer doğru cevapla çakışırsa yanlış pozitif oluşabilir. Deney sırasında aynı GPU'da başka iş yükleri de çalıştığı için süre ölçümleri karşılaştırma amacıyla kullanılmamalı. Token sayıları bu durumdan etkilenmez. Sıcaklık 0'daki belirlenimcilik ve batch etkisi donanıma ve yazılım sürümlerine bağlıdır.

## Sonraki adımlar

Bu pilotun ardından yapılabilecek en bilgilendirici genişletme, orta ve zor görevlerde de başarılı olabilen daha büyük bir modelin (ör. 3B ya da 7B) eklenmesidir. Kararsızlığın asıl ölçülmek istendiği yer de bu görevlerdir. T>0'da tekrar sayısını görev başına 30'a çıkarmak, batch=1, 2, 4, 8 gibi kademeli batch boyutlarıyla S4'ü genişletmek ve farklı bir GPU üzerinde T=0 sonuçlarını tekrarlamak diğer adımlar olabilir.

## Yeniden üretim

```bash
python src/runner.py --temps 0 --runs 30
python src/runner.py --temps 0.7 1.0 --seed-modes degisken --runs 10
python src/runner.py --temps 0 --seed-modes sabit --batch-size 8 --out results/runs_batch8.jsonl
python src/analysis.py --batch results/runs_batch8.jsonl
```

Ham koşu kayıtları `results/runs.jsonl` (1600 koşu), `results/runs_batch8.jsonl` (600 koşu) ve `results/runs_t07_sabit_seed.jsonl` (229 koşu, bırakılan blok) dosyalarındadır.
