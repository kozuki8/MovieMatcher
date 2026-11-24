# MovieMatcher

Sevdiğin filmi yaz, benzer 10 filmi gör.

Model kendi bilgisayarında çalışır ve her film için önerileri **bir kez** hesaplar.
Yayındaki site (`site/index.html`) sunucuda model çalıştırmaz; yalnızca hazır
dosyaları okur. Bu yüzden Vercel'de ücretsiz, hızlı ve bakım gerektirmeden
yayınlanır. Ayrı bir Python API'sine (FastAPI vb.) gerek yoktur.

## Klasör yapısı

```
movie_recommender.py    modelin kendisi
evaluate.py             ayarları sayıyla karşılaştırma
export_static.py        site dosyalarını üretir
site/
  index.html            web sitesi (tek dosya)
  data.json             filmler, önerileri ve afiş yolları   (export_static.py üretir)
  overviews.json        film özetleri                        (export_static.py üretir)
Data/                   veri setleri ve önbellekler (GitHub'a gitmez)
```

## 1. Site dosyalarını üret

Önce kısa bir deneme:

```
python export_static.py --limit 200
```

Çıktıdaki **"Aktif bloklar"** satırında `overview (embedding)`, `cf` ve `themes`
görünmeli. Eksik olan varsa `UYARI` yazar; eksik bir modelle yayına çıkma.

### Afişler

Afişler için iki yol var, betik kendisi seçer:

1. **CSV'de `poster_path` sütunu varsa** oradan okunur. Ek bir şey yapmana gerek yok.
2. **Yoksa TMDB'den çekilir.** TMDB hesabından (Settings → API) bir anahtar al ve
   PowerShell'de, betiği çalıştırdığın pencerede şunu yaz:

   ```
   $env:TMDB_API_KEY = "v3-anahtarın"
   ```

   (v4 "okuma erişim jetonun varsa `$env:TMDB_TOKEN = "jeton"` da olur.)
   İlk çalıştırmada her film için bir istek atılır (birkaç dakika). Sonuçlar
   `Data/tmdb_cache.json` dosyasına kaydedilir; sonraki çalıştırmalar hızlıdır.
   Bu yoldan TMDB'nin güncel film özeti de alınır ve siteye o konur.

Hiçbir kaynak yoksa site yine çalışır, ama afişler yerine "MM" işaretli yer
tutucular görünür. Anahtarı kimseyle paylaşma ve dosyalara yazma; sadece o
terminal oturumunda ortam değişkeni olarak ver.

### Tam üretim

```
python export_static.py
```

Film sayısına göre yarım saate kadar sürebilir; bilgisayarı uyku moduna sokma.
`--min-votes 500` gibi bir değerle sitede aranabilecek filmleri daha bilinen
filmlerle sınırlayabilirsin (varsayılan: en az 100 oy). Daha az film, daha küçük
dosya ve genelde daha güvenilir öneri demek. `--no-overviews` özetleri siteye
koymaz (`overviews.json` üretilmez; site "No overview available" yazar).

## 2. Bilgisayarında dene

Tarayıcı JSON dosyalarını çift tıklayarak açılan sayfadan okuyamaz, küçük bir
sunucu gerekir:

```
python -m http.server 8000 --directory site
```

Sonra tarayıcıda `http://localhost:8000` adresini aç.

## 3. GitHub'a yükle

Proje klasörünü OneDrive dışında tutmanı öneririm (örn. `C:\dev\MovieMatcher`);
OneDrive `.git` klasörüyle sorun çıkarabiliyor.

```
git init
git add .
git status
```

`git status` listesinde `Data/` ve `.npy`/`.npz` dosyaları **olmamalı**. Olması
gerekenler: `movie_recommender.py`, `evaluate.py`, `export_static.py`,
`README.md`, `.gitignore`, `site/index.html`, `site/data.json`,
`site/overviews.json`. Sonra:

```
git commit -m "MovieMatcher"
git branch -M main
git remote add origin https://github.com/KULLANICI/REPO.git
git push -u origin main
```

## 4. Vercel'de yayınla

1. vercel.com'a GitHub hesabınla giriş yap.
2. **Add New → Project** ile repoyu seç.
3. **Root Directory** alanına `site` yaz. (Unutulursa 404 verir.)
4. **Framework Preset**: `Other`. Build komutu ve çıktı klasörü boş kalsın.
5. **Deploy**'a bas.

Sonraki her `git push` siteyi otomatik günceller.

## Öneriler değişince

```
python export_static.py
git add site
git commit -m "Önerileri güncelle"
git push
```

## Paylaşılabilir adresler

Bir film seçilince adres `?f=<TMDB id>` olur (örn. `https://siten.vercel.app/?f=157336`).
Bu adresi paylaşınca aynı sonuç açılır.

## Kaynaklar ve lisans

- Sitenin altındaki kaynak notlarını gerçek veri kaynaklarına göre kontrol et.
  Afişler ve film bilgileri TMDB'den geliyorsa TMDB'nin atıf cümlesi sayfada
  kalmalı. TMDB'nin ücretsiz kullanımı ticari olmayan projeler içindir;
  ticari bir kullanım planlıyorsan koşullarını kendin oku.
- MovieLens verisi araştırma ve ticari olmayan kullanım içindir; sayfada atıf
  yapılıyor. Ticari kullanım için GroupLens'ten izin gerekir.
- Ham MovieLens dosyalarını ve `Data/` klasörünü GitHub'a koyma.
