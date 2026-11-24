"""
Web sitesi için öneri verisini hazırlar: site/data.json ve site/overviews.json.

Model her film için bir kez, kendi bilgisayarında çalıştırılır. Yayındaki site
model çalıştırmaz; yalnızca bu dosyalardaki hazır listeleri gösterir.

Kullanım (movie_recommender.py ile aynı klasörden):
    python export_static.py                   # >= 100 oyu olan filmler aranabilir
    python export_static.py --min-votes 500   # daha az ama daha bilinen film
    python export_static.py --limit 200       # hızlı deneme (en popüler 200 film)

Afişler (poster) ve özetler:
  * CSV'de "poster_path" sütunu varsa afişler oradan alınır.
  * Yoksa TMDB'den çekilir. Bunun için bir TMDB API anahtarı gerekir
    (themoviedb.org hesabı > Settings > API). PowerShell'de:
        $env:TMDB_API_KEY = "v3-anahtarın"        # ya da
        $env:TMDB_TOKEN   = "v4-okuma-jetonun"
    İlk çalıştırmada filmler için istek atılır (birkaç dakika); sonuçlar
    Data/tmdb_cache.json dosyasına kaydedilir, sonraki çalıştırmalar hızlıdır.
  * Anahtar da yoksa site afişsiz çalışır (MM işaretli yer tutucular görünür).

Süre: film başına yaklaşık 0,05-0,1 saniye; 15-25 bin film için yarım saate
kadar sürebilir. İlk denemeyi --limit ile yap.
"""

import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import model8 as mr

TMDB_CACHE_PATH = Path("Data/tmdb_cache.json")
OVERVIEW_LIMIT = 300  # siteye giden özet uzunluğu (karakter)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="site/data.json", help="çıktı dosyası")
    parser.add_argument(
        "--min-votes", type=int, default=100,
        help="aranabilir olması için gereken en az oy sayısı (vote_count)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="yalnızca en popüler N filmi işle (deneme için)",
    )
    parser.add_argument(
        "--posters", choices=["auto", "tmdb", "none"], default="auto",
        help="auto: CSV sütunu varsa onu, yoksa TMDB anahtarı varsa TMDB'yi kullan",
    )
    parser.add_argument(
        "--no-overviews", action="store_true",
        help="film özetlerini siteye koyma",
    )
    parser.add_argument("--workers", type=int, default=12, help="TMDB istek sayısı (paralel)")
    return parser.parse_args()


def report_blocks():
    """Hangi bloklar aktif? Eksik bir blokla yayına çıkmamak için uyarır."""
    blocks = list(mr.sparse_blocks)
    if getattr(mr, "embeddings", None) is not None:
        blocks.append("overview (embedding)")
    if getattr(mr, "theme_profiles", None) is not None:
        blocks.append("themes")
    print("Aktif bloklar:", ", ".join(blocks))

    if getattr(mr, "embeddings", None) is None:
        print("UYARI: Embedding yok (sentence-transformers kurulu değil ya da "
              "önbellek bulunamadı). Öneriler zayıf olur.")
    if getattr(mr, "CF_ENABLED", False) and "cf" not in mr.sparse_blocks:
        print("UYARI: MovieLens beğeni bloğu (cf) yüklenmedi.")
    if getattr(mr, "THEMES_ENABLED", False) and getattr(mr, "theme_profiles", None) is None:
        print("UYARI: Tag Genome tema bloğu (themes) yüklenmedi.")


# ---------------------------------------------------------------------
# Afiş ve özet kaynakları
# ---------------------------------------------------------------------
def posters_from_csv():
    """CSV'de poster_path sütunu varsa {tmdb_id: dosya_adı} döndürür."""
    header = pd.read_csv(mr.DATA_PATH, nrows=0).columns
    if "poster_path" not in header:
        return None

    df = pd.read_csv(mr.DATA_PATH, usecols=["id", "poster_path"]).dropna()
    result = {}
    for tmdb_id, path in zip(df["id"], df["poster_path"]):
        name = str(path).strip().rsplit("/", 1)[-1]
        if name and name.lower() != "nan":
            result[int(tmdb_id)] = name
    return result


def tmdb_credentials():
    key = os.environ.get("TMDB_API_KEY", "").strip()
    token = os.environ.get("TMDB_TOKEN", "").strip()
    return key, token


def fetch_tmdb(ids, workers):
    """
    TMDB'den {tmdb_id: {"p": poster_dosya_adı, "o": özet}} çeker.
    Sonuçlar Data/tmdb_cache.json dosyasında saklanır; başarısız olanlar
    önbelleğe yazılmaz ve bir sonraki çalıştırmada yeniden denenir.
    """
    key, token = tmdb_credentials()
    if not key and not token:
        return None

    base = os.environ.get("TMDB_API_BASE", "https://api.themoviedb.org/3").rstrip("/")

    cache = {}
    if TMDB_CACHE_PATH.exists():
        cache = json.loads(TMDB_CACHE_PATH.read_text(encoding="utf-8"))

    todo = [i for i in ids if str(i) not in cache]
    print(f"TMDB: {len(ids) - len(todo):,} film önbellekte, {len(todo):,} film çekilecek...")

    def fetch_one(tmdb_id):
        url = f"{base}/movie/{tmdb_id}"
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            url += f"?api_key={key}"
        request = urllib.request.Request(url, headers=headers)

        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    body = json.load(response)
                poster = (body.get("poster_path") or "").rsplit("/", 1)[-1]
                return tmdb_id, {"p": poster, "o": body.get("overview") or ""}
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return tmdb_id, {"p": "", "o": ""}
                if error.code == 401:
                    raise SystemExit("TMDB anahtarı reddedildi (401). Anahtarını kontrol et.")
                if error.code == 429:
                    time.sleep(float(error.headers.get("Retry-After", 2)))
                    continue
                time.sleep(1 + attempt)
            except Exception:
                time.sleep(1 + attempt)
        return tmdb_id, None

    done = failed = 0
    started = time.perf_counter()

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for tmdb_id, info in pool.map(fetch_one, todo):
                if info is None:
                    failed += 1
                else:
                    cache[str(tmdb_id)] = info
                done += 1
                if done % 500 == 0 or done == len(todo):
                    elapsed = time.perf_counter() - started
                    print(f"  TMDB {done:,}/{len(todo):,}  ({elapsed / 60:.1f} dk)")
                    TMDB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    TMDB_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")

    if failed:
        print(f"UYARI: {failed:,} film için TMDB yanıt vermedi; tekrar çalıştırınca yeniden denenir.")

    wanted = set(ids)
    return {int(k): v for k, v in cache.items() if int(k) in wanted}


def shorten(text, limit=OVERVIEW_LIMIT):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "…"


def main():
    args = parse_args()
    movies = mr.movies
    pop = movies["_pop_raw"].to_numpy()

    report_blocks()

    # 1) Aranabilir filmler: yeterince oy almış olanlar, popülerlik sırasıyla
    if mr.HAS_VOTE_COUNT:
        mask = movies["vote_count"].fillna(0).to_numpy() >= args.min_votes
    else:
        mask = np.ones(len(movies), dtype=bool)

    queries = np.flatnonzero(mask)
    queries = queries[np.argsort(-pop[queries], kind="stable")]
    if args.limit:
        queries = queries[: args.limit]

    print(f"{len(queries):,} film için öneri hesaplanacak...")

    # 2) Her film için önerileri hesapla (modelin kendi fonksiyonlarıyla)
    recs = {}
    started = time.perf_counter()

    for n, i in enumerate(queries, start=1):
        scores = mr.apply_filters(mr.score_all(i), i)
        top = mr.pick_diverse(scores, mr.TOP_N)
        if top:
            recs[int(i)] = [(int(j), float(scores[j])) for j in top]

        if n % 250 == 0 or n == len(queries):
            elapsed = time.perf_counter() - started
            eta = elapsed / n * (len(queries) - n)
            print(f"  {n:,}/{len(queries):,}  geçen {elapsed / 60:.1f} dk, "
                  f"kalan yaklaşık {eta / 60:.1f} dk")

    if not recs:
        raise SystemExit("Hiç öneri üretilemedi.")

    # 3) Sadece gerekli filmleri al (aranabilir olanlar + önerilenler),
    #    popülerlik sırasına diz ve numaralarını yeniden ver.
    used = set(recs)
    for items in recs.values():
        used.update(j for j, _ in items)

    order = sorted(used, key=lambda i: (-pop[i], i))
    new_index = {old: new for new, old in enumerate(order)}
    tmdb_ids = [int(movies.at[i, "id"]) for i in order]

    # 4) Afiş ve özetler
    poster_map, tmdb_info = {}, {}

    if args.posters != "none":
        if args.posters == "auto":
            csv_posters = posters_from_csv()
            if csv_posters is not None:
                poster_map = csv_posters
                print("Afişler: CSV'deki poster_path sütunundan alındı.")

        if not poster_map:
            tmdb_info = fetch_tmdb(tmdb_ids, args.workers)
            if tmdb_info is None:
                print("UYARI: Afiş kaynağı yok (CSV'de poster_path sütunu ve TMDB anahtarı "
                      "bulunamadı). Site afişsiz çalışacak. Ayrıntı için dosya başındaki "
                      "açıklamaya bak.")
            else:
                poster_map = {k: v["p"] for k, v in tmdb_info.items() if v.get("p")}
                print(f"Afişler: TMDB'den {len(poster_map):,} film için alındı.")

    def year_of(i):
        year = str(movies.at[i, "year"])
        return int(year) if year.isdigit() else 0

    data = {
        "meta": {
            "generated": date.today().isoformat(),
            "films": len(order),
            "searchable": len(recs),
            "top_n": mr.TOP_N,
        },
        "t": [str(movies.at[i, "title"]) for i in order],
        "y": [year_of(i) for i in order],
        "i": tmdb_ids,
        "g": [list(movies.at[i, "genres"][:3]) for i in order],
        "p": [poster_map.get(t, "") for t in tmdb_ids],
        "r": {},
    }

    for old, items in recs.items():
        flat = []
        for j, score in items:
            flat += [new_index[j], int(round(min(max(score, 0.0), 1.0) * 100))]
        data["r"][str(new_index[old])] = flat

    # 5) Yaz
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out.write_text(text, encoding="utf-8")

    raw_kb = len(text.encode("utf-8")) / 1024
    gz_kb = len(gzip.compress(text.encode("utf-8"))) / 1024
    print(f"\nYazıldı: {out}  ({raw_kb:,.0f} KB, sıkıştırılmış yaklaşık {gz_kb:,.0f} KB)")

    if not args.no_overviews:
        overviews = []
        for i, tmdb_id in zip(order, tmdb_ids):
            tmdb_text = (tmdb_info.get(tmdb_id) or {}).get("o", "") if tmdb_info else ""
            overviews.append(shorten(tmdb_text or movies.at[i, "overview"]))

        ov_path = out.with_name("overviews.json")
        ov_text = json.dumps(overviews, ensure_ascii=False, separators=(",", ":"))
        ov_path.write_text(ov_text, encoding="utf-8")
        ov_kb = len(ov_text.encode("utf-8")) / 1024
        ov_gz = len(gzip.compress(ov_text.encode("utf-8"))) / 1024
        print(f"Yazıldı: {ov_path}  ({ov_kb:,.0f} KB, sıkıştırılmış yaklaşık {ov_gz:,.0f} KB)")

    print(f"Aranabilir film: {len(recs):,}   tabloda toplam film: {len(order):,}")


if __name__ == "__main__":
    main()
