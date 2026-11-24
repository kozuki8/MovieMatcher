"""

Film öneri sistemi (model8, hibrit + MovieLens Tag Genome)


Model7 mimarisini korur ve MovieLens Tag Genome tema profillerini bağımsız

"themes" benzerlik bloğu olarak ekler. Tema verisi olmayan filmler, bu blok

yüzünden cezalandırılmaz; diğer blokların ağırlıkları o karşılaştırmada

yeniden normalize edilir.


Gerekli paketler:

    pip install pandas numpy scipy scikit-learn sentence-transformers

İsteğe bağlı (yazım hatası düzeltme önerisi):

    pip install rapidfuzz


MovieLens için (isteğe bağlı ama önerilir):

    grouplens.org/datasets/movielens adresinden ml-25m indirilip açılır.

    ratings.csv, links.csv, genome-scores.csv ve genome-tags.csv dosyaları

    Data/ml-25m/ (veya ml-25m/) klasörüne konur. Eksik bloklar atlanır.


Not: sentence-transformers kurulu değilse konu özeti için otomatik olarak

TF-IDF'e geri döner (sonuçlar daha zayıf olur).


Ayarları sayıyla karşılaştırmak için evaluate.py dosyasını kullan.

"""


import hashlib

import json

import re

import time

from pathlib import Path


import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix, diags, hstack, load_npz, save_npz

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import normalize


# =====================================================================

# AYARLAR

# =====================================================================

DATA_PATH = "Data/movies_v5_25k.csv"


# Embedding ve CF önbellekleri bu klasöre yazılır. Dosya adları girdinin

# özetini (hash) içerir; girdi veya ayar değişirse otomatik yenilenir.

CACHE_DIR = Path("Data")

EMB_MODEL = "all-MiniLM-L6-v2"


TOP_N = 10               # kaç film önerilecek

ACTOR_LIMIT = 5         # her filmden sadece ilk N oyuncu kullanılır

EMB_KEYWORD_LIMIT = 15  # embedding metnine giren anahtar kelime sayısı


# True yapılırsa embedding'lerden ortalama vektör çıkarılır; böylece tüm

# filmlerin ortak bileşeni silinir ve benzerlik değerleri daha ayırt edici

# olur. Açık/kapalı hâlini evaluate.py ile karşılaştır.

CENTER_EMBEDDINGS = False


# MovieLens (collaborative filtering) ayarları

CF_ENABLED = True

ML_DIR = Path("Data/ml-25m")      # MovieLens dosyalarının klasörü
if not ML_DIR.exists() and Path("ml-25m").exists():
    ML_DIR = Path("ml-25m")

CF_RATING_THRESHOLD = 4.0         # bu puan ve üstü "beğendi" sayılır

CF_MIN_POSITIVES = 30             # bundan az beğeni alan filmde CF sinyali yok

CF_SHRINK = 50                    # az oylu filmlerin CF etkisini yumuşatır

# MovieLens Tag Genome (film x 1.128 tema etiketi)
THEMES_ENABLED = True
GENOME_SCORES_PATH = ML_DIR / "genome-scores.csv"
GENOME_TAGS_PATH = ML_DIR / "genome-tags.csv"
GENOME_CHUNK_SIZE = 1_000_000


# Ağırlıklar göreli önemleri belirtir; kullanılabilir bloklar her film çifti için normalize edilir.
# MovieLens bulunamazsa ilgili cf/themes blokları otomatik atlanır.

WEIGHTS = {

    "overview": 0.26,    # tür + anahtar kelime + konu özeti (embedding)

    "keywords": 0.18,

    "genres": 0.03,

    "directors": 0.09,

    "actors": 0.12,

    "cf": 0.10,          # MovieLens beğeni benzerliği
    
    "themes": 1.40,      # MovieLens Tag Genome tema profili

}


# Popülerlik: skorun ne kadarının oy sayısından etkileneceği (0 = kapalı)

POP_WEIGHT = 0.10

# Bu kadar oydan az alan filmler öneri listesine girmez (vote_count varsa)

MIN_VOTES = 350


# Dönem cezası: yıl farkı büyüdükçe skor en fazla bu oranda düşer (0 = kapalı)

YEAR_PENALTY = 0.15

YEAR_SCALE = 60  # bu kadar yıllık fark en yüksek cezayı verir


# Çeşitlilik: aynı yönetmenin filmlerinden en fazla kaç tanesi önerilsin

MAX_PER_DIRECTOR = 5

# Çeşitlilik için en yüksek skorlu kaç aday arasından seçilsin

POOL_SIZE = 50

# True yapılırsa seçilen filmle aynı isimli filmler (remake) önerilmez

EXCLUDE_SAME_TITLE = True


# =====================================================================

# 1. VERİ YÜKLEME

# =====================================================================

header = pd.read_csv(DATA_PATH, nrows=0).columns


base_columns = [

    "id", "title", "release_date", "overview",

    "genres", "keywords", "actors", "directors",

]

# Varsa popülerlik sütunlarını da oku

extra_columns = [c for c in ("vote_count", "popularity") if c in header]


movies = pd.read_csv(DATA_PATH, usecols=base_columns + extra_columns)

movies = movies.reset_index(drop=True)


movies["overview"] = movies["overview"].fillna("")

movies["year"] = movies["release_date"].fillna("").astype(str).str[:4]


def parse_list(value):

    """JSON liste metnini Python listesine çevirir, bozuk veride boş döner."""

    if pd.isna(value):

        return []

    try:

        data = json.loads(value)

    except (TypeError, ValueError):

        return []

    return [str(x) for x in data] if isinstance(data, list) else []


for col in ("genres", "keywords", "actors", "directors"):

    movies[col] = movies[col].apply(parse_list)


movies["actors"] = movies["actors"].apply(lambda x: x[:ACTOR_LIMIT])

movies["keywords"] = movies["keywords"].apply(

    lambda items: [k.strip().lower().replace(" ", "_") for k in items]

)


# Popülerlik bilgisi

if "vote_count" in movies.columns:

    POP_COLUMN = "vote_count"

elif "popularity" in movies.columns:

    POP_COLUMN = "popularity"

else:

    POP_COLUMN = None


HAS_VOTE_COUNT = "vote_count" in movies.columns


if POP_COLUMN:

    movies["_pop_raw"] = movies[POP_COLUMN].fillna(0).astype(float)

    pop = np.log1p(movies["_pop_raw"].to_numpy())

    if pop.max() > 0:

        pop = pop / pop.max()

    POP_MULTIPLIER = (1 - POP_WEIGHT) + POP_WEIGHT * pop

else:

    movies["_pop_raw"] = 0.0

    POP_MULTIPLIER = np.ones(len(movies))


# Hızlı erişim için

director_lists = movies["directors"].tolist()

year_num = pd.to_numeric(movies["year"], errors="coerce").to_numpy(dtype=float)


# Arama için hazırlık

def normalize_title(text):

    """Küçük harfe çevirir, baştaki the/a/an'i atar ("The Man from Earth"

    ile "man from earth" aynı sayılır)."""

    text = str(text).casefold().strip()

    return re.sub(r"^(the|a|an)\s+", "", text)


title_keys = movies["title"].fillna("").map(normalize_title)

years = movies["year"]


print("Film sayısı:", len(movies))

print("Popülerlik sütunu:", POP_COLUMN or "yok")


# =====================================================================

# 2. İÇERİK ÖZELLİKLERİ

# =====================================================================

def as_tokens(items):

    """Liste halindeki değerleri olduğu gibi token olarak kullanır."""

    return items


def list_tfidf(series, **kwargs):

    """Liste sütunları için IDF'li vektör (nadir değerler daha ağırlıklı)."""

    model = TfidfVectorizer(

        analyzer=as_tokens, min_df=2, dtype=np.float32, **kwargs

    )

    return model.fit_transform(series)


def build_embedding_text(genres, keywords, overview):

    """Embedding'e girecek metin: tür + temalar + konu özeti."""

    parts = []

    if genres:

        parts.append("Genres: " + ", ".join(genres))

    if keywords:

        themes = [k.replace("_", " ") for k in keywords[:EMB_KEYWORD_LIMIT]]

        parts.append("Themes: " + ", ".join(themes))

    if overview:

        parts.append(overview)

    return ". ".join(parts)


def embedding_cache_path(texts):

    """Metinlerin ve modelin özetine göre önbellek dosya yolu."""

    digest = hashlib.md5(

        ("\n".join(texts) + "|" + EMB_MODEL).encode("utf-8")

    ).hexdigest()[:10]

    return CACHE_DIR / f"overview_embeddings_{digest}.npy"


def load_embeddings(texts):

    """Metinleri cümle embedding'ine çevirir; önbelleğe kaydeder."""

    path = embedding_cache_path(texts)


    if path.exists():

        return np.load(path)


    try:

        from sentence_transformers import SentenceTransformer

    except ImportError:

        return None


    print("Embedding hesaplanıyor (sadece ilk çalıştırmada)...")

    model = SentenceTransformer(EMB_MODEL)

    embeddings = model.encode(

        texts,

        batch_size=128,

        normalize_embeddings=True,

        show_progress_bar=True,

    ).astype(np.float32)


    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, embeddings)

    return embeddings


# =====================================================================

# 3. MOVIELENS (COLLABORATIVE FILTERING) BLOĞU

# =====================================================================

def load_cf_block():

    """

    MovieLens'te puanı >= CF_RATING_THRESHOLD olan "beğeniler"den film x

    kullanıcı matrisi çıkarır. İki film arasındaki nokta çarpımı, onları

    birlikte beğenen kullanıcıların ağırlıklı cosine benzerliğidir.


    - Çok film beğenen kullanıcıların oyu daha az ağırlık taşır

      (1 / log2(2 + beğeni sayısı)).

    - Az beğeni alan filmlerin etkisi CF_SHRINK ile yumuşatılır,

      CF_MIN_POSITIVES altındakilerin CF sinyali sıfırlanır.

    - Filmler MovieLens'e links.csv içindeki tmdbId ile eşlenir; yani

      CSV'deki "id" sütunu TMDB id'si olmalıdır.


    Sonuç diske kaydedilir; ilk çalıştırma birkaç dakika sürebilir.

    """

    if not CF_ENABLED:

        return None


    ratings_path = ML_DIR / "ratings.csv"

    links_path = ML_DIR / "links.csv"


    if not ratings_path.exists() or not links_path.exists():

        print(

            f"MovieLens bulunamadı ({ML_DIR}), CF bloğu atlanıyor; "

            "sadece içerik tabanlı çalışılacak."

        )

        return None


    links_digest = hashlib.md5(links_path.read_bytes()).hexdigest()

    key = hashlib.md5(

        (

            f"{movies['id'].tolist()}|{CF_RATING_THRESHOLD}|{CF_MIN_POSITIVES}"

            f"|{CF_SHRINK}|{ratings_path.stat().st_size}|{links_digest}"

        ).encode("utf-8")

    ).hexdigest()[:10]

    cache_path = CACHE_DIR / f"cf_block_{key}.npz"


    if cache_path.exists():

        print("CF bloğu önbellekten yüklendi.")

        return load_npz(cache_path).tocsr()


    # --- Film eşleştirme: MovieLens movieId -> bizim satır numarası ---

    links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"]).dropna()

    links["tmdbId"] = links["tmdbId"].astype(np.int64)


    tmdb_to_row = pd.Series(

        np.arange(len(movies)), index=movies["id"].astype(np.int64)

    )

    tmdb_to_row = tmdb_to_row[~tmdb_to_row.index.duplicated()]


    links["row"] = links["tmdbId"].map(tmdb_to_row)

    links = links.dropna(subset=["row"])

    ml_to_row = pd.Series(

        links["row"].astype(np.int64).to_numpy(),

        index=links["movieId"].astype(np.int64).to_numpy(),

    )


    matched = ml_to_row.nunique()

    print(f"MovieLens eşleşmesi: {matched}/{len(movies)} film")


    if matched < 0.3 * len(movies):

        print(

            "UYARI: Eşleşme çok düşük. CSV'deki 'id' sütunu TMDB id'si "

            "olmayabilir; CF bloğu atlanıyor."

        )

        return None


    # --- Beğenileri parça parça oku (ratings.csv büyük) ---

    row_parts, user_parts = [], []

    total = 0


    for chunk in pd.read_csv(

        ratings_path,

        usecols=["userId", "movieId", "rating"],

        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},

        chunksize=5_000_000,

    ):

        total += len(chunk)

        chunk = chunk[chunk["rating"] >= CF_RATING_THRESHOLD]

        mapped = chunk["movieId"].map(ml_to_row)

        keep = mapped.notna().to_numpy()


        row_parts.append(mapped[keep].to_numpy(dtype=np.int32))

        user_parts.append(chunk["userId"].to_numpy()[keep])

        print(f"  {total:,} puan okundu...")


    rows = np.concatenate(row_parts)

    users = np.concatenate(user_parts)


    if len(rows) == 0:

        print("UYARI: Eşleşen beğeni bulunamadı, CF bloğu atlanıyor.")

        return None


    _, user_idx = np.unique(users, return_inverse=True)

    n_users = int(user_idx.max()) + 1


    # Çok film beğenen kullanıcıların oyu daha az ağırlıklı

    likes_per_user = np.bincount(user_idx)

    user_weight = (1.0 / np.log2(2.0 + likes_per_user)).astype(np.float32)


    matrix = csr_matrix(

        (user_weight[user_idx], (rows, user_idx)),

        shape=(len(movies), n_users),

        dtype=np.float32,

    )


    counts = np.bincount(rows, minlength=len(movies))

    matrix = normalize(matrix, norm="l2")


    scale = np.sqrt(counts / (counts + CF_SHRINK))

    scale[counts < CF_MIN_POSITIVES] = 0

    matrix = (diags(scale.astype(np.float32)) @ matrix).tocsr()

    matrix.eliminate_zeros()


    print(

        f"CF sinyali olan film: {int((counts >= CF_MIN_POSITIVES).sum())}"

        f"/{len(movies)}  (kullanıcı: {n_users:,})"

    )


    cache_path.parent.mkdir(parents=True, exist_ok=True)

    save_npz(cache_path, matrix)

    return matrix


def load_theme_profiles():
    """MovieLens Tag Genome verisinden film x etiket tema matrisi kurar.

    CSV büyük olduğu için parça parça okunur. TMDB id -> MovieLens movieId
    eşleştirmesi links.csv üzerinden yapılır. Her film satırı L2 normalize edilir.

    Tema verisi olmayan filmler için satır sıfır kalır; karşılaştırma sırasında
    themes bloğu o film çifti için devre dışı bırakılır.
    """
    if not THEMES_ENABLED:
        print("Tag Genome kapalı.")
        return None, None

    scores_path = GENOME_SCORES_PATH
    tags_path = GENOME_TAGS_PATH
    links_path = ML_DIR / "links.csv"

    # Genome dosyaları proje kökündeki ml-25m klasöründe duruyorsa onu da bul.
    if not scores_path.exists() or not tags_path.exists():
        for genome_dir in (Path("ml-25m"), Path("Data/ml-25m")):
            candidate_scores = genome_dir / "genome-scores.csv"
            candidate_tags = genome_dir / "genome-tags.csv"
            if candidate_scores.exists() and candidate_tags.exists():
                scores_path = candidate_scores
                tags_path = candidate_tags
                if not links_path.exists() and (genome_dir / "links.csv").exists():
                    links_path = genome_dir / "links.csv"
                break

    if not scores_path.exists() or not tags_path.exists() or not links_path.exists():
        print(
            "Tag Genome dosyaları bulunamadı; themes bloğu atlanıyor. "
            f"Beklenen klasör: {ML_DIR}"
        )
        return None, None

    tags = pd.read_csv(tags_path, usecols=["tagId", "tag"])
    tag_ids = tags["tagId"].to_numpy(dtype=np.int32)
    n_tags = len(tag_ids)
    if n_tags == 0:
        print("genome-tags.csv boş; themes bloğu atlanıyor.")
        return None, None

    # Cache anahtarı; kaynak dosyalar değişirse profil matrisi yeniden kurulur.
    links_stat = links_path.stat()
    scores_stat = scores_path.stat()
    tags_stat = tags_path.stat()
    movie_id_digest = hashlib.md5(
        pd.util.hash_pandas_object(movies["id"], index=False).values.tobytes()
    ).hexdigest()
    cache_key = hashlib.md5(
        (
            f"{movie_id_digest}|{scores_stat.st_size}|{scores_stat.st_mtime_ns}|"
            f"{tags_stat.st_size}|{tags_stat.st_mtime_ns}|"
            f"{links_stat.st_size}|{links_stat.st_mtime_ns}|{n_tags}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_path = CACHE_DIR / f"theme_profiles_{cache_key}.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            profiles = cached["profiles"].astype(np.float32, copy=False)
            available = cached["available"].astype(bool, copy=False)
        if profiles.shape == (len(movies), n_tags):
            print(
                f"Tag Genome önbellekten yüklendi: "
                f"{int(available.sum()):,}/{len(movies):,} film"
            )
            return profiles, available

    # MovieLens movieId -> movies tablosundaki satır numarası.
    links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"]).dropna()
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce")
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna(subset=["movieId", "tmdbId"])
    links["movieId"] = links["movieId"].astype(np.int64)
    links["tmdbId"] = links["tmdbId"].astype(np.int64)

    tmdb_to_row = pd.Series(
        np.arange(len(movies), dtype=np.int32),
        index=pd.to_numeric(movies["id"], errors="coerce").astype(np.int64),
    )
    tmdb_to_row = tmdb_to_row[~tmdb_to_row.index.duplicated(keep="first")]
    links["row"] = links["tmdbId"].map(tmdb_to_row)
    links = links.dropna(subset=["row"]).drop_duplicates("movieId", keep="first")
    ml_to_row = dict(
        zip(links["movieId"].to_numpy(), links["row"].to_numpy(dtype=np.int32))
    )

    # Genome tagId değerleri normalde 1..N şeklindedir; farklıysa güvenli map kullan.
    sorted_tag_ids = np.sort(tag_ids)
    contiguous_tags = np.array_equal(
        sorted_tag_ids, np.arange(1, n_tags + 1, dtype=np.int32)
    )
    tag_to_col = {int(tag_id): i for i, tag_id in enumerate(tag_ids)}

    profiles = np.zeros((len(movies), n_tags), dtype=np.float32)
    available = np.zeros(len(movies), dtype=bool)
    total_rows = 0
    matched_rows = 0

    print("Tag Genome tema profilleri oluşturuluyor...")
    for chunk in pd.read_csv(
        scores_path,
        usecols=["movieId", "tagId", "relevance"],
        dtype={
            "movieId": np.int32,
            "tagId": np.int32,
            "relevance": np.float32,
        },
        chunksize=GENOME_CHUNK_SIZE,
    ):
        total_rows += len(chunk)
        mapped_rows = chunk["movieId"].map(ml_to_row)
        keep = mapped_rows.notna().to_numpy()
        if not keep.any():
            continue

        row_idx = mapped_rows[keep].to_numpy(dtype=np.int32)
        tag_values = chunk.loc[keep, "tagId"].to_numpy(dtype=np.int32)
        relevance = chunk.loc[keep, "relevance"].to_numpy(dtype=np.float32)

        if contiguous_tags:
            col_idx = tag_values - 1
            valid = (col_idx >= 0) & (col_idx < n_tags)
        else:
            col_idx = pd.Series(tag_values).map(tag_to_col).fillna(-1).to_numpy(dtype=np.int32)
            valid = col_idx >= 0

        if valid.any():
            # Genome dosyasında film/etiket çifti tektir; doğrudan indeksleme yeterli.
            profiles[row_idx[valid], col_idx[valid]] = relevance[valid]
            available[row_idx[valid]] = True
            matched_rows += int(valid.sum())

        print(f"  {total_rows:,} genome satırı tarandı...", end="\r")

    # Her film profilini birim uzunluğa getir; dot product = cosine similarity.
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    np.divide(profiles, np.maximum(norms, 1e-12), out=profiles)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, profiles=profiles, available=available)

    print(
        f"\nTag Genome hazır: {int(available.sum()):,}/{len(movies):,} film "
        f"(%{available.mean() * 100:.2f}), {n_tags:,} etiket, "
        f"{matched_rows:,} eşleşen puan"
    )
    return profiles, available


# =====================================================================

# 4. ÖZELLİK MATRİSLERİNİ KUR

# =====================================================================

start = time.perf_counter()


sparse_blocks = {}


embedding_texts = [

    build_embedding_text(g, k, o)

    for g, k, o in zip(movies["genres"], movies["keywords"], movies["overview"])

]


embeddings = load_embeddings(embedding_texts)


if embeddings is not None:

    if CENTER_EMBEDDINGS:

        embeddings = embeddings - embeddings.mean(axis=0)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        embeddings = embeddings / np.maximum(norms, 1e-9)


    # Hiç bilgisi olmayan filmler birbirine benzemesin

    empty_text = np.array([t == "" for t in embedding_texts])

    embeddings[empty_text] = 0

else:

    print(

        "UYARI: sentence-transformers kurulu değil, konu özeti için "

        "TF-IDF kullanılıyor.\n"

        "       Daha iyi sonuç için: pip install sentence-transformers"

    )

    overview_model = TfidfVectorizer(

        max_features=20000,

        stop_words="english",

        min_df=2,

        sublinear_tf=True,

        dtype=np.float32,

    )

    sparse_blocks["overview"] = overview_model.fit_transform(

        movies["overview"]

    )


sparse_blocks["keywords"] = list_tfidf(movies["keywords"], max_features=15000)

sparse_blocks["genres"] = list_tfidf(movies["genres"])

sparse_blocks["directors"] = list_tfidf(movies["directors"])

sparse_blocks["actors"] = list_tfidf(movies["actors"])


cf_block = load_cf_block()

if cf_block is not None:

    sparse_blocks["cf"] = cf_block


# Dense tema profilleri sparse bloklardan ayrı tutulur.
theme_profiles, theme_available = load_theme_profiles()


# Her blok L2 normalize (veya <= 1 normlu) olduğu için, sqrt(ağırlık) ile

# ölçekleyip yan yana koyunca tek bir çarpım ağırlıklı cosine toplamını verir.

sparse_features = hstack(

    [

        matrix * float(np.sqrt(WEIGHTS[name]))

        for name, matrix in sparse_blocks.items()

    ]

).tocsr()


print(

    "Model hazırlama süresi:",

    round(time.perf_counter() - start, 2),

    "saniye",

)


# =====================================================================

# 5. FİLM BULMA

# =====================================================================

def find_movie(query):

    """

    Filmi isme göre bulur. "oldboy 2003" veya "oldboy (2003)" gibi

    girişlerde yılı ayırır. Birden fazla eşleşme varsa en popüler olan

    başa gelir.

    """

    query = query.strip()

    attempts = []


    match = re.match(r"^(.+?)\s*\(?(\d{4})\)?$", query)

    if match:

        attempts.append((match.group(1).strip(), match.group(2)))

    attempts.append((query, None))  # "Blade Runner 2049" gibi başlıklar için


    for title, year in attempts:

        mask = title_keys == normalize_title(title)

        if year:

            mask &= years == year

        found = movies[mask]

        if not found.empty:

            return found.sort_values("_pop_raw", ascending=False)


    # Tam eşleşme yoksa başlığın içinde geçenleri dene (en popüler başa gelir)

    key = normalize_title(query)

    if len(key) >= 3:

        found = movies[title_keys.str.contains(key, regex=False)]

        if not found.empty:

            return found.sort_values("_pop_raw", ascending=False)


    return movies.iloc[0:0]


def suggest_titles(query, limit=5):

    """Yazım hatası varsa yakın başlıkları önerir (rapidfuzz gerekir)."""

    try:

        from rapidfuzz import fuzz, process

    except ImportError:

        return []


    hits = process.extract(

        normalize_title(query),

        title_keys,

        scorer=fuzz.WRatio,

        limit=limit * 3,

        score_cutoff=70,

    )


    suggestions = []

    for _choice, _score, key in hits:

        title = movies.loc[key, "title"]

        if title not in suggestions:

            suggestions.append(title)

        if len(suggestions) == limit:

            break

    return suggestions


# =====================================================================

# 6. BENZERLİK

# =====================================================================

def year_factor(index):

    """Seçilen filmden yıl olarak uzak filmlerin skorunu hafifçe düşürür."""

    if not YEAR_PENALTY:

        return np.ones(len(movies))


    diff = np.abs(year_num - year_num[index])

    diff = np.nan_to_num(diff, nan=30.0)  # yılı bilinmeyenlere orta ceza

    return 1 - YEAR_PENALTY * np.minimum(diff / YEAR_SCALE, 1)


def score_all(index):
    """Tüm filmlerin skorunu verir; eksik tema verisi ağırlık kaybettirmez."""
    base_scores = (sparse_features @ sparse_features[index].T).toarray().ravel()

    if embeddings is not None:
        base_scores += WEIGHTS.get("overview", 0.0) * (embeddings @ embeddings[index])

    # Tema dışındaki, gerçekten yüklenmiş blokların toplam ağırlığı.
    base_weight = sum(WEIGHTS.get(name, 0.0) for name in sparse_blocks)
    if embeddings is not None:
        base_weight += WEIGHTS.get("overview", 0.0)

    denominators = np.full(len(movies), base_weight, dtype=np.float32)
    theme_weight = WEIGHTS.get("themes", 0.0)

    if (
        theme_profiles is not None
        and theme_available[index]
        and theme_weight > 0
    ):
        theme_sims = theme_profiles @ theme_profiles[index]
        valid_theme_pairs = theme_available
        base_scores[valid_theme_pairs] += theme_weight * theme_sims[valid_theme_pairs]
        denominators[valid_theme_pairs] += theme_weight

    scores = np.divide(
        base_scores,
        denominators,
        out=np.zeros_like(base_scores, dtype=np.float32),
        where=denominators > 0,
    )
    return scores * POP_MULTIPLIER * year_factor(index)


def block_similarities(index):
    """Her bloğun ayrı benzerlik vektörü; evaluate.py ağırlık denemeleri için."""
    sims = {}
    for name, matrix in sparse_blocks.items():
        sims[name] = (matrix @ matrix[index].T).toarray().ravel()

    if embeddings is not None:
        sims["overview"] = embeddings @ embeddings[index]

    if theme_profiles is not None:
        if theme_available[index]:
            theme_sims = theme_profiles @ theme_profiles[index]
            # NaN, bu film çifti için tema verisinin mevcut olmadığını belirtir.
            theme_sims = np.asarray(theme_sims, dtype=np.float32)
            theme_sims[~theme_available] = np.nan
        else:
            theme_sims = np.full(len(movies), np.nan, dtype=np.float32)
        sims["themes"] = theme_sims

    return sims


def combine(sims, weights, index):
    """Ağırlıklı benzerlik; eksik bloklar ilgili film çifti için normalize edilir."""
    numerator = np.zeros(len(movies), dtype=np.float32)
    denominator = np.zeros(len(movies), dtype=np.float32)

    for name, similarities in sims.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue

        values = np.asarray(similarities, dtype=np.float32)
        valid = np.isfinite(values)
        numerator += np.where(valid, values, 0.0) * weight
        denominator += valid.astype(np.float32) * weight

    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return scores * POP_MULTIPLIER * year_factor(index)


def component_sims(i, j):
    """Hata ayıklama: iki film arasındaki blok bazlı benzerlikler."""
    result = {}
    for name, matrix in sparse_blocks.items():
        result[name] = float((matrix[i] @ matrix[j].T).toarray()[0, 0])

    if embeddings is not None:
        result["overview"] = float(embeddings[i] @ embeddings[j])

    if theme_profiles is not None:
        if theme_available[i] and theme_available[j]:
            result["themes"] = float(theme_profiles[i] @ theme_profiles[j])
        else:
            result["themes"] = float("nan")

    return result


# =====================================================================

# 7. FİLTRE VE SEÇİM

# =====================================================================

def apply_filters(scores, index):

    """Kendisini, (istenirse) aynı isimlileri ve az oy alanları eler."""

    scores = scores.copy()

    scores[index] = -np.inf


    if EXCLUDE_SAME_TITLE:

        scores[(title_keys == title_keys[index]).to_numpy()] = -np.inf


    if HAS_VOTE_COUNT and MIN_VOTES:

        scores[(movies["vote_count"].fillna(0) < MIN_VOTES).to_numpy()] = -np.inf


    return scores


def pick_diverse(scores, k):

    """

    En yüksek skorlu adaylar arasından, aynı yönetmenin filmlerinden

    en fazla MAX_PER_DIRECTOR tanesini alarak k film seçer.

    """

    pool = np.argsort(scores)[::-1][:max(POOL_SIZE, k)]

    chosen = []

    director_count = {}


    for i in pool:

        if scores[i] <= 0:

            break


        directors = director_lists[i]

        if any(director_count.get(d, 0) >= MAX_PER_DIRECTOR for d in directors):

            continue


        chosen.append(i)

        for d in directors:

            director_count[d] = director_count.get(d, 0) + 1


        if len(chosen) == k:

            break


    return chosen


def top_recommendations(scores, index, k=TOP_N):

    """Filtre + çeşitlilik uygulayıp en iyi k filmin indekslerini verir."""

    return pick_diverse(apply_filters(scores, index), k)


# =====================================================================

# 8. ÖNERİ

# =====================================================================

def recommend(query, debug=False, verbose=True):

    found = find_movie(query)


    if found.empty:

        if verbose:

            print("Film bulunamadı!")

            suggestions = suggest_titles(query)

            if suggestions:

                print("Bunu mu demek istediniz:", ", ".join(suggestions))

        return []


    index = found.index[0]

    selected = movies.loc[index]


    if verbose:

        print("\nSeçilen film:", selected["title"], "|", selected["release_date"])


        if len(found) > 1:

            others = ", ".join(

                f"{row.title} ({row.year})"

                for row in found.iloc[1:6].itertuples()

            )

            print("Aynı isimli diğer filmler:", others)

            print("(Belirli birini seçmek için yıl ekle, örn: oldboy 2013)")


    scores = apply_filters(score_all(index), index)

    top_indices = pick_diverse(scores, min(TOP_N, len(movies) - 1))


    results = []

    for i in top_indices:

        row = movies.loc[i]

        item = {

            "id": int(row["id"]),

            "title": row["title"],

            "year": row["year"],

            "score": float(scores[i]),

        }

        if debug:

            item["parts"] = component_sims(index, i)

            item["pop_multiplier"] = float(POP_MULTIPLIER[i])

        results.append(item)


    return results


def print_results(results, debug=False):

    for item in results:

        print(f"{item['title']} ({item['year']})  Benzerlik: {item['score']:.4f}")


        if debug:

            parts = "  ".join(

                f"{name}={value:.2f}" for name, value in item["parts"].items()

            )

            print(f"    {parts}  pop_çarpanı={item['pop_multiplier']:.2f}")


# =====================================================================

# 9. TERMİNALDEN TEST

# =====================================================================

if __name__ == "__main__":

    debug_mode = False


    print("\nKomutlar: q = çıkış, /debug = blok skorlarını aç/kapat")


    while True:

        film = input("\nFilm adı (çıkış q): ").strip()


        if not film:

            continue

        if film.lower() == "q":

            break

        if film.lower() == "/debug":

            debug_mode = not debug_mode

            print("Debug:", "açık" if debug_mode else "kapalı")

            continue


        start = time.perf_counter()


        results = recommend(film, debug=debug_mode)

        print_results(results, debug=debug_mode)


        print("Sorgu süresi:", round(time.perf_counter() - start, 4), "saniye")
