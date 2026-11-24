"""
Model8 hızlı değerlendirme betiği.

Kullanım:
    python evaluate.py

- Her test filmi için ilk 10 öneriyi gösterir.
- İlk 10 ve ilk 20 başarı oranlarını hesaplar.
- themes ağırlığını model8.py içindeki değerinde sabit tutar.
- Overview, keywords, cf, genres, directors ve actors ağırlıklarını
  sırayla tarar; iki koordinat turuyla birlikte en iyi dengeyi arar.
- 50 farklı filmle genel öneri başarısını ölçer.
- Inception ve Black Swan genel test setinde kalır; özel hedef sıralaması yapılmaz.

Not: Bu betik tam kombinasyonlu grid search yapmaz. Her turda bir ağırlığı
    değiştirip diğerlerini sabit tutan koordinat araması kullanır. Böylece
    tüm kombinasyonları denemeden makul sürede daha geniş optimizasyon yapar.
"""

import numpy as np
import model8 as mr


# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------

K = 10          # Ekranda gösterilecek öneri sayısı
K_TUNE = 20     # Başarı değerlendirmesinde kullanılacak üst sınır

# themes=1.40 sabit tutulur. Diğer bloklar aşağıdaki değerlerle taranır.
# Her turda bir blok değiştirilir; diğer bloklar o ana kadarki en iyi
# değerlerinde kalır. Böylece tüm kombinasyonları denemeden hızlı arama yapılır.
TUNE_VALUES = {
    "overview":  [0.10, 0.18, 0.26, 0.35, 0.45, 0.60, 0.80],
    "keywords":  [0.10, 0.18, 0.26, 0.35, 0.45, 0.60, 0.80],
    "cf":        [0.10, 0.18, 0.26, 0.35, 0.45, 0.60, 0.80],
    "genres":    [0.03, 0.05, 0.07, 0.10, 0.15, 0.22, 0.30],
    "directors": [0.03, 0.05, 0.09, 0.12, 0.18, 0.25, 0.35],
    "actors":    [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25],
}
TUNE_ORDER = ["overview", "keywords", "cf", "genres", "directors", "actors"]
TUNE_ROUNDS = 2


# ---------------------------------------------------------
# TEST FİLMLERİ
# ---------------------------------------------------------
# Beklenen filmler mümkün olduğunca yıl ile belirtilir.
# Veri setinde bulunan üç film doğru yıllarıyla geri eklendi.
# Dune: Part Two (2024) ise CSV'ye ayrıca eklendi.
# Insidious için veri setindeki yıl (2011) kullanılır.

TEST_CASES = {
    "Interstellar": [
        "Gravity (2013)",
        "The Martian (2015)",
        "Arrival (2016)",
        "Contact (1997)",
        "Sunshine (2007)",
    ],

    "Fight Club": [
        "American Psycho (2000)",
        "Trainspotting (1996)",
        "Joker (2019)",
        "Taxi Driver (1976)",
    ],

    "Oldboy 2003": [
        "Lady Vengeance (2005)",
        "Sympathy for Mr. Vengeance (2002)",
        "I Saw the Devil (2010)",
        "Decision to Leave (2022)",
    ],

    "Se7en": [
        "Zodiac (2007)",
        "The Silence of the Lambs (1991)",
        "Prisoners (2013)",
        "Memories of Murder (2003)",
        "Cure (1997)",
    ],

    "Parasite 2019": [
        "Burning (2018)",
        "Snowpiercer (2013)",
        "Shoplifters (2018)",
        "Memories of Murder (2003)",
    ],

    "Dune 2021": [
        "Dune: Part Two (2024)",
        "Blade Runner 2049 (2017)",
        "Arrival (2016)",
    ],

    "Iron Man": [
        "Iron Man 2 (2010)",
        "Iron Man 3 (2013)",
        "The Avengers (2012)",
        "Captain America: The First Avenger (2011)",
    ],

    "Memories of Murder": [
        "Zodiac (2007)",
        "Se7en (1995)",
        "The Chaser (2008)",
        "I Saw the Devil (2010)",
    ],

    "The Matrix": [
        "The Matrix Reloaded (2003)",
        "Dark City (1998)",
        "Minority Report (2002)",
        "Ghost in the Shell (1995)",
    ],

    "Toy Story": [
        "Toy Story 2 (1999)",
        "Toy Story 3 (2010)",
        "Monsters, Inc. (2001)",
        "Finding Nemo (2003)",
    ],

    "The Dark Knight": [
        "Batman Begins (2005)",
        "The Dark Knight Rises (2012)",
        "Heat (1995)",
        "V for Vendetta (2006)",
    ],

    "Arrival 2016": [
        "Interstellar (2014)",
        "Blade Runner 2049 (2017)",
        "Contact (1997)",
        "The Martian (2015)",
        "Sunshine (2007)",
    ],

    "Seven Samurai 1954": [
        "The Hidden Fortress (1958)",
        "Yojimbo (1961)",
        "Sanjuro (1962)",
        "Harakiri (1962)",
        "13 Assassins (2010)",
    ],

    "The Last Samurai 2003": [
        "Gladiator (2000)",
        "Braveheart (1995)",
        "Kingdom of Heaven (2005)",
        "The Patriot (2000)",
        "Letters from Iwo Jima (2006)",
    ],

    "Deadpool 2016": [
        "Deadpool 2 (2018)",
        "Kick-Ass (2010)",
        "Guardians of the Galaxy (2014)",
        "Zombieland (2009)",
        "The Suicide Squad (2021)",
    ],

    "Harry Potter and the Philosopher's Stone 2001": [
        "Harry Potter and the Chamber of Secrets (2002)",
        "Harry Potter and the Prisoner of Azkaban (2004)",
        "Harry Potter and the Goblet of Fire (2005)",
        "The Lord of the Rings: The Fellowship of the Ring (2001)",
        "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe (2005)",
    ],

    "Whiplash 2014": [
        "Birdman or (The Unexpected Virtue of Ignorance) (2014)",
        "The Pianist (2002)",
        "Sound of Metal (2020)",
        "The Social Network (2010)",
    ],

    "Spirited Away 2001": [
        "Princess Mononoke (1997)",
        "Howl's Moving Castle (2004)",
        "My Neighbor Totoro (1988)",
        "Kiki's Delivery Service (1989)",
        "The Tale of the Princess Kaguya (2013)",
    ],

    "Mad Max: Fury Road 2015": [
        "Mad Max 2 (1981)",
        "Mad Max (1979)",
        "Dredd (2012)",
        "Children of Men (2006)",
        "Blade Runner 2049 (2017)",
    ],

    # Inception ve Black Swan genel test setinde kalıyor.
    # Özel ikili sıralama karşılaştırması yapılmıyor.
    "Inception": [
        "The Prestige (2006)",
        "Shutter Island (2010)",
        "The Matrix (1999)",
        "Memento (2000)",
        "Source Code (2011)",
        "Looper (2012)",
    ],

    "Black Swan": [
        "Requiem for a Dream (2000)",
        "Whiplash (2014)",
        "The Wrestler (2008)",
        "The Machinist (2004)",
        "mother! (2017)",
        "The Neon Demon (2016)",
    ],

    "The Godfather 1972": [
        "The Godfather Part II (1974)",
        "Goodfellas (1990)",
        "Casino (1995)",
        "Scarface (1983)",
        "Once Upon a Time in America (1984)",
    ],

    "Pulp Fiction 1994": [
        "Reservoir Dogs (1992)",
        "Jackie Brown (1997)",
        "Kill Bill: Vol. 1 (2003)",
        "Snatch (2000)",
        "Fargo (1996)",
    ],

    "The Shawshank Redemption 1994": [
        "The Green Mile (1999)",
        "Forrest Gump (1994)",
        "One Flew Over the Cuckoo's Nest (1975)",
        "Good Will Hunting (1997)",
        "Dead Poets Society (1989)",
    ],

    "The Lord of the Rings: The Fellowship of the Ring 2001": [
        "The Lord of the Rings: The Two Towers (2002)",
        "The Lord of the Rings: The Return of the King (2003)",
        "The Hobbit: An Unexpected Journey (2012)",
        "Harry Potter and the Prisoner of Azkaban (2004)",
        "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe (2005)",
    ],

    "Joker 2019": [
        "Taxi Driver (1976)",
        "The King of Comedy (1982)",
        "Nightcrawler (2014)",
        "American Psycho (2000)",
        "There Will Be Blood (2007)",
    ],

    "Titanic 1997": [
        "The Notebook (2004)",
        "Atonement (2007)",
        "Pearl Harbor (2001)",
        "Romeo + Juliet (1996)",
        "The English Patient (1996)",
    ],

    "Jurassic Park 1993": [
        "The Lost World: Jurassic Park (1997)",
        "Jurassic World (2015)",
        "Jaws (1975)",
        "King Kong (2005)",
        "Godzilla (2014)",
    ],

    "The Lion King 1994": [
        "Aladdin (1992)",
        "Beauty and the Beast (1991)",
        "Mulan (1998)",
        "The Jungle Book (1967)",
        "Finding Nemo (2003)",
    ],

    "The Silence of the Lambs 1991": [
        "Se7en (1995)",
        "Zodiac (2007)",
        "Red Dragon (2002)",
        "Hannibal (2001)",
        "Prisoners (2013)",
    ],

    "The Prestige 2006": [
        "The Illusionist (2006)",
        "Memento (2000)",
        "Shutter Island (2010)",
        "The Game (1997)",
        "Inception (2010)",
    ],

    "Blade Runner 2049 2017": [
        "Blade Runner (1982)",
        "Ex Machina (2015)",
        "Ghost in the Shell (1995)",
        "Minority Report (2002)",
        "Arrival (2016)",
    ],

    "The Grand Budapest Hotel 2014": [
        "Moonrise Kingdom (2012)",
        "The Royal Tenenbaums (2001)",
        "Fantastic Mr. Fox (2009)",
        "The Darjeeling Limited (2007)",
        "The Lobster (2015)",
    ],

    "La La Land 2016": [
        "Whiplash (2014)",
        "Once (2007)",
        "Singin' in the Rain (1952)",
        "Moulin Rouge! (2001)",
        "Chicago (2002)",
    ],

    "The Conjuring 2013": [
        "Insidious (2011)",
        "Sinister (2012)",
        "Annabelle (2014)",
        "The Exorcist (1973)",
        "The Babadook (2014)",
    ],

    "Get Out 2017": [
        "Us (2019)",
        "Nope (2022)",
        "The Cabin in the Woods (2012)",
        "The Thing (1982)",
        "Parasite (2019)",
    ],

    "The Avengers 2012": [
        "Avengers: Age of Ultron (2015)",
        "Avengers: Infinity War (2018)",
        "Guardians of the Galaxy (2014)",
        "Iron Man (2008)",
        "Thor: Ragnarok (2017)",
    ],

    "The Social Network 2010": [
        "Steve Jobs (2015)",
        "Moneyball (2011)",
        "The Big Short (2015)",
        "Spotlight (2015)",
        "The Imitation Game (2014)",
    ],

    "Forrest Gump 1994": [
        "The Shawshank Redemption (1994)",
        "The Green Mile (1999)",
        "Cast Away (2000)",
        "Big Fish (2003)",
        "Rain Man (1988)",
    ],

    "Back to the Future 1985": [
        "Back to the Future Part II (1989)",
        "Back to the Future Part III (1990)",
        "Groundhog Day (1993)",
        "E.T. the Extra-Terrestrial (1982)",
        "Bill & Ted's Excellent Adventure (1989)",
    ],

    "Alien 1979": [
        "Aliens (1986)",
        "The Thing (1982)",
        "Predator (1987)",
        "Event Horizon (1997)",
        "The Fly (1986)",
    ],

    "The Truman Show 1998": [
        "Eternal Sunshine of the Spotless Mind (2004)",
        "Groundhog Day (1993)",
        "Being John Malkovich (1999)",
        "Stranger Than Fiction (2006)",
        "Pleasantville (1998)",
    ],

    "Goodfellas 1990": [
        "The Godfather (1972)",
        "Casino (1995)",
        "Donnie Brasco (1997)",
        "Scarface (1983)",
        "The Departed (2006)",
    ],

    "The Incredibles 2004": [
        "Ratatouille (2007)",
        "Finding Nemo (2003)",
        "Monsters, Inc. (2001)",
        "Toy Story (1995)",
        "Up (2009)",
    ],

    "Finding Nemo 2003": [
        "Monsters, Inc. (2001)",
        "Toy Story (1995)",
        "Up (2009)",
        "The Incredibles (2004)",
        "Shark Tale (2004)",
    ],

    "Coco 2017": [
        "Inside Out (2015)",
        "Soul (2020)",
        "Encanto (2021)",
        "Up (2009)",
        "Ratatouille (2007)",
    ],

    "The Notebook 2004": [
        "A Walk to Remember (2002)",
        "Pride & Prejudice (2005)",
        "Titanic (1997)",
        "Dear John (2010)",
        "The Vow (2012)",
    ],

    "The Bourne Identity 2002": [
        "The Bourne Supremacy (2004)",
        "The Bourne Ultimatum (2007)",
        "Casino Royale (2006)",
        "Taken (2008)",
        "The Fugitive (1993)",
    ],

    "The Terminator 1984": [
        "Terminator 2: Judgment Day (1991)",
        "RoboCop (1987)",
        "The Matrix (1999)",
        "Predator (1987)",
        "Total Recall (1990)",
    ],

    "The Exorcist 1973": [
        "The Omen (1976)",
        "Rosemary's Baby (1968)",
        "The Conjuring (2013)",
        "Poltergeist (1982)",
        "Hereditary (2018)",
    ],
}


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------

def split_year(text):
    """'Başlık (2003)' -> ('Başlık', '2003')."""
    import re

    match = re.match(r"^(.*?)\s*\((\d{4})\)$", str(text).strip())
    if match:
        return match.group(1).strip(), match.group(2)
    return str(text).strip(), None


def title_mask(text):
    title, year = split_year(text)
    mask = np.array(
        mr.title_keys == mr.normalize_title(title),
        dtype=bool,
        copy=True,
    )
    if year:
        mask &= np.asarray(mr.years).astype(str) == year
    return mask


def exists_in_dataset(text):
    return bool(title_mask(text).any())


def is_match(expected, index):
    title, year = split_year(expected)
    if mr.title_keys[index] != mr.normalize_title(title):
        return False
    return year is None or str(mr.years[index]) == year


def case_hits(expected, top_indices):
    return [
        title
        for title in expected
        if any(is_match(title, int(i)) for i in top_indices)
    ]


def fast_top_recommendations(scores, index, k):
    """Model8'in çeşitlilik kuralını koruyarak tam sıralamadan ilk k'yı seçer.

    Eski pick_diverse fonksiyonu 25.000 filmin tamamını sıralıyordu.
    Burada önce yalnızca en yüksek puanlı aday havuzu seçilir, sonra bu
    küçük havuz sıralanır. Bu, tekrar eden değerlendirmeleri hızlandırır.
    """
    filtered = mr.apply_filters(scores, index)
    pool_size = min(max(int(mr.POOL_SIZE), int(k)), len(filtered))

    if pool_size == 0:
        return []

    # En yüksek puanlı pool_size adayı bul; yalnızca bu küçük grubu sırala.
    split_at = len(filtered) - pool_size
    candidate_indices = np.argpartition(filtered, split_at)[split_at:]
    candidate_indices = candidate_indices[
        np.argsort(filtered[candidate_indices])[::-1]
    ]

    chosen = []
    director_count = {}

    for raw_index in candidate_indices:
        i = int(raw_index)
        if filtered[i] <= 0:
            break

        directors = mr.director_lists[i]
        if any(
            director_count.get(d, 0) >= mr.MAX_PER_DIRECTOR
            for d in directors
        ):
            continue

        chosen.append(i)
        for director in directors:
            director_count[director] = director_count.get(director, 0) + 1

        if len(chosen) == k:
            break

    return chosen


def evaluate(prepared, weights, verbose=False):
    """İlk 10 ve ilk 20 başarı oranlarını hesaplar."""
    total_top10 = 0.0
    total_top20 = 0.0

    for case in prepared:
        scores = mr.combine(case["sims"], weights, case["index"])
        top20 = fast_top_recommendations(scores, case["index"], K_TUNE)
        top10 = top20[:K]

        hits10 = case_hits(case["expected"], top10)
        hits20 = case_hits(case["expected"], top20)

        denom10 = min(len(case["expected"]), K)
        denom20 = min(len(case["expected"]), K_TUNE)

        total_top10 += len(hits10) / denom10
        total_top20 += len(hits20) / denom20

        if verbose:
            titles = ", ".join(
                f"{mr.movies.at[i, 'title']} ({mr.movies.at[i, 'year']})"
                for i in top10
            )
            print(f"\n{case['query']} -> {len(hits10)}/{denom10}")
            print(f"   ilk 10 öneri: {titles}")

            if hits10:
                print(f"   bulunanlar: {', '.join(hits10)}")

            missed = [title for title in case["expected"] if title not in hits10]
            if missed:
                filtered = mr.apply_filters(scores, case["index"])
                details = []
                for expected_title in missed:
                    indices = np.flatnonzero(title_mask(expected_title))
                    ranks = [
                        1 + int(np.sum(filtered > filtered[int(i)]))
                        for i in indices
                        if np.isfinite(filtered[int(i)])
                    ]
                    details.append(
                        f"{expected_title}: #{min(ranks)}"
                        if ranks else f"{expected_title}: bulunamadı"
                    )
                print(f"   ilk 10 dışında: {', '.join(details)}")

    count = len(prepared)
    return total_top10 / count, total_top20 / count


def prepare_cases():
    """Filmleri bulur ve benzerlik bloklarını yalnızca bir kez hesaplar."""
    prepared = []

    for query, expected in TEST_CASES.items():
        found = mr.find_movie(query)
        if found.empty:
            print(f"! '{query}' veri setinde yok, atlandı.")
            continue

        available = [title for title in expected if exists_in_dataset(title)]
        missing = [title for title in expected if title not in available]

        if missing:
            print(
                f"  {query}: veri setinde olmayan beklenen filmler -> "
                f"{', '.join(missing)}"
            )

        if not available:
            print(f"! '{query}' için veri setinde beklenen film bulunamadı.")
            continue

        index = int(found.index[0])
        prepared.append({
            "query": query,
            "index": index,
            "expected": available,
            "sims": mr.block_similarities(index),
        })

    return prepared


# ---------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------

def balanced_score(score10, score20):
    """İlk 10 ve ilk 20 başarısına eşit önem veren seçim ölçütü."""
    return (score10 + score20) / 2.0


def run_weight_search(prepared, initial_weights):
    """Tema ağırlığını sabit tutup diğer ağırlıkları koordinat aramasıyla ayarlar."""
    weights = {name: float(value) for name, value in initial_weights.items()}
    fixed_theme = weights.get("themes")

    baseline10, baseline20 = evaluate(prepared, weights, verbose=False)
    print(
        f"\nBaşlangıç skoru: ilk10={baseline10:.3f}, "
        f"ilk20={baseline20:.3f}, dengeli={balanced_score(baseline10, baseline20):.3f}"
    )
    if fixed_theme is not None:
        print(f"Themes sabit tutuluyor: {fixed_theme:.2f}")

    history = []
    for round_no in range(1, TUNE_ROUNDS + 1):
        print(f"\n=== Ağırlık optimizasyonu: Tur {round_no}/{TUNE_ROUNDS} ===")
        changed = False

        for name in TUNE_ORDER:
            if name not in weights:
                print(f"\n{name}: modelde etkin değil, atlandı.")
                continue
            if name == "cf" and not any("cf" in case["sims"] for case in prepared):
                print("\ncf: MovieLens verisi bulunmadığı için atlandı.")
                continue

            current_value = weights[name]
            candidates = sorted(set(float(x) for x in TUNE_VALUES[name]) | {current_value})
            rows = []

            for candidate in candidates:
                trial = dict(weights)
                trial[name] = candidate
                score10, score20 = evaluate(prepared, trial, verbose=False)
                objective = balanced_score(score10, score20)
                rows.append((candidate, score10, score20, objective))

            # Dengeli skor eşitse ilk 10, sonra ilk 20 tercih edilir.
            # Tam eşitlikte mevcut değeri koruyarak gereksiz değişimi önler.
            best = max(
                rows,
                key=lambda row: (
                    row[3],
                    row[1],
                    row[2],
                    np.isclose(row[0], current_value),
                ),
            )

            print(f"\n{name} (diğer ağırlıklar sabit):")
            print("  değer | ilk 10 | ilk 20 | dengeli")
            for candidate, score10, score20, objective in rows:
                marker = "  <-- seçildi" if np.isclose(candidate, best[0]) else ""
                print(
                    f"  {candidate:>5.2f} | {score10:.3f}  | {score20:.3f}  "
                    f"| {objective:.3f}{marker}"
                )

            if not np.isclose(best[0], current_value):
                changed = True
                weights[name] = best[0]
                print(f"  Güncellendi: {current_value:.2f} -> {best[0]:.2f}")
            else:
                print(f"  Değişmedi: {current_value:.2f}")

            history.append({
                "round": round_no,
                "name": name,
                "selected": weights[name],
                "score10": best[1],
                "score20": best[2],
            })

        if not changed:
            print("\nBu turda ağırlık değişmedi; arama erken durduruldu.")
            break

    final10, final20 = evaluate(prepared, weights, verbose=False)
    return weights, (baseline10, baseline20), (final10, final20), history


if __name__ == "__main__":
    prepared = prepare_cases()
    if not prepared:
        raise SystemExit("Değerlendirilecek film bulunamadı.")

    first_sims = prepared[0]["sims"]
    has_cf = any("cf" in case["sims"] for case in prepared)
    has_themes = any(
        "themes" in case["sims"]
        and np.isfinite(np.asarray(case["sims"]["themes"])).any()
        for case in prepared
    )
    theme_cases = sum(
        "themes" in case["sims"]
        and np.isfinite(np.asarray(case["sims"]["themes"])).any()
        for case in prepared
    )

    print(
        f"\n{len(prepared)} test filmi değerlendiriliyor "
        f"(ilk {K} ve ilk {K_TUNE})."
    )
    print("MovieLens (cf) bloğu:", "aktif" if has_cf else "yok")
    print(
        "Tag Genome (themes) bloğu:",
        f"aktif ({theme_cases}/{len(prepared)} test filminde)"
        if has_themes else "kullanılabilir tema verisi yok",
    )

    print("\n=== Başlangıç model8 ağırlıkları ===")
    for name, weight in mr.WEIGHTS.items():
        print(f"{name}: {weight:.3f}")

    # Başlangıç modelinin 50 filmdeki önerilerini ve metriklerini göster.
    current_top10, current_top20 = evaluate(prepared, mr.WEIGHTS, verbose=True)
    print(
        f"\nBaşlangıç model8 skoru: ilk10={current_top10:.3f}, "
        f"ilk20={current_top20:.3f}"
    )

    if not has_themes:
        print("\nUYARI: Kullanılabilir Tag Genome profili yok; themes ağırlığı etkisiz olabilir.")

    optimized, baseline, final, history = run_weight_search(prepared, mr.WEIGHTS)

    print("\n=== OPTİMİZASYON SONUCU ===")
    print(
        f"Başlangıç: ilk10={baseline[0]:.3f}, ilk20={baseline[1]:.3f}, "
        f"dengeli={balanced_score(*baseline):.3f}"
    )
    print(
        f"Optimize:  ilk10={final[0]:.3f}, ilk20={final[1]:.3f}, "
        f"dengeli={balanced_score(*final):.3f}"
    )
    print("\nÖnerilen WEIGHTS ayarları (model8.py içine elle aktarılabilir):")
    for name in mr.WEIGHTS:
        print(f'    "{name}": {optimized.get(name, mr.WEIGHTS[name]):.2f},')

    print("\nOptimizasyon tamamlandı. model8.py otomatik değiştirilmedi.")
