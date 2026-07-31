# Movie Recommendation System — Content-Based & Collaborative Filtering

## 📌 Objective
Build a movie recommendation system using two complementary approaches: **content-based
filtering** (genre similarity) and **collaborative filtering** (SVD matrix factorization).

## 📊 Dataset
**MovieLens Small Dataset (ml-latest-small)** — 100,836 ratings, 610 users, 9,742 movies
Source: [GroupLens — MovieLens](https://grouplens.org/datasets/movielens/latest/)

*(The dataset is not uploaded to this repo — the notebook loads it automatically from a public mirror.)*

## 🛠️ Libraries Used
- `pandas` / `numpy` — data loading and manipulation
- `scikit-learn` — TF-IDF vectorization, cosine similarity (content-based filtering)
- `scikit-surprise` — SVD matrix factorization (collaborative filtering), the same algorithm
  family that won the Netflix Prize
- `matplotlib` — visualization (rating distribution)

## 🔍 Methodology
1. **Data Understanding** — loaded 100,836 ratings across 9,742 movies, explored rating
   distribution (skews positive) and genre distribution (Drama, Comedy, Thriller most common).
2. **Content-Based Filtering** — converted each movie's genre list into a TF-IDF vector,
   computed pairwise cosine similarity between all movies, and built a function to recommend
   similar movies given a title.
3. **Collaborative Filtering (SVD)** — trained an SVD matrix factorization model (50 latent
   factors) using `scikit-surprise`, evaluated with an 80/20 split and 5-fold cross-validation.
4. **Top-N Recommendations** — retrained on the full dataset and built a function generating
   personalized top-10 movie recommendations for any user, excluding movies they've already rated.

## 📈 Results

| Metric (Collaborative Filtering / SVD) | Value |
|------------------------------------------|-------|
| RMSE | ≈ 0.87 |
| MAE  | ≈ 0.67 |

(Consistent across 5-fold cross-validation: RMSE 0.870 ± 0.001, MAE 0.669 ± 0.002)

**Content-based example:** Recommending "Toy Story (1995)" surfaces other animated
family/comedy films (Antz, Toy Story 2, Monsters Inc.) — thematically coherent but genre-bound.

**Collaborative filtering example:** Recommendations for a sample user span multiple genres
(The Godfather, The Shawshank Redemption, The Dark Knight, Lord of the Rings) — reflecting
learned cross-genre taste patterns rather than genre overlap alone.

## ✅ Conclusion
This project built a movie recommendation system using two complementary techniques.
Content-based filtering, using TF-IDF vectors over movie genres and cosine similarity,
successfully recommends movies with a similar thematic feel to a given movie, but is
inherently limited to genre overlap and cannot capture broader taste patterns across genres.
Collaborative filtering, implemented via SVD matrix factorization (the same technique family
that won the Netflix Prize), achieved a strong RMSE of approximately 0.87 and MAE of
approximately 0.67 on a 0.5-5.0 rating scale, and produced genuinely personalized
recommendations spanning multiple genres based purely on patterns in user rating behavior,
without requiring any movie metadata. A key limitation of collaborative filtering is the
**"cold start" problem**: it cannot generate meaningful recommendations for brand-new users
with no rating history, or brand-new movies with no ratings yet, since it has no signal to
learn from in either case — a scenario where content-based filtering (or a hybrid of both
approaches) would need to take over until enough interaction data accumulates. In production
recommendation systems, these two approaches are frequently combined into a **hybrid model**
to get the benefits of both: content-based filtering for cold-start scenarios, and
collaborative filtering for richer, cross-genre personalization once sufficient data exists.

## 📂 Files
- `MovieRecommendationSystem.ipynb` — full notebook with both recommendation approaches, evaluation, and examples
