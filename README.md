# 📰 Fake News Detector (AI / Machine Learning / Python / Streamlit)

Paste a news headline or article and the app tells you if it **looks FAKE or REAL**, how confident it is,
**which words influenced the decision**, and a simple **red-flag checklist** (ALL-CAPS, clickbait, missing sources).

## Features
- Cleans text and converts it to numbers with **TF-IDF** (words + word pairs)
- Trains and compares **3 ML models**: Logistic Regression, Naive Bayes, Linear SVM (best one is used automatically)
- Shows Accuracy, Precision, Recall, F1 and a confusion matrix
- **Explainable**: shows the words that pushed your text towards FAKE or REAL
- Train on **your own CSV** or the Kaggle dataset directly from the app
- Works out of the box (no download needed) using a small demo dataset

## Quick start (5 minutes)
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first launch trains the model automatically (a few seconds). That's it.

## Use REAL data for real results (recommended)
1. Download the free **"Fake and Real News Dataset"** (ISOT) from Kaggle.
2. Put `Fake.csv` and `True.csv` inside the `data/` folder.
3. Run `python -m src.train` (or press *Retrain* in the app's **Train / Dataset** tab).

Or use your own file: a CSV with columns `text` and `label` (`FAKE`/`REAL` or `1`/`0`) saved as `data/news.csv`,
or uploaded in the app.

> The built-in demo dataset is **synthetic** (made from sentence templates). Its scores (about 100%) are not meaningful.
> Real datasets usually give about 95-99% on ISOT, and much less on news from other sources.

## Project structure
```
fake_news_detector/
├── app.py                    # Streamlit web app (4 tabs)
├── requirements.txt
├── data/                     # sample_news.csv (demo). Put Fake.csv + True.csv here
├── models/                   # saved model (created automatically)
├── src/
│   ├── config.py             # paths and settings
│   ├── generate_sample_data.py  # creates the demo dataset
│   ├── data_loader.py        # loads + validates data, normalises labels
│   ├── preprocess.py         # text cleaning
│   ├── train.py              # TF-IDF + 3 models + metrics + save best
│   ├── predict.py            # predict + word-level explanation
│   └── signals.py            # rule-based red-flag checklist
└── tests/test_basic.py       # quick self-check:  python -m tests.test_basic
```

## How it works
1. **Preprocess**: lowercase, remove links/HTML/numbers/symbols and agency tags like *(Reuters)* (they leak the answer).
2. **TF-IDF**: each article becomes a vector; distinctive words get higher weights.
3. **Train/test split**: 80% train, 20% unseen test (stratified, fixed seed 42).
4. **Models**: Logistic Regression, Multinomial Naive Bayes, Linear SVM (calibrated for probabilities).
5. **Pick best** by F1-score on the fake class and save with `joblib`.
6. **Explain**: contribution of a word = its TF-IDF value in your text x its Logistic Regression weight.

## Viva / interview questions
- **Why TF-IDF and not plain word counts?** It down-weights very common words and highlights distinctive ones.
- **Why F1 and not only accuracy?** F1 balances precision and recall, which matters when classes are uneven.
- **Why Logistic Regression for explanation?** Its weights can be read directly: positive = fake, negative = real.
- **Can it detect false facts?** No. It learns writing style, not truth. It cannot check facts against the world.
- **How to improve?** Larger and more recent data, multilingual data, transformer models (BERT), source and metadata features.

## Limitations
The model reflects its training data. It can be fooled by well-written false stories and may flag honest but dramatic
writing. Use it as a hint, and always verify with trusted sources and fact-checking sites.
