"""Fake News Detector - Streamlit app.   Run:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import FAKE, LABEL_NAMES, MODEL_PATH
from src.data_loader import prepare_frame
from src.predict import explain_text, load_bundle, predict_text
from src.signals import analyse_signals, signal_score
from src.train import train_and_save

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

EXAMPLE_FAKE = ("SHOCKING!!! Secret documents EXPOSED: the government is hiding the truth about the new vaccine. "
                "Doctors HATE this miracle trick that the mainstream media banned. Share before it is deleted!!!")
EXAMPLE_REAL = ("The finance ministry announced on Tuesday that the quarterly growth figure rose by 6 percent, "
                "according to data published by the national statistics office. A spokesperson said the report "
                "will be reviewed by a committee before any final decision.")


@st.cache_resource(show_spinner="Loading model (first run trains it, takes a few seconds)...")
def get_bundle():
    return load_bundle()


def set_text(value):
    st.session_state["news_text"] = value


def explanation_chart(items):
    words = [w for w, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(5.5, 0.4 * len(items) + 1))
    ax.barh(words, vals, color=["#D64545" if v > 0 else "#2E9E6B" for v in vals])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("<- pushes towards REAL        pushes towards FAKE ->")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


bundle = get_bundle()
is_demo = "DEMO" in bundle.get("source", "")

st.title("📰 Fake News Detector")
st.caption("A machine-learning tool that reads a news text and estimates whether it looks FAKE or REAL.")

if is_demo:
    st.warning("**Demo mode:** the model is trained on a small *synthetic* dataset so the app works out of the box. "
               "Its scores are not meaningful for real news. Open the **Train / Dataset** tab to train on real data.")

with st.sidebar:
    st.header("Current model")
    st.write(f"**Algorithm:** {bundle['best_name']}")
    st.write(f"**Data:** {bundle['source']}")
    st.write(f"**Articles:** {bundle['n_fake'] + bundle['n_real']:,}")
    best = bundle["metrics"].iloc[0]
    st.metric("Test accuracy", f"{best['Accuracy']:.1%}")
    st.metric("Test F1 (fake class)", f"{best['F1']:.3f}")
    st.divider()
    st.caption("This tool is an aid, not a judge. Always verify important news with trusted sources.")

tab_check, tab_perf, tab_train, tab_about = st.tabs(["🔍 Check News", "📊 Model Performance", "🧠 Train / Dataset", "ℹ️ About"])

# ---------------------------------------------------------------- Check News
with tab_check:
    st.subheader("Paste a headline or article")
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("Try a fake-style example", on_click=set_text, args=(EXAMPLE_FAKE,))
    c2.button("Try a real-style example", on_click=set_text, args=(EXAMPLE_REAL,))
    text = st.text_area("News text", key="news_text", height=200,
                        placeholder="Paste the headline and body of the news here...")

    if st.button("Analyze", type="primary"):
        if len(text.split()) < 5:
            st.error("Please paste at least a full sentence (5+ words) for a meaningful result.")
        else:
            res = predict_text(bundle, text)
            name = LABEL_NAMES[res["label"]]
            left, right = st.columns([1, 1])
            with left:
                if res["label"] == FAKE:
                    st.error(f"### 🚨 Looks {name}")
                else:
                    st.success(f"### ✅ Looks {name}")
                st.write(f"Model confidence: **{res['confidence']:.1%}**")
                st.progress(res["p_fake"], text=f"Probability of FAKE: {res['p_fake']:.1%}")
                if res["confidence"] < 0.65:
                    st.info("The model is unsure about this text. Treat the result with extra caution.")
                if len(text.split()) < 15:
                    st.info("Short text: results are more reliable with a full article.")
            with right:
                st.markdown("**Red-flag checklist** (simple rules, independent of the model)")
                sig = analyse_signals(text)
                st.dataframe(pd.DataFrame([{"Check": n, "Found": str(v), "Status": "⚠️ flag" if s == "flag" else "✅ ok"}
                                           for n, v, s in sig]), hide_index=True)
                st.caption(f"{signal_score(sig)} of {len(sig)} checks raised a flag.")

            st.markdown("**Why did the model decide this?** Words with the biggest influence in your text:")
            items = explain_text(bundle, text)
            if items:
                fig_exp = explanation_chart(items)
                st.pyplot(fig_exp)
                plt.close(fig_exp)
            else:
                st.info("None of the words in this text were seen during training, so no explanation is available.")

# ---------------------------------------------------------------- Performance
with tab_perf:
    st.subheader("How well do the models perform?")
    st.write(f"Each model was trained on {bundle['n_train']:,} articles and tested on {bundle['n_test']:,} unseen articles. "
             f"**{bundle['best_name']}** scored best and is used by the app.")
    m = bundle["metrics"]
    st.dataframe(m.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                 hide_index=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**Accuracy and F1 by model**")
        st.bar_chart(m.set_index("Model")[["Accuracy", "F1"]])
    with b:
        st.markdown(f"**Confusion matrix ({bundle['best_name']})**")
        cm = bundle["confusion"]
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["FAKE", "REAL"]); ax.set_yticks([0, 1], ["FAKE", "REAL"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                        color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("Precision: of the articles called FAKE, how many really are. Recall: of all FAKE articles, how many were caught. "
               "F1: the balance of the two.")
    st.markdown("**Words the model associates most with each class**")
    w1, w2 = st.columns(2)
    w1.write("🚨 **FAKE:** " + ", ".join(w for w, _ in bundle["top_words"]["fake"][:12]))
    w2.write("✅ **REAL:** " + ", ".join(w for w, _ in bundle["top_words"]["real"][:12]))

# ---------------------------------------------------------------- Train
with tab_train:
    st.subheader("Train on your own data")
    st.write(f"The current model was trained on: **{bundle['source']}** "
             f"({bundle['n_fake']:,} fake, {bundle['n_real']:,} real).")
    st.markdown("""
**Best results with real data.** Download the free *Fake and Real News Dataset* from Kaggle and put `Fake.csv` and
`True.csv` inside the `data/` folder, then press the button below. Or upload any CSV here with two columns:
`text` and `label` (values `FAKE`/`REAL` or `1`/`0`).
""")
    up = st.file_uploader("Upload a CSV file", type="csv")
    if up is not None:
        try:
            df = prepare_frame(pd.read_csv(up))
            st.success(f"File accepted: {len(df):,} articles ({int((df['label'] == FAKE).sum()):,} fake, "
                       f"{int((df['label'] != FAKE).sum()):,} real).")
            st.dataframe(df.head(5), hide_index=True)
            if len(df) < 50:
                st.warning("Very small dataset: results will not be reliable.")
            if st.button("Train on uploaded file", type="primary"):
                with st.spinner("Training three models..."):
                    train_and_save(df, f"uploaded file ({up.name})")
                get_bundle.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Could not use this file: {e}")
    st.divider()
    if st.button("Retrain from the data/ folder (Kaggle files, news.csv or demo data)"):
        from src.data_loader import load_dataset
        with st.spinner("Training three models..."):
            d, s = load_dataset()
            train_and_save(d, s)
        get_bundle.clear()
        st.rerun()

# ---------------------------------------------------------------- About
with tab_about:
    st.subheader("How it works")
    st.markdown("""
1. **Clean** the text: lowercase, remove links, symbols and agency tags such as *(Reuters)*.
2. **TF-IDF** converts every article into numbers: words and word pairs that are distinctive get high weight.
3. **Three classic ML models** are trained and compared: Logistic Regression, Naive Bayes and Linear SVM.
4. The model with the best F1-score is saved and used for predictions.
5. **Explainability:** Logistic Regression word weights show which words pushed your text towards FAKE or REAL.
6. A separate **red-flag checklist** looks for ALL-CAPS, many exclamation marks, clickbait phrases and missing sources.
""")
    st.subheader("Limitations (please read)")
    st.markdown("""
- The model learns the **writing style** of fake and real articles in its training data. It does **not** check facts
  against the real world. A well-written false story can look REAL, and an honest but dramatic story can look FAKE.
- Accuracy on one dataset does not guarantee accuracy on news from another country, language or time period.
- Use the result as a hint to investigate further: check the source, the date, other outlets and fact-checking sites.
""")
