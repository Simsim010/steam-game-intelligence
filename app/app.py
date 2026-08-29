import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Steam Game Intelligence",
    page_icon="🎮",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sentiment_model.joblib"
)

TFIDF_PATH = os.path.join(
    BASE_DIR,
    "model",
    "tfidf_vectorizer.joblib"
)

GAME_ANALYSIS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "game_analysis.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return model, tfidf


# ============================================================
# LOAD GAME DATA
# ============================================================

@st.cache_data
def load_game_data():

    data = pd.read_csv(GAME_ANALYSIS_PATH)

    # Eğer index CSV'ye kaydedildiyse
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    return data


model, tfidf = load_model()
game_analysis = load_game_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎮 Steam Game Intelligence")

st.sidebar.markdown(
    """
    ### Navigation

    Use the menu below to explore the project.
    """
)

page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Dashboard",
        "💬 Review Analyzer",
        "🎮 Game Analysis",
        "📊 Model Performance",
        "🔍 Error Analysis"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🎮 Steam Game Intelligence")

    st.markdown(
        """
        ### Machine Learning-Based Steam Review Analysis

        This application analyzes Steam game reviews using
        **TF-IDF + Logistic Regression**.

        The model was trained to classify Steam reviews as
        positive or negative and generate a sentiment score.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Logistic Regression"
        )

    with col2:
        st.metric(
            "Accuracy",
            "92.22%"
        )

    with col3:
        st.metric(
            "ROC-AUC",
            "0.9593"
        )

    with col4:
        st.metric(
            "Final Threshold",
            "0.20"
        )

    st.divider()

    # --------------------------------------------------------
    # GAME STATISTICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Analyzed Games",
            f"{len(game_analysis):,}"
        )

    with col2:
        st.metric(
            "Reviews per Game",
            "≥ 100"
        )

    with col3:
        st.metric(
            "Game-Level Correlation",
            "0.9824"
        )

    st.divider()

    st.subheader("📌 Project Overview")

    st.markdown(
        """
        **Pipeline**

        `Steam Reviews`
        ↓
        `Data Cleaning`
        ↓
        `TF-IDF`
        ↓
        `Logistic Regression`
        ↓
        `Sentiment Score`
        ↓
        `Game-Level Analysis`

        ### Key Findings

        - Logistic Regression slightly outperformed Linear SVM.
        - The final classification threshold was optimized to **0.20**.
        - Accuracy reached **92.22%**.
        - ROC-AUC reached **0.9593**.
        - Game-level AI sentiment showed a correlation of **0.9824**
          with the actual positive review rate.
        """
    )


# ============================================================
# REVIEW ANALYZER
# ============================================================

elif page == "💬 Review Analyzer":

    st.title("💬 Steam Review Analyzer")

    st.markdown(
        """
        Enter a Steam review and let the machine learning model
        estimate its sentiment.
        """
    )

    review = st.text_area(
        "Enter your review:",
        height=180,
        placeholder=(
            "Example: This game is absolutely amazing "
            "and I highly recommend it!"
        )
    )

    analyze = st.button(
        "🤖 Analyze Review",
        use_container_width=True
    )

    if analyze:

        if not review.strip():

            st.warning(
                "Please enter a review first."
            )

        else:

            # ------------------------------------------------
            # TF-IDF TRANSFORMATION
            # ------------------------------------------------

            review_tfidf = tfidf.transform(
                [review]
            )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probability = model.predict_proba(
                review_tfidf
            )[0][1]

            # ------------------------------------------------
            # FINAL THRESHOLD
            # ------------------------------------------------

            threshold = 0.20

            prediction = (
                probability >= threshold
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            if prediction:

                st.success(
                    "🟢 Positive Review"
                )

            else:

                st.error(
                    "🔴 Negative Review"
                )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "AI Sentiment Score",
                    f"{probability:.2%}"
                )

            with col2:

                st.metric(
                    "Threshold",
                    f"{threshold:.2f}"
                )

            with col3:

                if probability >= 0.80:

                    confidence = "High"

                elif probability >= 0.50:

                    confidence = "Medium"

                else:

                    confidence = "Low"

                st.metric(
                    "Confidence",
                    confidence
                )

            # ------------------------------------------------
            # PROBABILITY BAR
            # ------------------------------------------------

            st.subheader(
                "Sentiment Probability"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                "The score represents the model's estimated "
                "probability of a positive review."
            )

            # ------------------------------------------------
            # REVIEW
            # ------------------------------------------------

            st.subheader(
                "Analyzed Review"
            )

            st.info(review)


# ============================================================
# GAME ANALYSIS
# ============================================================

elif page == "🎮 Game Analysis":

    st.title("🎮 Game-Level Analysis")

    st.markdown(
        """
        Compare the actual Steam positive review rate
        with the AI-generated sentiment score.
        """
    )

    # --------------------------------------------------------
    # GAME SELECTOR
    # --------------------------------------------------------

    selected_game = st.selectbox(
        "Select a game:",
        sorted(
            game_analysis["name"].dropna().unique()
        )
    )

    game = game_analysis[
        game_analysis["name"] == selected_game
    ].iloc[0]

    st.divider()

    # --------------------------------------------------------
    # GAME METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Review Count",
            f"{int(game['review_count']):,}"
        )

    with col2:

        st.metric(
            "Actual Positive Rate",
            f"{game['actual_positive_rate']:.2%}"
        )

    with col3:

        st.metric(
            "AI Sentiment Score",
            f"{game['ai_sentiment_score']:.2%}"
        )

    with col4:

        difference = (
            game["score_difference"]
        )

        st.metric(
            "AI - Actual",
            f"{difference:+.2%}"
        )

    st.divider()

    # --------------------------------------------------------
    # COMPARISON CHART
    # --------------------------------------------------------

    st.subheader(
        "Actual Positive Rate vs AI Sentiment"
    )

    chart_data = pd.DataFrame(
        {
            "Score": [
                game["actual_positive_rate"],
                game["ai_sentiment_score"]
            ]
        },
        index=[
            "Actual Positive Rate",
            "AI Sentiment Score"
        ]
    )

    st.bar_chart(
        chart_data
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if abs(difference) < 0.05:

        st.success(
            "The AI sentiment score is very close "
            "to the actual positive review rate."
        )

    elif difference > 0:

        st.info(
            "The AI model estimates a more positive sentiment "
            "than the actual Steam recommendation rate."
        )

    else:

        st.warning(
            "The AI model estimates a less positive sentiment "
            "than the actual Steam recommendation rate."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.markdown(
        """
        Two machine learning models were evaluated:

        - Logistic Regression
        - Linear Support Vector Machine
        """
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Linear SVM"
            ],
            "Accuracy": [
                0.922211,
                0.8987
            ],
            "ROC-AUC": [
                0.959301,
                0.9557
            ]
        }
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.subheader(
        "🏆 Selected Model"
    )

    st.success(
        """
        Logistic Regression was selected as the final model
        because it achieved the best overall performance.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "92.22%"
        )

    with col2:

        st.metric(
            "Precision",
            "93.74%"
        )

    with col3:

        st.metric(
            "Recall",
            "96.96%"
        )

    with col4:

        st.metric(
            "F1 Score",
            "95.32%"
        )

    st.divider()

    st.subheader(
        "Model Comparison"
    )

    chart_data = results.set_index(
        "Model"
    )[
        ["Accuracy", "ROC-AUC"]
    ]

    st.bar_chart(
        chart_data
    )

    st.divider()

    st.subheader(
        "🎯 Threshold Optimization"
    )

    threshold_results = pd.DataFrame(
        {
            "Threshold": [
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50
            ],
            "Accuracy": [
                0.911149,
                0.918743,
                0.922211,
                0.922814,
                0.921580,
                0.918846,
                0.913842,
                0.907619,
                0.898682
            ],
            "F1": [
                0.947748,
                0.951636,
                0.953222,
                0.953155,
                0.952012,
                0.949908,
                0.946348,
                0.941948,
                0.935691
            ]
        }
    )

    st.dataframe(
        threshold_results,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        The selected threshold is **0.20** because it provides
        a strong balance between Precision, Recall and F1 Score.
        """
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif page == "🔍 Error Analysis":

    st.title("🔍 Error Analysis")

    st.markdown(
        """
        The model's incorrect predictions were analyzed
        to understand its limitations.
        """
    )

    # --------------------------------------------------------
    # ERROR METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Correct",
            "90.59%"
        )

    with col2:

        st.metric(
            "False Negative",
            "7.69%"
        )

    with col3:

        st.metric(
            "False Positive",
            "1.72%"
        )

    st.divider()

    st.subheader(
        "Prediction Error Distribution"
    )

    error_data = pd.DataFrame(
        {
            "Percentage": [
                90.59,
                7.69,
                1.72
            ]
        },
        index=[
            "Correct",
            "False Negative",
            "False Positive"
        ]
    )

    st.bar_chart(
        error_data
    )

    st.divider()

    st.subheader(
        "⚠️ Common Sources of Errors"
    )

    st.markdown(
        """
        The model can struggle with:

        - Sarcasm and irony
        - Mixed sentiment
        - Contradictory statements
        - Numerical ratings such as `1/10` and `10/10`
        - Game-specific terminology
        - Technical complaints
        - Updates and patches
        - Reviews where text sentiment and Steam recommendation differ
        """
    )

    st.divider()

    st.subheader(
        "📈 Game-Level Prediction Error"
    )

    error_games = game_analysis.copy()

    error_games["absolute_error"] = (
        error_games["score_difference"].abs()
    )

    error_games = error_games.sort_values(
        "absolute_error",
        ascending=False
    ).head(15)

    st.dataframe(
        error_games[
            [
                "name",
                "review_count",
                "actual_positive_rate",
                "ai_sentiment_score",
                "score_difference",
                "absolute_error"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        """
        These games show the largest difference between
        the actual positive review rate and the average
        AI sentiment score.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Steam Game Intelligence | "
    "TF-IDF + Logistic Regression"
)