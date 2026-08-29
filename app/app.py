import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# SAYFA AYARLARI
# ============================================================

st.set_page_config(
    page_title="Steam Oyun Analizi",
    page_icon="🎮",
    layout="wide"
)


# ============================================================
# DOSYA YOLLARI
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
# MODELİ YÜKLE
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return model, tfidf


# ============================================================
# OYUN VERİLERİNİ YÜKLE
# ============================================================

@st.cache_data
def load_game_data():

    data = pd.read_csv(GAME_ANALYSIS_PATH)

    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    return data


model, tfidf = load_model()
game_analysis = load_game_data()


# ============================================================
# YAN MENÜ
# ============================================================

st.sidebar.title("🎮 Steam Oyun Analizi")

st.sidebar.markdown(
    """
    ### Menü

    Steam oyun incelemelerini analiz etmek,
    oyunların genel memnuniyet düzeylerini
    karşılaştırmak ve yapay zekâ modelinin
    performansını incelemek için menüden
    bir bölüm seçebilirsin.
    """
)

page = st.sidebar.radio(
    "Bölüm seç:",
    [
        "🏠 Ana Sayfa",
        "💬 Yorum Analizi",
        "🎮 Oyun Analizi",
        "📊 Model Performansı",
        "🔍 Hata Analizi"
    ]
)


# ============================================================
# ANA SAYFA
# ============================================================

if page == "🏠 Ana Sayfa":

    st.title("🎮 Steam Oyun Analizi")

    st.markdown(
        """
        ## Yapay Zekâ Destekli Steam Yorum Analizi

        Bu uygulama, Steam oyun yorumlarını makine öğrenmesi
        kullanarak analiz eder.

        Model, yorumların **olumlu veya olumsuz** olduğunu
        tahmin eder ve her yorum için bir **olumlu duygu skoru**
        oluşturur.

        Kullanılan temel yöntemler:

        - **TF-IDF:** Yorumları sayısal özelliklere dönüştürür.
        - **Lojistik Regresyon:** Yorumun olumlu veya olumsuz
          olduğunu tahmin eder.
        - **Oyun bazlı analiz:** Yorum sonuçları oyun seviyesinde
          karşılaştırılır.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL SONUÇLARI
    # --------------------------------------------------------

    st.subheader("📈 Model Sonuçları")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Lojistik Regresyon"
        )

    with col2:
        st.metric(
            "Doğruluk",
            "92.22%"
        )

    with col3:
        st.metric(
            "ROC-AUC",
            "0.9593"
        )

    with col4:
        st.metric(
            "Seçilen Eşik",
            "0.20"
        )

    st.divider()

    # --------------------------------------------------------
    # VERİ SETİ İSTATİSTİKLERİ
    # --------------------------------------------------------

    st.subheader("🎮 Veri Seti Özeti")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Analiz Edilen Oyun",
            f"{len(game_analysis):,}"
        )

    with col2:
        st.metric(
            "Oyun Başına Minimum Yorum",
            "100"
        )

    with col3:
        st.metric(
            "Oyun Bazlı Korelasyon",
            "0.9824"
        )

    st.divider()

    # --------------------------------------------------------
    # PROJE AKIŞI
    # --------------------------------------------------------

    st.subheader("🔄 Proje Akışı")

    st.markdown(
        """
        **Steam Yorumları**

        ↓

        **Veri Temizleme**

        ↓

        **TF-IDF ile Metin Dönüştürme**

        ↓

        **Lojistik Regresyon**

        ↓

        **Duygu Skoru**

        ↓

        **Oyun Bazlı Analiz**
        """
    )

    st.divider()

    st.subheader("💡 Temel Bulgular")

    st.markdown(
        """
        - Lojistik Regresyon, test edilen modeller arasında
          en iyi sonucu verdi.
        - Optimize edilen sınıflandırma eşiği **0.20** olarak
          belirlendi.
        - Modelin doğruluk oranı **%92.22** seviyesine ulaştı.
        - ROC-AUC değeri **0.9593** olarak ölçüldü.
        - Oyun bazında yapay zekâ duygu skoru ile gerçek
          olumlu yorum oranı arasında **0.9824** korelasyon
          bulundu.
        """
    )


# ============================================================
# YORUM ANALİZİ
# ============================================================

elif page == "💬 Yorum Analizi":

    st.title("💬 Steam Yorum Analizi")

    st.markdown(
        """
        Aşağıdaki kutuya bir Steam oyun yorumu yaz.

        Yapay zekâ modeli yorumun olumlu veya olumsuz olduğunu
        tahmin edecek ve olumlu duygu skorunu hesaplayacaktır.
        """
    )

    review = st.text_area(
        "Steam yorumunu buraya yaz:",
        height=180,
        placeholder=(
            "Örnek: Bu oyun gerçekten harika! "
            "Grafikleri çok güzel ve oynaması çok eğlenceli."
        )
    )

    analyze = st.button(
        "🤖 Yorumu Analiz Et",
        use_container_width=True
    )

    if analyze:

        if not review.strip():

            st.warning(
                "⚠️ Lütfen önce bir yorum yaz."
            )

        else:

            # ------------------------------------------------
            # TF-IDF DÖNÜŞÜMÜ
            # ------------------------------------------------

            review_tfidf = tfidf.transform(
                [review]
            )

            # ------------------------------------------------
            # OLASILIK
            # ------------------------------------------------

            probability = model.predict_proba(
                review_tfidf
            )[0][1]

            # ------------------------------------------------
            # SEÇİLEN EŞİK
            # ------------------------------------------------

            threshold = 0.20

            prediction = (
                probability >= threshold
            )

            st.divider()

            # ------------------------------------------------
            # TAHMİN SONUCU
            # ------------------------------------------------

            if prediction:

                st.success(
                    "🟢 Olumlu Yorum"
                )

            else:

                st.error(
                    "🔴 Olumsuz Yorum"
                )

            # ------------------------------------------------
            # SONUÇLAR
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Olumlu Duygu Skoru",
                    f"{probability:.2%}"
                )

            with col2:

                st.metric(
                    "Kullanılan Eşik",
                    f"{threshold:.2f}"
                )

            with col3:

                if probability >= 0.80:

                    confidence = "Yüksek"

                elif probability >= 0.50:

                    confidence = "Orta"

                else:

                    confidence = "Düşük"

                st.metric(
                    "Model Güveni",
                    confidence
                )

            st.divider()

            # ------------------------------------------------
            # DUYGU SKORU
            # ------------------------------------------------

            st.subheader(
                "📊 Olumlu Duygu Skoru"
            )

            st.progress(
                float(probability)
            )

            st.caption(
                "Bu değer, modelin yorumun olumlu olma "
                "olasılığına ilişkin tahminini gösterir."
            )

            st.divider()

            # ------------------------------------------------
            # YORUM
            # ------------------------------------------------

            st.subheader(
                "📝 Analiz Edilen Yorum"
            )

            st.info(review)

            st.divider()

            # ------------------------------------------------
            # KISA AÇIKLAMA
            # ------------------------------------------------

            if probability >= 0.80:

                st.success(
                    "Model bu yorumu güçlü şekilde olumlu "
                    "olarak değerlendiriyor."
                )

            elif probability >= 0.50:

                st.info(
                    "Model yorumda genel olarak olumlu "
                    "bir duygu olduğunu düşünüyor."
                )

            elif probability >= 0.20:

                st.warning(
                    "Model yorumu düşük seviyede olumlu "
                    "olarak değerlendiriyor."
                )

            else:

                st.error(
                    "Model yorumda olumlu duygu yerine "
                    "olumsuz duygu bulunduğunu düşünüyor."
                )


# ============================================================
# OYUN ANALİZİ
# ============================================================

elif page == "🎮 Oyun Analizi":

    st.title("🎮 Oyun Bazlı Analiz")

    st.markdown(
        """
        Bir oyun seçerek Steam kullanıcılarının gerçek olumlu
        yorum oranını yapay zekânın oluşturduğu duygu skoru ile
        karşılaştırabilirsin.
        """
    )

    selected_game = st.selectbox(
        "🎯 Bir oyun seç:",
        sorted(
            game_analysis["name"]
            .dropna()
            .unique()
        )
    )

    game = game_analysis[
        game_analysis["name"] == selected_game
    ].iloc[0]

    st.divider()

    # --------------------------------------------------------
    # OYUN BİLGİLERİ
    # --------------------------------------------------------

    st.subheader(
        f"🎮 {selected_game}"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Yorum Sayısı",
            f"{int(game['review_count']):,}"
        )

    with col2:

        st.metric(
            "Gerçek Olumlu Yorum Oranı",
            f"{game['actual_positive_rate']:.2%}"
        )

    with col3:

        st.metric(
            "Yapay Zekâ Duygu Skoru",
            f"{game['ai_sentiment_score']:.2%}"
        )

    with col4:

        difference = game["score_difference"]

        st.metric(
            "Fark",
            f"{difference:+.2%}"
        )

    st.divider()

    # --------------------------------------------------------
    # KARŞILAŞTIRMA
    # --------------------------------------------------------

    st.subheader(
        "📊 Gerçek Oran ve Yapay Zekâ Skoru Karşılaştırması"
    )

    chart_data = pd.DataFrame(
        {
            "Skor": [
                game["actual_positive_rate"],
                game["ai_sentiment_score"]
            ]
        },
        index=[
            "Gerçek Olumlu Yorum Oranı",
            "Yapay Zekâ Duygu Skoru"
        ]
    )

    st.bar_chart(
        chart_data
    )

    st.divider()

    # --------------------------------------------------------
    # YORUM
    # --------------------------------------------------------

    st.subheader(
        "💡 Sonuç"
    )

    if abs(difference) < 0.05:

        st.success(
            "✅ Yapay zekâ skoru ile gerçek olumlu yorum "
            "oranı birbirine oldukça yakın."
        )

    elif difference > 0:

        st.info(
            "ℹ️ Yapay zekâ, kullanıcıların gerçek olumlu "
            "yorum oranından daha yüksek bir olumlu duygu "
            "seviyesi tahmin ediyor."
        )

    else:

        st.warning(
            "⚠️ Yapay zekâ, kullanıcıların gerçek olumlu "
            "yorum oranından daha düşük bir olumlu duygu "
            "seviyesi tahmin ediyor."
        )


# ============================================================
# MODEL PERFORMANSI
# ============================================================

elif page == "📊 Model Performansı":

    st.title("📊 Model Performansı")

    st.markdown(
        """
        Projede iki farklı makine öğrenmesi modeli test edildi:

        - **Lojistik Regresyon**
        - **Doğrusal SVM**

        Her iki model de TF-IDF ile oluşturulan metin özellikleri
        kullanılarak eğitildi.
        """
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Lojistik Regresyon",
                "Doğrusal SVM"
            ],
            "Doğruluk": [
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
    # SEÇİLEN MODEL
    # --------------------------------------------------------

    st.subheader(
        "🏆 Seçilen Model"
    )

    st.success(
        """
        Lojistik Regresyon final model olarak seçildi.

        Bunun temel nedeni test edilen modeller arasında
        daha yüksek doğruluk ve ROC-AUC değerlerine ulaşmasıdır.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Doğruluk",
            "92.22%"
        )

    with col2:

        st.metric(
            "Kesinlik",
            "93.74%"
        )

    with col3:

        st.metric(
            "Duyarlılık",
            "96.96%"
        )

    with col4:

        st.metric(
            "F1 Skoru",
            "95.32%"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL KARŞILAŞTIRMA
    # --------------------------------------------------------

    st.subheader(
        "📈 Model Karşılaştırması"
    )

    chart_data = results.set_index(
        "Model"
    )[
        ["Doğruluk", "ROC-AUC"]
    ]

    st.bar_chart(
        chart_data
    )

    st.divider()

    # --------------------------------------------------------
    # EŞİK OPTİMİZASYONU
    # --------------------------------------------------------

    st.subheader(
        "🎯 Eşik Değeri Optimizasyonu"
    )

    threshold_results = pd.DataFrame(
        {
            "Eşik": [
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
            "Doğruluk": [
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
        🎯 Final eşik değeri **0.20** olarak seçildi.

        Bu değer, modelin olumlu yorumları yakalama oranını
        yüksek tutarken genel performans açısından güçlü
        bir denge sağlamaktadır.
        """
    )


# ============================================================
# HATA ANALİZİ
# ============================================================

elif page == "🔍 Hata Analizi":

    st.title("🔍 Model Hata Analizi")

    st.markdown(
        """
        Modelin yanlış yaptığı tahminleri inceleyerek hangi
        durumlarda zorlandığını analiz ediyoruz.
        """
    )

    # --------------------------------------------------------
    # HATA ORANLARI
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "✅ Doğru Tahmin",
            "90.59%"
        )

    with col2:

        st.metric(
            "❌ Yanlış Negatif",
            "7.69%"
        )

    with col3:

        st.metric(
            "❌ Yanlış Pozitif",
            "1.72%"
        )

    st.divider()

    # --------------------------------------------------------
    # HATA DAĞILIMI
    # --------------------------------------------------------

    st.subheader(
        "📊 Tahmin Hatalarının Dağılımı"
    )

    error_data = pd.DataFrame(
        {
            "Yüzde": [
                90.59,
                7.69,
                1.72
            ]
        },
        index=[
            "Doğru Tahmin",
            "Yanlış Negatif",
            "Yanlış Pozitif"
        ]
    )

    st.bar_chart(
        error_data
    )

    st.divider()

    # --------------------------------------------------------
    # MODELİN ZORLANDIĞI DURUMLAR
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Modelin Zorlandığı Durumlar"
    )

    st.markdown(
        """
        Model özellikle aşağıdaki durumlarda hata yapabilir:

        - 😏 İroni ve alay
        - 🔀 Karışık duygu içeren yorumlar
        - ⚖️ Birbirine zıt ifadeler
        - 🔢 `1/10` ve `10/10` gibi sayısal puanlamalar
        - 🎮 Oyuna özel terimler
        - 🛠️ Teknik problemler
        - 🔄 Güncellemeler ve yamalar
        - 💬 Yorumun yazılı duygu ifadesi ile Steam
          öneri etiketinin birbirinden farklı olması
        """
    )

    st.divider()

    # --------------------------------------------------------
    # OYUN BAZLI HATA
    # --------------------------------------------------------

    st.subheader(
        "📈 Oyun Bazında En Büyük Farklar"
    )

    error_games = game_analysis.copy()

    error_games["mutlak_hata"] = (
        error_games["score_difference"].abs()
    )

    error_games = error_games.sort_values(
        "mutlak_hata",
        ascending=False
    ).head(15)

    display_data = error_games[
        [
            "name",
            "review_count",
            "actual_positive_rate",
            "ai_sentiment_score",
            "score_difference",
            "mutlak_hata"
        ]
    ].copy()

    display_data.columns = [
        "Oyun",
        "Yorum Sayısı",
        "Gerçek Olumlu Oran",
        "Yapay Zekâ Skoru",
        "Fark",
        "Mutlak Hata"
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        """
        Bu tablo, gerçek olumlu yorum oranı ile yapay zekânın
        ortalama duygu skoru arasındaki farkın en yüksek olduğu
        oyunları göstermektedir.
        """
    )


# ============================================================
# ALT BİLGİ
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "🎮 Steam Oyun Analizi"
)

st.sidebar.caption(
    "TF-IDF + Lojistik Regresyon"
)