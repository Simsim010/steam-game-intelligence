# 🎮 Steam Game Reviews - Sentiment Analysis

## 📌 Project Overview

This project analyzes Steam game reviews using Natural Language Processing (NLP) and Machine Learning techniques.

The main goal is to build a sentiment classification model that predicts whether a Steam review is positive or negative, and then use the model to generate sentiment scores at both the review and game level.

The project also investigates the relationship between:

- The actual Steam recommendation rate (`voted_up`)
- The AI-generated sentiment score
- Prediction errors
- Game-level sentiment differences

Two machine learning models were evaluated:

- Logistic Regression
- Linear Support Vector Machine (SVM)

The final model was selected based on Accuracy and ROC-AUC performance.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Clean and prepare a large-scale Steam review dataset.
2. Analyze the distribution of positive and negative reviews.
3. Convert review text into numerical features using TF-IDF.
4. Train machine learning models for sentiment classification.
5. Compare Logistic Regression and Linear SVM.
6. Evaluate model performance using Accuracy, Precision, Recall, F1 Score and ROC-AUC.
7. Optimize the classification threshold.
8. Generate AI sentiment scores for Steam reviews.
9. Aggregate sentiment scores at the game level.
10. Compare AI sentiment with the actual positive review rate.
11. Analyze false positives and false negatives.
12. Identify challenging reviews for the model.
13. Investigate which words are strongly associated with positive and negative predictions.

---

## 📂 Dataset

The project uses a Steam game review dataset containing approximately 730,000 reviews.

The dataset includes information such as:

- Game name
- Review text
- Review recommendation (`voted_up`)
- Word count
- Release date
- Other review-related information

The target variable is:

```text
voted_up
where:
- True → Positive recommendation
- False → Negative recommendation
🧹 Data Cleaning
The first step was to inspect and clean the dataset.
Removing unnecessary columns
The release_date column was removed because it was not required for the sentiment classification task.
df = df.drop(columns=["release_date"])
Checking class distribution
The distribution of positive and negative reviews was examined using:
df["voted_up"].value_counts()
and:
df["voted_up"].value_counts(normalize=True)
A bar chart was also created to visualize the distribution.
🔍 Review Analysis
The review column was examined using:
df["review"].describe()
The word_count variable was also analyzed to understand review length.
The shortest and longest reviews were inspected:
df.loc[df["word_count"].idxmin(), "review"]
df.loc[df["word_count"].idxmax(), "review"]
🧹 Duplicate Removal
Duplicate rows and duplicate reviews were checked.
df.duplicated().sum()
df["review"].duplicated().sum()
Duplicate rows were removed:
df = df.drop_duplicates()
🔢 Removing Numeric-Only Reviews
Some reviews contained only numbers such as:
10 10
1 10
10 10 10
These reviews can create misleading text features.
A regular expression was used to identify numeric-only reviews:
numeric_only = df["review"].str.fullmatch(
    r"\s*\d+(?:\s+\d+)*\s*"
)
These reviews were removed:
df = df[~numeric_only].copy()
🔤 Removing Reviews Without Meaningful Text
Reviews containing no alphabetic characters were identified.
has_letters = df["review"].str.contains(
    r"[^\W\d_]",
    regex=True
)
Reviews without letters were removed:
df = df[has_letters].copy()
This helped reduce noise caused by reviews consisting mainly of symbols, numbers, or formatting characters.
✂️ Train-Test Split
The cleaned review dataset was divided into training and testing sets.
The test set contained 20% of the data.
from sklearn.model_selection import train_test_split

X = df["review"]
y = df["voted_up"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
Stratification was used to preserve the positive/negative class distribution in both datasets.
🔢 TF-IDF Feature Extraction
Since machine learning models cannot directly process raw text, the reviews were converted into numerical features using TF-IDF.
The following configuration was used:
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=50000,
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True
)
The model uses:
- Unigrams → individual words
- Bigrams → two-word combinations
- Maximum 50,000 features
- Minimum document frequency of 5
- Maximum document frequency of 95%
The training data was used to fit the vectorizer:
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
🤖 Machine Learning Models
Two classification algorithms were evaluated.
1. Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)
The class_weight="balanced" parameter was used to account for class imbalance.
2. Linear Support Vector Machine
A Linear SVM was also trained:
from sklearn.svm import LinearSVC

svm_model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train_tfidf, y_train)
📊 Model Evaluation
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
Initial Model Comparison
Model	Accuracy	ROC-AUC
Logistic Regression	0.9000	0.9593
Linear SVM	0.8987	0.9557


Logistic Regression achieved slightly better performance than Linear SVM.
Therefore, Logistic Regression was selected as the final model.
📈 ROC-AUC
For Logistic Regression, prediction probabilities were obtained using:
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
The ROC-AUC score was:
ROC-AUC = 0.9593
This indicates that the model has a strong ability to distinguish positive and negative Steam reviews.
🎯 Threshold Optimization
The default classification threshold of 0.50 was not used automatically.
Instead, multiple thresholds were tested to determine whether a different threshold could improve classification performance.
The tested thresholds ranged from:
0.10 → 0.90
The best F1 Score was obtained at:
Threshold = 0.20
The results at the selected threshold were:
Metric	Score
Threshold	0.2000
Accuracy	0.9222
Precision	0.9374
Recall	0.9696
F1 Score	0.9532
ROC-AUC	0.9593


The optimized threshold significantly improved the classification metrics compared with the default 0.50 threshold.
The final prediction rule was therefore based on:
final_threshold = 0.20

y_pred_final = (
    y_prob >= final_threshold
)
🧠 Model Interpretation
The Logistic Regression coefficients were analyzed to understand which words contributed most strongly to positive and negative predictions.
🟢 Strong Positive Features
Some of the strongest positive terms were:
Word / Phrase	Coefficient
10 10	13.5507
best	12.0642
great	10.9421
masterpiece	10.0159
fun	9.9003
amazing	9.2621
good	9.2009
perfect	9.1950
not bad	8.9236
peak	8.8002
love	8.3969
excellent	7.8953
better than	7.4930
fantastic	7.4495
addictive	7.4256
awesome	7.3769
very fun	7.3658
solid	6.9008
definitely	6.8873
incredible	6.7840


These features generally indicate positive sentiment.
🔴 Strong Negative Features
Some of the strongest negative terms were:
Word / Phrase	Coefficient
not worth	-12.2085
unplayable	-11.6916
boring	-10.9586
can recommend	-9.4480
not fun	-8.6985
worst	-8.6577
not	-8.5229
garbage	-8.2323
spyware	-8.1585
eula	-8.0868
refund	-8.0809
disappointing	-7.8990
crashes	-7.8576
than this	-7.7567
not recommend	-7.7004
mediocre	-7.6484
awful	-7.6048
poorly	-7.4553
overrated	-7.4246
at best	-7.3699


These features generally contributed to negative predictions.
🎮 Game-Level Sentiment Analysis
After training the final model, the model was applied to reviews belonging to games with at least 100 reviews.
The AI produced a probability score for each review:
df_games["ai_sentiment_score"]
This score represents the model's estimated probability that the review is positive.
The review-level predictions were then aggregated by game.
For each game, three important metrics were calculated:
game_analysis = df_games.groupby("name").agg(
    review_count=("voted_up", "count"),
    actual_positive_rate=("voted_up", "mean"),
    ai_sentiment_score=("ai_sentiment_score", "mean")
)
Where:
- review_count → Number of reviews
- actual_positive_rate → Actual percentage of positive Steam recommendations
- ai_sentiment_score → Average AI sentiment score
🟢 Most Positive Games
Based on the actual positive review rate, some of the most positively reviewed games were:
Game	Reviews	Actual Positive Rate	AI Sentiment
Stardew Valley	996	0.9809	0.8883
The Henry Stickmin Collection	998	0.9800	0.8796
Portal 2	999	0.9790	0.8685
Portal	998	0.9780	0.8705
Plants vs. Zombies GOTY Edition	996	0.9779	0.8517
Factorio	1000	0.9770	0.8280
People Playground	1000	0.9740	0.8670
ULTRAKILL	998	0.9739	0.8267
Slime Rancher	999	0.9720	0.8786
Aseprite	997	0.9719	0.8639


These games had very high recommendation rates.
🔴 Most Negative Games
Some of the games with the lowest actual positive review rates were:
Game	Reviews	Actual Positive Rate	AI Sentiment
Borderlands: The Pre-Sequel	999	0.2302	0.2337
BattleBit Remastered	1000	0.2550	0.2797
Squad	1000	0.3000	0.2952
Wolfenstein: Youngblood	999	0.3473	0.3412
Borderlands Game of the Year	998	0.3737	0.3426
Creativerse	1000	0.3760	0.3523
Battlefield 2042	1000	0.3770	0.3234
EA SPORTS FC 25	999	0.3844	0.3494
Call of Duty: Modern Warfare II	1000	0.3880	0.3900
ATLAS	998	0.4269	0.3800


The AI sentiment scores generally followed the actual recommendation rates.
📐 Game-Level Correlation
The relationship between actual positive review rate and AI sentiment score was analyzed using correlation.
The result was:
Correlation = 0.9824
This is a very strong positive correlation.
It indicates that games with higher actual recommendation rates generally also received higher AI sentiment scores.
📏 Mean Absolute Error
The Mean Absolute Error between the actual positive rate and the AI sentiment score was:
MAE = 0.1002
This means that, on average, the AI sentiment score differed from the actual positive review rate by approximately 0.10.
The result demonstrates that the AI score tracks the overall game-level recommendation rate reasonably well, while also showing that sentiment and recommendation are not exactly the same concept.
⚠️ Game-Level Prediction Differences
The difference was calculated as:
score_difference = (
    ai_sentiment_score
    - actual_positive_rate
)
A positive value means that the AI estimated sentiment higher than the actual recommendation rate.
A negative value means that the AI estimated sentiment lower than the actual recommendation rate.
The games with the largest differences were examined to identify cases where the review text sentiment differed substantially from the Steam recommendation behavior.
🚨 Error Analysis
The final predictions were divided into:
- Correct
- False Negative
- False Positive
The results were:
Prediction Type	Count	Percentage
Correct	660,726	90.59%
False Negative	56,123	7.69%
False Positive	12,536	1.72%


False Negative errors were more common than False Positive errors.
❌ False Negative Analysis
A False Negative occurs when:
Actual = Positive
Prediction = Negative
The model therefore interpreted the review as negative even though the user recommended the game.
Examples showed that some positive recommendations contained strongly negative language.
For example, a user may complain extensively about a game but still recommend it.
This creates a difficult classification problem because the sentiment expressed in the text and the final recommendation label may not always agree.
❌ False Positive Analysis
A False Positive occurs when:
Actual = Negative
Prediction = Positive
Some negative reviews contained strongly positive words.
Examples included reviews such as:
"best game ever"
or:
"great game"
but the review was labeled as not recommended.
This can happen because of:
- Sarcasm
- Irony
- Contradictory statements
- Numerical ratings
- Context-dependent language
- Complaints mixed with positive statements
🧩 Difficult Reviews
The most difficult predictions were identified using the absolute prediction error.
For each review:
error = abs(
    actual_label - ai_sentiment_score
)
The analysis revealed several challenging patterns.
1. Sarcasm and Irony
Example:
"best game ever"
may be used sarcastically.
Traditional TF-IDF models have difficulty recognizing this.
2. Mixed Sentiment
A review can contain both positive and negative statements:
"The game is really fun, but the servers are terrible."
The model must determine the overall recommendation from conflicting information.
3. Numerical Ratings
Expressions such as:
1/10
10/10
can create problems because the model may learn numerical patterns without fully understanding their context.
4. Contradictory Statements
For example:
"This game sucks. Can't stop playing, love it."
contains both negative and positive language.
5. Technical Complaints
Some users recommend a game despite mentioning:
- crashes
- performance issues
- login problems
- bugs
- updates
- DLC problems
This creates a difference between textual sentiment and recommendation behavior.
📊 Game-Level Visualization
Several visualizations were created during the analysis.
Actual Positive Rate
A bar chart was created to display the most positively reviewed games.
Actual vs AI Sentiment
A comparison chart was created using:
- Actual Positive Rate
- AI Sentiment Score
This allowed the difference between human recommendation behavior and AI sentiment estimation to be visually examined.
Correlation Plot
A scatter plot was used to compare:
Actual Positive Rate
against:
AI Sentiment Score
A perfect agreement line was added:
y = x
The points were generally close to this line, supporting the strong correlation of 0.9824.
🏆 Final Model
The final selected model is:
Logistic Regression
with:
TF-IDF
using:
- Unigrams
- Bigrams
- Maximum 50,000 features
- Sublinear TF-IDF weighting
The final model achieved:
Metric	Result
Accuracy	92.22%
Precision	93.74%
Recall	96.96%
F1 Score	95.32%
ROC-AUC	95.93%
Optimal Threshold	0.20


💡 Key Findings
The project produced several important findings.
Finding 1
TF-IDF combined with Logistic Regression can achieve strong performance on large-scale Steam review sentiment classification.
Finding 2
Logistic Regression slightly outperformed Linear SVM.
Finding 3
Threshold optimization significantly improved classification performance.
The optimal threshold was:
0.20
with an F1 Score of:
0.9532
Finding 4
The AI sentiment score strongly follows game-level recommendation behavior.
The correlation was:
0.9824
Finding 5
The AI sentiment score and Steam recommendation are related but not identical.
A user may write a negative-sounding review and still recommend the game.
Finding 6
False Negatives were more common than False Positives.
Finding 7
Sarcasm, irony, mixed sentiment, numerical ratings and contextual language were among the main sources of prediction errors.
🚀 Future Improvements
The project can be extended in several ways.
1. Transformer-Based NLP Models
Transformer models such as BERT could be tested.
These models can better capture context and relationships between words.
Possible models include:
- BERT
- RoBERTa
- DistilBERT
- DeBERTa
2. Advanced Text Preprocessing
Future preprocessing could include:
- Emoji processing
- Slang normalization
- Repeated character handling
- Internet abbreviations
- Special Steam terminology
- Better handling of numerical ratings
3. Multilingual Sentiment Analysis
Steam reviews can contain multiple languages.
A language detection system could be added and multilingual models could be evaluated.
Possible approaches include:
- Multilingual BERT
- XLM-RoBERTa
- Language-specific models
4. Probability Calibration
The Logistic Regression probabilities could be calibrated to make the sentiment scores more reliable.
This would be especially useful because the project uses the probabilities as game-level sentiment scores.
5. Game-Level Prediction
Additional game-level variables could be incorporated.
Possible features include:
- Review count
- Price
- Release year
- Playtime
- Game genre
- DLC count
- Developer
- Publisher
This could allow the project to move from simple sentiment analysis toward a broader game evaluation system.
6. Explainable AI
Future versions could provide explanations for individual predictions.
For example:
Prediction: Positive
Confidence: 94%

Important positive words:
- great
- fun
- excellent
- amazing
This could make the model easier to interpret.
🛠️ Technologies Used
The project was developed using Python.
Programming Language
- Python
Data Analysis
- Pandas
- NumPy
Visualization
- Matplotlib
Machine Learning
- Scikit-learn
NLP
- TF-IDF
- N-grams
- Logistic Regression
- Linear SVM
Environment
- Jupyter Notebook
📁 Project Structure
steam-game-reviews-sentiment/
│
├── data/
│   └── steam_game_reviews_730945.csv
│
├── notebooks/
│   └── steam_sentiment_analysis.ipynb
│
├── README.md
│
└── requirements.txt
📦 Installation
Clone the repository:
git clone https://github.com/USERNAME/REPOSITORY.git
Navigate to the project directory:
cd steam-game-reviews-sentiment
Install the required libraries:
pip install pandas numpy matplotlib scikit-learn jupyter
Start Jupyter Notebook:
jupyter notebook
Then open the project notebook.
▶️ How to Run
1. Place the Steam dataset inside the data/ directory.
2. Open the Jupyter Notebook.
3. Run the cells from top to bottom.
4. The notebook performs:
   - Data cleaning
   - Exploratory analysis
   - TF-IDF transformation
   - Model training
   - Model evaluation
   - Threshold optimization
   - Game-level sentiment analysis
   - Error analysis
   - Visualization
📌 Conclusion
This project demonstrates how Natural Language Processing and traditional machine learning can be used to analyze a large-scale collection of Steam game reviews.
After data cleaning and TF-IDF feature extraction, Logistic Regression and Linear SVM were evaluated.
Logistic Regression was selected as the final model because it achieved slightly better performance.
After threshold optimization, the model achieved:
Accuracy: 92.22%
Precision: 93.74%
Recall: 96.96%
F1 Score: 95.32%
ROC-AUC: 95.93%
The model was also used to generate sentiment scores for individual reviews and aggregate these scores at the game level.
The game-level sentiment score showed a very strong correlation with the actual positive review rate:
Correlation = 0.9824
with a Mean Absolute Error of:
MAE = 0.1002
The error analysis showed that the main challenges were sarcasm, irony, mixed sentiment, numerical ratings, contradictory statements and cases where the written review sentiment differed from the user's final recommendation.
Overall, the project demonstrates that TF-IDF combined with Logistic Regression can provide an effective and interpretable baseline for large-scale Steam review sentiment analysis.
👩‍💻 Author
Simge Altun
Computer Engineering Minor & Electrical-Electronics Engineering Student
⭐ If you found this project interesting, feel free to explore the notebook and the analysis.
