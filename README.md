# Internship
HR prodigy Infotech
# ============================================================
# TASK 4: SENTIMENT ANALYSIS
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# 2. LOAD DATASET
columns = ["Tweet_ID", "Entity", "Sentiment", "Text"]

try:
    train_df = pd.read_csv("twitter_training.csv", header=None, names=columns)
    val_df = pd.read_csv("twitter_validation.csv", header=None, names=columns)

    df = pd.concat([train_df, val_df], ignore_index=True)

except FileNotFoundError:
    print("Error: Please place twitter_training.csv and "
          "twitter_validation.csv in the notebook folder.")
    raise

print("Dataset loaded successfully!")
print("Total records:", df.shape[0])
print("\nFirst five records:")
print(df.head())

# 3. DATA EXPLORATION
print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# 4. DATA CLEANING
df = df.dropna(subset=["Text", "Sentiment", "Entity"])
df = df.drop_duplicates()

# 5. TEXT CLEANING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Clean_Text"] = df["Text"].apply(clean_text)

print("\nCleaned Text:")
print(df[["Text", "Clean_Text"]].head())

# 6. SENTIMENT DISTRIBUTION
sentiment_order = ["Positive", "Neutral", "Negative", "Irrelevant"]

palette = {
    "Positive": "#55A868",
    "Neutral": "#4C72B0",
    "Negative": "#C44E52",
    "Irrelevant": "#8172B2"
}

sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(9, 6))
sns.countplot(
    data=df,
    x="Sentiment",
    order=sentiment_order,
    hue="Sentiment",
    palette=palette,
    legend=False
)

plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig("overall_sentiment_distribution.png", dpi=150)
plt.show()

# 7. SENTIMENT PERCENTAGES
print("\nSentiment Percentages:")
print((sentiment_counts / len(df) * 100).round(2))

# 8. TOP 10 MOST DISCUSSED ENTITIES
top_entities = df["Entity"].value_counts().head(10).index

print("\nTop 10 Entities:")
print(df["Entity"].value_counts().head(10))

entity_sentiment = (
    df[df["Entity"].isin(top_entities)]
    .groupby(["Entity", "Sentiment"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=sentiment_order, fill_value=0)
)

entity_sentiment.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 7),
    color=[palette[s] for s in sentiment_order]
)

plt.title("Sentiment Distribution for Top 10 Entities")
plt.xlabel("Entity")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("entity_sentiment_distribution.png", dpi=150)
plt.show()

# 9. POSITIVE AND NEGATIVE SENTIMENT PERCENTAGES
entity_pct = entity_sentiment.div(
    entity_sentiment.sum(axis=1).replace(0, 1),
    axis=0
) * 100

entity_pct = entity_pct.sort_values("Positive")

entity_pct[["Positive", "Negative"]].plot(
    kind="barh",
    figsize=(12, 7),
    color=[palette["Positive"], palette["Negative"]]
)

plt.title("Positive vs Negative Sentiment by Entity")
plt.xlabel("Percentage of Tweets")
plt.ylabel("Entity")
plt.tight_layout()
plt.savefig("positive_negative_sentiment.png", dpi=150)
plt.show()

# 10. WORD CLOUD FOR POSITIVE TWEETS
positive_text = " ".join(
    df.loc[df["Sentiment"] == "Positive", "Clean_Text"]
)

if positive_text.strip():
    positive_wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=100
    ).generate(positive_text)

    plt.figure(figsize=(12, 7))
    plt.imshow(positive_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Positive Tweets")
    plt.tight_layout()
    plt.savefig("positive_wordcloud.png", dpi=150)
    plt.show()
else:
    print("No positive tweet text available for the word cloud.")

# 11. WORD CLOUD FOR NEGATIVE TWEETS
negative_text = " ".join(
    df.loc[df["Sentiment"] == "Negative", "Clean_Text"]
)

if negative_text.strip():
    negative_wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        max_words=100
    ).generate(negative_text)

    plt.figure(figsize=(12, 7))
    plt.imshow(negative_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud - Negative Tweets")
    plt.tight_layout()
    plt.savefig("negative_wordcloud.png", dpi=150)
    plt.show()
else:
    print("No negative tweet text available for the word cloud.")

# 12. TWEET LENGTH ANALYSIS
df["Text_Length"] = df["Clean_Text"].apply(
    lambda text: len(text.split())
)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df[df["Sentiment"].isin(["Positive", "Neutral", "Negative"])],
    x="Sentiment",
    y="Text_Length",
    order=["Positive", "Neutral", "Negative"],
    showfliers=False
)

plt.title("Tweet Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Words")
plt.tight_layout()
plt.savefig("tweet_length_analysis.png", dpi=150)
plt.show()

# 13. AVERAGE TWEET LENGTH
print("\nAverage Tweet Length by Sentiment:")
print(
    df.groupby("Sentiment")["Text_Length"]
    .mean()
    .round(2)
)

# 14. SENTIMENT ANALYSIS SUMMARY
print("\n" + "=" * 50)
print("SENTIMENT ANALYSIS SUMMARY")
print("=" * 50)

print("Total tweets analyzed:", len(df))
print("\nSentiment counts:")
print(df["Sentiment"].value_counts())

print("\nMost common sentiment:")
print(df["Sentiment"].value_counts().idxmax())

print("\nMost discussed entity:")
print(df["Entity"].value_counts().idxmax())

print("\nAverage number of words per tweet:",
      round(df["Text_Length"].mean(), 2))

print("\nSentiment analysis completed successfully!")
print("Charts have been saved as PNG files."

# ============================================================
# TASK 1: POPULATION DISTRIBUTION
# Prodigy InfoTech - Data Science Internship
# Objective: Visualize population distribution using a
# bar chart and histogram.
# Dataset: World Bank Total Population
# ============================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("=" * 60)
print("POPULATION DISTRIBUTION ANALYSIS")
print("=" * 60)


# STEP 2: LOAD THE DATASET

try:
    df = pd.read_csv(
        "API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv",
        skiprows=4
    )

    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

except FileNotFoundError:
    raise FileNotFoundError(
        "Population dataset not found. Please upload "
        "API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv "
        "to your notebook folder."
    )


# STEP 3: DATA CLEANING

# Remove unnecessary unnamed columns
df = df.drop(
    columns=[
        col for col in df.columns
        if "Unnamed" in col
    ]
)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Check missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum().head(10))

# Convert the 2024 population column to numeric
df["2024"] = pd.to_numeric(
    df["2024"],
    errors="coerce"
)

print(
    "\nMissing population values for 2024:",
    df["2024"].isnull().sum()
)


# STEP 4: SELECT ACTUAL COUNTRIES

# Load country metadata to exclude regional aggregates
try:
    meta = pd.read_csv(
        "Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv"
    )

    # Select country codes that have a valid geographic region
    country_codes = meta.loc[
        meta["Region"].notnull(),
        "Country Code"
    ].dropna()

    # Keep actual countries only
    df_countries = df[
        df["Country Code"].isin(country_codes)
    ].copy()

except FileNotFoundError:
    raise FileNotFoundError(
        "Country metadata file not found. Please upload "
        "Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_38144.csv "
        "to your notebook folder."
    )

# Remove rows without population values for 2024
df_countries = df_countries.dropna(
    subset=["2024"]
)

# Remove invalid or non-positive population values
df_countries = df_countries[
    df_countries["2024"] > 0
]

print("\nData cleaning completed!")
print(
    "Number of countries remaining:",
    len(df_countries)
)

print("\nCleaned dataset:")
print(
    df_countries[
        ["Country Name", "Country Code", "2024"]
    ].head()
)


# STEP 5: BAR CHART OF TOP 15 MOST POPULOUS COUNTRIES

# Select the top 15 countries by population
top15 = df_countries.nlargest(
    15,
    "2024"
)[
    ["Country Name", "2024"]
].copy()

# Convert population into millions
top15["Population_Millions"] = (
    top15["2024"] / 1_000_000
)

# Reverse order for horizontal bar chart
top15 = top15.sort_values(
    "Population_Millions",
    ascending=True
)

# Create the bar chart
plt.figure(figsize=(12, 7))

bars = plt.barh(
    top15["Country Name"],
    top15["Population_Millions"],
    color="#4C72B0
)

# Add labels and title
plt.xlabel("Population (millions)")
plt.ylabel("Country")
plt.title(
    "Top 15 Most Populous Countries (2024)",
    fontsize=15
)

# Add population labels to each bar
for bar in bars:
    width = bar.get_width()

    plt.text(
        width + 10,
        bar.get_y() + bar.get_height() / 2,
        f"{width:,.0f}M",
        va="center",
        fontsize=9
    )

plt.tight_layout()

# Save the chart
plt.savefig(
    "top15_population_bar_chart.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# STEP 6: DISPLAY TOP 15 COUNTRIES

print("\nTop 15 Most Populous Countries in 2024:")

print(
    top15.sort_values(
        "Population_Millions",
        ascending=False
    )[
        ["Country Name", "Population_Millions"]
    ].to_string(
        index=False,
        formatters={
            "Population_Millions": "{:,.2f}".format
        }
    )
)


# STEP 7: HISTOGRAM OF POPULATION DISTRIBUTION

# Extract population values
population = df_countries["2024"].dropna()

# Create logarithmically spaced bins
bins = np.logspace(
    np.log10(population.min()),
    np.log10(population.max()),
    20
)

# Create histogram
plt.figure(figsize=(12, 6))

plt.hist(
    population,
    bins=bins,
    color="#DD8452",
    edgecolor="black"
)

# Use logarithmic scale on the x-axis
plt.xscale("log")

# Add chart labels
plt.xlabel("Population (log scale)")
plt.ylabel("Number of countries")
plt.title(
    "Distribution of Country Population (2024)",
    fontsize=15
)

plt.tight_layout()

# Save the histogram
plt.savefig(
    "population_histogram.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# STEP 8: DESCRIPTIVE STATISTICS

print("\nPopulation Statistics for 2024")
print("=" * 40)

print(
    "Total countries:",
    len(population)
)

print(
    "Total population across selected countries:",
    f"{population.sum():,.0f}"
)

print(
    "Mean population:",
    f"{population.mean():,.0f}"
)

print(
    "Median population:",
    f"{population.median():,.0f}"
)

print(
    "Minimum population:",
    f"{population.min():,.0f}"
)

print(
    "Maximum population:",
    f"{population.max():,.0f}"
)

print(
    "Standard deviation:",
    f"{population.std():,.0f}"
)


# STEP 9: ANALYZE POPULATION DISTRIBUTION

mean_population = population.mean()
median_population = population.median()

print("\nDistribution Analysis")
print("=" * 40)

if mean_population > median_population:
    print(
        "The distribution is positively skewed. "
        "A few highly populated countries increase the mean."
    )

elif mean_population < median_population:
    print(
        "The distribution is negatively skewed."
    )

else:
    print(
        "The mean and median population are approximately equal."
    )

# Find countries with populations above 100 million
large_countries = df_countries[
    df_countries["2024"] > 100_000_000
][
    ["Country Name", "2024"]
].sort_values(
    "2024",
    ascending=False
)

print(
    "\nCountries with populations above 100 million:"
)

print(
    large_countries.to_string(
        index=False,
        formatters={
            "2024": "{:,.0f}".format
        }
    )
)


# STEP 10: DISPLAY COUNTRIES WITH THE SMALLEST POPULATIONS

small_countries = df_countries.nsmallest(
    10,
    "2024"
)[
    ["Country Name", "2024"]
]

print("\n10 Countries with the Smallest Populations:")

print(
    small_countries.to_string(
        index=False,
        formatters={
            "2024": "{:,.0f}".format
        }
    )
)


# STEP 11: FINAL OBSERVATIONS

print("\n")
print("=" * 60)
print("FINAL OBSERVATIONS")
print("=" * 60)

print(
    "1. The bar chart identifies the 15 most populous countries."
)

print(
    "2. The histogram illustrates population differences "
    "among countries."
)

print(
    "3. Population sizes vary considerably across countries."
)

print(
    "4. A small number of countries have very large populations."
)

print(
    "5. The logarithmic x-axis makes the histogram easier "
    "to interpret."
)

print(
    "6. Data cleaning helps exclude regional aggregates "
    "and missing population values."
)


# STEP 12: CONCLUSION

print("\n")
print("=" * 60)
print("CONCLUSION")
print("=" * 60)

print(
    "Population data was cleaned and analyzed using Python, "
    "Pandas, NumPy, Matplotlib, and Seaborn."
)

print(
    "The horizontal bar chart compares the populations of "
    "the 15 most populous countries in 2024."
)

print(
    "The histogram shows how population is distributed "
    "across countries."
)

print(
    "Descriptive statistics provide additional information "
    "about the population data."
)

print(
    "These visualizations help us understand global population "
    "patterns and identify differences between countries."
)

print("\nAnalysis completed successfully!")
