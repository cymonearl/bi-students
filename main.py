import os
import kagglehub 
import polars as pl
from kagglehub import KaggleDatasetAdapter
from plotnine import ggplot, geom_bar, aes, geom_point
from plotnine import geom_smooth, theme_minimal, labs, geom_boxplot
import seaborn as sns
import matplotlib.pyplot as plt

# df = kagglehub.load_dataset(
#     KaggleDatasetAdapter.POLARS,
#     "minahilfatima12328/performance-trends-in-education",
#     "StudentPerformanceFactors.csv"
# )

# df = df.collect()
# df.write_csv("StudentPerformanceFactors.csv")


df = pl.read_csv("StudentPerformanceFactors.csv")
df.head(0)
df = df.filter(pl.col("Exam_Score") <= 100)

df.to_pandas

df.describe()

for col in df.columns:
    string_vals = [v for v in df[col].unique() if isinstance(v, str)]
    if string_vals:
        print(f"\nColumn: {col}")
        print(string_vals)

# Plotting

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for col in df.columns:
    # Convert the column to pandas (required by Plotnine)
    col_df = df.select(col).to_pandas()

    # Skip numeric columns if you only want categories
    if col_df[col].dtype == "O" or col_df[col].dtype.name == "category":
        p = (
            ggplot(col_df, aes(x=col))
            + geom_bar(fill="#4C72B0", color="white")
            + labs(title=f"Distribution of {col}", x=col, y="Count")
            + theme_minimal()
        )
        file_path = os.path.join(output_dir, f"{col}_barplot.png")
        p.save(file_path, width=6, height=4, dpi=100)

plt.close("all")

(
    ggplot(df, aes(x = "Hours_Studied")) +
        geom_bar()
)
(
    ggplot(df, aes(x = "Attendance")) +
        geom_bar()
)
(
    ggplot(df, aes(x = "Sleep_Hours")) +
        geom_bar()
)
(
    ggplot(df, aes(x = "Physical_Activity")) +
        geom_bar()
)
(
    ggplot(df, aes(x = "Distance_from_Home")) +
        geom_bar()
)
(
    ggplot(df, aes(x = "Exam_Score")) +
        geom_bar()
)
(
    ggplot(df, aes(x = "Hours_Studied", y = "Exam_Score")) +
        geom_boxplot()
)

# Check Correlations

numeric_df = df.select(pl.col(pl.NUMERIC_DTYPES))
corr_matrix = numeric_df.to_pandas().corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,          # show correlation values
    fmt=".2f",           # format numbers
    cmap="coolwarm",     # color palette
    center=0,            # center at 0 for equal red/blue balance
    linewidths=0.5,      # grid line thickness
    square=True          # make each cell square
)

plt.title("Correlation Matrix Heatmap", fontsize=14, weight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

(
    ggplot(df, aes(x = "Exam_Score", y = "Hours_Studied")) +
    geom_point(color = "steelblue", size = 3) +
    geom_smooth(method = "lm", se = "TRUE", color = "red") +
    labs(title = "Scatter plot with Linear Fit (lm)",
        x = "X Variable", y = "Y Variable") +
    theme_minimal()
)
(
    ggplot(df, aes(x = "Exam_Score", y = "Attendance")) +
    geom_point(color = "steelblue", size = 3) +
    geom_smooth(method = "lm", se = "TRUE", color = "red") +
    labs(title = "Scatter plot with Linear Fit (lm)",
        x = "X Variable", y = "Y Variable") +
    theme_minimal()
)
(
    ggplot(df, aes(x = "Exam_Score", y = "Previous_Scores")) +
    geom_point(color = "steelblue", size = 3) +
    geom_smooth(method = "lm", se = "TRUE", color = "red") +
    labs(title = "Scatter plot with Linear Fit (lm)",
        x = "X Variable", y = "Y Variable") +
    theme_minimal()
)
(
    ggplot(df, aes(x = "Exam_Score", y = "Tutoring_Sessions")) +
    geom_point(color = "steelblue", size = 3) +
    geom_smooth(method = "lm", se = "TRUE", color = "red") +
    labs(title = "Scatter plot with Linear Fit (lm)",
        x = "X Variable", y = "Y Variable") +
    theme_minimal()
)


