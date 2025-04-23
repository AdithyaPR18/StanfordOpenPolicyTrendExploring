import polars as pl
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import  numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import random

pl.Config.set_tbl_rows(5000000000)
file_path = "openpolicing.parquet"

# department_counts_df = (
#     pl.scan_parquet(file_path)
#     .filter(pl.col("department_name").is_not_null())
#     .group_by("department_name")
#     .agg(pl.len().alias("count"))
#     .sort("count", descending=True)
#     .collect()
# )
#
# race_df = (
#     pl.scan_parquet(file_path)
#     .filter(pl.col("subject_race").is_not_null())
#     .group_by("subject_race")
#     .agg(pl.len().alias("count"))
#     .sort("count", descending=True)
#     .collect()
# )
#
# age_df = (
#     pl.scan_parquet(file_path)
#     .filter(pl.col("subject_age").is_not_null())
#     .group_by("subject_age")
#     .agg(pl.len().alias("count"))
#     .sort("count", descending=True)
#     .collect()
# )
#
# sex_df = (
#     pl.scan_parquet(file_path)
#     .filter(pl.col("subject_sex").is_not_null())
#     .group_by("subject_sex")
#     .agg(pl.len().alias("count"))
#     .sort("count", descending=True)
#     .collect()
# )
#
#
# type_df = (
#     pl.scan_parquet(file_path)
#     .filter(pl.col("type").is_not_null())
#     .group_by("type")
#     .agg(pl.len().alias("count"))
#     .sort("count", descending=True)
#     .collect()
# )
#
# reason_for_stop = (
#     pl.scan_parquet(file_path)
#     .filter(pl.col("reason_for_stop").is_not_null())
#     .group_by("reason_for_stop")
#     .agg(pl.len().alias("count"))
#     .sort("count", descending=True)
#     .collect()
# )
#
#
# print(department_counts_df)
# print(race_df)
# print(age_df)
# print(sex_df)
# print(type_df)
# print(reason_for_stop)
#
# dept_race_df = (
#     pl.scan_parquet(file_path)
#     .filter(
#         pl.col("department_name").is_not_null() &
#         pl.col("subject_race").is_not_null()
#     )
#     .group_by(["department_name", "subject_race"])
#     .agg(pl.len().alias("count"))
#     .sort(["department_name", "count"], descending=[False, True])
#     .collect()
# )
#
# print(dept_race_df)
#
# reason_race_df = (
#     pl.scan_parquet(file_path)
#     .filter(
#         pl.col("reason_for_stop").is_not_null() &
#         pl.col("subject_race").is_not_null()
#     )
#     .group_by(["reason_for_stop", "subject_race"])
#     .agg(pl.len().alias("count"))
#     .sort(["reason_for_stop", "count"], descending=[False, True])
#     .collect()
# )
#
# print(reason_race_df)
#
# df = pl.read_parquet(file_path)
# df.get(1, "time")
#
# build your 30‑minute bin expression and alias it correctly
# mins_bin = (
#     # hours*60 + minutes
#     pl.col("time").str.slice(0, 2).cast(pl.UInt16) * 60
#     + pl.col("time").str.slice(3, 2).cast(pl.UInt16)
# )
# mins_bin = (mins_bin // 30 * 30)
#
# # 2) Turn that into an "HH:MM" string, alias it "time"
# time_bin_str = (
#     (mins_bin // 60).cast(str).str.zfill(2)
#     + ":"
#     + (mins_bin % 60).cast(str).str.zfill(2)
# ).alias("time")
#
# arrest_30m_bins = (
#     pl.scan_parquet(file_path)
#     .filter(
#         pl.col("arrest_made").str.to_lowercase() == "true",
#         pl.col("time").is_not_null(),
#         pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$")
#     )
#     .group_by(time_bin_str)               # group on your "HH:MM" label
#     .agg(pl.len().alias("count"))         # count with pl.len()
#     .sort("time")                         # sort chronologically
#     .collect()
# )
#

#print(arrest_30m_bins)
#
# mins_bin = (
#     pl.col("time").str.slice(0, 2).cast(pl.UInt16) * 60
#     + pl.col("time").str.slice(3, 2).cast(pl.UInt16)
# ) // 30 * 30
#
# # 2) format it to "HH:MM" and alias to "time"
# time_bin_str = (
#     (mins_bin // 60).cast(str).str.zfill(2)
#     + ":"
#     + (mins_bin % 60).cast(str).str.zfill(2)
# ).alias("time")
#
# # 3) scan, filter, then group by both the 30‑min bin AND race
# arrest_time_race = (
#     pl.scan_parquet(file_path)
#     .filter(
#         pl.col("arrest_made").str.to_lowercase() == "true",
#         pl.col("time").is_not_null(),
#         pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$")
#     )
#     .group_by([time_bin_str, pl.col("subject_race")])
#     .agg(pl.len().alias("count"))
#     .sort(["time", "subject_race"])
#     .collect()
# )
#
# #print(arrest_time_race)
#
# df = arrest_time_race.to_pandas()
# df_pivot = df.pivot(index="time", columns="subject_race", values="count")
#
# plt.figure()
# for race in df_pivot.columns:
#     plt.plot(df_pivot.index, df_pivot[race], label=race)
#
# plt.xlabel("Time of Day (30‑min intervals)")
# plt.ylabel("Number of Arrests")
# plt.xticks(rotation=90)
#
# # include the legend in the figure again
# plt.legend(title="Race", loc="best")
#
# plt.tight_layout()
# plt.show()
#
# mins_bin = (
#     pl.col("time").str.slice(0,2).cast(pl.UInt16)*60
#     + pl.col("time").str.slice(3,2).cast(pl.UInt16)
# ) // 30 * 30
#
# time_bin = (
#     (mins_bin // 60).cast(str).str.zfill(2)
#     + ":"
#     + (mins_bin % 60).cast(str).str.zfill(2)
# ).alias("time")
#
# reason_time_race = (
#     pl.scan_parquet(file_path)
#       .select(["time","subject_race","reason_for_stop","arrest_made"])
#       .filter(
#          pl.col("arrest_made").str.to_lowercase() == "true",
#          pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$")
#       )
#       .group_by([time_bin, "subject_race", "reason_for_stop"])
#       .agg(pl.len().alias("count"))
#       .sort(["time","subject_race","count"], descending=[False,False,True])
#       .collect()
# )
#
# print(reason_time_race)
#
# top_reasons = (
#   reason_time_race
#     .group_by("reason_for_stop")
#     .agg(pl.sum("count").alias("total"))
#     .sort("total", descending=True)
#     .head(10)
#     .select("reason_for_stop")
#     .to_series()
#     .to_list()
# )
#
# filtered = reason_time_race.filter(
#   pl.col("reason_for_stop").is_in(top_reasons)
# )
#
# print(top_reasons)
# print(filtered)

# --- compute_or function remains the same ---
# --- compute_or function remains the same ---



# --- You can remove or comment out the old compute_or function ---
# def compute_or(pdf: pd.DataFrame, title: str):
#    ... (old code) ...


def compute_or(pdf: pd.DataFrame, title: str):
  pdf = pdf.copy()
  pdf["arrest_made"] = pdf["arrest_made"].astype(int)

  X = pdf.drop(columns=["arrest_made", "mins"])
  y = pdf["arrest_made"]
  strat = pdf["subject_race"]

  Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.3, stratify=strat, random_state=42
  )

  ct = ColumnTransformer([
    ("race",
     OneHotEncoder(handle_unknown="ignore", drop=["white"], sparse_output=True),
     ["subject_race"]),
    ("others",
     OneHotEncoder(handle_unknown="ignore", sparse_output=True),
     ["time", "reason_for_stop", "department_name", "subject_age", "subject_sex"])
  ], remainder="drop")

  pipe = Pipeline([("ct", ct), ("clf", LogisticRegression(max_iter=1000))])
  pipe.fit(Xtr, ytr)

  feat_names = pipe.named_steps["ct"].get_feature_names_out()
  black_feat = next(f for f in feat_names if f.endswith("subject_race_black"))
  idx = list(feat_names).index(black_feat)
  oratio = np.exp(pipe.named_steps["clf"].coef_[0][idx])
  auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])

  print(f"{title}: OR(Black vs White) = {oratio:.2f}, AUC = {auc:.3f}")


def main():
  SAMPLE_SIZE = 10_000_000
  file_path = "openpolicing.parquet"

  # 1) Lazy‐load, filter, compute mins, and take a single 10 M‐row sample
  base = (
    pl.scan_parquet(file_path)
    .select([
      "arrest_made", "time", "reason_for_stop",
      "department_name", "subject_race",
      "subject_age", "subject_sex"
    ])
    .filter(
      pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"),
      pl.col("subject_race").is_not_null(),
      pl.col("arrest_made").is_not_null()
    )
    .with_columns(
      (pl.col("time")
       .str.strptime(pl.Time, "%H:%M:%S")
       .dt.hour() * 60
       + pl.col("time").str.slice(3, 2).cast(pl.UInt32)
       ).alias("mins")
    )
    .limit(SAMPLE_SIZE)
    .collect()
  )

  # 2) Binarize arrest_made
  sample = base.with_columns(
    (pl.col("arrest_made").str.to_lowercase() == "true").alias("arrest_made")
  )

  # 3) Optional: print race distribution
  print("▶ Race counts in 10 M sample:")
  print(
    sample
    .group_by("subject_race")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
  )

  # 4) Convert to pandas and run the model once
  compute_or(sample.to_pandas(), "All times")



# --- compute_or function (required for modeling) ---
def compute_or(pdf: pd.DataFrame, title: str):

  pdf = pdf.copy()
  # Ensure arrest_made is integer (0 or 1)
  pdf["arrest_made"] = pdf["arrest_made"].astype(int)

  # Define features (X) and target (y)
  # Drop 'mins' as 'time' (categorical representation of time) is used
  X = pdf.drop(columns=["arrest_made", "mins"])
  y = pdf["arrest_made"]

  # Use subject_race for stratified splitting
  # Check if multiple race categories exist for stratification
  if pdf["subject_race"].nunique() > 1:
      strat = pdf["subject_race"]
      print(f"Stratifying by subject_race for '{title}'.")
  else:
      strat = None # Cannot stratify if only one category exists
      print(f"Warning: Only one race category found in sample for '{title}'. Cannot stratify split.")


  # Split data into training and testing sets (70% train, 30% test)
  Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.3, stratify=strat, random_state=42 # Use stratify if possible
  )

  # Create a ColumnTransformer for preprocessing
  # OneHotEncode categorical features, using 'white' as reference for race
  ct = ColumnTransformer([
    ("race",
     OneHotEncoder(handle_unknown="ignore", drop=["white"], sparse_output=True),
     ["subject_race"]),
    ("others",
     OneHotEncoder(handle_unknown="ignore", sparse_output=True),
     # Treat time and age as categorical for this model
     ["time", "reason_for_stop", "department_name", "subject_age", "subject_sex"])
  ], remainder="drop", verbose_feature_names_out=False) # Keep feature names cleaner

  # Create a pipeline: Preprocessing -> Logistic Regression
  pipe = Pipeline([("ct", ct), ("clf", LogisticRegression(max_iter=1000))])

  # Train the model
  print(f"Fitting model for '{title}'...")
  pipe.fit(Xtr, ytr)
  print("Fitting complete.")

  # Calculate Odds Ratio for Black vs White
  feat_names = pipe.named_steps["ct"].get_feature_names_out()
  # Manually construct the expected feature name after OHE
  black_feat_name = "subject_race_black"

  try:
    # Find the index of the feature corresponding to 'black' race
    idx = list(feat_names).index(black_feat_name)
    # Get the coefficient and calculate Odds Ratio
    oratio = np.exp(pipe.named_steps["clf"].coef_[0][idx])
    print(f"Coefficient for {black_feat_name}: {pipe.named_steps['clf'].coef_[0][idx]:.4f}")

  except ValueError:
     # Handle cases where 'black' might not be present after splitting or filtering
    print(f"Warning: Feature '{black_feat_name}' not found in model features for '{title}'. Odds Ratio cannot be calculated.")
    print("Available features:", feat_names)
    oratio = np.nan # Assign NaN if feature not found

  # Evaluate the model using AUC on the test set
  print(f"Calculating AUC for '{title}'...")
  auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
  print("AUC calculation complete.")

  # Print the results
  print(f"\n>>> RESULT for {title}: OR(Black vs White) = {oratio:.2f}, AUC = {auc:.3f}\n")


def again():
  # Define sample size for the night-time analysis
  # Using 10M based on the last successful detailed run provided
  NIGHT_SAMPLE_SIZE = 10_000_000
  file_path = "openpolicing.parquet" # Make sure this file exists

  # --- Define Time Ranges in Minutes ---
  night_start_min = 17 * 60  # 5 PM = 1020
  night_end_min = 3 * 60    # 3 AM = 180

  print(f"Processing data from: {file_path}")
  print(f"Target sample size for Night/Evening: {NIGHT_SAMPLE_SIZE:,}")
  print(f"Night/Evening period: >= {night_start_min} mins OR < {night_end_min} mins")

  # --- Collect, Process, and Model Night Sample ONLY ---
  print("\nCollecting Night/Evening sample...")
  try:
      # Use the robust approach: filter fully, then limit
      base_lazy_night = (
          pl.scan_parquet(file_path)
          .select([
              "arrest_made", "time", "reason_for_stop",
              "department_name", "subject_race",
              "subject_age", "subject_sex"
          ])
          # Apply all necessary filters first - ensures data quality for model
          .filter(
              pl.col("time").str.contains(r"^\d{2}:\d{2}:\d{2}$"),
              pl.col("arrest_made").is_not_null(), # Target must be non-null
              pl.col("reason_for_stop").is_not_null(),
              pl.col("department_name").is_not_null(),
              pl.col("subject_race").is_not_null(),
              pl.col("subject_age").is_not_null(),
              pl.col("subject_sex").is_not_null()
          )
          # Calculate 'mins' using reliable string slicing
          .with_columns(
              (pl.col("time").str.slice(0, 2).cast(pl.UInt32, strict=False) * 60
               + pl.col("time").str.slice(3, 2).cast(pl.UInt32, strict=False)
               ).alias("mins")
          )
          # Ensure 'mins' calculation was successful (not null from cast errors)
          .filter(pl.col("mins").is_not_null())
          # Apply the time filter for night/evening (5 PM to 3 AM)
          .filter(
              (pl.col("mins") >= night_start_min) | (pl.col("mins") < night_end_min)
          )
          # Limit *after* all filtering
          .limit(NIGHT_SAMPLE_SIZE)
      )
      # Execute the query and collect data into memory
      night_collected = base_lazy_night.collect()

      # Process and Model Night Sample
      if night_collected.height > 0:
          print(f"\n--- Processing Night/Evening Sample (5 PM - 3 AM) ---")
          print(f"Actual sample size: {night_collected.height:,}")
          # Binarize arrest_made (True/False)
          night_sample = night_collected.with_columns(
              (pl.col("arrest_made").str.to_lowercase() == "true").alias("arrest_made")
          )
          # Optional: Print race distribution in the sample
          print("▶ Race counts in Night/Evening sample:")
          print(
              night_sample
              .group_by("subject_race")
              .agg(pl.len().alias("count"))
              .sort("count", descending=True)
          )
          # Convert to Pandas and run the modeling function
          compute_or(night_sample.to_pandas(), "Night/Evening (5 PM - 3 AM)")
      else:
          # This shouldn't happen based on previous runs, but included for safety
          print("\n--- No data found or collected for Night/Evening sample after filtering ---")
          print("    (Check filters and data completeness for this time period if unexpected)")

  except Exception as e:
      print(f"\nAn error occurred during the Night/Evening analysis: {e}")
      # Optional: uncomment below for detailed error traceback
      # import traceback
      # print(traceback.format_exc())


if __name__ == "__main__":
  main()
  again()