# Databricks notebook source
# DBTITLE 1,Pipeline Overview
# MAGIC %md
# MAGIC # Silver to Gold Pipeline
# MAGIC
# MAGIC ## Purpose
# MAGIC Transform cleaned credit card data into a single analytics-ready gold table for ML training.
# MAGIC
# MAGIC ## Data Flow
# MAGIC * **Input**: `end_to_end_credit_default.uci_credit_card_silver`
# MAGIC * **Output**: `end_to_end_credit_default.uci_credit_card_gold`
# MAGIC
# MAGIC ## Feature Engineering
# MAGIC 1. **Drop ID column** (not predictive)
# MAGIC 2. **One-hot encode categorical features**: SEX, EDUCATION, MARRIAGE
# MAGIC 3. **Keep all numeric features**: payment history, bill amounts, payment amounts
# MAGIC 4. **Keep target variable**: de_pay
# MAGIC
# MAGIC ## Next Steps
# MAGIC Create a separate **ML Training Notebook** that:
# MAGIC * Loads this gold table
# MAGIC * Applies model-specific preprocessing (scaling, sampling)
# MAGIC * Trains multiple models (Logistic Regression, XGBoost, etc.)
# MAGIC * Tracks experiments with MLflow

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC Select * from end_to_end_credit_default.uci_credit_card_silver

# COMMAND ----------

# DBTITLE 1,Load silver table
# Load silver table
df_silver = spark.table('end_to_end_credit_default.uci_credit_card_silver')

print(f"Silver table records: {df_silver.count():,}")
print(f"Silver table columns: {len(df_silver.columns)}")
print(f"\nSchema:")
df_silver.printSchema()

display(df_silver.limit(5))

# COMMAND ----------

# DBTITLE 1,Linear Regression Gold
from pyspark.sql.functions import col, when

# Start with silver table and drop ID (non-predictive)
df_gold = df_silver.drop('ID')

# One-hot encode categorical features
# Using efficient .withColumns() instead of .withColumn() in loop
categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

print("Creating one-hot encoded features...\n")

for cat_col in categorical_cols:
    # Get unique categories
    categories = [row[0] for row in df_gold.select(cat_col).distinct().collect()]
    categories = sorted([int(c) for c in categories])
    
    print(f"{cat_col}: {categories}")
    
    # Build dict for withColumns() - more efficient than loop with withColumn()
    new_columns = {}
    for category in categories:
        new_columns[f"{cat_col}_{category}"] = when(col(cat_col) == category, 1).otherwise(0)
    
    # Add all new columns at once
    df_gold = df_gold.withColumns(new_columns)
    
    # Drop original categorical column
    df_gold = df_gold.drop(cat_col)

print(f"\nGold table columns: {len(df_gold.columns)}")
print(f"\nNew binary feature columns:")
for col_name in sorted([c for c in df_gold.columns if any(cat in c for cat in categorical_cols)]):
    print(f"  - {col_name}")


# COMMAND ----------

# DBTITLE 1,Preview gold table features
# Preview the transformed data
print(f"Final gold table shape: {df_gold.count():,} rows × {len(df_gold.columns)} columns\n")
print("Column list:")
for idx, col_name in enumerate(df_gold.columns, 1):
    print(f"  {idx:2d}. {col_name}")

print("\nSample data:")
display(df_gold.limit(5))

# COMMAND ----------

# DBTITLE 1,Data quality validation
from pyspark.sql.functions import sum as spark_sum, count

print("=" * 70)
print("GOLD TABLE DATA QUALITY VALIDATION")
print("=" * 70)

# 1. Null value check
print("\n1. Null value check:")
null_counts = df_gold.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df_gold.columns])
null_df = null_counts.toPandas().T
null_df.columns = ['null_count']
nulls_found = null_df[null_df['null_count'] > 0]
if len(nulls_found) > 0:
    print("   ⚠ Warning: Columns with nulls found")
    display(nulls_found)
else:
    print("   ✓ No null values found")

# 2. Record count validation
silver_count = df_silver.count()
gold_count = df_gold.count()
print(f"\n2. Record count validation:")
print(f"   Silver: {silver_count:,} records")
print(f"   Gold:   {gold_count:,} records")
print(f"   Lost:   {silver_count - gold_count:,} records")

if silver_count == gold_count:
    print("   ✓ No records lost during transformation")
else:
    print("   ⚠ Warning: Record count mismatch!")

# 3. Feature engineering summary
print(f"\n3. Feature engineering summary:")
print(f"   Silver features: {len(df_silver.columns) - 2}  (excluding ID, de_pay)")
print(f"   Gold features:   {len(df_gold.columns) - 1}  (excluding de_pay)")
print(f"   New features:    {len(df_gold.columns) - len(df_silver.columns) + 1}")

# 4. Target distribution
default_rate = df_gold.filter(col('de_pay') == 1).count() / gold_count
print(f"\n4. Target variable distribution:")
print(f"   Default rate: {default_rate:.2%}")
df_gold.groupBy('de_pay') \
    .count() \
    .withColumn('percentage', col('count') / gold_count * 100) \
    .orderBy('de_pay') \
    .show()

print("=" * 70)

# COMMAND ----------

# DBTITLE 1,Write to gold table
# Write to gold table in Delta format
gold_table = 'end_to_end_credit_default.uci_credit_card_gold'

print(f"Writing to {gold_table}...")

df_gold.write \
    .mode('overwrite') \
    .format('delta') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(gold_table)

print(f"\n✓ Successfully wrote {gold_count:,} records to {gold_table}")
print(f"\n" + "=" * 70)
print("GOLD TABLE READY FOR ML TRAINING")
print("=" * 70)
