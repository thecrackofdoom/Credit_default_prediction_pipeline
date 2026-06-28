# Databricks notebook source
# DBTITLE 1,Pipeline Overview
# MAGIC %md
# MAGIC # Bronze to Silver Data Pipeline
# MAGIC
# MAGIC ## Purpose
# MAGIC This notebook transforms raw credit card default data from the bronze layer into cleaned, standardized data ready for analysis and modeling.
# MAGIC
# MAGIC ## Data Source
# MAGIC * **Bronze Table**: `end_to_end_credit_default.uci_credit_card`
# MAGIC * **Silver Table**: `end_to_end_credit_default.uci_credit_card_silver`
# MAGIC
# MAGIC ## Transformations Applied
# MAGIC
# MAGIC ### 1. Column Renaming
# MAGIC * `default.payment.next.month` → `de_pay` (target variable)
# MAGIC * `PAY_0` → `PAY_1` (first payment status)
# MAGIC
# MAGIC ### 2. Category Consolidation
# MAGIC * **EDUCATION**: Map rare/unknown categories (0, 5, 6) to 'others' (4)
# MAGIC * **MARRIAGE**: Map unknown (0) to 'others' (3)
# MAGIC
# MAGIC ### 3. Payment Status Normalization
# MAGIC * **PAY_1 through PAY_6**: Consolidate non-delay indicators (-2, -1, 0) to 0
# MAGIC   * -2 = no consumption
# MAGIC   * -1 = pay duly
# MAGIC   * 0 = pay duly
# MAGIC   * Positive values = months of payment delay
# MAGIC
# MAGIC ### 4. Data Quality Validation
# MAGIC * Null value checks
# MAGIC * Record count validation
# MAGIC * Target distribution analysis
# MAGIC
# MAGIC ## Architecture
# MAGIC * Uses **PySpark** for distributed processing (scales beyond single-machine memory)
# MAGIC * Writes to **Delta Lake** format (ACID transactions, time travel, schema evolution)
# MAGIC * Idempotent design (safe to re-run)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC Select * from end_to_end_credit_default.uci_credit_card

# COMMAND ----------

# DBTITLE 1,Load bronze table as PySpark DataFrame
# Load bronze table
df_bronze = spark.table('end_to_end_credit_default.uci_credit_card')

print(f"Bronze table records: {df_bronze.count():,}")
display(df_bronze.limit(30))


# COMMAND ----------

# DBTITLE 1,Rename columns and consolidate EDUCATION categories
from pyspark.sql.functions import when, col

# Rename columns for clarity
df_silver = df_bronze \
    .withColumnRenamed('default.payment.next.month', 'de_pay') \
    .withColumnRenamed('PAY_0', 'PAY_1')

# Consolidate EDUCATION: map rare categories (0, 5, 6) to 'others' (4)
# 0=unknown, 5-6=undefined
df_silver = df_silver.withColumn(
    'EDUCATION',
    when(col('EDUCATION').isin([0, 5, 6]), 4)
    .otherwise(col('EDUCATION'))
)

print("EDUCATION value distribution after consolidation:")
df_silver.groupBy('EDUCATION').count().orderBy('EDUCATION').show()

# COMMAND ----------

# DBTITLE 1,Consolidate MARRIAGE categories
# Consolidate MARRIAGE: map unknown (0) to 'others' (3)
# Business rule: 0=unknown in dataset documentation
df_silver = df_silver.withColumn(
    'MARRIAGE',
    when(col("MARRIAGE") == 0, 3).otherwise(col("MARRIAGE"))
)

print("MARRIAGE value distribution after consolidation:")
df_silver.groupBy('MARRIAGE').count().orderBy('MARRIAGE').show()


# COMMAND ----------

# DBTITLE 1,Normalize payment status columns (PAY_1 to PAY_6)
# Normalize payment status columns: map negative values and 0 to 0 (no delay)
# Business rule: -2=no consumption, -1=pay duly, 0=pay duly → all treated as 0 (no delay)
# Positive values indicate months of delay (1=1 month delay, 2=2 months, etc.)

pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

for pay_col in pay_columns:
    df_silver = df_silver.withColumn(
        pay_col,
        when(col(pay_col).isin([-2, -1, 0]), 0)
        .otherwise(col(pay_col))
    )

print("Payment status distribution after normalization (showing PAY_1 as example):")
df_silver.groupBy('PAY_1').count().orderBy('PAY_1').show()

# COMMAND ----------

# DBTITLE 1,Data quality validation
# Data Quality Checks
print("=" * 60)
print("DATA QUALITY VALIDATION")
print("=" * 60)

# Check for null values
from pyspark.sql.functions import sum as spark_sum, isnan, count

print("\n1. Null value check:")
null_counts = df_silver.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df_silver.columns])
null_counts.show(vertical=True)

# Record count validation
bronze_count = df_bronze.count()
silver_count = df_silver.count()
print(f"\n2. Record count validation:")
print(f"   Bronze records: {bronze_count:,}")
print(f"   Silver records: {silver_count:,}")
print(f"   Records lost: {bronze_count - silver_count:,}")

if bronze_count == silver_count:
    print("   ✓ No records lost during transformation")
else:
    print("   ⚠ Warning: Mismatch!")

# Default rate
default_rate = df_silver.filter(col('de_pay') == 1).count() / silver_count
print(f"\n3. Target variable distribution:")
print(f"   Default rate: {default_rate:.2%}")

# COMMAND ----------

# DBTITLE 1,Write cleaned data to silver table
# Write to silver table in Delta format
silver_table = 'end_to_end_credit_default.uci_credit_card_silver'

print(f"Writing cleaned data to {silver_table}...")

df_silver.write \
    .mode('overwrite') \
    .format('delta') \
    .option('overwriteSchema', 'true') \
    .saveAsTable(silver_table)

print(f"\n✓ Successfully wrote {silver_count:,} records to {silver_table}")
print(f"\nSilver table location: {spark.sql(f'DESCRIBE EXTENDED {silver_table}').filter('col_name == "Location"').collect()[0]['data_type']}")

# COMMAND ----------

