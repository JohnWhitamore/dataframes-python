from pathlib import Path
import polars as pl
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
Polars is really good for ETL-type data wrangling.
However, it doesn't play nicely with matplotlib.
For that reason, the code uses Polars for ETL and
then switches to Pandas to collect the results of
the statistical model and pass them into matplotlib
for plotting.

It seems to be a case of Polars AND Pandas, rather
than Polars OR Pandas.
"""


# Define data path
load_path = Path.cwd().parent / "data" / "synthetic_data.csv.gz"

"""
Polars
"""

# Load and parse the csv file

# ... automatically infers compression type from ".gz" extension. no need to specify.
df = pl.read_csv(load_path, try_parse_dates=True)

# ... rename column for brevity
df = df.rename({"synth_sales_data": "sales"})

# ... extract day of week: 1 is Monday, 7 is Sunday. ISO weekday standard.
df = df.with_columns([
    pl.col("date").dt.weekday().alias("dow")
])

# ... cast to string to make one-hot work. (unnecessary in this example)
df = df.with_columns(
    pl.col("dow").cast(pl.Utf8)
)

# ... create dummy variables i.e. one-hot encoding of days of week
days_of_week = df.select("dow").to_dummies()

# ... join ("horizontally stack") day of week columns to dataframe
df = df.hstack(days_of_week)

# ... drop the day-of-week column ("dow") as no longer needed
df = df.drop("dow")

# ... drop one weekday to avoid multicollinearity
df = df.drop("dow_1")

"""
Still Polars
"""

# Aggregate to product-date level from store-product-date level

# ... a list of day-of-week columns (because we dropped dow_1)
dow_columns = [col for col in df.columns if col.startswith("dow_")]

# ... aggregation
product_df = df.group_by(["product_id", "date"]).agg([
    pl.sum("sales"),
    *[pl.first(col) for col in dow_columns]
])

print(product_df.columns)
print(product_df.head())


# OLS regression per product

# ... initialise a dictionary to hold model results
results = {}

# ... loop through products
for product in product_df.partition_by("product_id", as_dict=False):
    
    # ... day of week
    X = product.select(dow_columns).to_numpy()
    
    # ... bias
    X = sm.add_constant(X)
    
    # ... dependent variable
    y = product["sales"].to_numpy()
    
    # ... fit the model
    model = sm.OLS(y, X).fit()
    
    # ... place the model results in a dict
    results[product[0, "product_id"]] = model
    
"""
Switch to Pandas
"""

# Create a pandas data frame in which to hold the model results

# ... initialise a list of rows of parameter values
param_rows = []

# ... loop
for product_id, model in results.items():
    
    param_names = model.model.exog_names
    
    row = pd.Series(
        model.params,
        index=param_names
    )
    
    row["product_id"] = product_id
    param_rows.append(row)

# ... create a Pandas dataframe from the rows of parameter values
coef_df = pd.DataFrame(param_rows)

"""
statsmodel has renamed the dow columns to become x columns instead.
"""

# Handle the new column names. We didn't need to do this with Pandas.

# ... identify columns relating to days of the week
weekday_cols = [col for col in coef_df.columns if col.startswith("x")]


# ... sort by weekday number for clarity: x2 (Tue), ..., x7 (Sun)
weekday_cols = sorted(
    weekday_cols,
    key=lambda x: int(x[1:])  # Extract number after 'x'
)

"""
And now to matplotlib
"""

# Create the boxplot

# ... re-shape from wide to long format
melted = coef_df.melt(
    id_vars="product_id",
    value_vars=weekday_cols,
    var_name="weekday",
    value_name="coefficient"
)

# ... groups boxplots by weekday
melted.boxplot(
    column="coefficient",
    by="weekday",
    grid=False
)
# ... axes and labels
plt.axhline(y=0, linestyle="--", color="gray", linewidth=1)
plt.title("Distribution of Day-of-Week Effects")
plt.suptitle("")
plt.ylabel("Sales Coefficient")
plt.xlabel("Weekday Dummy")

# ... display the plot
plt.tight_layout()
plt.show()