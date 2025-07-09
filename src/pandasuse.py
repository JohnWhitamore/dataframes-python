from pathlib import Path
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
A classic Pandas workflow:
- load data from compressed csv and cast dates to datetime64[ns]
- wrangle the data: we add day of week columns
- aggregate the data from store-product-date to product-date level
- run a statistical model on the data for each product
- present the results in a dataframe for further analysis or action
- plot a descriptive chart
"""

# Define path to data
cwd = Path.cwd()
load_path = cwd.parent / "data" / "synthetic_data.csv.gz"

# Load and format the data

# ... load
df = pd.read_csv(load_path, compression="gzip")

# ... cast the date column to datetime64[ns]
df["date"] = pd.to_datetime(df["date"], format = "%Y-%m-%d")
print(df["date"].dtype)

# ... rename synth_sales_data to sales
df.rename(columns={"synth_sales_data": "sales"}, inplace=True)


# Create day of week columns, one-hot encoded

# ... extract the day of week from the date column
# ... 0 is Monday; 6 is Sunday
df["day_of_week"] = df["date"].dt.dayofweek

# ... one-hot encoding

# ... given k categories, get_dummies creates k columns one-hot encoded
# ... i have cast them from Bool to float in order to use them in regression
# ... drop one of the day-of-week columns to avoid multicollinearity

days_of_week = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=True).astype(float)

# ... join with original dataframe
df = pd.concat([df, days_of_week], axis=1)

# ... drop the day_of_week column (no longer needed)
df.drop(columns="day_of_week", inplace=True)


# Aggregate to create product-date level data

# ... identify day-of-week dummy columns
dow_columns = [col for col in df.columns if col.startswith("dow_")]

# ... aggregate sales (but not day of week!) at product-date level

# ... {...} is a dictionary comprehension that builds a mapping for 
# ... each day of week column like {"dow_0": ("dow_0", "first")}
# ... the ** operator unpacks the dictionary comprehension to give
# ... sales = ("synth_sales_data", "sum"),
# ... dow_0 = ("dow_0", "first"),
# ... dow_1 = ("dow_1", "first"), etc

product_df = df.groupby(["product_id", "date"]).agg(
    sales=("sales", "sum"),
    **{col: (col, "first") for col in dow_columns}
).reset_index()


# Run OLS regression on product-level sales data

# ... dictionary to store regression results
results = {}

# .. loop over products
for product_id, group in product_df.groupby("product_id"):
    
    # ... day of week
    X = group[dow_columns]
    
    # ... bias
    X = sm.add_constant(X)
    
    # ... dependent variable
    y = group["sales"]
    
    # ... fit OLS regression
    model = sm.OLS(y, X).fit()
    
    # ... store the results
    results[product_id] = model


# Build a dataframe of coefficients with product_id as a column

# ... note the use of ** to unpack dictionaries again

coef_df = pd.DataFrame([
    {
        "product_id": product_id,
        **model.params.to_dict(),
        **{f"p_{k}": v for k, v in model.pvalues.items()}
    }
    for product_id, model in results.items()
])

# ... display results
print(coef_df.shape)
print(coef_df.head())
print(coef_df.columns)

# Plot a chart

coef_df.boxplot(column=[col for col in coef_df.columns if col.startswith("dow_")])
plt.title("Distribution of Day-of-Week Effects Across Products")
plt.ylabel("Sales Effect")
plt.show()