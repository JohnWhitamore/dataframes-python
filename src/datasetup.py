import numpy as np
import pandas as pd
from pathlib import Path


"""
# pathlib:
# - cwd: current working directory
# - cwd.parent: up one level from the current working directory
# - note that / is not in quotation marks. pathlib overloads / to become an operator that joins strings.
# - we go from the current "src/" folder, up to the parent and back down to the "data/" folder.
"""

# Specify paths to data
cwd = Path.cwd()
load_path = cwd.parent / "data" / "synthetic_data.npz"
write_path = cwd.parent / "data" / "synthetic_data.csv"

"""
npz:
- very fast
- the .names attribute (data.names) contains the names of the arrays in the file
"""

# Load the npz file
data = np.load(load_path)

# Access individual arrays by name
dates = data["dates"]
synth_sales_data = data["synth_sales_data"]
fitted_line = data["fitted_line"]

# ... print the shapes of the arrays
print(dates.shape)
print(synth_sales_data.shape)
print(fitted_line.shape)

# Numbers of stores (S), products (P) and time-steps (N)
S, P, N = synth_sales_data.shape

# Create index arrays
store_ids = np.arange(S).reshape(S, 1, 1)
product_ids = np.arange(P).reshape(1, P, 1)
date_ids = np.arange(N).reshape(1, 1, N)

# Broadcast to full shape
# ... ravel() flattens a multi-dimensional array into a 1D array
# ... ravel() returns a view (memory efficient but can fail)
# ... flatten() returns a copy (inefficient but safer)
store_col = np.broadcast_to(store_ids, synth_sales_data.shape).ravel()
product_col = np.broadcast_to(product_ids, synth_sales_data.shape).ravel()
date_col = np.broadcast_to(date_ids, synth_sales_data.shape).ravel()

# Flatten the data
sales_flat = synth_sales_data.ravel()
fitted_flat = fitted_line.ravel()

# Convert integer offsets to actual dates

# ... define the start date
start_date = pd.Timestamp("2025-06-01")

# ... add offsets to the start date
date_strings = [start_date + pd.Timedelta(days=int(i)) for i in date_col]

# ... create a date column containing timestamp data
date_col = pd.to_datetime(date_strings)

# Build the DataFrame
df = pd.DataFrame({
    "store_id": store_col,
    "product_id": product_col,
    "date": date_col,
    "synth_sales_data": sales_flat,
    "fitted_line": fitted_flat
})

print(df.head())

# Save to CSV
df.to_csv(write_path, index=False)