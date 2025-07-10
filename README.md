# dataframes-python

Clean examples of dataframe code. Uses synthetic retail sales data as a running example.

- 12 stores
- 3970 products
- 63 days (9 weeks)

### packages used
`pathlib`: for managing load and write paths.  
`Numpy`: to load and re-shape .npz data.  
`Pandas`: widely used dataframe.  
`Polars`: good for ETL, data wrangling and large datasets. Limited interop with matplotlib.  
`statsmodels`: analyse data.  
`matplotlib.pyplot`: plot charts.

### data/

`synthetic_data.npz`. Compressed Numpy file containing three arrays:

- `dates`: (63, ) array of integers (not dates).  
- `synth_sales_data`: (12, 3970, 63) array of integer sales quantities.  
- `fitted_line`: (12, 3970, 63) array of doubles fitted through the sales quantities. 

`synthetic_data.csv.gz`. Compressed csv file containing the same data as the .npz file but in long format. The integer date values have been converted to `datetime64`.

### src/

`datasetup.py`: loads npz data, reshapes it and saves it as compressed csv.

`pandasuse.py`:
- loads data from compressed csv.  
- wrangles data: cast dates to datetime64; rename a column; add day-of-week columns.  
- aggregate data: from store-product-date level to product-date level.  
- analyse the data: using statsmodel.  
- package up the model results into a dataframe.  
- plot a chart.

`polarsuse.py`
- deliberately mirrors the workflow in `pandasuse.py` to allow a side-by-side comparison.
- Polars excels at ETL and data wrangling - fast, expressive, and supports lazy evaluation (ideal for large datasets).
- since Polars has limited integration with matplotlib, this example transitions to Pandas for modelling and plotting.
