# dataframes-python

Clean examples of dataframe code. Uses synthetic retail sales data as a running example.

- 12 stores
- 3970 products
- 63 days (9 weeks)

### packages used


### data/

`synthetic_data.npz`. Compressed Numpy file containing three arrays:

`dates`: (63, ) array of integers (not dates).  
`synth_sales_data`: (12, 3970, 63) array of integer sales quantities.  
`fitted_line`: (12, 3970, 63) array of doubles fitted through the sales quantities.  

### src/
