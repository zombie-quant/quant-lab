import pandas as pd

close = pd.Series([10.0, 11.0, 12.1, 11.5, 12.0])
print("原始价格:\n", close.to_list())

print("\n1) pct_change():\n", close.pct_change().round(4).to_list())

print("\n2) pct_change().shift(-1):\n", close.pct_change().shift(-1).round(4).to_list())

print("\n3) pct_change(2):\n", close.pct_change(2).round(4).to_list())
