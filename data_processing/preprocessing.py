import pandas as pd

input_file = "RateBeer/RateBeer.inter"
output_file = "RateBeer/RateBeer_processed.inter"

df = pd.read_csv(input_file, sep="\t")

df_processed = df.drop_duplicates(subset=["user_id:token", "item_id:token"], keep="first")

df_processed.to_csv(output_file, sep="\t", index=False)

print(f"The processed data has been saved to {output_file}")
