import pandas as pd
import numpy as np
import os


def split_data_with_popularity_control(input_path, output_dir):

    # Function：Divide the data set

    try:
        # ==================== Loading ====================
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The input file does not exist: {input_path}")

        df = pd.read_csv(input_path, sep='\t')
        print(f"Data sample size: {len(df)}")

        required_columns = {'user_id:token', 'item_id:token'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"The lack of necessary column: {missing}")

        item_popularity = df['item_id:token'].value_counts().reset_index()
        item_popularity.columns = ['item_id:token', 'popularity']
        item_popularity['rank'] = item_popularity['popularity'].rank(method='first', ascending=False)
        total_items = len(item_popularity)

        def assign_tier(rank):
            if rank <= total_items * 0.2:
                return 'high'
            elif rank <= total_items * 0.8:
                return 'medium'
            else:
                return 'low'

        item_popularity['tier'] = item_popularity['rank'].apply(assign_tier)

        df = df.merge(item_popularity[['item_id:token', 'popularity', 'tier']], on='item_id:token', how='left')

        # ==================== Sampling ====================
        train_records = []
        test_records = []

        for user_id, user_group in df.groupby('user_id:token'):
            test_samples = []

            for tier in ['high', 'medium', 'low']:
                tier_data = user_group[user_group['tier'] == tier]
                if len(tier_data) == 0:
                    continue

                tier_test_size = min(len(tier_data), max(1, int(0.2 * len(tier_data))))
                sampled = tier_data.sample(n=tier_test_size, random_state=42)
                test_samples.append(sampled)

            if test_samples:
                test_df_user = pd.concat(test_samples)
                train_df_user = user_group[~user_group.index.isin(test_df_user.index)]

                test_records.append(test_df_user)
                train_records.append(train_df_user)
            else:
                train_records.append(user_group)

        train_df = pd.concat(train_records).drop(columns=['popularity', 'tier', 'rank'], errors='ignore')
        test_df = pd.concat(test_records).drop(columns=['popularity', 'tier', 'rank'], errors='ignore')

        overlap = set(train_df.index) & set(test_df.index)
        if overlap:
            raise RuntimeError(f"Data breach! Number of overlapping records: {len(overlap)}")

        # ==================== Save ====================
        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "BeerAdvocate_train.inter")
        test_path = os.path.join(output_dir, "BeerAdvocate_test.inter")

        train_df.to_csv(train_path, sep='\t', index=False)
        test_df.to_csv(test_path, sep='\t', index=False)

    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":

    split_data_with_popularity_control(
        input_path="RateBeer/RateBeer_processed.inter",
        output_dir="RateBeer"
    )
