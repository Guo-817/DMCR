import os
from collections import defaultdict


def process_data(input_file):

    with open(input_file, 'r') as f:
        headers = f.readline().strip().split('\t')

    rating_columns = headers[2:]

    rating_data = {col: defaultdict(list) for col in rating_columns}

    with open(input_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            user_id = parts[0]
            item_id = parts[1]

            for i, col in enumerate(rating_columns):
                rating = parts[2 + i]
                if rating == '1':
                    rating_data[col][user_id].append(item_id)

    for col in rating_columns:
        output_filename = f"RateBeer/{col.split(':')[0]}.txt"
        with open(output_filename, 'w') as f_out:
            for user_id, item_ids in rating_data[col].items():
                line = f"{user_id}\t{' '.join(item_ids)}\n"
                f_out.write(line)


if __name__ == "__main__":
    input_file = "RateBeer/RateBeer_train.inter"
    process_data(input_file)
    print("Processing completed！")
