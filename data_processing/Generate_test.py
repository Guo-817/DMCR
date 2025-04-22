from collections import defaultdict


def process_single_column(input_file, target_column):

    with open(input_file, 'r') as f:
        headers = f.readline().strip().split('\t')
        try:
            col_index = headers.index(target_column)
        except ValueError:
            print(f"Error: Column name '{target_column}' do not exist！")
            return

    # user_id -> [item_ids]
    user_items = defaultdict(list)

    with open(input_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            user_id = parts[0]
            item_id = parts[1]
            rating = parts[col_index]

            if rating == '1':
                user_items[user_id].append(item_id)

    output_filename = f"RateBeer/{target_column.split(':')[0]}_output.txt"
    with open(output_filename, 'w') as f_out:
        for user_id, item_ids in user_items.items():
            f_out.write(f"{user_id}\t{' '.join(item_ids)}\n")


if __name__ == "__main__":
    input_file = "RateBeer/RateBeer_test.inter"
    target_column = "rating:float"
    process_single_column(input_file, target_column)
    print("Processing completed！")
