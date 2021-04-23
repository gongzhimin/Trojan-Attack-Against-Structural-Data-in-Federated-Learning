import modin.pandas as pd

data_dir = "./data/train_data.csv"
data = pd.read_csv(data_dir, sep='|')

# find the maximum value of each column
# max{data} < 2^32 - 1
for i, column in data.items():
    if column.dtype != "int64":
        break
    max_value = column.max()
    min_value = column.min()
    if max_value > 2147483647 or min_value <  -2147483648:
        print(i, " overflow!")
    else:
        print(i, " fine")

# delete validness field
del data["communication_onlinerate"]

# convert int64 to int32
data = pd.DataFrame(data, dtype="int32")

# save the data as csv file
data.to_csv("../data/new_train_data.csv", index_label="index")

# make data slice
size = len(data)
selected = int(0.75 * size)
new_data = data[: selected]
new_data.to_csv("../data/train_data_75.csv", index_label="index")