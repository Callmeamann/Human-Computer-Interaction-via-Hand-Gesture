import csv

input_file = "temp.csv"
output_file = "temp.csv"
desired_integer = 10

with open(input_file, mode="r", newline="") as file:
    reader = csv.reader(file)
    data = [list(map(float, row)) for row in reader]

for line in data:
    line[0] = desired_integer

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Modified data saved to {output_file}")
