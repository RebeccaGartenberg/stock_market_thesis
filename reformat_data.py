import datetime
import yaml

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

stock_symbol_list = params.get("stock_symbols")
data_path = params.get("data_path")
year = params.get("year")

for stock_symbol in stock_symbol_list:
    file = open(f"./{data_path}/{stock_symbol}_{year}.csv", "r+", newline ='')
    lines = file.readlines()

    # Rewrite each edited line at the end of the file
    for line in lines:
        if "datetime.datetime" in line or "datetime.timezone" in line:
            line = line.replace("datetime.datetime", "datetime").replace("datetime.timezone", "timezone")
            file.write(line)
    file.close()

    file = open(f"./{data_path}/{stock_symbol}_{year}.csv", "r", newline ='')
    lines = file.readlines()
    file = open(f"./{data_path}/{stock_symbol}_{year}.csv", "w", newline ='')

    # Overwrite file with the new lines that were appended previously
    for line in lines:
        if "datetime.datetime" not in line and "datetime.timezone" not in line:
            file.write(line)
    file.close()
