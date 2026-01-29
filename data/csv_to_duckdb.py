import duckdb
import pandas as pd
from data.gdrive_download import *

data_dict = {
    "train.csv": "1r_4AM9FYosvw_Ubd_8mu8YlIHRiqTjgs",
    "test.csv": "1SkMch2UNTCMDmDcK6OL2fd-ZGNGj2SSB"
}
folder = "data"
database = "AmesHousePrice.duckdb"
os.makedirs(folder, exist_ok=True)
database_path = os.path.join(folder, database)
train_path = os.path.join(folder, "train.csv")
test_path = os.path.join(folder, "test.csv")

for filename, file_id in data_dict.items():
    download_from_drive(file_id, filename, folder)

conn = duckdb.connect(database = database_path, read_only = False)

# Not converting value 'None' to NA because None is a valid category for MasVnrType
# None in MasVnrType means (no masonry veneer type), but need to investigate NA value
default_na = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", 
              "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", 
              "NULL", "NaN", "n/a", "na", "nan", "null"]
nullstr_sql = "[" + ", ".join(f"'{x}'" for x in default_na) + "]"

for table, path in {"train": train_path, "test": test_path}.items():
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * FROM read_csv_auto(
            '{path}',
            nullstr={nullstr_sql}
        );  
                """)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()

print(f"Saved DuckDB database to {database_path}")
