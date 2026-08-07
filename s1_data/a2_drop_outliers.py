import duckdb
import os

"""
Drop partial sales from the training set: >4000 sqft homes sold far below market.
These are high-leverage points under squared-error loss and they also produce
single-observation one-hot levels that downstream models memorise.

Filtered here, ahead of imputation and the model-specific prep scripts, so every
s1_data script splits the same rows.
"""

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

dropped = conn.execute("""
    delete from train
    where GrLivArea > 4000 and SalePrice < 300000
    returning Id, GrLivArea, SalePrice;
""").fetchall()

remaining = conn.execute("select count(*) from train").fetchone()[0]

for row_id, grlivarea, saleprice in dropped:
    print(f"Dropped Id {row_id}: GrLivArea {grlivarea:.0f}, SalePrice {saleprice}")
print(f"Dropped {len(dropped)} outliers, {remaining} training rows remain")

print(conn.execute("SHOW TABLES").fetchall())

conn.close()
