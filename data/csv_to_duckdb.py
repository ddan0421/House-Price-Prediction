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
        create or replace table {table} as
        select 
            cast(Id as int) as "Id"
            , cast(MSSubClass as int) as "MSSubClass"
            , cast(MSZoning as varchar) as "MSZoning"
            , cast(LotFrontage as int) as "LotFrontage"
            , cast(LotArea as int) as "LotArea"
            , cast(Street as varchar) as "Street"
            , cast(Alley as varchar) as "Alley"
            , cast(LotShape as varchar) as "LotShape"
            , cast(LandContour as varchar) as "LandContour"
            , cast(Utilities as varchar) as "Utilities"
            , cast(LotConfig as varchar) as "LotConfig"
            , cast(LandSlope as varchar) as "LandSlope"
            , cast(Neighborhood as varchar) as "Neighborhood"
            , cast(Condition1 as varchar) as "Condition1"
            , cast(Condition2 as varchar) as "Condition2"
            , cast(BldgType as varchar) as "BldgType"
            , cast(HouseStyle as varchar) as "HouseStyle"
            , cast(OverallQual as int) as "OverallQual"
            , cast(OverallCond as int) as "OverallCond"
            , cast(YearBuilt as int) as "YearBuilt"
            , cast(YearRemodAdd as int) as "YearRemodAdd"
            , cast(RoofStyle as varchar) as "RoofStyle"
            , cast(RoofMatl as varchar) as "RoofMatl"
            , cast(Exterior1st as varchar) as "Exterior1st"
            , cast(Exterior2nd as varchar) as "Exterior2nd"
            , cast(MasVnrType as varchar) as "MasVnrType"
            , cast(MasVnrArea as int) as "MasVnrArea"
            , cast(ExterQual as varchar) as "ExterQual"
            , cast(ExterCond as varchar) as "ExterCond"
            Foundation
            BsmtQual
            BsmtCond
            BsmtExposure
            BsmtFinType1
            BsmtFinSF1
            BsmtFinType2
            BsmtFinSF2
            BsmtUnfSF
            TotalBsmtSF
            Heating
            HeatingQC
            CentralAir
            Electrical
            1stFlrSF
            2ndFlrSF
            LowQualFinSF
            GrLivArea
            BsmtFullBath
            BsmtHalfBath
            FullBath
            HalfBath
            BedroomAbvGr
            KitchenAbvGr
            KitchenQual
            TotRmsAbvGrd
            Functional
            Fireplaces
            FireplaceQu
            GarageType
            GarageYrBlt
            GarageFinish
            GarageCars
            GarageArea
            GarageQual
            GarageCond
            PavedDrive
            WoodDeckSF
            OpenPorchSF
            EnclosedPorch
            3SsnPorch
            ScreenPorch
            PoolArea
            PoolQC
            Fence
            MiscFeature
            MiscVal
            MoSold
            YrSold
            SaleType
            SaleCondition



        from read_csv_auto(
            '{path}',
            nullstr={nullstr_sql}
        );        
                """)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()

print(f"Saved DuckDB database to {database_path}")
