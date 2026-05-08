import duckdb
from s1_data.gdrive_download import download_from_drive

data_dict = {
    "train.csv": "1r_4AM9FYosvw_Ubd_8mu8YlIHRiqTjgs",
    "test.csv": "1SkMch2UNTCMDmDcK6OL2fd-ZGNGj2SSB"
}
base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)
train_path = os.path.join(base_folder, "train.csv")
test_path = os.path.join(base_folder, "test.csv")

for filename, file_id in data_dict.items():
    download_from_drive(file_id, filename, base_folder)

conn = duckdb.connect(database = database_path, read_only = False)

# Not converting value 'None' to NA because None is a valid category for MasVnrType
# None in MasVnrType means (no masonry veneer type), but need to investigate NA value
default_na = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", 
              "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", 
              "NULL", "NaN", "n/a", "na", "nan", "null"]
nullstr_sql = "[" + ", ".join(f"'{x}'" for x in default_na) + "]"

for table, path in {"train": train_path, "test": test_path}.items():
    target_col = ', cast(SalePrice as int) as "SalePrice"' if table == "train" else ""
    conn.execute(f"""
        create or replace table {table} as
        select 
            cast(Id as int) as "Id"
            , cast(MSSubClass as int) as "MSSubClass"
            , cast(MSZoning as varchar) as "MSZoning"
            , cast(LotFrontage as double) as "LotFrontage"
            , cast(LotArea as double) as "LotArea"
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
            , cast(MasVnrArea as double) as "MasVnrArea"
            , cast(ExterQual as varchar) as "ExterQual"
            , cast(ExterCond as varchar) as "ExterCond"
            , cast(Foundation as varchar) as "Foundation"
            , cast(BsmtQual as varchar) as "BsmtQual"
            , cast(BsmtCond as varchar) as "BsmtCond"
            , cast(BsmtExposure as varchar) as "BsmtExposure"
            , cast(BsmtFinType1 as varchar) as "BsmtFinType1"
            , cast(BsmtFinSF1 as double) as "BsmtFinSF1"
            , cast(BsmtFinType2 as varchar) as "BsmtFinType2"
            , cast(BsmtFinSF2 as double) as "BsmtFinSF2" 
            , cast(BsmtUnfSF as double) as "BsmtUnfSF"
            , cast(TotalBsmtSF as double) as "TotalBsmtSF"
            , cast(Heating as varchar) as "Heating"
            , cast(HeatingQC as varchar) as "HeatingQC"
            , cast(CentralAir as varchar) as "CentralAir"
            , cast(Electrical as varchar) as "Electrical"
            , cast("1stFlrSF" as double) as "1stFlrSF" 
            , cast("2ndFlrSF" as double) as "2ndFlrSF"
            , cast(LowQualFinSF as double) as "LowQualFinSF"
            , cast(GrLivArea as double) as "GrLivArea"
            , cast(BsmtFullBath as int) as "BsmtFullBath"
            , cast(BsmtHalfBath as int) as "BsmtHalfBath"
            , cast(FullBath as int) as "FullBath"
            , cast(HalfBath as int) as "HalfBath"
            , cast(BedroomAbvGr as int) as "BedroomAbvGr"
            , cast(KitchenAbvGr as int) as "KitchenAbvGr"
            , cast(KitchenQual as varchar) as "KitchenQual"
            , cast(TotRmsAbvGrd as int) as "TotRmsAbvGrd"
            , cast(Functional as varchar) as "Functional"
            , cast(Fireplaces as int) as "Fireplaces"
            , cast(FireplaceQu as varchar) as "FireplaceQu"
            , cast(GarageType as varchar) as "GarageType"
            , cast(GarageYrBlt as int) as "GarageYrBlt"
            , cast(GarageFinish as varchar) as "GarageFinish"
            , cast(GarageCars as int) as "GarageCars"
            , cast(GarageArea as double) as "GarageArea"
            , cast(GarageQual as varchar) as "GarageQual"
            , cast(GarageCond as varchar) as "GarageCond"
            , cast(PavedDrive as varchar) as "PavedDrive"
            , cast(WoodDeckSF as double) as "WoodDeckSF"
            , cast(OpenPorchSF as double) as "OpenPorchSF"
            , cast(EnclosedPorch as double) as "EnclosedPorch"
            , cast("3SsnPorch" as double) as "3SsnPorch"
            , cast(ScreenPorch as double) as "ScreenPorch"
            , cast(PoolArea as double) as "PoolArea"
            , cast(PoolQC as varchar) as "PoolQC"
            , cast(Fence as varchar) as "Fence"
            , cast(MiscFeature as varchar) as "MiscFeature"
            , cast(MiscVal as int) as "MiscVal"
            , cast(MoSold as int) as "MoSold"
            , cast(YrSold as int) as "YrSold"
            , cast(SaleType as varchar) as "SaleType"
            , cast(SaleCondition as varchar) as "SaleCondition"
            {target_col}
        from read_csv_auto(
            '{path}',
            nullstr={nullstr_sql}
        );        
                """)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()

print(f"Saved DuckDB database to {database_path}")
