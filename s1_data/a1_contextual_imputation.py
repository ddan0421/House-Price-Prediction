import pandas as pd
import warnings
import duckdb
warnings.filterwarnings("ignore", category=FutureWarning)

from data.gdrive_download import download_from_drive
file_id = "1wUrGG6OijEhwA4YvZx8VzsyqkJFZPv-G"
filename = "AmesHousePrice.duckdb"
folder = "data"
download_from_drive(file_id, filename, folder)


conn = duckdb.connect("data/AmesHousePrice.duckdb")


for source in ["train", "test"]:
    target_col = ', SalePrice' if source == "train" else ""
    query = f"""
        create or replace table {source}_cleaned as
            with cte as (
                select
                    *
                    , ifnull(Alley, 'no_alley') AS Alley
                    , ifnull(Exterior1st, 'Other') AS Exterior1st
                    , ifnull(Exterior2nd, 'Other') AS Exterior2nd
                    , ifnull(MasVnrType, 'None') AS MasVnrType
                    , ifnull(MasVnrArea, 0) AS MasVnrArea
                    , ifnull(BsmtQual, 'no_basement') AS BsmtQual
                    , ifnull(BsmtCond, 'no_basement') AS BsmtCond
                    , ifnull(BsmtExposure, 'no_basement') AS BsmtExposure
                    , ifnull(BsmtFinType1, 'no_basement') AS BsmtFinType1
                    , ifnull(BsmtFinSF1, 0) AS BsmtFinSF1
                    , ifnull(BsmtFinType2, 'no_basement') AS BsmtFinType2
                    , ifnull(BsmtFinSF2, 0) AS BsmtFinSF2
                    , ifnull(FireplaceQu, 'no_fireplace') AS FireplaceQu
                    , ifnull(GarageType, 'no_garage') AS GarageType
                    , ifnull(GarageYrBlt, 0) AS GarageYrBlt
                    , ifnull(GarageFinish, 'no_garage') AS GarageFinish
                    , ifnull(GarageQual, 'no_garage') AS GarageQual
                    , ifnull(GarageCond, 'no_garage') AS GarageCond
                    , ifnull(PoolQC, 'no_pool') AS PoolQC
                    , ifnull(Fence, 'no_fence') AS Fence
                    , ifnull(MiscFeature, 'no_MiscFeature') AS MiscFeature
                    , ifnull(SaleType, 'Oth') AS SaleType
                from {source}
            )
            select
                Id
                , MSSubClass
                , MSZoning
                , LotFrontage
                , LotArea
                , Street
                , Alley
                , LotShape
                , LandContour
                , Utilities
                , LotConfig
                , LandSlope
                , Neighborhood
                , Condition1
                , Condition2
                , BldgType
                , HouseStyle
                , OverallQual
                , OverallCond
                , YearBuilt
                , YearRemodAdd
                , RoofStyle
                , RoofMatl
                , Exterior1st
                , Exterior2nd
                , MasVnrType
                , case 
                    when MasVnrType = 'None' then 0
                    else MasVnrArea            
                  end as MasVnrArea
                , ExterQual
                , ExterCond
                , Foundation
                , BsmtQual
                , BsmtCond
                , BsmtExposure
                , BsmtFinType1
                , case 
                    when BsmtFinType1 = 'no_basement' then 0
                    when BsmtFinType1 = 'Unf' then 0
                    else BsmtFinSF1
                  end as BsmtFinSF1
                , BsmtFinType2
                , case 
                    when BsmtFinType2 = 'no_basement' then 0
                    when BsmtFinType2 = 'Unf' then 0
                    else BsmtFinSF2
                  end as BsmtFinSF2
                , case 
                    when BsmtFinType1 = 'no_basement' 
                     and BsmtFinType2 = 'no_basement' then 0
                    else BsmtUnfSF
                  end as BsmtUnfSF  
                , case 
                    when BsmtFinType1 = 'no_basement' 
                     and BsmtFinType2 = 'no_basement' then 0
                    else TotalBsmtSF
                  end as TotalBsmtSF                           
                , Heating
                , HeatingQC
                , CentralAir
                , Electrical
                , "1stFlrSF"
                , "2ndFlrSF"
                , LowQualFinSF
                , GrLivArea
                , case 
                    when BsmtFinType1 = 'no_basement' 
                     and BsmtFinType2 = 'no_basement' then 0
                    else BsmtFullBath
                  end as BsmtFullBath 
                , case 
                    when BsmtFinType1 = 'no_basement' 
                     and BsmtFinType2 = 'no_basement' then 0
                    else BsmtHalfBath
                  end as BsmtHalfBath                 
                , FullBath
                , HalfBath
                , BedroomAbvGr
                , KitchenAbvGr
                , KitchenQual
                , TotRmsAbvGrd
                , Functional
                , Fireplaces
                , FireplaceQu
                , case 
                    when GarageType = 'Detchd'
                     and GarageYrBlt = 0
                     and GarageFinish = 'no_garage'
                     and GarageQual = 'no_garage'
                     and GarageCond = 'no_garage'
                    then 'no_garage'
                    else GarageType
                  end as GarageType
                , GarageYrBlt
                , GarageFinish
                , case 
                    when GarageType = 'Detchd'
                     and GarageYrBlt = 0
                     and GarageFinish = 'no_garage'
                     and GarageQual = 'no_garage'
                     and GarageCond = 'no_garage'
                    then 0
                    else GarageCars
                  end as GarageCars
                , case 
                    when GarageType = 'Detchd'
                     and GarageYrBlt = 0
                     and GarageFinish = 'no_garage'
                     and GarageQual = 'no_garage'
                     and GarageCond = 'no_garage'
                    then 0
                    else GarageArea
                  end as GarageArea                                
                , GarageQual
                , GarageCond
                , PavedDrive
                , WoodDeckSF
                , OpenPorchSF
                , EnclosedPorch
                , "3SsnPorch"
                , ScreenPorch
                , case
                    when PoolQC = 'no_pool' then 0
                    else PoolArea
                  end as PoolArea
                , PoolQC
                , Fence
                , MiscFeature
                , MiscVal
                , MoSold
                , YrSold
                , SaleType
                , SaleCondition
                {target_col}
            from cte;

    """
    conn.execute(query)
   
print(conn.execute("SHOW TABLES").fetchall())

train = conn.execute("""select * from train_cleaned;""").fetch_df()
test = conn.execute("""select * from test_cleaned;""").fetch_df()

train.to_csv("data/train_clean_01.csv", index=False)
test.to_csv("data/test_clean_01.csv", index=False)

conn.close()
