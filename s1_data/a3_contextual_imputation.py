import pandas as pd
import duckdb
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)


for source in ["train", "test"]:
    target_col = ', SalePrice' if source == "train" else ""
    query = f"""
        create or replace table {source}_contextual_imputed as
            with cte as (
                select
                    *
                    , ifnull(Alley, 'no_alley') AS _Alley
                    , ifnull(Exterior1st, 'Other') AS _Exterior1st
                    , ifnull(Exterior2nd, 'Other') AS _Exterior2nd
                    , ifnull(MasVnrType, 'None') AS _MasVnrType
                    , ifnull(MasVnrArea, 0) AS _MasVnrArea
                    , ifnull(BsmtQual, 'no_basement') AS _BsmtQual
                    , ifnull(BsmtCond, 'no_basement') AS _BsmtCond
                    , ifnull(BsmtExposure, 'no_basement') AS _BsmtExposure
                    , ifnull(BsmtFinType1, 'no_basement') AS _BsmtFinType1
                    , ifnull(BsmtFinSF1, 0) AS _BsmtFinSF1
                    , ifnull(BsmtFinType2, 'no_basement') AS _BsmtFinType2
                    , ifnull(BsmtFinSF2, 0) AS _BsmtFinSF2
                    , ifnull(FireplaceQu, 'no_fireplace') AS _FireplaceQu
                    , ifnull(GarageType, 'no_garage') AS _GarageType
                    , ifnull(GarageYrBlt, 0) AS _GarageYrBlt
                    , ifnull(GarageFinish, 'no_garage') AS _GarageFinish
                    , ifnull(GarageQual, 'no_garage') AS _GarageQual
                    , ifnull(GarageCond, 'no_garage') AS _GarageCond
                    , ifnull(PoolQC, 'no_pool') AS _PoolQC
                    , ifnull(Fence, 'no_fence') AS _Fence
                    , ifnull(MiscFeature, 'no_MiscFeature') AS _MiscFeature
                    , ifnull(SaleType, 'Oth') AS _SaleType
                from {source}
            )
            select
                Id
                , MSSubClass
                , MSZoning
                , LotFrontage
                , LotArea
                , Street
                , _Alley as Alley
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
                , _Exterior1st as Exterior1st
                , _Exterior2nd as Exterior2nd
                , case 
                    when _MasVnrType = 'None' then 'NoMasVnr'
                    else _MasVnrType
                  end as MasVnrType
                , case 
                    when _MasVnrType = 'None' then 0
                    else _MasVnrArea            
                  end as MasVnrArea
                , ExterQual
                , ExterCond
                , Foundation
                , _BsmtQual as BsmtQual
                , _BsmtCond as BsmtCond
                , _BsmtExposure as BsmtExposure
                , _BsmtFinType1 as BsmtFinType1
                , case 
                    when _BsmtFinType1 = 'no_basement' then 0
                    when _BsmtFinType1 = 'Unf' then 0
                    else _BsmtFinSF1
                  end as BsmtFinSF1
                , _BsmtFinType2 as BsmtFinType2
                , case 
                    when _BsmtFinType2 = 'no_basement' then 0
                    when _BsmtFinType2 = 'Unf' then 0
                    else _BsmtFinSF2
                  end as BsmtFinSF2
                , case 
                    when _BsmtFinType1 = 'no_basement' 
                     and _BsmtFinType2 = 'no_basement' then 0
                    else BsmtUnfSF
                  end as BsmtUnfSF  
                , case 
                    when _BsmtFinType1 = 'no_basement' 
                     and _BsmtFinType2 = 'no_basement' then 0
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
                    when _BsmtFinType1 = 'no_basement' 
                     and _BsmtFinType2 = 'no_basement' then 0
                    else BsmtFullBath
                  end as BsmtFullBath 
                , case 
                    when _BsmtFinType1 = 'no_basement' 
                     and _BsmtFinType2 = 'no_basement' then 0
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
                , _FireplaceQu as FireplaceQu
                , case 
                    when _GarageType = 'Detchd'
                     and _GarageYrBlt = 0
                     and _GarageFinish = 'no_garage'
                     and _GarageQual = 'no_garage'
                     and _GarageCond = 'no_garage'
                    then 'no_garage'
                    else _GarageType
                  end as GarageType
                , _GarageYrBlt as GarageYrBlt
                , _GarageFinish as GarageFinish
                , case 
                    when _GarageType = 'Detchd'
                     and _GarageYrBlt = 0
                     and _GarageFinish = 'no_garage'
                     and _GarageQual = 'no_garage'
                     and _GarageCond = 'no_garage'
                    then 0
                    else GarageCars
                  end as GarageCars
                , case 
                    when _GarageType = 'Detchd'
                     and _GarageYrBlt = 0
                     and _GarageFinish = 'no_garage'
                     and _GarageQual = 'no_garage'
                     and _GarageCond = 'no_garage'
                    then 0
                    else GarageArea
                  end as GarageArea                                
                , _GarageQual as GarageQual
                , _GarageCond as GarageCond
                , PavedDrive
                , WoodDeckSF
                , OpenPorchSF
                , EnclosedPorch
                , "3SsnPorch"
                , ScreenPorch
                , case
                    when _PoolQC = 'no_pool' then 0
                    else PoolArea
                  end as PoolArea
                , _PoolQC as PoolQC
                , _Fence as Fence
                , _MiscFeature as MiscFeature
                , MiscVal
                , MoSold
                , YrSold
                , _SaleType as SaleType
                , SaleCondition
                {target_col}
            from cte;

    """
    conn.execute(query)
   
print(conn.execute("SHOW TABLES").fetchall())

conn.close()
