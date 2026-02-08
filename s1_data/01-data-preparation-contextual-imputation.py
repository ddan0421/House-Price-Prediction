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
train = conn.execute("""select * from train;""").fetch_df()
test = conn.execute("""select * from test;""").fetch_df()


train_features = train.drop("SalePrice", axis=1)
feature_df = pd.concat([train_features, test], axis=0, ignore_index=False)


def missing_col(df):
    missing_col = df.isna().sum()
    missing_col_df = pd.DataFrame(missing_col[missing_col > 0])
#     print(missing_col_df.index.tolist())
    return missing_col_df

missing_check = missing_col(feature_df)


####################### Contextual Missing Data Handling  #######################
# Alley missing
"""
Alley: Type of alley access to property
       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
"""
feature_df["Alley"].fillna("no_alley", inplace=True) 


# Exterior1st
"""
Exterior1st: Exterior covering on house
       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
"""
feature_df["Exterior1st"].fillna("Other", inplace=True) 

# Exterior2nd
feature_df["Exterior2nd"].fillna("Other", inplace=True) 

# MasVnrType Missing
"""
MasVnrType: Masonry veneer type
       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone

There is NA type. Fill NA with 'Unknown'. 
Missing MasVnrType values were treated as an explicit “Unknown” category to 
distinguish unobserved veneer information from the valid “None” category indicating known absence of veneer.
"""
feature_df["MasVnrType"].fillna("no_MasVnrType", inplace=True)


# MasVnrArea Missing
"""
MasVnrArea: Masonry veneer area in square feet
- may need to fill in NA with 0 so that
the model knows:
- Area = 0 and “this was actually unknown”

That combination is distinct from:
- Area = 0 and Type = None

- Also need to set MasVnrArea to 0 if MasVnrType = None
"""
# feature_df["MasVnrArea"].fillna(0, inplace=True)


# BsmtQual Missing
"""
BsmtQual: Evaluates the height of the basement
       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
"""
feature_df["BsmtQual"].fillna("no_basement", inplace=True) 

# BsmtCond Missing
"""
BsmtCond: Evaluates the general condition of the basement
       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
"""
feature_df["BsmtCond"].fillna("no_basement", inplace=True) 



# BsmtExposure Missing
"""
BsmtExposure: Refers to walkout or garden level walls
       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
"""
feature_df["BsmtExposure"].fillna("no_basement", inplace=True)


# BsmtFinType1 Missing
"""
BsmtFinType1: Rating of basement finished area
       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
"""
feature_df["BsmtFinType1"].fillna("no_basement", inplace=True)

# BsmtFinSF1 Missing
"""
Type 1 finished square feet
- If BsmtFinType1 is no_basement, then BsmtFinSF1 should be 0
"""
feature_df["BsmtFinSF1"].fillna(0, inplace=True)


# BsmtFinType2 Missing
"""
BsmtFinType2: Rating of basement finished area (if multiple types)
       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
"""
feature_df["BsmtFinType2"].fillna("no_basement", inplace=True)

# BsmtFinSF2 Missing
"""
if BsmtFinType2 is no_basement, then BsmtFinSF2 should be 0
"""
feature_df["BsmtFinSF2"].fillna(0, inplace=True)



# BsmtUnfSF
"""
BsmtUnfSF = TotalBsmtSF - BsmtFinSF1 - BsmtFinSF2
If BsmtFinType1 and BsmtFinType2 are no_basement, and BsmtUnfSF is null, then BsmtUnfSF should be 0
"""
feature_df.loc[(feature_df["BsmtFinType1"] == "no_basement") & (feature_df["BsmtFinType2"] == "no_basement") & (feature_df["BsmtUnfSF"].isna()), "BsmtUnfSF"] = 0


# TotalBsmtSF
"""
TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF
If BsmtFinType1 and BsmtFinType2 are no_basement, and TotalBsmtSF is null, then TotalBsmtSF should be 0
"""
feature_df.loc[(feature_df["BsmtFinType1"] == "no_basement") & (feature_df["BsmtFinType2"] == "no_basement") & (feature_df["TotalBsmtSF"].isna()), "TotalBsmtSF"] = 0


# BsmtFullBath
"""
If BsmtFinType1 and BsmtFinType2 are no_basement, and BsmtFullBath is null, then BsmtFullBath should be 0

"""
feature_df.loc[(feature_df["BsmtFinType1"] == "no_basement") & (feature_df["BsmtFinType2"] == "no_basement") & (feature_df["BsmtFullBath"].isna()), "BsmtFullBath"] = 0


# BsmtHalfBath
"""
If BsmtFinType1 and BsmtFinType2 are no_basement, and BsmtHalfBath is null, then BsmtHalfBath should be 0

"""
feature_df.loc[(feature_df["BsmtFinType1"] == "no_basement") & (feature_df["BsmtFinType2"] == "no_basement") & (feature_df["BsmtHalfBath"].isna()), "BsmtHalfBath"] = 0




# FireplaceQu Missing
"""
FireplaceQu: Fireplace quality
       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
"""
feature_df["FireplaceQu"].fillna("no_fireplace", inplace=True)





# GarageType Missing
"""
GarageType: Garage location
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
"""
feature_df["GarageType"].fillna("no_garage", inplace=True)


# GarageYrBlt Missing
"""
if GarageType is no_garage, then GarageYrBlt should be 0
I have observed that GarageYrBlt is missing in some cases where GarageType is either 'no_garage' or 'Detchd' (detached garage). Similarly, for GarageQual and GarageCond, the NA values occur for both 'no_garage' and 'Detchd'.
Therefore, we need to interpret all missing values in GarageYrBlt, GarageQual, and GarageCond as 'no_garage'. For instances where GarageType is 'Detchd' and GarageYrBlt, GarageQual, and GarageCond are missing, we should fill these values with 0 and treat them as 'no_garage'.
"""
feature_df["GarageYrBlt"].fillna(0, inplace=True)



# GarageFinish Missing
"""
GarageFinish: Interior finish of the garage
       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
"""
feature_df["GarageFinish"].fillna("no_garage", inplace=True)


# GarageQual Missing
"""
GarageQual: Garage quality
       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
"""
feature_df["GarageQual"].fillna("no_garage", inplace=True)



# GarageCond Missing
"""
GarageCond: Garage condition
       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
"""
feature_df["GarageCond"].fillna("no_garage", inplace=True)



#  GarageCars Missing and GarageArea Missing by DEALING the Detached Garage Anomalies
feature_df.loc[(feature_df["GarageType"] == "Detchd") & (feature_df["GarageYrBlt"] == 0) & (feature_df["GarageFinish"]=="no_garage") & (feature_df["GarageQual"]=="no_garage") & (feature_df["GarageCond"]=="no_garage"), "GarageType"] = "no_garage"
feature_df.loc[(feature_df["GarageType"] == "Detchd") & (feature_df["GarageYrBlt"] == 0) & (feature_df["GarageFinish"]=="no_garage") & (feature_df["GarageQual"]=="no_garage") & (feature_df["GarageCond"]=="no_garage"), "GarageCars"] = 0
feature_df.loc[(feature_df["GarageType"] == "Detchd") & (feature_df["GarageYrBlt"] == 0) & (feature_df["GarageFinish"]=="no_garage") & (feature_df["GarageQual"]=="no_garage") & (feature_df["GarageCond"]=="no_garage"), "GarageArea"] = 0

feature_df["GarageCars"].fillna(0, inplace=True)
feature_df["GarageArea"].fillna(0, inplace=True)


# PoolQC Missing
"""
PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
"""
feature_df["PoolQC"].fillna("no_pool", inplace=True)

# Fence Missing
"""
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
"""
feature_df["Fence"].fillna("no_fence", inplace=True)



# MiscFeature Missing
"""
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
"""
feature_df["MiscFeature"].fillna("no_MiscFeature", inplace=True)

# SaleType Missing (fill the missing with other)
"""
SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
"""
feature_df["SaleType"].fillna("Oth", inplace=True)


train_clean = pd.concat([feature_df.iloc[:1460, :], train["SalePrice"]], axis=1)
test_clean = feature_df.iloc[1460:, :]

train_clean.to_csv("data/train_clean_01.csv", index=False)
test_clean.to_csv("data/test_clean_01.csv", index=False)


