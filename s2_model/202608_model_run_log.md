# s2_model_run_log

Terminal color codes were removed so this renders cleanly on GitHub (GFM does not apply custom ANSI/HTML colors in repo views).

```text
============================================================
Running s2_model.a1_regression
============================================================


Model final parameters:
const                                      10.934712
Age_House                                  -0.069244
Area_vs_Nbhd                               -0.003505
BedroomAbvGr                               -0.030798
BsmtExposure_encoded                        0.014825
BsmtFullBath                                0.016809
CentralAir_Electrical_N_SBrkr              -0.053028
EnclosedPorch                               0.005535
ExterQual_encoded                           0.016797
Exterior1st_Exterior2nd_BrkFace             0.082567
Exterior1st_Exterior2nd_BrkFace_Wd Sdng     0.153055
FinishedAreaPct                            -0.020907
Fireplaces                                  0.013816
Foundation_PConc                            0.037960
FullBath                                    0.009094
Functional_encoded                          0.039258
GarageArea                                  0.015687
GarageCars                                  0.012246
Garage_AgeCars                             -0.003053
Garage_Space                                0.010959
HalfBath                                    0.012502
KitchenAbvGr                               -0.019165
KitchenQual_encoded                         0.019848
Living_Rooms                               -0.099386
Neighborhood_Condition_BrkSide_Norm         0.074584
Neighborhood_Condition_Crawfor_Norm         0.122237
Neighborhood_Condition_Edwards_PosN        -1.302573
Neighborhood_Condition_NoRidge_Norm         0.072870
Neighborhood_Condition_NridgHt_Norm         0.076684
Neighborhood_Condition_Somerst_Norm         0.050362
Neighborhood_Condition_StoneBr_Norm         0.131832
OverallCond                                 0.044165
OverallQual                                 0.050781
Porch_Age                                  -0.002371
Ratio_2ndFlr_Living                        -0.001540
Ratio_Bedroom_Rooms                         0.019696
RoofStyle_RoofMatl_Hip_ClyTile             -1.784498
SaleCondition_Normal                        0.048911
SaleType_New                                0.091256
ScreenPorch                                 0.010634
TotRmsAbvGrd                                0.076577
cbrt_MasVnrArea                            -0.003770
cbrt_OpenPorchSF                            0.002427
log_1stFlrSF                                0.002053
log_2ndFlrSF                               -0.026065
log_GrLivArea                               0.229642
log_LotArea                                 0.047459
log_Yrs_Since_Remodel                      -0.012713
sqrt_BsmtFinSF1                             0.029845
sqrt_TotalBsmtSF                            0.001959
sqrt_WoodDeckSF                             0.010280
dtype: float64

Model fitting summary:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.929
Model:                            OLS   Adj. R-squared:                  0.926
Method:                 Least Squares   F-statistic:                     292.6
Date:                Mon, 10 Aug 2026   Prob (F-statistic):               0.00
Time:                        02:35:06   Log-Likelihood:                 986.41
No. Observations:                1168   AIC:                            -1871.
Df Residuals:                    1117   BIC:                            -1613.
Df Model:                          50                                         
Covariance Type:            nonrobust                                         
===========================================================================================================
                                              coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------
const                                      10.9347      0.053    204.498      0.000      10.830      11.040
Age_House                                  -0.0692      0.008     -9.177      0.000      -0.084      -0.054
Area_vs_Nbhd                               -0.0035      0.008     -0.432      0.666      -0.019       0.012
BedroomAbvGr                               -0.0308      0.016     -1.894      0.058      -0.063       0.001
BsmtExposure_encoded                        0.0148      0.004      4.147      0.000       0.008       0.022
BsmtFullBath                                0.0168      0.004      3.849      0.000       0.008       0.025
CentralAir_Electrical_N_SBrkr              -0.0530      0.018     -2.977      0.003      -0.088      -0.018
EnclosedPorch                               0.0055      0.007      0.774      0.439      -0.008       0.020
ExterQual_encoded                           0.0168      0.010      1.768      0.077      -0.002       0.035
Exterior1st_Exterior2nd_BrkFace             0.0826      0.029      2.828      0.005       0.025       0.140
Exterior1st_Exterior2nd_BrkFace_Wd Sdng     0.1531      0.033      4.578      0.000       0.087       0.219
FinishedAreaPct                            -0.0209      0.020     -1.054      0.292      -0.060       0.018
Fireplaces                                  0.0138      0.004      3.408      0.001       0.006       0.022
Foundation_PConc                            0.0380      0.010      3.708      0.000       0.018       0.058
FullBath                                    0.0091      0.005      1.680      0.093      -0.002       0.020
Functional_encoded                          0.0393      0.005      7.646      0.000       0.029       0.049
GarageArea                                  0.0157      0.008      1.902      0.057      -0.000       0.032
GarageCars                                  0.0122      0.009      1.404      0.161      -0.005       0.029
Garage_AgeCars                             -0.0031      0.009     -0.352      0.725      -0.020       0.014
Garage_Space                                0.0110      0.010      1.084      0.278      -0.009       0.031
HalfBath                                    0.0125      0.005      2.655      0.008       0.003       0.022
KitchenAbvGr                               -0.0192      0.004     -4.976      0.000      -0.027      -0.012
KitchenQual_encoded                         0.0198      0.008      2.475      0.013       0.004       0.036
Living_Rooms                               -0.0994      0.042     -2.372      0.018      -0.182      -0.017
Neighborhood_Condition_BrkSide_Norm         0.0746      0.020      3.719      0.000       0.035       0.114
Neighborhood_Condition_Crawfor_Norm         0.1222      0.019      6.399      0.000       0.085       0.160
Neighborhood_Condition_Edwards_PosN        -1.3026      0.120    -10.817      0.000      -1.539      -1.066
Neighborhood_Condition_NoRidge_Norm         0.0729      0.024      3.083      0.002       0.026       0.119
Neighborhood_Condition_NridgHt_Norm         0.0767      0.018      4.316      0.000       0.042       0.112
Neighborhood_Condition_Somerst_Norm         0.0504      0.016      3.059      0.002       0.018       0.083
Neighborhood_Condition_StoneBr_Norm         0.1318      0.026      5.026      0.000       0.080       0.183
OverallCond                                 0.0442      0.004     11.839      0.000       0.037       0.051
OverallQual                                 0.0508      0.005     11.139      0.000       0.042       0.060
Porch_Age                                  -0.0024      0.008     -0.313      0.755      -0.017       0.013
Ratio_2ndFlr_Living                        -0.0015      0.044     -0.035      0.972      -0.087       0.084
Ratio_Bedroom_Rooms                         0.0197      0.012      1.639      0.101      -0.004       0.043
RoofStyle_RoofMatl_Hip_ClyTile             -1.7845      0.136    -13.119      0.000      -2.051      -1.518
SaleCondition_Normal                        0.0489      0.011      4.362      0.000       0.027       0.071
SaleType_New                                0.0913      0.020      4.607      0.000       0.052       0.130
ScreenPorch                                 0.0106      0.003      3.197      0.001       0.004       0.017
TotRmsAbvGrd                                0.0766      0.022      3.424      0.001       0.033       0.120
cbrt_MasVnrArea                            -0.0038      0.004     -0.954      0.340      -0.012       0.004
cbrt_OpenPorchSF                            0.0024      0.004      0.640      0.522      -0.005       0.010
log_1stFlrSF                                0.0021      0.032      0.065      0.948      -0.060       0.064
log_2ndFlrSF                               -0.0261      0.021     -1.259      0.208      -0.067       0.015
log_GrLivArea                               0.2296      0.049      4.688      0.000       0.134       0.326
log_LotArea                                 0.0475      0.004     11.594      0.000       0.039       0.055
log_Yrs_Since_Remodel                      -0.0127      0.006     -1.994      0.046      -0.025      -0.000
sqrt_BsmtFinSF1                             0.0298      0.005      6.377      0.000       0.021       0.039
sqrt_TotalBsmtSF                            0.0020      0.020      0.098      0.922      -0.037       0.041
sqrt_WoodDeckSF                             0.0103      0.004      2.876      0.004       0.003       0.017
==============================================================================
Omnibus:                      339.375   Durbin-Watson:                   2.029
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2737.408
Skew:                          -1.109   Prob(JB):                         0.00
Kurtosis:                      10.165   Cond. No.                         595.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Model summary saved to models/ols_model_summary.txt

10-Fold CV RMSE (log-transformed scale): 0.12047609997528097
Optimal Parameter: {'alpha': 1.0}
Optimal Estimator: Ridge()
Ridge model saved to models/final_model_ridge.pkl
10-Fold CV RMSE: 0.12406826374177338
Optimal Parameter: {'alpha': 0.001}
Optimal Estimator: Lasso(alpha=0.001)
Selected features for Lasso:
Index(['Age_House', 'Area_vs_Nbhd', 'BsmtExposure_encoded', 'BsmtFullBath',
       'CentralAir_Electrical_N_SBrkr', 'EnclosedPorch', 'ExterQual_encoded',
       'Exterior1st_Exterior2nd_BrkFace',
       'Exterior1st_Exterior2nd_BrkFace_Wd Sdng', 'FinishedAreaPct',
       'Fireplaces', 'Foundation_PConc', 'FullBath', 'Functional_encoded',
       'GarageArea', 'GarageCars', 'Garage_AgeCars', 'Garage_Space',
       'HalfBath', 'KitchenAbvGr', 'KitchenQual_encoded',
       'Neighborhood_Condition_BrkSide_Norm',
       'Neighborhood_Condition_Crawfor_Norm',
       'Neighborhood_Condition_NridgHt_Norm',
       'Neighborhood_Condition_Somerst_Norm',
       'Neighborhood_Condition_StoneBr_Norm', 'OverallCond', 'OverallQual',
       'Porch_Age', 'SaleCondition_Normal', 'SaleType_New', 'ScreenPorch',
       'TotRmsAbvGrd', 'cbrt_MasVnrArea', 'cbrt_OpenPorchSF', 'log_1stFlrSF',
       'log_2ndFlrSF', 'log_GrLivArea', 'log_LotArea', 'log_Yrs_Since_Remodel',
       'sqrt_BsmtFinSF1', 'sqrt_WoodDeckSF'],
      dtype='object')
Lasso model saved to models/final_model_lasso.pkl
10-Fold CV RMSE: 0.12064331066708427
Optimal Parameter: {'alpha': 0.001, 'l1_ratio': 0.1}
Optimal Estimator: ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=42)
ElasticNet model saved to models/final_model_enet.pkl
OLS Model Performance:
Root Mean Squared Error: 0.1265
Ridge Model Performance:
Root Mean Squared Error: 0.1276
Lasso Model Performance:
Root Mean Squared Error: 0.1318
ElasticNet Model Performance:
Root Mean Squared Error: 0.1281

============================================================
Running s2_model.a2_svr
============================================================

RBF SVR gamma='scale' resolves to 0.00441
RBF SVR 10-Fold CV RMSE: 0.114743172597216
RBF SVR Optimal Parameter: {'C': 3, 'epsilon': 0.04, 'gamma': np.float64(0.0033096523125370194)}
RBF SVR Optimal Estimator: SVR(C=3, epsilon=0.04, gamma=np.float64(0.0033096523125370194), tol=1e-05)
RBF SVR model saved to models/final_model_svr_rbf.pkl
LinearSVR 10-Fold CV RMSE: 0.12193226892578832
LinearSVR Optimal Parameter: {'C': 50, 'dual': False, 'epsilon': 0.01, 'loss': 'squared_epsilon_insensitive', 'tol': 0.0001}
LinearSVR Optimal Estimator: LinearSVR(C=50, dual=False, epsilon=0.01, loss='squared_epsilon_insensitive',
          max_iter=50000, random_state=42)
LinearSVR model saved to models/final_model_linear_svr.pkl
RBF SVR Model Performance:
Root Mean Squared Error: 0.1303
LinearSVR Model Performance:
Root Mean Squared Error: 0.1263

============================================================
Running s2_model.a3_knn
============================================================

KNN 10-Fold CV RMSE: 0.1533552206858842
KNN Optimal Parameter: {'n_neighbors': 6, 'p': 1, 'weights': 'distance'}
KNN Optimal Estimator: KNeighborsRegressor(n_neighbors=6, p=1, weights='distance')
KNN model saved to models/final_model_knn.pkl
KNN Model Performance:
Root Mean Squared Error: 0.1653

============================================================
Running s2_model.a4_trees
============================================================

10-Fold CV RMSE: 0.18894078301372924
Optimal Parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.01}
Optimal Estimator: DecisionTreeRegressor(max_depth=10, min_weight_fraction_leaf=0.01,
                      random_state=42)
decision tree model saved to models/final_model_dt.pkl
Decision Tree Performance:
Root Mean Squared Error: 0.1905
10-Fold CV RMSE: 0.14793457211311073
Optimal Parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Optimal Estimator: RandomForestRegressor(max_depth=10, max_features='sqrt', n_estimators=200,
                      random_state=42)
random forest model saved to models/final_model_rf.pkl
Random Forst Tree Performance:
Root Mean Squared Error: 0.1675
10-Fold CV RMSE: 0.142158980890348
Optimal Parameters: {'bootstrap': True, 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Optimal Estimator: ExtraTreesRegressor(bootstrap=True, max_depth=10, max_features=None,
                    n_estimators=200, random_state=42)
ExtraTreesRegressor model saved to models/final_model_et.pkl
Extra Trees Regressor Performance:
Root Mean Squared Error: 0.1543

============================================================
Running s2_model.a5_xgb
============================================================

10-Fold CV RMSE: 0.12156935295732203
Optimal Parameters: {'tree_method': 'hist', 'subsample': 0.8, 'colsample_bytree': 0.75, 'learning_rate': 0.05, 'max_depth': 2, 'min_child_weight': 3, 'reg_alpha': 0, 'reg_lambda': 1}
Best boosting round from CV: 1382
Optimal Estimator: XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=0.75, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.05, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=2, max_leaves=None,
             min_child_weight=3, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=1382, n_jobs=-1,
             num_parallel_tree=None, random_state=42, ...)
xgboost model saved to models/final_model_xgb.pkl
|   iter    |  target   | colsam... |   gamma   | learni... | max_depth | min_ch... | reg_alpha | reg_la... | subsample |
-------------------------------------------------------------------------------------------------------------------------
| 1         | -0.2004   | 0.6247    | 4.754     | 0.2209    | 6.789     | 2.404     | 0.78      | 0.2904    | 0.9331    |
| 2         | -0.1964   | 0.7607    | 3.54      | 0.01107   | 9.759     | 8.492     | 1.062     | 0.9091    | 0.5917    |
| 3         | -0.1809   | 0.5825    | 2.624     | 0.1324    | 4.33      | 6.507     | 0.6975    | 1.461     | 0.6832    |
| 4         | -0.1968   | 0.6736    | 3.926     | 0.0639    | 6.114     | 6.332     | 0.2323    | 3.038     | 0.5853    |
| 5         | -0.2024   | 0.439     | 4.744     | 0.2899    | 8.467     | 3.742     | 0.4884    | 3.421     | 0.7201    |
| 6         | -0.1897   | 0.4732    | 2.476     | 0.01514   | 9.275     | 3.329     | 3.313     | 1.559     | 0.76      |
| 7         | -0.1758   | 0.728     | 0.9243    | 0.291     | 8.201     | 9.455     | 4.474     | 2.989     | 0.9609    |
| 8         | -0.1593   | 0.4531    | 0.9799    | 0.01834   | 4.603     | 4.498     | 1.357     | 4.144     | 0.6784    |
| 9         | -0.204    | 0.5686    | 2.713     | 0.04657   | 8.418     | 1.671     | 4.934     | 3.861     | 0.5994    |
| 10        | -0.1989   | 0.4033    | 4.077     | 0.2135    | 7.832     | 7.941     | 0.3702    | 1.792     | 0.5579    |
| 11        | -0.1289   | 0.4       | 0.0       | 0.03227   | 3.358     | 5.881     | 2.529     | 4.215     | 0.8744    |
| 12        | -0.1681   | 0.8116    | 0.4783    | 0.2844    | 2.046     | 7.771     | 4.912     | 4.472     | 0.8581    |
| 13        | -0.1337   | 0.5209    | 0.06785   | 0.06491   | 2.296     | 5.957     | 1.316     | 4.261     | 0.7963    |
| 14        | -0.15     | 0.4335    | 0.121     | 0.2781    | 2.032     | 4.373     | 3.353     | 4.72      | 0.6911    |
| 15        | -0.1358   | 0.5013    | 0.03942   | 0.1945    | 3.573     | 8.519     | 0.9449    | 4.435     | 0.9051    |
| 16        | -0.1274   | 0.5633    | 0.02423   | 0.01314   | 2.168     | 9.578     | 0.374     | 1.321     | 0.9044    |
| 17        | -0.1467   | 0.7648    | 0.1316    | 0.05818   | 2.583     | 8.494     | 3.375     | 0.1067    | 0.5643    |
| 18        | -0.136    | 0.9675    | 0.1402    | 0.2037    | 2.246     | 6.607     | 0.5612    | 0.3525    | 0.5946    |
| 19        | -0.1454   | 0.7519    | 0.05021   | 0.2732    | 2.989     | 7.011     | 1.949     | 2.238     | 0.7832    |
| 20        | -0.1286   | 0.4347    | 0.03817   | 0.02729   | 4.955     | 9.54      | 0.8043    | 0.08217   | 0.7374    |
| 21        | -0.1554   | 0.402     | 0.8544    | 0.2568    | 3.027     | 9.753     | 0.3298    | 0.04972   | 0.9453    |
| 22        | -0.1329   | 0.5102    | 0.1245    | 0.0503    | 2.2       | 1.192     | 0.727     | 0.5497    | 0.677     |
| 23        | -0.1681   | 0.9999    | 0.6144    | 0.1644    | 2.807     | 1.748     | 4.402     | 0.2692    | 0.7825    |
| 24        | -0.1508   | 0.7143    | 0.7245    | 0.1399    | 2.033     | 9.639     | 0.1132    | 4.475     | 0.5394    |
| 25        | -0.1285   | 0.7189    | 0.03293   | 0.1875    | 2.059     | 2.892     | 0.2141    | 2.213     | 0.8105    |
| 26        | -0.1361   | 0.7715    | 0.05541   | 0.1416    | 5.989     | 9.748     | 0.3532    | 2.146     | 0.845     |
| 27        | -0.1338   | 0.5496    | 0.1976    | 0.04899   | 2.284     | 3.003     | 0.08648   | 0.04105   | 0.7394    |
| 28        | -0.1761   | 0.7478    | 2.494     | 0.0688    | 2.145     | 1.074     | 0.05668   | 3.617     | 0.8922    |
| 29        | -0.131    | 0.6666    | 0.04462   | 0.1491    | 6.033     | 5.878     | 0.01231   | 0.4288    | 0.6796    |
| 30        | -0.1358   | 0.7567    | 0.03319   | 0.2247    | 9.966     | 5.194     | 0.06042   | 0.4862    | 0.6986    |
| 31        | -0.1316   | 0.8708    | 0.02627   | 0.08303   | 6.474     | 1.971     | 0.5477    | 0.1474    | 0.5611    |
| 32        | -0.1338   | 0.5525    | 0.1324    | 0.01964   | 9.926     | 1.745     | 0.2735    | 3.697     | 0.8355    |
| 33        | -0.145    | 0.7677    | 0.6534    | 0.06159   | 9.279     | 1.312     | 0.08182   | 0.4807    | 0.5457    |
| 34        | -0.1403   | 0.4714    | 0.2902    | 0.1663    | 9.419     | 4.765     | 0.1867    | 4.641     | 0.8608    |
| 35        | -0.1336   | 0.845     | 0.0688    | 0.187     | 7.403     | 1.245     | 0.03776   | 4.185     | 0.695     |
| 36        | -0.1366   | 0.8793    | 0.006709  | 0.2031    | 6.601     | 8.968     | 0.09829   | 0.3977    | 0.9358    |
| 37        | -0.1373   | 0.6868    | 0.07893   | 0.2471    | 7.598     | 2.907     | 0.09313   | 3.095     | 0.7329    |
| 38        | -0.1336   | 0.6216    | 0.04088   | 0.1494    | 3.689     | 2.026     | 0.07452   | 2.164     | 0.8188    |
| 39        | -0.1475   | 0.9       | 0.2729    | 0.2419    | 9.787     | 9.54      | 0.3939    | 4.986     | 0.9684    |
| 40        | -0.1304   | 0.7368    | 0.04282   | 0.101     | 4.821     | 6.777     | 0.2214    | 4.692     | 0.7467    |
| 41        | -0.1367   | 0.4225    | 0.01143   | 0.2184    | 6.466     | 6.825     | 1.606     | 4.004     | 0.8703    |
| 42        | -0.1446   | 0.4517    | 0.08394   | 0.1136    | 6.477     | 9.93      | 3.028     | 0.06004   | 0.876     |
| 43        | -0.1381   | 0.4825    | 0.03229   | 0.05371   | 7.324     | 4.597     | 2.154     | 0.3006    | 0.9257    |
| 44        | -0.1329   | 0.9536    | 0.01329   | 0.289     | 2.042     | 1.525     | 0.2071    | 4.39      | 0.6177    |
| 45        | -0.1308   | 0.4857    | 0.003158  | 0.2156    | 4.05      | 6.593     | 0.1975    | 3.033     | 0.786     |
| 46        | -0.1343   | 0.8514    | 0.06792   | 0.136     | 4.205     | 8.347     | 0.1651    | 1.402     | 0.8444    |
| 47        | -0.1258   | 0.453     | 0.03638   | 0.04762   | 4.845     | 6.413     | 0.1681    | 0.0444    | 0.6486    |
| 48        | -0.1291   | 0.5717    | 0.0811    | 0.08783   | 2.127     | 4.361     | 0.08783   | 4.578     | 0.718     |
| 49        | -0.1267   | 0.4       | 0.0       | 0.005     | 2.0       | 1.0       | 0.0       | 5.0       | 1.0       |
| 50        | -0.1414   | 0.8182    | 0.06709   | 0.0943    | 3.873     | 6.194     | 1.506     | 4.98      | 0.9682    |
=========================================================================================================================
Best Parameters: {'colsample_bytree': np.float64(0.4529629485112321), 'gamma': np.float64(0.03638453632112959), 'learning_rate': np.float64(0.047620018801133), 'max_depth': np.float64(4.845403106144805), 'min_child_weight': np.float64(6.412780642865168), 'reg_alpha': np.float64(0.16805950905792122), 'reg_lambda': np.float64(0.04440351889485605), 'subsample': np.float64(0.6485559222418549)}
Best RMSE Score: 0.1258483081399436
Best boosting round from CV: 503
XGB Bayes model saved to models/final_model_xgb_bayes.pkl
XGBoost (GridSearch) Performance:
Root Mean Squared Error: 0.1315
XGBoost (Bayes Opt) Performance:
Root Mean Squared Error: 0.1332

============================================================
Running s2_model.a6_lgbm
============================================================

10-Fold CV RMSE: 0.12888460330033538
Optimal Parameters: {'subsample': 0.85, 'subsample_freq': 1, 'colsample_bytree': 0.85, 'reg_alpha': 0.0, 'min_split_gain': 0.0, 'learning_rate': 0.05, 'num_leaves': 4, 'min_child_samples': 20, 'reg_lambda': 3.0}
Best boosting round from CV: 952
Optimal Estimator: LGBMRegressor(colsample_bytree=0.85, learning_rate=0.05, n_estimators=952,
              n_jobs=-1, num_leaves=4, objective='regression', random_state=42,
              reg_lambda=3.0, subsample=0.85, subsample_freq=1, verbose=-1)
lgbm model saved to models/final_model_lgbm.pkl
|   iter    |  target   | colsam... | learni... | min_ch... | num_le... | reg_alpha | reg_la... | subsample | subsam... |
-------------------------------------------------------------------------------------------------------------------------
| 1         | -0.1447   | 0.6247    | 0.2855    | 37.94     | 121.3     | 0.7801    | 0.78      | 0.529     | 6.197     |
| 2         | -0.1521   | 0.7607    | 0.2139    | 5.926     | 194.1     | 4.162     | 1.062     | 0.5909    | 2.1       |
| 3         | -0.1375   | 0.5825    | 0.1598    | 24.44     | 61.08     | 3.059     | 0.6975    | 0.6461    | 3.198     |
| 4         | -0.1446   | 0.6736    | 0.2366    | 13.99     | 104.8     | 2.962     | 0.2323    | 0.8038    | 2.023     |
| 5         | -0.1374   | 0.439     | 0.2849    | 48.45     | 162.4     | 1.523     | 0.4884    | 0.8421    | 3.641     |
| 6         | -0.1371   | 0.4732    | 0.1511    | 6.547     | 182.2     | 1.294     | 3.313     | 0.6559    | 4.12      |
| 7         | -0.1434   | 0.728     | 0.05953   | 48.63     | 155.9     | 4.697     | 4.474     | 0.7989    | 6.531     |
| 8         | -0.1316   | 0.4531    | 0.06281   | 7.035     | 67.76     | 1.943     | 1.357     | 0.9144    | 3.141     |
| 9         | -0.1384   | 0.5686    | 0.1651    | 11.34     | 161.2     | 0.3728    | 4.934     | 0.8861    | 2.192     |
| 10        | -0.1441   | 0.4033    | 0.2456    | 36.81     | 146.9     | 3.856     | 0.3702    | 0.6792    | 1.695     |
| 11        | -0.1359   | 0.9179    | 0.1889    | 19.89     | 16.46     | 1.555     | 1.626     | 0.8648    | 4.825     |
| 12        | -0.1447   | 0.9323    | 0.1443    | 10.38     | 143.8     | 3.804     | 2.806     | 0.8855    | 3.963     |
| 13        | -0.1353   | 0.7136    | 0.1311    | 6.144     | 25.15     | 0.1571    | 3.182     | 0.6572    | 4.051     |
| 14        | -0.1331   | 0.9445    | 0.07854   | 23.47     | 152.1     | 1.144     | 0.3849    | 0.6449    | 1.967     |
| 15        | -0.1448   | 0.9578    | 0.2434    | 33.5      | 174.8     | 4.018     | 0.9329    | 0.9463    | 4.236     |
| 16        | -0.1393   | 0.8845    | 0.2693    | 19.31     | 25.57     | 1.14      | 2.136     | 0.909     | 6.164     |
| 17        | -0.1339   | 0.4042    | 0.1557    | 23.78     | 47.53     | 0.5993    | 1.688     | 0.9715    | 2.939     |
| 18        | -0.1474   | 0.7113    | 0.2124    | 21.36     | 194.5     | 4.812     | 1.259     | 0.7486    | 2.805     |
| 19        | -0.1293   | 0.5709    | 0.01588   | 32.43     | 102.5     | 0.2574    | 1.393     | 0.9541    | 2.437     |
| 20        | -0.1416   | 0.4869    | 0.1494    | 49.35     | 51.44     | 3.361     | 3.808     | 0.6188    | 5.369     |
| 21        | -0.1416   | 0.4869    | 0.1494    | 49.35     | 51.44     | 3.361     | 3.808     | 0.6188    | 5.369     |
| 22        | -0.1403   | 0.8562    | 0.2949    | 36.68     | 155.6     | 1.581     | 3.536     | 0.8125    | 1.855     |
| 23        | -0.1328   | 0.4578    | 0.1338    | 21.72     | 45.35     | 1.624     | 3.727     | 0.8431    | 5.371     |
| 24        | -0.1319   | 0.9141    | 0.04444   | 32.63     | 100.5     | 1.089     | 0.2431    | 0.602     | 1.776     |
| 25        | -0.1309   | 0.8583    | 0.05075   | 31.27     | 103.9     | 0.2929    | 4.971     | 0.7469    | 2.233     |
| 26        | -0.1439   | 0.6605    | 0.2882    | 34.26     | 103.8     | 1.066     | 3.446     | 0.545     | 5.123     |
| 27        | -0.132    | 0.4553    | 0.06916   | 31.53     | 101.5     | 0.4367    | 2.585     | 0.8142    | 1.993     |
| 28        | -0.1305   | 0.4548    | 0.04325   | 31.72     | 104.4     | 1.149     | 2.326     | 0.8399    | 2.094     |
| 29        | -0.1307   | 0.565     | 0.02285   | 32.15     | 103.7     | 0.6611    | 0.4007    | 0.9355    | 1.844     |
| 30        | -0.1354   | 0.8769    | 0.141     | 29.76     | 101.9     | 0.1591    | 0.7883    | 0.5651    | 1.743     |
| 31        | -0.1353   | 0.7698    | 0.189     | 33.74     | 103.9     | 1.422     | 2.83      | 0.6995    | 1.498     |
| 32        | -0.131    | 0.7473    | 0.02991   | 30.75     | 105.3     | 0.4749    | 2.995     | 0.6       | 1.823     |
| 33        | -0.1328   | 0.5636    | 0.07942   | 30.16     | 104.4     | 1.768     | 4.036     | 0.7674    | 1.248     |
| 34        | -0.1324   | 0.4795    | 0.01578   | 8.195     | 67.43     | 0.3525    | 1.224     | 0.9201    | 5.002     |
| 35        | -0.1389   | 0.7603    | 0.07794   | 6.836     | 65.63     | 2.757     | 1.69      | 0.8105    | 4.624     |
| 36        | -0.1408   | 0.9506    | 0.1739    | 6.566     | 69.82     | 0.718     | 1.564     | 0.9673    | 1.63      |
| 37        | -0.1369   | 0.7809    | 0.2992    | 32.44     | 102.0     | 0.8626    | 0.4795    | 0.9383    | 1.721     |
| 38        | -0.1379   | 0.9       | 0.2494    | 31.54     | 103.6     | 0.775     | 1.255     | 0.5947    | 2.101     |
| 39        | -0.1283   | 0.5325    | 0.01571   | 31.61     | 104.9     | 0.03146   | 3.502     | 0.6419    | 1.737     |
| 40        | -0.142    | 0.8456    | 0.2172    | 32.14     | 105.0     | 0.7712    | 3.544     | 0.5424    | 2.816     |
| 41        | -0.1309   | 0.8893    | 0.006581  | 30.66     | 105.0     | 1.04      | 2.023     | 0.726     | 1.675     |
| 42        | -0.1379   | 0.4757    | 0.289     | 30.69     | 104.6     | 1.7       | 4.106     | 0.7133    | 1.582     |
| 43        | -0.1409   | 0.9134    | 0.2662    | 30.43     | 105.2     | 0.2231    | 3.311     | 0.6326    | 2.023     |
| 44        | -0.1451   | 0.5887    | 0.1331    | 13.07     | 169.9     | 4.676     | 1.713     | 0.8347    | 4.747     |
| 45        | -0.1388   | 0.889     | 0.2072    | 28.48     | 42.04     | 0.0754    | 0.5285    | 0.9309    | 5.46      |
| 46        | -0.1371   | 0.9688    | 0.09859   | 20.9      | 165.3     | 2.008     | 1.407     | 0.6675    | 3.44      |
| 47        | -0.1486   | 0.5948    | 0.1786    | 14.54     | 127.8     | 4.902     | 0.07267   | 0.8262    | 6.596     |
| 48        | -0.1476   | 0.8131    | 0.2708    | 10.46     | 184.8     | 3.634     | 2.943     | 0.8677    | 6.949     |
| 49        | -0.1361   | 0.6541    | 0.1468    | 32.06     | 102.1     | 0.2702    | 3.517     | 0.5118    | 2.457     |
| 50        | -0.1426   | 0.4568    | 0.2415    | 7.103     | 67.16     | 3.092     | 1.702     | 0.9863    | 3.276     |
| 51        | -0.1359   | 0.7889    | 0.1765    | 33.05     | 102.5     | 1.29      | 2.504     | 0.8019    | 1.953     |
| 52        | -0.1314   | 0.8458    | 0.03492   | 39.63     | 105.3     | 0.2218    | 3.319     | 0.5568    | 1.297     |
| 53        | -0.1438   | 0.4949    | 0.1024    | 18.8      | 152.6     | 4.689     | 2.255     | 0.7502    | 5.952     |
| 54        | -0.1406   | 0.9795    | 0.2609    | 42.32     | 126.0     | 2.918     | 3.356     | 0.9708    | 6.681     |
| 55        | -0.1425   | 0.7872    | 0.1101    | 25.5      | 64.08     | 3.928     | 0.0537    | 0.8012    | 6.059     |
| 56        | -0.1422   | 0.795     | 0.1383    | 32.91     | 4.366     | 4.319     | 2.125     | 0.7189    | 3.81      |
| 57        | -0.1381   | 0.6607    | 0.1802    | 30.61     | 103.7     | 0.4535    | 2.409     | 0.6162    | 1.655     |
| 58        | -0.1363   | 0.7144    | 0.1872    | 40.09     | 96.46     | 1.393     | 0.8552    | 0.7647    | 3.93      |
| 59        | -0.1344   | 0.6985    | 0.06136   | 13.79     | 155.8     | 0.2614    | 0.1525    | 0.9506    | 6.614     |
| 60        | -0.1353   | 0.5445    | 0.08866   | 7.036     | 119.9     | 0.7228    | 4.164     | 0.7365    | 5.519     |
| 61        | -0.1518   | 0.5281    | 0.1433    | 43.4      | 104.8     | 4.714     | 2.315     | 0.5443    | 4.147     |
| 62        | -0.1372   | 0.8161    | 0.2379    | 32.68     | 100.2     | 1.799     | 0.2554    | 0.8208    | 2.413     |
| 63        | -0.1391   | 0.8408    | 0.2893    | 33.2      | 102.4     | 0.2715    | 1.431     | 0.8044    | 1.197     |
| 64        | -0.1448   | 0.6565    | 0.2465    | 40.32     | 21.8      | 3.184     | 0.02414   | 0.6335    | 4.449     |
| 65        | -0.1401   | 0.9936    | 0.1625    | 20.91     | 45.82     | 1.265     | 4.495     | 0.5607    | 5.422     |
| 66        | -0.1343   | 0.9828    | 0.1219    | 32.1      | 103.6     | 0.267     | 3.2       | 0.6388    | 1.431     |
| 67        | -0.1369   | 0.6384    | 0.1099    | 23.07     | 47.86     | 0.136     | 1.955     | 0.5178    | 3.838     |
| 68        | -0.1485   | 0.7476    | 0.29      | 8.311     | 66.54     | 0.8591    | 1.022     | 0.6065    | 5.352     |
| 69        | -0.1351   | 0.4591    | 0.0572    | 26.38     | 157.0     | 2.864     | 0.8709    | 0.9726    | 4.411     |
| 70        | -0.136    | 0.8615    | 0.1178    | 32.52     | 176.0     | 2.009     | 2.925     | 0.5947    | 1.228     |
| 71        | -0.1314   | 0.4152    | 0.07796   | 35.53     | 96.1      | 0.6788    | 2.584     | 0.7917    | 3.057     |
| 72        | -0.1387   | 0.9906    | 0.05026   | 10.26     | 147.7     | 2.58      | 3.623     | 0.7764    | 5.888     |
| 73        | -0.1341   | 0.652     | 0.01474   | 27.18     | 80.9      | 2.718     | 3.287     | 0.9516    | 2.608     |
| 74        | -0.1373   | 0.7793    | 0.1504    | 13.52     | 55.16     | 1.722     | 0.7842    | 0.9561    | 2.521     |
| 75        | -0.1497   | 0.8025    | 0.2801    | 19.46     | 25.9      | 2.688     | 3.268     | 0.5064    | 6.62      |
| 76        | -0.1335   | 0.5551    | 0.1288    | 41.65     | 101.5     | 0.5086    | 1.318     | 0.7805    | 3.495     |
| 77        | -0.1364   | 0.8765    | 0.1016    | 14.14     | 32.7      | 1.953     | 4.918     | 0.88      | 4.903     |
| 78        | -0.1295   | 0.6066    | 0.01635   | 31.76     | 103.3     | 0.6928    | 2.619     | 0.7084    | 1.471     |
| 79        | -0.1595   | 0.995     | 0.2404    | 33.23     | 104.3     | 4.689     | 3.545     | 0.5137    | 4.407     |
| 80        | -0.1373   | 0.7015    | 0.2791    | 40.25     | 105.5     | 0.2279    | 3.042     | 0.7049    | 1.978     |
| 81        | -0.1346   | 0.9418    | 0.03068   | 26.75     | 80.88     | 2.5       | 3.081     | 0.6919    | 3.329     |
| 82        | -0.1408   | 0.7264    | 0.2733    | 29.91     | 104.6     | 0.8715    | 4.747     | 0.8098    | 1.547     |
| 83        | -0.1345   | 0.4191    | 0.1065    | 32.14     | 103.7     | 1.977     | 1.804     | 0.7993    | 2.351     |
| 84        | -0.1383   | 0.795     | 0.171     | 32.36     | 100.2     | 0.992     | 0.4732    | 0.5037    | 2.088     |
| 85        | -0.1384   | 0.7631    | 0.1521    | 23.65     | 152.1     | 1.046     | 1.203     | 0.5145    | 1.595     |
| 86        | -0.1325   | 0.5336    | 0.1056    | 32.23     | 102.6     | 0.6281    | 1.2       | 0.825     | 2.732     |
| 87        | -0.1336   | 0.5398    | 0.1834    | 42.39     | 102.0     | 0.9592    | 1.821     | 0.7434    | 4.087     |
| 88        | -0.1416   | 0.6793    | 0.285     | 32.62     | 103.7     | 0.628     | 1.172     | 0.7065    | 2.928     |
| 89        | -0.1299   | 0.5817    | 0.02003   | 32.8      | 103.4     | 1.495     | 2.4       | 0.9458    | 2.068     |
| 90        | -0.1397   | 0.8614    | 0.1702    | 31.29     | 105.0     | 1.745     | 2.214     | 0.5716    | 1.767     |
| 91        | -0.1344   | 0.9488    | 0.1653    | 32.46     | 103.3     | 1.414     | 2.13      | 0.6533    | 1.311     |
| 92        | -0.1304   | 0.4078    | 0.02737   | 36.92     | 96.34     | 0.303     | 2.922     | 0.5677    | 3.423     |
| 93        | -0.1362   | 0.536     | 0.2065    | 22.11     | 45.33     | 1.556     | 3.325     | 0.8911    | 5.304     |
| 94        | -0.1314   | 0.7604    | 0.07972   | 34.82     | 95.81     | 0.7551    | 2.87      | 0.9216    | 2.722     |
| 95        | -0.1296   | 0.5554    | 0.008351  | 32.55     | 103.8     | 1.154     | 0.2258    | 0.9113    | 1.296     |
| 96        | -0.1355   | 0.5758    | 0.1764    | 31.42     | 104.6     | 1.049     | 2.886     | 0.9475    | 1.441     |
| 97        | -0.1328   | 0.5158    | 0.05321   | 31.87     | 104.8     | 0.716     | 0.4359    | 0.5588    | 1.395     |
| 98        | -0.1346   | 0.8084    | 0.07323   | 32.13     | 102.0     | 0.6874    | 1.655     | 0.5047    | 2.843     |
| 99        | -0.135    | 0.6874    | 0.173     | 32.32     | 102.8     | 1.581     | 3.541     | 0.8568    | 1.724     |
| 100       | -0.1379   | 0.9895    | 0.2415    | 31.75     | 102.6     | 0.4289    | 0.999     | 0.9725    | 1.884     |
=========================================================================================================================
Best Parameters: {'colsample_bytree': np.float64(0.5324981920983456), 'learning_rate': np.float64(0.015707051319786915), 'min_child_samples': np.float64(31.613535655929358), 'num_leaves': np.float64(104.91948324786794), 'reg_alpha': np.float64(0.03146264671192445), 'reg_lambda': np.float64(3.501834997723873), 'subsample': np.float64(0.6419254502909304), 'subsample_freq': np.float64(1.737011589154804)}
Best RMSE Score: 0.12825219582182507
Best boosting round from CV: 1444
LGBM Bayes model saved to models/final_model_lgbm_bayes.pkl
LGBM (GridSearch) Performance:
Root Mean Squared Error: 0.1351
LGBM (Bayes Opt) Performance:
Root Mean Squared Error: 0.1344

============================================================
Running s2_model.a7_catboost
============================================================

Learning rate set to 0.04196
0:	learn: 0.3806785	total: 78.6ms	remaining: 1m 18s
200:	learn: 0.1006103	total: 5.66s	remaining: 22.5s
400:	learn: 0.0796899	total: 11.5s	remaining: 17.1s
600:	learn: 0.0664873	total: 17.2s	remaining: 11.4s
800:	learn: 0.0564137	total: 23.1s	remaining: 5.75s
999:	learn: 0.0497047	total: 29s	remaining: 0us
CatBoost model saved to models/final_model_catboost_basic.cbm
CatBoost Regressor Performance:
Root Mean Squared Error: 0.1348

============================================================
Running s2_model.a8_stacking
============================================================

Processing Fold 1...
Processing Fold 2...
Processing Fold 3...
Processing Fold 4...
Processing Fold 5...
Processing Fold 6...
Processing Fold 7...
Processing Fold 8...
Processing Fold 9...
Processing Fold 10...

[Meta-learner] NNLS kept 5 of 14 base models
  svr_rbf      0.5796
  ridge        0.1399
  lgbm_bayes   0.1020
  svr_linear   0.0989
  xgb          0.0925
  intercept    -0.1578
  weight sum   1.0129
[Meta-learner] Zero weight: ['xgb_bayes', 'lasso', 'enet', 'knn', 'dt', 'rf', 'et', 'lgbm', 'cat_basic']

[Comparison] RMSE per model
       model  oof_rmse  val_rmse
     svr_rbf   0.11700   0.13032
        enet   0.12446   0.12814
       ridge   0.12448   0.12762
       lasso   0.12680   0.13179
         xgb   0.12720   0.13155
   cat_basic   0.12895   0.13477
   xgb_bayes   0.12923   0.13317
  svr_linear   0.12987   0.12629
  lgbm_bayes   0.13183   0.13441
        lgbm   0.13207   0.13505
          et   0.14581   0.15425
          rf   0.15112   0.16749
         knn   0.15747   0.16533
          dt   0.19486   0.19046
stack (nnls)   0.11629   0.12452
Meta-learner saved to models/meta_learner_nnls.json
Surviving model list saved to models/meta_learner_active_models.txt

============================================================
All s2_model scripts completed successfully.
============================================================
```
