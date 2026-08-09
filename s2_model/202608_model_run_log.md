# s2_model_run_log

Terminal color codes were removed so this renders cleanly on GitHub (GFM does not apply custom ANSI/HTML colors in repo views).

```text
============================================================
Running s2_model.a1_regression
============================================================


Model final parameters:
const                                      10.950783
Age_House                                  -0.071796
BedroomAbvGr                               -0.047392
BsmtExposure_encoded                        0.010707
BsmtFullBath                                0.018122
CentralAir_Electrical_N_SBrkr              -0.063217
EnclosedPorch                              -0.000694
ExterQual_encoded                           0.016571
Exterior1st_Exterior2nd_BrkFace             0.091894
Exterior1st_Exterior2nd_BrkFace_Wd Sdng     0.147885
FinishedAreaPct                            -0.003037
Fireplaces                                  0.012928
Foundation_PConc                            0.035601
FullBath                                    0.013275
Functional_encoded                          0.034773
GarageArea                                  0.013694
GarageCars                                  0.008493
Garage_AgeCars                             -0.001225
Garage_Space                                0.014526
HalfBath                                    0.010379
KitchenAbvGr                               -0.016539
KitchenQual_encoded                         0.022158
Living_Rooms                               -0.117089
Neighborhood_Condition_BrkSide_Norm         0.080782
Neighborhood_Condition_Crawfor_Norm         0.134180
Neighborhood_Condition_NoRidge_Norm         0.093755
Neighborhood_Condition_NridgHt_Norm         0.083939
Neighborhood_Condition_Somerst_Norm         0.051809
Neighborhood_Condition_StoneBr_Norm         0.140549
OverallCond                                 0.043866
OverallQual                                 0.050327
Porch_Age                                   0.004208
Ratio_2ndFlr_Living                        -0.009445
Ratio_Bedroom_Rooms                         0.027781
SaleCondition_Normal                        0.065466
SaleType_New                                0.120280
ScreenPorch                                 0.008748
TotRmsAbvGrd                                0.104961
cbrt_MasVnrArea                            -0.002455
cbrt_OpenPorchSF                            0.004745
log_1stFlrSF                               -0.001911
log_2ndFlrSF                               -0.021394
log_GrLivArea                               0.219520
log_LotArea                                 0.046211
log_Yrs_Since_Remodel                      -0.007903
sqrt_BsmtFinSF1                             0.031586
sqrt_TotalBsmtSF                            0.019318
sqrt_WoodDeckSF                             0.011415
dtype: float64

Model fitting summary:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              SalePrice   R-squared:                       0.928
Model:                            OLS   Adj. R-squared:                  0.925
Method:                 Least Squares   F-statistic:                     307.0
Date:                Sun, 09 Aug 2026   Prob (F-statistic):               0.00
Time:                        00:33:07   Log-Likelihood:                 957.94
No. Observations:                1166   AIC:                            -1820.
Df Residuals:                    1118   BIC:                            -1577.
Df Model:                          47                                         
Covariance Type:            nonrobust                                         
===========================================================================================================
                                              coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------------------------------
const                                      10.9508      0.054    201.670      0.000      10.844      11.057
Age_House                                  -0.0718      0.008     -9.479      0.000      -0.087      -0.057
BedroomAbvGr                               -0.0474      0.017     -2.730      0.006      -0.081      -0.013
BsmtExposure_encoded                        0.0107      0.004      2.947      0.003       0.004       0.018
BsmtFullBath                                0.0181      0.004      4.058      0.000       0.009       0.027
CentralAir_Electrical_N_SBrkr              -0.0632      0.019     -3.358      0.001      -0.100      -0.026
EnclosedPorch                              -0.0007      0.007     -0.095      0.924      -0.015       0.014
ExterQual_encoded                           0.0166      0.010      1.703      0.089      -0.003       0.036
Exterior1st_Exterior2nd_BrkFace             0.0919      0.027      3.403      0.001       0.039       0.145
Exterior1st_Exterior2nd_BrkFace_Wd Sdng     0.1479      0.038      3.920      0.000       0.074       0.222
FinishedAreaPct                            -0.0030      0.022     -0.136      0.892      -0.047       0.041
Fireplaces                                  0.0129      0.004      3.153      0.002       0.005       0.021
Foundation_PConc                            0.0356      0.011      3.390      0.001       0.015       0.056
FullBath                                    0.0133      0.005      2.430      0.015       0.003       0.024
Functional_encoded                          0.0348      0.005      6.513      0.000       0.024       0.045
GarageArea                                  0.0137      0.008      1.625      0.104      -0.003       0.030
GarageCars                                  0.0085      0.009      0.930      0.352      -0.009       0.026
Garage_AgeCars                             -0.0012      0.009     -0.134      0.893      -0.019       0.017
Garage_Space                                0.0145      0.011      1.338      0.181      -0.007       0.036
HalfBath                                    0.0104      0.005      2.168      0.030       0.001       0.020
KitchenAbvGr                               -0.0165      0.004     -4.125      0.000      -0.024      -0.009
KitchenQual_encoded                         0.0222      0.008      2.681      0.007       0.006       0.038
Living_Rooms                               -0.1171      0.042     -2.790      0.005      -0.199      -0.035
Neighborhood_Condition_BrkSide_Norm         0.0808      0.021      3.766      0.000       0.039       0.123
Neighborhood_Condition_Crawfor_Norm         0.1342      0.020      6.847      0.000       0.096       0.173
Neighborhood_Condition_NoRidge_Norm         0.0938      0.022      4.326      0.000       0.051       0.136
Neighborhood_Condition_NridgHt_Norm         0.0839      0.018      4.747      0.000       0.049       0.119
Neighborhood_Condition_Somerst_Norm         0.0518      0.017      2.980      0.003       0.018       0.086
Neighborhood_Condition_StoneBr_Norm         0.1405      0.026      5.492      0.000       0.090       0.191
OverallCond                                 0.0439      0.004     11.340      0.000       0.036       0.051
OverallQual                                 0.0503      0.005     10.932      0.000       0.041       0.059
Porch_Age                                   0.0042      0.008      0.541      0.589      -0.011       0.019
Ratio_2ndFlr_Living                        -0.0094      0.046     -0.204      0.839      -0.100       0.081
Ratio_Bedroom_Rooms                         0.0278      0.013      2.218      0.027       0.003       0.052
SaleCondition_Normal                        0.0655      0.012      5.651      0.000       0.043       0.088
SaleType_New                                0.1203      0.021      5.855      0.000       0.080       0.161
ScreenPorch                                 0.0087      0.003      2.555      0.011       0.002       0.015
TotRmsAbvGrd                                0.1050      0.023      4.557      0.000       0.060       0.150
cbrt_MasVnrArea                            -0.0025      0.004     -0.606      0.545      -0.010       0.005
cbrt_OpenPorchSF                            0.0047      0.004      1.223      0.222      -0.003       0.012
log_1stFlrSF                               -0.0019      0.033     -0.058      0.954      -0.067       0.063
log_2ndFlrSF                               -0.0214      0.021     -1.004      0.316      -0.063       0.020
log_GrLivArea                               0.2195      0.049      4.474      0.000       0.123       0.316
log_LotArea                                 0.0462      0.004     11.297      0.000       0.038       0.054
log_Yrs_Since_Remodel                      -0.0079      0.007     -1.214      0.225      -0.021       0.005
sqrt_BsmtFinSF1                             0.0316      0.005      6.603      0.000       0.022       0.041
sqrt_TotalBsmtSF                            0.0193      0.022      0.874      0.382      -0.024       0.063
sqrt_WoodDeckSF                             0.0114      0.004      3.115      0.002       0.004       0.019
==============================================================================
Omnibus:                      350.405   Durbin-Watson:                   2.041
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2658.849
Skew:                          -1.174   Prob(JB):                         0.00
Kurtosis:                      10.015   Cond. No.                         294.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Model summary saved to models/ols_model_summary.txt

10-Fold CV RMSE (log-transformed scale): 0.11119944783717262
Optimal Parameter: {'alpha': 0.1}
Optimal Estimator: Ridge(alpha=0.1)
Ridge model saved to models/final_model_ridge.pkl
10-Fold CV RMSE: 0.11438362866791667
Optimal Parameter: {'alpha': 0.001}
Optimal Estimator: Lasso(alpha=0.001)
Selected features for Lasso:
Index(['Age_House', 'BedroomAbvGr', 'BsmtExposure_encoded', 'BsmtFullBath',
       'CentralAir_Electrical_N_SBrkr', 'EnclosedPorch', 'ExterQual_encoded',
       'Exterior1st_Exterior2nd_BrkFace',
       'Exterior1st_Exterior2nd_BrkFace_Wd Sdng', 'Fireplaces',
       'Foundation_PConc', 'FullBath', 'Functional_encoded', 'GarageArea',
       'GarageCars', 'Garage_AgeCars', 'Garage_Space', 'HalfBath',
       'KitchenAbvGr', 'KitchenQual_encoded',
       'Neighborhood_Condition_BrkSide_Norm',
       'Neighborhood_Condition_Crawfor_Norm',
       'Neighborhood_Condition_NoRidge_Norm',
       'Neighborhood_Condition_NridgHt_Norm',
       'Neighborhood_Condition_Somerst_Norm',
       'Neighborhood_Condition_StoneBr_Norm', 'OverallCond', 'OverallQual',
       'Porch_Age', 'SaleCondition_Normal', 'SaleType_New', 'ScreenPorch',
       'TotRmsAbvGrd', 'cbrt_MasVnrArea', 'cbrt_OpenPorchSF', 'log_1stFlrSF',
       'log_2ndFlrSF', 'log_GrLivArea', 'log_LotArea', 'log_Yrs_Since_Remodel',
       'sqrt_BsmtFinSF1', 'sqrt_TotalBsmtSF', 'sqrt_WoodDeckSF'],
      dtype='object')
Lasso model saved to models/final_model_lasso.pkl
10-Fold CV RMSE: 0.11121022205742996
Optimal Parameter: {'alpha': 0.001, 'l1_ratio': 0.1}
Optimal Estimator: ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=42)
ElasticNet model saved to models/final_model_enet.pkl
OLS Model Performance:
Root Mean Squared Error: 0.1197
Ridge Model Performance:
Root Mean Squared Error: 0.1197
Lasso Model Performance:
Root Mean Squared Error: 0.1175
ElasticNet Model Performance:
Root Mean Squared Error: 0.1185

============================================================
Running s2_model.a2_svr
============================================================

RBF SVR 10-Fold CV RMSE: 0.11167960696249897
RBF SVR Optimal Parameter: {'C': 10, 'epsilon': 0.01, 'gamma': 0.001, 'tol': 0.0001}
RBF SVR Optimal Estimator: SVR(C=10, epsilon=0.01, gamma=0.001, tol=0.0001)
RBF SVR model saved to models/final_model_svr_rbf.pkl
LinearSVR 10-Fold CV RMSE: 0.11229037553903301
LinearSVR Optimal Parameter: {'C': 50, 'dual': False, 'epsilon': 0.01, 'loss': 'squared_epsilon_insensitive', 'tol': 0.0001}
LinearSVR Optimal Estimator: LinearSVR(C=50, dual=False, epsilon=0.01, loss='squared_epsilon_insensitive',
          max_iter=50000, random_state=42)
LinearSVR model saved to models/final_model_linear_svr.pkl
RBF SVR Model Performance:
Root Mean Squared Error: 0.1137
LinearSVR Model Performance:
Root Mean Squared Error: 0.1174

============================================================
Running s2_model.a3_knn
============================================================

/mnt/custom-file-systems/efs/fs-43568745_fsap-02fff3db64dd335d1/house-price-prediction/.venv/lib/python3.12/site-packages/numpy/ma/core.py:2885: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,
KNN 10-Fold CV RMSE: 0.1530681312480007
KNN Optimal Parameter: {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
KNN Optimal Estimator: KNeighborsRegressor(p=1, weights='distance')
KNN model saved to models/final_model_knn.pkl
KNN Model Performance:
Root Mean Squared Error: 0.1657

============================================================
Running s2_model.a4_trees
============================================================

10-Fold CV RMSE: 0.18876415400039107
Optimal Parameters: {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0}
Optimal Estimator: DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=42)
decision tree model saved to models/final_model_dt.pkl
Decision Tree Performance:
Root Mean Squared Error: 0.1949
/mnt/custom-file-systems/efs/fs-43568745_fsap-02fff3db64dd335d1/house-price-prediction/.venv/lib/python3.12/site-packages/numpy/ma/core.py:2885: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,
10-Fold CV RMSE: 0.14324264353825195
Optimal Parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Optimal Estimator: RandomForestRegressor(max_depth=10, max_features='sqrt', min_samples_split=5,
                      n_estimators=200, random_state=42)
random forest model saved to models/final_model_rf.pkl
Random Forst Tree Performance:
Root Mean Squared Error: 0.1619
/mnt/custom-file-systems/efs/fs-43568745_fsap-02fff3db64dd335d1/house-price-prediction/.venv/lib/python3.12/site-packages/numpy/ma/core.py:2885: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,
10-Fold CV RMSE: 0.13756930234867643
Optimal Parameters: {'bootstrap': False, 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
Optimal Estimator: ExtraTreesRegressor(max_depth=10, max_features=None, min_samples_leaf=2,
                    min_samples_split=5, n_estimators=200, random_state=42)
ExtraTreesRegressor model saved to models/final_model_et.pkl
Extra Trees Regressor Performance:
Root Mean Squared Error: 0.1447

============================================================
Running s2_model.a5_xgb
============================================================

Fitting 10 folds for each of 108 candidates, totalling 1080 fits
10-Fold CV RMSE: 0.11991405513127065
Optimal Parameters: {'learning_rate': 0.07, 'max_depth': 4, 'min_child_weight': 2, 'reg_alpha': 0, 'reg_lambda': 0.281}
Optimal Estimator: XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=0.75, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.07, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=4, max_leaves=None,
             min_child_weight=2, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=200, n_jobs=1,
             num_parallel_tree=None, random_state=42, ...)
xgboost model saved to models/final_model_xgb.pkl
|   iter    |  target   | colsam... |   gamma   | learni... | max_depth | min_ch... | reg_alpha | reg_la... | subsample |
-------------------------------------------------------------------------------------------------------------------------
| 1         | -0.1982   | 0.6247    | 4.754     | 0.2209    | 6.789     | 2.404     | 0.78      | 0.2904    | 0.9331    |
| 2         | -0.1958   | 0.7607    | 3.54      | 0.01107   | 9.759     | 8.492     | 1.062     | 0.9091    | 0.5917    |
| 3         | -0.177    | 0.5825    | 2.624     | 0.1324    | 4.33      | 6.507     | 0.6975    | 1.461     | 0.6832    |
| 4         | -0.1953   | 0.6736    | 3.926     | 0.0639    | 6.114     | 6.332     | 0.2323    | 3.038     | 0.5853    |
| 5         | -0.1993   | 0.439     | 4.744     | 0.2899    | 8.467     | 3.742     | 0.4884    | 3.421     | 0.7201    |
| 6         | -0.1888   | 0.4732    | 2.476     | 0.01514   | 9.275     | 3.329     | 3.313     | 1.559     | 0.76      |
| 7         | -0.1722   | 0.728     | 0.9243    | 0.291     | 8.201     | 9.455     | 4.474     | 2.989     | 0.9609    |
| 8         | -0.1572   | 0.4531    | 0.9799    | 0.01834   | 4.603     | 4.498     | 1.357     | 4.144     | 0.6784    |
| 9         | -0.2048   | 0.5686    | 2.713     | 0.04657   | 8.418     | 1.671     | 4.934     | 3.861     | 0.5994    |
| 10        | -0.1973   | 0.4033    | 4.077     | 0.2135    | 7.832     | 7.941     | 0.3702    | 1.792     | 0.5579    |
| 11        | -0.1274   | 0.4       | 0.0       | 0.07299   | 3.556     | 6.061     | 2.556     | 3.867     | 0.875     |
| 12        | -0.1666   | 0.8116    | 0.4783    | 0.2844    | 2.046     | 7.771     | 4.912     | 4.472     | 0.8581    |
| 13        | -0.1294   | 0.4       | 0.0       | 0.005     | 2.0       | 4.998     | 2.436     | 2.803     | 1.0       |
| 14        | -0.1257   | 0.8033    | 0.09748   | 0.02638   | 2.343     | 8.648     | 0.4814    | 4.943     | 0.6555    |
| 15        | -0.1222   | 0.8377    | 0.02794   | 0.01278   | 2.225     | 9.009     | 0.2441    | 1.546     | 0.8861    |
| 16        | -0.1267   | 0.7591    | 0.02975   | 0.07511   | 4.744     | 9.811     | 0.1726    | 2.588     | 0.9286    |
| 17        | -0.128    | 0.7816    | 0.03595   | 0.2454    | 2.413     | 7.454     | 0.1479    | 2.639     | 0.6693    |
| 18        | -0.1492   | 0.8683    | 0.07658   | 0.1571    | 3.724     | 9.267     | 3.82      | 0.04887   | 0.8722    |
| 19        | -0.1447   | 0.4124    | 0.4394    | 0.1452    | 2.131     | 9.973     | 0.954     | 3.275     | 0.9071    |
| 20        | -0.1643   | 0.7949    | 0.5143    | 0.1122    | 2.128     | 1.185     | 4.873     | 0.6533    | 0.6856    |
| 21        | -0.1328   | 0.7291    | 0.01162   | 0.1742    | 2.05      | 4.699     | 2.907     | 4.91      | 0.5906    |
| 22        | -0.1268   | 0.5156    | 0.05429   | 0.2366    | 3.343     | 9.503     | 0.1961    | 0.4772    | 0.512     |
| 23        | -0.1392   | 0.7508    | 0.2822    | 0.1761    | 7.509     | 8.998     | 0.04827   | 4.627     | 0.9159    |
| 24        | -0.1305   | 0.6621    | 0.1174    | 0.2374    | 2.015     | 1.097     | 0.1642    | 0.08361   | 0.7204    |
| 25        | -0.1284   | 0.4849    | 0.08732   | 0.07238   | 4.583     | 7.868     | 0.9545    | 4.91      | 0.5644    |
| 26        | -0.126    | 0.475     | 0.07017   | 0.08547   | 2.167     | 5.989     | 0.1075    | 4.483     | 0.7962    |
| 27        | -0.1285   | 0.9379    | 0.1821    | 0.118     | 2.081     | 4.977     | 0.3194    | 0.487     | 0.6107    |
| 28        | -0.1298   | 0.8697    | 0.09826   | 0.01619   | 3.246     | 7.693     | 0.727     | 0.3833    | 0.9161    |
| 29        | -0.1214   | 0.7014    | 0.04695   | 0.02049   | 2.076     | 2.218     | 0.1155    | 2.012     | 0.7715    |
| 30        | -0.1531   | 0.6209    | 0.8545    | 0.2131    | 2.042     | 1.705     | 0.1213    | 4.548     | 0.961     |
| 31        | -0.1273   | 0.4721    | 0.04389   | 0.1727    | 3.921     | 3.828     | 0.7458    | 0.4191    | 0.6411    |
| 32        | -0.127    | 0.5267    | 0.04826   | 0.2263    | 4.86      | 1.786     | 0.09886   | 1.542     | 0.7997    |
| 33        | -0.136    | 0.8515    | 0.08039   | 0.2051    | 8.086     | 1.787     | 0.5721    | 0.2985    | 0.926     |
| 34        | -0.1269   | 0.6526    | 0.06614   | 0.01888   | 4.75      | 9.993     | 0.1783    | 4.971     | 0.8884    |
| 35        | -0.131    | 0.5177    | 0.04191   | 0.1793    | 6.205     | 3.858     | 0.005747  | 1.207     | 0.9959    |
| 36        | -0.1362   | 0.8523    | 0.1764    | 0.2685    | 9.888     | 1.341     | 0.3545    | 4.384     | 0.7159    |
| 37        | -0.1347   | 0.9954    | 0.09141   | 0.07895   | 3.731     | 7.919     | 0.3292    | 4.601     | 0.9979    |
| 38        | -0.1225   | 0.6131    | 0.005479  | 0.0203    | 7.888     | 9.816     | 0.1728    | 0.7332    | 0.8445    |
| 39        | -0.127    | 0.4347    | 0.1013    | 0.05643   | 9.92      | 4.557     | 0.2996    | 3.422     | 0.7601    |
| 40        | -0.1309   | 0.8741    | 0.03564   | 0.08899   | 9.192     | 6.043     | 0.4347    | 0.5418    | 0.9592    |
| 41        | -0.1187   | 0.4608    | 0.01911   | 0.0201    | 6.118     | 9.766     | 0.4378    | 0.3067    | 0.5262    |
| 42        | -0.1355   | 0.7741    | 0.08472   | 0.1321    | 6.716     | 9.785     | 1.76      | 0.08425   | 0.6578    |
| 43        | -0.1395   | 0.5543    | 0.282     | 0.2587    | 6.504     | 8.564     | 0.03074   | 0.3878    | 0.9231    |
| 44        | -0.131    | 0.6074    | 0.1495    | 0.1602    | 9.87      | 3.696     | 0.04021   | 0.487     | 0.8244    |
| 45        | -0.1274   | 0.6827    | 0.03732   | 0.08656   | 9.913     | 9.265     | 0.4642    | 4.414     | 0.8286    |
| 46        | -0.1296   | 0.5515    | 0.1671    | 0.1265    | 9.997     | 7.515     | 0.09606   | 1.425     | 0.6836    |
| 47        | -0.135    | 0.9393    | 0.03171   | 0.2749    | 8.406     | 9.648     | 0.2835    | 2.21      | 0.5063    |
| 48        | -0.1311   | 0.7086    | 0.1269    | 0.2358    | 9.076     | 7.236     | 0.1923    | 4.973     | 0.5516    |
| 49        | -0.12     | 0.5098    | 0.03588   | 0.05352   | 4.162     | 1.983     | 0.1046    | 0.3127    | 0.6532    |
| 50        | -0.1264   | 0.4       | 0.0       | 0.005     | 10.0      | 5.616     | 2.183     | 0.0       | 0.5       |
=========================================================================================================================
Best Parameters: {'colsample_bytree': np.float64(0.4607990231324068), 'gamma': np.float64(0.01911061553543092), 'learning_rate': np.float64(0.02010039502393073), 'max_depth': np.float64(6.117987956655009), 'min_child_weight': np.float64(9.765570734465339), 'reg_alpha': np.float64(0.4377880616772295), 'reg_lambda': np.float64(0.3067186102487529), 'subsample': np.float64(0.5261959460826902)}
Best RMSE Score: 0.1186602858467519
Best boosting round from CV: 925
XGB Bayes model saved to models/final_model_xgb_bayes.pkl
XGBoost (GridSearch) Performance:
Root Mean Squared Error: 0.1231
XGBoost (Bayes Opt) Performance:
Root Mean Squared Error: 0.1259

============================================================
Running s2_model.a6_lgbm
============================================================

Fitting 10 folds for each of 54 candidates, totalling 540 fits
10-Fold CV RMSE: 0.12271935737992301
Optimal Parameters: {'learning_rate': 0.11, 'min_child_samples': 10, 'num_leaves': 4, 'reg_lambda': 1.1}
Optimal Estimator: LGBMRegressor(colsample_bytree=0.85, learning_rate=0.11, min_child_samples=10,
              n_estimators=200, n_jobs=1, num_leaves=4, objective='regression',
              random_state=42, reg_lambda=1.1, subsample=0.85, subsample_freq=1,
              verbose=-1)
lgbm model saved to models/final_model_lgbm.pkl
|   iter    |  target   | colsam... | learni... | min_ch... | num_le... | reg_alpha | reg_la... | subsample | subsam... |
-------------------------------------------------------------------------------------------------------------------------
| 1         | -0.1398   | 0.6247    | 0.2855    | 37.94     | 121.3     | 0.7801    | 0.78      | 0.529     | 6.197     |
| 2         | -0.1483   | 0.7607    | 0.2139    | 5.926     | 194.1     | 4.162     | 1.062     | 0.5909    | 2.1       |
| 3         | -0.1366   | 0.5825    | 0.1598    | 24.44     | 61.08     | 3.059     | 0.6975    | 0.6461    | 3.198     |
| 4         | -0.1369   | 0.6736    | 0.2366    | 13.99     | 104.8     | 2.962     | 0.2323    | 0.8038    | 2.023     |
| 5         | -0.1318   | 0.439     | 0.2849    | 48.45     | 162.4     | 1.523     | 0.4884    | 0.8421    | 3.641     |
| 6         | -0.1319   | 0.4732    | 0.1511    | 6.547     | 182.2     | 1.294     | 3.313     | 0.6559    | 4.12      |
| 7         | -0.1396   | 0.728     | 0.05953   | 48.63     | 155.9     | 4.697     | 4.474     | 0.7989    | 6.531     |
| 8         | -0.1293   | 0.4531    | 0.06281   | 7.035     | 67.76     | 1.943     | 1.357     | 0.9144    | 3.141     |
| 9         | -0.1284   | 0.5686    | 0.1651    | 11.34     | 161.2     | 0.3728    | 4.934     | 0.8861    | 2.192     |
| 10        | -0.1415   | 0.4033    | 0.2456    | 36.81     | 146.9     | 3.856     | 0.3702    | 0.6792    | 1.695     |
| 11        | -0.1346   | 0.9179    | 0.1889    | 19.89     | 16.46     | 1.555     | 1.626     | 0.8648    | 4.825     |
| 12        | -0.1396   | 0.9323    | 0.1443    | 10.38     | 143.8     | 3.804     | 2.806     | 0.8855    | 3.963     |
| 13        | -0.13     | 0.7136    | 0.1311    | 6.144     | 25.15     | 0.1571    | 3.182     | 0.6572    | 4.051     |
| 14        | -0.1289   | 0.9445    | 0.07854   | 23.47     | 152.1     | 1.144     | 0.3849    | 0.6449    | 1.967     |
| 15        | -0.1426   | 0.9578    | 0.2434    | 33.5      | 174.8     | 4.018     | 0.9329    | 0.9463    | 4.236     |
| 16        | -0.1329   | 0.8845    | 0.2693    | 19.31     | 25.57     | 1.14      | 2.136     | 0.909     | 6.164     |
| 17        | -0.1266   | 0.4042    | 0.1557    | 23.78     | 47.53     | 0.5993    | 1.688     | 0.9715    | 2.939     |
| 18        | -0.1477   | 0.7113    | 0.2124    | 21.36     | 194.5     | 4.812     | 1.259     | 0.7486    | 2.805     |
| 19        | -0.1238   | 0.5709    | 0.01588   | 32.43     | 102.5     | 0.2574    | 1.393     | 0.9541    | 2.437     |
| 20        | -0.1387   | 0.4869    | 0.1494    | 49.35     | 51.44     | 3.361     | 3.808     | 0.6188    | 5.369     |
| 21        | -0.1387   | 0.4869    | 0.1494    | 49.35     | 51.44     | 3.361     | 3.808     | 0.6188    | 5.369     |
| 22        | -0.1334   | 0.8562    | 0.2949    | 36.68     | 155.6     | 1.581     | 3.536     | 0.8125    | 1.855     |
| 23        | -0.1281   | 0.4578    | 0.1338    | 21.72     | 45.35     | 1.624     | 3.727     | 0.8431    | 5.371     |
| 24        | -0.1311   | 0.7573    | 0.02465   | 33.27     | 99.7      | 2.667     | 1.093     | 0.6169    | 2.45      |
| 25        | -0.1256   | 0.7734    | 0.03673   | 31.37     | 105.2     | 0.5447    | 0.6331    | 0.9629    | 1.109     |
| 26        | -0.1296   | 0.4752    | 0.1738    | 34.58     | 103.6     | 0.5341    | 3.808     | 0.6315    | 3.144     |
| 27        | -0.1326   | 0.6699    | 0.1495    | 29.07     | 102.9     | 0.8829    | 0.5996    | 0.5081    | 1.165     |
| 28        | -0.1253   | 0.6077    | 0.07922   | 34.33     | 105.8     | 0.992     | 0.7105    | 0.5141    | 1.235     |
| 29        | -0.1257   | 0.5105    | 0.06505   | 34.05     | 104.2     | 0.2485    | 1.03      | 0.5375    | 2.968     |
| 30        | -0.125    | 0.8941    | 0.04833   | 31.73     | 105.3     | 0.5364    | 0.1263    | 0.9268    | 3.278     |
| 31        | -0.1308   | 0.7857    | 0.09157   | 32.38     | 104.7     | 1.838     | 0.3066    | 0.584     | 2.226     |
| 32        | -0.1257   | 0.8266    | 0.0241    | 31.49     | 105.5     | 0.5715    | 1.867     | 0.5894    | 2.562     |
| 33        | -0.13     | 0.8586    | 0.1335    | 33.41     | 110.1     | 0.828     | 1.174     | 0.6618    | 1.967     |
| 34        | -0.135    | 0.6134    | 0.2904    | 30.87     | 105.7     | 0.2363    | 0.2624    | 0.6266    | 2.983     |
| 35        | -0.1281   | 0.4395    | 0.1674    | 33.83     | 103.4     | 0.4253    | 1.548     | 0.5432    | 1.26      |
| 36        | -0.1273   | 0.9831    | 0.09983   | 32.42     | 105.1     | 0.4957    | 1.126     | 0.7328    | 4.332     |
| 37        | -0.1336   | 0.7951    | 0.2734    | 31.58     | 104.3     | 0.7339    | 0.7629    | 0.8518    | 2.703     |
| 38        | -0.1303   | 0.6067    | 0.2206    | 32.61     | 106.5     | 1.639     | 0.9788    | 0.7091    | 1.92      |
| 39        | -0.1238   | 0.6079    | 0.01881   | 33.65     | 105.4     | 0.5966    | 0.7542    | 0.5519    | 2.005     |
| 40        | -0.135    | 0.8456    | 0.2172    | 32.14     | 105.0     | 0.7712    | 3.544     | 0.5424    | 2.816     |
| 41        | -0.1324   | 0.9805    | 0.2169    | 34.13     | 105.2     | 1.497     | 0.2752    | 0.8829    | 3.086     |
| 42        | -0.1291   | 0.6853    | 0.2531    | 35.24     | 104.4     | 0.2601    | 2.227     | 0.799     | 2.09      |
| 43        | -0.1326   | 0.9063    | 0.156     | 33.51     | 103.6     | 0.9416    | 1.545     | 0.6397    | 3.158     |
| 44        | -0.142    | 0.7793    | 0.2358    | 34.29     | 102.9     | 0.02936   | 1.445     | 0.5088    | 2.753     |
| 45        | -0.13     | 0.4828    | 0.1509    | 24.41     | 48.16     | 0.9064    | 1.934     | 0.695     | 2.697     |
| 46        | -0.1225   | 0.4162    | 0.02078   | 34.77     | 104.2     | 0.9872    | 0.6757    | 0.8337    | 1.911     |
| 47        | -0.1332   | 0.6531    | 0.1901    | 33.29     | 105.0     | 0.4513    | 1.198     | 0.519     | 2.805     |
| 48        | -0.1236   | 0.5178    | 0.06764   | 34.67     | 103.7     | 1.296     | 2.567     | 0.7704    | 1.653     |
| 49        | -0.1296   | 0.6598    | 0.255     | 34.96     | 103.8     | 1.775     | 2.932     | 0.8042    | 1.388     |
| 50        | -0.1255   | 0.7324    | 0.0438    | 34.49     | 104.5     | 1.133     | 1.399     | 0.9837    | 1.308     |
| 51        | -0.1255   | 0.9079    | 0.06665   | 31.69     | 106.0     | 0.4891    | 0.4452    | 0.5992    | 1.248     |
| 52        | -0.1342   | 0.492     | 0.2526    | 33.79     | 103.7     | 1.567     | 1.97      | 0.9887    | 1.699     |
| 53        | -0.1261   | 0.5307    | 0.08532   | 34.74     | 103.9     | 1.934     | 0.1205    | 0.7793    | 1.399     |
| 54        | -0.1259   | 0.4908    | 0.1661    | 35.35     | 103.5     | 1.205     | 2.48      | 0.8494    | 1.492     |
| 55        | -0.1265   | 0.9925    | 0.07549   | 30.64     | 106.3     | 0.07457   | 0.8139    | 0.6409    | 1.024     |
| 56        | -0.1367   | 0.4923    | 0.2754    | 32.69     | 106.0     | 1.104     | 2.024     | 0.6997    | 1.619     |
| 57        | -0.1294   | 0.8647    | 0.1664    | 31.59     | 102.1     | 0.3957    | 1.209     | 0.8617    | 1.406     |
| 58        | -0.1299   | 0.7608    | 0.1666    | 32.36     | 105.8     | 1.312     | 0.1505    | 0.6876    | 3.57      |
| 59        | -0.1236   | 0.6142    | 0.01043   | 32.19     | 107.0     | 0.804     | 1.099     | 0.8797    | 1.909     |
| 60        | -0.1262   | 0.7128    | 0.138     | 31.69     | 102.2     | 0.5384    | 0.7287    | 0.92      | 1.515     |
| 61        | -0.1239   | 0.8403    | 0.03738   | 35.44     | 103.6     | 0.169     | 1.3       | 0.6727    | 1.429     |
| 62        | -0.1258   | 0.4073    | 0.09907   | 33.79     | 103.9     | 1.027     | 0.06716   | 0.5068    | 1.49      |
| 63        | -0.1339   | 0.5948    | 0.2803    | 34.71     | 105.2     | 1.832     | 0.8717    | 0.615     | 1.09      |
| 64        | -0.135    | 0.5815    | 0.2206    | 34.88     | 103.4     | 1.715     | 1.314     | 0.5512    | 1.372     |
| 65        | -0.1248   | 0.9953    | 0.01651   | 35.35     | 102.7     | 0.5127    | 2.398     | 0.5434    | 1.891     |
| 66        | -0.1312   | 0.6018    | 0.2446    | 33.98     | 104.4     | 0.384     | 1.276     | 0.7504    | 1.983     |
| 67        | -0.1288   | 0.6561    | 0.1505    | 33.48     | 106.1     | 0.3112    | 0.4598    | 0.8257    | 1.762     |
| 68        | -0.1283   | 0.8642    | 0.1205    | 34.84     | 104.5     | 0.1246    | 0.4798    | 0.7292    | 1.514     |
| 69        | -0.1274   | 0.6722    | 0.0409    | 21.19     | 45.08     | 2.092     | 4.48      | 0.8042    | 5.736     |
| 70        | -0.1362   | 0.7701    | 0.2011    | 31.65     | 106.8     | 0.7383    | 1.463     | 0.527     | 2.895     |
| 71        | -0.1286   | 0.5007    | 0.2393    | 33.66     | 104.9     | 1.205     | 0.4618    | 0.9363    | 1.97      |
| 72        | -0.1339   | 0.9906    | 0.05026   | 10.26     | 147.7     | 2.58      | 3.623     | 0.7764    | 5.888     |
| 73        | -0.1293   | 0.652     | 0.01474   | 27.18     | 80.9      | 2.718     | 3.287     | 0.9516    | 2.608     |
| 74        | -0.1337   | 0.7793    | 0.1504    | 13.52     | 55.16     | 1.722     | 0.7842    | 0.9561    | 2.521     |
| 75        | -0.1456   | 0.8025    | 0.2801    | 19.46     | 25.9      | 2.688     | 3.268     | 0.5064    | 6.62      |
| 76        | -0.1327   | 0.9306    | 0.2473    | 30.32     | 105.4     | 1.004     | 1.381     | 0.7861    | 1.971     |
| 77        | -0.1258   | 0.8674    | 0.09775   | 35.47     | 104.9     | 0.4128    | 1.088     | 0.8528    | 1.636     |
| 78        | -0.128    | 0.5888    | 0.1871    | 31.54     | 102.7     | 1.033     | 1.196     | 0.9004    | 2.429     |
| 79        | -0.1302   | 0.6077    | 0.246     | 34.68     | 102.9     | 0.6124    | 2.612     | 0.9147    | 1.688     |
| 80        | -0.1302   | 0.5738    | 0.2315    | 11.16     | 161.2     | 0.5944    | 4.636     | 0.9075    | 2.092     |
| 81        | -0.1305   | 0.9418    | 0.03068   | 26.75     | 80.88     | 2.5       | 3.081     | 0.6919    | 3.329     |
| 82        | -0.1323   | 0.9583    | 0.1213    | 31.16     | 102.7     | 1.281     | 0.9984    | 0.6255    | 2.806     |
| 83        | -0.1266   | 0.5655    | 0.08388   | 23.35     | 46.57     | 1.225     | 2.058     | 0.6915    | 1.99      |
| 84        | -0.1289   | 0.6175    | 0.2082    | 35.49     | 103.2     | 1.634     | 3.613     | 0.8863    | 1.417     |
| 85        | -0.1266   | 0.7026    | 0.09871   | 33.36     | 104.5     | 1.079     | 0.326     | 0.6838    | 1.818     |
| 86        | -0.1273   | 0.5336    | 0.1056    | 32.23     | 102.6     | 0.6281    | 1.2       | 0.825     | 2.732     |
| 87        | -0.1272   | 0.6664    | 0.07222   | 22.49     | 47.61     | 1.379     | 2.69      | 0.8017    | 1.572     |
| 88        | -0.1356   | 0.9184    | 0.2366    | 34.05     | 105.0     | 0.4636    | 1.551     | 0.5435    | 1.047     |
| 89        | -0.1339   | 0.6973    | 0.2296    | 21.06     | 44.94     | 2.254     | 4.247     | 0.6767    | 5.097     |
| 90        | -0.1389   | 0.6182    | 0.2898    | 35.23     | 105.3     | 1.024     | 0.5639    | 0.518     | 1.346     |
| 91        | -0.1357   | 0.4631    | 0.2219    | 26.2      | 126.1     | 3.641     | 1.334     | 0.9282    | 2.148     |
| 92        | -0.1388   | 0.5178    | 0.1287    | 47.0      | 53.58     | 4.021     | 0.08855   | 0.8012    | 5.875     |
| 93        | -0.1283   | 0.536     | 0.2065    | 22.11     | 45.33     | 1.556     | 3.325     | 0.8911    | 5.304     |
| 94        | -0.1326   | 0.5816    | 0.1686    | 35.58     | 102.9     | 1.154     | 3.857     | 0.6031    | 1.866     |
| 95        | -0.1255   | 0.6923    | 0.07824   | 34.65     | 103.1     | 1.349     | 2.793     | 0.7042    | 1.664     |
| 96        | -0.1322   | 0.7397    | 0.2866    | 21.92     | 47.48     | 1.167     | 3.203     | 0.8516    | 2.284     |
| 97        | -0.1234   | 0.5158    | 0.05321   | 31.87     | 104.8     | 0.716     | 0.4359    | 0.5588    | 1.395     |
| 98        | -0.1313   | 0.8084    | 0.07323   | 32.13     | 102.0     | 0.6874    | 1.655     | 0.5047    | 2.843     |
| 99        | -0.1241   | 0.7796    | 0.0785    | 34.28     | 103.8     | 0.1795    | 0.6002    | 0.8071    | 2.215     |
| 100       | -0.1349   | 0.9895    | 0.2415    | 31.75     | 102.6     | 0.4289    | 0.999     | 0.9725    | 1.884     |
=========================================================================================================================
Best Parameters: {'colsample_bytree': np.float64(0.4162096440927156), 'learning_rate': np.float64(0.020782248775194148), 'min_child_samples': np.float64(34.77432062997563), 'num_leaves': np.float64(104.23806403260609), 'reg_alpha': np.float64(0.9872382756743581), 'reg_lambda': np.float64(0.6757087782429311), 'subsample': np.float64(0.8337293782622301), 'subsample_freq': np.float64(1.9109163959435957)}
Best RMSE Score: 0.1225002865550335
Best boosting round from CV: 1779
LGBM Bayes model saved to models/final_model_lgbm_bayes.pkl
LGBM (GridSearch) Performance:
Root Mean Squared Error: 0.1340
LGBM (Bayes Opt) Performance:
Root Mean Squared Error: 0.1343

============================================================
Running s2_model.a7_catboost
============================================================

|   iter    |  target   |   depth   | l2_lea... | learni... | min_da... | random... |
-------------------------------------------------------------------------------------
| 1         | -0.1243   | 4.749     | 19.06     | 0.1544    | 18.36     | 0.7801    |
| 2         | -0.1277   | 4.312     | 2.104     | 0.1772    | 18.43     | 3.54      |
| 3         | -0.1247   | 4.041     | 19.43     | 0.1715    | 7.158     | 0.9091    |
| 4         | -0.1239   | 4.367     | 6.781     | 0.1192    | 13.53     | 1.456     |
| 5         | -0.1243   | 5.224     | 3.65      | 0.07966   | 11.62     | 2.28      |
| 6         | -0.126    | 5.57      | 4.794     | 0.1174    | 18.18     | 0.2323    |
| 7         | -0.1239   | 5.215     | 4.24      | 0.04106   | 28.52     | 4.828     |
| 8         | -0.1228   | 5.617     | 6.788     | 0.0466    | 20.84     | 2.201     |
| 9         | -0.1243   | 4.244     | 10.41     | 0.03585   | 27.37     | 1.294     |
| 10        | -0.1232   | 5.325     | 6.923     | 0.1184    | 16.85     | 0.9243    |
| 11        | -0.1232   | 5.325     | 6.923     | 0.1184    | 16.85     | 0.9243    |
| 12        | -0.1276   | 5.18      | 11.72     | 0.194     | 10.27     | 0.1726    |
| 13        | -0.1243   | 5.309     | 9.464     | 0.05403   | 9.975     | 0.9954    |
| 14        | -0.1261   | 5.579     | 4.409     | 0.1067    | 26.27     | 4.111     |
| 15        | -0.1296   | 5.979     | 1.262     | 0.1799    | 14.98     | 0.7411    |
| 16        | -0.129    | 5.712     | 5.677     | 0.1567    | 25.37     | 4.364     |
| 17        | -0.1254   | 5.475     | 6.826     | 0.07047   | 21.28     | 2.172     |
| 18        | -0.1215   | 4.668     | 4.83      | 0.1056    | 26.75     | 3.655     |
| 19        | -0.1243   | 4.386     | 11.81     | 0.1473    | 12.66     | 2.663     |
| 20        | -0.1268   | 5.863     | 10.79     | 0.1908    | 28.14     | 3.252     |
| 21        | -0.1259   | 4.456     | 8.794     | 0.1537    | 2.539     | 1.825     |
| 22        | -0.1296   | 5.989     | 14.94     | 0.1675    | 28.88     | 0.6412    |
| 23        | -0.1274   | 4.84      | 18.96     | 0.1924    | 18.32     | 0.4804    |
| 24        | -0.1253   | 5.148     | 13.71     | 0.08819   | 29.74     | 1.566     |
| 25        | -0.1232   | 4.519     | 14.68     | 0.1146    | 13.28     | 2.967     |
| 26        | -0.1265   | 5.511     | 1.997     | 0.08634   | 29.98     | 2.296     |
| 27        | -0.127    | 5.475     | 8.815     | 0.1889    | 10.55     | 3.059     |
| 28        | -0.1234   | 4.399     | 16.53     | 0.07139   | 11.72     | 1.62      |
| 29        | -0.1231   | 4.062     | 9.802     | 0.1003    | 9.417     | 3.663     |
| 30        | -0.1265   | 5.208     | 8.305     | 0.1774    | 2.684     | 1.248     |
=====================================================================================
Best Parameters: {'depth': np.float64(4.6683180434250495), 'l2_leaf_reg': np.float64(4.829902823626683), 'learning_rate': np.float64(0.10561984864676233), 'min_data_in_leaf': np.float64(26.75328398408857), 'random_strength': np.float64(3.654558109708332)}
Best 5-Fold CV RMSE: 0.12154100408638628
10-Fold CV RMSE: 0.124643142817081
Best boosting round from CV: 539
CatBoost params saved to models/catboost_best_params.json
0:	learn: 0.3746414	total: 27.8ms	remaining: 15s
200:	learn: 0.0893322	total: 4.6s	remaining: 7.74s
400:	learn: 0.0658035	total: 9.31s	remaining: 3.2s
538:	learn: 0.0563723	total: 12.5s	remaining: 0us
CatBoost model saved to models/final_model_catboost_basic.cbm
CatBoost Regressor Performance:
Root Mean Squared Error: 0.1285

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

[Meta-learner] NNLS kept 8 of 14 base models
  svr_rbf      0.3609
  ridge        0.3255
  svr_linear   0.1132
  xgb          0.0577
  cat_basic    0.0564
  lgbm_bayes   0.0550
  lgbm         0.0355
  et           0.0149
  intercept    -0.2315
  weight sum   1.0190
[Meta-learner] Zero weight: ['xgb_bayes', 'lasso', 'enet', 'knn', 'dt', 'rf']

[Comparison] held-out RMSE on y_val
       model  val_rmse
     svr_rbf   0.11368
  svr_linear   0.11742
       lasso   0.11752
        enet   0.11850
       ridge   0.11966
         xgb   0.12306
   xgb_bayes   0.12590
   cat_basic   0.12854
        lgbm   0.13405
  lgbm_bayes   0.13430
          et   0.14467
          rf   0.16195
         knn   0.16565
          dt   0.19486
stack (nnls)   0.11175
Meta-learner saved to models/meta_learner_nnls.json
Surviving model list saved to models/meta_learner_active_models.txt

============================================================
All s2_model scripts completed successfully.
============================================================
```
