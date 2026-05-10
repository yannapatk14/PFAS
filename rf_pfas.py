import joblib

# โหลด model
rf = joblib.load(r'C:\Hydrogeology\output\rf_pfas_tokyo.pkl')

# ดูข้อมูล model
print(type(rf))                        # RandomForestClassifier
print(rf.n_estimators)                 # 300
print(rf.oob_score_)                   # OOB score
print(rf.feature_importances_)         # feature importance array
print(rf.classes_)                     # [0, 1]