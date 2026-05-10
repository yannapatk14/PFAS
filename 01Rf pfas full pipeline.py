# =============================================================================
# RF PFAS Tokyo — Full Pipeline (Step 1-6)
# Working directory: C:\Hydrogeology\data\
# =============================================================================
# ติดตั้ง library ก่อน run (พิมพ์ใน terminal ทีเดียว):
#   pip install pandas numpy scikit-learn imbalanced-learn rasterio pyproj shap matplotlib joblib
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ใช้ non-interactive backend (ไม่ต้องการ display)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — แก้ path ตรงนี้ที่เดียว
# ============================================================
DATA_DIR  = r'C:\Hydrogeology\data'          # โฟลเดอร์ที่เก็บไฟล์ทั้งหมด
OUTPUT_DIR = r'C:\Hydrogeology\output'        # โฟลเดอร์บันทึกผลลัพธ์

CSV_FILE  = os.path.join(DATA_DIR, 'Kuroda_Ready_for_GIS.csv')

RASTERS = {
    'DTW':  os.path.join(DATA_DIR, 'DTW_tokyo_EPSG6677_100m.tif'),
    'LULC': os.path.join(DATA_DIR, 'Tokyo_LULC_100m_EPSG6677.tif'),
    'K':    os.path.join(DATA_DIR, 'Tokyo_K_Value.tif'),
    'Sand': os.path.join(DATA_DIR, 'Tokyo_Sand_100m_EPSG6677.tif'),
}

PFAS_THRESHOLD = 100  # ng/L — threshold แบ่ง contaminated / clean

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1 — Data Preparation
# ============================================================
print("=" * 60)
print("STEP 1: Data Preparation")
print("=" * 60)

df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
print(f"  Loaded: {df.shape[0]} wells, {df.shape[1]} columns")

# -- แปลงค่า PFAS: <LOQ, n.a., <0.25  →  numeric
def parse_pfas(val):
    val = str(val).strip()
    if val in ['<LOQ', 'n.a.', 'n.a. d', '-', 'nan']:
        return np.nan
    if val.startswith('<'):
        try:
            return float(val.replace('<', '')) / 2   # LOD/2 substitution
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

PFAS_COLS = ['PFHxS','PFHpS','PFOS','PFDS','PFHpA','PFOA',
             'PFNA','PFDA','PFUnDA','PFDoDA','PFTrDA','PFTeDA',
             'ΣPFAAs a','FOSA']

for col in PFAS_COLS:
    df[col] = df[col].apply(parse_pfas)

# -- Target variable
df['contaminated'] = (df['ΣPFAAs a'] > PFAS_THRESHOLD).astype(int)
print(f"  Target distribution: {df['contaminated'].value_counts().to_dict()}")

# -- Well depth: แปลง "50 c" → 50.0
df['Well_depth_m'] = (df['Well depth']
                      .astype(str)
                      .str.extract(r'(\d+\.?\d*)')[0]
                      .astype(float))

# -- Encode categorical
from sklearn.preprocessing import LabelEncoder

le_aquifer = LabelEncoder()
df['aquifer_code'] = le_aquifer.fit_transform(
    df['Type of aquifer'].fillna('Unknown'))

le_zone = LabelEncoder()
df['zone_code'] = le_zone.fit_transform(df['ZONE'].fillna('Unknown'))

print(f"  Aquifer types: {le_aquifer.classes_.tolist()}")
print(f"  Step 1 complete ✓")

# ============================================================
# STEP 2 — Extract Raster Values to Well Points
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Extract Raster Values")
print("=" * 60)

try:
    import rasterio
    from pyproj import Transformer

    coords_wgs84 = list(zip(df['Longitude'], df['Latitude']))

    for name, path in RASTERS.items():
        if not os.path.exists(path):
            print(f"  [SKIP] {name}: ไม่พบไฟล์ {os.path.basename(path)}")
            df[name] = np.nan
            continue

        with rasterio.open(path) as src:
            # reproject WGS84 → raster CRS
            epsg_code = src.crs.to_epsg()
            if epsg_code and epsg_code != 4326:
                transformer = Transformer.from_crs(
                    "EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
                coords_proj = [transformer.transform(lon, lat)
                               for lon, lat in coords_wgs84]
            else:
                coords_proj = coords_wgs84

            vals = [list(src.sample([xy]))[0][0] for xy in coords_proj]
            df[name] = vals

            # แทนค่า nodata ด้วย NaN
            nodata = src.nodata
            if nodata is not None:
                df[name] = df[name].replace(nodata, np.nan)

        n_valid = df[name].notna().sum()
        print(f"  {name}: {n_valid}/{len(df)} wells extracted  "
              f"(mean={df[name].mean():.2f})")

    print("  Step 2 complete ✓")

except ImportError as e:
    print(f"  [ERROR] ไม่พบ library: {e}")
    print("  กรุณาติดตั้ง: pip install rasterio pyproj")
    print("  ข้าม Step 2 → ใช้เฉพาะ hydrochemical features")
    for name in RASTERS:
        df[name] = np.nan

# ============================================================
# STEP 3 — Build Feature Matrix X
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Build Feature Matrix")
print("=" * 60)

# features ที่มี raster — ถ้า raster ไม่มีให้จะเป็น NaN ทั้งหมด
GEO_FEATURES  = [k for k in RASTERS if df[k].notna().any()]
HYDRO_FEATURES = ['Temperature', 'pH', 'DO', 'Eh', 'EC',
                  'Well_depth_m', 'aquifer_code', 'zone_code']

ALL_FEATURES = HYDRO_FEATURES + GEO_FEATURES
print(f"  Features used ({len(ALL_FEATURES)}): {ALL_FEATURES}")

X = df[ALL_FEATURES].copy()
y = df['contaminated'].copy()

# drop rows ที่มี NaN ใน features หลัก
mask = X[HYDRO_FEATURES].notna().all(axis=1) & y.notna()
X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
df_clean = df[mask].reset_index(drop=True)

# เติม NaN ใน geo features ด้วย median (ถ้ามีข้อมูลบางส่วน)
for col in GEO_FEATURES:
    if X[col].isna().any():
        med = X[col].median()
        X[col] = X[col].fillna(med)
        print(f"  Filled NaN in {col} with median={med:.2f}")

print(f"  Final dataset: {len(X)} wells  |  "
      f"contaminated={y.sum()}  clean={len(y)-y.sum()}")
print("  Step 3 complete ✓")

# ============================================================
# STEP 4 — Resampling + Train/Test Split
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Resampling + Train/Test Split")
print("=" * 60)

from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score

# SMOTE — ต้องมี k_neighbors < minority class size
try:
    from imblearn.over_sampling import SMOTE
    min_class_n = y.value_counts().min()
    k = min(3, min_class_n - 1)
    if k >= 1:
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"  SMOTE applied (k={k}): "
              f"{dict(pd.Series(y_res).value_counts())}")
    else:
        print("  [SKIP SMOTE] minority class น้อยเกินไป → ใช้ original data")
        X_res, y_res = X.copy(), y.copy()
except ImportError:
    print("  [SKIP SMOTE] ไม่มี imbalanced-learn → ใช้ original data")
    X_res, y_res = X.copy(), y.copy()

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2,
    stratify=y_res, random_state=42)

print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
print("  Step 4 complete ✓")

# ============================================================
# STEP 5 — Train RandomForestClassifier
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Train Random Forest")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)
import joblib

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features='sqrt',
    class_weight='balanced',
    oob_score=True,
    n_jobs=-1,
    random_state=42)

rf.fit(X_train, y_train)

print(f"\n  OOB Score : {rf.oob_score_:.3f}")

y_pred  = rf.predict(X_test)
y_prob  = rf.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_prob)
print(f"  Test AUC  : {auc:.3f}")
print(f"\n  Classification Report:\n"
      f"{classification_report(y_test, y_pred, target_names=['Clean','Contaminated'])}")

# -- LeaveOneOut CV บน original data (สำคัญมากเพราะ n=53)
print("  Running LeaveOneOut CV (n_original)...")
rf_loo = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight='balanced', n_jobs=-1, random_state=42)
loo = LeaveOneOut()
loo_scores = cross_val_score(rf_loo, X, y, cv=loo, scoring='roc_auc')
print(f"  LOO AUC   : {loo_scores.mean():.3f} ± {loo_scores.std():.3f}")

# -- Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Clean','Contaminated'])
disp.plot(ax=ax, colorbar=False)
ax.set_title('Confusion Matrix — RF PFAS Tokyo')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f"  Saved: confusion_matrix.png")

# -- Save model
model_path = os.path.join(OUTPUT_DIR, 'rf_pfas_tokyo.pkl')
joblib.dump(rf, model_path)
print(f"  Model saved: {model_path}")
print("  Step 5 complete ✓")

# ============================================================
# STEP 6 — Feature Importance + SHAP
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Feature Importance + SHAP")
print("=" * 60)

# -- Feature importance (MDI)
imp = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2196F3' if v < imp.median() else '#F44336' for v in imp]
imp.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Importance (MDI) — RF PFAS Tokyo', fontsize=12)
ax.set_xlabel('Importance')
ax.axvline(imp.median(), color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
plt.close()
print(f"  Saved: feature_importance.png")
print(f"\n  Top 3 features:")
for fname, fval in imp.sort_values(ascending=False).head(3).items():
    print(f"    {fname:<20} {fval:.4f}")

# -- SHAP
try:
    import shap
    explainer  = shap.TreeExplainer(rf)
    shap_vals  = explainer.shap_values(X)

    # shap_vals อาจเป็น list [class0, class1] หรือ 3D array
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(sv, X, feature_names=ALL_FEATURES,
                      show=False, plot_size=None)
    plt.title('SHAP Summary — RF PFAS Tokyo')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap_summary.png")

except ImportError:
    print("  [SKIP SHAP] ติดตั้งด้วย: pip install shap")

# -- Spatial prediction raster (ถ้ามี raster)
if GEO_FEATURES:
    print("\n  Building spatial prediction raster...")
    try:
        import rasterio
        from rasterio.transform import from_bounds

        ref_path = RASTERS['DTW']
        if os.path.exists(ref_path):
            with rasterio.open(ref_path) as src:
                meta   = src.meta.copy()
                arr    = src.read(1).astype(float)
                nodata = src.nodata if src.nodata else -9999
                rows, cols = arr.shape

            # สร้าง feature grid (ใช้ค่า median สำหรับ non-spatial features)
            medians = X[HYDRO_FEATURES].median().to_dict()

            # สร้าง dummy grid สำหรับ predict (ตัวอย่าง simplified)
            print(f"  Raster size: {rows} x {cols} pixels")
            print("  (Spatial prediction ใช้เวลานาน — แนะนำ run แยกต่างหาก)")

    except Exception as e:
        print(f"  [SKIP spatial raster] {e}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"Output files → {OUTPUT_DIR}")
print("=" * 60)