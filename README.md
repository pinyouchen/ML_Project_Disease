# GAD & Mental Health Classification Framework

這是一個針對 **廣泛性焦慮症 (GAD)** 及相關精神疾病 (SSD, MDD, Panic) 進行二元分類 (Disease vs. Healthy) 的機器學習框架。本專案採用嚴謹的特徵工程、多模型集成 (Ensemble/Stacking) 以及符合頂尖期刊標準 (Top-Journal Quality) 的視覺化分析。

## 🚀 專案特色

* **多模態特徵整合**：支援 HRV 生理訊號、人口學變數 (Age, Sex, BMI)、心理量表及臨床特徵的深度整合。
* **穩健的訓練流程**：
    * **Stratified 5-Fold Cross-Validation**：確保驗證結果可靠。
    * **Isolation Forest**：自動偵測並排除訓練集中的異常值 (固定隨機性以確保重現)。
    * **動態採樣策略**：整合 SMOTE, BorderlineSMOTE, ADASYN 處理資料不平衡。
    * **模型集成**：結合 XGBoost, LightGBM, Random Forest, ExtraTrees 等模型，並實作 Stacking 與 Top-3 Ensemble。
* **頂級視覺化分析**：
    * **Global OOF SHAP**：基於全體樣本的特徵重要性解釋 (強制使用樹模型解析)。
    * **ROC & PR Curves with CI**：顯示 5-Fold 的平均曲線與信賴區間 (±1 std)。
    * **PCA 散佈圖**：視覺化資料分佈與類別邊界。
    * **雷達圖 (Radar Chart)**：多維度效能指標展示。
    * **聚合混淆矩陣**：展示整體的分類準確度。

## 📂 檔案結構

```text
GAD_ML_Project/
│
├── main.py             # 程式入口：接收參數並啟動對應任務
├── tasks.py            # 任務邏輯：定義 Cross-Validation 流程、資料清洗、呼叫訓練與繪圖
├── model_trainer.py    # 模型核心：定義分類器參數 (V6.12)、採樣策略、Stacking 邏輯
├── processors.py       # 資料處理：特徵工程 (V6.2)、缺失值填補、標準化
├── visualization.py    # 繪圖模組：產生符合期刊標準的高解析度圖表 (含 PCA, SHAP)
├── utils.py            # 通用工具：評估指標 (Spec/NPV)、模型保存與載入
│
├── data.xlsx           # (需自行準備) 原始資料 Excel 檔
└── runs/               # (自動生成) 實驗輸出結果，包含模型與圖表
🛠️ 環境安裝
請確保您的環境已安裝 Python 3.8+，並安裝以下套件：

Bash

pip install numpy pandas scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn openpyxl
⚡ 快速開始 (Usage)
執行完整流程 (Full V6.2)
這是目前最強大的版本，包含完整的特徵工程、Isolation Forest 異常移除、以及所有的視覺化圖表。

在終端機 (Terminal) 執行：

1. 執行 HRV Baseline (對應 test2_data2_binary_hrv.py):
python main.py --task hrv --file "D:\ML_Project\dataset\data.xlsx" --sheet "Data2"

2. 執行 Psych Baseline (對應 test2_data2_binary_psych.py):
python main.py --task psych --file "D:\ML_Project\dataset\data.xlsx" --sheet "Data2"

3. 執行 Baseline All (HRV+Demo+Psych) (對應 test2_data2_binary_all.py):
python main.py --task baseline_all --file "D:\ML_Project\dataset\data.xlsx" --sheet "Data2"

4. 執行 Full V6.2 (完整特徵) (對應 test2_alldata_binary_update.py):
python main.py --task full_v62 --file "D:\ML_Project\dataset\data.xlsx" --sheet "Filled_AllData"

5.執行 HRV(4個特徵) + Demo baseline (對應 test2_alldata_binary.py):
python main.py --task baseline --file "D:\ML_Project\dataset\data.xlsx" --sheet "Data2"

6. 執行外部驗證 (對應 external_validate_A_Data1.py): (你需要先跑過上面的訓練，拿到 runs/xxx/models 的路徑)
python main.py --task validate --file "D:\ML_Project\dataset\data.xlsx" --sheet "Data1" --models_dir "D:\ML_Project\runs\Baseline_HRV_Demo_20251125_162744\models"

參數說明：

--task: 指定任務模式 (full_v62 為建議選項，另有 hrv, psych, baseline_all)。

--file: 指定 Excel 資料檔的完整路徑。

--sheet: 指定 Excel 工作表名稱。

📊 輸出結果 (Outputs)
程式執行完畢後，會在 runs/ 資料夾下產生以時間戳記命名的目錄 (例如 Run_v612_Full..._20251125_120000)，內容包括：

Results_Summary.xlsx：包含每個 Label 的最佳模型參數、5-Fold 平均效能 (F1, AUC, Acc, Spec, NPV) 等詳細數據。

models/：儲存每個 Label 訓練出的最佳模型 (.joblib) 與對應的預處理器。

plots/：

SSD/, MDD/, Panic/, GAD/：各疾病的獨立圖表 (SHAP Summary, PCA Scatter, ROC/PR Curve, Confusion Matrix)。

Summary_Comparison/：所有疾病的綜合比較圖 (ROC 比較, PR 比較, Metrics 比較)。

📝 注意事項
SHAP 圖表：程式會優先抓取 XGBoost 或 LightGBM 模型來計算 SHAP 值，以確保解釋性圖表能順利產出。

效能重現性：已固定 random_state=42 並將 IsolationForest 的 n_jobs 設為 1 以消除平行運算的隨機性，確保結果穩定。


互動式可視化
python generate_interactive.py --parent_dir "D:\ML_Project\runs"