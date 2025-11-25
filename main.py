import argparse
import os
from tasks import run_binary_task, run_external_validation
# 🔥 記得這裡要 import DataProcessorBaseline
from processors import (
    ProcessorHRV, 
    ProcessorPsych, 
    ProcessorBaselineAll, 
    ProcessorFullV62, 
    DataProcessorBaseline
)

def main():
    parser = argparse.ArgumentParser(description="GAD Machine Learning Project Runner")
    
    # 參數設定
    parser.add_argument('--task', type=str, required=True, 
                        # 🔥 新增 'baseline' 選項
                        choices=['hrv', 'psych', 'baseline_all', 'full_v62', 'baseline', 'validate'],
                        help='選擇要執行的任務: hrv, psych, baseline_all, full_v62, baseline (HRV+Demo), validate')
    
    parser.add_argument('--file', type=str, default=r"D:\ML_Project\dataset\data.xlsx",
                        help='Excel 資料檔案路徑')
    
    parser.add_argument('--sheet', type=str, default=None,
                        help='Excel 工作表名稱 (若不指定則自動選擇預設)')
    
    parser.add_argument('--models_dir', type=str, default=None,
                        help='(僅驗證用) 模型資料夾路徑')

    args = parser.parse_args()

    # 1. HRV 任務 (HRV + Demo)
    if args.task == 'hrv':
        sheet = args.sheet if args.sheet else 'Data2'
        run_binary_task("HRV_Only", args.file, sheet, ProcessorHRV)
        
    # 2. Psych 任務 (Psych + Demo)
    elif args.task == 'psych':
        sheet = args.sheet if args.sheet else 'Data2'
        run_binary_task("Psych_Only", args.file, sheet, ProcessorPsych)

    # 3. Baseline All (HRV + Psych + Demo)
    elif args.task == 'baseline_all':
        sheet = args.sheet if args.sheet else 'Data2'
        run_binary_task("Baseline_All", args.file, sheet, ProcessorBaselineAll)

    # 4. Full V6.2 (All Data + Advanced Features + Stacking)
    elif args.task == 'full_v62':
        sheet = args.sheet if args.sheet else 'Filled_AllData'
        run_binary_task("Full_V62", args.file, sheet, ProcessorFullV62, use_stacking=True)

    # 🔥 5. Baseline (對應 test2_data2_binary.py，使用 DataProcessorBaseline)
    elif args.task == 'baseline':
        sheet = args.sheet if args.sheet else 'Data2'
        # 這裡使用 DataProcessorBaseline 類別
        run_binary_task("Baseline_HRV_Demo", args.file, sheet, DataProcessorBaseline)

    # 6. 外部驗證
    elif args.task == 'validate':
        if not args.models_dir:
            print("❌ 執行外部驗證需要指定 --models_dir (訓練好的模型資料夾)")
            return
        sheet = args.sheet if args.sheet else 'Data1'
        run_external_validation(args.models_dir, args.file, sheet)

if __name__ == "__main__":
    main()