import argparse
from tasks import run_binary_task, run_external_validation
from processors import (
    ProcessorBaseline4,
    ProcessorPsych,
    ProcessorHRV8,
    ProcessorData2Full,
    ProcessorBaselineMerge
)

# 你的檔案路徑
DEFAULT_XLSX_PATH = r"D:\ML_Project\dataset\data.xlsx"

def main():
    parser = argparse.ArgumentParser()
    # 定義 7 個任務選項
    parser.add_argument('--task', type=str, required=True, 
                        choices=[
                            '1_baseline_d2',   # Task 1
                            '2_ext_val_d1',    # Task 2
                            '3_psych_d2',      # Task 3
                            '4_hrv8_d2',       # Task 4
                            '5_full_d2',       # Task 5
                            '6_baseline_all',  # Task 6
                        ])
    parser.add_argument('--file', type=str, default=DEFAULT_XLSX_PATH)
    parser.add_argument('--models_dir', type=str, default=None, help='For Task 2 only')
    args = parser.parse_args()

    # ==========================
    # Task Dispatcher
    # ==========================
    
    # 1. Baseline (Data2) - 4 HRV
    if args.task == '1_baseline_d2':
        run_binary_task("Task1_Baseline_D2", args.file, 'Data2', ProcessorBaseline4)

    # 2. External Validation (Train on Task1, Test on Data1)
    elif args.task == '2_ext_val_d1':
        if not args.models_dir:
            print("❌ Task 2 需要指定 --models_dir (指向 Task 1 的模型資料夾)")
            return
        run_external_validation(args.models_dir, args.file, 'Data1', ProcessorBaseline4)

    # 3. Psych Comparison (Data2)
    elif args.task == '3_psych_d2':
        run_binary_task("Task3_Psych_D2", args.file, 'Data2', ProcessorPsych)

    # 4. Advanced HRV (Data2) - 8 HRV
    elif args.task == '4_hrv8_d2':
        run_binary_task("Task4_HRV8_D2", args.file, 'Data2', ProcessorHRV8)

    # 5. Full Data2 (Upper Bound)
    elif args.task == '5_full_d2':
        run_binary_task("Task5_Full_D2", args.file, 'Data2', ProcessorData2Full)

if __name__ == "__main__":
    main()