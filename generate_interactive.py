import os
import argparse
import pandas as pd
import base64
import glob
from interactive_visualization import InteractiveVisualizer

def encode_image_to_base64(image_path):
    """將圖片檔案轉為 Base64 字串"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except:
        return None

def process_all_runs_hybrid(parent_dir, output_path):
    print(f"🚀 開始生成混合型儀表板 (來源: {parent_dir})")
    print("   ℹ️  模式: 讀取 Excel 數據 + 嵌入原始 PNG 圖片 (不需資料集)")
    
    master_data = {}
    viz = InteractiveVisualizer()
    
    # 取得所有 run 資料夾
    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    for folder_name in subdirs:
        run_dir = os.path.join(parent_dir, folder_name)
        plots_dir = os.path.join(run_dir, "plots")
        
        # 如果連 plots 資料夾都沒有，可能不是有效的 run
        if not os.path.exists(plots_dir): continue
        
        print(f"\n📂 讀取: {folder_name}")
        run_data = {} # {Label: {PlotName: Fig/Base64}}
        
        # 1. [互動圖] 讀取 Excel 生成 Metrics Bar & Radar
        summary_path = os.path.join(run_dir, "Results_Summary.xlsx")
        if os.path.exists(summary_path):
            try:
                df_summary = pd.read_excel(summary_path)
                metrics_cols = ['F1(avg)', 'AUC(avg)', 'ACC(avg)', 'P(avg)', 'R(avg)']
                comp_data = []
                
                for _, row in df_summary.iterrows():
                    label = row['Label']
                    if label not in run_data: run_data[label] = {}
                    
                    # 生成互動式 Metrics Bar
                    metrics_dict = {}
                    for col in metrics_cols:
                        if col in df_summary.columns:
                            clean = col.replace('(avg)', '')
                            metrics_dict[clean] = row[col]
                            comp_data.append({'Label': label, 'Metric': clean, 'Value': row[col]})
                            
                    run_data[label]["01_Interactive_Metrics"] = viz.get_metrics_bar(metrics_dict, label)
                    run_data[label]["02_Interactive_Radar"] = viz.get_radar_chart(metrics_dict, label)
                
                # 生成 Comparison 互動圖
                if comp_data:
                    if "Comparison" not in run_data: run_data["Comparison"] = {}
                    run_data["Comparison"]["Interactive_Comparison"] = viz.get_multilabel_comparison(pd.DataFrame(comp_data))
            except: pass

        # 2. [靜態圖] 掃描 plots 資料夾下的 PNG
        # plots/SSD/*.png, plots/Summary_Comparison/*.png
        
        # 遍歷 plots 底下的子資料夾 (Label 名稱)
        for label_folder in os.listdir(plots_dir):
            label_path = os.path.join(plots_dir, label_folder)
            if os.path.isdir(label_path):
                # 處理標籤名稱 (例如 Summary_Comparison -> Comparison)
                dict_key = "Comparison" if "Comparison" in label_folder else label_folder
                if dict_key not in run_data: run_data[dict_key] = {}
                
                # 掃描所有 png
                png_files = glob.glob(os.path.join(label_path, "*.png"))
                for png_path in png_files:
                    fname = os.path.basename(png_path).replace(".png", "")
                    # 移除檔名中重複的 label (例如 ROC_Curve_SSD -> ROC Curve)
                    clean_name = fname.replace(f"_{label_folder}", "").replace("_", " ")
                    
                    base64_str = encode_image_to_base64(png_path)
                    if base64_str:
                        # 加個前綴讓它排在互動圖後面
                        run_data[dict_key][f"Img: {clean_name}"] = base64_str
        
        if run_data:
            master_data[folder_name] = run_data

    # 3. 輸出
    InteractiveVisualizer.save_master_dashboard(output_path, master_data)

def main():
    parser = argparse.ArgumentParser(description="Generate Hybrid Dashboard")
    parser.add_argument('--parent_dir', type=str, required=True, help='runs 資料夾路徑')
    
    args = parser.parse_args()
    
    if os.path.exists(args.parent_dir):
        output_path = os.path.join(args.parent_dir, "All_Experiments_Dashboard.html")
        process_all_runs_hybrid(args.parent_dir, output_path)
    else:
        print(f"❌ 資料夾不存在: {args.parent_dir}")

if __name__ == "__main__":
    main()