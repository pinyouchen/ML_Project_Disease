import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class InteractiveVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Plotly

    @staticmethod
    def save_master_dashboard(output_path, all_runs_data):
        """
        將所有實驗的圖表整合為單一 HTML (支援 RWD 手機響應式)
        """
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Unified ML Dashboard (Responsive)</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { background-color: #f4f6f9; font-family: 'Segoe UI', Roboto, sans-serif; overflow-x: hidden; }
                
                /* 電腦版樣式 (預設) */
                .sidebar { 
                    height: 100vh; position: fixed; top: 0; left: 0; width: 280px; 
                    background: #212529; overflow-y: auto; padding: 20px 0; z-index: 1000; 
                    transition: all 0.3s;
                }
                .main-content { 
                    margin-left: 280px; padding: 40px; width: calc(100% - 280px); 
                    transition: all 0.3s;
                }
                .plot-container { min-height: 500px; display: flex; align-items: center; justify-content: center; }

                /* 手機版樣式 (螢幕寬度小於 768px) */
                @media (max-width: 768px) {
                    .sidebar {
                        position: relative; /* 取消固定定位 */
                        width: 100%;
                        height: auto;
                        max-height: 300px; /* 限制選單高度 */
                        padding: 10px;
                    }
                    .main-content {
                        margin-left: 0; /* 取消左邊距 */
                        width: 100%;
                        padding: 15px; /* 減少內距 */
                    }
                    .sidebar-header h4 { font-size: 1.2rem; }
                    .nav-pills .nav-link { padding: 8px 15px; font-size: 0.9rem; }
                    .plot-container { min-height: 350px; } /* 手機圖表高度縮小 */
                    
                    /* 讓 Comparison 表格橫向捲動 */
                    .card-body { overflow-x: auto; }
                }

                .sidebar-header { color: white; text-align: center; padding-bottom: 20px; border-bottom: 1px solid #495057; margin-bottom: 20px; }
                .nav-pills .nav-link { color: #dee2e6; border-radius: 0; transition: 0.3s; }
                .nav-pills .nav-link:hover { background-color: #343a40; color: #fff; padding-left: 30px; }
                .nav-pills .nav-link.active { background-color: #0d6efd; color: #fff; border-left: 5px solid #fff; }
                
                /* 手機版選單點擊效果修正 */
                @media (max-width: 768px) {
                    .nav-pills .nav-link:hover { padding-left: 15px; } /* 手機不需位移特效 */
                }

                .card { border: none; border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.05); margin-bottom: 30px; background: white; }
                .card-header { background: white; border-bottom: 1px solid #eee; padding: 15px 20px; border-radius: 12px 12px 0 0 !important; font-weight: bold; }
                
                /* 圖片樣式 */
                .img-container { text-align: center; padding: 10px; overflow: hidden; }
                .img-container img { max-width: 100%; height: auto; border-radius: 5px; cursor: zoom-in; }
                
                /* Modal */
                #imageModal .modal-content { background-color: rgba(0,0,0,0.95); border: none; }
                #imageModal .btn-close { filter: invert(1); z-index: 1050; opacity: 1; }
                #modalImage { max-height: 85vh; max-width: 100vw; object-fit: contain; }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <div class="sidebar-header">
                    <h4>📊 ML Dashboard</h4>
                    <small>Mobile Optimized</small>
                </div>
                <div class="nav flex-column nav-pills" id="v-pills-tab" role="tablist">
        """
        
        run_names = sorted(list(all_runs_data.keys()))
        for i, run_name in enumerate(run_names):
            active = "active" if i == 0 else ""
            display_name = run_name.split('_202')[0] if '_202' in run_name else run_name
            html_content += f"""
                <a class="nav-link {active}" id="v-pills-{i}-tab" data-bs-toggle="pill" 
                   href="#v-pills-{i}" role="tab">{display_name}</a>
            """
            
        html_content += """
                </div>
            </div>
            
            <div class="main-content">
                <div class="tab-content">
        """
        
        for i, run_name in enumerate(run_names):
            active = "show active" if i == 0 else ""
            run_data = all_runs_data[run_name]
            
            html_content += f"""
                <div class="tab-pane fade {active}" id="v-pills-{i}" role="tabpanel">
                    <h2 class="mb-4 fw-bold text-break">{run_name}</h2>
            """
            
            labels = list(run_data.keys())
            if "Comparison" in labels:
                labels.remove("Comparison")
                labels.insert(0, "Comparison")
            
            html_content += f"""
                    <div class="card">
                        <div class="card-header">
                            <ul class="nav nav-tabs card-header-tabs flex-nowrap overflow-auto" id="subtabs-{i}" role="tablist" style="white-space: nowrap;">
            """
            
            for j, label in enumerate(labels):
                l_active = "active" if j == 0 else ""
                html_content += f"""
                                <li class="nav-item">
                                    <button class="nav-link {l_active}" data-bs-toggle="tab" data-bs-target="#sub-{i}-{j}" type="button">{label}</button>
                                </li>
                """
            
            html_content += """
                            </ul>
                        </div>
                        <div class="card-body">
                            <div class="tab-content">
            """
            
            for j, label in enumerate(labels):
                l_active = "show active" if j == 0 else ""
                plots = run_data[label]
                
                html_content += f"""
                                <div class="tab-pane fade {l_active}" id="sub-{i}-{j}">
                                    <div class="row">
                """
                
                plot_keys = sorted(list(plots.keys()))
                
                for pname in plot_keys:
                    content = plots[pname]
                    
                    if hasattr(content, 'to_html'):
                        # Plotly: 開啟響應式 (responsive=True)
                        div_content = content.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True, 'displayModeBar': False})
                        badge = '<span class="badge bg-primary float-end">Interactive</span>'
                    elif isinstance(content, str) and len(content) > 100:
                        img_src = f"data:image/png;base64,{content}"
                        div_content = f"""
                        <div class="img-container">
                            <img src="{img_src}" alt="{pname}" 
                                 data-bs-toggle="modal" data-bs-target="#imageModal" 
                                 data-bs-src="{img_src}">
                        </div>
                        """
                        badge = '<span class="badge bg-secondary float-end">Image</span>'
                    else:
                        div_content = "No Data"
                        badge = ""

                    col_width = "col-12 col-lg-6" # 手機佔滿(12), 電腦佔半(6)
                    if "SHAP" in pname or "Comparison" in pname or "Feature" in pname:
                        col_width = "col-12" # 寬圖永遠佔滿
                    
                    html_content += f"""
                                        <div class="{col_width} mb-4">
                                            <div class="card border h-100">
                                                <div class="card-header bg-light small fw-bold text-uppercase d-flex justify-content-between align-items-center">
                                                    <span>{pname}</span> {badge}
                                                </div>
                                                <div class="card-body p-1">{div_content}</div>
                                            </div>
                                        </div>
                    """
                
                html_content += """
                                    </div>
                                </div>
                """
                
            html_content += """
                            </div>
                        </div>
                    </div>
                </div>
            """

        html_content += """
                </div>
            </div>

            <div class="modal fade" id="imageModal" tabindex="-1" aria-hidden="true">
              <div class="modal-dialog modal-dialog-centered modal-fullscreen">
                <div class="modal-content bg-dark">
                  <div class="modal-header border-0">
                    <h5 class="modal-title text-white" id="imgModalLabel">Image View</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                  </div>
                  <div class="modal-body d-flex justify-content-center align-items-center">
                    <img src="" id="modalImage" class="img-fluid" alt="Enlarged Image">
                  </div>
                </div>
              </div>
            </div>

            <script>
                var imageModal = document.getElementById('imageModal');
                imageModal.addEventListener('show.bs.modal', function (event) {
                    var button = event.relatedTarget;
                    var imgSrc = button.getAttribute('data-bs-src');
                    var imgAlt = button.getAttribute('alt');
                    var modalImg = imageModal.querySelector('#modalImage');
                    var modalTitle = imageModal.querySelector('.modal-title');
                    modalImg.src = imgSrc;
                    modalTitle.textContent = imgAlt;
                });
            </script>
        </body>
        </html>
        """
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n✨ RWD 響應式儀表板已生成: {output_path}")

    # ===========================
    # 以下繪圖函式保持不變 (功能性邏輯)
    # ===========================
    
    def get_metrics_bar(self, metrics_dict, label):
        if not metrics_dict: return None
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(metrics_dict.keys()), y=list(metrics_dict.values()),
            marker_color='#636EFA', text=[f"{v:.3f}" for v in metrics_dict.values()],
            textposition='auto'
        ))
        # Autosize 設定
        fig.update_layout(title=f'Metrics ({label})', margin=dict(l=20, r=20, t=40, b=20), autosize=True, template="plotly_white")
        return fig

    def get_radar_chart(self, metrics_dict, label):
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        categories.append(categories[0]); values.append(values[0])
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=label))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=f"Radar ({label})", margin=dict(l=40, r=40, t=40, b=40), autosize=True, template="plotly_white")
        return fig

    def get_multilabel_comparison(self, df):
        fig = px.bar(df, x="Metric", y="Value", color="Label", barmode="group", title="Metrics Comparison", template="plotly_white")
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), autosize=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig