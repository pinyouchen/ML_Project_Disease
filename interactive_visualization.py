import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

class InteractiveVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Plotly

    @staticmethod
    def save_master_dashboard(output_path, all_runs_data):
        """
        將所有實驗的圖表整合為單一 HTML，並支援圖片點擊放大
        """
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Unified ML Dashboard (Hybrid + Zoom)</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { background-color: #f4f6f9; font-family: 'Segoe UI', Roboto, sans-serif; overflow-x: hidden; }
                .sidebar { height: 100vh; position: fixed; top: 0; left: 0; width: 280px; background: #212529; overflow-y: auto; padding: 20px 0; z-index: 1000; }
                .sidebar-header { color: white; text-align: center; padding-bottom: 20px; border-bottom: 1px solid #495057; margin-bottom: 20px; }
                .nav-pills .nav-link { color: #dee2e6; padding: 12px 25px; border-radius: 0; transition: 0.3s; }
                .nav-pills .nav-link:hover { background-color: #343a40; color: #fff; padding-left: 30px; }
                .nav-pills .nav-link.active { background-color: #0d6efd; color: #fff; border-left: 5px solid #fff; }
                .main-content { margin-left: 280px; padding: 40px; width: calc(100% - 280px); }
                .card { border: none; border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.05); margin-bottom: 30px; background: white; }
                .card-header { background: white; border-bottom: 1px solid #eee; padding: 15px 20px; border-radius: 12px 12px 0 0 !important; font-weight: bold; }
                
                /* 圖片容器樣式 (小圖) */
                .img-container { text-align: center; padding: 10px; overflow: hidden; }
                .img-container img { 
                    max-width: 100%; height: auto; border-radius: 5px; transition: transform 0.3s; 
                    cursor: zoom-in; /* 滑鼠游標變放大鏡 */
                }
                .img-container img:hover { transform: scale(1.02); }
                
                /* Modal 樣式 (放大後的圖) */
                #imageModal .modal-content { background-color: rgba(0,0,0,0.9); border: none; }
                #imageModal .btn-close { filter: invert(1); z-index: 1050; } /* 白色關閉按鈕 */
                #modalImage { max-height: 90vh; object-fit: contain; } /* 防止圖片超過螢幕高度 */

                .plot-container { min-height: 500px; display: flex; align-items: center; justify-content: center; }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <div class="sidebar-header">
                    <h4>📊 ML Dashboard</h4>
                    <small>Integrated Report</small>
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
                    <h2 class="mb-4 fw-bold">{run_name}</h2>
            """
            
            labels = list(run_data.keys())
            if "Comparison" in labels:
                labels.remove("Comparison")
                labels.insert(0, "Comparison")
            
            html_content += f"""
                    <div class="card">
                        <div class="card-header">
                            <ul class="nav nav-tabs card-header-tabs" id="subtabs-{i}" role="tablist">
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
                        # Plotly Figure (互動)
                        div_content = content.to_html(full_html=False, include_plotlyjs=False)
                        badge = '<span class="badge bg-primary float-end">Interactive</span>'
                    elif isinstance(content, str) and len(content) > 100:
                        # Base64 Image (靜態) -> 🔥 修改這裡加入 Modal 觸發屬性
                        img_src = f"data:image/png;base64,{content}"
                        div_content = f"""
                        <div class="img-container">
                            <img src="{img_src}" alt="{pname}" 
                                 data-bs-toggle="modal" data-bs-target="#imageModal" 
                                 data-bs-src="{img_src}">
                        </div>
                        """
                        badge = '<span class="badge bg-secondary float-end" style="cursor:zoom-in">Click to Zoom</span>'
                    else:
                        div_content = "No Data"
                        badge = ""

                    col_width = "col-md-12" if "SHAP" in pname or "Comparison" in pname or "Feature" in pname else "col-md-6"
                    
                    html_content += f"""
                                        <div class="{col_width} mb-4">
                                            <div class="card border h-100">
                                                <div class="card-header bg-light small fw-bold text-uppercase">
                                                    {pname} {badge}
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

        # 🔥 在 HTML 底部加入 Modal 結構與 Javascript
        html_content += """
                </div>
            </div>

            <div class="modal fade" id="imageModal" tabindex="-1" aria-hidden="true">
              <div class="modal-dialog modal-dialog-centered modal-xl">
                <div class="modal-content">
                  <div class="modal-body p-0 text-center position-relative">
                    <button type="button" class="btn-close position-absolute top-0 end-0 m-3" data-bs-dismiss="modal" aria-label="Close"></button>
                    <img src="" id="modalImage" class="img-fluid" alt="Enlarged Image">
                  </div>
                </div>
              </div>
            </div>

            <script>
                var imageModal = document.getElementById('imageModal');
                imageModal.addEventListener('show.bs.modal', function (event) {
                    // Button (image) that triggered the modal
                    var button = event.relatedTarget;
                    // Extract info from data-bs-* attributes
                    var imgSrc = button.getAttribute('data-bs-src');
                    var imgAlt = button.getAttribute('alt');
                    // Update the modal's content.
                    var modalImg = imageModal.querySelector('#modalImage');
                    modalImg.src = imgSrc;
                    modalImg.alt = imgAlt;
                });
            </script>

        </body>
        </html>
        """
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n✨ 混合型 HTML 儀表板 (含放大功能) 已生成: {output_path}")

    # 純數據繪圖 (保持不變)
    def get_metrics_bar(self, metrics_dict, label):
        if not metrics_dict: return None
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(metrics_dict.keys()), y=list(metrics_dict.values()),
            marker_color='#636EFA', text=[f"{v:.3f}" for v in metrics_dict.values()],
            textposition='auto'
        ))
        fig.update_layout(title=f'Performance Metrics ({label})', yaxis=dict(range=[0, 1.05]), template="plotly_white", height=400)
        return fig

    def get_radar_chart(self, metrics_dict, label):
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        categories.append(categories[0]); values.append(values[0])
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=label))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=f"Radar ({label})", template="plotly_white", height=400)
        return fig

    def get_multilabel_comparison(self, df):
        fig = px.bar(df, x="Metric", y="Value", color="Label", barmode="group", title="Metrics Comparison", template="plotly_white")
        return fig