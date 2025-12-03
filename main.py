import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTabWidget, QSplitter, 
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QSlider,
    QCheckBox, QGroupBox, QScrollArea, QProgressDialog, QMessageBox,
    QDialog, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QAction

# --- Matplotlib 集成 ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import style as mpl_style

# 设置 Matplotlib 风格，使其更贴合现代 UI
plt.style.use('ggplot') 

def apply_modern_dark_theme(app):
    """应用现代深色主题样式表（修正表格文字看不清的问题）"""
    dark_bg = "#2b2b2b"
    darker_bg = "#1e1e1e"
    alternate_bg = "#353535"
    accent_color = "#3daee9"
    text_color = "#f0f0f0"
    border_color = "#555555"
    style_sheet = f"""
    QWidget {{
        background-color: {dark_bg};
        color: {text_color};
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 10pt;
    }}
    QFrame {{
        border: none;
    }}
    QSplitter::handle {{
        background-color: {border_color};
        width: 2px;
    }}
    QPushButton {{
        background-color: #3e3e42;
        border: 1px solid {border_color};
        border-radius: 6px;
        padding: 6px 12px;
        color: white;
    }}
    QPushButton:hover {{
        background-color: {accent_color};
        border-color: {accent_color};
    }}
    QPushButton:pressed {{
        background-color: #2c8bc2;
    }}
    QTableWidget {{
        background-color: {darker_bg};
        alternate-background-color: {alternate_bg};
        gridline-color: {border_color};
        border: 1px solid {border_color};
        border-radius: 4px;
        color: {text_color};
        selection-background-color: {accent_color};
        selection-color: white;
    }}
    QTableWidget::item {{
        padding: 5px;
        border: none;
    }}
    QHeaderView::section {{
        background-color: #333333;
        color: {text_color};
        padding: 6px;
        border: none;
        border-bottom: 1px solid {accent_color};
        border-right: 1px solid {border_color};
        font-weight: bold;
    }}
    QTableCornerButton::section {{
        background-color: #333333;
        border: 1px solid {border_color};
    }}
    QGroupBox {{
        border: 1px solid {border_color};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        left: 10px;
        color: {accent_color};
        font-weight: bold;
    }}
    QTabWidget::pane {{
        border: 1px solid {border_color};
        border-radius: 4px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: {dark_bg};
        border: 1px solid {border_color};
        padding: 6px 12px;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: #aaaaaa;
    }}
    QTabBar::tab:selected {{
        background: {darker_bg};
        color: {accent_color};
        border-bottom: 1px solid {darker_bg};
    }}
    QSlider::groove:horizontal {{
        border: 1px solid {border_color};
        height: 6px;
        background: #1e1e1e;
        margin: 2px 0;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {accent_color};
        border: 1px solid {accent_color};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    """
    app.setStyleSheet(style_sheet)

# ==========================================
# 1. 简化的逻辑层 (Model) - 方便直接运行
# ==========================================
class ImageAnalyzer:
    def __init__(self):
        self.original_image = None # BGR format
        self.processed_image = None
    
    def load_image(self, filepath):
        # 解决中文路径问题
        self.original_image = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        return self.original_image is not None

    def get_canny_edges(self, low, high):
        if self.original_image is None: return None
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        return edges

    def calculate_stats(self):
        """计算 RGB 统计数据用于演示"""
        if self.original_image is None: return []
        
        stats = []
        # 分离通道 BGR -> RGB
        chans = cv2.split(self.original_image)
        colors = ['Blue', 'Green', 'Red'] # OpenCV 默认 BGR
        
        for i, color in enumerate(colors):
            c_data = chans[i]
            stats.append({
                "Channel": color,
                "Min": np.min(c_data),
                "Max": np.max(c_data),
                "Mean": f"{np.mean(c_data):.2f}",
                "Std": f"{np.std(c_data):.2f}"
            })
        return stats

    def get_histogram_data(self):
        if self.original_image is None: return None
        colors = ('b', 'g', 'r')
        data = []
        for i, col in enumerate(colors):
            hist = cv2.calcHist([self.original_image], [i], None, [256], [0, 256])
            data.append((col, hist))
        return data

# ==========================================
# 2. 自定义组件
# ==========================================
class MplCanvas(FigureCanvas):
    """Matplotlib 画布控件"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e1e')
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class ImageLabel(QLabel):
    """支持保持比例缩放的图片显示控件"""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("No Image Loaded")
        self.setStyleSheet("border: 2px dashed #aaa; color: #aaa; font-size: 16px;")
        self.setScaledContents(False) # 我们自己控制缩放

    def set_cv_image(self, cv_img):
        if cv_img is None: return
        
        # 转换 OpenCV (BGR/Gray) -> QImage
        if len(cv_img.shape) == 2: # 灰度图
            h, w = cv_img.shape
            bytes_per_line = w
            q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        else: # 彩色图
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            # OpenCV 是 BGR, Qt 需要 RGB
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
        # 缩放图片以适应窗口，保持比例
        pixmap = QPixmap.fromImage(q_img)
        self.setPixmap(pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))
        self.setStyleSheet("") # 移除虚线边框

class ComparisonWindow(QDialog):
    def __init__(self, image_paths, analyzer, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.analyzer = analyzer
        self.setWindowTitle(f"深度对比分析 - 选定 {len(image_paths)} 张图像")
        self.resize(1000, 600)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.image_paths))
        headers = [os.path.basename(p) for p in self.image_paths]
        self.table.setHorizontalHeaderLabels(headers)
        metrics = [
            "边缘数量 (Edge Count)",
            "亮度均值 (Gray Mean)",
            "亮度标准差 (Gray Std)",
            "红通道均值 (R Mean)",
            "蓝通道均值 (B Mean)"
        ]
        self.table.setRowCount(len(metrics))
        self.table.setVerticalHeaderLabels(metrics)
        left_layout.addWidget(QLabel("<b>📊 数据横向对比</b>"))
        left_layout.addWidget(self.table)

        right_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(QLabel("<b>📈 亮度直方图叠加 (Luminance Overlay)</b>"))
        right_layout.addWidget(self.canvas)

        layout.addLayout(left_layout, stretch=1)
        layout.addLayout(right_layout, stretch=1)

        self.perform_comparison()

    def perform_comparison(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Grayscale Distribution Comparison")
        self.canvas.axes.set_xlabel("Pixel Intensity")
        self.canvas.axes.set_ylabel("Frequency")

        line_styles = ['-', '--', '-.', ':']

        for col_idx, path in enumerate(self.image_paths):
            file_name = os.path.basename(path)
            if not self.analyzer.load_image(path):
                continue
            img = self.analyzer.original_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            edges = cv2.Canny(gray, 50, 150)
            edge_count = np.count_nonzero(edges)
            b_mean = np.mean(img[:, :, 0])
            r_mean = np.mean(img[:, :, 2])
            self.table.setItem(0, col_idx, QTableWidgetItem(str(edge_count)))
            self.table.setItem(1, col_idx, QTableWidgetItem(f"{mean_val:.2f}"))
            self.table.setItem(2, col_idx, QTableWidgetItem(f"{std_val:.2f}"))
            self.table.setItem(3, col_idx, QTableWidgetItem(f"{r_mean:.2f}"))
            self.table.setItem(4, col_idx, QTableWidgetItem(f"{b_mean:.2f}"))
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            style = line_styles[col_idx % len(line_styles)]
            self.canvas.axes.plot(hist, label=file_name, linestyle=style, linewidth=2)
        self.canvas.axes.legend()
        self.canvas.draw()
from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
import pandas as pd

class BatchWorker(QThread):
    # 定义信号：
    # progress_updated: 传回 (当前进度百分比, 当前处理的文件名)
    progress_updated = pyqtSignal(int, str)
    # finished: 传回 (处理结果列表)
    task_finished = pyqtSignal(list)
    # error_occurred: 传回错误信息
    error_occurred = pyqtSignal(str)

    def __init__(self, file_paths, analyzer_instance, canny_params):
        super().__init__()
        self.file_paths = file_paths
        self.analyzer = analyzer_instance
        self.canny_params = canny_params # {'th1': 50, 'th2': 150}
        self.is_running = True # 用于控制取消标志

    def run(self):
        """线程的主入口"""
        results = []
        total_files = len(self.file_paths)
        
        for i, file_path in enumerate(self.file_paths):
            if not self.is_running:
                break # 用户点击了取消

            file_name = os.path.basename(file_path)
            
            # 发送进度信号
            progress_percent = int((i / total_files) * 100)
            self.progress_updated.emit(progress_percent, f"正在处理: {file_name}")

            try:
                # 1. 加载图像 (复用 Analyzer 的逻辑)
                if self.analyzer.load_image(file_path):
                    # 2. 计算统计数据 (模拟全色彩空间)
                    # 注意：为了性能，这里我们简化调用，实际项目中应调用 analyzer.calculate_stats(space='all')
                    stats = self.analyzer.calculate_stats() 
                    
                    # 3. Canny 计数
                    edges = self.analyzer.get_canny_edges(
                        self.canny_params['th1'], 
                        self.canny_params['th2']
                    )
                    edge_count = np.count_nonzero(edges)

                    # 4. 整理单张图片的数据行
                    row_data = {
                        "File": file_name,
                        "Edge_Count": edge_count
                    }
                    # 展平 stats 列表到字典中
                    for stat in stats:
                        ch = stat['Channel']
                        row_data[f"{ch}_Mean"] = stat['Mean']
                        row_data[f"{ch}_Std"] = stat['Std']
                    
                    results.append(row_data)
                else:
                    print(f"Failed to load {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

            # 模拟一点耗时，不然处理太快看不清进度条 (实际使用时去掉这行)
            time.sleep(0.05) 

        # 循环结束，发送完成信号
        self.task_finished.emit(results)

    def stop(self):
        self.is_running = False
# ==========================================
# 3. 主窗口 GUI (View & Controller)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = ImageAnalyzer()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("OpenCV 图像分析工具 Pro")
        self.resize(1280, 800)

        # 主布局容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 使用 QSplitter 实现可拖拽调整大小的三栏布局
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- 左侧面板：配置区 ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(250)
        
        # 1. 加载区
        grp_load = QGroupBox("图像加载")
        load_layout = QVBoxLayout()
        self.btn_load = QPushButton("📂 打开单张图像")
        self.btn_load.clicked.connect(self.load_image_dialog)
        self.btn_batch = QPushButton("📚 批量处理")
        self.btn_batch.setEnabled(True)
        self.btn_batch.clicked.connect(self.start_batch_process)
        self.btn_compare = QPushButton("⚖️ 多图对比分析")
        self.btn_compare.clicked.connect(self.open_comparison_dialog)
        load_layout.addWidget(self.btn_load)
        load_layout.addWidget(self.btn_batch)
        load_layout.addWidget(self.btn_compare)
        grp_load.setLayout(load_layout)
        
        # 2. Canny 参数区
        grp_canny = QGroupBox("Canny 边缘检测参数")
        canny_layout = QVBoxLayout()
        
        self.lbl_th1 = QLabel("低阈值: 50")
        self.slider_th1 = QSlider(Qt.Orientation.Horizontal)
        self.slider_th1.setRange(0, 255)
        self.slider_th1.setValue(50)
        self.slider_th1.valueChanged.connect(self.update_canny_preview)
        
        self.lbl_th2 = QLabel("高阈值: 150")
        self.slider_th2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_th2.setRange(0, 255)
        self.slider_th2.setValue(150)
        self.slider_th2.valueChanged.connect(self.update_canny_preview)

        canny_layout.addWidget(self.lbl_th1)
        canny_layout.addWidget(self.slider_th1)
        canny_layout.addWidget(self.lbl_th2)
        canny_layout.addWidget(self.slider_th2)
        grp_canny.setLayout(canny_layout)

        # 3. 导出区
        grp_export = QGroupBox("操作")
        export_layout = QVBoxLayout()
        self.btn_export = QPushButton("💾 导出分析结果")
        export_layout.addWidget(self.btn_export)
        grp_export.setLayout(export_layout)

        left_layout.addWidget(grp_load)
        left_layout.addWidget(grp_canny)
        left_layout.addStretch() # 弹簧，顶上去
        left_layout.addWidget(grp_export)

        # --- 中间面板：预览区 ---
        center_panel = QTabWidget()
        self.view_original = ImageLabel()
        self.view_edges = ImageLabel()
        center_panel.addTab(self.view_original, "原始图像 (Original)")
        center_panel.addTab(self.view_edges, "边缘检测 (Canny)")

        # --- 右侧面板：数据区 ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMinimumWidth(350)

        # 1. 统计表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["通道", "Min", "Max", "Mean", "Std"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setFixedHeight(200)

        # 2. 直方图
        self.hist_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
        right_layout.addWidget(QLabel("<b>📊 量化指标 (RGB)</b>"))
        right_layout.addWidget(self.table)
        right_layout.addWidget(QLabel("<b>📈 直方图分布</b>"))
        right_layout.addWidget(self.hist_canvas)
        
        self.init_hover_tools()

        # 添加到 Splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        
        # 设置初始比例 (左:中:右)
        splitter.setSizes([250, 600, 400])

        main_layout.addWidget(splitter)

    # --- 逻辑处理槽函数 ---

    def load_image_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开图像', '.', "Image files (*.jpg *.png *.bmp)")
        if fname:
            if self.analyzer.load_image(fname):
                # 1. 显示原图
                self.view_original.set_cv_image(self.analyzer.original_image)
                # 2. 触发分析
                self.update_stats()
                self.update_canny_preview()
                self.update_histogram()

    def update_canny_preview(self):
        """实时更新 Canny 边缘检测"""
        th1 = self.slider_th1.value()
        th2 = self.slider_th2.value()
        
        # 更新 Label 文字
        self.lbl_th1.setText(f"低阈值: {th1}")
        self.lbl_th2.setText(f"高阈值: {th2}")

        edges = self.analyzer.get_canny_edges(th1, th2)
        if edges is not None:
            self.view_edges.set_cv_image(edges)

    def update_stats(self):
        """更新统计表格"""
        stats = self.analyzer.calculate_stats()
        self.table.setRowCount(len(stats))
        for row, data in enumerate(stats):
            self.table.setItem(row, 0, QTableWidgetItem(str(data["Channel"])))
            self.table.setItem(row, 1, QTableWidgetItem(str(data["Min"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(data["Max"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(data["Mean"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(data["Std"])))

    def update_histogram(self):
        """绘制直方图"""
        hist_data = self.analyzer.get_histogram_data()
        if hist_data is None: return

        self.hist_canvas.axes.cla() # 清除旧图
        self.hist_canvas.axes.set_title("RGB Histogram")
        self.hist_canvas.axes.set_xlabel("Pixel Intensity")
        self.hist_canvas.axes.set_ylabel("Count")
        
        for color, hist in hist_data:
            self.hist_canvas.axes.plot(hist.ravel(), color=color, alpha=0.7)
        
        self.hist_canvas.axes.set_xlim([0, 256])
        self.hist_canvas.draw()

    # ==========================================
    # 新增：直方图交互模块
    # ==========================================
    def init_hover_tools(self):
        """初始化直方图的悬停提示框"""
        self.hist_annot = self.hist_canvas.axes.annotate(
            "",
            xy=(0,0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#2b2b2b", ec="#3daee9", lw=1, alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#3daee9"),
            color="#f0f0f0"
        )
        self.hist_annot.set_visible(False)
        self.hist_canvas.mpl_connect("motion_notify_event", self.on_histogram_hover)

    def on_histogram_hover(self, event):
        """处理鼠标在直方图上的移动事件"""
        if event.inaxes == self.hist_canvas.axes:
            x_idx = int(round(event.xdata))
            if x_idx < 0 or x_idx > 255:
                return
            tooltip_text = f"强度 (Intensity): {x_idx}\n"
            tooltip_text += "--------------------\n"
            found_data = False
            for line in self.hist_canvas.axes.lines:
                y_data = line.get_ydata()
                if x_idx < len(y_data):
                    val = y_data[x_idx]
                    color_code = line.get_color()
                    channel_name = "Channel"
                    if color_code == 'r': channel_name = "Red"
                    elif color_code == 'g': channel_name = "Green"
                    elif color_code == 'b': channel_name = "Blue"
                    tooltip_text += f"• {channel_name}: {int(val)}\n"
                    found_data = True
            if found_data:
                self.hist_annot.xy = (x_idx, event.ydata)
                self.hist_annot.set_text(tooltip_text.strip())
                self.hist_annot.set_visible(True)
                self.hist_canvas.draw_idle()
            else:
                if self.hist_annot.get_visible():
                    self.hist_annot.set_visible(False)
                    self.hist_canvas.draw_idle()
        else:
            if hasattr(self, 'hist_annot') and self.hist_annot.get_visible():
                self.hist_annot.set_visible(False)
                self.hist_canvas.draw_idle()

    def start_batch_process(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图像用于批量分析", ".", "Image files (*.jpg *.png *.bmp *.tif)"
        )
        if not files:
            return

        canny_params = {
            'th1': self.slider_th1.value(),
            'th2': self.slider_th2.value()
        }

        self.progress_dialog = QProgressDialog("准备开始...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("批量处理中")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)

        self.worker = BatchWorker(files, self.analyzer, canny_params)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.task_finished.connect(self.batch_finished)
        self.progress_dialog.canceled.connect(self.worker.stop)
        self.worker.start()

    def update_progress(self, percent, message):
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(message)

    def batch_finished(self, results):
        self.progress_dialog.setValue(100)
        self.progress_dialog.close()

        if not results:
            QMessageBox.warning(self, "提示", "处理被取消或未生成数据。")
            return

        reply = QMessageBox.question(
            self, "完成",
            f"批量处理完成！共处理 {len(results)} 张图像。\n是否立即导出结果到 CSV？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.save_batch_results(results)

    def save_batch_results(self, results):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "batch_results.csv", "CSV Files (*.csv)")
        if save_path:
            try:
                df = pd.DataFrame(results)
                cols = ['File', 'Edge_Count'] + [c for c in df.columns if c not in ['File', 'Edge_Count']]
                df = df[cols]
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"数据已保存至:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def open_comparison_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 2-5 张图像进行对比", ".", "Image files (*.jpg *.png *.bmp)"
        )
        if not files:
            return
        if len(files) < 2:
            QMessageBox.warning(self, "提示", "请至少选择 2 张图片进行对比。")
            return
        if len(files) > 5:
            QMessageBox.warning(self, "提示", "为了保证图表可读性，建议一次最多对比 5 张图片。")
            return
        compare_win = ComparisonWindow(files, self.analyzer, parent=self)
        compare_win.exec()

    # 窗口大小改变时触发，用于重绘图片适应大小
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 简单触发一下更新显示，保证图片缩放正确
        if self.analyzer.original_image is not None:
             self.view_original.set_cv_image(self.analyzer.original_image)
             self.update_canny_preview()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_modern_dark_theme(app)
    
    # 设置全局字体大小，适应不同分辨率
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())