基于YOLOv8和pyqt5的织物布料检测系统
@[toc]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/522747120d274eb5989eed4c72622a70.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/64d09039e4db491cac43aed9e7c58b9a.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5809a9d7170040eb87a9584d2289f8f6.png)
通过yolov8训练织物布料数据集

基于YOLOv8和PyQt5的织物布料缺陷检测系统。为了实现这个系统，我们需要编写详细的代码来支持图像加载、缺陷检测、结果显示等功能。

仅供参考

下面将提供一个简化的示例代码框架，包括主要模块和功能。

### 项目结构

```
fabric_defect_detection/
├── datasets/               # 数据集（用于训练模型）
│   ├── images/             # 图像文件
│   └── labels/             # 标注文件
├── models/                 # 模型定义
│   └── yolov8.py           # YOLOv8 模型
├── utils/                  # 工具函数
│   ├── db_utils.py         # 数据库操作工具
│   └── config_utils.py     # 配置文件管理工具
├── app.py                  # 主程序（PyQt5界面）
└── config.yaml             # 系统配置文件
```

### 安装依赖

在终端中运行以下命令安装所需库：

```bash
pip install ultralytics opencv-python pyqt5 pandas numpy dlib
```

### 数据库设计

使用 SQLite 存储检测结果。

#### 数据库表结构

```sql
CREATE TABLE defects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    defect_type TEXT NOT NULL,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER,
    confidence REAL
);
```

### 主程序 (`app.py`)

#### 界面设计

```python
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from utils.db_utils import DatabaseManager
from utils.config_utils import ConfigManager

class FabricDefectDetection(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.db = DatabaseManager("defects.db")
        self.config = ConfigManager("config.yaml")
        self.model = YOLO('yolov8n.pt')  # 加载 YOLOv8 模型
        self.image_path = None

    def initUI(self):
        self.setWindowTitle("织物布料缺陷检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("请选择图片")
        self.label.setAlignment(Qt.AlignCenter)

        self.button_load = QPushButton("选择图片", self)
        self.button_load.clicked.connect(self.load_image)

        self.button_detect = QPushButton("开始检测", self)
        self.button_detect.clicked.connect(self.detect_defects)

        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["序号", "文件路径", "类别", "置信度", "坐标位置"])

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button_load)
        layout.addWidget(self.button_detect)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        """加载图片"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp)", options=options)
        if file_name:
            self.image_path = file_name
            img = cv2.imread(file_name)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.label.setPixmap(QPixmap.fromImage(q_img))

    def detect_defects(self):
        """进行缺陷检测"""
        if not self.image_path:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return

        img = cv2.imread(self.image_path)
        results = self.model(img)
        annotated_frame = results[0].plot()

        # 将 OpenCV 图像转换为 PyQt 图像
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_img))

        # 提取检测结果并保存到数据库
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confidences, classes):
                x_min, y_min, x_max, y_max = map(int, box)
                defect_type = self.config.get_defect_name(cls.item())
                self.db.add_defect(self.image_path, defect_type, x_min, y_min, x_max, y_max, conf.item())

        # 更新表格显示
        self.update_table()

    def update_table(self):
        """更新表格显示"""
        self.table.clearContents()
        self.table.setRowCount(0)
        defects = self.db.get_all_defects()
        for i, defect in enumerate(defects):
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(row_position, 1, QTableWidgetItem(defect['image_path']))
            self.table.setItem(row_position, 2, QTableWidgetItem(defect['defect_type']))
            self.table.setItem(row_position, 3, QTableWidgetItem(f"{defect['confidence']:.2f}"))
            self.table.setItem(row_position, 4, QTableWidgetItem(f"({defect['x_min']}, {defect['y_min']}, {defect['x_max']}, {defect['y_max']})"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FabricDefectDetection()
    window.show()
    sys.exit(app.exec_())
```

### 数据库管理 (`db_utils.py`)

```python
import sqlite3

class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                x_min INTEGER,
                y_min INTEGER,
                x_max INTEGER,
                y_max INTEGER,
                confidence REAL
            )
        ''')
        self.conn.commit()

    def add_defect(self, image_path, defect_type, x_min, y_min, x_max, y_max, confidence):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO defects (image_path, defect_type, x_min, y_min, x_max, y_max, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (image_path, defect_type, x_min, y_min, x_max, y_max, confidence))
        self.conn.commit()

    def get_all_defects(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM defects')
        return cursor.fetchall()

    def close(self):
        self.conn.close()
```

### 配置管理 (`config_utils.py`)

```python
class ConfigManager:
    def __init__(self, config_file):
        self.config = {}
        with open(config_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                self.config[key] = value

    def get_defect_name(self, class_id):
        return self.config[str(class_id)]
```


