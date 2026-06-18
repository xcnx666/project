# 智慧园区安防巡检系统 — 项目操作文档

> 适用版本:`cv1`
> 文档日期:2026-06-18
> 技术栈:Python + Flask + YOLOv8 (Ultralytics) + PaddleOCR + OpenCV + SQLite

---

## 一、项目概述

智慧园区安防巡检系统是一个基于计算机视觉的园区综合安防监控平台。通过接入视频流(默认本地摄像头),系统实时进行:

- 火焰 / 烟雾检测
- 人脸识别(已知 / 陌生人)
- 车辆识别 + 车牌 OCR(可选,代码中默认注释)
- 积水检测(可选,代码中默认注释)

一旦检测到异常事件,系统会自动:
1. 在视频画面上叠加告警框与文字;
2. 抓拍当前帧保存到 `snapshots/` 目录;
3. 将告警记录与图片 BLOB 写入 SQLite 数据库;
4. 通过 Web 界面提供告警列表、实时画面与历史截图查询。

---

## 二、目录结构

```
智慧园区安防巡检系统/
├── README.md                  # 项目简介
└── cv1/
    ├── app.py                 # Web 主程序(检测 + Flask API)
    ├── model_train.py         # YOLO 模型训练脚本
    ├── requirements           # Python 依赖清单
    ├── models/                # (运行后自动创建)放置 *.pt 模型
    ├── templates/             # (运行后自动创建)前端 HTML
    ├── snapshots/             # (运行后自动创建)告警截图
    ├── show.db                # (运行后自动创建)SQLite 数据库
    └── user.txt               # (注册后产生)用户凭证文件
```

---

## 三、环境准备

### 3.1 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核及以上 |
| 内存 | 8 GB | 16 GB |
| GPU | 可选(CPU 可跑但较慢) | NVIDIA GPU + CUDA |
| 摄像头 | USB 摄像头 / 笔记本自带 | 1080P 监控摄像头 |
| 操作系统 | Windows 10 / macOS / Linux | Windows 10/11(原项目) |

### 3.2 软件要求

- Python **3.10 ~ 3.13**(注释中声明兼容 3.13)
- pip 21+
- 推荐使用虚拟环境

### 3.3 创建虚拟环境并安装依赖

```bash
cd 智慧园区安防巡检系统/cv1

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements
```

> **注意**:`requirements` 文件中 PaddlePaddle 与 PaddleOCR 较重,若不需要车牌识别可临时注释掉相关行,跳过安装。

### 3.4 准备模型文件

`app.py` 默认从以下 Windows 路径加载模型(需修改):

```python
fire_model  = YOLO(r"C:\Users\39608\Desktop\demo1\models\fire_best.pt")
car_model   = YOLO(r"C:\Users\39608\Desktop\demo1\models\vehicle.pt")
plate_model = YOLO(r"C:\Users\39608\Desktop\demo1\models\plate.pt")
face_model  = YOLO(r"C:\Users\39608\Desktop\demo1\models\face_best5.pt")
```

操作步骤:
1. 在 `cv1/` 下新建 `models/` 文件夹(运行 `app.py` 时也会自动创建);
2. 将训练好的 4 个 `.pt` 模型文件放入 `models/` 目录;
3. 修改 `app.py` 顶部的模型路径为相对路径,例如:

```python
fire_model  = YOLO("models/fire_best.pt")
car_model   = YOLO("models/vehicle.pt")
plate_model = YOLO("models/plate.pt")
face_model  = YOLO("models/face_best5.pt")
```

> 如果只启用部分检测,可以注释掉未使用的模型行,同时把 `main()` 里对应检测函数注释。

---

## 四、运行项目

### 4.1 启动 Web 服务

```bash
cd cv1
python app.py
```

启动后默认监听 `http://0.0.0.0:5000`,浏览器访问 `http://127.0.0.1:5000/` 即可。

### 4.2 默认登录账号

`app.py` 中内置 6 个测试账号(密码均为 `123`):

| 用户名 | 密码 |
|--------|------|
| lwc    | 123  |
| zjt    | 123  |
| gdp    | 123  |
| ljx    | 123  |
| fyf    | 123  |
| xjg    | 123  |

> 正式使用前请删除/修改这些账号,并将用户数据迁移到数据库。

### 4.3 视频源说明

`main(source)` 中的 `source` 参数支持:

- `0` — 默认调用本机第一个摄像头(`cv2.VideoCapture(0)`)
- 数字字符串(如 `"1"`)— 其它摄像头
- 视频文件路径(如 `"test.mp4"`)— 离线视频分析
- RTSP/HTTP 流地址(如 `"rtsp://user:pass@ip/stream"`)

通过 API 切换视频源:
```
GET /api/video?path=rtsp://xxx
GET /api/video?path=video.mp4
```

---

## 五、核心功能详解

### 5.1 火焰检测 (`fire_recognize`)

- 模型:`fire_best.pt`
- 置信度阈值:`fire_conf = 0.5`
- 触发动作:在画面上绘制红色矩形与 “Fire” 文字,抓拍保存,告警计数 `num += 1`

### 5.2 人脸检测 (`face_recognize`)

- 模型:`face_best5.pt`
- 置信度阈值:`face_conf = 0.5`
- 逻辑:
  - 置信度 **< 0.5** → 标记为 **陌生人 (Stranger)**,红色框,抓拍保存;
  - 置信度 **≥ 0.5** → 显示模型识别的姓名/标签,绿色框。

> 若想提升识别准确率,请使用包含目标人脸的数据集对 `face_best5.pt` 进行微调。

### 5.3 车辆 / 车牌识别(`car_recognize`,已注释)

启用步骤:
1. 取消 `app.py` 中 `car_recognize` 函数的注释;
2. 取消 `PaddleOCR` 初始化代码的注释;
3. 取消 `main()` 中 `frame, stranger_alert = car_recognize(frame)` 的注释;
4. 安装 PaddlePaddle 与 PaddleOCR;
5. 在 `WHITELIST` 集合中添加允许通行的车牌号。

### 5.4 积水检测(`water_recognize`,已注释)

启用方式同上,需额外提供 `water_model` 训练好的权重。

### 5.5 告警抓拍与防抖

```python
MIN_SAVE_INTERVAL = 3.0   # 同一类型告警 3 秒内不重复抓拍
last_save_time = {}       # 按告警类型记录上次抓拍时间
```

抓拍同时会:
- 写入 `snapshots/<type>_<conf>_<timestamp>.jpg`;
- 在 `alerts` 表新增一条告警记录;
- 在 `alert_images` 表保存图片 BLOB 与文件路径。

### 5.6 实时视频流

`/` 路由返回首页模板,`/video_feed` 路由使用 `multipart/x-mixed-replace` 协议推送 MJPEG 帧,前端通过 `<img src="/video_feed">` 即可显示实时画面。

---

## 六、API 接口一览

| 方法 | 路径 | 说明 | 鉴权 |
|------|------|------|------|
| GET  | `/` | 首页 | 否 |
| GET  | `/video_feed` | 实时视频流(MJPEG) | 否 |
| POST | `/api/login` | 登录(JSON: username/password) | 否 |
| POST | `/api/Register` | 注册(JSON: username/password) | 否 |
| GET  | `/api/user` | 获取当前登录用户 | 是(Session) |
| GET  | `/api/video?path=...` | 切换视频源并返回 MJPEG 流 | 否 |
| GET  | `/api/num` | 获取累计告警次数 | 否 |
| GET  | `/api/snapshots?limit=20` | 告警截图列表(JSON) | 否 |
| GET  | `/api/snapshots/<id>` | 获取指定 ID 的截图二进制 | 否 |

请求示例(使用 `curl`):

```bash
# 登录
curl -X POST http://127.0.0.1:5000/api/login \
     -H "Content-Type: application/json" \
     -d '{"username":"lwc","password":"123"}'

# 查询最近 10 条告警
curl http://127.0.0.1:5000/api/snapshots?limit=10

# 获取第 1 张告警图片
curl http://127.0.0.1:5000/api/snapshots/1 -o alert.jpg
```

---

## 七、数据库结构

数据库文件:`show.db`(SQLite,启用 WAL 模式以支持并发写)。

### 7.1 alerts(告警主表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| timestamp | TEXT | ISO 格式时间戳 |
| alert_type | TEXT | Fire / Stranger_Face / Stranger_Car_xxx / Ponding |
| confidence | REAL | 置信度(0~1) |
| description | TEXT | 描述 |

### 7.2 alert_images(告警图片表)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增主键 |
| alert_id | INTEGER | 外键 → alerts.id,级联删除 |
| timestamp | TEXT | 时间戳 |
| image_data | BLOB | JPEG 编码后的图片二进制 |
| file_path | TEXT | `snapshots/` 中的实际文件路径 |

查看数据库(可使用 DB Browser for SQLite 或 sqlite3 CLI):
```bash
sqlite3 show.db
> .tables
> SELECT id, alert_type, confidence, timestamp FROM alerts ORDER BY id DESC LIMIT 10;
```

---

## 八、模型训练

`model_train.py` 提供了 YOLOv8 训练示例:

```python
models = [
    ('fire', r'.../data/fire1/fire.yaml'),
    # ('face', r'.../data/face/face.yaml')
]
weight = r'yolov8n'
```

使用步骤:
1. 准备 YOLO 格式数据集(目录结构:`images/`、`labels/`、`*.yaml` 索引文件);
2. 修改 `models` 列表中的 yaml 路径;
3. 视硬件调整 `epochs`、`batch`、`device`(无 GPU 改为 `device='cpu'`,并去掉 `freeze=10`);
4. 执行:
   ```bash
   python model_train.py
   ```
5. 训练完成后,最佳权重位于 `runs/detect/fire训练成功/weights/best.pt`,复制到 `cv1/models/` 即可。

> **重要**:Windows 上若 `device='0'` 报错,说明未安装 GPU 版 PyTorch 或 CUDA,改用 `device='cpu'`。

---

## 九、常见问题 (FAQ)

**Q1:启动时报 `cannot open camera 0`?**
- 确认摄像头未被其它程序占用;
- 在 macOS 系统设置中授权终端访问摄像头;
- 尝试改为 `cv2.VideoCapture(1)`。

**Q2:`ultralytics` 安装失败 / 报 `numpy` 版本冲突?**
- 使用虚拟环境,严格按 `requirements` 安装;
- 升级 pip:`python -m pip install --upgrade pip`。

**Q3:PaddleOCR 安装后导入报错?**
- 临时注释 `app.py` 中 PaddleOCR 相关代码;
- 或在 Python 3.10 环境中单独安装 PaddlePaddle(部分版本不支持 3.13)。

**Q4:Web 页面打开是空白?**
- 确认 `templates/index.html` 存在;
- 浏览器按 F12 查看控制台报错信息。

**Q5:告警数 `num` 一直累加不重置?**
- 这是设计行为。如需每天清零,在 `/api/num` 中加定时任务或将 `num` 改为按日持久化。

**Q6:模型路径是 Windows 反斜杠,macOS / Linux 怎么办?**
- 改用正斜杠 `/` 或相对路径;最稳妥是放在 `cv1/models/` 下用相对路径引用。

**Q7:如何在前端显示告警列表?**
- 调用 `GET /api/snapshots` 获取 JSON,在前端模板中渲染即可。

---

## 十、后续可扩展方向

1. **多路视频流**:将 `main()` 拆为独立线程,支持多路 RTSP 同步监控。
2. **告警推送**:接入企业微信 / 钉钉 / 邮件 webhook,实时推送告警。
3. **更细粒度权限**:将用户从 `dict` 改为 SQLite 表,增加角色(管理员 / 操作员)。
4. **历史回放**:把抓拍图片按时段归档,提供按日期 / 类型筛选的检索页。
5. **数据看板**:用 ECharts 展示每日 / 每月告警趋势、类型分布。
6. **模型热更新**:检测 `models/*.pt` 文件 hash 变化,自动重载模型。
7. **Docker 化部署**:编写 `Dockerfile` 与 `docker-compose.yml`,实现一键启动。

---

## 十一、联系与维护

- 项目维护:智慧园区安防巡检系统开发组
- 文档版本:v1.0
- 最后更新:2026-06-18

如发现 Bug 或有改进建议,请在项目仓库提交 Issue。
