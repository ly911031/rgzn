提升模型的泛化能力  C正则化
链式求导  C 由外向内
图像预处理，提高像素对比度且较好保存细节 直方图
数据处理前，整体检查 维度

@所有人 
下面👇🏻是今天课程的回看链接

日期：2025-11-10 18:28:01
录制文件：https://meeting.tencent.com/crm/2GLQJYYBf8

日期：2025-11-11 18:09:19
转写文件：https://meeting.tencent.com/ctm/l6Mq1ZPeeb

日期：2025-11-11 18:30:30
录制文件：https://meeting.tencent.com/crm/KePkdWbZbd



2025-11-12 18:26:54
录制文件：https://meeting.tencent.com/crm/2jBqxBdB8e

录制：人工智能训练师赛项培训
日期：2025-11-13 18:31:50
录制文件：https://meeting.tencent.com/crm/2qY77r6jdf



conda activate dify
cp -r /tmp_package/*   /root/bayes-tmp
cd /root/
python start_dify.py


硅基流动 sk-pzcykueiiwmpnprluubanvvzfzuhpvllrdvgubdiwakxmgfz


训练/学习平台链接：https://211.139.108.69:31003/login
账号：报名时填的手机号
密码：Admin@9000

#进入package文件夹
cd /root/bayes-tmp/package
#解压Labelme所依赖的系统级图形界面库
tar -zxf labelme_dep.tar.gz
#进入解压后的labelme_dep文件夹
cd /root/bayes-tmp/package/labelme_dep
#安装Labelme所依赖的系统级图形界面库
dpkg -i *.deb
tar -xf  /root/bayes-tmp/package/户型图片识别.tar.gz  -C ./

#进入package文件夹
cd /root/bayes-tmp/package
#解压py_deps安装包并进入解压后的目录
tar -zxf py_deps.tar.gz
cd /root/bayes-tmp/package/py_deps
#安装python3-pyqt5
dpkg -i ./*

#进入package文件夹
cd /root/bayes-tmp/package
#解压labelme安装包并进入解压后的目录
tar -zxf labelme.tar.gz
cd /root/bayes-tmp/package/labelme
#安装所有.whl文件
pip install *.whl

根目录启动 start labelme

启动yolo 模型

四、YOLO模型部署安装
#进入目录
cd /root/bayes-tmp
#创建yolo环境指定python版本
conda create -n yolo11 python=3.10  -y
#激活环境：
conda activate yolo11


缓存目录
mkdir -p /root/bayes-tmp/pip-cache /pip-packages

export PIP_CACHE_DIR=
export PYTHONUSERBASE=

export PIP_CACHE_DIR=/root/bayes-tmp/pip-cache
export PYTHONUSERBASE=/root/bayes-tmp/pip-packages


#进入目录
cd /root/bayes-tmp/package/
#解压 deqs.tar.gz 到当前目录
tar -zxf deps.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/deps
# 安装ultralytics及依赖包，批量安装所有.whl 文件
pip install *.whl

#再进入package文件夹
cd /root/bayes-tmp/package
#解压 reqs.tar.gz 到当前目录
tar -zxf reqs.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/reqs
# 安装依赖包，批量安装所有 .whl 文件
pip install *.whl

#进入package文件夹
cd /root/bayes-tmp/package
#解压 modelscope.tar.gz 到当前目录
tar -zxf modelscope.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/modelscope
# 批量安装所有 .whl 文件
pip install *.whl



董玉涛 2025/11/13 19:28:43
112  mkdir -p  /root/bayes-tmp/Model_yoll 
  113  cd /root/bayes-tmp/package
  114  cp yolo11s.pt /root/bayes-tmp/Model_yoll/
  115  cd /root/bayes-tmp/package
  116  tar -zxf libgl1.tar.gz
  117  cd /root/bayes-tmp/package/libgl1
  118  dpkg -i *.deb
  119  python -c "import ultralytics; print(ultralytics.__version__)"

Leon_Zhang 2025/11/13 19:28:48
ok




conda create -n bge -y 
conda activate bge
export HF_ENDPOINT=https://hf-mirror.com
export XINFERENCE_MODEL_SRC=modelscope             
export XINFERENCE_HOME=/root/bayes-tmp 

进入package文件夹
cd /root/bayes-tmp/package
#解压 torchvision_torchaudio_torch.tar.gz 到当前目录
tar -zxf torchvision_torchaudio_torch.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/torchvision_torchaudio_torch
# 批量安装所有 .whl 文件
pip install *.whl


#进入package文件夹
cd /root/bayes-tmp/package
#解压Xinference.tar.gz 到当前目录
tar -zxf Xinference.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/Xinference
# 批量安装所有 .whl 文件
pip install *.whl

xinference-local --host 0.0.0.0 --port 9997  

export XINFERENCE_MODEL_SRC=modelscope             
export XINFERENCE_HOME=/root/bayes-tmp 


export PIP_CACHE_DIR=/root/bayes-tmp/.cache
export PYTHONUSERBASE=/root/bayes-tmp/pip-packages
export HF_ENDPOINT=https://hf-mirror.com           
export XINFERENCE_MODEL_SRC=modelscope             
export XINFERENCE_HOME=/root/bayes-tmp 

#新建/modelscope/models/Xorbits目录
mkdir -p  /root/bayes-tmp/modelscope/models/Xorbits

#进入package文件夹
cd /root/bayes-tmp/package
#解压Xinference.tar.gz 到指定目录
tar -zxf bge-small-zh-v1.5.tar.gz -C /root/bayes-tmp/modelscope/models/Xorbits
tar -zxf bge-reranker-base.tar.gz -C /root/bayes-tmp/modelscope/models/Xorbits 
# 启动 bge-small-zh-v1.5（嵌入模型）
xinference launch --model-name bge-small-zh-v1.5 --model-type embedding --model-path  /root/bayes-tmp/modelscope/models/Xorbits/bge-small-zh-v1.5
# 启动 bge-reranker-base（重排序模型）
xinference launch --model-name bge-reranker-base --model-type rerank --model-path /root/bayes-tmp/modelscope/models/Xorbits/bge-reranker-base


curl http://localhost:9997/v1/models        
xinference list     


curl http://localhost:9997/v1/embeddings \
 -H "Content-Type: application/json" \
 -d '{
  "input": "测试ebmeddings",
  "model": "bge-small-zh-v1.5"
}'

curl -X 'POST' 'http://localhost:9997/v1/rerank' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bge-reranker-base",
    "query": "A man is eating pasta.",
    "documents": [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ]
}'

xinference-local --host 0.0.0.0 --port 9997   


conda activate yolo11  
mkdir -p /root/bayes-tmp/mydataset/户型图标记json文件    # 保存标注的 JSON 文件
mkdir -p /root/bayes-tmp/mydataset/户型图标记图片           # 存放原始图片
#进入package文件夹
cd /root/bayes-tmp/package
#解压yolo_deps.tar.gz 到当前目录
tar -zxf yolo_deps.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/yolo_deps
# 批量安装所有 .whl 文件
pip install *.whl 


import os
import json
import random
import shutil
from PIL import Image

# 路径配置
json_dir = "/root/bayes-tmp/mydataset/户型图标记json文件"
image_dir = "/root/bayes-tmp/mydataset/户型图标记"
output_root = "/root/bayes-tmp/mydataset/dataset"  # 输出标准YOLO格式结构

# 创建目录
for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(output_root, sub), exist_ok=True)

# 类别列表
class_names = []

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

# 收集所有文件
samples = [f for f in os.listdir(json_dir) if f.endswith(".json")]
random.shuffle(samples)
split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

def process_samples(sample_list, subset):
    for file in sample_list:
        json_path = os.path.join(json_dir, file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_name = os.path.basename(data["imagePath"].replace("\\", "/"))
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"❌ 找不到图片：{image_path}")
            continue

        with Image.open(image_path) as img:
            w, h = img.size


# 输出路径
        base_name = os.path.splitext(file)[0]
        txt_path = os.path.join(output_root, f"labels/{subset}/{base_name}.txt")

        with open(txt_path, 'w', encoding='utf-8') as out_file:
            for shape in data['shapes']:
                label = shape['label'].strip().replace("　", "").replace(" ", "")
                if label not in class_names:
                    class_names.append(label)
                class_id = class_names.index(label)

                points = shape['points']
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                box = [min(xs), min(ys), max(xs), max(ys)]
                yolo_box = convert_to_yolo((w, h), box)
                out_file.write(f"{class_id} {' '.join([str(round(x, 6)) for x in yolo_box])}\n")

        # 拷贝图片到 images/train 或 images/val
        dst_img_path = os.path.join(output_root, f"images/{subset}/{image_name}")
        shutil.copy(image_path, dst_img_path)

# 执行转换
process_samples(train_samples, "train")
process_samples(val_samples, "val")

# 写 classes.txt
with open(os.path.join(output_root, "classes.txt"), 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(name + "\n")

# 写 data.yaml
yaml_path = os.path.join(output_root, "data.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(f"train: {os.path.join(output_root, 'images/train')}\n")
    f.write(f"val: {os.path.join(output_root, 'images/val')}\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names:\n")
    for name in class_names:
        f.write(f"  - {name}\n")

print("✅ 全部转换完成！")
print("📂 数据集目录：", output_root)
print("📄 类别文件：", os.path.join(output_root, "classes.txt"))
print("📄 配置文件：", yaml_path)

董玉涛 2025/11/13 20:25:10
训练代码
import os
import torch
from ultralytics import YOLO

# 限制GPU内存
torch.cuda.set_per_process_memory_fraction(0.4, device=0)

# 模型路径（确认正确）
model_path = '/root/bayes-tmp/Model_yoll/yolo11s.pt'

# 验证文件存在性+大小（双重确认）
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # 转成MB
    print(f"本地模型存在，大小：{file_size:.1f}MB")
else:
    raise FileNotFoundError(f"模型文件不存在！路径：{model_path}")

# 强制加载本地模型（添加 verbose=True 查看加载日志）
try:
    model = YOLO(model_path, verbose=True)
    print("本地模型加载成功！")
except Exception as e:
    print(f"本地模型加载失败：{str(e)}")
    raise  # 抛出错误，不再自动下载

# 训练（参数不变）
model.train(
    data='/root/bayes-tmp/mydataset/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_yolo19cls',
    workers=4,
    device=0,
    amp=False
)



王澄淼 2025/11/13 20:25:21
数据量小，不需要训练100次吧？

暴露APID地址
董玉涛 2025/11/13 21:02:37
import os
import json
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO

app = FastAPI()

model_path = "/root/bayes-tmp/runs/detect/my_yolo19cls4/weights/best.pt"
print(f"✅ 正在加载模型：{model_path}")
model = YOLO(model_path)
print("✅ 模型加载成功！")

BASE_DIR = "/root/bayes-tmp"
ORIGINAL_DIR = os.path.join(BASE_DIR, "originals")
PREDICT_DIR = os.path.join(BASE_DIR, "predict_result")
YOLO_RUN_DIR = os.path.join(BASE_DIR, "runs/detect/predict")
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PREDICT_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    original_name = os.path.basename(file.filename)
    name, ext = os.path.splitext(original_name)
    final_image_name = original_name
    original_path = os.path.join(ORIGINAL_DIR, final_image_name)

    if os.path.exists(original_path):
        uid = uuid.uuid4().hex[:6]
        final_image_name = f"{name}_{uid}{ext}"
        original_path = os.path.join(ORIGINAL_DIR, final_image_name)

    with open(original_path, "wb") as f:
        f.write(await file.read())
    print(f"\n📥 原图保存至：{original_path}")

    # 删除 runs/detect/predict 目录，避免多次运行时目录不同
    if os.path.exists(YOLO_RUN_DIR):
        shutil.rmtree(YOLO_RUN_DIR)


董玉涛 2025/11/13 21:02:44
  print("🚀 执行 YOLO 推理...")
    results = model(original_path, save=True, save_dir=YOLO_RUN_DIR, show_conf=False)
    result = results[0]
    print(f"✅ 推理完成，识别目标数：{len(result.boxes)}")

    # ----------- 找到 YOLO 保存的预测图（支持任意扩展名）-----------
    yolo_pred_img = None
    for file in os.listdir(YOLO_RUN_DIR):
        fbase, fext = os.path.splitext(file)
        # 只看图片文件
        if fbase == os.path.splitext(final_image_name)[0] and fext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            yolo_pred_img = file
            break

    if not yolo_pred_img:
        # fallback: 找目录下唯一图片
        imgs = [f for f in os.listdir(YOLO_RUN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(imgs) == 1:
            yolo_pred_img = imgs[0]

    if yolo_pred_img:
        pred_image_path = os.path.join(PREDICT_DIR, yolo_pred_img)
        yolo_saved_path = os.path.join(YOLO_RUN_DIR, yolo_pred_img)
        shutil.copy(yolo_saved_path, pred_image_path)
        print(f"🖼 YOLO 预测图已复制到：{pred_image_path}")
    else:
        print("❌ 未找到 YOLO 预测输出图片，请检查保存目录。")
        return {"error": "❌ 模型预测图未找到，请检查 YOLO 是否推理成功。"}

    # 生成 JSON 文件
    objects = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = model.names[cls_id]
        xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
        objects.append({
            "label": label,
            "confidence": round(conf, 3),
            "bbox": xyxy
        })

    json_name = f"{os.path.splitext(yolo_pred_img)[0]}_pred.json"
    json_path = os.path.join(PREDICT_DIR, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=4, ensure_ascii=False)

    print(f"📄 JSON 文件保存至：{json_path}")

    labels = [obj["label"] for obj in objects]
    print(f"🔍 检测到标签：{labels}")

    # 用实际的预测图片名返回URL
    image_url = f"http://localhost:8000/image/{yolo_pred_img}"
    json_url = f"http://localhost:8000/json/{json_name}"

    return {
        "result": f"✅ 推理完成\n\n📌 标签：{labels}\n🖼️ 图片：{image_url}\n📄 JSON：{json_url}",
        "image_path": image_url,
        "json_path": json_url,
        "labels": labels
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    path = os.path.join(PREDICT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    return JSONResponse(content={"error": "Image not found"}, status_code=404)

@app.get("/json/{filename}")
def get_json(filename: str):
    path = os.path.join(PREDICT_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    return JSONResponse(content={"error": "JSON not found"}, status_code=404)


董玉涛 2025/11/13 21:14:07
export PATH="/opt/conda/envs/yolo11/bin/:$PATH"

董玉涛 2025/11/13 21:14:14
echo $PATH | grep "yolo11/bin"  

董玉涛 2025/11/13 21:14:20
#进入package文件夹
cd /root/bayes-tmp/package
#解压uvicorn .tar.gz 到当前目录
tar -zxf uvicorn.tar.gz
#进入解压后的目录
cd /root/bayes-tmp/package/uvicorn
# 批量安装所有 .whl 文件
pip install *.whl
#进入yolo_api所在文件夹
cd /root/bayes-tmp/
#执行命令
python -m uvicorn yolo_api:app --host 0.0.0.0 --port 8000

董玉涛 2025/11/13 21:14:28
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/root/bayes-tmp/mydataset/户型图标记/2室1厅1厨1卫005.png'
import os
import torch
from ultralytics import YOLO

# 限制GPU内存
torch.cuda.set_per_process_memory_fraction(0.4, device=0)

# 模型路径（确认正确）
model_path = '/root/bayes-tmp/Model_yoll/yolo11s.pt'

# 验证文件存在性+大小（双重确认）
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # 转成MB
    print(f"本地模型存在，大小：{file_size:.1f}MB")
else:
    raise FileNotFoundError(f"模型文件不存在！路径：{model_path}")

# 强制加载本地模型（添加 verbose=True 查看加载日志）
try:
    model = YOLO(model_path, verbose=True)
    print("本地模型加载成功！")
except Exception as e:
    print(f"本地模型加载失败：{str(e)}")
    raise  # 抛出错误，不再自动下载

# 训练（参数不变）
model.train(
    data='/root/bayes-tmp/mydataset/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_yolo19cls',
    workers=4,
    device=0
)
