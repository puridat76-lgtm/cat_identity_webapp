# Cat Identity Web App

เว็บแอปมินิมอลสำหรับทดสอบโปรเจกต์จำแนกแมวแบบ **Teachable Machine usage** โดยรองรับผลลัพธ์ 4 แบบ:

- `Haha / Loki / Mimi / Tiger / Toby / Vava` หรือชื่อโฟลเดอร์ที่คุณตั้งเอง
- `unknown`
- `not_cat`
- `low_quality`

> โปรเจกต์นี้เป็น **baseline backend + UI** ที่ทำงานได้จริงแบบไม่ต้องดาวน์โหลดโมเดลภายนอกเพิ่มตอนรัน โดยใช้การเปรียบเทียบ feature vector จากภาพอ้างอิงใน dataset ของคุณ

---

## 1) โครงสร้าง Dataset ที่ต้องใส่เอง

วางรูปภาพไว้แบบนี้:

```text
cat_identity_webapp/
└── data/
    ├── gallery/
    │   ├── Haha/
    │   │   ├── haha_01.jpg
    │   │   └── haha_02.jpg
    │   ├── Loki/
    │   ├── Mimi/
    │   ├── Tiger/
    │   ├── Toby/
    │   └── Vava/
    ├── not_cat/
    │   ├── hand_01.jpg
    │   ├── bottle_01.jpg
    │   └── dog_01.jpg
    └── unknown_cat/
        ├── internet_cat_01.jpg
        └── internet_cat_02.jpg
```

### คำอธิบาย
- `gallery/` = แมวที่ระบบต้องจำชื่อให้ได้
- `not_cat/` = ภาพที่ไม่ใช่แมว เช่น มือ ขวดน้ำ สุนัข
- `unknown_cat/` = แมวตัวอื่นที่ไม่ใช่แมวในระบบ

### คำแนะนำจำนวนรูป
- `gallery/<cat_name>/` อย่างน้อย 10–20 รูปต่อ 1 ตัวก่อนเริ่ม
- ถ้าจะให้เสถียรขึ้น ควรมี 50+ รูปต่อ 1 ตัว
- `not_cat/` และ `unknown_cat/` ใส่หลายแบบยิ่งช่วยให้ระบบปฏิเสธได้ดีขึ้น

---

## 2) ฟีเจอร์ในเว็บ

- หน้า `Cats` สำหรับเพิ่ม / ลบ / แก้ไข dataset หลัก
- หน้า `Not cat` และ `Unknown_cat` สำหรับจัดการ reference classes
- หน้า `Train` สำหรับ rebuild index พร้อมดู progress แบบ realtime
- หน้า `Predict` สำหรับอัปโหลดรูปหรือใช้กล้องแล้วทำนาย
- แสดงผลลัพธ์ final label และคะแนนเทียบกับ known / unknown / not_cat
- UI มินิมอล แยกหน้า ใช้งานง่าย

---

## 3) วิธีติดตั้ง

### 3.1 สร้าง virtual environment

**Windows (PowerShell)**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

---

## 4) วิธีรัน

### 4.1 ใส่ dataset ของคุณลงใน `data/`
จากนั้นสั่ง rebuild index:

```bash
python scripts/build_index.py
```

### 4.2 เปิดเว็บ
```bash
python app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่:
```text
http://127.0.0.1:5001
```

---

## 5) วิธีใช้งาน

### แบบ Upload
1. กด `เลือกรูป`
2. เลือกรูป
3. กด `Predict`

### แบบ Camera
1. กด `เปิดกล้อง`
2. กด `จับภาพ`
3. กด `Predict`

### ถ้าเพิ่งเพิ่ม dataset ใหม่
1. ไปหน้า `Train`
2. รอให้ระบบอ่านรูปใหม่
3. ค่อยทำนายอีกครั้ง

---

## 6) หลักการทำงานของระบบ

### Step 1: Quality check
- เช็กว่ารูปเล็กเกิน เบลอเกิน หรือมืดเกินไหม
- ถ้าไม่ผ่าน → `low_quality`

### Step 2: Feature extraction
- ดึง feature vector จากภาพ เช่น สี, texture, shape, DCT signature

### Step 3: Similarity matching
- เทียบกับภาพใน `gallery/`, `unknown_cat/`, `not_cat/`

### Step 4: Decision
- ถ้าคล้าย not_cat มากสุด → `not_cat`
- ถ้าคล้าย unknown_cat มาก หรือ known score ไม่ถึง threshold → `unknown`
- ถ้าคะแนน known ดีพอ → ตอบชื่อแมว

---

## 7) โครงสร้างไฟล์สำคัญ

```text
app.py                     # Flask app
services/features.py       # สร้าง feature vector
services/gallery.py        # สร้างและค้นหา index
services/quality.py        # ตรวจคุณภาพรูป
services/decision.py       # ตัดสินผลลัพธ์
services/pipeline.py       # รวม flow ทั้งหมด
scripts/build_index.py     # rebuild index จาก dataset
templates/base.html        # layout กลาง
templates/cats.html        # หน้า dataset แมว
templates/reference.html   # หน้า not_cat / unknown_cat
templates/train.html       # หน้า train
templates/predict.html     # หน้า predict
static/css/style.css       # สไตล์
static/js/common.js        # logic กลาง
static/js/cats.js          # หน้า dataset แมว
static/js/reference.js     # หน้า reference classes
static/js/train.js         # หน้า train
static/js/predict.js       # หน้า predict
```

---

## 8) การเทสระบบ

รัน test ได้ด้วย:

```bash
pytest -q
```

ไฟล์ทดสอบจะเช็กหลัก ๆ ดังนี้:
- เปิด `/api/status` ได้
- start train และดู status ได้
- predict known label ได้
- predict `not_cat` ได้
- predict `low_quality` ได้

---

## 12) Deploy ขึ้นเว็บจริง

ถ้าจะ deploy แบบเว็บจริงผ่าน GitHub + Gunicorn:

```bash
gunicorn --bind 0.0.0.0:$PORT wsgi:app
```

จุดสำคัญ:
- โปรเจกต์นี้เขียนไฟล์ลง `data/` ตอนอัปโหลดรูปและตอน Train
- ถ้า deploy บนเครื่องที่ filesystem ชั่วคราว ข้อมูลที่อัปโหลดจากหน้าเว็บอาจหายหลัง restart หรือ redeploy
- ตอนนี้แอปรองรับ environment variable เพื่อย้ายตำแหน่งเก็บข้อมูลได้แล้ว เช่น `DATA_DIR=/var/data`

ตัวแปรที่รองรับ:
- `DATA_DIR`
- `INDEX_DIR`
- `GALLERY_DIR`
- `NOT_CAT_DIR`
- `UNKNOWN_CAT_DIR`
- `CATS_META_PATH`
- `PORT`

---

## 9) ข้อจำกัดของ baseline นี้

โปรเจกต์นี้เป็น baseline ที่ใช้งานได้จริง แต่ยังไม่ใช่ deep learning identity model แบบเต็มระบบ จึงมีข้อจำกัด:

- ยังไม่ได้ crop หน้าแมวโดยเฉพาะ
- ยังไม่ได้ใช้ embedding model แบบ Siamese / Triplet / CLIP
- ความแม่นยำจะขึ้นกับคุณภาพ dataset มาก
- ถ้ารูปแมวหลายตัวหน้าคล้ายกันมาก อาจต้องอัปเกรดเป็น embedding model จริงในอนาคต

---

## 10) ถ้าจะอัปเกรดภายหลัง

แนวทางอัปเกรดในอนาคต:
- เพิ่ม cat detector / cat face cropper
- เปลี่ยน feature extractor เป็น CNN embedding
- ใช้ FAISS หรือ vector DB เต็มรูปแบบ
- เพิ่มหน้า admin สำหรับ register แมวใหม่
- เก็บ logs และ confusion cases

---

## 11) หมายเหตุสำคัญ

ถ้ายังไม่ใส่รูปใน `data/gallery/` แล้วกด predict ระบบจะฟ้องว่า gallery ว่าง ซึ่งเป็นพฤติกรรมปกติ
