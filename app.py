import os
import uuid
import cv2
import numpy as np
import requests
from datetime import datetime, date
from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from pony.orm import Database, Required, Optional, PrimaryKey, Set, db_session, select
from luno_python.client import Client

# Luno API Setup
LUNO_API_KEY_ID = "jnm42w8w23t8v"
LUNO_API_KEY_SECRET = "QSRtcDAysoiAs3IiRrDtqaXeO35SPzFMXU0niYUHNnc"
luno_client = Client(api_key_id=LUNO_API_KEY_ID, api_key_secret=LUNO_API_KEY_SECRET)

# Database Setup
db = Database()
db.bind(provider='mysql', host='localhost', user='kusehatweb3', passwd='Agustus2000', db='kusehatweb3')

class User(db.Entity):
    _table_ = "user"
    UserID = PrimaryKey(int, auto=True)
    NamaUser = Required(str)
    Email = Required(str, unique=True)
    Password = Required(str)
    Register_Date = Required(datetime)
    Login_Date = Optional(datetime)
    Saldo = Required(float, default=0.0)
    topups = Set("TopUp")
    exchanges = Set("Exchange")

class TopUp(db.Entity):
    _table_ = "topup"
    ID = PrimaryKey(int, auto=True)
    User = Required(User)
    Jumlah = Required(float)
    Metode = Required(str)
    Tanggal = Required(datetime)

class Exchange(db.Entity):
    _table_ = "exchange"
    ID = PrimaryKey(int, auto=True)
    User = Required(User)
    Tujuan = Required(str)
    Gambar = Required(str)
    Diagnosa = Required(str)
    Tanggal = Required(datetime)
    SaldoReward = Required(float)

db.generate_mapping(create_tables=True)

app = Flask(__name__)
app.secret_key = 'rahasia_kusehat'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Memuat Model dan Label (dilakukan sekali saat aplikasi dimulai)
try:
    model = load_model("model/keras_Model.h5", compile=False)
    with open("model/labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    class_names = []

GEMINI_API_KEY = "AIzaSyDwniC_zbYaVpRWRGjGk9HnhJWAe9IPZGM"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def get_gemini_explanation(disease_name):
    prompt = (
        f"Tolong jelaskan informasi tentang penyakit kulit berikut ini dalam format HTML yang rapi:\n\n"
        f"Nama penyakit: {disease_name}\n\n"
        f"1. Deskripsi\n2. Gejala dan Penyebab\n3. Cara Penyembuhan\n4. Rekomendasi Obat"
    )
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Periksa apakah respons memiliki struktur yang diharapkan
        if data and 'candidates' in data and data['candidates'][0] and 'content' in data['candidates'][0] and 'parts' in data['candidates'][0]['content'] and data['candidates'][0]['content']['parts'][0] and 'text' in data['candidates'][0]['content']['parts'][0]:
            return data['candidates'][0]['content']['parts'][0]['text']
        else:
            return "❌ Respons dari Gemini tidak valid."
            
    except requests.exceptions.RequestException as e:
        return f"❌ Gagal mengambil informasi dari Gemini AI: {e}"

def process_image_for_detection(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Gagal membaca file gambar."}
        
        image = cv2.resize(image, (224, 224))
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1

        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence = float(prediction[0][index])

        gemini_info = get_gemini_explanation(class_name)

        diagnosis = (
            f"📤 <b>Deteksi Upload:</b> <b>{class_name}</b><br>"
            f"🧪 <b>Kepercayaan:</b> {confidence:.2%}<br><br>"
            f"🧠 <b>Penjelasan Gemini AI:</b><br>{gemini_info}"
        )
        return {"diagnosis": diagnosis, "class_name": class_name}
    except Exception as e:
        return {"error": f"❌ Terjadi kesalahan saat proses AI: {e}"}

@app.route("/", methods=["GET", "POST"])
@db_session
def home():
    diagnosis = ""
    image_path = ""
    user = None

    if "user_id" in session:
        user = User.get(UserID=session["user_id"])

    if request.method == "POST" and "image" in request.files:
        if not user:
            return redirect(url_for("home"))

        today = date.today()
        upload_count = select(e for e in Exchange if e.User == user and e.Tanggal.date() == today and e.Tujuan == "deteksi").count()

        if upload_count >= 3:
            if user.Saldo >= 150000:
                user.Saldo -= 150000
                upload_count = 0
            else:
                return redirect(url_for("topup_page"))

        file = request.files.get("image")
        if not file or file.filename == "":
            diagnosis = "❌ Gambar tidak ditemukan."
        elif not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            diagnosis = "❌ Format file tidak didukung. Gunakan JPG, JPEG, atau PNG."
        else:
            filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            
            result = process_image_for_detection(image_path)
            if "error" in result:
                diagnosis = result["error"]
            else:
                diagnosis = result["diagnosis"]
                class_name = result["class_name"]
                Exchange(User=user, Tujuan="deteksi", Gambar=filename, Diagnosa=class_name, Tanggal=datetime.now(), SaldoReward=0.0)

    return render_template("index.html", diagnosis=diagnosis, image_path=image_path,
                           user=user, topup_address="", topup_error="", section="dashboard", today=date.today())

@app.route("/register", methods=["POST"])
@db_session
def register():
    nama = request.form.get("nama")
    email = request.form.get("email")
    password = request.form.get("password")

    if User.get(Email=email):
        return "❌ Email sudah terdaftar."

    User(NamaUser=nama, Email=email, Password=password, Register_Date=datetime.now())
    return "✅ Pendaftaran berhasil. Silakan login."

@app.route("/login", methods=["POST"])
@db_session
def login():
    email = request.form.get("email")
    password = request.form.get("password")
    user = User.get(Email=email, Password=password)

    if user:
        session['user_id'] = user.UserID
        return redirect(url_for("dashboard"))
    return "❌ Email atau Password salah."

@app.route("/dashboard")
@db_session
def dashboard():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("home"))
    user = User.get(UserID=user_id)
    return render_template("index.html", diagnosis="", image_path="", user=user,
                           topup_address="", topup_error="", section="dashboard")

@app.route("/update_user", methods=["POST"])
@db_session
def update_user():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("home"))
    user = User.get(UserID=user_id)
    old_password = request.form.get("old_password")

    if user.Password != old_password:
        return "❌ Password lama tidak cocok."

    user.NamaUser = request.form.get("nama")
    user.Email = request.form.get("email")
    user.Password = request.form.get("new_password")
    return redirect(url_for("dashboard"))

@app.route("/topup", methods=["GET"])
@db_session
def topup_page():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("home"))
    user = User.get(UserID=user_id)
    return render_template("index.html", diagnosis="", image_path="", user=user,
                           topup_address="", topup_error="", section="topup")
    
@app.route("/topup", methods=["POST"])
@db_session
def topup():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("home"))

    metode = request.form.get("metode")
    jumlah = float(request.form.get("jumlah", 0))
    alamat = ""
    error = ""

    try:
        if metode == "btc":
            res = luno_client.get_funding_address(asset="XBT")
            alamat = res["address"]
        elif metode == "eth":
            res = luno_client.get_funding_address(asset="ETH")
            alamat = res["address"]
        else:
            error = "❌ Metode tidak valid."
    except Exception as e:
        error = f"❌ Gagal mengambil alamat: {str(e)}"

    user = User.get(UserID=user_id)
    if not error:
        TopUp(User=user, Jumlah=jumlah, Metode=metode.upper(), Tanggal=datetime.now())
        user.Saldo += jumlah

    return render_template("index.html", diagnosis="", image_path="", user=user,
                           topup_address=alamat, topup_error=error)

# Reward logic
def get_exchange_reward(tujuan):
    tujuan = tujuan.lower()
    if tujuan == "dokter":
        return 100_000
    elif tujuan == "data_ai":
        return 200_000
    return 0

@app.route("/exchange", methods=["POST"])
@db_session
def exchange():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("home"))

    user = User.get(UserID=user_id)
    file = request.files.get("image")
    tujuan = request.form.get("tujuan")

    if not file or file.filename == "":
        return render_template("index.html", diagnosis="❌ Gambar tidak ditemukan untuk ditukar.",
                               image_path="", user=user, topup_address="", topup_error="")

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return render_template("index.html", diagnosis="❌ Format gambar tidak didukung.",
                               image_path="", user=user, topup_address="", topup_error="")

    try:
        filename = f"exchange_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        reward = get_exchange_reward(tujuan)
        tanggal = datetime.now()

        if reward > 0:
            user.Saldo += reward
            message = f"🎁 Gambar berhasil ditukar. Anda mendapat saldo IDR {reward:,}."
        else:
            message = "⚠️ Tujuan tidak dikenali. Tidak ada saldo diberikan."

        Exchange(User=user, Tujuan=tujuan, Gambar=filename,
                 Diagnosa="", Tanggal=tanggal, SaldoReward=reward)

        return render_template("index.html", diagnosis=message, image_path=image_path,
                               user=user, topup_address="", topup_error="")

    except Exception as e:
        return render_template("index.html", diagnosis=f"❌ Gagal memproses penukaran gambar: {e}",
                               image_path="", user=user, topup_address="", topup_error="")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

