import requests

# URL API lokal Anda
url = "http://127.0.0.1:8000/predict"

# Data yang akan dikirim
payload = {
    "text": "Pejabat itu korupsi uang rakyat dasar tikus!"
}

try:
    # Kirim request POST ke API
    response = requests.post(url, json=payload)
    
    # Cek apakah sukses
    if response.status_code == 200:
        print("✅ Prediksi Berhasil:")
        print(response.json())
    else:
        print("❌ Gagal:", response.text)

except Exception as e:
    print(f"Error koneksi: {e}")