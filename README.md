# 🧠 AIN313 – PA4
**OpenPose Tabanlı İskelet Çıkarımı ve Sınıflandırma**

Bu proje, videolardan OpenPose kütüphanesini kullanarak insan iskelet verilerini (keypoints) çıkarmak ve bu veriler üzerinden makine öğrenmesi yöntemleriyle aktivite tanıma gerçekleştirmek amacıyla geliştirilmiştir.

Readme dosyasi update alıcaktır
---

## Proje Klasör Yapısı

> **Önemli:** `data/` ve `tools/` klasörleri yüksek boyutlu dosyalar içerdiği için GitHub reposuna dahil edilmemiştir. Projeyi çalıştırmadan önce bu yapıyı yerel makinenizde oluşturmanız gerekmektedir.

```text
AIN313_PA4/
│
├── data/
│   ├── raw_videos/             # Orijinal .avi formatındaki videolar
│   │   ├── boxing/
│   │   ├── handclapping/
│   │   ├── handwaving/
│   │   ├── jogging/
│   │   ├── running/
│   │   └── walking/
│   │
│   ├── _tmp_openpose_json/     # OpenPose'dan çıkan geçici JSON dosyaları
│   └── poses_npz/              # İşlenmiş iskelet verileri (.npz)
│
├── tools/
│   └── openpose/               # OpenPose kütüphanesi ve modelleri
│
├── src/
│   ├── extract_poses.py        # İskelet çıkarma scripti
│   └── build_features.py       # Öznitelik çıkarımı scripti
│
├── .gitignore
├── requirements.txt 
└── README.md


# Repoyu klonlayın
git clone <repo_link>
cd AIN313_PA4

# Kütüphaneleri kurun
pip install -r requirements.txt

# İskelet çıkarma işlemini başlatın
python src/extract_poses.py

# Öznitelikleri oluşturun // hala yapmadik bunu, bunu yapicaz sonraki adim
python src/build_features.py

# poses_npz'yi burdan indirirsin, her hangi buyuk bir dosya gonderirken de bu drive'i kullanalim
https://drive.google.com/drive/folders/1WfONyscQ4ctaAS1yah5BenktwC46lKxU?usp=sharing

