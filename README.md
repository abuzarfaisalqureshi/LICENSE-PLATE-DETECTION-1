# 🚘 License Plate Detection System

### 🔍 See It. Detect It. Track Every Plate!

Welcome to the **License Plate Detection System**, a computer vision project powered by **YOLO and Streamlit** that detects vehicle license plates from images and videos.

Turn ordinary vehicle images and videos into smart detection results with an easy-to-use web application!

## ✨ Features

- 🖼️ **Image Detection** — Upload an image and detect license plates.
- 🎥 **Video Detection** — Process videos and view detected plates.
- 🎯 **Adjustable Confidence** — Control the detection confidence threshold.
- ⚡ **YOLO-Powered Detection** — Use a trained deep learning model.
- 🌐 **Interactive Web App** — Simple interface built with Streamlit.
- 📥 **Download Results** — Download processed videos.

## 🛠️ Technologies Used

- Python
- YOLO (Ultralytics)
- OpenCV
- Streamlit
- ImageIO-FFmpeg

## 📁 Project Structure

```text
License-Plate-Detection/
│
├── app.py
├── requirements.txt
├── best.pt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd License-Plate-Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Trained Model

Place your trained YOLO model, named `best.pt`, in the same folder as `app.py`.

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deploy on Streamlit Community Cloud

1. Upload the project files to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Select `app.py` as the main file.
5. Click **Deploy**.

**Note:** Make sure your trained model is accessible to the deployed application.

## 🎯 Project Goal

The goal of this project is to demonstrate how deep learning and computer vision can be used to detect vehicle license plates through an interactive web application.

## 🔮 Future Improvements

- 🔠 Automatic license plate text recognition using OCR.
- 🚗 Vehicle tracking across video frames.
- 📊 Detection statistics and reporting.
- 💾 Saving detection results for later analysis.

## 👨‍💻 Author

**Abuzar Faisal**

*Exploring AI, Deep Learning, and Computer Vision — one project at a time.*

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, YOLO, and Streamlit.**

