---

# 🦅 Hawkeye - AI Object Detection & Tracking

Hawkeye is an **AI-powered object detection system** built with **YOLOv8** and **Streamlit**.
It can detect and track objects in **images, videos, and live webcam feeds**, providing real-time summaries of what it sees.

---

## ✨ Features

* 🖼️ **Image Detection** → Upload an image and get detected objects with bounding boxes + counts
* 🎬 **Video Detection** → Upload a video and see real-time object detection frame by frame
* 📷 **Webcam Detection** → Live object detection directly from your webcam (works locally)
* 📊 **Results Box** → Summarizes object names and counts (e.g., Person (3), Car (2))
* 📥 **Download Option** → Save your processed image with detections
* 🎨 **Custom Themed UI** → Purple + Teal theme with a clean modern layout

---

## 🚀 How to Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Hawkeye.git
   cd Hawkeye
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run detection_app.py
   ```

4. Open the app in your browser at:

   ```
   http://localhost:8501
   ```

---

## 📂 Project Structure

```
📦 Hawkeye
 ┣ 📜 detection_app.py   # Main Streamlit app
 ┣ 📜 detector.py        # YOLOv8 detection logic
 ┣ 📜 requirements.txt   # Required dependencies
 ┣ 📜 README.md          # Project documentation
```

---

## 🌐 Deployment

* Deployable on **Streamlit Cloud** for **Image + Video Detection**.
* **Webcam mode works only locally** (not supported on Streamlit Cloud for security reasons).

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **NumPy**

---

## 🙌 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
* [Streamlit](https://streamlit.io) for an easy-to-build web interface
* Open datasets & stock resources for testing (COCO, Pexels, Pixabay)

---

## 📸 Demo Screenshots

*(Add here once you run and capture screenshots of Hawkeye detecting objects in images and videos.)*

---