🌱 Crop Disease Detection System using Deep Learning

An AI-powered crop disease detection system that identifies plant diseases from leaf images using deep learning. The system provides disease classification, severity estimation, explainable AI visualizations (Grad-CAM), reliability scoring, treatment recommendations, live webcam detection, and detection history logging.

🚀 Features

Upload leaf image for disease detection

Live webcam-based disease detection

Crop-type selection and crop-based prediction filtering

CNN-based multi-class disease classification

Severity estimation (percentage and level)

Grad-CAM explainable heatmaps

Reliability score for predictions

Top-K prediction display

Treatment and prevention recommendations

Detection history stored using SQLite

Clean and professional Streamlit UI

🧠 Tech Stack

Python

TensorFlow / Keras

OpenCV

Streamlit

NumPy

SQLite

📂 Project Structure
crop_disease_ai/
│
├── app.py
├── model/
│   └── trained_model.keras
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── utils/
│   ├── predict.py
│   ├── gradcam.py
│   ├── severity.py
│   ├── heatmap_score.py
│   ├── leaf_check.py
│   ├── crop_filter.py
│   ├── database.py
│   └── recommendations.py
├── database/
│   └── history.db
├── requirements.txt
└── README.md

⚙️ Installation
1. Clone the repository
git clone (https://github.com/jazzkay/crop_disease_detection_ai)
cd crop-disease-detection

2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py


Open browser at:

http://localhost:8501

🖼️ How It Works

User uploads leaf image or uses webcam

Leaf presence is verified

CNN predicts top disease classes

Predictions are filtered using selected crop

Severity and reliability are calculated

Grad-CAM highlights infected regions

Recommendations and history are displayed

📊 Dataset

A multi-crop, multi-disease image dataset collected from public agricultural sources and Kaggle, covering crops such as:

Rice

Maize

Cotton

Wheat

Sugarcane

🔬 Explainable AI

Grad-CAM visualizations show which regions of the leaf influenced the model’s decision, improving transparency and trust.

🧪 Limitations

Visually similar diseases may be confused

Model accuracy depends on image quality

Best results with clear, close-up leaf images

🔮 Future Improvements

Two-stage hierarchical classification (crop → disease)

Mobile application deployment

Cloud-based API service

Multi-language interface

PDF diagnostic report export

👩‍💻 Author

Jaspreet Kaur
