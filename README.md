# 🐠 Coral Health Classification Using CNN & Transfer Learning 🌊

This project aims to **automatically classify coral reef images** as either **Healthy 🪸** or **Bleached ⚪**, using **Convolutional Neural Networks (CNNs)** and **transfer learning** techniques (ResNet50 and EfficientNetB0).

It includes:
- Full **data exploration, preprocessing, and augmentation**
- **Model training and evaluation**
- A deployed **Streamlit web app** for real‑time coral health prediction

---

## 📦 Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification).

It consists of two folders:
bleached_corals/ → 485 images healthy_corals/ → 483 images


Each image shows a coral specimen labeled as either **bleached** or **healthy**.

---

## 🧠 Project Workflow

1. **Exploratory Data Analysis (EDA)**
   - Image counts, class distributions, and sample visualization

2. **Data Augmentation**
   - Random flips, rotations, zooms, and shifts using `ImageDataGenerator`  
   - Helps improve model generalization with a small dataset

3. **Dataset Splitting**
   - Train / Validation / Test sets using stratified splits

4. **Model Training**
   - Two transfer learning models used:
     - 🌀 **ResNet50**
     - ⚡ **EfficientNetB0**
   - Each model fine‑tuned and evaluated for accuracy & loss

5. **Model Evaluation**
   - Accuracy and loss plots for training vs. validation
   - Confusion matrix and classification reports per class

6. **Deployment**
   - A lightweight **Streamlit app** allows users to upload coral images and predict health condition

---

## 💻 Streamlit Web App

You can try the interactive demo here:  
👉 **[Coral Health Classifier App](YOUR_APP_LINK_HERE)**  

Upload a coral photo, and the app predicts whether it's **Healthy** or **Bleached**, along with the **confidence score**.

---

## 🧾 Notebook Overview

The Jupyter Notebook includes:

- `Section 1–2:` Data loading and EDA  
- `Section 3:` Dataset splitting  
- `Section 4:` Data augmentation  
- `Section 5–6:` Training CNN models (ResNet & EfficientNet)  
- `Section 7–10:` Evaluation (metrics, plots, confusion matrix)  
- `Section 11:` Test image visualizations with predictions  
- `Section 12:` Future improvement ideas  
- **Streamlit app integration guide**

---

## 🚀 Installation & Setup

Clone the repository and install dependencies:

```bash
git clone [github.com](https://github.com/yourusername/coral-health-classification.git)
cd coral-health-classification

pip install -r requirements.txt

Run the training notebook or the Streamlit app:
streamlit run app.py

🧩 Technologies Used
- Python 3.10+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Pillow

📈 Results Summary
|Model	|Test Accuracy	|Notes|
|ResNet50	|~76.7%	|Good baseline|
|EfficientNetB0|	~81%	|Higher accuracy and better generalization|

🌺 Future Improvements
- Add Grad‑CAM heatmaps for explainability
- Integrate advanced hyperparameter tuning
- Expand dataset diversity & size
- Deploy model via Streamlit Cloud or Hugging Face Spaces

📞 Acknowledgments
- Dataset from [kaggle.com]
- TensorFlow & Keras for the transfer learning architectures
- Streamlit for the web app interface
