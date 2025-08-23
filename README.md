# 🌱 Plant Disease Prediction (38 Classes) using VGG16 + Streamlit

This project uses **Transfer Learning (VGG16)** to classify **38 plant leaf diseases** with fine-tuning, class balancing, and augmentation.  
It includes a **Streamlit app** for interactive predictions.

---

## 📂 Project Structure
- `notebooks/training.ipynb` → Full Colab-ready workflow (data preprocessing, training, fine-tuning, evaluation)
- `streamlit_app/app.py` → Streamlit web app for predictions
- `models/plant_vgg16_best.h5` → Trained model
- `models/class_indices.json` → Class mapping
- `dataset/README.md` → Dataset download instructions

---

## 📊 Dataset
We use the **PlantVillage dataset (38 classes, 50K+ images)**:  
🔗 [Kaggle Link](https://www.kaggle.com/datasets/emmarex/plantdisease)

> ⚠️ The dataset is not uploaded due to GitHub size limits.  
Download from Kaggle and place it under `dataset/`.

---

## 🚀 Training
Run the notebook in Colab:
```bash
notebooks/training.ipynb
