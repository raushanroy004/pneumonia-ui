🩺 Paediatric Pneumonia Detector (Chest X-Ray · DenseNet121)

A deep-learning powered diagnostic tool built using DenseNet121 to classify pediatric chest X-rays as Normal or Pneumonia, with Grad-CAM heatmap explainability and a clean Streamlit UI for real-time medical assistance.
This project aims to support radiologists with faster, more reliable, and interpretable diagnosis.

🚀 Features

✔ CNN-based pneumonia detection using DenseNet121

✔ 99.8% AUC score on pediatric chest X-ray dataset

✔ Grad-CAM visualizations to highlight infected lung regions

✔ Streamlit UI for easy image upload and live prediction

✔ Real-time probability scores & classification output

✔ Supports medical decision-making in low-resource environments

🧠 Model & Techniques

Architecture: DenseNet121 with modified classifier head

Loss Function: BCEWithLogitsLoss

Optimizer: AdamW

Data Augmentation: rotation, horizontal flip, color jitter

Preprocessing: grayscale → RGB conversion, resizing to 224×224

Explainability: Grad-CAM heatmaps for model interpretation

Frameworks: PyTorch, Torchvision, Matplotlib, Streamlit

🔄 Workflow

Upload Chest X-Ray Image

Preprocessing: RGB conversion, normalization, resizing

CNN Inference: DenseNet121 predicts pneumonia probability

Explainability: Grad-CAM highlights affected lung regions

Output: Final label (Normal/Pneumonia) + heatmap overlay

<br>

(See workflow diagram in the project PDF 

Paediatric Pneumonia Detector (…

)

📊 Results:

AUC Score: 99.8%

Strong performance on test dataset (Confusion Matrix on page 10) 

Paediatric Pneumonia Detector (…

Grad-CAM shows medically meaningful regions, proving reliable decision-making (page 11) 

Paediatric Pneumonia Detector (…

🌐 Live Demo

🔗 Streamlit App:
https://pneumonia-ui-fnrwuargjgvswidka9jbyd.streamlit.app/

📁 Dataset

Pediatric Chest X-Ray dataset from Kaggle
(With Normal & Pneumonia classes)

🔮 Future Improvements:-

Multi-disease classification (TB, COVID-19, lung cancer)

Integration with patient history & clinical data

More advanced XAI methods beyond Grad-CAM

Multi-center dataset expansion for better generalization
