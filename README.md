🦴 Bone Injury Diagnostic AI System
A Python-based AI diagnostic system that uses trained models to detect bone injuries from X-ray images and generates detailed diagnostic reports with treatment suggestions.

🔍 Overview
This system detects bone fractures (e.g., wrist, hand, elbow, shoulder, forearm, humerus, finger) from X-ray images using PyTorch CNN models , and generates PDF diagnostic reports with:

Patient information
Fracture diagnosis
Treatment suggestions (based on medical guidelines)
Embedded X-ray image
Healing timeline
🧩 Features
✅ Supports 7 body parts:

Wrist
Hand
Elbow
Shoulder
Forearm
Humerus
Finger
✅ Binary classification per body part:

Fracture / No Fracture
✅ Rule-based treatment suggestions from medical knowledge

✅ PDF report generation with embedded X-ray image

✅ Manual input of X-ray path (no need to move files)

🛠️ Technologies Used
Python
Core language
PyTorch
Model training & inference
ReportLab
PDF report generation
Pillow (PIL)
Image processing
TorchVision
Pretrained CNN models
Scikit-learn
Accuracy metrics
Tkinter / Streamlit (optional)
GUI/Web interface

📁 Folder Structure

Bone-Diagnosis/
├── input_xrays/
│   ├── wrist/
│   ├── hand/
│   └── ... other body parts
├── models/
│   ├── wrist.pth
│   ├── hand.pth
│   └── ... one model per body part
├── Model_Selector.py
├── Predictor.py
├── Treatment.py
└── Report_Generator.py


🧪 How to Run
Step-by-step:
Clone or open the project directory
Install dependencies:
bash


1
pip install torch torchvision reportlab Pillow scikit-learn
Run the diagnostic report generator:
bash


1
python generate_diagnostic_report_manual_upload.py
Input:
Patient data
Select body part
Enter full path to X-ray image (e.g., "C:\X-rays\wrist_image.png")
Output:
A PDF report saved as [PatientName]_Diagnosis.pdf

⚠️ Notes
Each body part has its own trained .pth model
Models are binary classifiers: fracture vs normal
Does not currently support displacement detection
Uses rule-based treatment suggestions from medical books
Designed for easy extension to multi-class classification

🚀 Future Improvements
Add multi-class fracture severity detection (non-displaced, displaced, comminuted)
Integrate medical book data for smarter treatment suggestions
Auto-detect body part from image
Build a Streamlit web app version
Use DICOM metadata if available
Support HIPAA-compliant patient history storage
