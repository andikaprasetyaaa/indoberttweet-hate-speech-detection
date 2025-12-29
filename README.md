# 🛡️ Indonesian Hate Speech Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

*A Deep Learning-based system for detecting hate speech in Indonesian language using BERT models*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Models](#models) • [API Documentation](#api-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **state-of-the-art hate speech detection system** specifically designed for Indonesian language text. Using transformer-based models (IndoBERT), the system can accurately classify whether a given text contains hate speech with high precision.

### 🔬 Why This Matters

- **Social Media Moderation**: Automatically filter harmful content
- **Real-time Detection**: Fast inference with FastAPI backend
- **Explainable AI**: Identifies trigger words that contribute to hate speech classification
- **Bilingual Support**: Trained on Indonesian language context

---

## ✨ Features

- 🤖 **Multiple Model Support**: IndoBERT Base and IndoBERTweet implementations
- 🎯 **High Accuracy**: 91%+ F1-Score on validation set
- ⚡ **Fast Inference**: Optimized for production use
- 🔍 **Explainable Predictions**: Word-level impact analysis
- 🌐 **REST API**: FastAPI backend for easy integration
- 💻 **Web Interface**: Streamlit-based user-friendly UI
- 📊 **Comprehensive Metrics**: Confusion Matrix, ROC Curve, AUC Score
- 🔧 **Easy Deployment**: Docker-ready configuration

---

## 📁 Project Structure
```
DLCLASSIFYRACIST/
│
├── 📂 __pycache__/                     # Python cache files
│
├── 📂 dataset/
│   ├── dataset_label.json              # Raw labeled dataset
│   ├── dataset_processed.csv           # Processed dataset with numeric labels
│   └── preprocessed_tweets.csv         # Cleaned and normalized tweets
│
├── 📂 preprocessing/                   # Text and label preprocessing utilities
│
├── 📂 detection/                       # (Currently empty / reserved)
│
├── 📂 model_indobertbase/               # Trained IndoBERT Base model
│
├── 📂 model_indoberttweet/              # Trained IndoBERTweet model
│
├── 📂 model_indoberttweet2/             # Fine-tuned IndoBERTweet model
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
│
├── 📄 convert_label.ipynb               # Label conversion notebook
├── 📄 read_dataset.ipynb                # Dataset exploration notebook
├── 📄 detection_hate_speech_indobertbase.ipynb
├── 📄 detection_hate_speech_indoberttweet.ipynb
├── 📄 prediction.ipynb                  # Prediction 
│
├── 📄 main.py                           # FastAPI backend
├── 📄 frontend.py                       # Streamlit frontend
├── 📄 test_api.py                       # API testing script
│
├── 📄 .gitignore                        # Git ignore rules
└── 📄 README.md                         # Project documentation
```

---

## 🔧 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/DLClassifyRacist.git
cd DLClassifyRacist
```

### Step 2: Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n hate_speech python=3.9
conda activate hate_speech

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

Download the trained models from:
🔗 **[Google Drive - Trained Models](https://drive.google.com/drive/folders/1kD4cpCt2MEiqsFKAdLWrkICHd1co7d20?usp=sharing)**

Extract and place the models in the `models/` directory:
```
models/
├── model_indobertbase/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── model_indoberttweet/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

---

## 📊 Dataset

### Dataset Structure

The dataset contains **3,240 labeled Indonesian text samples**:

| Label | Count | Percentage |
|-------|-------|------------|
| `hate_speech` | 1,291 | 39.8% |
| `not_hate_speech` | 1,949 | 60.2% |

### Data Fields

- `id`: Unique identifier
- `cleaned_text`: Preprocessed Indonesian text
- `label`: Binary classification (`hate_speech` / `not_hate_speech`)
- `label_numeric`: Numeric encoding (0 = not_hate_speech, 1 = hate_speech)

### Preprocessing Steps

1. **Text Cleaning**:
   - Remove URLs, mentions, hashtags
   - Normalize whitespace
   - Convert to lowercase
   - Remove special characters

2. **Label Encoding**:
```python
   df['label_numeric'] = df['label'].apply(
       lambda x: 1 if x == 'hate_speech' else 0
   )
```

3. **Data Splitting**:
   - Training: 80% (2,595 samples)
   - Validation: 10% (321 samples)
   - Testing: 10% (324 samples)

---

## 🤖 Models

### 1. IndoBERT Base (`indobenchmark/indobert-base-p2`)

- **Architecture**: BERT Base (12 layers, 768 hidden size)
- **Training**: 3 epochs
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens

**Performance**:
- F1 Score: **91.52%**
- Precision: **91.67%**
- Recall: **91.59%**
- AUC: **0.9777**

### 2. IndoBERTweet (`indolem/indobertweet-base-uncased`)

- **Architecture**: BERT Base optimized for social media
- **Training**: 3 epochs
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens

**Performance**:
- F1 Score: **91.33%**
- Precision: **91.55%**
- Recall: **91.28%**
- AUC: **0.9746**

### Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| General text | IndoBERT Base |
| Social media posts | IndoBERTweet |
| Formal documents | IndoBERT Base |
| Twitter/casual text | IndoBERTweet |

---

## 🚀 Usage

### 1. Quick Prediction (Jupyter Notebook)
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
MODEL_DIR = './models/model_indobertbase'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Predict
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    labels = {0: "Not Hate Speech", 1: "Hate Speech"}
    return labels[prediction]

# Test
text = "Selamat pagi, semoga harimu menyenangkan!"
print(predict(text))  # Output: "Not Hate Speech"
```

### 2. Running the FastAPI Backend
```bash
# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

**API Documentation**: `http://localhost:8000/docs`

### 3. Running the Web Interface
```bash
# Start Streamlit app
streamlit run api/frontend.py
```

The web interface will open automatically in your browser at: `http://localhost:8501`

### 4. Testing the API
```bash
python scripts/test_api.py
```

Or use curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Contoh teks untuk dianalisis"}'
```

---

## 📡 API Documentation

### Endpoint: `/predict`

**Method**: `POST`

**Request Body**:
```json
{
  "text": "Your Indonesian text here"
}
```

**Response**:
```json
{
  "text": "Your Indonesian text here",
  "prediction": "Ujaran Kebencian",
  "original_confidence": 0.9234,
  "is_hate_speech": true,
  "trigger_analysis": [
    {
      "word": "bodoh",
      "confidence_without_word": 0.3421,
      "impact_drop": 0.5813
    }
  ],
  "process_time_seconds": 0.234
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Original input text |
| `prediction` | string | Classification label |
| `original_confidence` | float | Probability of hate speech (0-1) |
| `is_hate_speech` | boolean | Binary classification result |
| `trigger_analysis` | array | Words contributing to classification |
| `process_time_seconds` | float | Inference time in seconds |

---

## 📈 Results

### Confusion Matrix

<div align="center">

| | Predicted: Not Hate | Predicted: Hate |
|---|---|---|
| **Actual: Not Hate** | 174 (TN) | 19 (FP) |
| **Actual: Hate** | 9 (FN) | 122 (TP) |

</div>

### Classification Report
```
              precision    recall  f1-score   support

   Non-Racist     0.9508    0.9016    0.9255       193
       Racist     0.8623    0.9297    0.8947       128

     accuracy                         0.9128       321
    macro avg     0.9066    0.9156    0.9101       321
 weighted avg     0.9167    0.9159    0.9152       321
```

### ROC Curve

- **AUC Score**: 0.9777 (IndoBERT Base)
- **AUC Score**: 0.9746 (IndoBERTweet)

---

## 🎨 Web Interface Features

### Main Dashboard

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b0ae05d5-3edb-4b26-aa8e-eba8a39dbb86" />


1. **Text Input**: Large text area for entering Indonesian text
2. **Real-time Analysis**: Instant classification results
3. **Confidence Score**: Visual confidence meter
4. **Trigger Words**: Highlighted words contributing to classification
5. **Processing Time**: Performance metrics display

### Features

- ✅ Clean, modern UI built with Streamlit
- ✅ Real-time prediction
- ✅ Word-level explanation
- ✅ Confidence visualization
- ✅ Responsive design

---

## 🔬 Training Pipeline

### 1. Data Preprocessing
```bash
jupyter notebook notebooks/convert_label.ipynb
```

- Converts string labels to numeric format
- Handles class imbalance with computed weights
- Saves processed dataset as CSV

### 2. Model Training
```bash
# Train IndoBERT Base
jupyter notebook notebooks/detection_hate_speech_indobertbase.ipynb

# Train IndoBERTweet
jupyter notebook notebooks/detection_hate_speech_indoberttweet.ipynb
```

**Training Configuration**:
- Optimizer: AdamW
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Loss Function: CrossEntropyLoss with class weights
- Early Stopping: Based on validation F1 score

### 3. Evaluation
```bash
jupyter notebook notebooks/detection_hate_speech_indobertbase.ipynb
# Run evaluation cells for:
# - Confusion Matrix
# - ROC Curve
# - Classification Report
# - AUC Score
```

---

## 🐳 Docker Deployment (Optional)

### Build Docker Image
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Run Container
```bash
docker build -t hate-speech-detector .
docker run -p 8000:8000 hate-speech-detector
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
```bash
# Solution: Reduce batch size or use CPU
device = torch.device("cpu")
```

**Issue**: `Model files not found`
```bash
# Solution: Ensure models are downloaded and extracted correctly
# Check path: models/model_indobertbase/pytorch_model.bin
```

**Issue**: `Tokenizer errors`
```bash
# Solution: Reinstall transformers
pip install --upgrade transformers
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where possible
- Write unit tests for new features

---

## 📚 References

- [IndoBERT Paper](https://arxiv.org/abs/2009.05387)
- [IndoBERTTweet Paper](10.18653/v1/2020.coling-main.66)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- IndoBERT team for the pre-trained models
- Hugging Face for the transformers library
- Indonesian NLP community for dataset contributions
- All contributors who helped improve this project

---

## 📞 Contact

For questions or feedback:

- **Email**: andikaprasetyacomid@gmail.com.com
- **Instagram**: https://www.instagram.com/diiicka?igsh=dXU0NHoxemJ0MTlu 

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ and 🤖 for Indonesian NLP

</div>
```
