# Automatic-image-captioning-with-confidence-score
Image Caption Generator using Pretrained ViT-GPT2
📌 Project Overview

The Image Caption Generator is an Artificial Intelligence application that automatically generates descriptive captions for images. The project uses a pretrained Vision Transformer (ViT) and GPT-2 model from the Hugging Face Transformers library to analyze image content and generate meaningful natural language descriptions.

The application is developed using Python and deployed through a Streamlit web interface, allowing users to upload images and receive captions in real time. The system also displays a confidence/accuracy score to indicate the reliability of the generated caption.

🚀 Features
Upload images through a simple web interface
Generate captions automatically using a pretrained ViT-GPT2 model
Display confidence/accuracy score for generated captions
Real-time image caption generation
GPU support for faster inference
User-friendly Streamlit interface
Automatic image preprocessing using ViTImageProcessor
🛠️ Technologies Used
Python
PyTorch
Hugging Face Transformers
Streamlit
Pillow (PIL)
VisionEncoderDecoderModel
ViTImageProcessor
GPT-2
Vision Transformer (ViT)
🏗️ System Architecture
Input Image
      │
      ▼
Image Preprocessing
(ViTImageProcessor)
      │
      ▼
Vision Transformer (ViT)
Image Encoder
      │
      ▼
Feature Extraction
      │
      ▼
GPT-2 Decoder
Caption Generation
      │
      ▼
Tokenizer
      │
      ▼
Generated Caption
      │
      ▼
Confidence / Accuracy Score
⚙️ How It Works
The user uploads an image through the Streamlit application.
The image is converted into RGB format and preprocessed using ViTImageProcessor.
The Vision Transformer (ViT) extracts visual features from the image.
The GPT-2 decoder generates a meaningful caption based on the extracted features.
The tokenizer converts the generated tokens into readable text.
The generated caption and confidence score are displayed to the user.
📊 Model Used

This project uses a pretrained VisionEncoderDecoderModel consisting of:

Vision Transformer (ViT) as the image encoder
GPT-2 as the language decoder

The model is already trained on large image-caption datasets and can generate captions without additional training.

📂 Project Structure
Image-Caption-Generator/
│
├── app.py
├── requirements.txt
├── README.md
├── images/
└── output/
▶️ Installation

Clone the repository:

git clone <repository-link>
cd Image-Caption-Generator

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
💡 Applications
Assistive technology for visually impaired people
Social media caption generation
Multimedia content analysis
Digital asset management
Image search and retrieval systems
Smart education platforms
E-commerce product description generation
