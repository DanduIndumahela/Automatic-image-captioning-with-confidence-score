import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# -----------------------------
# UI Configuration
# -----------------------------
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image to generate a caption with an accuracy score.")

# -----------------------------
# Load Model (Cached for speed)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

# -----------------------------
# ✅ PROPER ACCURACY CALCULATION
# -----------------------------
def generate_caption_with_accuracy(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate output with scores
    outputs = model.generate(
        pixel_values, 
        max_length=16, 
        return_dict_in_generate=True, 
        output_scores=True
    )

    # Decode IDs to text
    output_ids = outputs.sequences[0]
    caption = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Calculate Confidence (Accuracy)
    # Stack logits for all generated tokens
    logits = torch.stack(outputs.scores, dim=1) 
    probs = F.softmax(logits, dim=-1)

    # Align generated IDs with probabilities (ignoring the BOS token)
    gen_ids = output_ids[1:].unsqueeze(0).unsqueeze(-1)
    token_probs = probs.gather(2, gen_ids).squeeze()

    # The mean probability represents the "Accuracy" of the model's guess
    if token_probs.dim() == 0:
        accuracy = token_probs.item()
    else:
        accuracy = token_probs.mean().item()

    return caption, accuracy * 100

# -----------------------------
# Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Output Section
    # -----------------------------
    if st.button("Generate Caption"):
        with st.spinner("Analyzing image and calculating accuracy..."):
            caption, accuracy = generate_caption_with_accuracy(image)

            # Display Caption
            st.success(f"**📌 Caption:** {caption}")
            
            # Display Accuracy Score
            st.info(f"**📊 Accuracy:** {accuracy:.2f}%")
            
            # Visual Progress Bar for Accuracy
            st.progress(int(min(accuracy, 100)))
# -----------------------------