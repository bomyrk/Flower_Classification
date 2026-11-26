import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model_b.pth'  # your saved ResNet18 checkpoint
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')  # set via env var
# Ollama (free, local) — set OLLAMA_MODEL to enable, e.g. OLLAMA_MODEL=llama3.2:3b
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma3:4b')
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
# Set OLLAMA_VISION=1 only if OLLAMA_MODEL is a vision model (llava, moondream, gemma3, ...)
OLLAMA_VISION = '1' #os.environ.get('OLLAMA_VISION', '') == '1'

# Class names (0 to 101) — canonical Flowers102 order, same as torchvision.datasets.Flowers102.classes
CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia?", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower",
    "tree poppy", "gazania", "azalea", "water lily", "rose",
    "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
    "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily"
]
assert len(CLASS_NAMES) == 102
# --- Load model ---
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 102)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
except FileNotFoundError:
    print(f"WARNING: {MODEL_PATH} not found — run the notebook training cell first. "
          "The app will launch but predictions are disabled.")
    model = None
# --- Transform (same as val/test) ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict(image):
    if image is None:
        return {}, ""

    image = image.convert('RGB')

    if model is None:
        return {}, "Model checkpoint not found. Run the notebook training cell to generate best_model_b.pth."

    # Run model
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
    probs = torch.softmax(output, dim=1)[0]
    top5_probs, top5_idx = torch.topk(probs, 5)

    label_dict = {}
    for i in range(5):
        label_dict[CLASS_NAMES[top5_idx[i].item()]] = top5_probs[i].item()

    # Get AI extra info (OpenAI if key set, otherwise local Ollama)
    ai_info = ""
    try:
        from openai import OpenAI

        predicted_class = CLASS_NAMES[top5_idx[0].item()]
        prompt_text = (#f"This flower is classified as '{predicted_class}'. "
                       f"Give me: 1) What species of flower is this? Answer with just the species name from this list {CLASS_NAMES}"
                       f"2) a one-line description, 3) native habitat, "
                       f"4) one fun fact, 5) care difficulty (easy/medium/hard). "
                       f"Keep it under 100 words.")

        if OPENAI_API_KEY:
            client = OpenAI(api_key=OPENAI_API_KEY)
            model_name = "gpt-4o"
        elif OLLAMA_MODEL:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            model_name = OLLAMA_MODEL
        else:
            client = None

        if client is not None:
            content = [{"type": "text", "text": prompt_text}]
            if OPENAI_API_KEY or OLLAMA_VISION:
                import base64
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format='JPEG')
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=200
            )
            ai_info = response.choices[0].message.content
    except Exception as e:
        ai_info = f"AI info unavailable: {str(e)}"

    return label_dict, ai_info

# --- Gradio UI ---
with gr.Blocks(title="Flower Classifier + AI Info") as demo:
    gr.Markdown("# 🌸 Flower Species Classifier\nUpload a flower photo — get the species + AI-generated info.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload a flower image", type="pil")
        with gr.Column():
            label_output = gr.Label(label="Top 5 Predictions")
            ai_output = gr.Textbox(label="AI Info", lines=6)

    btn = gr.Button("Classify")
    btn.click(fn=predict, inputs=img_input, outputs=[label_output, ai_output])

    gr.Examples(
        examples=[
            ".data/flowers-102/jpg/image_06734.jpg",  # pink primrose
            ".data/flowers-102/jpg/image_03095.jpg",  # snapdragon
            ".data/flowers-102/jpg/image_01491.jpg",  # poinsettia
            ".data/flowers-102/jpg/image_02753.jpg",  # bishop of llandaff
            ".data/flowers-102/jpg/image_07466.jpg",  # bougainvillea
        ],
        inputs=img_input
    )

if __name__ == "__main__":
    demo.launch()   
