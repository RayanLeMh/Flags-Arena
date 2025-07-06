import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
CORS(app)
# Load pretrained models
resnet = models.resnet50(weights='IMAGENET1K_V1')
squeezenet = models.squeezenet1_1(weights='IMAGENET1K_V1')
resnet.eval()
squeezenet.eval()

# Preprocessing for the models
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load ImageNet class labels
with open('imagenet_class_index.json') as f:
    imagenet_classes = json.load(f)

lion_indices = [int(idx) for idx, (code, _) in imagenet_classes.items() if code == "n02129165"]

# cat_indices = []
cat_codes = [
    "n02123045",  # tabby cat
    "n02123159",  # tiger cat
    "n02123394",  # Persian cat
    "n02123597",  # Siamese cat
    "n02124075",  # Egyptian cat
]

cat_indices = [int(idx) for idx, (code, _) in imagenet_classes.items() if code in cat_codes]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = Image.open(file.stream).convert('RGB')
        
        # Process image for model inference
        img_tensor = preprocess(img)
        img_batch = img_tensor.unsqueeze(0)
        
        # Get predictions from both models
        with torch.no_grad():
            resnet_output = resnet(img_batch)
            squeezenet_output = squeezenet(img_batch)
        
        # Get top predictions
        _, resnet_preds = torch.max(resnet_output, 1)
        resnet_idx = resnet_preds.item()
        
        # For SqueezeNet, get probabilities
        squeezenet_probs = torch.nn.functional.softmax(squeezenet_output, dim=1)[0]
        squeezenet_top5_prob, squeezenet_top5_idx = torch.topk(squeezenet_probs, 5)
        
        # Get class names
        resnet_class = imagenet_classes[str(resnet_idx)]
        
        squeezenet_top5 = []
        for i in range(5):
            idx = squeezenet_top5_idx[i].item()
            prob = squeezenet_top5_prob[i].item()
            squeezenet_top5.append({
                'class': imagenet_classes[str(idx)],
                'probability': round(prob * 100, 2)
            })
        
        # Check if the challenge is solved
        is_lion = resnet_idx in lion_indices

        top1_squeezenet_idx = squeezenet_top5_idx[0].item()
        is_squeezenet_cat = top1_squeezenet_idx in cat_indices
        # is_cat_in_squeezenet = any(idx.item() in cat_indices for idx in squeezenet_top5_idx)
        
        challenge_solved = is_lion and is_squeezenet_cat
        
        # Prepare results
        result = {
            'resnet_class': resnet_class,
            'squeezenet_top5': squeezenet_top5,
            'is_lion': is_lion,
            'is_squeezenet_cat': is_squeezenet_cat,
            'challenge_solved': challenge_solved
        }
        
        if challenge_solved:
            result['flag'] = open('flag.txt').read().strip()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if not os.path.exists('imagenet_class_index.json'):
        print("Error: imagenet_class_index.json not found!")
        print("Please download it from https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
        exit(1)
    app.run(host="0.0.0.0", port=5000, debug=True)