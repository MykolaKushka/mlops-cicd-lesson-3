# Lesson 3 — Image Classification with PyTorch and Docker

## Project Description
This project demonstrates how to create, package, and run a deep learning model using **PyTorch** and **Docker**.

The model classifies images using a pre-trained **MobileNetV2** network from the ImageNet dataset.  
Two versions of the Docker image were built:  
1. fat-model — full version with all dependencies.  
2. slim-model — lightweight version optimized for size.  

---

## Environment Setup (Local)
To recreate the virtual environment:

python3 -m venv venv  
source venv/bin/activate  
pip install --upgrade pip  
pip install torch torchvision pillow  

Run inference locally:  
python inference.py  

---

## Docker Instructions

1️ Build the fat Docker image:  
docker build -t fat-model -f Dockerfile.fat .  

2️ Run the model in a container:  
docker run --rm fat-model  

3️ Build the slim Docker image:  
docker build -t slim-model -f Dockerfile.slim .  

4️ Run inference with mounted image:  
docker run --rm -v $(pwd)/test.jpg:/app/test.jpg slim-model  

---

## Project Structure

lesson-3/  
│  
├── inference.py          — Main Python script for inference  
├── model.pt              — TorchScript saved model  
├── Dockerfile.fat        — Full-size Docker image definition  
├── Dockerfile.slim       — Slim image definition  
├── test.jpg              — Sample test image  
└── README.md             — Project documentation  

---

## Output Example
When running inference (both locally or in Docker):

1. sandbar: 9.71%  
2. seashore: 6.88%  
3. cliff: 3.33%  

---

## Cleanup
To free up space after testing:

docker system prune -a  
rm -rf venv  
rm -rf ~/.cache/torch  

---

## Author
**Кушка Микола**  
ITGo Data Science and Data Analytics — Lesson 3  
Autumn 2025
