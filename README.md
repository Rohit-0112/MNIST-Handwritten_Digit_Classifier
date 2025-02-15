
MNIST Handwritten Digit Classifier
This project is a deep learning-based classifier for recognizing handwritten digits from the MNIST dataset using PyTorch.

🚀 Project Overview
The MNIST Handwritten Digit Classifier is built using PyTorch, NumPy, and Pandas to train a neural network on the famous MNIST dataset. It can recognize digits (0-9) with high accuracy.


📊 Dataset
The dataset contains 28x28 grayscale images of handwritten digits (0-9).
Training Set: 60,000 images
Test Set: 10,000 images
Automatically downloaded using torchvision.datasets.MNIST

🔧 Setup & Installation
1️⃣ Clone the repository
git clone https://github.com/Rohit-0112/MNIST-Handwritten_Digit_Classifier.git

cd MNIST-Handwritten_Digit_Classifier

2️⃣ Create & activate a virtual environment

python -m venv mnist_env
source mnist_env/bin/activate  # On Windows: mnist_env\Scripts\activate


3️⃣ Install dependencies

pip install -r requirements.txt


4️⃣ Run the Jupyter Notebook

jupyter notebook


🏋️ Training the Model
To train the model, run the Jupyter Notebook (mnist_model-checkpoint.ipynb) or execute:
python train.py


📌 Technologies Used
Python 🐍
PyTorch 🔥
NumPy & Pandas 📊
Matplotlib & Seaborn 📈
Jupyter Notebook


🎯 Results & Accuracy

## 🔹 Training Details
- Loss Function: **CrossEntropyLoss**
- Optimizer: **Adam (learning rate = 0.001)**
- Training Epochs: **5**
- Achieved Test Accuracy: **97.02 %**


🤝 Contributing
Feel free to fork this repository, create a branch, make changes, and submit a pull request!
