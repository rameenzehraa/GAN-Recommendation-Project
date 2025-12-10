# GAN vs Collaborative Filtering: Robustness Analysis

## Project Overview
Comparing the robustness of **GAN-based** and **Collaborative Filtering (CF)** recommendation systems against adversarial noise and data corruption.

## Team Members
- **Nabira Khan**  
- **Rameen Zehra**  
- **Aisha Asif**  

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Set up repository:**
```bash
git clone https://github.com/YOUR_USERNAME/GAN-Recommendation-Project.git
cd GAN-Recommendation-Project
Create virtual environment:

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Download dataset:

cd data
# Windows PowerShell
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-1m.zip" -OutFile "ml-1m.zip"
Expand-Archive -Path "ml-1m.zip" -DestinationPath "." -Force

# Mac/Linux
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
cd ..
Test the setup:

python src/data_loader.py
python src/noise_injector.py

```
```Project Structure

GAN-Recommendation-Project/
├── data/                          # Dataset files
├── models/                        # Saved trained models
├── results/                       # Evaluation results & charts
├── notebooks/                     # Jupyter notebooks / analysis.py
├── src/                          
│   ├── data_loader.py            
│   ├── noise_injector.py         
│   ├── collaborative_filtering.py 
│   ├── gan_model.py              
│   ├── train.py                  
│   ├── evaluation.py             
│   └── visualize.py              
├── README.md
└── requirements.txt

```
Timeline

Days 1-2: Setup & Data

Days 2-4: Model Implementation

Days 4-5: Testing & Evaluation

Days 5-6: Analysis & Visualization

Day 7: Report & Presentation


