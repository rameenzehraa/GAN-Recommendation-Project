# GAN vs Collaborative Filtering: Robustness Analysis

## Project Overview
Comparing the robustness of GAN-based and Collaborative Filtering recommendation systems against adversarial noise and data corruption.

## Team Members
- **Person 1:** Data Engineering & CF Implementation
- **Person 2:** GAN Model Development & Training
- **Person 3:** Evaluation & Analysis

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/GAN-Recommendation-Project.git
cd GAN-Recommendation-Project
```

2. **Create virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download dataset:**
```bash
cd data
# Windows PowerShell
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-1m.zip" -OutFile "ml-1m.zip"
Expand-Archive -Path "ml-1m.zip" -DestinationPath "." -Force

# Mac/Linux
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip

cd ..
```

5. **Test the setup:**
```bash
python src/data_loader.py
python src/noise_injector.py
```

## Project Structure
```
GAN-Recommendation-Project/
├── data/                          # Dataset files
├── models/                        # Saved trained models
├── results/                       # Evaluation results & charts
├── notebooks/                     # Jupyter notebooks for exploration
├── src/                          
│   ├── data_loader.py            # ✅ COMPLETE - Loads MovieLens data
│   ├── noise_injector.py         # ✅ COMPLETE - Injects noise
│   ├── collaborative_filtering.py # 🔴 TODO - Person 1
│   ├── gan_model.py              # 🔴 TODO - Person 2
│   ├── train.py                  # 🔴 TODO - Person 2
│   ├── evaluation.py             # 🔴 TODO - Person 3
│   └── visualize.py              # 🔴 TODO - Person 3
├── README.md
└── requirements.txt
```

## Work Division

### Person 1: Data & Baseline Implementation
**Tasks:**
- ✅ Data loader (DONE)
- ✅ Noise injector (DONE)
- 🔴 Implement Collaborative Filtering in `collaborative_filtering.py`
- 🔴 Create training pipeline for CF
- 🔴 Save/load CF model utilities

**Files to work on:**
- `src/collaborative_filtering.py`

### Person 2: GAN Model & Training
**Tasks:**
- 🔴 Implement GAN architecture in `gan_model.py`
- 🔴 Create training loop in `train.py`
- 🔴 Train both models (GAN + CF)
- 🔴 Hyperparameter tuning

**Files to work on:**
- `src/gan_model.py`
- `src/train.py`

### Person 3: Evaluation & Analysis
**Tasks:**
- 🔴 Implement evaluation metrics in `evaluation.py`
- 🔴 Create visualization code in `visualize.py`
- 🔴 Test models on clean + noisy data
- 🔴 Generate comparison charts
- 🔴 Write analysis report

**Files to work on:**
- `src/evaluation.py`
- `src/visualize.py`

## Current Status
- [x] Project setup
- [x] Data loader
- [x] Noise injection
- [ ] Collaborative Filtering model
- [ ] GAN model
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Results & analysis

## Timeline
- **Days 1-2:** Setup & Data ✅
- **Days 2-4:** Model Implementation (IN PROGRESS)
- **Days 4-5:** Testing & Evaluation
- **Days 5-6:** Analysis & Visualization
- **Day 7:** Report & Presentation

## Contact
- Create issues for questions/problems
- Use pull requests for code contributions
