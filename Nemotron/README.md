# Text Classification Pipeline Comparison
## Nvidia Nemotron Model Reasoning Challenge

A comprehensive machine learning analysis comparing five distinct pipelines for multi-class text classification.

---

## 📋 Quick Start

### View Results Immediately
1. **Performance Summary:** Open `Nemotron/output/pipeline_results.csv`
2. **Visualizations:** View `.png` files in `Nemotron/output/`
3. **Full Analysis:** Read `TEXT_CLASSIFICATION_REPORT.md`

### Run the Code
```bash
cd CoderGym
python Nemotron/text_classification_pipelines.py
```

---

## 📊 Project Overview

**Objective:** Build and compare multiple ML pipelines for text classification

**Dataset:** 9,500 text examples from Nvidia Nemotron Challenge
- 6 task types: bit manipulation, cipher text, gravity, roman, symbol transform, unit conversion
- Balanced classes (16-17% each)
- 80/20 train-test split

**Pipelines Evaluated:** 5
- Naive Bayes (TF-IDF) ⭐ **RECOMMENDED**
- SVM (Count+SVD)
- Random Forest (TF-IDF)
- K-Nearest Neighbors (Count+SVD)
- Neural Network (FastText)

---

## 🎯 Key Findings

### Performance Results

| Model | Accuracy | Train Time | Inference |
|-------|----------|-----------|-----------|
| **Naive Bayes** | **100%** | **0.140s** | **0.025s** |
| Random Forest | 100% | 0.689s | 0.056s |
| Neural Network | 100% | 19.903s | 0.002s |
| SVM | 99.74% | 8.444s | 0.517s |
| KNN | 72.37% | 0.002s | 1.611s |

### Recommendation

**Use Naive Bayes (TF-IDF) pipeline because:**
- ✅ Achieves perfect accuracy (100%)
- ✅ Fastest training + inference
- ✅ Simple and interpretable
- ✅ Production-ready

---

## 📁 File Structure

```
Nemotron/
├── text_classification_pipelines.py          # Main source code
├── train_with_task_type.csv                  # Labeled dataset
├── TEXT_CLASSIFICATION_REPORT.md             # Full 9-section report
├── PROJECT_SUMMARY.md                        # Detailed summary
└── output/
    ├── pipeline_results.csv                  # CSV results table
    ├── 01_metrics_comparison.png             # Performance metrics chart
    ├── 02_speed_comparison.png               # Speed comparison chart
    ├── 03_performance_speed_tradeoff.png     # Accuracy vs Time scatter
    ├── 04_f1_vs_efficiency.png               # F1-Score vs Efficiency
    └── 05_normalized_metrics.png             # Normalized comparison
```

---

## 📖 Documentation Guide

### For Quick Overview
→ **PROJECT_SUMMARY.md** (2 min read)

### For Complete Analysis  
→ **TEXT_CLASSIFICATION_REPORT.md** (10-15 min read)

**Report Sections:**
1. Executive Summary
2. Data Preprocessing Details
3. Feature Extraction Methods
4. ML Algorithm Descriptions
5. Comparative Analysis
6. Visualization Results (with interpretations)
7. Detailed Findings
8. Recommendations
9. Conclusions & Limitations

### For Implementation Details
→ **text_classification_pipelines.py** (Well-commented code)

---

## 🔍 Visualizations

All visualizations clearly show the trade-offs between performance and computational efficiency:

1. **Metrics Comparison (01_metrics_comparison.png)**
   - Shows accuracy, F1-score, precision, recall
   - Three models achieve perfect scores

2. **Speed Comparison (02_speed_comparison.png)**
   - Training and inference times
   - Range: 0.002s (KNN train) to 19.903s (NN train)

3. **Performance vs Speed (03_performance_speed_tradeoff.png)**
   - Scatter plot: X-axis time, Y-axis accuracy
   - Naive Bayes and Random Forest dominate

4. **Efficiency Analysis (04_f1_vs_efficiency.png)**
   - F1-score vs inverse time (efficiency)
   - Identifies optimal performance/speed balance

5. **Normalized Metrics (05_normalized_metrics.png)**
   - All metrics on 0-1 scale
   - Clear three-tier separation

---

## 💻 Technical Specifications

### Environment
```
Python 3.8+
scikit-learn 0.24+
PyTorch 1.9+
pandas 1.2+
matplotlib 3.3+
seaborn 0.11+
numpy 1.19+
```

### Code Architecture
- **Object-oriented design** with clear separation of concerns
- **DataPreprocessor**: Data loading and preprocessing
- **FeatureExtractor**: Three text representation methods
- **TextClassifier**: Base class for all models
- **PipelineEvaluator**: Unified evaluation framework
- **PipelineVisualizer**: High-quality visualization generation

### Model Configurations
- **Naive Bayes:** Multinomial variant
- **SVM:** RBF kernel (gamma='scale')
- **Random Forest:** 100 estimators
- **KNN:** k=5, Euclidean distance
- **Neural Network:** Input→256→128→Output (dropout=0.3)

---

## 🚀 How to Use

### 1. Generate All Results
```bash
python Nemotron/text_classification_pipelines.py
```
**Output:** All CSV results and visualizations generated

### 2. View Results  
```bash
# View metrics in CSV format
cat Nemotron/output/pipeline_results.csv

# View visualizations in your image viewer
# Linux: eog Nemotron/output/*.png
# Windows: start Nemotron/output/
# Mac: open Nemotron/output/
```

### 3. Analyze Findings
1. Open `TEXT_CLASSIFICATION_REPORT.md` in any markdown viewer
2. Review visualizations in the output directory
3. Check `PROJECT_SUMMARY.md` for quick reference

---

## 📈 Performance Breakdown

### Feature Extraction Impact

**TF-IDF (5000 features):**
- Works excellently with Naive Bayes and Random Forest
- Both achieve 100% accuracy
- Fast processing of sparse vectors

**Count+SVD (100 features):**
- Good with SVM (99.74% accuracy)
- Poor with KNN (72.37% accuracy)  
- Dense representation offers trade-offs

**FastText-like (100 features):**
- Perfect for Neural Networks (100% accuracy)
- Character n-grams capture morphological patterns
- Best used with deep learning

### Algorithm Strengths

| Algorithm | Strength | Use Case |
|-----------|----------|----------|
| Naive Bayes | Speed + Accuracy | **Production default** |
| Random Forest | Interpretability | Feature analysis |
| Neural Network | Fastest inference | Real-time systems |
| SVM | Robustness | Conservative approach |
| KNN | Simplicity | Baseline only |

---

## 🔬 Methodology

### Data Preprocessing
1. Combine prompt + answer into single text
2. Convert to lowercase
3. Remove whitespace
4. Filter empty samples
5. Stratified 80/20 split

### Evaluation Approach
- **Metrics:** Accuracy, F1-Score, Precision, Recall
- **Speed:** Training time + Inference time
- **Cross-validation:** Single train-test split (should extend with k-fold)

### Limitations Noted
- Single train-test split (recommend k-fold CV)
- Perfect accuracy suggests possible overfitting
- No hyperparameter grid search
- Small test set (1,900 samples)

---

## 📝 Key Insights

### Why Naive Bayes Excels
1. **Conditional Independence:** Text classification has independent word features
2. **High-dimensional Space:** TF-IDF's 5000 features suit probabilistic classifiers
3. **Balanced Dataset:** No class imbalance complications
4. **Clear Patterns:** Nemotron tasks have distinctive linguistic patterns

### Why KNN Struggles
1. **Curse of Dimensionality:** 100D space makes distances less meaningful
2. **Dense Vectors:** Count+SVD creates ambiguous neighbor relationships
3. **Lack of Adaptation:** Fixed k=5 isn't optimal across feature space regions

### Why Neural Network Trains Slowly
1. **Backpropagation:** Iterative optimization (50 epochs)
2. **Feature Learning:** Learns representations from scratch
3. **50 training iterations:** High computational overhead
4. **GPU beneficial:** Could significantly improve with GPU acceleration

---

## 🎓 Learning Outcomes

This project demonstrates:
1. **Multiple ML paradigms:** Probabilistic, kernel, ensemble, and neural approaches
2. **Feature representation impact:** How text vectorization affects model performance
3. **Performance-efficiency trade-offs:** Speed vs accuracy considerations
4. **Evaluation methodology:** Comprehensive metric and visualization practices
5. **Production readiness:** Real-world model selection criteria

---

## 🤔 Frequently Asked Questions

**Q: Why does Naive Bayes achieve 100% accuracy?**
A: The Nemotron task types have clear, separable linguistic patterns. High-dimensional TF-IDF vectors (5000 features) provide sufficient information for perfect separation.

**Q: Should I trust the 100% accuracy?**
A: Be cautious with single train-test split evaluation. Recommend k-fold cross-validation for robust estimates. 100% may indicate some overfitting.

**Q: Why is KNN so slow at inference?**
A: KNN must compute distances to all 7,600 training samples for each prediction, resulting in 1.6 seconds for 1,900 test samples.

**Q: Can I use this for other text classification tasks?**
A: Yes! The code is modular and easily adaptable. Modify preprocessing for your domain and adjust hyperparameters as needed.

**Q: What about imbalanced classes?**
A: This dataset is well-balanced. For imbalanced data, consider class weights or resampling techniques.

---

## 📚 References & Resources

**Dataset:** [Nvidia Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

**Libraries Used:**
- scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- PyTorch: [https://pytorch.org](https://pytorch.org)
- pandas: [https://pandas.pydata.org](https://pandas.pydata.org)

---

## 📞 Contact & Support

For questions about this analysis:
1. Review the detailed sections in `TEXT_CLASSIFICATION_REPORT.md`
2. Check code comments in `text_classification_pipelines.py`
3. Examine visualization interpretations in `PROJECT_SUMMARY.md`

---

## ✅ Checklist of Deliverables

- ✅ **Source Code:** Comprehensive, well-commented implementation
- ✅ **Data Preprocessing:** Detailed description of all steps
- ✅ **Feature Extraction:** Three distinct methods explained
- ✅ **Five ML Pipelines:** Fully implemented and evaluated
- ✅ **Performance Metrics:** Accuracy, F1, Precision, Recall
- ✅ **Speed Analysis:** Training and inference times
- ✅ **Visualizations:** 5 comparison figures with interpretations
- ✅ **Written Report:** Comprehensive 9-section analysis
- ✅ **Recommendations:** Clear guidance on model selection
- ✅ **Results CSV:** Structured export of all metrics

---

**Status:** ✅ Project Complete

**Last Updated:** April 2026

**All deliverables available in the Nemotron directory**
