"""
Text Classification Pipeline Comparison
This module implements and compares multiple ML pipelines for classifying Nemotron task types.

Pipelines Implemented:
1. TF-IDF + Naive Bayes
2. Word2Vec + Support Vector Machine (SVM)
3. TF-IDF + Random Forest
4. Neural Network with Embeddings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== DATA LOADING AND PREPROCESSING ====================

class DataPreprocessor:
    """Handles data loading and preprocessing for text classification."""
    
    def __init__(self, data_path: str):
        """Initialize preprocessor and load data."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.y_train_encoded = None
        self.y_test_encoded = None
        
    def load_and_preprocess(self, test_size: float = 0.2):
        """Load data and perform preprocessing."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Task types distribution:\n{self.df['task_type'].value_counts()}")
        
        # Basic text preprocessing
        self.df['text'] = self.df['prompt'].fillna('') + ' ' + self.df['answer'].fillna('')
        self.df['text'] = self.df['text'].str.lower().str.strip()
        
        # Remove rows with empty text
        self.df = self.df[self.df['text'].str.len() > 0].reset_index(drop=True)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.df['task_type'])
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['text'].values,
            self.df['task_type'].values,
            test_size=test_size,
            random_state=42,
            stratify=self.df['task_type'].values
        )
        
        # Encode labels for train/test split
        self.y_train_encoded = self.label_encoder.transform(self.y_train)
        self.y_test_encoded = self.label_encoder.transform(self.y_test)
        
        print(f"\nTrain set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

# ==================== FEATURE EXTRACTION ====================

class FeatureExtractor:
    """Extracts features using various text representation methods."""
    
    @staticmethod
    def tfidf_features(X_train: np.ndarray, X_test: np.ndarray, 
                       max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TF-IDF features."""
        print("  Extracting TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), 
                                     min_df=2, max_df=0.8)
        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()
        return X_train_tfidf, X_test_tfidf, vectorizer
    
    @staticmethod
    def word2vec_features(X_train: np.ndarray, X_test: np.ndarray,
                          vector_size: int = 100, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Word2Vec-like features using CountVectorizer and SVD."""
        print("  Extracting Word2Vec-like features (via CountVectorizer + SVD)...")
        from sklearn.decomposition import TruncatedSVD
        
        # Create count vectors
        count_vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.8)
        X_train_counts = count_vectorizer.fit_transform(X_train)
        
        # Apply SVD to reduce dimensions and create dense embeddings
        svd = TruncatedSVD(n_components=min(vector_size, X_train_counts.shape[1]-1))
        X_train_w2v = svd.fit_transform(X_train_counts)
        
        X_test_counts = count_vectorizer.transform(X_test)
        X_test_w2v = svd.transform(X_test_counts)
        
        return X_train_w2v, X_test_w2v, count_vectorizer
    
    @staticmethod
    def fasttext_like_features(X_train: np.ndarray, X_test: np.ndarray,
                               embedding_dim: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple embedding-like features using character n-grams."""
        print("  Extracting FastText-like features...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Use character n-grams
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), 
                                     max_features=embedding_dim, 
                                     min_df=2, max_df=0.8)
        X_train_ft = vectorizer.fit_transform(X_train).toarray()
        X_test_ft = vectorizer.transform(X_test).toarray()
        
        return X_train_ft, X_test_ft, vectorizer

# ==================== ML MODELS ====================

class TextClassifier:
    """Base class for text classifiers."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.train_time = 0
        self.inference_time = 0
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train model and return training time."""
        raise NotImplementedError
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        raise NotImplementedError

class NaiveBayesClassifier(TextClassifier):
    """Naive Bayes classifier for TF-IDF features."""
    
    def __init__(self):
        super().__init__("Naive Bayes (TF-IDF)")
        self.model = MultinomialNB()
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train Naive Bayes."""
        start = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start
        return self.train_time
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        start = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start
        return predictions
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X_test)

class SVMClassifier(TextClassifier):
    """Support Vector Machine classifier."""
    
    def __init__(self, kernel='rbf'):
        super().__init__(f"SVM ({kernel}) (Count+SVD)")
        self.model = SVC(kernel=kernel, gamma='scale', probability=True)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train SVM."""
        start = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start
        return self.train_time
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        start = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start
        return predictions
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X_test)

class RandomForestTextClassifier(TextClassifier):
    """Random Forest classifier for text."""
    
    def __init__(self, n_estimators: int = 100):
        super().__init__(f"Random Forest (TF-IDF)")
        self.model = RandomForestClassifier(n_estimators=n_estimators, 
                                           random_state=42, n_jobs=-1)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train Random Forest."""
        start = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start
        return self.train_time
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        start = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start
        return predictions
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X_test)

class KNNClassifier(TextClassifier):
    """K-Nearest Neighbors classifier."""
    
    def __init__(self, n_neighbors: int = 5):
        super().__init__(f"KNN (k={n_neighbors}, Count+SVD)")
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train KNN."""
        start = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start
        return self.train_time
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        start = time.time()
        predictions = self.model.predict(X_test)
        self.inference_time = time.time() - start
        return predictions
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities - KNN doesn't have predict_proba by default."""
        # Return normalized distances as pseudo-probabilities
        distances, indices = self.model.kneighbors(X_test)
        # Invert distances to get similarities
        similarities = 1 / (1 + distances)
        # Normalize
        proba = similarities / similarities.sum(axis=1, keepdims=True)
        return proba

class NeuralNetworkClassifier(TextClassifier):
    """Neural Network classifier using PyTorch."""
    
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__("Neural Network (FastText Features)")
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        # Build network
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 50, batch_size: int = 32) -> float:
        """Train neural network."""
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        start = time.time()
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
        self.train_time = time.time() - start
        return self.train_time
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        self.inference_time = time.time() - start
        
        return predictions
        
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
            
        return proba

# ==================== EVALUATION ====================

class PipelineEvaluator:
    """Evaluates classification pipelines."""
    
    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder
        self.results = []
        
    def evaluate(self, model: TextClassifier, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 feature_type: str) -> Dict:
        """Evaluate a model pipeline."""
        print(f"\nEvaluating: {model.name}")
        print(f"  Training samples: {len(X_train)}, Feature dimension: {X_train.shape[1]}")
        
        # Train model
        model.train(X_train, y_train)
        print(f"  Training time: {model.train_time:.4f}s")
        
        # Make predictions
        y_pred = model.predict(X_test)
        print(f"  Inference time: {model.inference_time:.4f}s")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Store results
        result = {
            'Model': model.name,
            'Feature Type': feature_type,
            'Accuracy': accuracy,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Train Time (s)': model.train_time,
            'Inference Time (s)': model.inference_time,
            'Predictions': y_pred,
            'True Labels': y_test
        }
        
        self.results.append(result)
        return result
        
    def get_results_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        results_copy = [r.copy() for r in self.results]
        for r in results_copy:
            r.pop('Predictions', None)
            r.pop('True Labels', None)
        return pd.DataFrame(results_copy)

# ==================== VISUALIZATION ====================

class PipelineVisualizer:
    """Creates visualizations for pipeline comparison."""
    
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df
        self.output_dir = "Nemotron/output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_metrics_comparison(self):
        """Plot comparison of evaluation metrics."""
        metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            sorted_data = self.results_df.sort_values(metric, ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_data)))
            
            bars = ax.barh(sorted_data['Model'], sorted_data[metric], color=colors)
            ax.set_xlabel(metric, fontweight='bold')
            ax.set_title(f'{metric} by Model')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, sorted_data[metric])):
                ax.text(val + 0.02, i, f'{val:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/01_metrics_comparison.png")
        plt.close()
        
    def plot_speed_comparison(self):
        """Plot comparison of training and inference speed."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Computational Speed Comparison', fontsize=16, fontweight='bold')
        
        # Training time
        ax = axes[0]
        sorted_data = self.results_df.sort_values('Train Time (s)', ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sorted_data)))
        bars = ax.barh(sorted_data['Model'], sorted_data['Train Time (s)'], color=colors)
        ax.set_xlabel('Training Time (seconds)', fontweight='bold')
        ax.set_title('Training Time by Model')
        
        for i, (bar, val) in enumerate(zip(bars, sorted_data['Train Time (s)'])):
            ax.text(val + 0.001, i, f'{val:.4f}s', va='center')
        
        # Inference time
        ax = axes[1]
        sorted_data = self.results_df.sort_values('Inference Time (s)', ascending=False)
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(sorted_data)))
        bars = ax.barh(sorted_data['Model'], sorted_data['Inference Time (s)'], color=colors)
        ax.set_xlabel('Inference Time (seconds)', fontweight='bold')
        ax.set_title('Inference Time by Model')
        
        for i, (bar, val) in enumerate(zip(bars, sorted_data['Inference Time (s)'])):
            ax.text(val + 0.0001, i, f'{val:.4f}s', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_speed_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/02_speed_comparison.png")
        plt.close()
        
    def plot_performance_speed_tradeoff(self):
        """Plot accuracy vs speed tradeoff."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use total time for analysis
        self.results_df['Total Time'] = self.results_df['Train Time (s)'] + self.results_df['Inference Time (s)']
        
        # Create scatter plot
        scatter = ax.scatter(self.results_df['Total Time'], 
                            self.results_df['Accuracy'],
                            s=300, alpha=0.6, c=range(len(self.results_df)), 
                            cmap='viridis', edgecolors='black', linewidth=2)
        
        # Add model labels
        for idx, row in self.results_df.iterrows():
            ax.annotate(row['Model'], 
                       (row['Total Time'], row['Accuracy']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='yellow', alpha=0.3))
        
        ax.set_xlabel('Total Computational Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax.set_title('Performance vs Speed Trade-off Analysis', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_performance_speed_tradeoff.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/03_performance_speed_tradeoff.png")
        plt.close()
        
    def plot_f1_vs_efficiency(self):
        """Plot F1-Score vs Efficiency (inverse of total time)."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        self.results_df['Total Time'] = self.results_df['Train Time (s)'] + self.results_df['Inference Time (s)']
        self.results_df['Efficiency'] = 1 / (self.results_df['Total Time'] + 0.001)  # Add small value to avoid division by zero
        
        scatter = ax.scatter(self.results_df['F1-Score'], 
                            self.results_df['Efficiency'],
                            s=300, alpha=0.6, c=range(len(self.results_df)), 
                            cmap='plasma', edgecolors='black', linewidth=2)
        
        for idx, row in self.results_df.iterrows():
            ax.annotate(row['Model'], 
                       (row['F1-Score'], row['Efficiency']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='lightblue', alpha=0.3))
        
        ax.set_xlabel('F1-Score (Accuracy)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Efficiency (1/Time)', fontweight='bold', fontsize=12)
        ax.set_title('F1-Score vs Computational Efficiency', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_f1_vs_efficiency.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/04_f1_vs_efficiency.png")
        plt.close()
        
    def plot_all_metrics_normalized(self):
        """Plot normalized comparison of all metrics."""
        # Normalize metrics to 0-1 scale
        metrics_to_plot = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        normalized_df = self.results_df[['Model'] + metrics_to_plot].copy()
        
        for col in metrics_to_plot:
            normalized_df[col] = normalized_df[col]  # Already in 0-1 range
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(normalized_df))
        width = 0.2
        
        for idx, metric in enumerate(metrics_to_plot):
            offset = (idx - 1.5) * width
            ax.bar(x + offset, normalized_df[metric], width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Normalized Performance Metrics Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(normalized_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/05_normalized_metrics.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/05_normalized_metrics.png")
        plt.close()

# ==================== MAIN PIPELINE ====================

def main():
    """Main execution function."""
    print("=" * 80)
    print("TEXT CLASSIFICATION PIPELINE COMPARISON")
    print("Dataset: Nvidia Nemotron Model Reasoning Challenge")
    print("=" * 80)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor("Nemotron/train_with_task_type.csv")
    X_train, X_test, y_train, y_test = preprocessor.load_and_preprocess(test_size=0.2)
    
    # Feature Extraction
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION")
    print("=" * 80)
    
    # TF-IDF Features
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = FeatureExtractor.tfidf_features(X_train, X_test)
    print(f"  TF-IDF feature shape: {X_train_tfidf.shape}")
    
    # Word2Vec Features
    X_train_w2v, X_test_w2v, w2v_model = FeatureExtractor.word2vec_features(X_train, X_test)
    print(f"  Word2Vec-like (Count+SVD) feature shape: {X_train_w2v.shape}")
    
    # FastText-like Features
    X_train_ft, X_test_ft, ft_vectorizer = FeatureExtractor.fasttext_like_features(X_train, X_test)
    print(f"  FastText-like feature shape: {X_train_ft.shape}")
    
    # Normalize numerical features for distance-based models
    scaler_w2v = StandardScaler()
    X_train_w2v_scaled = scaler_w2v.fit_transform(X_train_w2v)
    X_test_w2v_scaled = scaler_w2v.transform(X_test_w2v)
    
    scaler_ft = StandardScaler()
    X_train_ft_scaled = scaler_ft.fit_transform(X_train_ft)
    X_test_ft_scaled = scaler_ft.transform(X_test_ft)
    
    # Encode target labels
    num_classes = len(preprocessor.label_encoder.classes_)
    
    # Initialize evaluator and models
    print("\n" + "=" * 80)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    evaluator = PipelineEvaluator(preprocessor.label_encoder)
    
    # Pipeline 1: TF-IDF + Naive Bayes
    print("\n[Pipeline 1] TF-IDF + Naive Bayes")
    nb_model = NaiveBayesClassifier()
    evaluator.evaluate(nb_model, X_train_tfidf, preprocessor.y_train_encoded,
                      X_test_tfidf, preprocessor.y_test_encoded, "TF-IDF")
    
    # Pipeline 2: Word2Vec + SVM
    print("\n[Pipeline 2] Count+SVD + SVM (RBF)")
    svm_model = SVMClassifier(kernel='rbf')
    evaluator.evaluate(svm_model, X_train_w2v_scaled, preprocessor.y_train_encoded,
                      X_test_w2v_scaled, preprocessor.y_test_encoded, "Count+SVD")
    
    # Pipeline 3: TF-IDF + Random Forest
    print("\n[Pipeline 3] TF-IDF + Random Forest")
    rf_model = RandomForestTextClassifier(n_estimators=100)
    evaluator.evaluate(rf_model, X_train_tfidf, preprocessor.y_train_encoded,
                      X_test_tfidf, preprocessor.y_test_encoded, "TF-IDF")
    
    # Pipeline 4: Word2Vec + KNN
    print("\n[Pipeline 4] Count+SVD + KNN (k=5)")
    knn_model = KNNClassifier(n_neighbors=5)
    evaluator.evaluate(knn_model, X_train_w2v_scaled, preprocessor.y_train_encoded,
                      X_test_w2v_scaled, preprocessor.y_test_encoded, "Count+SVD")
    
    # Pipeline 5: Neural Network with FastText-like Features
    print("\n[Pipeline 5] Neural Network (FastText Features)")
    nn_model = NeuralNetworkClassifier(input_size=X_train_ft_scaled.shape[1],
                                       num_classes=num_classes,
                                       hidden_dims=[256, 128],
                                       dropout=0.3)
    evaluator.evaluate(nn_model, X_train_ft_scaled, preprocessor.y_train_encoded,
                      X_test_ft_scaled, preprocessor.y_test_encoded, "FastText-like")
    
    # Get results and save
    results_df = evaluator.get_results_dataframe()
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Create output directory if it doesn't exist
    os.makedirs("Nemotron/output", exist_ok=True)
    
    # Save results to CSV
    results_df.to_csv("Nemotron/output/pipeline_results.csv", index=False)
    print(f"\nResults saved to: Nemotron/output/pipeline_results.csv")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    visualizer = PipelineVisualizer(results_df)
    visualizer.plot_metrics_comparison()
    visualizer.plot_speed_comparison()
    visualizer.plot_performance_speed_tradeoff()
    visualizer.plot_f1_vs_efficiency()
    visualizer.plot_all_metrics_normalized()
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPARISON COMPLETE")
    print("=" * 80)
    
    return results_df, evaluator

if __name__ == "__main__":
    results_df, evaluator = main()
