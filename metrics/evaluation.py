from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred, labels=['negative', 'neutral', 'positive']):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': dict(zip(labels, precision)),
            'recall': dict(zip(labels, recall)),
            'f1': dict(zip(labels, f1))
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive']):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    @staticmethod
    def plot_feature_importance(model, vectorizer, top_n=20):
        coef = model.model.coef_
        feature_names = vectorizer.get_feature_names()
        
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(['negative', 'neutral', 'positive']):
            top_indices = np.argsort(coef[i])[-top_n:]
            top_features = [feature_names[idx] for idx in top_indices]
            top_scores = coef[i][top_indices]
            
            plt.subplot(3, 1, i+1)
            plt.barh(top_features, top_scores)
            plt.title(f'Top {top_n} features for {label}')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()