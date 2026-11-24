#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Citation-label prediction model: simple deployment script.
Predict citation-type labels for new observations.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def load_model_components(model_dir='saved_models'):
    """Load the model and all supporting components."""
    components = {}
    
    # Find the latest model file.
    model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError(f"在目录 '{model_dir}' 中未找到模型文件")
    
    latest_model = sorted(model_files)[-1]
    components['model'] = joblib.load(os.path.join(model_dir, latest_model))
    components['label_encoder'] = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    components['scaler'] = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
    
    with open(os.path.join(model_dir, 'ai_features_info.json'), 'r', encoding='utf-8') as f:
        components['ai_info'] = json.load(f)
    
    return components

def predict_citation_single(text, citation, embedding_vector, ai_labels=None, model_dir='saved_models'):
    """
    Predict the citation type of one observation.

    Parameters
    ----------
    text : str
        Paragraph text.
    citation : str
        Cleaned citation.
    embedding_vector : list
        A 1024-dimensional text vector.
    ai_labels : dict, optional
        AI labels, such as {'GPT': 'Background', 'Gemini': 'Technical basis'}.
    model_dir : str
        Directory containing model components.

    Returns
    -------
    dict
        Prediction results.
    """
    # Create one observation.
    data = {
        'Para_text': [text],
        'Ref_Clean_citation': [citation],
        'embedding_vector': [embedding_vector]
    }
    
    # Add AI labels.
    if ai_labels:
        for ai_name, ai_label in ai_labels.items():
            data[f'Citation_Type_{ai_name}'] = [ai_label]
    
    df = pd.DataFrame(data)
    
    # Load components.
    components = load_model_components(model_dir)
    
    # Extract features and predict.
    result = predict_new_data(df, components)
    
    return result.iloc[0].to_dict()

def predict_new_data(new_data, components=None, model_dir='saved_models'):
    """Predict new observations."""
    if components is None:
        components = load_model_components(model_dir)
    
    model = components['model']
    label_encoder = components['label_encoder']
    scaler = components['scaler']
    ai_info = components['ai_info']
    
    # Feature extraction
    X_new = np.array(new_data['embedding_vector'].tolist())
    
    # Numeric features
    para_length = new_data['Para_text'].apply(len).values.reshape(-1, 1)
    has_brackets = new_data['Ref_Clean_citation'].str.contains(r'[\[\]()]', na=False).fillna(False).astype(int).values.reshape(-1, 1)
    has_digits = new_data['Ref_Clean_citation'].str.contains(r'\d', na=False).fillna(False).astype(int).values.reshape(-1, 1)
    
    numeric_features = np.hstack([para_length, has_brackets, has_digits])
    numeric_features_scaled = scaler.transform(numeric_features)
    
    # AI features
    # Encode all AI labels once, then align exactly to the saved training schema.
    available = [label for label in ai_info['ai_labels'] if label in new_data.columns]
    if available:
        ai_frame = pd.get_dummies(new_data[available].fillna('Unknown').astype(str),
                                  prefix_sep='_')
    else:
        ai_frame = pd.DataFrame(index=new_data.index)
    ai_features_new = ai_frame.reindex(
        columns=ai_info['ai_features_columns'], fill_value=0
    ).to_numpy(dtype=float)
    
    # Combine features.
    X_new_enhanced = np.hstack([X_new, numeric_features_scaled, ai_features_new])
    
    # Predict.
    y_pred_proba = model.predict_proba(X_new_enhanced)
    y_pred = model.predict(X_new_enhanced)
    
    # Prepare results.
    results = new_data.copy()
    results['预测标签'] = label_encoder.inverse_transform(y_pred)
    results['预测概率'] = np.max(y_pred_proba, axis=1)
    
    for i, label in enumerate(label_encoder.classes_):
        results[f'预测概率_{label}'] = y_pred_proba[:, i]
    
    results['预测置信度'] = pd.cut(results['预测概率'], 
                                bins=[0, 0.5, 0.7, 0.85, 1.0],
                                labels=['低', '中', '高', '很高'])
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python citation_predictor.py <输入文件> [输出文件]")
        print("示例: python citation_predictor.py new_data.csv predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load data.
    if input_file.endswith('.csv'):
        new_data = pd.read_csv(input_file, encoding='utf-8')
    elif input_file.endswith('.pkl'):
        new_data = pd.read_pickle(input_file)
    else:
        print("错误: 只支持.csv和.pkl格式")
        sys.exit(1)
    
    # Predict.
    try:
        predictions = predict_new_data(new_data)
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_预测结果.csv"
        
        predictions.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"预测完成! 结果保存到: {output_file}")
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        sys.exit(1)
