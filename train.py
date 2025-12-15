import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import sys
import time

from model import STGAT

# train.py の該当部分のみ変更
DATA_PATH = 'dataset_v11_pipeline.pt' 
SAVE_MODEL_PATH = 'stgat_model_v11.pth'
SAVE_CSV_PATH = 'evaluation_results_v11.csv'

HIDDEN_DIM = 128
LR = 0.0005
EPOCHS = 30
ACCUMULATION_STEPS = 16 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed()
    
    if not torch.cuda.is_available():
        print("エラー: GPUが検出されません。")
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    print("Loading dataset...")
    try:
        full_dataset = torch.load(DATA_PATH, weights_only=False)
    except:
        full_dataset = torch.load(DATA_PATH)

    labels = [d['label'].item() for d in full_dataset]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight_val = neg_count / (pos_count + 1e-5)
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    print(f"Stats: Make={pos_count}, Miss={neg_count}, Weight={pos_weight_val:.2f}")

    train_data, test_data = train_test_split(full_dataset, test_size=0.2, stratify=labels, random_state=42)

    # データから自動的に特徴量次元数を取得 (ここでは9になるはず)
    num_node_features = full_dataset[0]['graphs'][0].num_node_features
    num_edge_features = full_dataset[0]['graphs'][0].num_edge_features
    
    print(f"Input Features: Node={num_node_features}, Edge={num_edge_features}")

    model = STGAT(node_features=num_node_features, 
                  edge_features=num_edge_features, 
                  hidden_channels=HIDDEN_DIM, 
                  out_channels=1).to(device)
                  
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    print(f"\nStarting training for {EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        random.shuffle(train_data)
        
        loop = tqdm(train_data, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i, play_data in enumerate(loop):
            graphs = play_data['graphs']
            label = play_data['label'].float().to(device)

            out = model(graphs) 
            loss = criterion(out.view(-1), label.view(-1))
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * ACCUMULATION_STEPS

        avg_loss = total_loss / len(train_data)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} (Total Time: {elapsed:.0f}s)")

    # --- 保存と評価 ---
    print("\nEvaluating & Saving...")
    model.eval()
    
    results_list = []

    with torch.no_grad():
        for play_data in tqdm(test_data, desc="Evaluating"):
            graphs = play_data['graphs']
            label = play_data['label'].item()
            
            game_id = play_data.get('game_id', 'unknown')
            event_id = play_data.get('event_id', -1)
            
            logits = model(graphs)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0
            
            results_list.append({
                'game_id': game_id,
                'event_id': event_id,
                'actual': label,
                'predicted': pred,
                'prob_make': round(prob, 4),
                'correct': (label == pred)
            })

    # CSV出力
    df_res = pd.DataFrame(results_list)
    df_res.to_csv(SAVE_CSV_PATH, index=False)
    print(f"✅ 評価結果を保存しました: {SAVE_CSV_PATH}")

    # モデル保存
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"✅ モデルを保存しました: {SAVE_MODEL_PATH}")

    # コンソール表示
    acc = accuracy_score(df_res['actual'], df_res['predicted'])
    prec = precision_score(df_res['actual'], df_res['predicted'], zero_division=0)
    rec = recall_score(df_res['actual'], df_res['predicted'], zero_division=0)

    print("\n=== Final Result ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

if __name__ == "__main__":
    main()