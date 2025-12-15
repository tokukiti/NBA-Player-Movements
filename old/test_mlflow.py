import mlflow
import random
import time

# 実験の名前をセット
mlflow.set_experiment("GNN_Research_Test")

print("実験スタート（MLflow）")

with mlflow.start_run():
    # 1. パラメータ記録（学習率など）
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    
    # 2. 学習ループ
    for epoch in range(10):
        # 模擬的なLossとAccuracy
        loss = 1.0 / (epoch + 1) + random.random() * 0.1
        acc = 0.5 + (epoch * 0.05)
        
        # 3. メトリクス記録（グラフになる数値）
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", acc, step=epoch)
        
        print(f"Epoch {epoch} finished.")
        time.sleep(0.2)

print("完了！ブラウザをリロードして確認してください。")