import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from STaRNet.model import STaRNet
from util import preprocess
from config.params import PARAMS

# ========== [1] 파라미터 파싱 ==========
parser = argparse.ArgumentParser(description='Train EEG model using true HOCV')
parser.add_argument('--model', type=str, default='STaRNet')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--save_root', type=str, default='./results')
args = parser.parse_args()

# ========== [2] 환경 설정 ==========
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# 결과 저장 폴더: 예) ./results/HOCV/STaRNet/20250606_101523/
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(args.save_root, args.model, timestamp)
os.makedirs(save_dir, exist_ok=True)

# ========== [3] Hold-Out CV 방식으로 Subject별 학습/테스트 ==========
def train_subject_holdout(subject_id: int):
    """
    논문 §5.1: BCI Competition IV 2a 데이터에 대해 HOCV 사용:
      • Session1(288 trials) → Train
      • Session2(288 trials) → Test
      • 총 epochs=500, lr=0.001, batch_size=16
      • k-Fold나 EarlyStopping 없이, 학습 후 마지막 모델로 테스트
    """
    print(f"\n[INFO] Subject {subject_id} hold-out training start")

    data_dir = os.path.join('./dataset/BCICIV_2a')

    # -------------------------------------------------
    # 1) 데이터 로딩: 세션1 → train_dataset, 세션2 → test_dataset
    # -------------------------------------------------
    train_dataset, _, _, test_dataset = \
        preprocess.get_HO_EEG_data(subject_id, data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # -------------------------------------------------
    # 2) 모델/손실함수/최적화기 초기화
    # -------------------------------------------------
    model = STaRNet(**PARAMS).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # -------------------------------------------------
    # 3) 학습: 논문에 맞춰 Epoch 500 회 학습 (EarlyStopping 없음)
    # -------------------------------------------------
    for epoch in tqdm(range(args.num_epochs), desc=f"[Sub {subject_id}]"):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)    # (batch, 1, 22, 1000)
            targets = targets.to(device)  # (batch,)

            optimizer.zero_grad()
            outputs = model(inputs)  # (batch, num_classes)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
 
        # (선택) Epoch마다 학습 Loss 출력
        avg_loss = running_loss / len(train_dataset)
        tqdm.write(f"Epoch [{epoch+1:03d}/{args.num_epochs:03d}]  Train Loss: {avg_loss:.4f}")

    # -------------------------------------------------
    # 4) 학습 종료 후, 최종 epoch(500) 가중치 저장
    # -------------------------------------------------
    final_state = model.state_dict()
    model_path = os.path.join(
        save_dir,
        f'{args.model}_sub{subject_id}_epoch{args.num_epochs}.pth'
    )
    torch.save(final_state, model_path)
    print(f"[✓] Subject {subject_id} model saved → {model_path}")

    # -------------------------------------------------
    # 5) 학습된 모델 평가: 세션2 (Test set) → 한 번만 계산
    # -------------------------------------------------
    model.load_state_dict(final_state)
    model.eval()

    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    accuracy = correct_predictions / total_samples
    print(f"[RESULT] Subject {subject_id} Test Accuracy: {accuracy*100:.2f}%")

    # -------------------------------------------------
    # 6) Test 결과 파일 저장
    # -------------------------------------------------
    result_file = os.path.join(save_dir, f'test_result_sub{subject_id}.txt')
    with open(result_file, 'w') as f:
        f.write(f"Subject {subject_id} Test Accuracy: {accuracy:.4f}\n")
    print(f"[✓] Subject {subject_id} test results saved → {result_file}\n")

    return accuracy

# ========== [4] 메인: Subject 1~9 순차적으로 학습/테스트 ==========
if __name__ == '__main__':
    all_accuracies = []
    for subject_id in range(1, 10):  # Subjects 1~9
        accuracy = train_subject_holdout(subject_id)
        all_accuracies.append(accuracy)

    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"\n[FINAL] All Subjects Test Accuracies: {all_accuracies}")
    print(f"[FINAL] Average Accuracy: {avg_accuracy*100:.2f}%")

    summary_path = os.path.join(save_dir, 'summary_all_subjects.txt')
    with open(summary_path, 'w') as f:
        f.write(f"All Accuracies: {all_accuracies}\nAverage Accuracy: {avg_accuracy:.4f}\n")
    print(f"[✓] Summary saved → {summary_path}")

