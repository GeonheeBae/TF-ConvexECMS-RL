import torch

if torch.cuda.is_available():
    print("✅ GPU를 사용할 수 있습니다.")
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}개")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")