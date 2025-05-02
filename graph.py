import matplotlib.pyplot as plt

# 모델 이름
models = [f"EfficientNet-B{i}" for i in range(8)]

# 입력값 (단위 변환 없이 사용)
params = [4263748, 6769384, 7982794, 11003632, 17907216, 28750584, 41196704, 64299160]
flops  =  [34626304, 51029264, 58688128, 85144064, 131968312, 205805920, 292279816, 447795808]
accs   =  [0.418, 0.407, 0.413, 0.420, 0.394, 0.438, 0.423, 0.375]

# ─────────────── Params vs Accuracy ───────────────
plt.figure(figsize=(8, 6))
plt.plot(params, accs, marker='o', linestyle='-', color='blue')
for i, model in enumerate(models):
    plt.text(params[i], accs[i] + 0.002, model, fontsize=8, ha='center')
plt.xlabel("Number of Parameters")
plt.ylabel("Validation Accuracy")
plt.title("Model Size (Params) vs Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────── FLOPs vs Accuracy ───────────────
plt.figure(figsize=(8, 6))
plt.plot(flops, accs, marker='o', linestyle='-', color='green')
for i, model in enumerate(models):
    plt.text(flops[i], accs[i] + 0.002, model, fontsize=8, ha='center')
plt.xlabel("FLOPs")
plt.ylabel("Validation Accuracy")
plt.title("Model FLOPs vs Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
