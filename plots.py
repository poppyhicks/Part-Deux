import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


file_path = "./metrics/training_metrics_20250506_220144.csv"
epochs = 50
notes = "Bottleneck. Tolerance matching; unknown classes converted to closest match. "

baseline_path = "./metrics/training_metrics_20250428_112711.csv"
root, ext = os.path.splitext(file_path)
plot_name = root[-6:]
baseline = pd.read_csv(baseline_path)
comp1 = pd.read_csv(file_path)

plt.figure(figsize=(10,8))
plt.plot(baseline["Epoch"].iloc[2:50].astype(float), baseline["Val Loss"].iloc[2:50].astype(float), label = "Baseline")
plt.plot(comp1["Epoch"].iloc[2:epochs].astype(float), comp1["Val Loss"].iloc[2:epochs].astype(float), label = "Comparison")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title(notes)
plt.legend()
plt.savefig(f"plots/val_loss_{plot_name}.png")



plt.figure(figsize=(10,8))
plt.plot(baseline["Epoch"].iloc[0:50].astype(float), baseline["Mean IoU"].iloc[0:50].astype(float), label = "Baseline")
plt.plot(comp1["Epoch"].iloc[0:epochs].astype(float), comp1["Mean IoU"].iloc[0:epochs].astype(float), label = "Comparison")
plt.xlabel("Epoch")
plt.ylabel("Mean IoU")
plt.title(notes)
plt.legend()
plt.savefig(f"plots/mean_iou_{plot_name}.png")