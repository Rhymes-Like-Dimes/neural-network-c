import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("training_metrics.csv", comment="#")

fig, ax1 = plt.subplots(figsize=(8, 5))

#Plot loss
ax1.plot(df["Epoch"], df["Loss"], color="red", marker="o", linestyle="-", label="Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="red")
ax1.tick_params(axis='y', labelcolor="red")
ax1.set_ylim(0, 0.3) 

#Other
ax1.grid(True, linestyle=":")
ax1.set_xticks(range(0, 16))
ax1.set_xlim(0, 15)

#Plot accuracy
ax2 = ax1.twinx()
ax2.plot(df["Epoch"], df["Accuracy"], color="blue", marker="s", linestyle="-", label="Accuracy")
ax2.set_ylabel("Accuracy (%)", color="blue")
ax2.tick_params(axis='y', labelcolor="blue")
ax2.set_ylim(90, 100)

#Title
plt.title("Loss and Accuracy per Epoch")
fig.tight_layout()
plt.show()



