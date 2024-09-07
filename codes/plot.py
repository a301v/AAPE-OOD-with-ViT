import pandas as pd
import matplotlib.pyplot as plt

# Data for top 50%
data_top = {
    "epsilon": ['0.1', '0.01', '0.001', '0.0001', 'baseline', '-0.0001', '-0.001', '-0.01'],
    "roc": [0.9371, 0.9379, 0.9846, 0.9856, 0.9857, 0.9857, 0.9859, 0.9348]
}

# Data for bottom 50%
data_bottom = {
    "epsilon": ['0.1', '0.01', '0.001', '0.0001', 'baseline', '-0.0001', '-0.001', '-0.01'],
    "roc": [0.8607, 0.8907, 0.9844, 0.9856, 0.9857, 0.9857, 0.986, 0.8544]
}

# Create DataFrames
df_top = pd.DataFrame(data_top)
df_bottom = pd.DataFrame(data_bottom)

# Plot both top 50% and bottom 50% on the same graph
plt.figure(figsize=(10, 6))
plt.plot(df_top["epsilon"], df_top["roc"], marker='o', linestyle='-', label="Top 50%")
plt.plot(df_bottom["epsilon"], df_bottom["roc"], marker='o', linestyle='-', label="Bottom 50%", color='red')
plt.xlabel("epsilon")
plt.ylabel("AUROC")
plt.title("AUROC : Perturb Top 50% and Bottom 50%")
plt.grid(True)
plt.legend()
plt.show()



# Data for top 50%
data_top = {
    "epsilon": ['0.1', '0.01', '0.001', '0.0001', 'baseline', '-0.0001', '-0.001', '-0.01'],
    "aupr": [0.9465, 0.945, 0.9865, 0.9874, 0.9875, 0.9875, 0.9877, 0.9463]
}

# Data for bottom 50%
data_bottom = {
    "epsilon": ['0.1', '0.01', '0.001', '0.0001', 'baseline', '-0.0001', '-0.001', '-0.01'],
    "aupr": [0.8782, 0.9008, 0.9846, 0.9874, 0.9875, 0.9876, 0.988, 0.8891]
}

# Create DataFrames
df_top = pd.DataFrame(data_top)
df_bottom = pd.DataFrame(data_bottom)

# Plot both top 50% and bottom 50% on the same graph
plt.figure(figsize=(10, 6))
plt.plot(df_top["epsilon"], df_top["aupr"], marker='o', linestyle='-', label="Top 50%")
plt.plot(df_bottom["epsilon"], df_bottom["aupr"], marker='o', linestyle='-', label="Bottom 50%", color='red')
plt.xlabel("epsilon")
plt.ylabel("AUPR")
plt.title("AUPR : Perturb Top 50% and Bottom 50%")
plt.grid(True)
plt.legend()
plt.show()
