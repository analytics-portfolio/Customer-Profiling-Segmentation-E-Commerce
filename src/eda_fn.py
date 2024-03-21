import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the features
def plot_feature_distribution(df, 
                              feature_list, 
                              row_num, row_col, 
                              title,
                              figsize=(15,12)):
    fig, axes = plt.subplots(nrows=row_num, ncols=row_col, figsize=figsize
    fig.suptitle(title, size = 18)
    for r in range(row_num):
        for c in range(row_col):
            list_id = r*row_col+c
            axes[r,c].set_title(feature_list[list_id])
            sns.histplot(df[feature_list[list_id]], kde=True, ax=axes[r,c])
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)


# Plot feature correlation heatmap matrix
def plot_feature_correlation_heatmap(df, title, figsize=(12,10)):
    plt.figure(figsize=figsize)
    plt.title(title, size = 18)
    corr_df = df.corr()
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr_df, mask=mask, annot=True, center=0, fmt='.2f', linewidths=2)
    plt.xticks(rotation=45, ha='right')
    plt.title('Correlation Matrix', fontsize=14)
    plt.show()
