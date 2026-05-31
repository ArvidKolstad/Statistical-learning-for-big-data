import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def visualize_dataset():
    df_input= pd.read_csv("./data/X_TR.csv")
    df_labels = pd.read_csv("./data/y_TR.csv")
    unique_labels= np.unique(df_labels.to_numpy())
    df_total=pd.concat( [df_input,df_labels], axis=1)

    pca = PCA(n_components=820 ,copy=True,whiten=False, svd_solver = "full", )
    inputs = df_input.to_numpy()
    pca.fit(inputs)
    cum_variance = pca.explained_variance_ratio_.cumsum()
    fig,ax = plt.subplots()
    ax.plot(cum_variance)
    ax.set_xlabel("PC")
    ax.set_ylabel("Cumulative variance ratio")
    fig.savefig("./figures/vis_data/cumulative_variance_ratio_pca.pdf")

    class_dist =np.zeros(np.max(unique_labels))

    for idx,cls in enumerate(unique_labels):
        class_dist[idx]= np.sum(df_total["class"]==cls)/df_total.shape[0]


    fig, ax = plt.subplots()
    class_labels = "1","2","3","4","5","6","7"
    ax.pie(class_dist, labels = class_labels, autopct = "%1.1f%%")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig("./figures/vis_data/class_balance.pdf")
    print(f"Mean value: {np.mean(df_input.mean()):.3f}")
    print(f"STD value: {np.mean(df_input.std()):.3f}")
    print(f"Max value: {np.max(df_input.max()):.3f}")
    print(f"Min value: {np.min(df_input.min()):.3f}")
    print(f"Total samples: {df_input.shape[0]}")
    print(f"Input dim: {df_input.shape[1]}")
    print(f"Unique Classes: {unique_labels}")


def get_data_different_sample_sizes():

    df_input= pd.read_csv("./data/X_TR.csv")
    df_labels = pd.read_csv("./data/y_TR.csv")
    df_big = pd.concat([df_input, df_labels], axis=1)
    df_big = df_big.sample(frac=1, random_state=42).reset_index(drop=True)

    unique_classes = df_big['class'].unique()
    sizes = [500, 1000, len(df_big)]
    splits = {}

    for size in sizes:
        chunks = []
        for cls in unique_classes:
            cls_df = df_big[df_big['class'] == cls]          
            n = round(size * len(cls_df) / len(df_big))     
            chunks.append(cls_df.iloc[:n])                 
        
        df_split = (pd.concat(chunks)
                      .sample(frac=1, random_state=42)    
                      .reset_index(drop=True))
        
        splits[size] = df_split

    for size, subset in splits.items():
        print(size)
        print(subset['class'].value_counts(normalize=True).round(3))
        np.save(f"./data/{size}_labels",subset.to_numpy())
        np.save(f"./data/{size}_input", subset.drop(columns="class").to_numpy())


 





def main():
    visualize_dataset()
    get_data_different_sample_sizes()
    

if __name__=="__main__":
    main()
