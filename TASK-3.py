import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog, StringVar, Label, OptionMenu, Button, messagebox
from pathlib import Path

def choose_file():
    global file_path
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                           filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    file_path_var.set(file_path)

def perform_clustering():
    df = pd.read_csv(file_path)
    selected_feature = feature_var.get()

    if pd.api.types.is_numeric_dtype(df[selected_feature]):
       
        features = df[[selected_feature]]

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        num_clusters = len(df[selected_feature].unique())

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(scaled_features)
    else:
       
        grouped_data = df.groupby(selected_feature).size().reset_index(name='count')
        num_clusters = len(grouped_data)

       
        df['cluster'] = df[selected_feature].map(dict(zip(grouped_data[selected_feature], range(num_clusters))))

    for cluster_id in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(cluster_data.describe())

    desktop_path = str(Path.home() / "Desktop")
    output_file_path = desktop_path + "/segmented-data.xlsx"
    df.to_excel(output_file_path, index=False)

    print(f"Segmented data saved to: {output_file_path}")
    messagebox.showinfo("Segmentation Information", f"Segmentation based on {selected_feature}. {num_clusters} clusters created.")

root = Tk()
root.title("Customer Segmentation")

file_path_var = StringVar()
feature_var = StringVar()

Label(root, text="Choose CSV File:").grid(row=0, column=0, padx=10, pady=10)
Button(root, text="Browse", command=choose_file).grid(row=0, column=1, padx=10, pady=10)
Label(root, textvariable=file_path_var).grid(row=1, column=0, columnspan=2, padx=10, pady=10)

Label(root, text="Select Feature for Segmentation:").grid(row=2, column=0, padx=10, pady=10)
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Internsavy\Customers.csv")
all_features_list = list(df.columns)
OptionMenu(root, feature_var, *all_features_list).grid(row=2, column=1, padx=10, pady=10)

Button(root, text="Perform Segmentation", command=perform_clustering).grid(row=3, column=0, columnspan=2, pady=20)

root.mainloop()
