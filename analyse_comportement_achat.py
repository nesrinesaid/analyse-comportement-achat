# --- Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

# --- Étape 1 : Chargement et nettoyage des données ---
print("Chargement des données...")
df = pd.read_excel("Online Retail.xlsx")

# Suppression des lignes sans CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Suppression des lignes avec quantités ou prix négatifs
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Création de la colonne 'TotalPrice'
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# --- Étape 2 : Analyse exploratoire ---
print("Analyse exploratoire en cours...")
print(df.describe())
print("Pays avec le plus de transactions :\n", df['Country'].value_counts().head())

# Ventes par pays (hors Royaume-Uni)
df_non_uk = df[df['Country'] != 'United Kingdom']
ventes_pays = df_non_uk.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)

# Visualisation
plt.figure(figsize=(12, 6))
ventes_pays.plot(kind='bar', title='Ventes totales par pays (hors UK)')
plt.ylabel('Ventes totales (£)')
plt.tight_layout()
plt.show()

# --- Étape 3 : Analyse RFM ---
print("Calcul des variables RFM...")
date_reference = df['InvoiceDate'].max() + timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (date_reference - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Récence',
    'InvoiceNo': 'Fréquence',
    'TotalPrice': 'Montant'
}).reset_index()

# Normalisation des variables RFM
scaler = StandardScaler()
rfm_norm = scaler.fit_transform(rfm[['Récence', 'Fréquence', 'Montant']])

# --- Étape 4 : Clustering K-means ---
print("Application du clustering K-means...")
inerties = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(rfm_norm)
    inerties.append(kmeans.inertia_)

# Méthode du coude
plt.figure()
plt.plot(range(2, 11), inerties, marker='o')
plt.title('Méthode du coude - K-means')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()

# Application de K-means avec k=4
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster_KMeans'] = kmeans.fit_predict(rfm_norm)

# --- Visualisation principale des clusters K-means ---
print("Création de la visualisation pour K-means...")

# Visualisation 3D des clusters
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Assigner une couleur et un marqueur à chaque cluster
colors = ['blue', 'red', 'green', 'purple']
markers = ['o', '^', 's', 'D']

# Limiter à 1000 points par cluster pour une meilleure performance
sample_size = 1000

for i in range(4):
    cluster_data = rfm[rfm['Cluster_KMeans'] == i]
    # Échantillonnage si plus de 1000 points
    if len(cluster_data) > sample_size:
        cluster_data = cluster_data.sample(sample_size, random_state=42)

    ax.scatter(
        cluster_data['Récence'],
        cluster_data['Fréquence'],
        cluster_data['Montant'],
        c=colors[i],
        marker=markers[i],
        label=f'Cluster {i}',
        alpha=0.7,
        s=50  # taille des points
    )

# Ajout des centroides
centroids = kmeans.cluster_centers_
# Dénormaliser les centroides pour correspondre aux valeurs RFM originales
centroids_denormalized = scaler.inverse_transform(centroids)

for i, centroid in enumerate(centroids_denormalized):
    ax.scatter(
        centroid[0], centroid[1], centroid[2],
        c='black',
        marker='X',
        s=200,
        label=f'Centroïde {i}' if i == 0 else None
    )
    # Ajouter une annotation à chaque centroïde
    ax.text(
        centroid[0], centroid[1], centroid[2],
        f'C{i}',
        fontsize=12,
        fontweight='bold'
    )

# Ajouter une grille pour une meilleure perception de la profondeur
ax.grid(True)

# Paramètres esthétiques
ax.set_xlabel('Récence (jours)', fontsize=12, labelpad=10)
ax.set_ylabel('Fréquence (transactions)', fontsize=12, labelpad=10)
ax.set_zlabel('Montant (£)', fontsize=12, labelpad=10)
ax.set_title('Segmentation des clients par K-means (RFM)', fontsize=16, pad=20)

# Amélioration de la légende
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, title='Segments')

# Profil des clusters (pour l'affichage)
cluster_profiles = rfm.groupby('Cluster_KMeans')[['Récence', 'Fréquence', 'Montant']].mean().round(2)
print("Profil des clusters K-means:")
print(cluster_profiles)

# Rotation automatique pour une meilleure visualisation
for angle in range(0, 360, 30):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(0.1)

plt.tight_layout()
plt.show()

# --- Étape 5 : Clustering hiérarchique (CAH) ---
print("Application du clustering hiérarchique...")
liaison = linkage(rfm_norm, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(liaison)
plt.title("Dendrogramme - Clustering hiérarchique")
plt.show()

# Attribution des clusters (ex. 4)
rfm['Cluster_CAH'] = fcluster(liaison, t=4, criterion='maxclust')

# --- Étape 6 : Règles d'association avec Apriori ---
print("Extraction des règles d'association avec Apriori...")

# Limiter le jeu de données à 2000 transactions pour éviter la surcharge mémoire
subset_df = df[df['Quantity'] > 1].head(2000)

# Sélectionner les 50 produits les plus fréquents pour réduire la dimension
top_products = df['Description'].value_counts().head(50).index
subset_df = subset_df[subset_df['Description'].isin(top_products)]

# Création du panier binaire
basket = subset_df.pivot_table(
    index='InvoiceNo',
    columns='Description',
    values='Quantity',
    aggfunc='sum'
).fillna(0)

# Conversion explicite en booléen (True/False)
basket = basket.astype(bool)

# Utilisation d'Apriori au lieu de FP-growth
print("Calcul des itemsets fréquents avec Apriori...")
itemsets_frequents = apriori(basket, min_support=0.01, use_colnames=True)

# Extraction des règles
regles = association_rules(itemsets_frequents, metric="lift", min_threshold=1.0)

# Affichage des principales règles
regles = regles.sort_values(by='lift', ascending=False)
print("Top 10 des règles d'association :")
print(regles[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# --- Enregistrement des résultats ---
rfm.to_csv("Segments_RFM.csv", index=False)
regles.to_csv("Regles_Association.csv", index=False)

print("✅ Analyse terminée avec succès.")