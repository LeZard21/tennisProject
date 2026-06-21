import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("Adatok betöltése és előkészítése a KNN-hez...")
df = pd.read_csv("atp_meccsek_elo_pontokkal.csv")
df['tourney_date'] = pd.to_datetime(df['tourney_date'])
df = df.dropna(subset=['surface'])

# --- Érmefeldobás ---
np.random.seed(42)
coin_flip = np.random.rand(len(df)) > 0.5

df['target'] = coin_flip.astype(int)
df['diff_overall'] = np.where(coin_flip, df['elo_w_overall'], df['elo_l_overall']) - \
                     np.where(coin_flip, df['elo_l_overall'], df['elo_w_overall'])
df['diff_surface'] = np.where(coin_flip, df['elo_w_surface'], df['elo_l_surface']) - \
                     np.where(coin_flip, df['elo_l_surface'], df['elo_w_surface'])

# --- Időrendi Train/Test Split ---
split_date = pd.to_datetime('2024-01-01')
train_df = df[df['tourney_date'] < split_date]
test_df = df[df['tourney_date'] >= split_date]


features = ['diff_overall', 'diff_surface']

X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

k_values = range(1, 1002, 20) # 1, 11, 21, 31 ... 151
accuracies = []

print("\nKezdődik a modellek tesztelése (Ez eltarthat 1-2 percig)...")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    print(f"k={k:3} tesztelve. Pontosság: {acc * 100:.2f}%")

best_acc = max(accuracies)
best_k = k_values[accuracies.index(best_acc)]
print("\n" + "="*50)
print(f"A LEGJOBB BEÁLLÍTÁS: K = {best_k} (Pontosság: {best_acc * 100:.2f}%)")
print("="*50)

plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title("A KNN modell pontossága a K szomszédok számának függvényében")
plt.xlabel("K érték (Szomszédok száma)")
plt.ylabel("Teszt Pontosság (Decimális)")
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Legjobb K: {best_k}')
plt.grid(True)
plt.legend()
plt.show()