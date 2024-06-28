import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Membaca data dari file CSV
df = pd.read_csv('data-cleaned.csv')

# Memisahkan fitur dan target
x = df[['V', 'H', 'S']]
y = df['M']

# Melakukan train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

# Mendefinisikan base models
base_models = [
    ('knn3', KNeighborsClassifier(n_neighbors=3)),
    ('knn5', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB())
]

# Mendefinisikan meta model
meta_model = LogisticRegression()

# Membuat stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Melatih stacking classifier
stacking_clf.fit(x_train, y_train)

# Menghitung akurasi model pada data uji
y_test_pred = stacking_clf.predict(x_test)
accuracy = accuracy_score(y_test, y_test_pred)


# Melakukan prediksi untuk data baru
y_new_pred = stacking_clf.predict(x_test)

if y_new_pred == 1:
    print(f"Prediction for new data: {y_new_pred[0]} (Tidak Ada Land Mines)")
elif y_new_pred == 2:
    print(f"Prediction for new data: {y_new_pred[0]} (Anti tank)")
elif y_new_pred == 3:
    print(f"Prediction for new data: {y_new_pred[0]} (Anti Presonnel)")
elif y_new_pred == 4:
    print(f"Prediction for new data: {y_new_pred[0]} (Bobby Trapped Anti Presonnel)")
elif y_new_pred == 5:
    print(f"Prediction for new data: {y_new_pred[0]} (M14 Anti-personnel)")

with open('model-pickle.pkl', 'wb') as file:
  pickle.dump(stacking_clf, file)
