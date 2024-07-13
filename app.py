# Import potrebnih biblioteka
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# 1. korak (Učitavanje i analiza podataka)
# Učitavanje dataseta
file_path = 'kardiologija_hospitalizacija 2024-06-28.xlsx'
df = pd.read_excel(file_path)

# Pregled prvih nekoliko redova dataseta
print("\n--- Prvih nekoliko redova dataseta ---\n")
print(df.head())

# Dobijanje osnovnih informacija o datasetu (tipovi podataka, ne-null vrijednosti)
print("\n--- Osnovne informacije o datasetu ---\n")
print(df.info())

# Dobijanje osnovnih statistika o numeričkim kolonama
print("\n--- Osnovne statistike numeričkih kolona ---\n")
print(df.describe())

# Provjera broja nedostajućih vrijednosti u svakoj koloni
print("\n--- Broj nedostajućih vrijednosti po kolonama ---\n")
print(df.isnull().sum())

# 2. korak (Čišćenje podataka)
# Popunjavanje nedostajućih vrijednosti u numeričkim kolonama sa srednjim vrijednostima
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer_num = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

# Popunjavanje nedostajućih vrijednosti u kategorijskim kolonama sa najčešćim vrijednostima (modom)
categorical_cols = df.select_dtypes(include='object').columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

# Provjera da li su sve nedostajuće vrijednosti popunjene
print("\n--- Broj nedostajućih vrijednosti nakon čišćenja ---\n")
print(df.isnull().sum())

# 3. korak (Priprema podataka za model)
# Pretvaranje svih vrijednosti u kategorijskim kolonama u stringove
df[categorical_cols] = df[categorical_cols].astype(str)

# Pretvaranje kategorijskih varijabli u numeričke
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Podjela podataka na trening i test setove (80% trening, 20% test)
X = df.drop('dcNyha', axis=1)  # X sadrži sve kolone osim ciljne promenljive 'dcNyha'
y = df['dcNyha']  # Ciljna promenljiva 'dcNyha'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Provjera dimenzija trening i test setova
print("\n--- Dimenzije trening i test setova ---\n")
print(f"Dimenzije trening seta x: {X_train.shape}, y: {y_train.shape}")
print(f"Dimenzije test seta x: {X_test.shape}, y: {y_test.shape}")

# 4. korak (Treniranje modela)
# Treniranje RandomForestClassifier na trening setu
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predikcija na test setu
y_pred = model.predict(X_test)

# 5. korak (Evaluacija modela)
# Evaluacija modela
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\n--- Evaluacija modela ---\n")
print(f"Tačnost: {accuracy}")
print(f"Preciznost: {precision}")
print(f"Odziv: {recall}")
print(f"F1 score: {f1}")

# Detaljan izvještaj o klasifikaciji
print("\n--- Izvještaj o klasifikaciji ---\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Razmatranje problema sa modelom
# Na primjer, loša klasifikacija NYHA II klase
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print("\n--- Detalji za NYHA II klasu ---\n")
print(f"Preciznost: {class_report['1']['precision']}")
print(f"Odziv: {class_report['1']['recall']}")
print(f"F1 score: {class_report['1']['f1-score']}")

# 6. korak (Poboljšanje modela)
# Korištenje drugih algoritama
# Treniranje XGBoost modela
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Predikcija na test setu
y_pred_xgb = xgb_model.predict(X_test)

# Evaluacija modela
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0)

print("\n--- Evaluacija XGB modela ---\n")
print(f"Tačnost: {accuracy_xgb}")
print(f"Preciznost: {precision_xgb}")
print(f"Odziv: {recall_xgb}")
print(f"F1 score: {f1_xgb}")

# Detaljan izvještaj o klasifikaciji
print("\n--- Izvještaj o klasifikaciji (XGBoost) ---\n")
print(classification_report(y_test, y_pred_xgb, zero_division=0))

# Primjena SMOTE sa smanjenim brojem susjeda
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Treniranje RandomForestClassifier na balansiranom setu
model_smote = RandomForestClassifier(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

# Predikcija na test setu
y_pred_smote = model_smote.predict(X_test)

# Evaluacija modela
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote, average='weighted', zero_division=0)
recall_smote = recall_score(y_test, y_pred_smote, average='weighted', zero_division=0)
f1_smote = f1_score(y_test, y_pred_smote, average='weighted', zero_division=0)

print("\n--- Evaluacija modela sa SMOTE ---\n")
print(f"Tačnost: {accuracy_smote}")
print(f"Preciznost: {precision_smote}")
print(f"Odziv: {recall_smote}")
print(f"F1 score: {f1_smote}")

# Detaljan izvještaj o klasifikaciji
print("\n--- Izvještaj o klasifikaciji (SMOTE) ---\n")
print(classification_report(y_test, y_pred_smote, zero_division=0))