# Optimal Banking Model — README

Dieses Repository enthält eine minimalistische, präsentationsfertige Pipeline, aufgeteilt in mehrere Module:

```
prepare_data.py    # Datenvorbereitung & Preprocessing
imputate.py        # Imputation fehlender Werte
remove_features.py # Feature-Reduktion
predict.py         # Training, Prediction und Business-Summary
main.py            # End-to-End Pipeline Runner
präsi.ipynb        # Jupyter Notebook mit zusätzlichen Details zu was und warum und welche Reihenfolge gemacht wurde
```

---

## 1) Umgebung

Python 3.9+ empfohlen.

Installiere die Abhängigkeiten:

```bash
pip install -U scikit-learn pandas numpy joblib
```

---

## 2) Datenformat

Eingabe-CSV sollte numerische (oder bereits kodierte) Feature-Spalten und eine Zielspalte `target`, die Werte dieser Spalte werden erst später umgewandelt und die Werte 0 oder 1 gegeben

## 3) Wie ausführen?

Du hast zwei Optionen:

### Option A: End-to-End mit `main.py` (einfach & schnell)

Dies führt die gesamte Pipeline automatisch aus:

1. **prepare_data** → erstellt `kredit_clean.csv`
2. **imputate** → erstellt `kredit_imputed.csv`
3. **remove_features** → erstellt `kredit_final.csv`
4. **predict** → trainiert & evaluiert das Modell

```bash
python main.py
```

---

### Option B: Schritt-für-Schritt im Jupyter Notebook (mehr Einblick)

Falls du die Details, Zwischenschritte und Visualisierungen sehen willst:

```bash
pip install jupyterlab
jupyter lab
```

Dann das Notebook **`präsi.ipynb`** öffnen und die Zellen nacheinander ausführen.  
So bekommst du zusätzliche Erklärungen und Ergebnisse.

---

## 4) Projektstruktur

```
.
├── prepare_data.py
├── imputate.py
├── remove_features.py
├── predict.py
├── main.py
├── präsi.ipynb
└── README.md
```
