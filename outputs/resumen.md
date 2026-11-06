# Resumen de Tablas del Paper Científico

## Tabla 2 — Modelos sensoriales (n=8, LOOCV, Ridge alpha=1.0)

| Atributo sensorial | MAE  | RMSE | R²    | Baseline MAE (media) | Baseline RMSE (media) |
| ------------------ | ---- | ---- | ----- | -------------------- | --------------------- |
| Brillo             | 0.31 | 0.38 | 0.23  | 0.42                 | 0.50                  |
| Claridad           | 0.55 | 0.78 | -2.21 | 0.46                 | 0.50                  |
| Dulzura            | 0.35 | 0.43 | -0.55 | 0.30                 | 0.40                  |
| Acidez             | 0.76 | 1.10 | -6.56 | 0.41                 | 0.46                  |
| Adhesividad        | 0.72 | 0.83 | 0.21  | 0.93                 | 1.06                  |
| Elasticidad        | 1.40 | 2.00 | -4.08 | 0.79                 | 1.01                  |
| Arenosidad         | 0.35 | 0.52 | -1.90 | 0.32                 | 0.35                  |
| Dureza             | 0.51 | 0.69 | -2.85 | 0.29                 | 0.40                  |
| Gomosidad          | 0.71 | 0.86 | -2.12 | 0.38                 | 0.56                  |
| Masticabilidad     | 0.58 | 0.68 | -1.52 | 0.39                 | 0.49                  |
| Promedio           | 0.62 | 0.83 | -2.13 | 0.47                 | 0.57                  |

## Tabla 3a — Modelo de viabilidad: Métricas (n=25, LOOCV, LogisticRegression)

| Métrica           | Valor | Baseline mayoritaria |
| ----------------- | ----- | -------------------- |
| accuracy          | 0.56  | 0.68                 |
| balanced_accuracy | 0.54  | 0.50                 |
| precision         | 0.36  | 0.00                 |
| recall            | 0.50  | 0.00                 |
| f1                | 0.42  | 0.00                 |
| roc_auc           | 0.46  | 0.50                 |

## Tabla 3b — Modelo de viabilidad: Matriz de confusión

|                 | Predicho: no viable | Predicho: viable |
| --------------- | ------------------- | ---------------- |
| Real: no viable | 10                  | 7                |
| Real: viable    | 4                   | 4                |

## Tabla 5 — Validación funcional con datos sintéticos

| Tamaño muestral (n) | Modelo sensorial seleccionado                         | Modelo de viabilidad seleccionado                      | Modelo de aceptabilidad seleccionado                   | Resultado               |
| ------------------- | ----------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ | ----------------------- |
| 10                  | Ridge(alpha=1.0)                                      | LogisticRegression(C=0.5, class_weight="balanced")     | RandomForestClassifier(n_estimators=30, max_depth=4)   | OK, sin errores         |
| 30                  | RandomForestRegressor(n_estimators=30, max_depth=3)   | RandomForestClassifier(n_estimators=50, max_depth=4)   | RandomForestClassifier(n_estimators=50, max_depth=4)   | OK, transición correcta |
| 60                  | RandomForestRegressor(n_estimators=100, max_depth=10) | RandomForestClassifier(n_estimators=50, max_depth=4)   | RandomForestClassifier(n_estimators=50, max_depth=4)   | OK, transición correcta |
| 100                 | RandomForestRegressor(n_estimators=100, max_depth=10) | RandomForestClassifier(n_estimators=100, max_depth=10) | RandomForestClassifier(n_estimators=100, max_depth=10) | OK, transición correcta |
| 200                 | RandomForestRegressor(n_estimators=100, max_depth=10) | RandomForestClassifier(n_estimators=100, max_depth=10) | RandomForestClassifier(n_estimators=100, max_depth=10) | OK, sin errores         |
