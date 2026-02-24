import numpy as np
from scipy import stats
from itertools import combinations

np.random.seed(42)

# -------------------------
# 1️⃣ Generamos datos truchos
# -------------------------

# Pozo A: media 100, varianza más chica
pozo_A = np.random.normal(loc=100, scale=10, size=30)

# Pozo B: media 110, varianza más grande
pozo_B = np.random.normal(loc=110, scale=20, size=25)

# Pozo C: media 95, varianza intermedia
pozo_C = np.random.normal(loc=95, scale=15, size=40)

# Pozo D: media 120, varianza muy chica
pozo_D = np.random.normal(loc=120, scale=8, size=20)

# Pozo E: media 105, varianza muy grande
pozo_E = np.random.normal(loc=105, scale=30, size=35)

# Pozo F: media 98, varianza chica (similar a A, control negativo)
pozo_F = np.random.normal(loc=98, scale=12, size=28)

pozos = {
    "A": pozo_A,
    "B": pozo_B,
    "C": pozo_C,
    "D": pozo_D,
    "E": pozo_E,
    "F": pozo_F,
}

# -------------------------
# 2️⃣ Estadísticos básicos
# -------------------------

print("=== Estadísticos descriptivos ===")
for nombre, datos in pozos.items():
    print(f"  Pozo {nombre}: n={len(datos)}, media={np.mean(datos):.2f}, var={np.var(datos, ddof=1):.2f}")

# -------------------------
# 3️⃣ Test de Welch entre todos los pares
# -------------------------

pares = list(combinations(pozos.keys(), 2))
n_pozos = len(pozos)
n_tests = len(pares)

print(f"\n=== Conteo de pozos y tests ===")
print(f"Cantidad de pozos: {n_pozos}")
print(f"Cantidad de tests de Welch a hacer: {n_tests}  (todos los pares posibles)")

print("\n=== Tests de Welch (scipy) ===")
for (nombre_x, nombre_y) in pares:
    datos_x = pozos[nombre_x]
    datos_y = pozos[nombre_y]
    t_stat, p_val = stats.ttest_ind(datos_x, datos_y, equal_var=False)
    significativo = "✓ significativo" if p_val < 0.05 else "✗ no significativo"
    print(f"  Pozo {nombre_x} vs Pozo {nombre_y}: t={t_stat:.4f}, p={p_val:.6f}  → {significativo}")