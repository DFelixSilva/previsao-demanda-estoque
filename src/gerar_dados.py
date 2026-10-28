"""
Gera um dataset SINTÉTICO de vendas diárias para 5 SKUs ao longo de 2 anos.

Por que sintético: permite controlar os padrões (tendência, sazonalidade, promoções)
e verificar se o modelo os captura. Os parâmetros imitam um varejo pet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# sku, demanda base/dia, tendência anual, força da sazonalidade anual, lead time (dias), custo unitário (R$)
SKUS = [
    ("SKU-001 Ração Premium 10kg", 40, 0.15, 0.10, 5, 120.0),
    ("SKU-002 Areia Higiênica 4kg", 65, 0.05, 0.05, 3, 25.0),
    ("SKU-003 Antipulgas 3un", 18, 0.10, 0.45, 7, 90.0),          # sazonal: pico no verão
    ("SKU-004 Petisco Dental", 30, 0.30, 0.10, 4, 15.0),          # produto em crescimento
    ("SKU-005 Shampoo Neutro 500ml", 12, -0.10, 0.20, 6, 22.0),   # produto em declínio
]

# efeito médio por dia da semana (seg..dom)
EFEITO_SEMANA = np.array([0.95, 0.90, 0.92, 0.98, 1.10, 1.25, 0.90])


def gerar(inicio="2024-01-01", dias=730, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datas = pd.date_range(inicio, periods=dias, freq="D")
    t = np.arange(dias)
    tabelas = []

    for sku, base, tendencia, saz, lead_time, custo in SKUS:
        nivel = base * (1 + tendencia * t / 365)
        # pico em janeiro (verão no Brasil), vale em julho
        sazonal = 1 + saz * np.cos(2 * np.pi * (datas.dayofyear.to_numpy() - 15) / 365)
        semana = EFEITO_SEMANA[datas.dayofweek.to_numpy()]

        # promoções planejadas: ~1 a cada 6 semanas, duram 3 dias, +60% de demanda
        promo = np.zeros(dias, dtype=int)
        for ini in rng.choice(np.arange(10, dias - 3), size=dias // 42, replace=False):
            promo[ini:ini + 3] = 1

        media = nivel * sazonal * semana * (1 + 0.6 * promo)
        demanda = rng.poisson(media)

        tabelas.append(pd.DataFrame({
            "data": datas, "sku": sku, "demanda": demanda, "promocao": promo,
            "lead_time_dias": lead_time, "custo_unitario": custo,
        }))

    return pd.concat(tabelas, ignore_index=True)


if __name__ == "__main__":
    df = gerar()
    destino = Path(__file__).resolve().parents[1] / "data" / "vendas_diarias.csv"
    df.to_csv(destino, index=False)
    print(f"{len(df):,} linhas salvas em {destino}")
