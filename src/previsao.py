"""
Previsão de demanda diária por SKU.

Compara 3 abordagens no mesmo período de teste:
  1. Ingênuo sazonal  -> demanda de 7 dias atrás
  2. Média móvel      -> média dos 28 dias, disponível 7 dias antes
  3. Gradient Boosting -> modelo global (todos os SKUs) com lags, calendário e promoções

Regra anti-vazamento: toda feature usa dados de pelo menos 7 dias antes da data prevista.
Assim cada previsão é válida para horizontes de até 7 dias, o que cobre o lead time
de todos os SKUs (3 a 7 dias). Promoções entram como feature porque são planejadas
com antecedência.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONTE_MIN = 7
DIAS_TESTE = 120
DIAS_VALIDACAO = 90

FEATURES = [
    "lag_7", "lag_14", "lag_21", "lag_28", "media_28", "media_7",
    "dia_semana", "dia_ano_sin", "dia_ano_cos", "promocao", "sku_cod",
]


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku", "data"]).copy()
    g = df.groupby("sku")["demanda"]
    for lag in (7, 14, 21, 28):
        df[f"lag_{lag}"] = g.shift(lag)
    df["media_28"] = g.transform(lambda s: s.shift(HORIZONTE_MIN).rolling(28).mean())
    df["media_7"] = g.transform(lambda s: s.shift(HORIZONTE_MIN).rolling(7).mean())
    df["dia_semana"] = df["data"].dt.dayofweek
    df["dia_ano_sin"] = np.sin(2 * np.pi * df["data"].dt.dayofyear / 365)
    df["dia_ano_cos"] = np.cos(2 * np.pi * df["data"].dt.dayofyear / 365)
    df["sku_cod"] = df["sku"].astype("category").cat.codes
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def treinar(df_treino: pd.DataFrame) -> HistGradientBoostingRegressor:
    modelo = HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.05,
        categorical_features=[FEATURES.index("sku_cod")], random_state=42,
    )
    return modelo.fit(df_treino[FEATURES], df_treino["demanda"])


def dividir(df: pd.DataFrame, dias_finais: int):
    corte = df["data"].max() - pd.Timedelta(days=dias_finais)
    return df[df["data"] <= corte], df[df["data"] > corte]


def wape(real, previsto) -> float:
    return float(np.abs(real - previsto).sum() / real.sum())


def metricas(real, previsto) -> dict:
    return {
        "WAPE_%": round(100 * wape(real, previsto), 1),
        "MAE": round(float(np.mean(np.abs(real - previsto))), 2),
        "Viés_%": round(100 * float((previsto.sum() - real.sum()) / real.sum()), 1),
    }


def executar(df_bruto: pd.DataFrame):
    df = criar_features(df_bruto)
    treino, teste = dividir(df, DIAS_TESTE)

    # 1) erro do modelo em validação -> usado no estoque de segurança
    ajuste, validacao = dividir(treino, DIAS_VALIDACAO)
    validacao = validacao.assign(previsto=treinar(ajuste).predict(validacao[FEATURES]))
    sigma = (validacao.assign(erro=validacao["demanda"] - validacao["previsto"])
             .groupby("sku")["erro"].std().rename("sigma_erro"))

    # 2) modelo final, treinado com todo o histórico anterior ao teste
    modelo = treinar(treino)
    teste = teste.assign(
        prev_ingenuo=teste["lag_7"],
        prev_media=teste["media_28"],
        prev_modelo=modelo.predict(teste[FEATURES]),
    )

    nomes = {"prev_ingenuo": "Ingênuo sazonal (lag 7)",
             "prev_media": "Média móvel 28 dias",
             "prev_modelo": "Gradient Boosting"}
    geral = pd.DataFrame([{"modelo": n, **metricas(teste["demanda"], teste[c])}
                          for c, n in nomes.items()])
    por_sku = pd.DataFrame([
        {"sku": sku, "modelo": n, **metricas(g["demanda"], g[c])}
        for sku, g in teste.groupby("sku") for c, n in nomes.items()
    ])
    return teste, sigma, geral, por_sku
