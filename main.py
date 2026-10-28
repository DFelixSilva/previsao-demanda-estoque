"""Executa o pipeline completo: dados -> previsão -> simulação de estoque -> resultados."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import gerar_dados, politica_estoque, previsao

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "data" / "vendas_diarias.csv"
SAIDA = RAIZ / "resultados"
GRAF = SAIDA / "graficos"
SKU_EXEMPLO = "SKU-003 Antipulgas 3un"
AZUL, LARANJA, CINZA = "#2f6db5", "#e07b39", "#9aa5b1"


def carregar() -> pd.DataFrame:
    if not DADOS.exists():
        DADOS.parent.mkdir(exist_ok=True)
        gerar_dados.gerar().to_csv(DADOS, index=False)
    return pd.read_csv(DADOS, parse_dates=["data"])


def resumir(sim: pd.DataFrame, demanda_total: int) -> pd.DataFrame:
    total = sim.groupby("politica", sort=False).agg(
        unidades_perdidas=("unidades_perdidas", "sum"),
        dias_com_ruptura=("dias_com_ruptura", "sum"),
        capital_em_estoque_R=("capital_em_estoque_R$", "sum"),
        venda_perdida_custo_R=("venda_perdida_custo_R$", "sum"),
        pedidos=("pedidos", "sum"),
    )
    total.insert(0, "fill_rate_%", (100 * (1 - total["unidades_perdidas"] / demanda_total)).round(2))
    return total.rename(columns=lambda c: c.replace("_R", "_R$") if c.endswith("_R") else c)


def graficos(teste, geral, resumo, series):
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    GRAF.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(geral["modelo"], geral["WAPE_%"], color=[CINZA, CINZA, AZUL])
    for i, v in enumerate(geral["WAPE_%"]):
        ax.text(v + 0.3, i, f"{v}%", va="center")
    ax.set_xlabel("WAPE (%), menor é melhor")
    ax.set_title("Erro de previsão no período de teste (120 dias)")
    fig.tight_layout(); fig.savefig(GRAF / "01_erro_por_modelo.png"); plt.close(fig)

    g = teste[teste["sku"] == SKU_EXEMPLO]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(g["data"], g["demanda"], color=CINZA, lw=1, label="Demanda real")
    ax.plot(g["data"], g["prev_media"], color=LARANJA, lw=1.5, ls="--", label="Média móvel 28d")
    ax.plot(g["data"], g["prev_modelo"], color=AZUL, lw=2, label="Gradient Boosting")
    ax.set_title(f"{SKU_EXEMPLO}: a média móvel fica atrasada quando a demanda sobe")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(GRAF / "02_real_vs_previsto.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for rotulo, cor in (("Regra atual (folga fixa 20%)", LARANJA), ("Por previsão calibrada", AZUL)):
        datas, estoque = series[(SKU_EXEMPLO, rotulo)]
        ax.plot(datas, estoque, color=cor, lw=1.8, label=rotulo)
    ax.axhline(0, color="#c0392b", lw=1)
    ax.set_title(f"{SKU_EXEMPLO}: estoque ao fim do dia (zero = ruptura)")
    ax.set_ylabel("unidades"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(GRAF / "03_estoque_por_politica.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    cap = resumo["capital_em_estoque_R$"] / 1000
    ax.barh(cap.index, cap.values, color=[LARANJA, CINZA, AZUL])
    for i, v in enumerate(cap.values):
        ax.text(v + 0.5, i, f"R$ {v:,.1f} mil", va="center")
    ax.set_xlabel("Capital médio em estoque (R$ mil, a custo)")
    ax.set_title("Capital necessário por política")
    fig.tight_layout(); fig.savefig(GRAF / "04_capital_por_politica.png"); plt.close(fig)


def main():
    df = carregar()
    teste, sigma, geral, por_sku = previsao.executar(df)
    sim, series = politica_estoque.comparar(teste, sigma)
    resumo = resumir(sim, int(teste["demanda"].sum()))
    sens = politica_estoque.sensibilidade(teste, sigma)

    SAIDA.mkdir(exist_ok=True)
    geral.to_csv(SAIDA / "metricas_previsao.csv", index=False)
    por_sku.to_csv(SAIDA / "metricas_previsao_por_sku.csv", index=False)
    sim.to_csv(SAIDA / "simulacao_por_sku.csv", index=False)
    resumo.to_csv(SAIDA / "simulacao_resumo.csv")
    sens.to_csv(SAIDA / "sensibilidade.csv", index=False)
    graficos(teste, geral, resumo, series)

    pd.set_option("display.width", 220)
    print("\n== Previsão ==\n", geral.to_string(index=False))
    print("\n== Previsão por SKU ==\n", por_sku.to_string(index=False))
    print("\n== Simulação por SKU ==\n", sim.to_string(index=False))
    print("\n== Resumo ==\n", resumo.to_string())
    print("\n== Sensibilidade ==\n", sens.to_string(index=False))


if __name__ == "__main__":
    main()
