"""
Simulação de reposição: revisão diária, ponto de pedido (s) e nível máximo (S).

As duas políticas têm a MESMA estrutura. A única diferença é a estimativa de demanda:

  Regra atual:    taxa diária = média dos últimos 28 dias
                  ponto de pedido = taxa x lead time x folga
  Por previsão:   demanda no LT = soma das previsões do modelo para os próximos LT dias
                  ponto de pedido = demanda no LT + z x sigma do erro x raiz(LT)

  Em ambas:       nível máximo = ponto de pedido + taxa diária x dias de cobertura

Comparação justa: cada política é calibrada (folga ou z) até o MENOR valor que atinge
o mesmo fill rate alvo. Assim a pergunta é: quanto capital cada uma precisa para
entregar o mesmo nível de serviço?

Premissas: pedido emitido no fim do dia chega no início do dia t + LT;
demanda não atendida é venda perdida; sem custo de pedido nem restrição de lote mínimo.
"""

import numpy as np
import pandas as pd

FILL_RATE_ALVO = 99.5
COBERTURA_DIAS = 7  # reposição aproximadamente semanal


def simular(g, lead_time, politica, parametro, sigma=0.0, cobertura=COBERTURA_DIAS) -> dict:
    demanda = g["demanda"].to_numpy()
    previsto = g["prev_modelo"].to_numpy()
    media_28 = g["media_28"].to_numpy()
    n = len(demanda)

    estoque = media_28[0] * (lead_time + cobertura)  # mesmo ponto de partida para todas
    chegadas = np.zeros(n + lead_time + 1)
    vendido = perdido = dias_ruptura = pedidos = 0
    estoque_diario = []

    for t in range(n):
        estoque += chegadas[t]
        venda = min(estoque, demanda[t])
        estoque -= venda
        vendido += venda
        perdido += demanda[t] - venda
        dias_ruptura += int(demanda[t] > venda)
        estoque_diario.append(estoque)

        if politica == "regra_atual":
            taxa = media_28[t]
            ponto_pedido = taxa * lead_time * parametro
        else:
            janela = previsto[t + 1:t + 1 + lead_time]
            if len(janela) < lead_time:  # fim do período: repete a última previsão
                janela = np.concatenate([janela, np.full(lead_time - len(janela), previsto[-1])])
            taxa = janela.mean()
            ponto_pedido = janela.sum() + parametro * sigma * np.sqrt(lead_time)

        posicao = estoque + chegadas[t + 1:t + lead_time + 1].sum()
        if posicao <= ponto_pedido:
            chegadas[t + lead_time] += round(ponto_pedido + taxa * cobertura - posicao)
            pedidos += 1

    return {
        "fill_rate_%": round(100 * vendido / demanda.sum(), 2),
        "dias_com_ruptura": dias_ruptura,
        "unidades_perdidas": int(perdido),
        "estoque_medio_un": round(float(np.mean(estoque_diario)), 1),
        "pedidos": pedidos,
        "_serie": estoque_diario,
    }


def calibrar(g, lead_time, politica, sigma, alvo, cobertura):
    """Menor folga (regra atual) ou z (previsão) que atinge o fill rate alvo."""
    grade = np.arange(0.8, 4.01, 0.05) if politica == "regra_atual" else np.arange(-1.0, 4.01, 0.05)
    for p in grade:
        r = simular(g, lead_time, politica, p, sigma, cobertura)
        if r["fill_rate_%"] >= alvo:
            return round(float(p), 2), r
    return round(float(grade[-1]), 2), r


def comparar(teste, sigma, alvo=FILL_RATE_ALVO, cobertura=COBERTURA_DIAS):
    linhas, series = [], {}
    for sku, g in teste.sort_values("data").groupby("sku"):
        lt = int(g["lead_time_dias"].iloc[0])
        custo = float(g["custo_unitario"].iloc[0])
        s = float(sigma[sku])

        cenarios = {
            "Regra atual (folga fixa 20%)": (1.2, simular(g, lt, "regra_atual", 1.2, s, cobertura)),
            "Regra atual calibrada": calibrar(g, lt, "regra_atual", s, alvo, cobertura),
            "Por previsão calibrada": calibrar(g, lt, "previsao", s, alvo, cobertura),
        }
        for rotulo, (param, r) in cenarios.items():
            r = dict(r)
            series[(sku, rotulo)] = (g["data"].to_numpy(), r.pop("_serie"))
            linhas.append({
                "sku": sku, "politica": rotulo,
                "parametro": f"z = {param}" if "previsão" in rotulo else f"folga = {param}",
                **r,
                "capital_em_estoque_R$": round(r["estoque_medio_un"] * custo, 0),
                "venda_perdida_custo_R$": round(r["unidades_perdidas"] * custo, 0),
            })
    return pd.DataFrame(linhas), series


def sensibilidade(teste, sigma, coberturas=(7, 14), alvos=(99.0, 99.5, 99.9)):
    """Redução de capital da política por previsão vs. regra atual, ambas calibradas."""
    linhas = []
    for cob in coberturas:
        for alvo in alvos:
            sim, _ = comparar(teste, sigma, alvo, cob)
            cap = sim.groupby("politica")["capital_em_estoque_R$"].sum()
            base, prev = cap["Regra atual calibrada"], cap["Por previsão calibrada"]
            linhas.append({
                "cobertura_dias": cob, "fill_rate_alvo_%": alvo,
                "capital_regra_atual_R$": base, "capital_previsao_R$": prev,
                "reducao_capital_%": round(100 * (1 - prev / base), 1),
            })
    return pd.DataFrame(linhas)
