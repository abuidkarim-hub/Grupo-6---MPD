# =============================================================
# PANEL INTEGRADO MPD - GRUPO 6
# Retail Supply Chain: pronostico, inventario y elasticidad
# Ejecutar:  streamlit run app_mpd.py
# =============================================================

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

st.set_page_config(page_title="MPD - Grupo 6", layout="wide")

CATS = ["Furniture", "Office Supplies", "Technology"]
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"]
MESES_ALTA = [9, 11, 12]
AZUL, NARANJA, GRIS = "#1F4E78", "#ED7D31", "#A5A5A5"

st.title("Panel integrado de planificacion - Retail Supply Chain")
st.caption("Grupo 6 | Pronostico de demanda, inventario objetivo y elasticidad de portafolio")

# ------------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------------
st.sidebar.header("1. Datos")
archivo = st.sidebar.file_uploader("Cargar otro archivo (opcional)", type=["xlsx"])

# Si nadie carga un archivo, se usa el Excel del proyecto que esta junto al script
import os
RUTA_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Input_Modelo_Demanda.xlsx")
fuente = archivo if archivo is not None else RUTA_DEFECTO
if archivo is None:
    st.sidebar.caption("Usando Input_Modelo_Demanda.xlsx del proyecto.")


@st.cache_data
def cargar(f):
    serie = pd.read_excel(f, sheet_name="Serie_Categoria")
    trans = pd.read_excel(f, sheet_name="Transacciones")
    return serie, trans


serie, trans = cargar(fuente)

# ------------------------------------------------------------------
# PARAMETROS
# ------------------------------------------------------------------
st.sidebar.header("2. Parametros")
categoria = st.sidebar.selectbox("Categoria de producto", CATS)
confianza = st.sidebar.slider("Nivel de confianza de los escenarios", 0.80, 0.99, 0.95, 0.01)
nivel_serv = st.sidebar.slider("Nivel de servicio objetivo", 0.80, 0.99, 0.95, 0.01)
lead_time = st.sidebar.number_input("Tiempo de reposicion (meses)", 1, 6, 1)
anio_proy = 2018

z = stats.norm.ppf(nivel_serv)
st.sidebar.caption(f"z = {z:.3f} para un nivel de servicio de {nivel_serv:.0%}")


# ------------------------------------------------------------------
# MODELO DE PRONOSTICO (Dimension 1)
# ------------------------------------------------------------------
def ajustar(y, X):
    """Regresion lineal por minimos cuadrados. Devuelve coeficientes y error tipico."""
    A = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    gl = len(y) - A.shape[1]
    sigma = np.sqrt((resid ** 2).sum() / gl)
    cov = sigma ** 2 * np.linalg.inv(A.T @ A)
    return beta, sigma, np.sqrt(np.diag(cov)), gl


def mape(real, pred):
    return np.mean(np.abs((real - pred) / real)) * 100


def rmse(real, pred):
    return np.sqrt(np.mean((real - pred) ** 2))


@st.cache_data
def comparar_modelos(serie):
    """Modelo 1 (tendencia) vs Modelo 2 (tendencia + temporada alta), evaluados en 2017."""
    filas = []
    for cat in CATS:
        g = serie[serie.Categoria == cat]
        tr, te = g[g.Anio <= 2016], g[g.Anio == 2017]
        fila = {"Categoria": cat}
        for nom, cols in [("M1", ["Tendencia"]), ("M2", ["Tendencia", "Temporada_Alta"])]:
            b, *_ = ajustar(tr.Unidades_Vendidas.values, tr[cols].values.astype(float))
            pred = b[0] + te[cols].values.astype(float) @ b[1:]
            fila[f"MAPE {nom}"] = round(mape(te.Unidades_Vendidas.values, pred), 1)
            fila[f"RMSE {nom}"] = round(rmse(te.Unidades_Vendidas.values, pred), 1)
            fila[f"Exactitud {nom}"] = round(100 - mape(te.Unidades_Vendidas.values, pred), 1)
        fila["Modelo elegido"] = "Modelo 2" if fila["MAPE M2"] < fila["MAPE M1"] else "Modelo 1"
        filas.append(fila)
    return pd.DataFrame(filas)


def proyectar(serie, cat, confianza, factor=1.0):
    """Reentrena con los 48 meses y proyecta 2018 con escenarios por intervalo de confianza."""
    g = serie[serie.Categoria == cat]
    X = g[["Tendencia", "Temporada_Alta"]].values.astype(float)
    b, sigma, se, gl = ajustar(g.Unidades_Vendidas.values, X)
    tc = stats.t.ppf(1 - (1 - confianza) / 2, gl)

    mes = np.arange(1, 13)
    tend = np.arange(49, 61)
    alta = np.isin(mes, MESES_ALTA).astype(float)

    base = b[0] + b[1] * factor * tend + b[2] * alta
    pes = b[0] + (b[1] - tc * se[1]) * factor * tend + (b[2] - tc * se[2]) * alta
    opt = b[0] + (b[1] + tc * se[1]) * factor * tend + (b[2] + tc * se[2]) * alta

    df = pd.DataFrame({"Mes": mes, "Nombre_Mes": MESES, "Tendencia": tend,
                       "Temporada_Alta": alta.astype(int),
                       "Pesimista": pes, "Base": base, "Optimista": opt})
    return df, b, sigma


# ------------------------------------------------------------------
# PESTANAS
# ------------------------------------------------------------------
t1, t2, t3, t4, t5 = st.tabs([
    "Datos de entrada", "D1 Pronostico", "D2 Inventario",
    "D3 Elasticidad", "D4 Panel integrado"])

# ============================ TAB 1 ============================
with t1:
    st.header("Consideraciones de datos de entrada")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transacciones", f"{len(trans):,}")
    c2.metric("Observaciones del modelo", f"{len(serie)}")
    c3.metric("Pedidos", f"{trans.Order_ID.nunique():,}")
    c4.metric("Productos", f"{trans.Product_ID.nunique():,}")

    st.subheader("Variables transaccionales")
    tv = trans[["Quantity", "Sales", "Discount", "Profit"]]
    desc = pd.DataFrame({
        "Media": tv.mean(), "Moda": tv.mode().iloc[0], "Mediana": tv.median(),
        "Desv. Est.": tv.std(), "Minimo": tv.min(), "Maximo": tv.max(),
        "Coef. Var.": tv.std() / tv.mean()})
    st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Evolucion mensual de las unidades vendidas")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for cat, col in zip(CATS, [AZUL, NARANJA, "#70AD47"]):
        g = serie[serie.Categoria == cat]
        ax.plot(g.Tendencia, g.Unidades_Vendidas, label=cat, lw=1.6, color=col)
    ax.set_xlabel("Mes (1 = enero 2014)")
    ax.set_ylabel("Unidades")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    st.pyplot(fig)

    st.subheader("Analisis de reduccion de dimensiones")

    @st.cache_data
    def candidatas(trans):
        d = trans.copy()
        d["lp"] = d.Sales / (1 - d.Discount)
        g = d.groupby([d.Anio, d.Mes, d.Category])
        m = g.apply(lambda x: pd.Series({
            "Unidades": x.Quantity.sum(),
            "Precio_unitario": x.Sales.sum() / x.Quantity.sum(),
            "Descuento_prom": 1 - x.Sales.sum() / x.lp.sum(),
            "Margen": x.Profit.sum() / x.Sales.sum(),
            "Tasa_devolucion": (x.Returned == "Si").mean(),
            "Part_Consumer": (x.Segment == "Consumer").mean(),
            "Part_West": (x.Region == "West").mean(),
            "Part_Standard": (x.Ship_Mode == "Standard Class").mean(),
            "Ticket_prom": x.Quantity.sum() / x.Order_ID.nunique()})).reset_index()
        m["Tendencia"] = (m.Anio - 2014) * 12 + m.Mes
        m["Temporada_Alta"] = m.Mes.isin(MESES_ALTA).astype(int)
        return m

    cand = candidatas(trans)
    Xc = cand.drop(columns=["Anio", "Mes", "Category", "Unidades"])
    yc = cand.Unidades
    Xs = StandardScaler().fit_transform(Xc)

    alpha = st.slider("Penalizacion Lasso (alpha)", 0.01, 0.40, 0.10, 0.01)

    c1, c2 = st.columns(2)
    with c1:
        p = PCA().fit(Xs)
        acum = np.cumsum(p.explained_variance_ratio_) * 100
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(range(1, len(acum) + 1), p.explained_variance_ratio_ * 100,
               color="#8FB8DE", edgecolor=AZUL)
        ax.plot(range(1, len(acum) + 1), acum, "o-", color="#C0504D", lw=2, ms=4)
        for u in (80, 90):
            ax.axhline(u, ls="--", lw=.8, color="grey")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Varianza explicada (%)")
        ax.set_title("PCA sobre las variables candidatas", fontsize=10, fontweight="bold")
        st.pyplot(fig)
        st.caption(f"Componentes para el 80 %: {int(np.argmax(acum >= 80) + 1)} | "
                   f"para el 90 %: {int(np.argmax(acum >= 90) + 1)} de {Xc.shape[1]}")

    with c2:
        ys = (yc - yc.mean()) / yc.std()
        co = pd.Series(Lasso(alpha=alpha, max_iter=50000).fit(Xs, ys).coef_,
                       index=Xc.columns).sort_values()
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.barh(range(len(co)), co.values,
                color=[GRIS if v == 0 else AZUL for v in co.values])
        ax.set_yticks(range(len(co)))
        ax.set_yticklabels(co.index, fontsize=8)
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("Coeficiente estandarizado")
        ax.set_title(f"Lasso (alpha = {alpha})", fontsize=10, fontweight="bold")
        st.pyplot(fig)
        viv = list(co[co != 0].index)
        st.caption(f"Sobreviven {len(viv)} de {len(co)}: {', '.join(viv)}")

    st.info("El ticket promedio y el precio unitario se calculan con las ventas del mismo mes, "
            "por lo que no estan disponibles antes de que el periodo ocurra. El modelo final "
            "retiene **Tendencia** y **Temporada_Alta**, las unicas conocidas de antemano.")

# ============================ TAB 2 ============================
with t2:
    st.header(f"Dimension 1: Pronostico de demanda - {categoria}")

    comp = comparar_modelos(serie)
    st.subheader("Comparacion de modelos sobre 2017")
    st.dataframe(comp.set_index("Categoria"), use_container_width=True)

    fila = comp[comp.Categoria == categoria].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Exactitud Modelo 1", f"{fila['Exactitud M1']:.1f} %")
    c2.metric("Exactitud Modelo 2", f"{fila['Exactitud M2']:.1f} %",
              f"{fila['Exactitud M2'] - fila['Exactitud M1']:+.1f} pp")
    c3.metric("Modelo elegido", fila["Modelo elegido"])

    st.subheader("Proyeccion 2018")
    factor = st.slider("Factor de crecimiento (analisis de sensibilidad)",
                       0.5, 1.5, 1.0, 0.1,
                       help="1.0 mantiene el crecimiento estimado por el modelo.")

    proy, b, sigma = proyectar(serie, categoria, confianza, factor)

    st.caption(f"Ecuacion: Unidades = {b[0]:.2f} + {b[1]:.2f} x Tendencia "
               f"+ {b[2]:.2f} x Temporada_Alta | Error tipico = {sigma:.2f} unidades")

    real17 = serie[(serie.Categoria == categoria) & (serie.Anio == 2017)].Unidades_Vendidas.sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real 2017", f"{real17:,.0f}")
    for col, nom in zip([c2, c3, c4], ["Pesimista", "Base", "Optimista"]):
        tot = proy[nom].sum()
        col.metric(nom, f"{tot:,.0f}", f"{tot / real17 - 1:+.0%}")

    hist = serie[serie.Categoria == categoria]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.Tendencia, hist.Unidades_Vendidas, color="black", lw=1.6, label="Historico")
    ax.fill_between(proy.Tendencia, proy.Pesimista, proy.Optimista,
                    color=AZUL, alpha=.15, label=f"Banda {confianza:.0%}")
    ax.plot(proy.Tendencia, proy.Base, "o-", color=NARANJA, lw=2, ms=4, label="Base 2018")
    ax.set_xlabel("Mes (1 = enero 2014)")
    ax.set_ylabel("Unidades")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    st.pyplot(fig)

    st.dataframe(proy[["Nombre_Mes", "Temporada_Alta", "Pesimista", "Base", "Optimista"]]
                 .style.format({"Pesimista": "{:.0f}", "Base": "{:.0f}", "Optimista": "{:.0f}"}),
                 use_container_width=True, hide_index=True)

# ============================ TAB 3 ============================
with t3:
    st.header(f"Dimension 2: Inventario objetivo - {categoria}")

    proy, b, sigma = proyectar(serie, categoria, confianza, 1.0)
    ss = int(np.ceil(z * sigma * np.sqrt(lead_time)))

    inv = proy.copy()
    inv["Stock_Seguridad"] = ss
    inv["Inventario_Objetivo"] = np.ceil(inv.Base + ss).astype(int)

    st.latex(r"SS = z \times \sigma \times \sqrt{L} \qquad "
             r"Inventario\ objetivo = Demanda\ base + SS")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Error tipico del pronostico", f"{sigma:.2f} u")
    c2.metric("Stock de seguridad", f"{ss} u")
    c3.metric("Inventario objetivo maximo", f"{inv.Inventario_Objetivo.max():,} u")
    c4.metric("Inventario acumulado del anio", f"{inv.Inventario_Objetivo.sum():,} u")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(inv.Nombre_Mes, inv.Stock_Seguridad, color=NARANJA, alpha=.45,
           label="Stock de seguridad")
    ax.plot(inv.Nombre_Mes, inv.Base, "o-", color="#70AD47", lw=2, ms=5,
            label="Demanda base")
    ax.plot(inv.Nombre_Mes, inv.Inventario_Objetivo, "s--", color=NARANJA, lw=2, ms=5,
            label="Inventario objetivo")
    ax.fill_between(inv.Nombre_Mes, inv.Pesimista, inv.Optimista, color=AZUL, alpha=.12,
                    label=f"Banda de demanda {confianza:.0%}")
    ax.set_ylabel("Unidades fisicas")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.dataframe(inv[["Nombre_Mes", "Base", "Stock_Seguridad", "Inventario_Objetivo"]]
                 .style.format({"Base": "{:.0f}"}),
                 use_container_width=True, hide_index=True)

    st.download_button("Descargar inventario objetivo (CSV)",
                       inv.to_csv(index=False).encode("utf-8"),
                       f"inventario_{categoria}.csv", "text/csv")

# ============================ TAB 4 ============================
with t4:
    st.header("Dimension 3: Elasticidad y gestion de portafolio")

    @st.cache_data
    def elasticidad(trans):
        d = trans.copy()
        d["Unit_Price"] = d.Sales / d.Quantity
        d["Has_Discount"] = (d.Discount > 0).astype(int)
        filas = []
        for cat in CATS:
            g = d[(d.Category == cat) & (d.Quantity > 0) & (d.Unit_Price > 0)]
            X = sm.add_constant(pd.DataFrame({
                "log_price": np.log(g.Unit_Price), "Has_Discount": g.Has_Discount}))
            m = sm.OLS(np.log(g.Quantity), X).fit()
            filas.append({
                "Categoria": cat,
                "Elasticidad precio (Ep)": round(m.params["log_price"], 4),
                "Sensibilidad al descuento": round(m.params["Has_Discount"], 4),
                "R2 ajustado": round(m.rsquared_adj, 4),
                "Participacion unidades (%)": round(g.Quantity.sum() / d.Quantity.sum() * 100, 2),
                "Participacion ventas (%)": round(g.Sales.sum() / d.Sales.sum() * 100, 2),
                "Pedidos con descuento (%)": round(g.Has_Discount.mean() * 100, 2),
                "Descuento promedio (%)": round(g.Discount.mean() * 100, 2),
                "Margen de utilidad (%)": round(g.Profit.sum() / g.Sales.sum() * 100, 2)})
        return pd.DataFrame(filas)

    ela = elasticidad(trans)

    st.subheader("Indicadores econometricos (modelo log-log)")
    st.latex(r"\log(Quantity) = \beta_0 + \beta_1 \log(Unit\_Price) + \beta_2 Has\_Discount")
    st.dataframe(ela[["Categoria", "Elasticidad precio (Ep)",
                      "Sensibilidad al descuento", "R2 ajustado"]].set_index("Categoria"),
                 use_container_width=True)

    st.subheader("Metricas comparativas del portafolio")
    st.dataframe(ela[["Categoria", "Participacion unidades (%)", "Participacion ventas (%)",
                      "Pedidos con descuento (%)", "Descuento promedio (%)",
                      "Margen de utilidad (%)"]].set_index("Categoria"),
                 use_container_width=True)

    st.subheader("Dispersion de precio unitario frente al volumen de ventas")
    d = trans.copy()
    d["Unit_Price"] = d.Sales / d.Quantity
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for cat, col in zip(CATS, ["#C0504D", AZUL, "#70AD47"]):
        g = d[d.Category == cat]
        ax.scatter(g.Unit_Price, g.Quantity, s=9, alpha=.45, label=cat, color=col)
    ax.set_xlabel("Precio unitario ($)")
    ax.set_ylabel("Volumen de ventas (cantidad)")
    ax.legend(title="Categoria", fontsize=8)
    ax.grid(alpha=.3)
    st.pyplot(fig)

    st.info("Las elasticidades son cercanas a cero y el R2 ajustado es practicamente nulo: "
            "la demanda de este retail no responde al precio ni al descuento, sino a la "
            "tendencia y la estacionalidad, lo que respalda la especificacion de la Dimension 1.")

# ============================ TAB 5 ============================
with t5:
    st.header("Dimension 4: Panel integrado de escenarios")

    resumen = []
    for cat in CATS:
        p, bb, sg = proyectar(serie, cat, confianza, 1.0)
        ssc = int(np.ceil(z * sg * np.sqrt(lead_time)))
        real = serie[(serie.Categoria == cat) & (serie.Anio == 2017)].Unidades_Vendidas.sum()
        resumen.append({
            "Categoria": cat, "Real 2017": int(real),
            "Demanda pesimista": int(p.Pesimista.sum()),
            "Demanda base": int(p.Base.sum()),
            "Demanda optimista": int(p.Optimista.sum()),
            "Error tipico": round(sg, 2),
            "Stock de seguridad": ssc,
            "Inventario objetivo acumulado": int(np.ceil(p.Base + ssc).sum()),
            "Pico maximo de inventario": int(np.ceil(p.Base + ssc).max())})
    res = pd.DataFrame(resumen)

    tot = res[["Real 2017", "Demanda base", "Inventario objetivo acumulado"]].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demanda total 2018 (base)", f"{tot['Demanda base']:,}")
    c2.metric("Variacion vs. 2017", f"{tot['Demanda base'] / tot['Real 2017'] - 1:+.1%}")
    c3.metric("Inventario objetivo total", f"{tot['Inventario objetivo acumulado']:,}")
    c4.metric("Exactitud promedio del modelo",
              f"{comparar_modelos(serie)['Exactitud M2'].mean():.1f} %")

    st.dataframe(res.set_index("Categoria"), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        x = np.arange(3)
        ax.bar(x - .2, res["Real 2017"], .4, label="Real 2017", color=GRIS)
        ax.bar(x + .2, res["Demanda base"], .4, label="Base 2018", color=AZUL)
        ax.errorbar(x + .2, res["Demanda base"],
                    yerr=[res["Demanda base"] - res["Demanda pesimista"],
                          res["Demanda optimista"] - res["Demanda base"]],
                    fmt="none", ecolor="black", capsize=4, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(CATS, fontsize=8)
        ax.set_ylabel("Unidades")
        ax.set_title("Demanda anual por categoria", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.bar(CATS, res["Demanda base"], color=AZUL, label="Demanda base")
        ax.bar(CATS, res["Stock de seguridad"] * 12, bottom=res["Demanda base"],
               color=NARANJA, label="Stock de seguridad anual")
        ax.set_ylabel("Unidades")
        ax.set_title("Composicion del inventario objetivo", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        plt.xticks(fontsize=8)
        st.pyplot(fig)

    st.download_button("Descargar resumen integrado (CSV)",
                       res.to_csv(index=False).encode("utf-8"),
                       "resumen_integrado.csv", "text/csv")

    st.success(
        f"Con un nivel de servicio de {nivel_serv:.0%} y {lead_time} mes(es) de reposicion, "
        f"la red requiere {tot['Inventario objetivo acumulado']:,} unidades de inventario "
        f"objetivo acumulado para atender una demanda base de {tot['Demanda base']:,} unidades "
        f"en 2018. Office Supplies concentra la mayor exigencia de almacenamiento.")
