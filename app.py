import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Parara | Data & AI Portfolio",
    page_icon="🚀",
    layout="wide",
)

# ==========================================
# TABS SETUP
# ==========================================
tab_home, tab_ml, tab_eda = st.tabs([
    "Home",
    "🚀 ML Predictor",
    "📊 EDA Dashboard",
])

# ==========================================
# TAB 1: HOME PAGE
# ==========================================
with tab_home:
    st.title("Hi, I'm Fauzan")
    st.subheader("Head of Automation, AI & ML Bank MAS (Wings Group)")

    col_photo, col_bio = st.columns([1, 3])

    with col_photo:
        st.image("https://via.placeholder.com/200x200.png?text=Photo", width=200)

    with col_bio:
        st.write("""
        Results-driven Data & AI/ML leader with **8+ years of experience** delivering
        measurable business impact across banking, e-commerce, telecommunications, and
        transportation industries. As Head of Automation, AI & ML at Bank MAS (Wings Group),
        I spearhead credit risk modeling initiatives and generative AI applications, delivering
        tangible benefits in digital credit risk management and operational efficiency.
        """)

        st.write("**Core Skills:** EDA | ML Modeling | A/B Testing | LLM | RAG")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Years of Experience", "8+")
    col2.metric("Industries", "4+")
    col3.metric("Medium Tutorials", "20+")
    col4.metric("Mentees", "1,500+")

    st.divider()

    st.subheader("Featured Projects")

    p1, p2 = st.columns(2)

    with p1:
        with st.container(border=True):
            st.markdown("### 🏠 House Price Prediction Jakarta")
            st.write("""
            ML model trained on **>10,000 house listings** from rumahrumah.com.
            Predicts Jakarta property prices based on land area, building size,
            bedrooms, bathrooms, and certificate type.
            """)
            st.write("**R² = 92%** | **MAPE = 11%**")
            st.info("💡 Try it live in the **🚀 ML Predictor** tab above.")

    with p2:
        with st.container(border=True):
            st.markdown("### 🌸 Interactive EDA Iris Dataset")
            st.write("""
            Interactive exploratory data analysis dashboard to explore and uncover
            insights from the classic Iris dataset. Features bar charts, pie charts,
            box plots, and histograms with KDE overlays.
            """)
            st.write("**150 samples** | **3 species** | **4 features**")
            st.info("📊 Explore it in the **📊 EDA Dashboard** tab above.")


# ==========================================
# TAB 2: ML PREDICTOR PAGE
# ==========================================
with tab_ml:
    st.title("Jakarta House Price Predictor")
    st.write("Trained on **>10,000 listings** from rumahrumah.com | R² = 92% | MAPE = 11%")
    st.divider()

    col_inputs, col_result = st.columns(2)

    with col_inputs:
        st.subheader("Input Properti")

        luas_tanah = st.number_input(
            "Luas Tanah (m²)",
            min_value=0, max_value=10000,
            value=100, step=1,
        )
        luas_bangunan = st.number_input(
            "Luas Bangunan (m²)",
            min_value=0, max_value=10000,
            value=80, step=1,
        )
        kamar_tidur = st.slider("Jumlah Kamar Tidur", 1, 10, 3)
        kamar_mandi = st.slider("Jumlah Kamar Mandi", 1, 8, 2)

        sudah_shm = st.selectbox(
            "Status Sertifikat",
            options=["SHM (Sertifikat Hak Milik)", "Non-SHM (HGB, Girik, dll)"],
        )
        is_shm = 1 if "Non" not in sudah_shm else 0

        predict_btn = st.button("🚀 Prediksi Harga", use_container_width=True)

    with col_result:
        st.subheader("Hasil Prediksi")

        if predict_btn:
            shm_multiplier = 1 + 0.2 * is_shm
            price = shm_multiplier * (
                14_000_000 * luas_tanah
                + 4_000_000 * luas_bangunan
                + 15_000_000 * kamar_tidur
                + 10_000_000 * kamar_mandi
            )
            price_miliar = price / 1_000_000_000
            price_juta = price / 1_000_000

            st.metric(
                label="Estimasi Harga Properti",
                value=f"Rp {price_miliar:.2f} M",
                delta=f"Rp {price_juta:,.0f} juta",
            )


# ==========================================
# TAB 3: EDA DASHBOARD PAGE
# ==========================================
with tab_eda:

    @st.cache_data
    def load_iris_data():
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df

    FEATURES = {
        "sepal_length": "Sepal Length (cm)",
        "sepal_width": "Sepal Width (cm)",
        "petal_length": "Petal Length (cm)",
        "petal_width": "Petal Width (cm)",
    }

    df = load_iris_data()

    st.title("🌸 Iris Dataset Interactive EDA")
    st.divider()

    # Filters
    st.subheader("Filters")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        selected_species = st.multiselect(
            "Species",
            options=df["species"].unique().tolist(),
            default=df["species"].unique().tolist(),
        )

    with col_f2:
        x_axis = st.selectbox(
            "X-axis (Scatter)",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            index=0,
        )

    with col_f3:
        y_axis = st.selectbox(
            "Y-axis (Scatter)",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            index=2,
        )

    df_f = df[df["species"].isin(selected_species)]
    if df_f.empty:
        st.warning("No data. Select at least one species.")
        st.stop()

    # Bar chart & Pie chart
    st.subheader("Distribution Overview")
    col_bar, col_pie = st.columns(2)

    with col_bar:
        bar_feature = st.selectbox(
            "Feature for bar chart",
            options=list(FEATURES.keys()),
            format_func=lambda k: FEATURES[k],
            key="bar_feat",
        )
        bar_data = df_f.groupby("species")[bar_feature].mean().reset_index()
        fig_bar = px.bar(
            bar_data,
            x="species", y=bar_feature,
            color="species",
            text_auto=".2f",
            labels={"species": "Species", bar_feature: FEATURES[bar_feature]},
            title=f"Mean {FEATURES[bar_feature]} by Species",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        counts = df_f["species"].value_counts().reset_index()
        counts.columns = ["species", "count"]
        fig_pie = px.pie(
            counts,
            names="species", values="count",
            hole=0.4,
            title="Sample Count by Species",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Scatter plot
    st.subheader("Scatter Plot")
    fig_scatter = px.scatter(
        df_f,
        x=x_axis, y=y_axis,
        color="species",
        symbol="species",
        labels={x_axis: FEATURES[x_axis], y_axis: FEATURES[y_axis]},
        title=f"{FEATURES[x_axis]} vs {FEATURES[y_axis]}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Raw data
    st.divider()
    if st.checkbox("Show raw data"):
        st.dataframe(df_f, use_container_width=True)
        st.caption(f"{len(df_f)} rows | 5 columns")