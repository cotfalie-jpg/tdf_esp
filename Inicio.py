import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="TF-IDF BAE 🌸",
    page_icon="🍼",
    layout="centered"
)

# 🎨 ESTILO BAE: Pasteles suaves y tipografía redondeada
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

.stApp {
    background: linear-gradient(180deg, #FFF8E1 0%, #FFFDF5 100%);
    color: #3E3E3E;
    font-family: 'Poppins', sans-serif;
}

/* Título */
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    color: #3E3E3E;
    margin-bottom: 0.3rem;
    animation: fadeInDown 1.2s ease;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #52796F;
    margin-bottom: 1.5rem;
    animation: fadeIn 2s ease;
}

/* Tarjetas */
.bae-box {
    background: #FFF9EC;
    border-radius: 20px;
    border: 2px solid #FFD89C;
    padding: 1.5rem;
    box-shadow: 0 8px 20px rgba(255, 220, 150, 0.2);
    animation: fadeIn 1.2s ease;
}

/* Botones */
.stButton>button {
    background: linear-gradient(135deg, #FAD689, #F6C667, #A0D8EF);
    color: #2F3E46;
    border: none;
    border-radius: 15px;
    padding: 0.8rem 2rem;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(246, 198, 103, 0.3);
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(160, 216, 239, 0.4);
}

/* Inputs */
textarea, input {
    border-radius: 12px !important;
    border: 2px solid #FFD89C !important;
}

/* Métricas */
.similarity-high { color: #5B8C5A; font-weight: 700; animation: glow 2s ease-in-out infinite alternate; }
.similarity-medium { color: #F6AE2D; font-weight: 700; }
.similarity-low { color: #E76F51; font-weight: 700; }

/* Animaciones */
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
@keyframes fadeInDown { from {opacity: 0; transform: translateY(-20px);} to {opacity: 1; transform: translateY(0);} }
@keyframes glow { from {text-shadow: 0 0 5px rgba(91,140,90,0.5);} to {text-shadow: 0 0 15px rgba(91,140,90,0.8);} }

/* Expander */
.st-expander {
    border-radius: 15px !important;
    background-color: #FFF9EC !important;
    border: 1px solid #FFD89C !important;
}

/* Colores secundarios */
.metric-card {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #A0D8EF;
}
</style>
""", unsafe_allow_html=True)

# TÍTULO
st.markdown('<div class="main-title">🌿 Analizador TF-IDF BAE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compara tus textos con amor pastel y encuentra similitudes 💛</div>', unsafe_allow_html=True)

# DOCUMENTOS DE EJEMPLO
default_docs = """La luna ilumina el cielo nocturno con su suave resplandor.
Las estrellas brillan como pequeños destellos mágicos.
Los bebés sueñan tranquilos bajo las mantas de algodón.
El sol da calor al mundo y pinta sonrisas en los días felices.
Las flores abren sus pétalos cuando llega la primavera."""

# CONFIGURACIÓN DEL STEMMER
stemmer = SnowballStemmer("spanish")
def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# FUNCIÓN DE ANÁLISIS
def analyze_documents(documents, question):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
    X = vectorizer.fit_transform(documents)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X).flatten()
    best_idx = similarities.argmax()
    df_tfidf = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"Doc {i+1}" for i in range(len(documents))]
    )
    return {
        "best_idx": best_idx,
        "best_doc": documents[best_idx],
        "best_score": similarities[best_idx],
        "similarities": similarities,
        "df_tfidf": df_tfidf
    }

# INTERFAZ
st.markdown('<div class="bae-box">', unsafe_allow_html=True)
st.markdown("**📄 Escribe tus frases (una por línea):**")
docs = st.text_area("", default_docs, height=140)
st.markdown("**❓ Escribe tu pregunta:**")
question = st.text_input("", "¿Qué ilumina el cielo nocturno?")

if st.button("✨ Analizar Similitud", use_container_width=True):
    documents = [d.strip() for d in docs.split("\n") if d.strip()]
    if not documents or not question.strip():
        st.warning("Por favor, completa los campos antes de analizar 🌸")
    else:
        with st.spinner("Analizando con ternura pastel... 🌼"):
            result = analyze_documents(documents, question)
            st.markdown("### 🧸 Resultado principal")
            st.success(f"**Documento más similar:** {result['best_doc']}")
            st.markdown(
                f"<div class='similarity-high'>Similitud: {result['best_score']:.3f}</div>",
                unsafe_allow_html=True
            )

            with st.expander("Ver detalles de similitud"):
                sim_df = pd.DataFrame({
                    "Documento": [f"Doc {i+1}" for i in range(len(result['similarities']))],
                    "Similitud": result["similarities"],
                    "Texto": documents
                }).sort_values("Similitud", ascending=False)
                st.dataframe(sim_df, use_container_width=True)

            with st.expander("Matriz TF-IDF"):
                st.dataframe(result["df_tfidf"].round(3), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# PIE DE PÁGINA
st.markdown("""
<div style='text-align:center; margin-top:2rem; color:#52796F; font-size:0.9rem;'>
Hecho con 💛 por <b>BAE</b> | Textos dulces y análisis con cariño 🌿
</div>
""", unsafe_allow_html=True)
