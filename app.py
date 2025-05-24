
import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Cargar archivo CSV
st.title("📝 Análisis de Opiniones de Clientes")
archivo = st.file_uploader("Sube el archivo CSV con la columna 'Opinion'", type="csv")

if archivo:
    df = pd.read_csv(archivo)
    st.write("Primeras opiniones:")
    st.dataframe(df.head(10))

    opiniones = df['Opinion'].astype(str).tolist()
    texto = " ".join(opiniones).lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    palabras = texto.split()

    stop_words = set(stopwords.words('spanish'))
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words and len(palabra) > 2]

    # Nube de palabras
    st.subheader("☁️ Nube de Palabras")
    nube = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras_filtradas))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(nube, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Palabras más frecuentes
    st.subheader("📊 Palabras más frecuentes")
    conteo = Counter(palabras_filtradas)
    palabras_comunes = conteo.most_common(10)
    etiquetas, valores = zip(*palabras_comunes)
    fig2, ax2 = plt.subplots()
    ax2.bar(etiquetas, valores, color='pink')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Análisis de sentimientos
    st.subheader("❤️ Análisis de Sentimientos")
    modelo_sentimientos = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(modelo_sentimientos)
    model = AutoModelForSequenceClassification.from_pretrained(modelo_sentimientos)
    clasificador = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    resultados = clasificador(opiniones)

    def interpretar(label):
        estrellas = int(label[0])
        if estrellas >= 4:
            return "Positivo"
        elif estrellas == 3:
            return "Neutro"
        else:
            return "Negativo"

    df['Sentimiento'] = [interpretar(r['label']) for r in resultados]
    conteo_sentimientos = df['Sentimiento'].value_counts()

    fig3, ax3 = plt.subplots()
    ax3.pie(conteo_sentimientos, labels=conteo_sentimientos.index, autopct='%1.1f%%', colors=['pink', 'violet', 'skyblue'])
    ax3.set_title("Distribución de Sentimientos")
    st.pyplot(fig3)

    st.subheader("🗣 Comentario Nuevo")
    nuevo_comentario = st.text_input("Escribe tu comentario aquí:")

    if nuevo_comentario:
        resultado = clasificador(nuevo_comentario)[0]
        sentimiento = interpretar(resultado['label'])
        palabras_clave = [w for w in re.findall(r'\b\w+\b', nuevo_comentario.lower()) if w not in stop_words]
        palabras_clave = palabras_clave[:5]

        st.write(f"✅ Sentimiento: **{sentimiento}**")
        st.write("🔍 Palabras clave:", ", ".join(palabras_clave) if palabras_clave else "No se encontraron palabras clave")
