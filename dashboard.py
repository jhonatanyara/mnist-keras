import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.title("Reconocimiento de Números (MNIST)")
st.write("Sube una imagen de un número dibujado a mano (del 0 al 9).")

# Cargar el modelo entrenado (se usa st.cache_resource para no recargarlo cada vez)
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_mnist.keras")

modelo = cargar_modelo()

# Botón para subir archivo
archivo_subido = st.file_uploader("Elige una imagen (PNG o JPG)", type=["png", "jpg", "jpeg"])

if archivo_subido is not None:
    # Abrir y mostrar la imagen
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Imagen cargada", width=200)

    # Preprocesar la imagen para que coincida con MNIST
    # 1. Convertir a escala de grises
    img_procesada = imagen.convert('L')
    # 2. Invertir colores (MNIST tiene fondo negro y trazo blanco)
    img_procesada = ImageOps.invert(img_procesada)
    # 3. Redimensionar a 28x28
    img_procesada = img_procesada.resize((28, 28))
    # 4. Convertir a arreglo numpy y normalizar los píxeles (0 a 1)
    img_array = np.array(img_procesada) / 255.0
    # 5. Agregar la dimensión del lote (batch) para Keras
    img_array = img_array.reshape(1, 28, 28)

    # Hacer la predicción
    prediccion = modelo.predict(img_array)
    numero_predicho = np.argmax(prediccion)

    # Mostrar el resultado en texto grande
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Predicción: {numero_predicho}</h1>", unsafe_allow_html=True)
