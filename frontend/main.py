import streamlit as st
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

api_endpoint = os.environ.get("API_ENDPOINT", "default_endpoint")

def main():
    st.set_page_config(page_title="AeroBot", layout='wide')
    apply_custom_css()
    chat_page()

def apply_custom_css():
    st.markdown("""
        <style>
            /* Cambiar el color de fondo de la página */
            .stApp {
                background-color: #e6f0ff;
            }

            /* Personalizar el título y los subtítulos */
            h1 {
                color: #333366;
            }

            h2, h3, h4, h5, h6 {
                color: #555599;
            }
        </style>
    """, unsafe_allow_html=True)

def chat_page():
    st.image("aero.png", use_column_width=True)
    st.markdown("<h1>AeroBot</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Chatbot para la Aerocivil de Colombia</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input("Haz una pregunta sobre procedimientos y normativas de la aerocivil:", "")
    with col2:
        ask_button = st.button("Obtener respuesta")

    if ask_button and user_question:
        chat_container = st.container()
        with chat_container:
            st.markdown(f"**Tú:** {user_question}")
        with st.spinner("Generando la respuesta..."):
            response = get_response(user_question, True)
            if response:
                with chat_container:
                    st.markdown(response)
            else:
                st.error("Por favor intenta de nuevo.")
    elif ask_button:
        st.warning("Por favor, ingresa una pregunta.")

def get_response(question, verbose):
    endpoint = api_endpoint
    headers = {"Content-Type": "application/json"}
    data = {"question": question, "verbose": verbose}

    try:
        response = requests.post(endpoint, headers=headers, json=data)
        if response.status_code == 200:
            response_json = response.json()
            answer = json.loads(response_json.get('respuesta', ''))
            sources = json.loads(response_json.get('fuente', ''))
            formatted_response = f"**Respuesta:**\n{answer}\n\n**Fuentes:**\n{'; '.join(sources)}"
            return formatted_response
        else:
            st.error(f"Falla en generar la respuesta: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Una excepción ha ocurrido: {e}")
        return None

if __name__ == "__main__":
    main()
