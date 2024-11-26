import streamlit as st
from openai import OpenAI
import pandas as pd
from utils import get_context_from_query, custom_prompt
import json
from Levenshtein import ratio

# Cargar las respuestas predefinidas desde el archivo Excel
respuestas_df = pd.read_excel("PyR_emb.xlsx")

# Abrir y leer el archivo 'credentials.json'
file_name = open('credentials.json')
config_env = json.load(file_name)

# Cargar DataFrame de embeddings previamente almacenado
df_vector_store = pd.read_pickle('df_vector_store.pkl')

# Preguntas sugeridas
suggested_questions = respuestas_df['Pregunta'].tolist()

# Función principal de la página
def main_page():
    # Configuración inicial de la sesión
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""

    # Crear columnas para dividir la interfaz
    col1, col2 = st.columns([1, 1])

    # **Columna izquierda: Chat**
    with col1:
        st.image('usta.png', use_container_width="always")
        st.header(body="Chat personalizado :robot_face:")
        st.subheader('Configuración del modelo :level_slider:')
        
        model_name = st.radio("**Elige un modelo**:", ("GPT-3.5", "GPT-4"))
        st.session_state.model = "gpt-3.5-turbo" if model_name == "GPT-3.5" else "gpt-4"
        
        st.session_state.temperature = st.slider(
            "**Nivel de creatividad de respuesta**  \n  [Poco creativo ►►► Muy creativo]",
            min_value=0.0, max_value=1.0, step=0.1, value=0.0)
        
        st.subheader("Preguntas sugeridas:")
        selected_question = st.selectbox("Selecciona una pregunta para el chatbot:", suggested_questions)
        
        user_query = st.chat_input("O escribe tu consulta manualmente:")
        prompt = user_query if user_query else selected_question
        
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Obtener contexto relevante
                Context_List = get_context_from_query(query=prompt, vector_store=df_vector_store, n_chunks=5)
                client = OpenAI(api_key=config_env["openai_key"])
                
                # Generar respuesta del modelo
                completion = client.chat.completions.create(
                    model=st.session_state.model,
                    temperature=st.session_state.temperature,
                    messages=[
                        {"role": "system", "content": f"{custom_prompt.format(source=str(Context_List))}"}] +
                        st.session_state.message_history +
                        [{"role": "user", "content": prompt}]
                )
                full_response = completion.choices[0].message.content
                message_placeholder.markdown(full_response)

            st.session_state.message_history.append({"role": "user", "content": prompt})
            st.session_state.message_history.append({"role": "assistant", "content": full_response})
            st.session_state.last_response = full_response

    # **Columna derecha: Similitud**
    with col2:
        st.subheader("Comparar respuestas")
        if st.session_state.last_response:
            # Obtener la respuesta esperada
            expected_answer = respuestas_df.loc[respuestas_df['Pregunta'] == selected_question, 'Respuesta Humana'].values[0]

            st.write(f"**Respuesta esperada:** {expected_answer}")
            st.write(f"**Respuesta del chatbot:** {st.session_state.last_response}")

            if st.button("Calcular similitud"):
                # Calcular similitud usando Levenshtein
                similitud = ratio(expected_answer, st.session_state.last_response)
                st.metric("Similitud Levenshtein", f"{similitud:.2%}")

# Punto de entrada
if __name__ == "__main__":
    main_page()
