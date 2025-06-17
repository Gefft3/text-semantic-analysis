import pandas as pd
import os
import sys
import gc
import tiktoken
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# Modelo de resposta estruturada
class Response(BaseModel):
    sentiment: int = Field(
        description="Variação entre -1, 0, 1 do sentimento do texto."
    )
    most_relevant_phrase: str = Field(
        description="Frase que mais impactou a classificação do sentimento."
    )


# Configura o modelo e prompt
def config_model():
    prompt = PromptTemplate.from_template(
        """
Objetivo principal:
Você é um classificador de sentimentos. 
Você irá receber um texto e deve classificar se ele é negativo, neutro ou positivo. 

Entrada esperadas:
- Texto a ser classificado

Atributos esperados na saída:
- sentiment: inteiro que indica a classificação do sentimento, odne -1 é negativo, 0 é neutro e 1 é positivo.
- most_relevant_phrase: string inteiro da frase que mais impactou na classificação do sentimento.

  
Formato da saída:
{{
  "sentiment": -1|0|1,
  "most_relevant_phrase": Frase completa que mais impactou a classificação do sentimento.,
}}

Texto:
{text}
"""
    )

    llm = ChatOllama(model="llama3.2", format="json", temperature=0.8)
    structured_llm = llm.with_structured_output(Response)
    _chain = prompt | structured_llm

    return prompt, _chain


# Chunkinização usando tokenização aproximada
def chunk_text(text, max_tokens=2000):
    if not isinstance(text, str):
        text = str(text)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]


# Carregamento de dados
def load_data(csv_path):
    return pd.read_csv(csv_path)


# Processamento do texto e geração de resposta
def process_text(name, text, path_logs):
    try:
        response = _chain.invoke({"text": text})

        formatted_prompt = prompt.format(text=text)

        with open(os.path.join(path_logs, "prompts.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"Nome: {name}\nprompt:\n{formatted_prompt}\n-----------------------\n"
            )

        with open(os.path.join(path_logs, "responses.txt"), "a", encoding="utf-8") as f:
            f.write(f"Nome: {name}\nresponse:\n{response}\n-----------------------\n")

        response_data = {
            "nome": name,
            "sentimento": response.sentiment,
            "frase": response.most_relevant_phrase,
        }
        return response_data

    except Exception as e:
        # Salva os erros
        with open(os.path.join(path_logs, "errors.txt"), "a", encoding="utf-8") as f:
            f.write(f"Nome: {name} | Erro: {e}\n-----------------------\n")


# Processa o CSV completo
def run(df, path_logs, path_data):
    df_classes = pd.DataFrame(columns=["nome", "sentimento", "frase"])

    for i, row in tqdm(df.iterrows(), total=len(df)):
        name = row.get("name", "")
        text = row.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        text_processed = process_text(name, text, path_logs)

        if df_classes.empty:
            df_classes = pd.DataFrame([text_processed])
        else:
            df_aux = pd.DataFrame([text_processed])
            df_classes = pd.concat([df_classes, df_aux], ignore_index=True)

        gc.collect()

    df_classes.to_csv(
        os.path.join(path_data, "classifications.csv"),
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    # Espera caminho relativo para pasta data
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", sys.argv[1])

    path_data = os.path.join(os.path.dirname(__file__), "..", "outputs")

    path_logs = os.path.join(os.path.dirname(__file__), "..", "logs")

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)

    df = load_data(csv_path)
    prompt, _chain = config_model()

    gc.collect()

    run(df, path_logs, path_data)

    gc.collect()
