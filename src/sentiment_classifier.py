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
        description="Frase completa que mais impactou a classificação do sentimento."
    )


# Configura o modelo e prompt
def config_model():
    prompt = PromptTemplate.from_template(
        """
Você é um classificador de sentimentos e extrator de frases. 

Sua tarefa é:
1. Classificar o sentimento geral como:
   -1 para negativo,
    0 para neutro,
    1 para positivo.
2. Extrair a frase completa do texto que mais influenciou essa classificação. 
A frase deve ser copiada exatamente como está no texto original.

Retorne os resultados no seguinte formato JSON:
{{
  "sentiment": -1 | 0 | 1,
  "most_relevant_phrase": "Frase completa que mais impactou a classificação do sentimento."
}}

Texto a ser analisado: 
{text}
"""
    )

    llm = ChatOllama(model="phi4", format="json", temperature=0.1)
    structured_llm = llm.with_structured_output(Response)
    _chain = prompt | structured_llm

    return prompt, _chain


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
        name = row.get("nome", "")
        text = row.get("texto", "")

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
    root_path = os.path.join(os.path.dirname(__file__), "..", "data")

    csv_path = os.path.join(root_path, "datasets", sys.argv[1])

    path_data = os.path.join(root_path, "outputs")

    path_logs = os.path.join(root_path, "logs")

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)

    df = load_data(csv_path)
    prompt, _chain = config_model()

    gc.collect()

    run(df, path_logs, path_data)

    gc.collect()
