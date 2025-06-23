import pandas as pd
import os
import sys
import gc
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# Modelo de resposta estruturada
class Response(BaseModel):
    classes_found: list[str] = Field(
        description="Lista de classes encontradas no texto."
    )
    classes_list: list[str] = Field(
        description="Lista de classes atualizada após a análise do texto."
    )


# Configura o modelo e prompt
def config_model():
    prompt = PromptTemplate.from_template(
        """
Você é um extrator de classes. 
Dada uma lista de classes e um texto, retorne um lista das classes que aparecem no texto.
Caso encontre uma categoria nova, adicione-a à lista de classes e retorne-a.

classes_found: lista de classes encontradas no texto.
classes_list: lista de classes atualizada após a análise do texto.

Retorne no formato JSON:
{{
    "classes_found": ["classe1", "classe2", ...],
    "classes_list": ["classe1", "classe2", ...]
}}

Lista de classes: {classes_list}

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
def process_text(name, text, classes_list, path_logs):
    try:
        response = _chain.invoke({"classes_list": classes_list, "text": text})

        formatted_prompt = prompt.format(text=text, classes_list=classes_list)

        with open(os.path.join(path_logs, "prompts.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"Nome: {name}\nprompt:\n{formatted_prompt}\n-----------------------\n"
            )

        with open(os.path.join(path_logs, "responses.txt"), "a", encoding="utf-8") as f:
            f.write(f"Nome: {name}\nresponse:\n{response}\n-----------------------\n")

        return response.classes_found, response.classes_list

    except Exception as e:
        # Salva os erros
        with open(os.path.join(path_logs, "errors.txt"), "a", encoding="utf-8") as f:
            f.write(f"Nome: {name} | Erro: {e}\n-----------------------\n")


# Processa o CSV completo
def run(df, path_logs, path_data):
    df_classes = pd.DataFrame(columns=["classe", "count"])

    classes_list = [
        "liderança",
        "pro atividade",
        "inovação",
        "colaboração",
        "resiliência",
    ]

    for i, row in tqdm(df.iterrows(), total=len(df)):
        name = row.get("nome", "")
        text = row.get("frase", "")

        classes_list, classes_found = process_text(name, text, classes_list, path_logs)

        if classes_found:
            for classe in classes_found:
                if classe not in df_classes["classe"].values:
                    df_classes = pd.concat(
                        [df_classes, pd.DataFrame({"classe": [classe], "count": [1]})],
                        ignore_index=True,
                    )
                else:
                    df_classes.loc[df_classes["classe"] == classe, "count"] += 1

        gc.collect()

    df_classes.to_csv(
        os.path.join(path_data, "classes_count.csv"),
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    # Espera caminho relativo para pasta data
    root_path = os.path.join(os.path.dirname(__file__), "..", "data")

    csv_path = os.path.join(root_path, "datasets", sys.argv[1])

    path_data = os.path.join(root_path, "extraction/outputs")

    path_logs = os.path.join(root_path, "extraction/logs")

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)

    df = load_data(csv_path)
    prompt, _chain = config_model()

    gc.collect()

    run(df, path_logs, path_data)

    gc.collect()
