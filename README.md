# 📊 Análise Churn Rate - Challenge Data Science - Parte 2

<br>

## 🚀 Tecnologias Utilizadas
<div>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-FA8C00?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Badge">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Requests-478778?style=for-the-badge&logo=requests&logoColor=white">
  <img src="https://img.shields.io/badge/Seaborn-1f77b4?style=for-the-badge&logo=seaborn&logoColor=white">
  <img src="https://img.shields.io/badge/Plotly-FF6600?style=for-the-badge&logo=plotly&logoColor=white">
</div>

<br>

## ▶️ Como Executar o Notebook

1. Clone este repositório (se estiver usando Git):
git clone https://github.com/Brunex-Alado/Analise-Dados-3

2.  Abra o Jupyter Notebook:
Você pode usar VS Code, Jupyter Lab, Jupyter Notebook ou Google Colab.

3. Instale as bibliotecas Pandas e Matplotlib:
pip install pandas
pip install matplotlib
pip install plotly
pip install request
pip install seaborn
pip install numpy

4. Execute todas as células em ordem:
O notebook lê automaticamente os dados via internet e gera todos os gráficos.

✅ Importante: Requer conexão com a internet para ler os dados CSV.

<br>

## 📝 Propósito da Análise

- Desenvolver **modelos preditivos** capazes de prever quais clientes têm maior chance de cancelar seus serviços.

- A empresa quer **antecipar o problema da evasão**, assim será construído um pipeline robusto para essa etapa inicial de modelagem.

<br>

## 🛠️ Preparação dos Dados

- Preparar os dados para a **modelagem** (*tratamento, encoding, normalização*).
- Realizar **análise de correlação** e seleção de variáveis.
- Treinar dois ou mais modelos de classificação.
- Avaliar o desempenho dos modelos com **métricas**.
- **Interpretar os resultados**, incluindo a importância das variáveis.
- Criar uma **conclusão estratégica** apontando os principais fatores que influenciam a evasão.
  
<br>  

## 🎯 Objetivo

- ✅ Pré-processamento de dados para Machine Learning

- ✅ Construção e avaliação de modelos preditivos

- ✅ Interpretação dos resultados e entrega de insights

- ✅ Comunicação técnica com foco estratégico

<br/>

## 🔍 Visualização da Proporção de Evasão

| ![image](https://raw.githubusercontent.com/Brunex-Alado/Analise-Dados-3/refs/heads/main/img/distribuicao_evasao_cliente.png) | 

<br/>

### ⚙️ **Normalização / Padronização dos Dados**

Como parte do pré-processamento, foi aplicada a **padronização** dos atributos numéricos, utilizando a técnica `StandardScaler`, que transforma os dados para que tenham média zero e desvio padrão um.

Esse passo é fundamental para o bom desempenho de algoritmos que são sensíveis à escala dos dados, como:

- **KNN** (K-Nearest Neighbors)
- **SVM** (Support Vector Machines)
- **Regressão Logística**
- **Redes Neurais**

A padronização foi aplicada **apenas após o balanceamento das classes e a separação entre dados de treino e teste**, para evitar vazamento de dados e garantir que as transformações ocorram apenas com base nos dados de treino. Com isso, garantimos uma comparação justa e válida durante a avaliação dos modelos.

<br>

## 🧠 Correlação e Seleção de Variáveis

| ![image](https://raw.githubusercontent.com/Brunex-Alado/Analise-Dados-3/refs/heads/main/img/matriz_correlacao_variaveis_numericas.png) | 

<br/>

### 🔍 **Análise de Correlação**

- A **correlação entre “Meses_de_Contrato” e “Valor_Mensal”  é fraca (0.25)**.

- A **correlação entre “Meses_de_Contrato” e “Cancelamento”  é fraca (0.35)**.

- A **correlação entre “Valor_Mensal” e “Cancelamento”  é fraca (0.19)**.

*Isso sugere que não há uma relação linear forte entre o tempo de contrato e o valor pago mensalmente, e entre cada uma das variáveis com o cancelamento.*

<br/>

| ![image](https://raw.githubusercontent.com/Brunex-Alado/Analise-Dados-3/refs/heads/main/img/analise_direcionada.png) | 

<br/>

### 🔍 **Análise Direcionada**

Gráfico 1 Boxplot - Distribuição dos meses de contrato para clientes que permaneceram e os que evadiram: Podemos observar que clientes que evadiram tendem a ter um tempo de contrato menor do que os que permaneceram. Isso pode indicar que a evasão ocorre mais frequentemente nos primeiros meses de contrato.

Gráfico 2 Stripplot - Valor Mensal por Evasão: Apesar de haver sobreposição, é possível notar uma concentração maior de valores mensais mais altos entre os clientes que evadiram, sugerindo que valores mais elevados podem estar relacionados à evasão.

<br/>

## 🤖 Modelagem Preditiva

| ![image](https://raw.githubusercontent.com/Brunex-Alado/Analise-Dados-3/refs/heads/main/img/comparativo_desempenho_modelos.png) | 

<br/>

## 🔍 Análise Exploratória e Modelagem Preditiva

- **Proporção de evasão:** cerca de 25,8% dos clientes cancelaram os serviços.
- **Modelos testados:** Regressão Logística, Random Forest e modelo Random.
- **Melhor modelo:** Regressão Logística, com acurácia de 79,96%, precisão de 65,44%, recall de 52,14% e F1-score de 58,04%.
- Variáveis mais relevantes para previsão:
  - **Tempo de Contrato** (clientes com contratos mais curtos têm maior risco)
  - **Valor Mensal** (valores mais altos correlacionam com evasão)
  - Serviços adicionais como segurança online e backup também influenciam.

<br/>

## 🚀 Interpretação e Conclusão

| ![image](https://raw.githubusercontent.com/Brunex-Alado/Analise-Dados-3/refs/heads/main/img/importancia_variaveis.png) | 

<br/>

## 🔍 Análise de Importância das Variáveis

O gráfico exibe as variáveis mais relevantes identificadas pelo modelo **Regressão Logística**, treinado com dados normalizados, para prever a **evasão de clientes**.

As variáveis posicionadas no topo do gráfico apresentam **maior influência nas decisões do modelo**, com base na magnitude dos coeficientes. Isso significa que pequenas variações nessas variáveis têm um impacto significativo na probabilidade de um cliente cancelar os serviços.

<br/>>

## ✅ Recomendações Estratégicas

- Desenvolver **ações de retenção personalizadas** para clientes com contratos curtos e valores mensais elevados.
- Oferecer **pacotes promocionais e descontos progressivos** para aumentar a fidelização.
- Utilizar o modelo de **Regressão Logística para monitoramento contínuo**, possibilitando intervenções proativas em clientes com maior risco de evasão.

<br/>