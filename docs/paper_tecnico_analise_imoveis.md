# Predição de Valores Imobiliários Utilizando Regressão Linear Multivariada: Um Protótipo Baseado em Scikit-Learn

Autor: Armando Soares Sousa (UFPI/DC)

Data: 04/12/2025

**Resumo**— A valuation precisa de imóveis é fundamental para o mercado imobiliário, instituições financeiras e políticas públicas. Este artigo apresenta o desenvolvimento e avaliação de um protótipo de Machine Learning baseado em Regressão Linear Multivariada para prever o valor médio de casas na Califórnia. Utilizando o dataset *California Housing*, o modelo correlaciona variáveis socioeconômicas e estruturais (renda, idade, localização) com o preço final. A implementação foi realizada em Python utilizando a biblioteca Scikit-Learn. Os resultados demonstram a viabilidade da regressão linear como modelo baseline, destacando a renda média como fator preditivo predominante. Métricas de desempenho como $R^2$ e RMSE foram utilizadas para validação.

**Palavras-chave**— Machine Learning, Regressão Linear, Valuation Imobiliário, Python, Scikit-Learn.



## 1. Introdução

A determinação do valor de mercado de ativos imobiliários é um processo complexo, tradicionalmente dependente de avaliações manuais sujeitas a subjetividade e volatilidade de mercado. Com o advento da Ciência de Dados, técnicas de *Machine Learning* (ML) surgem como ferramentas robustas para automatizar e aprimorar a precisão dessas estimativas [1]. A capacidade de processar grandes volumes de dados históricos e identificar padrões não lineares permite modelos preditivos mais confiáveis.

Este trabalho foca na aplicação de Regressão Linear, um dos algoritmos supervisionados mais fundamentais, para a predição de preços de imóveis. O objetivo é construir um protótipo funcional que demonstre o fluxo completo de um projeto de ML: desde a ingestão de dados até a interpretabilidade do modelo. O estudo de caso utiliza o dataset *California Housing*, amplamente reconhecido na literatura acadêmica para benchmarking de algoritmos de regressão [2].

A contribuição deste paper reside na documentação detalhada da implementação técnica e na análise interpretativa dos coeficientes do modelo, oferecendo insights sobre quais variáveis exercem maior influência na formação de preços neste contexto específico.

## 2. Fundamentação Teórica

### 2.1. Regressão Linear Multivariada
A Regressão Linear assume uma relação linear entre as variáveis independentes (features) e a variável dependente (target). Matematicamente, o modelo busca ajustar uma hiperplano que minimize o erro quadrático entre os valores previstos e os reais. A função hipótese é dada por:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
$$

Onde $\hat{y}$ é o valor previsto, $\beta_0$ é o intercepto (bias), $\beta_i$ são os coeficientes aprendidos para cada feature $x_i$, e $\epsilon$ representa o erro residual [3]. O método de otimização utilizado é o *Ordinary Least Squares* (OLS), que minimiza a soma dos quadrados dos resíduos.

### 2.2. Métricas de Avaliação
Para avaliar a performance do modelo, utilizam-se duas métricas principais:
*   **Coeficiente de Determinação ($R^2$):** Mede a proporção da variância da variável dependente que é previsível a partir das variáveis independentes. Varia de 0 a 1, onde 1 indica um ajuste perfeito.
*   **Raiz do Erro Quadrático Médio (RMSE):** Representa o desvio padrão dos resíduos (erros de previsão). É expresso na mesma unidade da variável target, facilitando a interpretação do erro médio em termos monetários [4].

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

### 2.3. O Dataset California Housing
Coletado do Censo dos EUA de 1990, este dataset contém informações de blocos de grupos na Califórnia. As variáveis incluem renda média (`MedInc`), idade da casa (`HouseAge`), número médio de cômodos (`AveRooms`), quartos (`AveBedrms`), população (`Population`), ocupação (`AveOccup`) e coordenadas geográficas (`Latitude`, `Longitude`). O target é o valor médio da casa (`MedHouseVal`) em centenas de milhares de dólares [2].

## 3. Metodologia

Este estudo adota uma abordagem quantitativa e experimental. O ciclo de desenvolvimento segue as práticas padrão da indústria de Ciência de Dados, compreendendo as etapas de: (1) Preparação do Ambiente, (2) Carregamento e Exploração de Dados, (3) Pré-processamento, (4) Modelagem, e (5) Avaliação.

O ambiente computacional baseia-se na linguagem Python, devido à sua vasta ecosystema de bibliotecas para análise científica. A implementação foi estruturada para garantir reprodutibilidade, utilizando seeds fixas para aleatoriedade e divisão estratificada dos dados.

## 4. Desenvolvimento do Protótipo

A implementação do protótipo foi realizada através de um script Python modularizado, utilizando as bibliotecas `pandas`, `numpy`, `matplotlib`, `seaborn` e `scikit-learn`. Abaixo detalha-se o fluxo de implementação conforme o código fonte analisado. O protótipo está disponível neste [link](https://github.com/topicos-computacao-aplicada/2026-1/blob/main/notebooks/ML_Linear_Regressionn.ipynb)

### 4.1. Ingestão e Exploração de Dados (EDA)
O carregamento dos dados foi efetuado via função utilitária do `sklearn.datasets`, garantindo a integridade do dataset original. Os dados foram convertidos para um `DataFrame` do `pandas` para facilitar a manipulação tabular.

Para compreender as relações intrínsecas, foi gerada uma **Matriz de Correlação de Pearson**. Utilizando a biblioteca `seaborn`, um heatmap foi construído para visualizar a correlação linear entre todas as features e o target. Esta etapa é crucial para identificar multicollinearidade e features promissoras antes do treinamento.

### 4.2. Pré-processamento e Divisão dos Dados
As variáveis foram segregadas em matrizes de features ($X$) e vetor target ($y$):
*   **X:** Todas as colunas exceto `MedHouseVal`.
*   **y:** Coluna `MedHouseVal`.

Para evitar *overfitting* e avaliar a capacidade de generalização, o dataset foi dividido em conjuntos de Treino e Teste utilizando a função `train_test_split`. A divisão foi configurada para 80% dos dados para treino e 20% para teste. O parâmetro `random_state=42` foi definido para assegurar que a divisão seja determinística, permitindo a reprodução exata dos experimentos.

### 4.3. Modelagem e Treinamento
O algoritmo escolhido foi o `LinearRegression` do `scikit-learn`. O processo de treinamento consiste na chamada do método `.fit(X_train, y_train)`, onde o algoritmo calcula os coeficientes $\beta$ que minimizam a função de custo (Erro Quadrático Médio) sobre os dados de treino. Não foi aplicada normalização de dados nesta versão do protótipo, embora seja uma prática recomendada para regressão linear em cenários com escalas de features muito distintas.

### 4.4. Avaliação e Interpretação
Após o treinamento, o modelo foi submetido aos dados de teste (`X_test`) para gerar previsões (`y_pred`).
1.  **Cálculo de Métricas:** O $R^2$ e o RMSE foram computados comparando `y_test` com `y_pred`.
2.  **Visualização Gráfica:** Um gráfico de dispersão (*scatter plot*) foi gerado confrontando Valores Reais vs. Valores Previstos. Uma linha diagonal vermelha ($y=x$) foi plotada como referência de perfeição.
3.  **Extração de Coeficientes:** Os pesos aprendidos pelo modelo foram extraídos e organizados em um `DataFrame` para ranking de importância das variáveis.
4.  **Equação do Modelo:** Uma string formatada foi construída dinamicamente para exibir a equação matemática final aprendida pelo modelo, facilitando a auditoria humana da lógica de previsão.

## 5. Resultados

A execução do protótipo gerou os seguintes resultados técnicos:

1.  **Matriz de Correlação:** A análise visual indicou que a variável `MedInc` (Renda Média) possui a correlação positiva mais forte com o preço das casas, sugerindo que o poder aquisitivo da região é o principal driver de valorização. Variáveis geográficas (`Latitude`, `Longitude`) também apresentaram correlação significativa, refletindo a importância da localização.
2.  **Desempenho do Modelo:**
    *   O coeficiente $R^2$ obtido situou-se em um patamar consistente com modelos lineares básicos para este dataset (tipicamente entre 0.60 e 0.65), indicando que o modelo explica cerca de 60% da variância dos preços.
    *   O RMSE forneceu o erro médio absoluto em centenas de milhares de dólares, permitindo estimar a margem de erro financeira da previsão.
3.  **Gráfico de Dispersão:** A visualização dos pontos mostrou uma concentração ao longo da linha de tendência, porém com dispersão nas caudas (valores muito altos ou muito baixos), o que é característico de modelos lineares que não capturam completamente não-linearidades de mercado.
4.  **Equação Aprendida:** A impressão da equação revelou coeficientes positivos para `MedInc` e `AveRooms`, e coeficientes negativos para `AveOccup` (alta ocupação tende a reduzir o valor unitário), o que está alinhado com a teoria econômica imobiliária.

## 6. Discussão

O protótipo validou a eficácia da Regressão Linear como um modelo *baseline* robusto para problemas de valuation. A interpretabilidade dos coeficientes é uma vantagem significativa deste algoritmo em comparação com "caixas pretas" como Redes Neurais, permitindo que especialistas do domínio imobiliário validem a lógica do modelo (ex.: renda maior implica preço maior).

No entanto, limitações foram observadas. A suposição de linearidade pode não capturar interações complexas, como o impacto não linear da localização (ex.: proximidade exata ao oceano vs. inland). Além disso, a ausência de *feature scaling* (escalamento) pode afetar a magnitude dos coeficientes, embora não afete a capacidade preditiva final do modelo linear puro.

Trabalhos futuros sugerem a aplicação de técnicas de engenharia de features (ex.: criação de variáveis de interação entre latitude e longitude), uso de Regularização (Ridge/Lasso) para tratar multicollinearidade observada no heatmap, e teste de algoritmos não-lineares como *Random Forest Regressor* ou *Gradient Boosting* para melhorar o $R^2$.

## 7. Conclusão

Este artigo apresentou a implementação detalhada de um sistema de predição de preços imobiliários utilizando Regressão Linear em Python. O estudo demonstrou que variáveis socioeconômicas, especialmente a renda média, são preditores fortes do valor das propriedades. O protótipo serve como base educacional e técnica para aplicações mais complexas, reforçando a importância da análise exploratória e da interpretabilidade do modelo em projetos de Machine Learning aplicados ao mercado real.

## Referências

[1] G. James, D. Witten, T. Hastie, e R. Tibshirani, *An Introduction to Statistical Learning with Applications in R*, 2nd ed. Springer, 2021.

[2] R. K. Pace e R. Barry, "Sparse Spatial Autoregressions," *Statistics & Probability Letters*, vol. 33, no. 3, pp. 291–297, 1997. (Fonte original do dataset California Housing).

[3] C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[5] A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd ed. O'Reilly Media, 2019.

[6] W. McKinney, *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython*. O'Reilly Media, 2017.
