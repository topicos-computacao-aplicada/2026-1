# Segmentação de Clientes via Aprendizado Não Supervisionado: Um Protótipo Baseado em K-Means para Estratégias de Marketing

Autor: Armando Soares Sousa (UFPI/DC)

Data: 10/11/2025

**Resumo**— No cenário contemporâneo de Marketing 4.0, a personalização de ofertas é crucial para a retenção e conversão de clientes. No entanto, a ausência de rótulos prévios em grandes bases de dados dificulta a categorização manual. Este trabalho apresenta um protótipo de segmentação de clientes utilizando o algoritmo de clusterização K-Means, uma técnica de aprendizado não supervisionado. O estudo de caso demonstra a implementação em Python, utilizando as bibliotecas Scikit-Learn e Pandas, para agrupar clientes com base em idade, renda e gastos. Os resultados indicam a formação de três clusters distintos, permitindo a identificação de perfis como "clientes premium" e "caçadores de promoções". Conclui-se que a aplicação de K-Means é viável para descoberta de padrões ocultos, embora requera atenção à escalabilidade e pré-processamento em cenários reais.

**Palavras-chave**— K-Means, Segmentação de Clientes, Aprendizado Não Supervisionado, Mineração de Dados, Marketing Analítico.

## I. INTRODUÇÃO

A transformação digital impulsionou a geração de volumes massivos de dados nas organizações, fenômeno conhecido como *Big Data*. No contexto de marketing, a capacidade de transformar esses dados em insights acionáveis é um diferencial competitivo. A segmentação de mercado, tradicionalmente feita por critérios demográficos estáticos, evoluiu para uma abordagem comportamental dinâmica [1].

O problema central reside na dificuldade de classificar clientes quando não existem categorias predefinidas (rótulos). Métodos de aprendizado supervisionado exigem dados históricos rotulados, o que nem sempre está disponível. Nesse cenário, o aprendizado não supervisionado, especificamente a clusterização, surge como uma solução para descobrir estruturas intrínsecas nos dados [2].

Este trabalho tem como objetivo desenvolver e analisar um protótipo de segmentação de clientes utilizando o algoritmo K-Means. A contribuição deste estudo reside na demonstração prática da aplicação da técnica para identificar perfis de consumo, facilitando a criação de campanhas direcionadas.

## II. FUNDAMENTAÇÃO TEÓRICA

### A. Aprendizado Não Supervisionado
O aprendizado de máquina divide-se principalmente em supervisionado e não supervisionado. Enquanto o primeiro mapeia entradas para saídas conhecidas, o segundo busca encontrar padrões em dados sem respostas corretas prévias. A clusterização é a tarefa principal deste paradigma, visando agrupar objetos de modo que aqueles no mesmo grupo (cluster) sejam mais similares entre si do que com aqueles em outros grupos [3].

### B. Algoritmo K-Means
Proposto inicialmente por MacQueen em 1967, o K-Means é um algoritmo iterativo que particiona um conjunto de dados em $K$ clusters distintos. O objetivo é minimizar a variância intra-cluster, formalizada pela função de inércia (soma dos quadrados das distâncias entre os pontos e o centroide do seu cluster) [4]:

$$ J = \sum_{j=1}^{k} \sum_{i=1}^{n} || x_i^{(j)} - c_j ||^2 $$

Onde:
*   $K$ é o número de clusters.
*   $x_i^{(j)}$ é um ponto de dados pertencente ao cluster $j$.
*   $c_j$ é o centroide do cluster $j$.

O algoritmo opera em dois passos principais repetidos até a convergência:
1.  **Atribuição:** Cada ponto é atribuído ao centroide mais próximo (geralmente usando distância Euclidiana).
2.  **Atualização:** Os centroides são recalculados como a média dos pontos atribuídos a eles.

### C. Segmentação de Clientes (RFM e Variações)
Na literatura de marketing, a segmentação frequentemente utiliza variáveis como Recência, Frequência e Valor Monetário (RFM). Neste protótipo, adapta-se o conceito utilizando variáveis demográficas e comportamentais diretas: Idade, Renda Anual e Score de Gastos, que servem como *features* para o vetor de entrada do algoritmo [5].

## III. METODOLOGIA

Este estudo adota uma abordagem quantitativa e experimental, estruturada como um *Proof of Concept* (PoC). O fluxo de trabalho segue o padrão CRISP-DM (Cross-Industry Standard Process for Data Mining), adaptado para um ambiente de desenvolvimento ágil.

1.  **Entendimento do Negócio:** Definir a necessidade de agrupar clientes para campanhas de marketing.
2.  **Entendimento dos Dados:** Simulação de uma base de dados contendo características numéricas de clientes.
3.  **Preparação dos Dados:** Organização em estrutura tabular (DataFrame) e seleção de *features*.
4.  **Modelagem:** Aplicação do algoritmo K-Means.
5.  **Avaliação:** Análise dos centroides e dos rótulos atribuídos.

As ferramentas utilizadas incluem a linguagem Python, devido à sua robusta ekosistema para ciência de dados, especificamente as bibliotecas `NumPy` para computação numérica, `Pandas` para manipulação de dados e `Scikit-Learn` para implementação do modelo de machine learning [6].

## IV. DESENVOLVIMENTO DO PROTÓTIPO

A implementação do protótipo foi realizada em um ambiente *notebook* (Google Colab), permitindo a execução interativa do código. Abaixo, detalha-se cada etapa técnica do script `ml_k_means.py`. O protótipo está disponível neste [link](https://github.com/topicos-computacao-aplicada/2026-1/blob/main/notebooks/ML_K_Means.ipynb).

### A. Importação de Bibliotecas
O início do script importa as dependências necessárias. O `numpy` é utilizado para operações de álgebra linear subjacentes ao cálculo de distâncias. O `pandas` gerencia a estrutura de dados tabular. O módulo `KMeans` é importado do `sklearn.cluster`, que contém a implementação otimizada do algoritmo.

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
```

### B. Criação e Preparação do Dataset
Para fins de prototipagem, foi construído um dicionário fictício contendo 10 observações de clientes. As variáveis escolhidas foram:
*   **Idade:** Variável demográfica.
*   **Renda:** Variável econômica (em mil dólares).
*   **Gastos:** Variável comportamental (score 1-100).

```python
data = {
    "idade": [23, 45, 31, 52, 40, 25, 36, 60, 48, 30],
    "renda": [15, 60, 35, 80, 50, 20, 40, 90, 70, 30],
    "gastos": [39, 81, 50, 75, 60, 20, 55, 85, 65, 45]
}
df = pd.DataFrame(data)
```
A conversão para `DataFrame` facilita a visualização e a posterior anexação dos resultados. As *features* são extraídas para um array NumPy `X`, formato exigido pelo Scikit-Learn.

```python
X = df[["idade", "renda", "gastos"]].values
```

### C. Configuração e Treinamento do Modelo
O hiperparâmetro crítico no K-Means é o número de clusters ($K$). Neste protótipo, definiu-se $K=3$, baseando-se na hipótese de negócio de existir três perfis principais (ex.: Baixo, Médio, Alto). O parâmetro `random_state=42` foi configurado para garantir reprodutibilidade dos experimentos, fixando a seed de inicialização dos centroides.

```python
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
```
O método `.fit(X)` executa o loop de otimização do algoritmo, ajustando as posições dos centroides até minimizar a inércia.

### D. Atribuição de Rótulos e Análise de Centroides
Após o treinamento, o modelo prediz o cluster de cada observação. Esses rótulos são armazenados em uma nova coluna no DataFrame original para análise contextual.

```python
df["cluster"] = kmeans.labels_
```
Para interpretar o significado de cada cluster, analisam-se os centroides. No script, há um carregamento externo de um arquivo `centroids.csv`, que representa as coordenadas médias de cada grupo. Isso permite inferir, por exemplo, se o Cluster 0 possui média de renda superior ao Cluster 1.

```python
df_centroids = pd.read_csv('centroids.csv')
```

## V. RESULTADOS

A execução do protótipo gerou um DataFrame enriched com a coluna `cluster`. Observou-se que o algoritmo conseguiu separar os dados em três grupos distintos baseados na proximidade euclidiana no espaço tridimensional (idade, renda, gastos).

A análise dos centroides (carregados via CSV) permitiu a seguinte interpretação preliminar dos perfis:
1.  **Cluster 0 (Jovens/Econômicos):** Tendência a menores valores de renda e gastos.
2.  **Cluster 1 (Perfil Médio):** Valores intermediários nas três dimensões.
3.  **Cluster 2 (Premium):** Altos valores de renda e gastos, frequentemente associados a idades mais maduras.

A tabela de resultados (DataFrame `df`) demonstrou que clientes com características similares (ex.: idade 45, renda 60, gastos 81) foram agrupados consistentemente, validando a capacidade do algoritmo de encontrar similaridades multidimensionais.

## VI. DISCUSSÃO

Os resultados obtidos corroboram a literatura que afirma a eficácia do K-Means para segmentação inicial de bases de clientes [7]. A capacidade de identificar grupos sem intervenção manual reduz o viés humano na categorização.

No entanto, limitações do protótipo devem ser consideradas para aplicações em produção:
1.  **Escala dos Dados:** O K-Means é sensível à escala das variáveis. No código apresentado, não houve normalização (StandardScaler). Em um cenário real, onde "Renda" varia em milhares e "Idade" em dezenas, a Renda dominaria o cálculo de distância. Futuras iterações devem incluir pré-processamento de escala [8].
2.  **Tamanho da Amostra:** O dataset de 10 linhas é insuficiente para generalização estatística. Serve apenas para validação de lógica de código.
3.  **Escolha do K:** O valor $K=3$ foi arbitrário. Em produção, recomenda-se o uso do *Elbow Method* ou *Silhouette Score* para determinar o número ótimo de clusters [9].

Apesar das limitações, o protótipo estabelece a base técnica para uma pipeline de CRM analítico, onde os rótulos gerados podem ser integrados a sistemas de e-mail marketing para disparo de ofertas personalizadas.

## VII. CONCLUSÃO

Este trabalho apresentou a implementação de um algoritmo de clusterização K-Means para segmentação de clientes. O protótipo demonstrou ser possível agrupar consumidores com base em atributos demográficos e financeiros, gerando insights acionáveis para estratégias de marketing.

Conclui-se que o aprendizado não supervisionado é uma ferramenta poderosa para exploração de dados quando rótulos não estão disponíveis. Como trabalhos futuros, sugere-se a aplicação do modelo em bases de dados reais de grande volume, a implementação de técnicas de redução de dimensionalidade (como PCA) e a integração do modelo em APIs para predição em tempo real.

## REFERÊNCIAS

[1] P. Kotler e H. Kartajaya, *Marketing 4.0: Moving from Traditional to Digital*. Hoboken: Wiley, 2017.

[2] T. Hastie, R. Tibshirani e J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York: Springer, 2009.

[3] A. K. Jain, "Data clustering: 50 years beyond K-means," *Pattern Recognition Letters*, vol. 31, no. 8, pp. 651-666, 2010.

[4] J. MacQueen, "Some methods for classification and analysis of multivariate observations," in *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1967, pp. 281-297.

[5] D. R. Bell, T. Ho e C. Tang, "Determining where to shop: Fixed and variable costs of shopping," *Journal of Marketing Research*, vol. 35, no. 3, pp. 352-369, 1998.

[6] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[7] M. J. A. Berry e G. S. Linoff, *Data Mining Techniques: For Marketing, Sales, and Customer Relationship Management*. Indianapolis: Wiley, 2004.

[8] I. Goodfellow, Y. Bengio e A. Courville, *Deep Learning*. Cambridge: MIT Press, 2016. (Referência contextual sobre pré-processamento).

[9] R. L. Thorndike, "Who belongs in the family?" *Psychometrika*, vol. 18, no. 4, pp. 267-276, 1953.
