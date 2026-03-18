# Cat x Dog recognition

Projeto de criação de um modelo de reconhecimento de imagens de cães e gatos. 

Mais detalhes técnicos sobre o projeto disponível em [Projetos.md](Projeto.md)

Logo abaixo seguem os passos essencias para a execução do projeto. 

## 1. Preparação do ambiente virtual 

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

## 2. Instação das dependências

```bash
pip install -r requirements
```

## 3. Execução do processo de integração e criação do modelo 
```bash
python3 main.py
```

Arquivos gerados após a execução: 
- [Loss_Accuracy.png](resultados/Loss_Accuracy.png)
- [reconhece_dog_cat.png](resultados/reconhece_doc_cat.png)
- cat_dog_classifier.pth

O log do resultado da 1a. execução do main.py pode ser visualizado em [Resultados da Execução](resultado_execucao.md)

Detalhes do arquivo classificador (cat_dog_classifier.pth) de imagens de cães e gatos disponível em [cat_dog_classifier](classificador_gerado.md)

## Mais detalhes

Mais detalhes no [paper técnico](https://github.com/topicos-computacao-aplicada/2026-1/blob/main/cat_dog/paper_tecnico.md) do protótipo.
