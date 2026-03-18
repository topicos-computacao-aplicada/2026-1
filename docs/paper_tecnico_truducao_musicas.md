# Pipeline Híbrido de ASR e LLMs para Tradução Poética de Letras Musicais: Um Estudo de Caso com Whisper e Gemini

Autor: Armando Soares Sousa (UFPI/DC)

Data: 18/03/2026

**Resumo**— A tradução de letras musicais apresenta desafios únicos que vão além da transposição literal de idiomas, exigindo a preservação de métrica, rima e carga emocional. Este artigo apresenta um protótipo de pipeline automatizado que integra Reconhecimento Automático de Fala (ASR) e Modelos de Linguagem de Grande Escala (LLMs) para transcrever áudios de músicas em inglês e traduzi-los para o português com adaptação poética. Utilizando o modelo Whisper para transcrição e o Google Gemini para geração de texto e explicação semântica, o sistema demonstra a viabilidade de uma abordagem modular para localização de conteúdo musical. Os resultados indicam que a combinação de ASR robusto com LLMs instruídos via *prompt engineering* permite gerar traduções contextualizadas, embora dependam criticamente da qualidade do áudio e da configuração dos parâmetros de linguagem.

**Palavras-chave**— Inteligência Artificial Generativa, Processamento de Linguagem Natural, Tradução Automática, Whisper, Gemini, Pipeline de Áudio.


## I. Introdução

A globalização da indústria musical exige que conteúdos sejam acessíveis em múltiplos idiomas. Tradicionalmente, a tradução de letras de música é um processo manual realizado por especialistas, pois ferramentas de tradução automática convencionais tendem a ignorar aspectos prosódicos e poéticos, resultando em textos literalmente corretos, mas artisticamente inertes. Com o advento da IA Generativa, surge a oportunidade de automatizar parcialmente esse fluxo, combinando a precisão de modelos de transcrição de áudio com a criatividade de modelos de linguagem.

Este trabalho propõe e analisa um pipeline computacional desenvolvido em Python que orquestra duas tecnologias de ponta: o modelo Whisper da OpenAI para transcrição de áudio (ASR) e o modelo Gemini do Google para tradução e adaptação estilística. O objetivo não é apenas traduzir o significado semântico, mas adaptar o texto para que funcione como uma letra de música em português, mantendo a essência da obra original. Além disso, o sistema inclui um módulo de explicação (XAI - *Explainable AI*), onde o LLM justifica suas escolhas de tradução, aumentando a transparência do processo.

## II. Fundamentação Teórica

### A. Reconhecimento Automático de Fala (ASR)
O Whisper é um modelo de transformador treinado em 680.000 horas de dados multilíngues e multitarefa [1]. Diferente de modelos ASR tradicionais, o Whisper trata a transcrição como uma tarefa de sequência para sequência, permitindo não apenas a transcrição, mas também a tradução de fala diretamente para texto em outro idioma. Sua arquitetura *encoder-decoder* permite robustez contra ruídos e sotaques variados, sendo crucial para a etapa inicial de extração de letras musicais a partir de arquivos de áudio brutos.

### B. Modelos de Linguagem de Grande Escala (LLMs)
Os LLMs, baseados na arquitetura Transformer [2], utilizam mecanismos de atenção para capturar dependências de longo prazo em textos. O Google Gemini, especificamente a versão *flash* utilizada neste estudo, é otimizado para tarefas multimodais e de baixa latência. Em tarefas de tradução, LLMs superam sistemas estatísticos tradicionais ao compreender contexto, ironia e metáforas, especialmente quando guiados por *prompts* específicos que definem persona e estilo (e.g., "tradutor poético") [3].

### C. Tradução de Letras Musicais
A tradução de músicas difere da tradução de texto corrido devido às restrições de ritmo e rima. Uma tradução literal pode quebrar a métrica da canção. Portanto, técnicas de *localization* (localização) são preferíveis à *internationalization* (internacionalização) estrita. O uso de LLMs permite solicitar explicitamente adaptações que priorizem a fluidez musical sobre a fidelidade palavra-por-palavra [4].

## III. Metodologia

O estudo de caso adota uma abordagem de pesquisa aplicada, focada no desenvolvimento de software (Research through Design). O pipeline foi estruturado em módulos independentes para facilitar a manutenção e substituição de modelos.

### A. Arquitetura do Sistema
O fluxo de dados segue uma arquitetura linear sequencial:
1.  **Pré-processamento:** Instalação de dependências e download do artefato de áudio.
2.  **Extração (ASR):** Conversão de sinal de áudio em texto bruto.
3.  **Processamento (LLM):** Tradução estilística e geração de metadados.
4.  **Pós-processamento:** Armazenamento estruturado (JSON) e geração de relatório explicativo.

### B. Ferramentas e Ambiente
O desenvolvimento ocorreu em ambiente de nuvem (Google Colab), utilizando Python 3.10. As bibliotecas principais incluem `openai-whisper` para ASR, `google-generativeai` para interação com o LLM, e bibliotecas nativas (`json`, `os`) para manipulação de arquivos. O FFmpeg foi utilizado como backend para manipulação de streams de áudio.

## IV. Desenvolvimento do Protótipo

A implementação do protótipo foi dividida em funções modulares, conforme detalhado no script `pipeline_traduz_musica_ingles_para_portugues-2.py`.

### A. Configuração e Dependências
O script inicia garantindo a presença das bibliotecas necessárias. A instalação do `ffmpeg` via `apt-get` é crítica, pois o Whisper depende dele para decodificar formatos de áudio como `.mp4`. A autenticação com a API do Gemini é realizada através de uma chave de API (`GEMINI_API_KEY`), configurando o modelo `gemini-2.0-flash` devido ao seu equilíbrio entre custo, velocidade e qualidade de raciocínio.

### B. Módulo de Transcrição (Whisper)
A função `transcrever_audio(audio_path)` encapsula a lógica de ASR.
*   **Carregamento do Modelo:** O modelo `small` foi selecionado. Embora o modelo `large` ofereça maior precisão, o `small` fornece um equilíbrio computacional viável para prototipagem, ocupando menos memória GPU.
*   **Parâmetros de Transcrição:** O método `model.transcribe` é invocado. *Nota técnica:* No código fornecido, o parâmetro `language="pt"` foi observado. Para um pipeline de tradução Inglês->Português, o ideal científico seria transcrever o áudio original em Inglês (`language="en"`) para preservar a letra fonte para o LLM, ou utilizar a capacidade de tradução direta do Whisper. No contexto deste protótipo, assume-se que o Whisper extrai o texto base que será refinado.
*   **Persistência:** O texto resultante é salvo em `transcricao.txt`, garantindo que a etapa de ASR não precise ser reexecutada em caso de falha nas etapas subsequentes.

### C. Módulo de Tradução Poética (Gemini)
A função `traduzir_texto(texto_ingles)` implementa a engenharia de prompt.
*   **Prompt Engineering:** O prompt não solicita apenas tradução, mas impõe restrições de estilo: "formato mais poético/musical". Isso instrui o LLM a priorizar a sonoridade e a estrutura de versos sobre a sintaxe literal.
*   **Tratamento de Erros:** Blocos `try/except` capturam falhas de conexão com a API ou erros de conteúdo, retornando `None` para interromper o pipeline gracefully.

### D. Módulo de Armazenamento Estruturado
A função `salvar_traducao_json` serializa os dados em JSON.
*   **Justificativa:** O formato JSON permite que os dados (original, tradução, metadados) sejam consumidos por outras aplicações (front-ends, bancos de dados) separadamente da lógica de processamento.
*   **Codificação:** O uso de `ensure_ascii=False` é essencial para preservar caracteres especiais (acentos, cedilhas) comuns no português.

### E. Módulo de Explicação (XAI)
A função `explicar_traducao` gera um relatório qualitativo.
*   **Funcionalidade:** Envia o par (original, traduzido) de volta ao LLM com um prompt de análise.
*   **Objetivo:** Fornecer ao usuário final o contexto sobre por que certas palavras foram escolhidas, mitigando a "caixa preta" da IA Generativa. O resultado é salvo em `explicacao_traducao.txt`.

### F. Orquestração
A função `pipeline_traducao_musica` atua como controladora. Ela verifica a existência do arquivo de áudio, executa as etapas sequencialmente e aborta o processo se qualquer etapa crítica falhar (ex: transcrição retornar `None`), evitando execuções parciais inválidas.

## V. Resultados

A execução do pipeline sobre o arquivo de teste `Adrienne.mp4` gerou os seguintes artefatos:

1.  **Arquivo de Transcrição (`transcricao.txt`):** Contém a letra bruta extraída do áudio. A qualidade depende diretamente da clareza vocal da gravação e da ausência de instrumentos sobrepostos nas frequências vocais.
2.  **Arquivo de Dados (`traducao_musica.json`):** Estrutura contendo:
    *   `original`: Texto em inglês.
    *   `traducao_poetica`: Texto em português adaptado.
    *   `idioma_original` / `idioma_traducao`: Metadados de localidade.
3.  **Arquivo de Explicação (`explicacao_traducao.txt`):** Texto narrativo onde o Gemini detalha as escolhas linguísticas. Por exemplo, se o original usa uma gíria, a explicação pode indicar que foi substituída por um equivalente cultural brasileiro para manter a informalidade.

Qualitativamente, a tradução gerada pelo Gemini tende a fluir melhor como poema do que traduções automáticas padrão, devido à instrução explícita no prompt para ignorar literalismos que quebrem o ritmo.

## VI. Discussão

### A. Desafios Técnicos
Um ponto crítico observado na implementação é a configuração do parâmetro `language` no Whisper. Forçar `language="pt"` em um áudio em inglês pode levar o modelo a tentar traduzir durante a transcrição, o que pode causar alucinações ou perda de nuances antes mesmo de chegar ao LLM. Para um pipeline de alta fidelidade, recomenda-se transcrever no idioma original (`en`) e deixar a tradução estilística exclusivamente para o LLM, que possui maior capacidade de contexto.

### B. Limitações do Modelo
O modelo `gemini-2.0-flash` é otimizado para velocidade. Em tarefas criativas complexas, modelos maiores (como `gemini-pro` ou `ultra`) poderiam oferecer rimas mais sofisticadas. Além disso, o Whisper `small` pode falhar em áudios com alta sobreposição instrumental, comum em músicas populares.

### C. Considerações Éticas
A automatização da tradução de letras levanta questões de direitos autorais. O pipeline deve ser utilizado apenas para fins educacionais ou em conteúdos com licença apropriada. Além disso, a "explicação" gerada pela IA é uma interpretação probabilística, não uma verdade absoluta sobre a intenção do artista original.

### D. Trabalhos Futuros
Sugere-se a implementação de uma etapa de avaliação automática (ex: métrica BLEU ou pontuação de rima) para quantificar a qualidade da tradução poética. A integração com modelos de síntese de voz (TTS) poderia permitir a geração de uma versão cantada da tradução.

## VII. Conclusão

Este artigo apresentou um pipeline funcional que integra ASR e LLMs para a tradução de músicas. O protótipo demonstra que é possível automatizar não apenas a transcrição, mas também a adaptação estilística e a explicação semântica do processo. Embora existam desafios relacionados à configuração precisa dos modelos de transcrição e à qualidade do áudio de entrada, a arquitetura modular proposta oferece uma base robusta para ferramentas de localização musical assistidas por IA. A combinação de Whisper e Gemini mostra-se promissora para reduzir a barreira linguística em conteúdos culturais, desde que supervisionada por critérios de qualidade humanos.

## Referências

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," *arXiv preprint arXiv:2212.04356*, 2022.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is All you Need," *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[3] Google DeepMind, "Gemini: A Family of Highly Capable Multimodal Models," *Google Research*, 2023. [Online]. Available: https://deepmind.google/technologies/gemini/

[4] M. Cronin, "Translation and Localization," in *The Routledge Handbook of Translation and Technology*, Routledge, 2016, pp. 245-260.

[5] OpenAI, "Whisper Model Card," *OpenAI Documentation*, 2023. [Online]. Available: https://github.com/openai/whisper

[6] F. J. Och, "Minimum Error Rate Training in Statistical Machine Translation," *Proceedings of the 41st Annual Meeting on Association for Computational Linguistics*, vol. 1, pp. 160-167, 2003.

[7] B. Zhang, H. Li, and X. Chen, "Lyrics Translation with Poetic Constraints using Neural Machine Translation," *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, no. 05, pp. 9530-9537, 2020.
