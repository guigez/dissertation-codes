# Pôster CBCTQ 2026 — texto em português

> Tradução do texto do pôster `poster_cbctq2026.pdf`, que está em inglês, para leitura. A ordem é a do pôster: cabeçalho, coluna da esquerda, coluna da direita e rodapé. Os nomes em inglês aparecem entre parênteses para facilitar localizar cada trecho no pôster. Os números seguem o padrão brasileiro (vírgula decimal); no pôster, estão com ponto.

---

**Congresso Brasileiro de Ciências e Tecnologias Quânticas** *(Brazilian Congress of Quantum Sciences and Technologies)*
Niterói–RJ, Brasil · 21 a 25 de setembro de 2026

## Uma ablação NISQ controlada de CNNs híbridas quântico-clássicas para classificação de imagens biomédicas

*(A Controlled NISQ Ablation of Hybrid Quantum–Classical CNNs for Biomedical Image Classification)*

**Guilherme H. Rodrigues**¹ \*, **Ulisses Dias**¹
¹ Faculdade de Tecnologia (FT), Universidade Estadual de Campinas (UNICAMP), Limeira–SP, Brasil
\* Autor apresentador · g290200@dac.unicamp.br

---

# Coluna da esquerda

## Resumo *(Abstract)*

O módulo quântico de uma CNN híbrida quântico-clássica (H-QCNN) ajuda a classificar imagens biomédicas no regime NISQ? Trocando *apenas* esse módulo por um bloco clássico pareado (31.818 frames termográficos de 11 ratos), não encontramos **nenhuma vantagem agregada**. Além disso, o módulo quântico tem **recall da classe minoritária significativamente menor** (0,518 vs 0,777; p = 0,008).

**Palavras-chave *(Keywords)*:** aprendizado de máquina quântico; modelos híbridos quântico-clássicos; circuitos quânticos variacionais; NISQ; ablação controlada; classificação de imagens biomédicas; termografia infravermelha.

## Introdução *(Introduction)*

Modelos híbridos quântico-clássicos servem para investigar possíveis vantagens em hardware NISQ [1, 2]. Neles, circuitos parametrizados [3] funcionam como módulos não lineares compactos. Para mostrar que o módulo quântico *em si* ajuda, é preciso um comparador clássico pareado, sob um protocolo idêntico. Ablações pareadas [4–6] costumam relatar vantagens, mas quase sempre usam divisões dos dados dependentes do sujeito ou não especificadas. Quando se usou validação independente do sujeito [7], a vantagem se mostrou sensível ao vazamento de informação entre pacientes.

## Objetivos *(Objectives)*

- Sob um protocolo mais rigoroso, trocar *apenas* o módulo por um **bloco clássico diferenciável com a mesma função**.
- Validar de forma **independente do sujeito** (*Leave-One-Rat-Out*, LORO: um rato fica de fora a cada rodada), com testes pareados e foco em uma **classe rara** (~4%) [8].
- Localizar qualquer efeito com análises de matriz de confusão e de atribuição (SHAP).

## Metodologia *(Methodology)*

**Dados e desenho *(Data and design)*.** Foram usados 31.818 frames termográficos de 11 ratos Wistar, distribuídos em quatro domínios de intensidade de exercício [8] (Fig. 1). As escolhas de projeto (AngleEmbedding, BasicEntanglerLayers e topologia C–Q–C) seguem os padrões mais comuns da nossa revisão PRISMA 2020 [9] de 81 estudos primários revisados por pares.

**Figura 1** — Um frame representativo de cada domínio; as barras mostram a participação de cada domínio nos 31.818 frames.

| No pôster | Domínio | Participação |
|---|---|---|
| Warm-up | Baixo (aquecimento) | 38,2% |
| Moderate | Moderado | 35,9% |
| Heavy | Intenso | 21,5% |
| Severe | Severo | 4,4% |

**Arquitetura *(Architecture)*.** Os dois modelos compartilham o mesmo pipeline com ResNet-50 [10] (Fig. 2). Um comparador não diferenciável (por exemplo, uma *random forest*) impediria o ajuste fino do backbone.

**Figura 2** — Ablação pareada: em cima, o pipeline compartilhado; embaixo, o circuito quântico e o bloco clássico que o substitui.

*Conteúdo da figura:*

- **Pipeline (em cima):** frame térmico → **ResNet-50** (pré-treinada na ImageNet, com ajuste fino da layer4; η = 10⁻⁵) → vetor de 2048 dimensões → **camada linear** 2048→8 → z ∈ ℝ⁸ → **módulo intercambiável** (η = 10⁻⁴) → saída em (−1, 1)⁸ → **camada linear** 8→4 domínios (η = 10⁻³).
  - O módulo intercambiável é uma de duas opções:
    - **quântico:** PQC de 8 qubits, 5 camadas, 40 parâmetros;
    - **clássico:** Linear + tanh, 8→8, 72 parâmetros.
- **Circuito (embaixo, à esquerda):** 8 qubits em |0⟩ → codificação U_enc(z) → bloco R_X(θ_l) seguido de uma escada de CNOTs em anel, repetido N_l = 5 vezes → medição → ⟨Z_i⟩.
- **Quadro azul, módulo quântico *(Quantum module)*:**
  - estado: |ψ(θ, z)⟩ = U_ent^(N_l)(θ) · U_enc(z) · |0⟩^(⊗N_q);
  - codificação R_X(π·σ(z_i));
  - N_l = 5 camadas de R_X(θ_{l,i}) + anel de CNOTs;
  - leitura: ⟨Z_i⟩;
  - **40 parâmetros**.
- **Quadro laranja, módulo clássico (controle) *(Classical module, control)*:**
  - h = tanh(Wz + b), com W ∈ ℝ^(8×8) e b ∈ ℝ⁸;
  - **72 parâmetros**;
  - saída na mesma faixa (−1, 1) de ⟨Z_i⟩.

**Protocolo pareado *(Paired protocol)*.** Os dois modelos são idênticos, exceto pelo módulo:

| No pôster | Tradução |
|---|---|
| Training | Entropia cruzada ponderada por classe, com suavização de rótulos (*label smoothing*); otimizador Adam com uma η para cada parte do modelo (Fig. 2) |
| Held-out | Os quatro ratos que têm frames da classe severa (1, 9, 10 e 11), cada um com duas *seeds*: 8 configurações pareadas (16 execuções) |
| Validation | LORO (dez *folds*), aumento de dados em tempo de teste (TTA, T = 5) e *ensemble* |
| Statistics | Teste de Wilcoxon pareado (postos sinalizados), sobre as configurações e sobre os 80 *folds* |
| Backend | PennyLane (`default.qubit`) + PyTorch, com simulação por vetor de estado |

---

# Coluna da direita

## Resultados *(Results)*

**Tabela 1** — Média sobre quatro ratos de teste × duas *seeds* (n = 8); p do teste de Wilcoxon pareado, quântico vs clássico.

| Modelo | Acur. balanceada | F1 | Precisão | Recall | Recall severo |
|---|---|---|---|---|---|
| Quântico (PQC, 40 parâmetros) | 0,529 | **0,438** | **0,541** | 0,529 | 0,518 |
| Clássico (pareado, 72 parâmetros) | **0,532** | 0,411 | 0,492 | **0,532** | **0,777** |
| p pareado (Wilcoxon) | 0,74 | 0,38 | — | 0,74 | **0,008** |

**Métricas gerais *(Overall metrics)*.** Os dois módulos têm o mesmo desempenho. O módulo quântico tem acurácia balanceada maior em apenas 2 das 8 configurações, sua vantagem em F1-macro não é significativa (p = 0,20 na comparação por *fold*) e os desvios-padrão altos vêm das diferenças entre os ratos.

**Classe minoritária *(Minority class)*.** A única diferença significativa **favorece o módulo clássico**, que é melhor em **todas as oito** configurações (recall da classe severa: 0,777 vs 0,518; p = 0,008). O que muda é o ponto de operação, e não a separabilidade das classes (Fig. 3).

**Figura 3** — Matrizes de confusão somando todos os ratos de teste. As células mostram contagens; a cor mostra a proporção dentro de cada linha. Os dois módulos quase não acertam a classe *intensa* (recall < 0,12). O módulo clássico prevê a classe *severa* com mais frequência (coluna destacada). Por isso, a precisão da classe severa é 0,17 no módulo clássico, contra 0,21 no quântico.

*Conteúdo da figura: linhas = domínio verdadeiro; colunas = domínio previsto.*

**Módulo quântico** (PQC, 40 parâmetros)

| Verdadeiro \ Previsto | Baixo | Moderado | Intenso | Severo |
|---|---|---|---|---|
| Baixo | 7.544 | 1.127 | 0 | 211 |
| Moderado | 1.233 | 5.574 | 26 | 2.071 |
| Intenso | 165 | 3.742 | 1.017 | 3.892 |
| Severo | 4 | 329 | 800 | 1.649 |

**Módulo clássico** (Linear(8→8) + tanh, 72 parâmetros)

| Verdadeiro \ Previsto | Baixo | Moderado | Intenso | Severo |
|---|---|---|---|---|
| Baixo | 7.363 | 474 | 0 | 1.045 |
| Moderado | 1.026 | 3.663 | 6 | 4.209 |
| Intenso | 140 | 1.308 | 696 | 6.672 |
| Severo | 5 | 116 | 245 | 2.416 |

**Dependência do sujeito e atribuição *(Subject dependence and attribution)*.** A única vitória do módulo quântico em acurácia balanceada acontece no rato 1 (o antigo sujeito de teste fixo). Nos outros três ratos o resultado se inverte, e em um deles o recall da classe severa é zero. O bloco clássico pareado reproduz, e até supera, a recuperação da classe minoritária que antes era atribuída ao pipeline híbrido. Mapas SHAP contrastivos [11] colocam o efeito no **backbone com ajuste fino**, e não no circuito quântico.

## Conclusões *(Conclusions)*

> Neste regime NISQ, o PQC de 40 parâmetros **não traz vantagem mensurável** sobre um bloco clássico diferenciável pareado e é **significativamente pior na classe minoritária**.

- Os ganhos aparentes vêm do ajuste fino do backbone clássico. Somado a [7], isso indica que as vantagens relatadas para H-QCNNs dependem do protocolo de validação.
- Contribuição: evidência independente do sujeito, que leva em conta o desbalanceamento das classes, e uma análise de atribuição que mostra que o efeito vem da parte clássica.
- **Próximo passo *(Next)*:** rodar em hardware NISQ real via Amazon Braket (QPUs supercondutoras e de íons aprisionados), com extrapolação para ruído zero (ZNE).

## Referências *(References)*

*As referências ficam no original.*

1. J. Preskill, "Quantum computing in the NISQ era and beyond," *Quantum*, vol. 2, p. 79, 2018.
2. M. Cerezo *et al.*, "Variational quantum algorithms," *Nature Reviews Physics*, vol. 3, pp. 625–644, 2021.
3. K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, "Quantum circuit learning," *Physical Review A*, vol. 98, p. 032309, 2018.
4. V. Kulkarni, S. Pawale, and A. Kharat, "A classical–quantum convolutional neural network for detecting pneumonia from chest radiographs," *Neural Computing and Applications*, 2023.
5. S. Rawas and D. AlSaeed, "Quantum enhanced machine learning for medical image analysis: a hybrid approach," *Applied Computing and Informatics*, 2026.
6. C. Long *et al.*, "Hybrid quantum-classical-quantum convolutional neural networks," *Scientific Reports*, 2025.
7. D. Guha *et al.*, "ResQ: A hybrid classical-quantum model for efficient breast cancer image classification," *Applied Soft Computing*, 2025.
8. P. H. B. Silva, *Classificação de domínios de intensidade de exercício em ratos por meio de imagens termográficas e redes neurais convolucionais*. M.S. thesis, Universidade Estadual de Campinas, Limeira, 2025.
9. M. J. Page *et al.*, "The PRISMA 2020 statement: an updated guideline for reporting systematic reviews," *BMJ*, vol. 372, p. n71, 2021.
10. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
11. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

---

# Rodapé

## Agradecimentos *(Acknowledgements)*

Os autores agradecem ao Programa de Pós-Graduação em Tecnologia (PPGT) da Faculdade de Tecnologia (FT) da UNICAMP. O presente trabalho foi realizado com apoio da Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES) – Código de Financiamento 001. Este trabalho também contou com apoio do FAEPEX/UNICAMP (Fundo de Apoio ao Ensino, Pesquisa e Extensão), processo 3207/26. U. Dias recebe Bolsa de Produtividade em Pesquisa do Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), processo 303911/2026-3.

---

## Glossário rápido (inglês do pôster → português)

| Inglês | Português |
|---|---|
| matched classical block | bloco clássico pareado |
| quantum / classical module | módulo quântico / clássico (as duas versões comparadas) |
| balanced accuracy | acurácia balanceada |
| recall | revocação (sensibilidade) |
| operating point | ponto de operação |
| held-out rats | ratos de teste (deixados de fora do treino) |
| seed | semente aleatória |
| fine-tuning / backbone | ajuste fino / rede-base extratora de características |
| state-vector simulation | simulação por vetor de estado |
| zero-noise extrapolation | extrapolação para ruído zero |
| trapped-ion QPU | processador quântico de íons aprisionados |
