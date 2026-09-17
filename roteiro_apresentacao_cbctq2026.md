# Roteiro da apresentação — pôster CBCTQ 2026

> **Como usar.** O gancho (~40 s) é para quem só passa em frente ao pôster. O roteiro principal (~3 min 30 s, em ritmo calmo) é para quem para e escuta. Os dois focam na metodologia. Os números ficam fora da fala e estão no fim, para você usar só se alguém perguntar. O que está **[entre colchetes]** é indicação de onde apontar no pôster — não faz parte da fala.

---

# Português

## Gancho (~40 s)

Esse trabalho testa uma pergunta simples: quando uma rede híbrida quântico-clássica funciona bem, é o circuito quântico que está fazendo a diferença?

Para responder, montamos duas redes idênticas. **[aponte a Fig. 2, o pipeline]** A única coisa que muda entre elas é um módulo de oito dimensões no meio do pipeline: em uma, é um circuito quântico variacional; na outra, um bloco clássico com o mesmo papel. Todo o resto — backbone, dados, otimizador, divisão dos dados, semente — é igual.

É uma ablação pareada: se aparecer diferença de desempenho, ela só pode vir do módulo.

## Roteiro principal (~3 min 30 s)

**1. O problema (~45 s)** · *se estiver sem tempo, é este o trecho a cortar*

A literatura de aprendizado de máquina quântico aplicada a imagens médicas tem bastante artigo relatando ganho com modelos híbridos. O problema está no desenho do experimento. Na maioria das vezes, o híbrido é comparado com um modelo clássico diferente dele, ou a divisão dos dados deixa o mesmo sujeito no treino e no teste. Quando isso acontece, não dá para saber de onde vem o ganho: do circuito quântico, do backbone clássico que foi ajustado junto, ou do vazamento de dados. Entre os poucos trabalhos que fazem uma comparação realmente pareada, quase todos usam divisão dependente do sujeito, ou nem descrevem qual divisão usaram.

**2. O desenho do experimento (~1 min 35 s)** · *o miolo da fala*

Nosso desenho ataca isso com três decisões. **[aponte a Fig. 2]**

A primeira é o comparador. O módulo quântico é um circuito de oito qubits: codificação por ângulo, e depois camadas de rotação seguidas de um anel de CNOTs. O comparador clássico é uma camada linear de oito para oito com tanh. Ele ocupa exatamente o mesmo lugar no pipeline, recebe o mesmo vetor de entrada e devolve a saída na mesma faixa, de menos um a um. E, principalmente, ele é diferenciável. Isso importa: se usássemos uma random forest, como parte da literatura faz, o gradiente não voltaria para o backbone, e o experimento deixaria de ser pareado.

A segunda é a validação. **[aponte o quadro do protocolo]** É Leave-One-Rat-Out: o rato usado no teste não aparece no treino, em nenhum frame. Em vídeo termográfico isso é essencial, porque frames vizinhos do mesmo animal são quase idênticos. Se você divide por frame, o modelo já viu o animal de teste, e o desempenho sobe sem que o modelo tenha aprendido nada de novo.

A terceira é o pareamento. Cada configuração roda os dois módulos com a mesma semente, os mesmos folds, a mesma augmentação em tempo de teste e o mesmo ensemble. Como os pares são idênticos em tudo, a comparação pode ser feita com um teste pareado, em vez de comparar duas médias soltas.

**3. O resultado (~30 s)**

E o resultado é negativo. **[aponte a Tabela 1 e depois a Fig. 3]** Nas métricas gerais, os dois módulos empatam. Na classe rara, a severa, quem vai melhor é o bloco clássico — e vai melhor em todas as configurações. As matrizes de confusão mostram que isso não é o clássico separando melhor as classes: é ele prevendo a classe severa com mais frequência. É um ponto de operação diferente, não uma capacidade diferente.

**4. Interpretação e próximo passo (~40 s)**

A nossa leitura é que o ganho que aparecia antes vinha do ajuste fino do backbone clássico, e não do circuito. Os mapas SHAP contrastivos apontam para o mesmo lugar. Isso não quer dizer que aprendizado de máquina quântico não sirva para imagem médica. Quer dizer que, sem um comparador pareado e sem validação independente do sujeito, não dá para afirmar que a vantagem veio do circuito. O próximo passo é repetir o experimento em hardware quântico real, para ver se o resultado nulo se mantém fora do simulador.

**Fecho:** Se quiser, eu abro qualquer parte com mais calma: o circuito, o protocolo ou as matrizes de confusão.

---

# English

## Hook (~40 s)

This work asks a simple question: when a hybrid quantum–classical network performs well, is the quantum circuit what makes the difference?

To answer it, we built two identical networks. **[point to Fig. 2, the pipeline]** The only thing that changes between them is an eight-dimensional module in the middle of the pipeline: in one it is a variational quantum circuit, in the other a classical block with the same role. Everything else — backbone, data, optimizer, data splits, random seed — is the same.

It is a paired ablation: if a performance difference shows up, it can only come from the module.

## Main script (~3 min 20 s)

**1. The problem (~40 s)** · *the part to cut if you are short on time*

In quantum machine learning for medical imaging, many papers report gains from hybrid models. The problem is the experimental design. Most of the time the hybrid model is compared against a different classical model, or the data split leaves the same subject in training and in test. When that happens, you cannot tell where the gain comes from: the quantum circuit, the classical backbone that was fine-tuned along with it, or data leakage. Among the few papers that do run a properly matched comparison, almost all use subject-dependent splits, or never say which split they used.

**2. The experimental design (~1 min 30 s)** · *the core of the talk*

Our design addresses this with three decisions. **[point to Fig. 2]**

The first is the comparator. The quantum module is an eight-qubit circuit: angle encoding, then rotation layers followed by a ring of CNOTs. The classical comparator is a linear layer, eight to eight, with tanh. It sits in exactly the same place in the pipeline, takes the same input vector, and returns its output in the same range, minus one to one. And, most importantly, it is differentiable. That matters: if we used a random forest, as part of the literature does, the gradient would not flow back to the backbone, and the experiment would stop being matched.

The second is the validation. **[point to the protocol box]** We use Leave-One-Rat-Out: the rat used for testing never appears in training, not in a single frame. With thermographic video this is essential, because neighbouring frames of the same animal are nearly identical. If you split by frame, the model has already seen the test animal, and performance goes up without the model having learned anything new.

The third is the pairing. Each configuration runs both modules with the same seed, the same folds, the same test-time augmentation and the same ensembling. Since the pairs are identical in everything else, we can compare them with a paired test, instead of comparing two separate averages.

**3. The result (~30 s)**

And the result is negative. **[point to Table 1, then Fig. 3]** On the overall metrics, the two modules score the same. On the rare class, the severe one, the classical block does better — and it does better in every configuration. The confusion matrices show this is not the classical block separating the classes better: it is predicting the severe class more often. That is a different operating point, not a different capability.

**4. Interpretation and next step (~35 s)**

Our reading is that the gain reported earlier came from fine-tuning the classical backbone, not from the circuit. The contrastive SHAP maps point to the same place. This does not mean quantum machine learning is useless for medical imaging. It means that, without a matched comparator and without subject-independent validation, you cannot claim the advantage came from the circuit. The next step is to repeat the experiment on real quantum hardware, to see whether the null result holds outside the simulator.

**Closing:** I'm happy to go deeper into any part: the circuit, the protocol, or the confusion matrices.

---

# Se perguntarem números

*Fora da fala. Use só quando alguém pedir.*

| Pergunta provável | Resposta curta |
|---|---|
| Quantos dados? | 31.818 frames de 11 ratos Wistar, em quatro domínios de intensidade; a classe severa é ~4% do total. |
| Quantas execuções? | Quatro ratos de teste (os que têm classe severa) × duas sementes = 8 pares, 16 execuções no total. |
| E as métricas gerais? | Acurácia balanceada 0,529 no quântico e 0,532 no clássico; F1-macro 0,438 e 0,411. Nenhuma diferença significativa. |
| E a classe severa? | Recall 0,518 no quântico e 0,777 no clássico, com p = 0,008; o clássico é melhor nas 8 configurações. |
| Tamanho dos módulos? | 40 parâmetros no circuito e 72 no bloco clássico. |
| Rodou em hardware real? | Ainda não: tudo por simulação de vetor de estado (PennyLane + PyTorch). É o próximo passo, via Amazon Braket. |

**Se alguém apertar no p = 0,008:** vale reconhecer a limitação na hora. Os 8 pares vêm de 4 ratos × 2 sementes, então não são independentes entre si — o p formal é otimista. O que sustenta a conclusão é a consistência: o bloco clássico ganha em todas as configurações, em todos os ratos de teste.
Em inglês: *"Fair point — the eight pairs come from four rats and two seeds, so they are not independent, and the p-value is optimistic. What holds the conclusion up is the consistency: the classical block wins in every configuration."*

**Se perguntarem por que a Fig. 3 e a Tabela 1 não batem:** a figura soma todos os frames de teste, e a tabela é a média por configuração. São duas formas de agregar, então os valores de recall não são os mesmos.
Em inglês: *"The figure pools all test frames; the table averages over configurations. Two different aggregations, so the recall values differ."*

# Frases de apoio em inglês

| Situação | Frase |
|---|---|
| Alguém chega no meio | *"Let me give you the short version first."* |
| Não entendeu a pergunta | *"Sorry, could you say that again?"* |
| Perguntaram algo que você não sabe | *"I don't have that number here — I can check it and send it to you."* |
| Quer encerrar com elegância | *"Thanks for stopping by — the poster stays up, feel free to come back."* |
