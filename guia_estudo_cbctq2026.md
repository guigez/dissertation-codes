# Guia de estudo — pôster CBCTQ 2026

> **Para que serve.** Material de preparação para a sessão de pôster: o detalhe por trás de cada afirmação do pôster, o que o trabalho **não** afirma, as fragilidades conhecidas e, no fim, um banco de perguntas com respostas prontas.
>
> **Três níveis de certeza aparecem marcados ao longo do texto:**
> - **[pôster]** — está escrito no pôster ou no resumo estendido aceito. Pode afirmar sem medo.
> - **[projeto]** — está nas suas notas de projeto e no desenho do pipeline, mas não aparece no pôster. Use ao responder perguntas.
> - **[verificar]** — reconstruído a partir das notas; confira no seu código antes de afirmar em público.

---

## 1. A pergunta, e de onde ela veio

**A pergunta do trabalho [pôster]:** quando uma CNN híbrida quântico-clássica vai bem em imagem biomédica, o componente quântico está contribuindo?

**De onde a pergunta nasceu [projeto].** Na qualificação havia três modelos: o baseline clássico do grupo, a H-QCNN v1 e a H-QCNN v2. A v2 ganhava do baseline (F1-macro 0,4437 contra 0,3817) e o texto creditava o ganho ao circuito. Só que entre a v2 e o baseline mudavam **cinco coisas ao mesmo tempo**: ajuste fino da layer4, validação LORO, taxas de aprendizado diferenciadas por parte do modelo, label smoothing e ensemble de dez modelos com TTA. O circuito eram **40 parâmetros em 14.981.204 treináveis — 0,0003% do modelo**.

Pior: a própria análise SHAP daquele capítulo creditava o ganho à camada clássica ajustada, enquanto a conclusão creditava ao componente quântico. Uma contradição interna dentro do mesmo trabalho.

A v1 também não servia de controle: dos 16.468 parâmetros treináveis dela, 16.392 eram a projeção `Linear(2048→8)`, ou seja 99,5%. Comparar v1 com o baseline mudava duas coisas de uma vez — comprimir 2.048 dimensões em 8 **e** trocar o classificador por um circuito.

**A ablação do pôster existe para resolver exatamente isso:** manter tudo fixo e trocar só o módulo. Essa é a contribuição metodológica, e é ela que sustenta o resultado nulo.

---

## 2. Os dados

### 2.1 Origem [projeto]

- Acervo construído por **Silva (2025)** — a referência [8] do pôster — a partir de vídeos termográficos de sessões de corrida em esteira, adquiridos em **2019 pelo LAFAE/FCA-Unicamp**.
- **Nenhum experimento novo com animais foi feito neste trabalho.** Isso é importante de dizer se perguntarem sobre ética: o dado é secundário.
- Câmera **FLIR One Pro**, a 28 cm da esteira, posicionada perpendicularmente. Taxa efetiva ~7 quadros por segundo.

### 2.2 Tamanho e classes [pôster + projeto]

| Classe (pôster) | Nome fisiológico | Quadros | Proporção |
|---|---|---|---|
| Warm-up | Baixo (aquecimento) | 12.141 | 38,16% |
| Moderate | Moderado (limiar de lactato) | 11.436 | 35,94% |
| Heavy | Intenso (máxima fase estável de lactato, MFEL) | 6.850 | 21,53% |
| Severe | Severo (supramáximo) | 1.391 | **4,37%** |

- Total: **31.818 quadros, 11 ratos Wistar**, razão de desbalanceamento ≈ 8,7 : 1.
- **Apenas 4 dos 11 animais percorreram os quatro domínios.** É por isso que os ratos de teste são 1, 9, 10 e 11: são os únicos que têm a classe severa, e sem ela não dá para medir recall da classe rara.

### 2.3 A propriedade que governa todo o desenho [projeto]

**O número efetivo de observações independentes é 11, não 31.818.** Quadros consecutivos a ~7 fps são quase duplicatas, e o rótulo é colinear ao tempo de sessão: qualquer coisa que varie com o tempo (aquecimento da esteira, condensação, deriva do sensor) é um atalho disponível para o classificador.

Consequências que você deve saber defender:
- a partição tem de ser por **sujeito**, nunca por quadro;
- intervalo de confiança por bootstrap tem de reamostrar **sujeitos**;
- a unidade de um teste pareado deveria ser o sujeito (veja a §8, onde isso vira uma limitação real do pôster).

### 2.4 Natureza da imagem [projeto]

As imagens são **pseudo-coloridas**: a FLIR renderiza um único canal de temperatura através de um colormap. Os três canais RGB **não são cor, são codificação de temperatura**. Isso proíbe algumas augmentações comuns:

- **Grayscale:** num colormap tipo *jet*, a luminância não é monotônica na temperatura — há inversões de direção, e temperaturas diferentes colapsam no mesmo cinza. Destrói informação de forma irreversível.
- **ColorJitter, jitter de matiz/saturação, brilho e contraste:** sobre um colormap, isso **é** a temperatura aparente. Perturbar é perturbar a variável de interesse.
- **Flip vertical:** inverte a orientação anatômica do animal na baia.
- Permitido: rotação ≤ 5°, translação ≤ 3%, escala 0,97–1,03; resize 224×224 e normalização ImageNet.

**[verificar]** A qualificação usou `Grayscale(3)` e `ColorJitter(±20%)`. Se o pipeline do pôster herdou essas transformações, isso é uma limitação conhecida e você deve reconhecê-la se perguntarem — não é fatal para a ablação (os dois módulos sofrem igual, o pareamento continua válido), mas prejudica o desempenho absoluto dos dois.

---

## 3. A arquitetura, peça por peça [pôster]

```
frame térmico 224×224
      ↓
ResNet-50 (ImageNet), conv1–layer3 congeladas, layer4 com ajuste fino   η = 10⁻⁵
      ↓  vetor de 2.048 dimensões (global average pooling)
Linear(2048 → 8)                                                        (projeção)
      ↓  z ∈ ℝ⁸
MÓDULO INTERCAMBIÁVEL                                                   η = 10⁻⁴
   ├── quântico:  PQC de 8 qubits, 5 camadas  → 40 parâmetros
   └── clássico:  Linear(8→8) + tanh          → 72 parâmetros
      ↓  saída em (−1, 1)⁸
Linear(8 → 4 domínios)                                                  η = 10⁻³
```

**Contagem de parâmetros treináveis [projeto]:** layer4 = 14.964.736; projeção = 16.392; módulo = 40 (quântico) ou 72 (clássico); classificador = 36. Total ≈ **14.981.204**.

Guarde esse número: **o módulo quântico é 0,0003% do que é treinado**. É a base da pergunta mais dura que você pode receber (§13, Q2), e é melhor você trazer o número do que deixar alguém trazer.

---

## 4. O circuito quântico [pôster + projeto]

**Especificação:**

| Item | Valor |
|---|---|
| Qubits | 8 (espaço de Hilbert de dimensão 256) |
| Codificação | `AngleEmbedding` com R_X, ângulo θᵢ = σ(zᵢ)·π (sigmoide escalada) |
| Ansatz | `BasicEntanglerLayers`, 5 camadas: R_X por qubit + anel de CNOTs |
| Parâmetros | 5 camadas × 8 qubits = **40** |
| Profundidade | 46 (1 de codificação + 5 × [1 rotação + 8 CNOTs em cascata]) |
| Leitura | ⟨Zᵢ⟩ por qubit → vetor em (−1, 1)⁸ |
| Execução | PennyLane `default.qubit`, modo analítico (vetor de estado exato), `diff_method="backprop"`, interface torch |

**Duas fragilidades técnicas que você deve conhecer antes que alguém as aponte [projeto]:**

1. **A primeira camada variacional é quase inerte.** A codificação é R_X e a primeira rotação variacional também é R_X, ambas no mesmo qubit e antes de qualquer CNOT. Como R_X(θᵢ)·R_X(w₁ᵢ) = R_X(θᵢ + w₁ᵢ), os 8 parâmetros da primeira camada são um **viés aditivo sobre os ângulos de codificação**, não uma transformação nova. A profundidade estruturalmente efetiva é 4 camadas, não 5. A correção é trocar o eixo da variacional (R_Y ou R_Z) ou usar `Rot(φ,θ,ω)`, e já está prevista para o próximo experimento.
2. **Em modo analítico não há estocasticidade.** Sem `shots`, não há medição nem colapso: o circuito é uma função determinística e diferenciável de ℝ⁸ → (−1,1)⁸. Qualquer motivação do tipo "regularização probabilística vinda da natureza estocástica da medição" **não se aplica** à configuração usada. Se alguém perguntar pelo mecanismo esperado, a resposta honesta é "um mapa de características não linear e compacto", não estocasticidade.

**Por que 8 qubits e 5 camadas?** Foi a configuração herdada da qualificação. É o extremo superior do que a área pratica — num piloto de extração de 10 estudos do corpus, a mediana ficou em 2–3 qubits, e há evidência de que aumentar qubits e profundidade **degrada** o desempenho (Islam et al., 2025; Halab et al., 2026). O próximo experimento varre k ∈ {2,4,6} e L ∈ {2,4,6}.

---

## 5. O comparador clássico, e por que ele é assim [pôster + projeto]

O controle é `Linear(8→8) + tanh`: **72 parâmetros**, saída na mesma faixa (−1, 1) de ⟨Zᵢ⟩, no mesmo ponto do pipeline, recebendo exatamente o mesmo vetor z.

Três decisões por trás disso:

1. **Tem de ser diferenciável.** Uma random forest — que é o que o baseline clássico do grupo usa — quebraria o gradiente para a layer4. Sem gradiente, o backbone deixa de ser ajustado, e aí a comparação volta a mudar duas coisas de uma vez: o módulo **e** o treinamento do backbone. Isso reintroduziria exatamente o confundidor que a ablação existe para eliminar.
2. **Tem de ter o mesmo gargalo dimensional.** Os dois módulos recebem 8 valores e devolvem 8 valores. Sem isso, parte da diferença seria diferença de compressão, não de módulo.
3. **Não é pareado em número de parâmetros** (72 vs 40) — e é por isso que o pôster diz apenas *matched*, e não *parameter-matched*. O que está pareado é a **função** e a **interface**: mesma entrada, mesma faixa de saída, mesmo lugar, mesmo treinamento.

**Bernini é baseline externo, não controle [projeto].** O trabalho do grupo (ResNet-50 + Random Forest) difere em eixos demais de protocolo para servir de comparador pareado. Ele ancora contexto entre estudos, e só.

---

## 6. O protocolo experimental [pôster]

| Item | O que é |
|---|---|
| Perda | Entropia cruzada ponderada por classe, com label smoothing |
| Otimizador | Adam com taxa por papel: layer4 10⁻⁵, módulo 10⁻⁴, classificador 10⁻³ |
| Ratos de teste | 1, 9, 10 e 11 (os quatro com classe severa) |
| Sementes | duas por rato → 8 configurações pareadas, 16 execuções |
| Validação interna | LORO sobre os 10 ratos restantes = 10 folds por configuração (daí os 80 folds) |
| Inferência | TTA com T = 5 e ensemble dos modelos dos folds |
| Estatística | Wilcoxon pareado, sobre as 8 configurações e sobre os 80 folds |
| Execução | PennyLane `default.qubit` + PyTorch, simulação por vetor de estado |

**O que faz o pareamento valer:** para cada configuração, os dois módulos veem o mesmo rato de teste, os mesmos folds, a mesma semente, a mesma augmentação e o mesmo ensemble. Uma única flag no código (`block_type`) controla a troca; todo o resto é literalmente o mesmo caminho de execução. **[projeto]** A cobertura dos grupos do otimizador foi verificada: cada parâmetro treinável está em exatamente um grupo, sem vazamento de parâmetro congelado, com os parâmetros do módulo isolados.

**[projeto]** Houve um bug de contaminação — um `rglob` que agregava predições antigas — identificado e corrigido. Se alguém perguntar sobre reprodutibilidade, é um bom exemplo de que os resultados foram auditados, não aceitos de primeira.

---

## 7. Interpretabilidade [pôster + projeto]

- SHAP contrastivo é calculado para os dois módulos na mesma configuração (rato 1, semente 42), de forma que os mapas são comparáveis entre si.
- O achado: os mapas não mudam de lugar quando o módulo muda. O que muda a atribuição é o ajuste fino do backbone. É isso que sustenta a frase do pôster de que o efeito mora no backbone, e não no circuito.
- **Antecedente [projeto]:** na qualificação, a comparação SHAP entre v1 e v2 mostrou a v1 ancorando decisões em ruído do aparato (a esteira) e a v2 na silhueta anatômica do animal. Esse é o achado mais defensável daquele capítulo.
- **Limitação:** SHAP é verificação qualitativa, não validação. O próximo passo é transformar isso em número, com a **razão de densidade de atribuição entre a região do animal e o fundo** — pergunta direta: o modelo extrai evidência do animal ou do equipamento?

---

## 8. Estatística [pôster + projeto]

- Teste: **Wilcoxon pareado (postos sinalizados)**, escolhido por não exigir normalidade das diferenças.
- Unidade: a **configuração** (rato × semente), n = 8; e, como análise secundária, o **fold**, n = 80.
- Com n = 8, o menor p bilateral possível é 2/2⁸ ≈ 0,0078 — que é exatamente o p = 0,008 relatado. Ou seja: **o resultado da classe severa é o mínimo p alcançável com esse n**, o que só acontece porque o clássico venceu em 8 de 8.

**A limitação a admitir de cara:** as 8 configurações vêm de 4 ratos × 2 sementes, então **não são independentes** — duas configurações do mesmo rato compartilham o mesmo animal de teste. Se a unidade fosse o rato (n = 4), o menor p bilateral possível seria 0,125, e nada seria significativo. O que sustenta a conclusão não é a magnitude do p, é a **consistência**: o bloco clássico vence em todas as configurações e em todos os quatro ratos.

O redesenho do próximo artigo resolve isso usando **o sujeito como unidade pareada com os 11 animais**, o que leva o menor p possível a 2/2¹¹ ≈ 0,00098.

---

## 9. Os resultados, e como ler cada número [pôster]

| Modelo | Acur. balanceada | F1-macro | Precisão | Recall | Recall severo |
|---|---|---|---|---|---|
| Quântico (PQC, 40 par.) | 0,529 | 0,438 | 0,541 | 0,529 | 0,518 |
| Clássico (pareado, 72 par.) | 0,532 | 0,411 | 0,492 | 0,532 | 0,777 |
| p pareado (Wilcoxon) | 0,74 | 0,38 | — | 0,74 | **0,008** |

Como ler:

- **Acurácia balanceada e recall macro são o mesmo número** (0,529 e 0,532): acurácia balanceada é a média dos recalls por classe.
- **O quântico "ganha" em F1-macro e precisão, mas p = 0,38.** Não é diferença; é ruído. Não venda isso como vantagem parcial — é justamente o tipo de leitura que o trabalho critica.
- **O quântico vence a acurácia balanceada em 2 das 8 configurações** — e essas duas são o rato 1 com as duas sementes. É o antigo sujeito de teste fixo, e também o rato usado como referência da atribuição. Coincidência que vale mencionar como alerta: era o único rato olhado antes.
- **Em um dos outros três ratos, o braço quântico tem recall zero na classe severa.**
- **A única diferença significativa é o recall da severa**, e favorece o clássico.

**As matrizes de confusão (Fig. 3):**
- Os dois módulos praticamente não recuperam a classe *intensa* (recall < 0,12). Ela é confundida principalmente com *moderado* e com *severo* — o que faz sentido fisiológico: é a classe do meio, entre o limiar de lactato e o supramáximo.
- O clássico chega a um recall maior na severa **prevendo severa com mais frequência**: a coluna destacada soma 14.342 predições contra 7.823 do quântico. Por isso a precisão da severa cai (0,17 no clássico contra 0,21 no quântico). É deslocamento do ponto de operação, não separação melhor das classes.
- **Os valores de recall da figura não batem com os da tabela** (severa: 0,59 e 0,87 na figura; 0,518 e 0,777 na tabela) porque a figura soma todos os quadros de teste e a tabela é a média por configuração. Duas agregações diferentes.

---

## 10. O que o trabalho afirma — e o que não afirma

**Afirma [pôster]:**
- Sob este protocolo, com este comparador, o módulo quântico de 40 parâmetros não traz vantagem mensurável, e é pior na classe rara.
- O ganho relatado antes vem do ajuste fino do backbone clássico, e a atribuição aponta para o mesmo lugar.
- Vantagens relatadas para H-QCNNs dependem do protocolo de validação; sem comparador pareado e sem validação independente do sujeito, a atribuição do ganho não se sustenta.

**Não afirma:**
- Que aprendizado de máquina quântico não funciona, ou não vai funcionar, para imagem médica.
- Que não exista vantagem quântica em outra arquitetura, outra codificação, outro tamanho de circuito ou outra modalidade.
- Que o resultado valha para hardware real — tudo aqui é simulação exata.
- Que exista aplicação clínica. O trabalho é sobre método, não sobre produto.

---

## 11. Limitações conhecidas (a lista honesta)

1. **Sensibilidade baixa por construção:** 40 parâmetros dentro de ~15 milhões treináveis. O experimento responde "trocar o módulo muda alguma coisa neste pipeline?", e não "um componente quântico pode contribuir quando ele é a maior parte do que se treina?".
2. **Pares não independentes:** 4 ratos × 2 sementes (§8).
3. **Só 11 sujeitos no acervo, e só 4 com a classe severa.** Nenhuma quantidade de quadros muda isso.
4. **Desbalanceamento severo** (4,37% na classe rara).
5. **Rótulo colinear ao tempo de sessão** — mitigado pela partição por sujeito e verificado por SHAP, não eliminado.
6. **Primeira camada variacional quase inerte** (§4).
7. **Tudo em simulação analítica:** sem shots, sem ruído, sem topologia de dispositivo.
8. **Uma única configuração de circuito** (8 qubits, 5 camadas); nenhuma varredura de largura e profundidade.
9. **Sem controles de circuito congelado ou de ablação de emaranhamento** — o que distinguiria "o circuito aprendeu algo" de "o circuito é um mapa aleatório fixo". Estão previstos para o próximo experimento.
10. **[verificar]** Possível uso de augmentações incompatíveis com imagem pseudo-colorida (§2.4).

---

## 12. O que vem depois [projeto]

**Redesenho do experimento (o artigo seguinte):**
- **Congelar o backbone e cachear as features de 2.048 dimensões**, e reduzir com **PCA ajustada só no treino do fold** (zero parâmetros treináveis). Assim o que se treina passa a ser quase só o circuito: a **fração quântica sobe de 0,0003% para ~78%**, e a pergunta "o componente quântico contribui?" fica literalmente respondível.
- **Controle casado** `MLP(k→2k→k)` com contagem de parâmetros a 4 unidades do circuito.
- **Dois controles novos: circuito aleatório congelado** — se o desempenho se mantém, o circuito é um mapa de características aleatório, não algo que aprendeu; ausente em 0 dos 81 estudos revisados — e **ablação de emaranhamento**, com o mesmo número de parâmetros e sem as portas de dois qubits, que só 2 dos 81 fazem.
- **Unidade pareada = sujeito**, com os 11 animais.
- **Pré-registro datado** da comparação primária, da métrica primária e da família de secundários, antes de qualquer execução.
- **Efeito mínimo detectável:** a menor diferença de F1-macro que o desenho detecta com 80% de potência. Vira uma afirmação para fora: *nesta escala de sujeitos, alegações de vantagem abaixo de X pontos não são sustentáveis pelos dados que as acompanham*.
- **Varredura de shots finitos** (∞, 1024, 128, 32): se a "regularização estocástica" existir, ela aparece aqui e some no modo analítico.

**Hardware real:** **treinar** em QPU está fora de alcance por três ordens de grandeza — sem retropropagação, o gradiente exige *parameter-shift*, 2 execuções por parâmetro por amostra, o que dá ~6,7 × 10⁸ execuções de circuito para os quatro folds. O que é viável é **inferência**: treinar em simulação, congelar os parâmetros e rodar só a passada direta de um subconjunto do teste no dispositivo, decompondo a degradação em (i) simulador analítico, (ii) shots finitos, (iii) ruído modelado pela calibração e (iv) hardware real. Essa decomposição não existe no corpus revisado.

**Fase 2 [projeto]:** segunda modalidade, radiografia abdominal pediátrica (artigo companheiro, 241 imagens, ResNet18). A predição registrada é que o gargalo de poucas dimensões **preserva** o sinal na termografia (distribuição de temperatura é de baixa dimensão) e **destrói** o sinal na radiografia (morfologia de alta frequência espacial).

---

## 13. Perguntas e respostas

> Respostas curtas, do jeito que dá para falar. As mais duras trazem também uma versão em inglês.

### Sobre o desenho do experimento

**Q1. Por que trocar só o módulo, em vez de comparar com um modelo clássico completo?**
Porque comparar modelos diferentes não responde à pergunta. Se eu mudo o classificador, o treinamento e a divisão dos dados ao mesmo tempo, qualquer diferença tem três explicações possíveis. Trocando só o módulo, sobra uma.

**Q2. Seu módulo quântico tem 40 parâmetros dentro de uma rede de 15 milhões. O resultado nulo não está garantido por construção?** ⚑
Essa é a limitação principal, e ela é real: a sensibilidade do experimento ao módulo é baixa por construção. Duas coisas seguram o resultado mesmo assim. Primeira: o comparador clássico está na mesma situação, então a comparação continua justa — e ele ganha na classe rara. Segunda: a atribuição mostra **onde** o efeito mora, que é no backbone. O próximo experimento ataca exatamente isso: congela o backbone e usa uma redução sem parâmetros, de modo que o circuito passe a ser cerca de 80% do que é treinado.
*EN:* "That's the main limitation, and it's real — the experiment has low sensitivity to the module by design. But the classical comparator sits in the same position and still wins on the rare class, and the attribution shows the effect lives in the backbone. The next study freezes the backbone so the quantum part is about 80% of what is trained."

**Q3. O comparador tem 72 parâmetros e o circuito tem 40. Isso não é pareado.**
Não em número de parâmetros, e por isso eu digo apenas *matched*. O que está pareado é a função: mesma entrada de 8 dimensões, mesma faixa de saída, mesmo ponto do pipeline, mesmo treinamento. E note a direção: o bloco clássico tem mais parâmetros e ainda assim o resultado é de empate nas métricas gerais — o viés, se existe, é contra a minha conclusão, não a favor.

**Q4. Por que não usar uma random forest como comparador, já que é o baseline do grupo?**
Porque ela não é diferenciável. Sem gradiente, a layer4 deixa de ser ajustada, e a comparação passa a mudar duas coisas ao mesmo tempo: o módulo e o treinamento do backbone. Voltaria o confundidor que a ablação existe para eliminar.

**Q5. Por que não congelar o backbone?**
Porque eu queria reproduzir o pipeline em que o ganho tinha sido relatado, e nele o backbone é ajustado. Congelar muda a pergunta, e é o desenho do próximo experimento — lá o congelamento é justamente o que torna o circuito grande o suficiente para ser medido.

**Q6. Por que Leave-One-Rat-Out?**
Porque o vídeo é amostrado a ~7 quadros por segundo: quadros vizinhos são quase idênticos. Se a divisão for por quadro, o modelo já viu o animal de teste, e o desempenho sobe sem aprendizado nenhum. Além disso, o rótulo é colinear com o tempo de sessão, então qualquer coisa que varie com o tempo vira atalho.

**Q7. Os dois módulos rodaram nos mesmos folds?**
Sim, nos mesmos folds, com a mesma semente, a mesma augmentação e o mesmo ensemble. Uma flag controla a troca do módulo; todo o resto é o mesmo caminho de código. É isso que autoriza o teste pareado.

### Sobre a parte quântica

**Q8. Por que 8 qubits e 5 camadas?**
É a configuração herdada do trabalho anterior, e é o extremo superior do que a área pratica — num piloto de dez estudos do corpus que revisei, a mediana ficou em 2 ou 3 qubits. Há evidência de que aumentar qubits e profundidade piora o desempenho. Varrer largura e profundidade é item do próximo experimento.

**Q9. A codificação é R_X e a primeira camada variacional também é R_X. A primeira camada não vira só um viés aditivo?** ⚑
Vira, sim. As duas rotações são no mesmo eixo e no mesmo qubit antes de qualquer CNOT, então elas somam: os 8 parâmetros da primeira camada são um deslocamento nos ângulos de codificação, e a profundidade efetiva é 4, não 5. Isso está identificado e a correção é trocar o eixo da variacional. Não muda a conclusão — se alguma coisa, significa que o circuito é ainda menos expressivo do que parece.
*EN:* "Yes — they commute, so the first variational layer is an additive bias on the encoding angles and the effective depth is four, not five. It's identified, and the fix is to change the rotation axis. If anything, it means the circuit is even less expressive than it looks."

**Q10. Como você sabe que o emaranhamento contribui?**
Não sei, e o trabalho não testa isso. O controle que responderia — o mesmo circuito sem as portas de dois qubits, com o mesmo número de parâmetros — está no próximo experimento. Vale dizer que existe evidência na literatura de que remover o emaranhamento muitas vezes iguala ou melhora o desempenho.

**Q11. E platôs áridos? Você não está medindo um circuito que não treina?**
Nesta escala, 8 qubits e 5 camadas, o platô árido não é a ameaça dominante; o risco concreto é sobreajuste e custo. Mas eu não tenho um controle que separe "não contribui" de "não treinou", e é para isso que serve o circuito aleatório congelado no próximo desenho: se o desempenho se mantiver com o circuito não treinado, ele está funcionando como mapa aleatório.

**Q12. Rodou em hardware real?**
Não, é tudo simulação por vetor de estado. Treinar em hardware está fora de alcance: sem retropropagação, o gradiente sai por *parameter-shift*, que custa duas execuções por parâmetro por amostra — dá algo como 10⁸ execuções de circuito. O que é viável é inferência com parâmetros treinados em simulação, e é o próximo passo.

**Q13. Sem shots e sem ruído, isso é um circuito quântico ou uma função determinística?**
No modo analítico é uma função determinística e diferenciável de ℝ⁸ em (−1,1)⁸. É exatamente por isso que qualquer motivação baseada na estocasticidade da medição não se aplica aqui. Uma varredura de shots finitos está prevista, e ela transforma essa objeção em experimento.

**Q14. Que mecanismo você esperava que desse vantagem?**
Um mapa de características não linear e compacto, que é a justificativa mais comum na literatura para esse tipo de módulo. O trabalho testa contribuição, não mecanismo — e a contribuição não apareceu.

### Sobre estatística

**Q15. p = 0,008 com n = 8, e os pares vêm de 4 ratos com 2 sementes. Isso não é independente.** ⚑
Concordo. As oito configurações não são independentes, e o p formal é otimista. Com o rato como unidade, n = 4 e o menor p possível seria 0,125 — nada seria significativo. O que sustenta a conclusão é a consistência: o bloco clássico ganha em todas as configurações e em todos os quatro ratos. No próximo desenho a unidade pareada passa a ser o sujeito, com os 11 animais.
*EN:* "Agreed — the eight pairs are not independent and the p-value is optimistic. What holds the conclusion up is consistency: the classical block wins in every configuration and in all four rats. The next design uses the subject as the pairing unit, with eleven animals."

**Q16. Por que Wilcoxon e não teste t?**
Por não depender de normalidade das diferenças, com um n pequeno em que testar normalidade também tem pouca potência. O ideal, e é o que vou fazer, é decidir por Shapiro-Wilk declarado antes de rodar.

**Q17. Você fez várias comparações. Corrigiu multiplicidade?**
No pôster são quatro comparações com p relatado, e só uma deu significativa. Não há correção aplicada, e isso é uma limitação: com muitas comparações a α = 0,05 sem correção, a chance de um falso positivo cresce rápido. O próximo desenho declara uma comparação primária antes de rodar e joga o resto numa família com Holm-Bonferroni, rotulada como exploratória.

**Q18. Por que os números da figura não batem com os da tabela?**
A figura soma todos os quadros de teste; a tabela é a média por configuração. São duas formas de agregar. Nas duas, a direção é a mesma.

### Sobre os dados

**Q19. Por que nenhum dos dois recupera a classe intensa?**
Ela é a classe do meio, entre o limiar de lactato e o supramáximo, e é a mais confundida com as vizinhas. Fisiologicamente é o limite mais sutil dos três, e a assinatura térmica dela se sobrepõe às outras.

**Q20. A diferença no recall da classe severa não é só um limiar diferente?** ⚑
É essencialmente isso, e eu digo isso no pôster. O bloco clássico prevê severa quase o dobro de vezes; o recall sobe e a precisão cai. É deslocamento de ponto de operação, não separação melhor. Por isso eu não apresento a diferença como "o clássico entende melhor a classe rara".
*EN:* "Essentially yes, and the poster says so. The classical block predicts severe about twice as often: recall goes up, precision goes down. It is an operating-point shift, not better separability."

**Q21. Como você sabe que o modelo olha o animal e não a esteira?**
Não sei com certeza, e esse é um achado real do trabalho anterior: a primeira versão ancorava decisões em ruído do aparato, e a versão seguinte, na silhueta do animal. A próxima etapa transforma isso em número, com a razão de densidade de atribuição entre a região do animal e o fundo.

**Q22. As imagens são pseudo-coloridas. Vocês usaram augmentação de cor?**
As cores são um colormap de temperatura, então mexer em brilho, contraste ou saturação é mexer na variável de interesse, e converter para cinza destrói informação de forma não monotônica. Esse é o padrão que adotei a partir da análise do dado. Se quiser conferir a configuração exata desta rodada, eu te mostro depois.

**Q23. O rótulo é colinear ao tempo da sessão. O modelo não está aprendendo o relógio?**
Pode estar aprendendo parte disso, e eu não elimino essa possibilidade — só mitigo, dividindo por sujeito e olhando a atribuição. Um sujeito novo tem outro aparato, outra condição térmica inicial, e é isso que torna o atalho menos útil.

### Sobre o resultado e o posicionamento

**Q24. Seu resultado quer dizer que QML não serve para imagem médica?**
Não. Quer dizer que, neste pipeline, com este circuito e este protocolo, o módulo quântico não contribui — e que boa parte dos ganhos relatados na literatura não sobrevive a um comparador pareado com validação independente do sujeito. É uma afirmação sobre método, não sobre o futuro da área.

**Q25. Isso não é só um resultado negativo? Onde está a contribuição?** ⚑
A contribuição é o desenho: isolar o módulo com um comparador de mesma função, validar sem vazamento entre sujeitos, e localizar o efeito com atribuição. Isolar o componente já foi feito antes; o que não existia era fazer isso com validação independente do sujeito, atenta ao desbalanceamento, e com uma análise que mostra onde o efeito realmente mora.
*EN:* "The contribution is the design: isolating the module with a same-role comparator, validating without subject leakage, and locating the effect with attribution. Isolating the component has been done; doing it subject-independently, with an imbalanced rare class, and showing where the effect actually lives, had not."

**Q26. Qual é o próximo experimento?**
Congelar o backbone e reduzir as features com PCA, para que o circuito passe a ser a maior parte do que se treina; acrescentar dois controles que não existem no corpus, o circuito aleatório congelado e a ablação de emaranhamento; usar o sujeito como unidade pareada, com 11 animais; e pré-registrar a comparação primária antes de rodar.

**Q27. Dá para usar isso na prática, para monitorar exercício?**
Não é o objetivo do trabalho e eu não faria essa afirmação. O desempenho absoluto é modesto, são 11 animais, e a pergunta aqui é metodológica.

---

## 14. Glossário rápido

| Termo | Em uma frase |
|---|---|
| **NISQ** | Regime atual dos computadores quânticos: poucos qubits, com ruído, sem correção de erro. |
| **PQC / VQC** | Circuito quântico parametrizado (ou variacional): portas com ângulos treináveis, otimizados como pesos de rede neural. |
| **Ansatz** | A estrutura fixa do circuito — quais portas, em que ordem — dentro da qual os parâmetros são treinados. |
| **AngleEmbedding** | Codificação clássica→quântica em que cada valor de entrada vira o ângulo de uma rotação de um qubit. |
| **BasicEntanglerLayers** | Template do PennyLane: uma rotação por qubit seguida de um anel de CNOTs, repetido L vezes. |
| **⟨Z⟩** | Valor esperado do observável Pauli-Z de um qubit; fica entre −1 e 1 e é a saída usada aqui. |
| **Shots** | Número de repetições da medição. Em modo analítico não há shots: o simulador devolve o valor exato. |
| **Parameter-shift** | Regra para obter gradiente em hardware, onde não há retropropagação; custa 2 execuções por parâmetro. |
| **ZNE** | Extrapolação a ruído zero: roda o circuito com ruído amplificado e extrapola para o caso sem ruído. |
| **Platô árido** | Região em que o gradiente do circuito some exponencialmente com o número de qubits, impedindo o treino. |
| **Backbone** | A rede pré-treinada que extrai as características da imagem — aqui, a ResNet-50. |
| **Ajuste fino (fine-tuning)** | Continuar treinando parte do backbone nos dados do problema, em vez de deixá-lo congelado. |
| **LORO** | Leave-One-Rat-Out: validação em que cada rodada deixa um animal inteiro fora do treino. |
| **TTA** | Test-time augmentation: várias versões levemente transformadas da mesma imagem na inferência, com as predições combinadas. |
| **Ensemble** | Combinação das predições de vários modelos (aqui, um por fold). |
| **Label smoothing** | Suavização dos rótulos alvo, que reduz excesso de confiança do modelo. |
| **F1-macro** | Média simples do F1 das classes; ao contrário da média ponderada, não deixa a classe rara sumir. |
| **Acurácia balanceada** | Média dos recalls por classe; aqui é igual ao recall macro. |
| **Recall / precisão** | Recall: dos exemplos de uma classe, quantos o modelo achou. Precisão: das predições de uma classe, quantas estavam certas. |
| **Wilcoxon pareado** | Teste não paramétrico para comparar duas condições medidas nas mesmas unidades. |
| **SHAP** | Método de atribuição que estima quanto cada região da imagem contribuiu para a predição. |
