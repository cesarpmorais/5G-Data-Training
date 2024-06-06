# [Aprendizado Por Reforço (RL)](https://www.geeksforgeeks.org/what-is-reinforcement-learning/)
Aprendizado por Reforço é um dos 3 paradigmas básicos de Machine Learning, junto do aprendizado supervisionado e não-supervisionado.
Ele é baseado em tentativa e recompensa. Quanto melhor for a solução do agente, maior a recompensa que ele receberá. O objetivo é encontrar uma solução ótima, ou quase-ótima, que maximiza a "função de recompensa". É muito útil em situações que não temos datasets que representam o problema, ou quando não sabemos o "passo-a-passo" da solução.

No aprendizado supervisionado, é possível, a cada passo do modelo, recompensar decisões corretas e tentar corrigir decisões erradas, já que temos as respostas corretas como guia. 
Entretanto, em RL, não sabemos exatamente como chegar ao fim desejado. Isso é decorrente da natureza dos problemas que tentamos resolver - novamente, não há um "passo a passo" definido. Por isso, o foco é tomar decisões sequencialmente, e cada decisão é dependente da próxima (ex: jogo de xadrez, sumarização de texto).

Um tradeoff clássico de RL é o de "exploration vs. exploitation". Este diz respeito à escolha de uma solução que já conhecemos, enquanto ese envolve tentar novas opções de solução, que custa tempo, mas pode chegar a uma solução final melhor. O agente deve decidir quando usar o exploit de soluções conhecidas, e quando explorar novas soluções.

RL possui 4 elementos principais (além do agente):
1. Policy: determina como o agente se comporta em um período de tempo. É uma função do estado do agente e do cenário do ambiente, e pode ser tão complexa quanto necessário;
2. Função de Recompensa: uma função que provê uma pontuação numérica baseada no estado do ambiente;
3. Valor do Estado: é a quantidade acumulada de recompensas que um agente pode receber no futuro se ficar nesse estado. É uma noção em longo-prazo, pois estados que aparentam não ser bons no momento podem levar a outros muito recompensantes.
4. Modelo do Ambiente: é um mecanismo que simula as condições ambientais e possibilita predições para o agente.

## Vantagens e Desvantagens
### Vantagens
- Pode ser usado em problemas complexos, incapazes de serem resolvidos por técnicas comuns;
- Dados de treinamento são gerados ao longo da interação do agente com o ambiente;
- É flexível e funciona em diversos casos;
- Funciona em ambientes altamente não-determinísticos.
### Desvantagens
- Caro em dados e computação - não se deve utilizar para problemas simples;
- Complexo de debugar e interpretar;
- Altamente dependente da qualidade da Função de Recompensa - deve ser corretamente modelada.

## [Tipos de algoritmos de RL - OpenAI](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
Um dos pontos mais importantes de qualquer algoritmo de RL é a existência ou não de um modelo, ou seja, uma função que prediz transição de estado e recompensas.

Caso seja possível prover tal modelo para a máquina, os agentes podem planejar e prever suas ações, o que resulta em maior eficiência de aprendizado.

Entretanto, não possuímos tal modelo na maioria dos casos, e nesse caso o agente deve descobrir o modelo por si mesmo. Esse cenários traz desafios: o modelo pode ter um viés exploitado pelo agente que não se traduz em boa performance no mundo real, por exemplo. Apesar disso, soluções Model-Free são geralmente mais fáceis de implementar e ajustar.

### [Model-Free RL](https://medium.com/@oumoudhmine/q-learning-and-ppo-driving-forces-in-openais-ai-mastery-655027c8670f)
A OpenAI usa dois métodos: Q-Learning e Proximal Policy Optimization (PPO). Como cada um funciona?

PPO tem como principal vantagem o fato de que estamos otimizando diretamente o nosso objetivo, através da policy. Q-Learning é menos estável nesse sentido, mas é mais eficiente, pois pode reutilizar melhor os dados. Existem algoritmos que usam os dois métodos em conjunto.

### Model-Based RL
Existem formas variadas de se usar modelos, mas algumas das principais são:

- Planejamento Puro: é a abordagem mais simples - a policy nunca é explicitada, mas sim técnicas de planejamento são usadas para selecionar ações. Um exemplo é o Model-Predictive Control, onde o agente constrói um plano para um horizonte fixo de tempo com base no modelo do ambiente. Após executar a primeira ação desse plano, o agente descarta o restante do plano e refaz o planejamento com base no novo estado observado. Isso permite que o agente se ajuste dinamicamente a novas informações a cada passo;

- Expert Iteration: Nesta abordagem, o agente alterna entre aprendizado e planejamento. Usando a policy atual, o agente gera ações candidatas para os próximos passos por meio de algum algoritmo de planejamento. O planejamento age como um "expert", refinando as ações sugeridas pela policy. Isso pode resultar em um agente que melhora sua policy ao imitar e aprender das ações mais refinadas geradas pelo planejamento. É um ciclo iterativo onde a policy e o modelo se aprimoram mutuamente.

- Data Augmentation for Model-Free Methods: a ideia é aproveitar o modelo para gerar experiências fictícias, que são usadas para treinar algoritmos Model-Free. Isso pode aumentar significativamente a quantidade de dados de treinamento disponíveis, melhorando a eficiência do aprendizado.

OBS: Explicações em alto nível, assunto fica bem complexo rapidamente. Chat-GPT ajudou para escrever o resumo!