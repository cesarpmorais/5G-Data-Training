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

## Vantagens
- Pode ser usado em problemas complexos, incapazes de serem resolvidos por técnicas comuns;
- Dados de treinamento são gerados ao longo da interação do agente com o ambiente;
- É flexível e funciona em diversos casos;
- Funciona em ambientes altamente não-determinísticos.
## Desvantagens
- Caro em dados e computação - não se deve utilizar para problemas simples;
- Complexo de debugar e interpretar;
- Altamente dependente da qualidade da Função de Recompensa - deve ser corretamente modelada.