# Previsão de Chuvas

Este repositório contém um modelo de aprendizado de máquina para prever chuvas.

## Introdução

Este modelo usa dados históricos de chuvas e outras variáveis meteorológicas para prever a probabilidade de chuva em uma determinada região.

## Impacto do Projeto

Este projeto de previsão de chuvas pode ter um impacto significativo na qualidade dos serviços prestados e na segurança em áreas de risco, como desabamentos e enchentes. Com previsões precisas de chuvas, as autoridades podem tomar medidas preventivas para minimizar os danos causados por eventos climáticos extremos. Por exemplo, eles podem evacuar áreas propensas a deslizamentos de terra ou enchentes antes que ocorram, ou redirecionar o tráfego para evitar áreas perigosas.

Além disso, as empresas podem usar as previsões para planejar suas operações e minimizar interrupções. Por exemplo, as companhias aéreas podem ajustar seus horários de voo para evitar tempestades, enquanto as empresas de construção podem planejar suas atividades de acordo com as condições climáticas previstas.

Em resumo, este projeto pode ajudar a melhorar a segurança e a qualidade dos serviços prestados em áreas afetadas por chuvas intensas, ao fornecer previsões precisas e oportunas.

## Resultados

Aqui estão algumas métricas importantes para avaliar o desempenho do nosso modelo de previsão de chuvas:

- Acurácia: 0.9863567635992827
- Matriz de Confusão:

|       | Predito Não-Chuva | Predito Chuva |
|-------|-------------------|---------------|
| Verdadeiro Não-Chuva | 21710            | 388           |
| Verdadeiro Chuva     | 0                | 6341          |

Essas métricas mostram que o modelo tem uma alta acurácia na previsão de chuvas e que a maioria das previsões está correta, como pode ser visto na matriz de confusão.



## Instalação

Para usar este modelo, você precisará ter o Python 3.x instalado em seu computador. Além disso, você precisará instalar as seguintes bibliotecas:

Você pode instalar essas bibliotecas usando o seguinte comando:

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report
```

## Uso

Para usar este modelo, primeiro abra o arquivo `Rain_detector.ipynb` em um ambiente Jupyter Notebook. Certifique-se de ter todas as bibliotecas necessárias instaladas.

Em seguida, execute as células do notebook para construir e treinar o modelo. Você pode ajustar os parâmetros do modelo, como o número de épocas e o tamanho do lote, se desejar.

Depois de treinar o modelo, você pode usá-lo para fazer previsões. Você pode fazer isso executando a célula apropriada no notebook ou usando o código a seguir em seu próprio script Python:

```python
from rain_prediction import RainPredictor

# Criar uma instância do modelo
predictor = RainPredictor()

# Carregar os pesos treinados
predictor.load_weights('model_weights.h5')

# Fazer previsões
predictions = predictor.predict(test_data)
```

## Contribuindo

Se você tiver sugestões ou melhorias para este modelo, sinta-se à vontade para enviar um pull request ou abrir uma issue.

## Licença

Este projeto está licenciado sob a licença MIT.

