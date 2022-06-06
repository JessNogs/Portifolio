#                           Mini Projeto 3

#  Prevendo a inadimplencia dos clientes com Machine Leanrning e Power BI

#Definindo a pasta de trabalho
setwd("C:/Users/jessica.gsilva/OneDrive - Grupo Aguas do Brasil/Área de Trabalho/PowerBI/Cap_15")
getwd()


#Instalando pacotes para o projeto
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

#Carregando os pacotes
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)

#Carregando o dataset
#Fonte: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
dados_clientes <- read.csv("dados/dataset.csv")

#Visualizando os dados e sua estrutura
View(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

#Análise Exploratória, Limpeza e Transformação de Dados
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

#Renomeando a coluna de classe
colnames(dados_clientes)   #ver o nome das colunas
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
View(dados_clientes)

##Verificando valores ausentes e removendo do dataset##

sapply(dados_clientes, function(x) sum(is.na(x)))  
#sapply - passa uma "lupa" pelos dados / sum (is.na) soma valores iguais a NA (nulos)

missmap(dados_clientes, main = "Valores Missing Obervados") 
#Missmap - mapa para verficar valores ausentes - pacote Amelia

#Se houvessem valores ausentes, para omiti-los poderiamos usar:

dados_clientes <- na.omit(dados_clientes)


##Convertendo os atributos genero, escolaridade, estado civil e idade para fatores (categorias)##

# 1- Renomeando as colunas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
View(dados_clientes)

# Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)  
#é reconhecido como um número inteiro, mas na vdd é categorico
#Da documentação: 1-homem 2-mulher

?cut  #Fução que transforma variavel numerica em categorica
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels = c("Masculino", 
                                        "Feminino"))
View(dados_clientes)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

# Escolaridade
View(dados_clientes$Escolaridade)

dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c("Pós-graduação", "Graduação", "Ensino Médio", "Outros"))
View(dados_clientes)
summary(dados_clientes$Escolaridade)
#Como não tinha só 1,2,3 e 4, gerou valores NA

# Estado Civil
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil,
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido","Casado", "Solteiro", "Outros"))
View(dados_clientes)
summary(dados_clientes$Estado_Civil)
#Como ia gerar valores NA, foi adotado uma estratégia para considerar esses valores como Desconhecido

# Idade
dados_clientes$Idade <- cut(dados_clientes$Idade,
                            c(0,30,50,100),
                            labels = c("Jovem", "Adulto", "Idoso"))
View(dados_clientes)
summary(dados_clientes$Idade)


##Convertendo a variável que indica pagamento##
# Usa-se a função as.factor pois não vamos alterar o valor das variáveis e sim apenas mudá-las para categoricas

dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#Dataset após conversões
str(dados_clientes)
sapply(dados_clientes, function (x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
?missmap
dados_clientes <-na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)
View(dados_clientes)
sapply(dados_clientes, function (x) sum(is.na(x)))

##Alterando a variavel dependente para o tipo fator (categórico)
str(dados_clientes$inadimplente)
dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente)
str(dados_clientes$inadimplente)
View(dados_clientes)

##Total de inadimplentes vs não-inadimplentes

table(dados_clientes$inadimplente)

##Porcentagem entre as classes
?prop.table
prop.table(table(dados_clientes$inadimplente))

## Plot da distribuição usando ggplot2

qplot(inadimplente, data = dados_clientes, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## Set seed
set.seed(12345)
?set.seed

## Amostragem Estratificada
#Seleciona as linhas de acordo com a variável inadimplente como strata
# Ou seja, cria uma porção de dados retirados dos dados originais, nesse caso 75%
# pois p = 0.75. 'List = false' serve apenas para visualizar em forma de matriz.
?createDataPartition
indice <- createDataPartition(dados_clientes$inadimplente, p =0.75, list = FALSE)
dim(indice)

## Definimos os dados de treinamento como subconjunto do conjunto de dados original com números
# de indice de linha (conforme identificado acima) e todas as colunas

# Ou seja, cria-se o conjunto de dados dados_treino que contém a porção de dados aleatórios (indice)
# [indice,] - número de linhas = indice e coluna vazio para selecionar todas.

dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$inadimplente)

#Porcentagem entre classes
prop.table(table(dados_treino$inadimplente))

#Número de registros no dataset de treinamento
dim(dados_treino)

# Comparamos as porcentagens entre classes de treinamento e dados originais

#A proporção deve-se manter igual a original

compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dados_clientes$inadimplente)))

colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas - apenas para visualização
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver a distribuição do treinamento vs original - para visualização
ggplot(melt_compara_dados, aes(x = X1, y = value)) + 
  geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que não está no dataset de treinamento está no dataset de teste. Observe o sinal - (menos)
# Cria-se o conjunto de dados de teste, com o restante dos dados originais, ou seja os 25%
# que sobraram.

dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)

#################### Modelo de Machine Learning ####################

# Construindo a primeira versão do modelo

#Na formula: o ~ representa uma formula: a esquerda é  coluna que quero prever
# E a direita, todas as variaveis preditoas. 
#O ponto final representa todas as variaveis preditoras, para não ter que digitar uma por uma


?randomForest
View(dados_treino)
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste - faz previsoes com dados de TESTE para não repetir dados
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix - avalia o modelo, dando sua acurácia e outras métricas
#Compara o que foi previsto com o real, mostrando o quanto o modelo errou
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
# --> Serve apenas para avaliar melhor o modelo

y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Balanceamento de classe - necessario para que o modelo aprenda por igual as classes
#Pode ser aplicado técnicas como:
#over-sampling: cria dados artificiais para as classes que possuem menos dados;
#under-sampling: diminui dados da classe que possui mais dados
#Nesse caso será aplicado over-sampling para a classe 1, que possui menos dados.


install.packages("DMwR")
library(DMwR)
?SMOTE

# Aplicando o SMOTE - SMOTE: Synthetic Minority Over-sampling Technique
# https://arxiv.org/pdf/1106.1813.pdf
table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))

#Usa set.sed pois é um processo randomico e essa função permite reproduzir esse
# processo na máquina
set.seed(9560)

#utiliza-se a função SOMTE para realizar o balanceamento das classes
# O sinal ~. significa tudo que vem antes de inadimplente (variaveis preditoras)
#Gravar os dados em outro lugar (dados_treni_bal) para não perder os dados de treino originais.
dados_treino_bal <- SMOTE(inadimplente ~ ., data  = dados_treino) 

table(dados_treino_bal$inadimplente)
prop.table(table(dados_treino_bal$inadimplente))

#O resultado não precisa ser 50/50 mas próximo disso, pois ja esta mais balanceado.

# Construindo a segunda versão do modelo
#Única diferença é que se usa os dados balanceados
modelo_v2 <- randomForest(inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

## ATENÇÃO: Não se faz balanceamento nos dados de teste!! Pois o modelo deve funcionar para qualquer conjunto de dados.##

# Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$inadimplente, positive = "1")
cm_v2

# A acuracia do modelo diminui de 81% para 79%. No entanto ele tem maior equilíbrio
# Apesar do modelo 1 ter maior acuracia, ele esta balanceado. A acuracia somente não diz nada, pode inclusive mascarar os dados


# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#Todas as metricas estão equilibradas agora pois o modelo acerta mais tanto a classe 0 como a 1 (0- não inadimplente / 1-inadimplente)
#No 1 modelo, acertava muito mais uma classe que a outra pois estava desbalanceado.


# Importância das variáveis preditoras para as previsões
# Para verificar quais variaveis preditoras sao mais importantes -  e não precisar usar todas
# Ex: Genero causa impacto na inadimplencia? Idade? Estado civil? Isso que queremos saber

View(dados_treino_bal)
varImpPlot(modelo_v2)

#varImpPlot mostra em grafico as variaveis mais impotantes:
#Quanto maior o valor em X (quanto mais a direita está o ponto), mais importante a variavel
#É utilizado o índice MeanDecreaseGini (uma ferramenta específica para isso)


    ##    PARA VISUALIZAR  MELHOR   ##
# Obtendo as variáveis mais importantes - para construir um grafico mais profissional
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var), 
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))

# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importância relativa das variáveis
ggplot(rankImportance, 
       aes(x = reorder(Variables, Importance), 
           y = Importance, 
           fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), 
            hjust = 0, 
            vjust = 0.55, 
            size = 4, 
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 



## Como foi mostrado que a maioria das variaveis não sao relevantes, iremos usar apenas as mais relevantes para 
#simplificar o modelo e melhorá-lo. 

## Como foi mostrado, as variaveis mais relevantes são: PAY_0, PAY_2, PAY_3,  PAY_AMT1, PAY_AMT2, PAY_5 e BILL_AMT1.

# Construindo a terceira versão do modelo apenas com as variáveis mais importantes

colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, 
                          data = dados_treino_bal)
modelo_v3

#Não tem como garantir que a versão 3 sera melhor que a 2, então é preciso testar

    ##Nunca fazer muitas modificações ao mesmo tempo##

# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$inadimplente, positive = "1")
cm_v3

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

#A acuracia e as outras métricas aumentaram pouco, mas ja é significativo.

#Nesse momento, os modelos estão na memória do computador isso significa que se fechar o RStudio, perdemos o modelo


# Salvando o modelo em disco - para não perder quando fechar
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo - para ler o modelo do disco e carregar novamente na memoria do PC
## Posso usar só o modelo a partir daqui

modelo_final <- readRDS("modelo/modelo_v3.rds")



# Previsões com novos dados de 3 clientes  #

#Não é dado de teste, são dados novos!
#Deve ter a mesma quatidade de variaveis que tem no modelo

# Dados dos clientes
PAY_0 <- c(0, 0, 0) 
PAY_2 <- c(0, 0, 0) 
PAY_3 <- c(1, 0, 0) 
PAY_AMT1 <- c(1100, 1000, 1200) 
PAY_AMT2 <- c(1500, 1300, 1150) 
PAY_5 <- c(0, 0, 0) 
BILL_AMT1 <- c(350, 420, 280) 

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)

## ERRO: os novos dados estão de um tipo diferente dos dados de treino!


# Checando os tipos de dados
str(dados_treino_bal)
str(novos_clientes)

# Os dados de PAY_0, PAY_2, PAY_3, PAY_5 devem ser do tipo fator e não número!

# Convertendo os tipos de dados  #

#Necessário converter as variaveis para fator, especificando também que deve ser no mesmo NIVEL
#(mesmas categorias) dos dados de treino

novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)

# Fim

