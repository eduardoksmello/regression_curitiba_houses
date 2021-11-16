#!/usr/bin/env python
# coding: utf-8

# # Objetivo Geral do Projeto
# 
# Simular o desenvolvimento de um sistema de coleta de dados e avaliação com o objetivo de prever o valor de um imóvel com base em suas demais características.

# # Pedidos do CEO (perguntas de negócio):
# 1. Realizar busca de imóveis no site www.portalimoveiscuritiba.com.br e salvar resultado em um ficheiro CSV.
# 2. Quais os imóveis disponíveis em Curitiba?
# 3. Destes imóveis, qual é o que possui mais quartos e qual tem mais espaços de garagem?
# 4. Quais são os imóveis mais caro e mais barato de Curitiba?
# 5. Criar uma classificação para os imóveis de Curitiba, separando-os em categorias proporcionais (baixo padrão/ alto padrão).
# 6. Criar um relatório ordenado pelo preço dos imóveis de Curitiba.
# 7. Criar uma classificação de acordo com a região/zona da cidade de Curitiba que o imóvel está localizado.
# 8. Desenvolver um sistema preditivo do preço do imóvel em função das características de quartos, garagens e região de Curitiba.

# # Procedimentos para a solução:
# 
# - Web Scraping
# - Análise Exploratória dos Dados
# - Processamento dos Dados
# - Modelo de Regressão
# - Respostas para perguntas de negócio

# # Parte 1: WEB SCRAPING

# # Objetivo específico
# 
# Realizar processo de coleta de dados estruturados da web de maneira automatizada (web scraping) de site de imóveis residenciais de Curitiba (Brasil).

# # Bibliotecas

# In[2]:


from urllib.request import urlopen, urlretrieve, Request
from urllib.error import URLError, HTTPError
import urllib.request as urllib_request
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Accent')
sns.set_style('darkgrid')

import warnings
warnings.filterwarnings("ignore")


# # Conexão com o site

# In[3]:


# atribuindo site para variável
url = "https://portalimoveiscuritiba.com.br/imoveis?nidtbxfin=1&nidtbxtpi=1&nidtagbac=&nidtbxloc="

# teste de erros
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'}

try:
    req = Request(url, headers = headers)
    response = urlopen(req, timeout=20)
    print(response.read())

except HTTPError as e:
    print(e.status, e.reason)

except URLError as e:
    print(e.reason)


# A conexão foi estabelecida sem erros.

# # Scraping

# In[4]:


# obtendo dados da HTML
response = urlopen(url, timeout=20) 
html = response.read().decode('utf-8') 

# tratamento de dados da html: eliminar espaços entre as TAGs
html = " ".join(html.split()).replace('> <', '><')

# criação do objeto BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
pages = int(soup.find('div', class_='pagerfanta').get_text().split()[-1].split('...')[-1])

# declarando variável cards que armazenará conteúdo do dataset
cards = []

# iterando todas as páginas do site
for i in range(pages):
    
    # obtendo o HTML
    response = urlopen('https://portalimoveiscuritiba.com.br/imoveis?nidtbxfin=1&nidtbxtpi=1&nidtagbac=&nidtbxloc=&page=' + str(i + 1))
    html = response.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    
    # obtendo as TAGs de interesse: todos os imóveis anunciados
    ads = soup.find('div', class_="tl-filterOuter").findAll('div', class_="row")
    
    for ad in ads:
        card = {}

        # Valor
        card['price'] = ad.find('h3', {'class': 'price'}).getText()

        # Localização
        address = ad.find('span', {'class': 'address'}).getText().split(' - ')
        card['city'] = address[0].lower()
        card['neighborhood'] = address[-1].lower()
        
        # Quartos e Garagens
        info = ad.find('ul', {'class': 'tl-meta-listed'}).get_text().split()
        card['bedrooms'] = info[1]
        card['garage'] = info[-1]
        
        # Adicionando resultado à lista cards
        cards.append(card)    


# # Ficheiro CSV

# In[5]:


# criando um DataFrame com os resultados
df = pd.DataFrame(cards)

# tratando os dados: registros ausentes no site serão padronizados para None
df['bedrooms'][df['bedrooms'] == 'Garagens'] = np.nan
df['garage'][df['garage'] == 'Garagens'] = np.nan


# In[6]:


# quantidade de cards obtidos 
len(cards)


# In[7]:


# quantidade de linhas e colunas do dataset
df.shape


# In[8]:


# tratamento e conversão da coluna 'price'
df['price'] = df['price'].replace('[R\$\.\,]', '', regex=True).astype(float)

# cortando as casas decimais de valores 
df['price'] = df['price'] / 100

# transformando em inteiros
df['price'] = df['price'].astype(int)


# In[9]:


# visualização do dataset
df


# In[10]:


# informações sobre o dataset
df.info()


# ## Pedido 1. O ficheiro 'dataset_todos_imoveis.cvs' contém os imóveis disponíveis no site.

# In[11]:


# salvando o dataset em um ficheiro CSV
df.to_csv('./dataset_todos_imoveis222.csv', sep=';', index = False, encoding = 'utf-8-sig')


# # PARTE 2: ANÁLISE EXPLORATÓRIA

# In[3]:


# carregando o dataset salvo
df = pd.read_csv('./dataset_todos_imoveis.csv', sep=';')

# visualização das primeiras linhas do dataset
df.head()


# In[4]:


# selecionando apenas imóveis de Curitiba
df.city.value_counts()


# In[5]:


# removendo imóveis que não são de Curitiba
df = df.loc[lambda df: df['city'] == 'curitiba', :]

# contagem de imóveis de Curitiba 
df.city.value_counts()


# # Tratamento dos dados nulos

# In[6]:


# verificando dados nulos no dataframe
df.isnull().sum()


# In[7]:


# criando um dataframe temporário sem linhas com quartos com valor nulo
df_temp = df[df['bedrooms'].notna()]

# média de quartos em dataframe temporário
bedrooms_mean = df_temp['bedrooms'].astype(int).mean()

# convertendo média para inteiros
bedrooms_mean = int(bedrooms_mean)
print(bedrooms_mean)


# In[8]:


# preenchendo valores nulos de quartos com média do dataframe
df['bedrooms'].fillna(bedrooms_mean, inplace =True)


# In[9]:


df.head()


# In[10]:


df.isnull().sum()


# In[11]:


df.info()


# In[12]:


df['bedrooms'] = df['bedrooms'].astype(int)


# In[13]:


# transformação da str 'garage' em float
df['garage'] = df['garage'].astype(float)

#observação de correlações
df.corr()


# In[14]:


# calculando a taxa de proporção de espaços de garagem em função do número de quartos dos imóveis
garage_rate = df['garage'].sum()/df['bedrooms'].sum()

print('A taxa de garagens por quartos do dataset é:', garage_rate)

# utilizando a taxa para substituir valores nulos
print('Os valores nulos em garagem serão preenchidos em função do número de quartos do imóvel.')
df['garage'].fillna((df['bedrooms']*garage_rate).round(), inplace = True)

print('O número de valores nulos em df é:', df.isnull().sum().sum())


# In[15]:


df.info()


# In[16]:


sns.boxplot(x='price',
            data=df,
            orient='h',
            width=0.3)


# In[17]:


df['price'].describe().round()


# ## Excluindo os outliers da variável dependente

# In[18]:


# excluindo imóveis com preço igual a zero
df = df.drop(df[df.price == 0].index)

# excluindo o imóvel com preço muito acima do resto
df = df.drop(df[df.price > 800000000].index)


# In[19]:


# organizando o index
df.reset_index(drop=True, inplace=True)
df.head()


# In[20]:


sns.boxplot(x='price',
            data=df,
            orient='h',
            width=0.3)


# In[21]:


df.describe().round()


# ## Pedido 2. Quais os imóveis disponíveis em Curitiba?

# In[22]:


# imóveis disponíveis em Curitiba
print('O dataset possui {} imóveis disponíveis em Curitiba'.format(len(df)))


# ## Pedido 3. Qual é o que possui mais quartos e qual tem mais espaços de garagem?

# In[23]:


# imóvel com mais quartos
df[df['bedrooms'] == df['bedrooms'].max()]


# In[24]:


# imóvel com mais espaços de garagem
df[df['garage'] == df['garage'].max()]


# In[25]:


print('Imóvel com maior número de quarto:', int(df['bedrooms'].max()), 'quartos.')
print('Imóvel com maior número de espaços de garagem:', int(df['garage'].max()), 'espaços.')


# ## Pedido 4. Quais são os imóveis mais caro e mais barato?

# In[26]:


# imóvel mais caro
df[df['price'] == df['price'].max()]


# In[27]:


# imóvel mais barato
df[df['price'] == df['price'].min()]


# In[28]:


print('Imóvel mais caro: R$', int(df['price'].max()))
print('Imóvel mais barato: R$', int(df['price'].min()))


# ## Pedido 5. Criar uma classificação para os imóveis de Curitiba, separando-os em categorias proporcionais (baixo padrão/ alto padrão).

# In[29]:


# mediana dos valores de imóveis
df['price'].median()


# In[30]:


df['class'] = ''
df.loc[ df['price'] > df['price'].median(), 'class'] = 'high'
df.loc[ df['price'] <= df['price'].median(), 'class'] = 'low'
df.head()


# In[31]:


print("A coluna 'class' apresenta a divisão do conjunto de imóveis de Curitiba em duas classes:\n",
      df['class'].value_counts())


# ## Pedido 6. Criar um relatório ordenado pelo preço dos imóveis de Curitiba.

# In[32]:


report_prices_ctba = df.sort_values(by = 'price',
                                   ignore_index=True)
report_prices_ctba.to_csv('./report_prices_ctba.csv',
                          sep=';', 
                          index = False,
                          encoding = 'utf-8-sig')


# In[33]:


report_prices_ctba


# ## Pedido 7. Criar uma classificação de acordo com a região/zona da cidade de Curitiba que o imóvel está localizado.

# In[34]:


# visualização do mapa de Curitiba (bairros e regiões)

from IPython.display import Image
from IPython.core.display import HTML 

Image(url= "http://www.mobilizacuritiba.org.br/files/2014/03/mapa-regionais.jpg")


# In[35]:


df.neighborhood.value_counts()


# In[36]:


# criação de uma lista com as zonas e bairros de Curitiba
# fonte: http://www.mobilizacuritiba.org.br/

neighborhood = []

with open("curitiba_neighborhood_area.txt", "r") as area_file:
    for line in area_file:
            line = line.strip()
            line = str(line).lower()
            neighborhood.append(line)
area_file.close()
neighborhood


# In[37]:


# divisão da lista em regiões de Curitiba
matriz = neighborhood[1:18:1]
portao = neighborhood[21:32:1]
bairro_novo = neighborhood[34:37:1]
boa_vista = neighborhood[39:52:1]
cic = neighborhood[54:58:1]
santa_felicidade = neighborhood[62:75:1]
boqueirao = neighborhood[78:82:1]
pinheirinho = neighborhood[85:89:1]
cajuru = neighborhood[94:99:1]


# In[38]:


# criação de dicionário (bairro: região)
region_curitiba = {}

# função que atribui aos bairros o nome de sua região
def add_region (region, name):
    for i in region:
        region_curitiba[i] = name


# In[39]:


# adicionando as regiões no dicionário
add_region(matriz, 'matriz')
add_region(portao, 'portao')
add_region(bairro_novo, 'bairro novo')
add_region(boa_vista, 'boa vista')
add_region(cic, 'cic')
add_region(santa_felicidade, 'santa felicidade')
add_region(boqueirao, 'boqueirao')
add_region(pinheirinho, 'pinheirinho')
add_region(cajuru, 'cajuru')


# In[40]:


region_curitiba['ecoville'] = 'santa felicidade'
region_curitiba['cidade industrial de curitiba'] = 'cic'
region_curitiba['champagnat'] = 'matriz'
region_curitiba['alto da rua xv'] = 'matriz'
region_curitiba['são francisco'] = 'matriz'
region_curitiba['neoville'] = 'portao'
region_curitiba['barigui'] = 'cic'
region_curitiba['campo santana'] = 'pinheirinho'
region_curitiba['tatuquara'] = 'pinheirinho'
region_curitiba['alto da gloria/juvevê'] = 'matriz'
region_curitiba['bastel'] = 'matriz'
region_curitiba['caiuá'] = 'portao'


# In[41]:


# dicionário de bairros e regiões de Curitiba
region_curitiba


# In[42]:


df['region'] = df['neighborhood']


# In[43]:


df['region'] = df['neighborhood'].map(region_curitiba)


# In[44]:


df


# In[45]:


print('Os imóveis podem ser divididos em função da região da cidade:')

df['region'].value_counts()


# In[69]:


# estatísticas descritivas de cada região
region_group = df.groupby('region')
region_group['price'].describe().round(2)


# ## Pedido 8. Desenvolver um sistema preditivo do preço do imóvel em função das características de quartos, garagens e região de Curitiba.

# In[46]:


df.info()


# In[50]:


# análise descritiva
df.describe().round(2)


# In[54]:


ax = sns.pairplot(df, kind='reg')
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1)
ax


# ### Exclusão de outlier da coluna garage

# In[101]:


df = df.drop(df[(df.garage>10)].index)


# In[102]:


df.garage.describe()


# # Pré-processamento dos dados

# ## Distribuição de frequências

# In[103]:


# distribuição de frequências da variável dependente
ax = sns.distplot(df['price'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Preço dos Imóveis (R$)', fontsize=16)
ax;


# A distribuição de frequência da variável dependente revela uma assimetria à direita.

# In[105]:


# distribuição de frequências das variáveis independente
plt.figure(figsize=(18, 6))

plt.subplot(1,2,1)
sns.distplot(df['bedrooms'], bins=7, kde=False)
plt.title('Histograma - Quartos')

plt.subplot(1, 2, 2)
sns.distplot(df['garage'], bins=6, kde=False)
plt.title('Histograma - Garagens');


# ### Aplicação da transformação logarítmica aos dados do dataset

# In[106]:


df['log_price'] = np.log(df['price'])
df['log_bedrooms'] = np.log(df['bedrooms'] + 1)
df['log_garage'] = np.log(df['garage'] + 1)


# In[107]:


df.head()


# In[108]:


df.describe().round(2)


# ### Distribuilção de frequências de variáveis transformadas

# In[109]:


# distribuição de frequências da variável dependente transformada (y)
ax = sns.distplot(df['log_price'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20)
ax.set_xlabel('Preço dos Imóveis (R$)', fontsize=16)
ax;


# ### Verificando relação linear entre VD e VI's

# In[110]:


ax = sns.pairplot(df,
                  y_vars='log_price',
                  x_vars=['log_bedrooms', 'log_garage'],
                  height=5)
ax.fig.suptitle('Dispersão entre as Variáveis Transformadas',
                fontsize=20, y=1.05)
ax;


# # Estimando um Modelo de Regressão Linear para o Preço

# In[123]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm


# In[112]:


# criação de dataset de treino e teste
y = df['log_price']
X = df[['log_bedrooms', 'log_garage']]


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)


# In[125]:


X_train_constant = sm.add_constant(X_train)

model_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst = True).fit()


# In[126]:


print(model_statsmodels.summary())


# In[127]:


# criação da função de one-hot encoder
def one_hot_encoder(dataset, columns, drop_first=False):
    dataset = pd.get_dummies(dataset, 
                             columns=columns,
                             drop_first=drop_first)
    return dataset


# In[136]:


# transformação das variáveis categóricas
df = one_hot_encoder(df, 
                     ['region'], 
                     drop_first=False)
df


# In[138]:


df_final = df.drop(columns=['price', 'city', 'neighborhood',
                            'bedrooms', 'garage', 'class'])
df_final.head()


# In[139]:


# criação de dataset de treino e teste
y = df_final['log_price']
X = df_final.drop(columns=['log_price'])


# In[140]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10)

X_train_constant = sm.add_constant(X_train)

model_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst = True).fit()

print(model_statsmodels.summary())


# In[142]:


model = LinearRegression()


# In[143]:


model.fit(X_train, y_train)


# In[144]:


print('R² = {}'.format(model.score(X_train, y_train).round(3)))


# In[145]:


y_previsto = model.predict(X_test)


# In[146]:


print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(3))


# In[148]:


model.intercept_


# In[149]:


np.exp(model.intercept_)


# In[150]:


model.coef_


# In[151]:


X.columns


# In[152]:


index = ['Intercepto', 'log_bedrooms', 'log_garage', 'region_bairro novo', 'region_boa vista',
       'region_boqueirao', 'region_cajuru', 'region_cic', 'region_matriz',
       'region_pinheirinho', 'region_portao', 'region_santa felicidade']


# In[153]:


pd.DataFrame(data=np.append(model.intercept_, model.coef_),
            index=index, columns=['Parâmetros'])


# ## Interpretação dos Coeficientes Estimados
# 
# <p style='font-size: 20px; line-height: 2; margin: 10px 50px; text-align: justify;'>
# <b>Intercepto</b> → Excluindo o efeito das variáveis explicativas ($X_2=X_3=0$) o efeito médio no Preço dos Imóveis seria de <b>R$ 69.356,02
# </b> (exp[11.147]).
# </p>
# 
# <p style='font-size: 20px; line-height: 2; margin: 10px 50px; text-align: justify;'>
# <b>Quartos</b> → Mantendo-se o valor das outras variáveis constante, um acréscimo em quartos gera, em média, um acréscimo de <b>0.43%</b> no Preço do Imóvel.
# </p>
# 
# <p style='font-size: 20px; line-height: 2; margin: 10px 50px; text-align: justify;'>
# <b>Garagem</b> →  Mantendo-se o valor das outras variáveis constante, um acréscimo em garagem gera, em média, um acréscimo de <b>1.49%</b> no Preço do Imóvel.
# </p>

# In[154]:


y_previsto_train = model.predict(X_train)


# In[155]:


ax = sns.scatterplot(x=y_previsto_train, 
                     y=y_train)
ax.figure.set_size_inches(12, 6)
ax.set_title('Previsão X Real', fontsize=18)
ax.set_xlabel('log do Preço - Previsão', fontsize=14)
ax.set_ylabel('log do Preço - Real', fontsize=14)
ax


# In[156]:


residuo = y_train - y_previsto_train


# In[157]:


ax = sns.distplot(residuo)
ax.figure.set_size_inches(12, 6)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize=18)
ax.set_xlabel('log do Preço', fontsize=14)
ax;

