import streamlit as st
import pydeck as pdk
import base64

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

import random

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

def create_map_deck(leads):
  leads = leads.head(30)
  vermelho = "https://img.icons8.com/plasticine/100/000000/marker.png"
  azul = "https://img.icons8.com/ultraviolet/40/000000/marker.png"

  icon_data = {
    "url": azul,
    "width": 128,
    "height":128,
    "anchorY": 128
  }
 
  leads['icon_data']= None
  for i in leads.index:
    leads['lat'][i] = leads['lat'][i] + random.uniform(-0.1, 0.1)
    leads['lng'][i] = leads['lng'][i] + random.uniform(-0.1, 0.1)
    leads['icon_data'][i] = icon_data
  
  for i in range(0, leads.shape[0], 5):
    leads['icon_data'][i] = {"url": vermelho,"width": 128,"height":128,"anchorY": 128}

  icon_layer = pdk.Layer(
      type='IconLayer',
      data=leads,
      get_position='[lng, lat]',
      get_icon='icon_data',
      get_size=4,
      pickable=True,
      size_scale=15
  )
  
  mapa = pdk.Deck(
      map_style='mapbox://styles/mapbox/light-v9',
          initial_view_state=pdk.ViewState(
              latitude=-16.1237611,
              longitude=-59.9219642,
              zoom=3,
          ),
          layers=[icon_layer]
      )
  return mapa

def get_table_download_link(df):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()
  href = f'<a href="data:file/csv;base64,{b64}" download="leads.csv">Download arquivo de sugestões</a>'
  #href = f'<a href="data:file/csv;base64,{b64}">Download arquivo de sugestões</a>'
  return href

def main():
    st.image('logo.png', width=400)
    st.title('AceleraDev Data Science - Projeto Final')
    st.subheader('Ronaldo Regis Posser - Sistema de recomendação de novos clientes')
    
    
    label_enc = LabelEncoder()

    base = pd.read_csv('base.zip')
    base['id'] = label_enc.fit_transform(base['id'])
    
    st.subheader('Visualização base de dados')

    st.markdown('A base possui ' + str(base.shape[0]) + ' empresas, com ' + str(base.shape[1]) + ' variáveis que diferenciam as empresas por faturamento, ramo de atividade e localização.')
    st.dataframe(base.head())

    #Engenharia de features
    base.set_index(['id'], inplace=True)

    features_cat = ['sg_uf',	'setor', 'idade_emp_cat', 'nm_divisao', 'de_saude_tributaria',
                'de_saude_rescencia', 'de_nivel_atividade', 'nm_meso_regiao',
                'nm_micro_regiao', 'de_faixa_faturamento_estimado']
    base_dummies = pd.get_dummies(base, columns=features_cat)

    #Treinamento modelo
    qtd_neighbors = 5
    model = NearestNeighbors(n_neighbors=qtd_neighbors, metric = 'cosine')



if __name__ == '__main__':
    main()
