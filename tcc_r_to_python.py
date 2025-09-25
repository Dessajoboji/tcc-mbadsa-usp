#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Função para baixar e processar um arquivo ZIP da CVM
def process_cvm_data(ref):
    url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ref}.zip"
    try:
        response = requests.get(url)
        response.raise_for_status() # Levanta um erro para códigos de status HTTP ruins

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file_name = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f, sep=';', encoding='latin1') # CVM usa latin1 ou utf-8
                
                # Tratando os meses em que as colunas estão com nomes diferentes
                if 'TP_FUNDO_CLASSE' in df.columns:
                    df.rename(columns={'TP_FUNDO_CLASSE': 'TP_FUNDO'}, inplace=True)
                if 'CNPJ_FUNDO_CLASSE' in df.columns:
                    df.rename(columns={'CNPJ_FUNDO_CLASSE': 'CNPJ_FUNDO'}, inplace=True)
                
                df['referencia'] = ref
                return df
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar ou processar {url}: {e}")
        return None

# ####### COLETANDO OS DADOS MENSAIS E EMPILHANDO ################################

data_final = datetime.now()
datas = [data_final - timedelta(days=30*i) for i in range(23, -1, -1)] # 24 meses
refs = [d.strftime("%Y%m") for d in datas]

dados_lista = []
for ref in refs:
    df_mes = process_cvm_data(ref)
    if df_mes is not None:
        dados_lista.append(df_mes)

dados_completos = pd.concat(dados_lista, ignore_index=True)

# ####### ESTATÍSTICAS DA BASE DE DADOS COMPLETA #################################

# quantidade de fundos na base em todo o período (sem filtrar)
# print(f"Quantidade de fundos na base completa: {dados_completos['CNPJ_FUNDO'].nunique()}")

# ####### FILTRANDO A BASE DE DADOS ##############################################

# coluna data está com classe de caracteres, deve-se transformar em data
dados_completos['DT_COMPTC'] = pd.to_datetime(dados_completos['DT_COMPTC'])

# calcular a data de corte = 365 dias corridos a menos
data_corte = dados_completos['DT_COMPTC'].max() - timedelta(days=365)

# fundos que possuem pelo menos 252 observações (dias úteis) desde a data de corte
cnpjs_filtrados1 = dados_completos[dados_completos['DT_COMPTC'] >= data_corte].groupby('CNPJ_FUNDO')['VL_QUOTA'].count()
cnpjs_filtrados1 = cnpjs_filtrados1[cnpjs_filtrados1 >= 252].index.tolist()

# FILTRANDO CNPJS COM MAIS DE 100 COTISTAS (RETIRA FUNDOS EXCLUSIVOS E POTENCIALMENTE FECHADOS)
cnpjs_filtrados2 = dados_completos.groupby('CNPJ_FUNDO')['NR_COTST'].mean()
cnpjs_filtrados2 = cnpjs_filtrados2[cnpjs_filtrados2 > 100].index.tolist()

# aplicando os filtros
dados_filtrados = dados_completos[
    (dados_completos['CNPJ_FUNDO'].isin(cnpjs_filtrados1)) &
    (dados_completos['CNPJ_FUNDO'].isin(cnpjs_filtrados2))
].copy()

# quantidade de fundos na base filtrada
# print(f"Quantidade de fundos na base filtrada: {dados_filtrados['CNPJ_FUNDO'].nunique()}")

# ####### CALCULANDO INDICADORES #################################################

indicadores = dados_filtrados.sort_values(by=['CNPJ_FUNDO', 'DT_COMPTC']).groupby('CNPJ_FUNDO').apply(lambda x:
    pd.Series({
        'retorno_diario_medio': x['VL_QUOTA'].pct_change().mean() * 100,
        'retorno_1m': (x['VL_QUOTA'].iloc[-1] / x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=30))]['VL_QUOTA'].iloc[0] - 1) * 100 if not x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=30))].empty else np.nan,
        'retorno_1a': (x['VL_QUOTA'].iloc[-1] / x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=365))]['VL_QUOTA'].iloc[0] - 1) * 100 if not x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=365))].empty else np.nan,
        'retorno_2a': (x['VL_QUOTA'].iloc[-1] / x['VL_QUOTA'].iloc[0] - 1) * 100,
        'vol_1m': x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=30))]['VL_QUOTA'].pct_change().std() * np.sqrt(252) if not x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=30))].empty else np.nan,
        'vol_1a': x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=365))]['VL_QUOTA'].pct_change().std() * np.sqrt(252) if not x[x['DT_COMPTC'] >= (x['DT_COMPTC'].max() - timedelta(days=365))].empty else np.nan,
        'vol_2a': x['VL_QUOTA'].pct_change().std() * np.sqrt(2 * 252),
        'pl_medio': x['VL_PATRIM_LIQ'].mean()
    })
).reset_index()

# ####### CLUSTERIZAÇÃO ##########################################################

indicadores2 = indicadores.drop(columns=['retorno_diario_medio'])

# Seleciona apenas as colunas numéricas dos indicadores
dados_cluster = indicadores2[['retorno_1m', 'retorno_1a', 'retorno_2a',
                               'vol_1m', 'vol_1a', 'vol_2a', 'pl_medio']]

# Padroniza as variáveis (média = 0, desvio padrão = 1)
scaler = StandardScaler()
dados_cluster_scaled = scaler.fit_transform(dados_cluster.fillna(0)) # Substituindo NA por 0 antes de escalar

### KMEANS

# VALIDAÇÃO: ELBOW (Para plotar, seria necessário um loop para diferentes k e calcular WSS)
# Exemplo de como calcular WSS para um dado k
# wss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(dados_cluster_scaled)
#     wss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WSS')
# plt.show()

# Supondo que o "cotovelo" mostrou k = 4
kmeans_result = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=123)
kmeans_result.fit(dados_cluster_scaled)

# VALIDAÇÃO: SILHOUETTE
sil_score = silhouette_score(dados_cluster_scaled, kmeans_result.labels_)
# print(f"Silhouette Score: {sil_score}")

# Adiciona o cluster no objeto indicadores
indicadores['cluster'] = kmeans_result.labels_

# Quantos fundos em cada cluster
# print(indicadores['cluster'].value_counts())

# Estatísticas médias por cluster
perfil_clusters = indicadores.groupby('cluster').agg(
    retorno_medio=('retorno_diario_medio', 'mean'),
    retorno_1m=('retorno_1m', 'mean'),
    retorno_1a=('retorno_1a', 'mean'),
    vol_1a=('vol_1a', 'mean'),
    pl_medio=('pl_medio', 'mean'),
    n_fundos=('CNPJ_FUNDO', 'count')
).reset_index()

# print(perfil_clusters)

### DBSCAN (Exemplo básico, pode precisar de ajuste de parâmetros)
# from sklearn.cluster import DBSCAN
# dbscan_result = DBSCAN(eps=0.5, min_samples=5).fit(dados_cluster_scaled)
# indicadores['cluster_dbscan'] = dbscan_result.labels_
# print(indicadores['cluster_dbscan'].value_counts())

print("Processamento e Clusterização concluídos em Python!")


