#!/usr/bin/env python
# coding: utf-8

# # Análise e Clusterização de Fundos de Investimento
# 
# Este notebook implementa uma análise de dados de fundos de investimento brasileiros, utilizando dados disponibilizados pela CVM (Comissão de Valores Mobiliários).

# ## Importação de Bibliotecas

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

# ## Funções Auxiliares

# Função para simular o comportamento do operador %m-% do lubridate em R
def subtract_months(date_obj, months):
    """
    Subtrai um número específico de meses de uma data, similar ao operador %m-% do lubridate em R
    """
    year = date_obj.year
    month = date_obj.month - months
    
    while month <= 0:
        year -= 1
        month += 12
    
    # Ajusta o dia para o último dia do mês se necessário
    day = min(date_obj.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
    
    return date(year, month, day)

# Função para baixar e processar um arquivo ZIP da CVM
def process_cvm_data(ref):
    """
    Baixa e processa um arquivo ZIP da CVM para uma referência específica (AAAAMM)
    """
    url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ref}.zip"
    try:
        print(f"Baixando dados para {ref}...")
        # Desabilitar a verificação SSL
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Levanta um erro para códigos de status HTTP ruins

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file_name = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f, sep=';', encoding='latin1')  # CVM usa latin1 ou utf-8
                
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

# Função para visualizar clusters (similar ao fviz_cluster do R)
def visualize_clusters(data, labels, centers=None, title="Cluster Visualization"):
    """
    Visualiza clusters usando PCA para redução de dimensionalidade
    """
    # Redução de dimensionalidade para visualização
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    
    # Criar DataFrame para plotagem
    df = pd.DataFrame({
        'x': components[:, 0],
        'y': components[:, 1],
        'cluster': labels
    })
    
    # Plotar pontos
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', s=80, alpha=0.8)
    
    # Plotar centróides se fornecidos
    if centers is not None:
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='red', marker='X', label='Centroids')
    
    plt.title(title, fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)
    plt.tight_layout()
    plt.show()

# Função para visualizar o silhouette plot (similar ao fviz_silhouette do R)
def visualize_silhouette(silhouette_vals, labels, avg_score):
    """
    Cria um silhouette plot similar ao fviz_silhouette do R
    """
    plt.figure(figsize=(12, 8))
    
    y_lower, y_upper = 0, 0
    cluster_labels = np.unique(labels)
    n_clusters = len(cluster_labels)
    
    # Criar uma paleta de cores para os clusters
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    for i, cluster in enumerate(cluster_labels):
        # Silhouette scores para o cluster atual
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper += size_cluster_i
        
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0, 
                 edgecolor='none', color=colors[i])
        
        # Adicionar rótulo do cluster
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
        
        y_lower += size_cluster_i
    
    # Adicionar linha vertical para o valor médio do silhouette
    plt.axvline(x=avg_score, color="red", linestyle="--", label=f"Average: {avg_score:.3f}")
    
    plt.yticks([])  # Esconder os ticks do eixo y
    plt.xlabel("Silhouette Coefficient", fontsize=12)
    plt.ylabel("Cluster", fontsize=12)
    plt.title("Silhouette Plot", fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

# Desabilitar avisos de segurança para requests sem verificação SSL
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ## Coleta e Processamento de Dados

# Usa a data do sistema como referência
data_final = date.today()

# Cria uma sequência de 24 meses, terminando no mês atual
# Simulando o comportamento do lubridate em R
datas = []
for i in range(23, -1, -1):
    datas.append(subtract_months(data_final, i))

# Converte de "yyyy-mm-dd" para "yyyymm"
refs = [d.strftime("%Y%m") for d in datas]

# Lista para guardar os arquivos csv
dados_lista = []

# Looping para baixar e processar os arquivos
for ref in refs:
    df = process_cvm_data(ref)
    if df is not None:
        dados_lista.append(df)

# Verificar se temos dados para processar
if not dados_lista:
    print("Nenhum dado foi baixado. Tentando usar os arquivos locais...")
    # Tentar usar os arquivos locais como fallback
    for ref in refs:
        for base_path in ["Informes Diários de Fundos de Investimentos", "projeto/dados_cvm"]:
            for path_pattern in [
                f"{base_path}/inf_diario_fi_{ref}/inf_diario_fi_{ref}.csv",
                f"{base_path}/1/inf_diario_fi_{ref}/inf_diario_fi_{ref}.csv",
                f"{base_path}/Infos 09-2023 08-2025/inf_diario_fi_{ref}.csv",
                f"{base_path}/inf_diario_fi_{ref}.csv"
            ]:
                if os.path.exists(path_pattern):
                    try:
                        print(f"Processando dados locais para {ref} de {path_pattern}...")
                        df = pd.read_csv(path_pattern, sep=';', encoding='latin1')
                        
                        # Tratando os meses em que as colunas estão com nomes diferentes
                        if 'TP_FUNDO_CLASSE' in df.columns:
                            df.rename(columns={'TP_FUNDO_CLASSE': 'TP_FUNDO'}, inplace=True)
                        if 'CNPJ_FUNDO_CLASSE' in df.columns:
                            df.rename(columns={'CNPJ_FUNDO_CLASSE': 'CNPJ_FUNDO'}, inplace=True)
                        
                        df['referencia'] = ref
                        dados_lista.append(df)
                        break
                    except Exception as e:
                        print(f"Erro ao processar {path_pattern}: {e}")

# Verificar novamente se temos dados para processar
if not dados_lista:
    print("Não foi possível obter dados. Verifique sua conexão ou os arquivos locais.")
    exit(1)

# Empilhando os dados da lista (equivalente ao bind_rows do dplyr)
dados_completos = pd.concat(dados_lista, ignore_index=True)

# ## Estatísticas da Base de Dados Completa

# Quantidade de fundos na base em todo o período (sem filtrar)
n_fundos_total = dados_completos['CNPJ_FUNDO'].nunique()
print(f"Quantidade de fundos na base completa: {n_fundos_total}")

# ## Filtragem da Base de Dados

# Transformar a coluna de data em datetime para conseguir manipular
dados_completos['DT_COMPTC'] = pd.to_datetime(dados_completos['DT_COMPTC'])

# Calcular a data de corte = 365 dias corridos a menos
data_corte = dados_completos['DT_COMPTC'].max() - pd.Timedelta(days=365)

# Fundos que possuem pelo menos 252 observações (dias úteis) desde a data de corte
# Contando apenas valores não-NA, similar ao R (sum(!is.na(VL_QUOTA)))
cnpjs_filtrados1 = dados_completos[dados_completos['DT_COMPTC'] >= data_corte].groupby('CNPJ_FUNDO')['VL_QUOTA'].apply(lambda x: x.notna().sum())
cnpjs_filtrados1 = cnpjs_filtrados1[cnpjs_filtrados1 >= 252].index.tolist()

# Filtrando CNPJs com mais de 100 cotistas (retira fundos exclusivos e potencialmente fechados)
# Calculando a média ignorando valores NA, similar ao R (mean(NR_COTST, na.rm = TRUE))
cnpjs_filtrados2 = dados_completos.groupby('CNPJ_FUNDO')['NR_COTST'].apply(lambda x: x.mean(skipna=True))
cnpjs_filtrados2 = cnpjs_filtrados2[cnpjs_filtrados2 > 100].index.tolist()

# Aplicando os filtros
dados_filtrados = dados_completos[
    (dados_completos['CNPJ_FUNDO'].isin(cnpjs_filtrados1)) &
    (dados_completos['CNPJ_FUNDO'].isin(cnpjs_filtrados2))
].copy()

# Quantidade de fundos na base filtrada
n_fundos_filtrados = dados_filtrados['CNPJ_FUNDO'].nunique()
print(f"Quantidade de fundos na base filtrada: {n_fundos_filtrados}")

# ## Cálculo de Indicadores

# Função auxiliar para calcular retornos e volatilidades por período
def calcular_indicadores(grupo):
    """
    Calcula indicadores para um grupo (CNPJ) específico
    """
    # Ordenar por data
    grupo = grupo.sort_values('DT_COMPTC')
    
    # Calcular retornos diários
    retornos = grupo['VL_QUOTA'].pct_change()
    
    # Data máxima no grupo
    data_max = grupo['DT_COMPTC'].max()
    
    # Datas de corte para diferentes períodos
    data_1m = subtract_months(data_max.date(), 1)
    data_1a = subtract_months(data_max.date(), 12)
    
    # Filtros para diferentes períodos
    filtro_1m = grupo['DT_COMPTC'] >= pd.Timestamp(data_1m)
    filtro_1a = grupo['DT_COMPTC'] >= pd.Timestamp(data_1a)
    
    # Valores para cálculo de retornos
    ultimo_valor = grupo['VL_QUOTA'].iloc[-1]
    
    # Primeiro valor para cada período (com tratamento de NA)
    primeiro_valor_1m = grupo.loc[filtro_1m, 'VL_QUOTA'].dropna().iloc[0] if not grupo.loc[filtro_1m].empty else np.nan
    primeiro_valor_1a = grupo.loc[filtro_1a, 'VL_QUOTA'].dropna().iloc[0] if not grupo.loc[filtro_1a].empty else np.nan
    primeiro_valor_total = grupo['VL_QUOTA'].dropna().iloc[0]
    
    # Cálculo de retornos
    retorno_1m = 100 * ((ultimo_valor / primeiro_valor_1m) - 1) if not np.isnan(primeiro_valor_1m) else np.nan
    retorno_1a = 100 * ((ultimo_valor / primeiro_valor_1a) - 1) if not np.isnan(primeiro_valor_1a) else np.nan
    retorno_2a = 100 * ((ultimo_valor / primeiro_valor_total) - 1)
    
    # Cálculo de volatilidades
    vol_1m = retornos[filtro_1m].std(skipna=True) * np.sqrt(252) if not grupo.loc[filtro_1m].empty else np.nan
    vol_1a = retornos[filtro_1a].std(skipna=True) * np.sqrt(252) if not grupo.loc[filtro_1a].empty else np.nan
    vol_2a = retornos.std(skipna=True) * np.sqrt(2 * 252)
    
    return pd.Series({
        'retorno_diario_medio': 100 * retornos.mean(skipna=True),
        'retorno_1m': retorno_1m,
        'retorno_1a': retorno_1a,
        'retorno_2a': retorno_2a,
        'vol_1m': vol_1m,
        'vol_1a': vol_1a,
        'vol_2a': vol_2a,
        'pl_medio': grupo['VL_PATRIM_LIQ'].mean(skipna=True)
    })

# Calcular indicadores para cada CNPJ
indicadores = dados_filtrados.groupby('CNPJ_FUNDO').apply(calcular_indicadores).reset_index()

# Exibir estatísticas dos indicadores
print("\nResumo dos indicadores:")
print(indicadores.describe())

# Fundo com maior retorno no último mês
print("\nFundo com maior retorno no último mês:")
print(indicadores.sort_values('retorno_1m', ascending=False).head(1))

# Fundo com menor retorno no último mês
print("\nFundo com menor retorno no último mês:")
print(indicadores.sort_values('retorno_1m').head(1))

# Fundo com menor volatilidade no último ano
print("\nFundo com menor volatilidade no último ano:")
print(indicadores.sort_values('vol_1a').head(1))

# Fundo com maior volatilidade no último ano
print("\nFundo com maior volatilidade no último ano:")
print(indicadores.sort_values('vol_1a', ascending=False).head(1))

# ## Clusterização

# Remover a coluna retorno_diario_medio para clusterização
indicadores2 = indicadores.drop(columns=['retorno_diario_medio'])

# Selecionar apenas as colunas numéricas dos indicadores
dados_cluster = indicadores2[['retorno_1m', 'retorno_1a', 'retorno_2a',
                             'vol_1m', 'vol_1a', 'vol_2a', 'pl_medio']]

# Padronizar as variáveis (média = 0, desvio padrão = 1)
# Primeiro escala os dados e depois substitui NA por 0, similar ao R
scaler = StandardScaler()
dados_cluster_scaled = scaler.fit_transform(dados_cluster)
dados_cluster_scaled = np.nan_to_num(dados_cluster_scaled)  # Substitui NaN por 0 após escalar

# ### K-means

# VALIDAÇÃO: ELBOW
plt.figure(figsize=(10, 6))
wss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='random', max_iter=300, n_init=25, random_state=123)
    kmeans.fit(dados_cluster_scaled)
    wss.append(kmeans.inertia_)

plt.plot(K_range, wss, marker='o')
plt.title('Método do Cotovelo (Elbow Method)', fontsize=16)
plt.xlabel('Número de Clusters', fontsize=12)
plt.ylabel('WSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.xticks(K_range)
plt.grid(True)
plt.savefig('elbow_method.png')  # Salvar o gráfico como imagem
plt.show()

# Supondo que o "cotovelo" mostrou k = 4
np.random.seed(123)  # Garante reprodutibilidade
kmeans_result = KMeans(n_clusters=4, init='random', max_iter=300, n_init=25, random_state=123)
kmeans_result.fit(dados_cluster_scaled)

# VALIDAÇÃO: SILHOUETTE
# Calcula silhouette
silhouette_vals = silhouette_samples(dados_cluster_scaled, kmeans_result.labels_)
avg_silhouette = silhouette_score(dados_cluster_scaled, kmeans_result.labels_)
print(f"\nValor médio do silhouette: {avg_silhouette:.3f}")

# Visualiza o silhouette plot
visualize_silhouette(silhouette_vals, kmeans_result.labels_, avg_silhouette)
plt.savefig('silhouette_plot.png')  # Salvar o gráfico como imagem

# Adiciona o cluster no objeto indicadores
indicadores['cluster'] = kmeans_result.labels_.astype(str)

# Visualiza os clusters em 2D (redução com PCA)
visualize_clusters(dados_cluster_scaled, kmeans_result.labels_, 
                  centers=kmeans_result.cluster_centers_,
                  title="Visualização dos Clusters (K-means)")
plt.savefig('kmeans_clusters.png')  # Salvar o gráfico como imagem

# Quantos fundos em cada cluster
cluster_counts = indicadores['cluster'].value_counts().sort_index()
print("\nQuantidade de fundos por cluster:")
print(cluster_counts)

# Estatísticas médias por cluster
perfil_clusters = indicadores.groupby('cluster').agg(
    retorno_medio=('retorno_diario_medio', lambda x: x.mean(skipna=True)),
    retorno_1m=('retorno_1m', lambda x: x.mean(skipna=True)),
    retorno_1a=('retorno_1a', lambda x: x.mean(skipna=True)),
    vol_1a=('vol_1a', lambda x: x.mean(skipna=True)),
    pl_medio=('pl_medio', lambda x: x.mean(skipna=True)),
    n_fundos=('CNPJ_FUNDO', 'count')
).reset_index()

print("\nPerfil dos clusters:")
print(perfil_clusters)

# ### DBSCAN

# Importar DBSCAN
from sklearn.cluster import DBSCAN

# Aplicar DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(dados_cluster_scaled)
indicadores['cluster_dbscan'] = db.labels_.astype(str)

# Visualizar clusters do DBSCAN
visualize_clusters(dados_cluster_scaled, db.labels_, title="Visualização dos Clusters (DBSCAN)")
plt.savefig('dbscan_clusters.png')  # Salvar o gráfico como imagem

# Quantidade de fundos por cluster no DBSCAN
dbscan_counts = indicadores['cluster_dbscan'].value_counts().sort_index()
print("\nQuantidade de fundos por cluster (DBSCAN):")
print(dbscan_counts)

print("\nProcessamento e Clusterização concluídos em Python!")
print("\nOs gráficos foram salvos como arquivos PNG no diretório atual.")