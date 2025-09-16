# ==============================================================================
# Script 1: Importação e Tratamento de Dados de Fundos de Investimento
# ==============================================================================

# Passo 1: Importar as bibliotecas necessárias
import pandas as pd
import glob
import os

print("Iniciando o pipeline de tratamento de dados...")

# Passo 2: Definir o caminho para os arquivos CSV
# IMPORTANTE: Crie uma pasta chamada 'dados_cvm' e coloque todos os seus
# arquivos CSV baixados da CVM dentro dela.
caminho_para_arquivos = 'dados_cvm/'
padrao_arquivos = os.path.join(caminho_para_arquivos, 'inf_diario_fi_*.csv')

# Encontrar todos os arquivos que correspondem ao padrão
lista_de_arquivos = glob.glob(padrao_arquivos)

if not lista_de_arquivos:
    raise FileNotFoundError(f"Nenhum arquivo CSV encontrado no caminho: {caminho_para_arquivos}. Verifique a pasta.")

print(f"Encontrados {len(lista_de_arquivos)} arquivos para importação.")

# Passo 3: Ler e concatenar todos os arquivos CSV em um único DataFrame
# Este processo pode demorar alguns minutos e consumir memória RAM.
lista_dfs = []
for arquivo in lista_de_arquivos:
    print(f"Lendo o arquivo: {os.path.basename(arquivo)}...")
    df_temp = pd.read_csv(
        arquivo,
        sep=';',
        encoding='utf-8',
        # Otimização: ler a data já no formato correto e apenas as colunas que vamos usar
        parse_dates=['DT_COMPTC'],
        usecols=['DT_COMPTC', 'CNPJ_FUNDO', 'DENOM_SOCIAL', 'SIT', 'VL_QUOTA', 'CAPTC_DIA', 'RESG_DIA', 'NR_COTST']
    )
    lista_dfs.append(df_temp)

# Unir todos os DataFrames em um só
df_bruto = pd.concat(lista_dfs, ignore_index=True)

print(f"Dados brutos carregados. Total de {len(df_bruto)} registros.")

# Passo 4: Filtrar os fundos que atendem aos nossos critérios
print("Iniciando a filtragem dos fundos...")

# Agrupar por CNPJ para analisar as características de cada fundo
# O método .agg() permite aplicar várias funções de agregação de uma vez
caracteristicas_fundos = df_bruto.groupby('CNPJ_FUNDO').agg(
    # Conta o número de meses distintos que o fundo aparece
    meses_de_dados=('DT_COMPTC', lambda x: x.dt.to_period('M').nunique()),
    # Verifica se 'RENDA FIXA' aparece no nome (retorna True ou False)
    eh_renda_fixa=('DENOM_SOCIAL', lambda x: x.str.contains('RENDA FIXA').any()),
    # Verifica se o fundo esteve sempre em funcionamento normal
    sempre_ativo=('SIT', lambda x: (x == 'EM FUNCIONAMENTO NORMAL').all())
)

# Aplicar os filtros para selecionar os CNPJs
fundos_selecionados_cnpj = caracteristicas_fundos[
    (caracteristicas_fundos['meses_de_dados'] >= 24) &
    (caracteristicas_fundos['eh_renda_fixa'] == True) &
    (caracteristicas_fundos['sempre_ativo'] == True)
].index

print(f"Número de fundos selecionados após o filtro: {len(fundos_selecionados_cnpj)}")

# Filtrar o DataFrame original para manter apenas os fundos selecionados
df_tratado = df_bruto[df_bruto['CNPJ_FUNDO'].isin(fundos_selecionados_cnpj)].copy()

# Remover colunas que não são mais necessárias para a próxima etapa
df_tratado.drop(columns=['DENOM_SOCIAL', 'SIT'], inplace=True)

print(f"Dados tratados. Total de {len(df_tratado)} registros para a análise.")

# Passo 5: Salvar o resultado em um formato eficiente (Parquet)
# Crie uma pasta chamada 'dados_tratados' para salvar o arquivo de saída.
caminho_saida = 'dados_tratados/dados_tratados.parquet'
df_tratado.to_parquet(caminho_saida, index=False)

print(f"Pipeline concluído! Dados limpos e salvos em: {caminho_saida}")

