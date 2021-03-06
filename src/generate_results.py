import matplotlib.pyplot as plt

from sys_variables import GRAPH_STYLE, GRAPH_COLORS, GRAPH_MAKERS


def graphics(results_df):
    """
    Gera todos os gráficos. Para qualquer modelo e todas as métricas cria um gráfico com os algoritmos nas linhas
    :param results_df: Pandas DataFrame com cinco colunas: ['round', 'model', 'algorithm', 'metric', 'value']
    """
    for config in results_df['config'].unique().tolist():
        # Para cada modelagem de dados
        for model in results_df['model'].unique().tolist():
            # Para cada metrica usada durante a validação dos algoritmos
            for metric in results_df['metric'].unique().tolist():
                # Cria e configura gráficos
                plt.figure()
                plt.grid(True)
                plt.xlabel('Rodada')
                plt.ylabel('Valor')
                results_df_by_filter = results_df[
                    (results_df['config'] == config) &
                    (results_df['model'] == model) &
                    (results_df['metric'] == metric)]
                # Para cada algoritmo usado cria-se uma linha no gráfico com cores e formatos diferentes
                for algorithm, style, colors, makers in zip(results_df_by_filter['algorithm'].unique().tolist(),
                                                            GRAPH_STYLE,
                                                            GRAPH_COLORS, GRAPH_MAKERS):
                    at_df = results_df[
                        (results_df['config'] == config) &
                        (results_df['algorithm'] == algorithm) &
                        (results_df['model'] == model) &
                        (results_df['metric'] == metric)]
                    at_df.sort_values("round")
                    plt.plot(
                        at_df['round'],
                        at_df['value'],
                        linestyle=style,
                        color=colors,
                        marker=makers,
                        label=algorithm
                    )
                # Configura legenda
                lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
                plt.xticks(sorted(results_df['round'].unique().tolist()))
                # Salva a figura com alta resolução e qualidade
                plt.savefig(
                    'results/'
                    + str(config)
                    + '_'
                    + model
                    + '_'
                    + metric
                    + '.png',
                    format='png',
                    dpi=1000,
                    quality=100,
                    bbox_extra_artists=(lgd,),
                    bbox_inches='tight'
                )
                plt.close()


def comparate(results_df):
    """
    :param results_df: Pandas DataFrame com seis colunas: ['round', 'config', 'model', 'algorithm', 'metric', 'value']
    """
    for config in results_df['config'].unique().tolist():
        # Para cada modelagem de dados
        for model in results_df['model'].unique().tolist():
            # Para cada metrica usada durante a validação dos algoritmos
            for metric in results_df['metric'].unique().tolist():
                results_df_by_filter = results_df[
                    (results_df['config'] == config) &
                    (results_df['model'] == model) &
                    (results_df['metric'] == metric)]
                # Para cada algoritmo usado
                for algorithm in results_df_by_filter['algorithm'].unique().tolist():
                    at_df = results_df[
                        (results_df['config'] == config) &
                        (results_df['algorithm'] == algorithm) &
                        (results_df['model'] == model) &
                        (results_df['metric'] == metric)]
                    print("Config; ", str(config), "\t| Algoritmo: ", str(algorithm), "\t| Model: ", str(model),
                          "\t| Metrica: ", str(metric), "RESULT; ", str(at_df['value'].sum() / at_df['value'].count()))
