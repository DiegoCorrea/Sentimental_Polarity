import matplotlib.pyplot as plt

from sys_variables import GRAPH_STYLE, GRAPH_COLORS, GRAPH_MAKERS


def generate(results_df):
    """
    Gera todos os gráficos. Para qualquer modelo e todas as métricas cria um gráfico com os algoritmos nas linhas
    :param results_df: Pandas DataFrame com cinco colunas: ['round', 'model', 'algorithm', 'metric', 'value']
    """
    # Para cada modelagem de dados
    for model in results_df['model'].unique().tolist():
        # Para cada metrica usada durante a validação dos algoritmos
        for metric in results_df['metric'].unique().tolist():
            # Cria e configura gráficos
            plt.figure()
            plt.grid(True)
            plt.xlabel('Rodada')
            plt.ylabel('Valor')
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
            plt.xticks(sorted(results_df['round'].unique().tolist()))
            # Para cada algoritmo usado cria-se uma linha no gráfico com cores e formatos diferentes
            for algorithm, style, colors, makers in zip(results_df['algorithm'].unique().tolist(), GRAPH_STYLE,
                                                        GRAPH_COLORS, GRAPH_MAKERS):
                at_df = results_df[
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
            # Salva a figura com alta resolução e qualidade
            plt.savefig(
                'results/'
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
