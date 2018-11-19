import matplotlib.pyplot as plt

graph_style = [':', '-', '--', '--']
graph_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
graph_makers = ['o', '^', 's', 'D']


def graphics_generate(results_df):
    for model in results_df['model'].unique().tolist():
        for metric in results_df['metrics'].unique().tolist():
            plt.figure()
            plt.grid(True)
            plt.xlabel('Rodada')
            plt.ylabel('Valor')
            for algorithm, style, colors, makers in zip(results_df['algorithm'].unique().tolist(), graph_style,
                                                        graph_colors, graph_makers):
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
            # plt.legend(loc='best')
            lgd = plt.legend(loc=4, bbox_to_anchor=(0.5, -0.1), ncol=2)
            plt.xticks(sorted(results_df['round'].unique().tolist()))
            plt.savefig(
                model
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
