import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mult_stacked_bar(table: pd.crosstab, colors: list, bar_width: float=0.8,
                     legend_title: str='', counts: list=[], 
                     x_label: str='', figsize: list=[10,8], savefile: bool=False, filename: str='plot.png'):
    '''
    Creates a stacked bar plot with multiple bars

        Parameters:
            table (pd.crosstab): Crosstab of data
            colors (list):       List of colors to use for each bar
            legend_title (str):  Title for the legend
            counts (list):       List of counts to use for each bar
            x_label (str):       Label for the x-axis
            figsize (list):      Size of the figure
            savefile (bool):     Whether or not to save the plot
            filename (str):      Filename to save the plot as
    '''
    
    # set up and augment bar graph
    ax = table.plot(kind='bar', width=bar_width, stacked=True,
                    fontsize=16, rot=0, figsize=figsize,
                    color=colors)

    # augment spines, ticks, and axes
    [ax.spines[i].set_visible(False) for i in ax.spines]
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', length=0)
    plt.xlabel('')
    if x_label:
        plt.xlabel(x_label, fontsize=20, labelpad=10)

    n_rows = table.shape[0]
    n_cols = table.shape[1]

    # put relative proportions in the patches
    for i in range(n_rows):
        prev = 0
        for j in range(n_cols):
            current = table.iloc[i, j]
            ypos = current / 2 + prev
            ax.text(i, ypos, f'{current * 100:.0f}%',
                    ha='center', va='center',
                    color='white', fontsize='14', weight='bold')
            prev += current

    # put counts above patches
    if counts:
        for i in range(n_rows):
            ax.text(i, 1.025, f'n={str(counts[i])}',
                    fontsize=14,
                    ha='center')

    # set up and augment legend
    plt.legend(bbox_to_anchor=[1., 0.5], loc='center left',
            title=legend_title, title_fontsize=24, fontsize=16,
            frameon=False, markerfirst=False)

    # save the figure
    if savefile:
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

    return ax

def make_subplot_viz(values: list, labels: list, fontsize: int=16, title_fontsize: int=20, figsize: tuple=(20,8),
                     xlabel: str='', savefile: bool=False, filename: str='budget.png', 
                     colors: list=['cornflowerblue','slategrey','mediumpurple']):
    '''
    Creates boxplots of the distribution of values
    
        Parameters:
            values (list):        List of pd.Series or nd.array objects to plot
            labels (list):        List of labels to use for each bar
            fontsize (int):       Font size for the labels
            title_fontsize (int): Font size for the title
            figsize (tuple):      Figure size
            xlabel (str):         Label for the x-axis
            colors (list):        List of colors to use for each bar
            savefile (bool):      Whether or not to save the figure
            filename (str):       Name of the file to save
    '''

    # create figure and axes and set style
    f, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
    sns.set_style('whitegrid')

    for i in range(len(axes)):

        # establish numerical variables
        count_media, mean_media, min_media, p25_media, p50_media, p75_media, max_media =\
            np.array(values[i].describe().drop(index='std')).reshape(7,)

        xtickvals = [count_media, mean_media, min_media, p25_media, p50_media, p75_media, max_media]
        xtickvals = [round(val) for val in xtickvals]
        if (mean_media - p50_media) / p50_media < 0.1:
            del xtickvals[1]

        # create and configure boxplot, plot axes, and spines
        sns.boxplot(ax=axes[i], data=values[i], width=0.33, orient='h', color=colors[i], showmeans=True,
                    medianprops={'linewidth': 2.5,
                                #  'color': 'k',
                                'alpha': 1},
                    meanprops={'marker': 'o',
                            'markerfacecolor': 'white', 
                            'markeredgecolor': 'black',
                            'markersize': 10},
                    flierprops={'marker': 'o', 
                                'markerfacecolor': colors[i],
                                #  'markeredgecolor': 'black',
                                'markersize': 10, 
                                'alpha': 0.5})

        axes[i].tick_params(axis='x', size=0, labelsize=fontsize)
        axes[i].set(yticklabels=[])
        axes[i].text(-8, 0, labels[i], ha='center', va='center', 
                    fontsize=fontsize, rotation='vertical')
        axes[i].text(-20, 0, f'n={xtickvals[0]}', ha='center', va='center', 
                    fontsize=fontsize, rotation='vertical')

        [axes[i].spines[j].set_visible(False) for j in axes[i].spines]

        # display numerical values
        for val in xtickvals[1:2]:   # mean
            axes[i].text(val, -0.27, str(val), ha='center', va='center', fontsize=fontsize)
        for val in xtickvals[3:-1]:   # median, p25, p50, p75
            axes[i].text(val, -0.27, str(val), ha='center', va='center', fontsize=fontsize)
        for val in xtickvals[-1:]:   # max
            axes[i].text(val, -0.16, str(val), ha='center', va='center', fontsize=fontsize)
        if xtickvals[2] not in [n for n in range(4)]:   # min
            for val in xtickvals[2:3]:
                axes[i].text(val, -0.16, str(val), ha='center', va='center', fontsize=16)

    plt.title(xlabel, fontsize=title_fontsize, va='center', ha='center', y=-0.4)

    if savefile:
        plt.savefig(filename, bbox_inches='tight', dpi=1000)

    return f

if __name__ == '__main__':
    df = pd.read_csv('movies_film_digital.csv')

    # set all genres outside of the top 3 to to Other
    df.primary_genre = df.primary_genre.apply(lambda x: \
                        x if x in df.primary_genre.value_counts().head(3).index else 'Other')

    # rename film types to be more readable
    pd.options.mode.chained_assignment = None  # default='warn'
    df.film_type.loc[df.film_type == 'F'] = 'Film'
    df.film_type.loc[df.film_type == 'D'] = 'Digital'
    df.film_type.loc[df.film_type == 'FD'] = 'Film and Digital'

    # scale down budget values by 10^6 and remove erroneous data
    df.inflation_adjusted_budget /= 10**6
    df.inflation_adjusted_budget = df.inflation_adjusted_budget[df.inflation_adjusted_budget != 0]

    # visualize the distribution of movie media by year
    table1 = pd.crosstab(df.production_year, df.film_type, normalize='index')
    table1_nn = pd.crosstab(df.production_year, df.film_type)
    counts1 = [table1_nn.iloc[n].sum() for n in range(len(table1_nn))]

    mult_stacked_bar(table=table1, colors=['cornflowerblue', 'slategrey', 'mediumpurple'],
                    counts=counts1, legend_title='Movie Media', figsize=[20,8],
                    savefile=False, filename='Film_by_Year.png')
    plt.show()

    # visualize the distribution of movie genres by year
    table2 = pd.crosstab(df.production_year, df.primary_genre, normalize='index')
    table2_nn = pd.crosstab(df.production_year, df.primary_genre)
    counts2 = [table2_nn.iloc[n].sum() for n in range(len(table2_nn))]

    mult_stacked_bar(table=table2, colors=['cornflowerblue', 'slategrey', 'mediumpurple', 'mediumseagreen'],
                    counts=counts2, legend_title='Movie Genre', figsize=[20,8],
                    savefile=False, filename='Genre_by_Year.png')
    plt.show()

    # visualize the distribution of movie media by genre
    table3 = pd.crosstab(df.primary_genre, df.film_type, normalize='index')
    table3_nn = pd.crosstab(df.primary_genre, df.film_type)
    counts3 = [table3_nn.iloc[n].sum() for n in range(len(table3_nn))]

    mult_stacked_bar(table=table3, colors=['cornflowerblue', 'slategrey', 'mediumpurple'],
                    bar_width = 0.4, counts=counts3, legend_title='Film Media',
                    figsize=[10, 8], savefile=False, filename='Film_by_Genre.png')
    plt.show()

    f = df[['film_type', 'inflation_adjusted_budget']]
    f = f[f.film_type == 'Film']
    film = pd.DataFrame(f.inflation_adjusted_budget)
    film.columns = ['Film']

    d = df[['film_type', 'inflation_adjusted_budget']]
    d = d[d.film_type == 'Digital']
    digital = pd.DataFrame(d.inflation_adjusted_budget)
    digital.columns = ['Digital']

    fd = df[['film_type', 'inflation_adjusted_budget']]
    fd = fd[fd.film_type == 'Film and Digital']
    f_and_d = pd.DataFrame(fd.inflation_adjusted_budget)
    f_and_d.columns = ['Film and Digital']

    make_subplot_viz(values=[film, digital, f_and_d], labels=['Film', 'Digital', 'Film and Digital'],
                    xlabel='Inflation Adjusted Budget (in millions $) by Film Media',
                    savefile=False, filename='budget.png')

    plt.show()
