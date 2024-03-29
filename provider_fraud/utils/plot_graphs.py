
import plotly.graph_objs as go
import networkx as nx
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def feat_cor(df_enriched):
    '''

    :param df_enriched:
    :return:
    '''
    corr_data = df_enriched[['PotentialFraud',
                    'AttendingPhysician',
                    'Provider_degree',
    #                 'Provider_closeness_centrality',
                    'Provider_eigenvector_centrality',
                    'Provider_pagerank',
                    'AttendingPhysician_degree',
    #                 'AttendingPhysician_closeness_centrality',
                    'AttendingPhysician_eigenvector_centrality',
                    'AttendingPhysician_pagerank']]

    # Compute the correlation matrix
    corr = corr_data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('fig/heatmap.png')
    plt.show()


def plot_graphs(df, source, target):
    '''

    :return:
    '''
    # Consider subgraph for plotting
    df_plotting = df.sample(n=1_000, random_state=1)

    H = nx.from_pandas_edgelist(df_plotting, source=source, target = target)
    pos = nx.random_layout(H)

    #Create Edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#010203'),
        hoverinfo='none',
        mode='lines')

    for edge in H.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale='RdBu',
            reversescale=False,
            color=[],
            size=12,
            colorbar=dict(
                thickness=35,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in H.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    #add color to node points
    for node, adjacencies in enumerate(H.adjacency()):
        node_trace['marker']['color']+=tuple([len(adjacencies[1])])
        node_info = str(adjacencies[0])

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Network Graph of Provider & Physician \n\n',
                    titlefont=dict(size=16),
                    showlegend=False,
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    plt.savefig('fig/graph.png')
    fig.show()

def cal_display_percentiles(train_bene_df, label_font_dict, title_font_dict,
                            x_col, y_col, title_lbl, x_filter_code):
    """
    Description : This function is created for calculating and generating the percentiles for pre-disease indicators.

    Input: It accepts below parameters:
        1. x_col : Disease indicator feature name.
        2. y_col : Feature like re-imbursement or deductible amount whose percentiles you want to generate.
        3. title_lbl : Label to be provided in the title of the plot.
        4. x_filter_code : Category code for which you want to generate the percentiles.

    Output: It returns the dataframe having percentiles and their respective values for the specific disease indicator feature.
    And, it displays the pointplot graph of the same.
    """
    percentiles = []
    percentiles_vals = []

    # Calculating & storing the various percentiles and their respective values
    for val in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                0.98, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0]:
        percentile = round(float(val * 100), 6)
        percentiles.append(percentile)

        percentile_val = round(train_bene_df[train_bene_df[x_col] == x_filter_code][y_col].quantile(val), 1)
        percentiles_vals.append(percentile_val)

    # Creating the temp dataframe for displaying the results
    tmp_percentiles = pd.DataFrame([percentiles, percentiles_vals]).T
    tmp_percentiles.columns = ['Percentiles', 'Values']

    # Here, I'm displaying the Percentiles values for all disease code features
    sns.set_style('whitegrid')  # Setting Seaborn style

    plt.figure(figsize=(15, 7))
    sns.pointplot(data=tmp_percentiles, x='Percentiles', y='Values', markers="o", palette='spring')
    sns.pointplot(data=tmp_percentiles, x='Percentiles', y='Values', markers="", color='grey',
                  linestyles="solid")
    # Providing the labels and title to the graph
    plt.xlabel("\nPercentiles", fontdict=label_font_dict)
    plt.xticks(rotation=90, size=12)
    plt.ylabel("Total Annual '{}' Sum \n".format(y_col), fontdict=label_font_dict)
    plt.grid(which='major', linestyle="-.", color='lightpink')
    plt.minorticks_on()
    plt.title("Percentile values of '{}' :: '{}'\n".format(y_col, title_lbl), fontdict=title_font_dict)
    plt.savefig('fig/' + title_lbl + '_graph.png')
    plt.show()
    return tmp_percentiles
