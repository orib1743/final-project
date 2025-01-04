import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the comparison results file
file_path = r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\Final-Project\Final-Project\Output_Files\child_level_comparison.csv"
df = pd.read_csv(file_path)


# 1. Bar Chart: Comparison of children and matches
def plot_bar_chart(df):
    plt.figure(figsize=(10, 6))
    df_plot = df[['Parent Title 2017', 'Total Children 2017', 'Total Children 2018', 'Matching Children']].copy()
    df_plot = df_plot.set_index('Parent Title 2017')

    df_plot.plot(kind='bar', stacked=False)
    plt.title('Comparison of Total Children and Matches for Parent Titles')
    plt.ylabel('Number of Children')
    plt.xlabel('Parent Title 2017')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Children 2017', 'Children 2018', 'Matching Children'])
    plt.tight_layout()

    # Save the bar chart
    plt.savefig(r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\bar_chart.png", dpi=300)
    plt.show()


# 2. Pie Chart: Percentage of Matching Children
def plot_pie_chart(df):
    total_matches = df['Matching Children'].sum()
    total_possible = df[['Total Children 2017', 'Total Children 2018']].min(axis=1).sum()
    unmatched = total_possible - total_matches

    labels = ['Matches', 'Unmatched']
    sizes = [total_matches, unmatched]
    colors = ['#4CAF50', '#FF9999']

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Percentage of Matching Children')

    # Save the pie chart
    plt.savefig(r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\pie_chart.png", dpi=300)
    plt.show()


# 3. Heatmap: Correlation between metrics
def plot_heatmap(df):
    plt.figure(figsize=(8, 6))
    df_heatmap = df[['Semantic Matches', 'Levenshtein_Matches', 'Jaccard_Matches', 'Matching Children']]
    sns.heatmap(df_heatmap.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Matching Metrics')

    # Save the heatmap
    plt.savefig(r"C:\Users\תומר סורוז'ון\Desktop\לימודים\Final Project\heatmap.png", dpi=300)
    plt.show()


# Generate the plots
plot_bar_chart(df)
plot_pie_chart(df)
plot_heatmap(df)
