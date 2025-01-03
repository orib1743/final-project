import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Create the output PDF file
output_pdf_path = "Output_Files/Statistics.pdf"
with PdfPages(output_pdf_path) as pdf:

    # Reading the files
    files = {
        "2017": {
            "title": "N_GRAM/ngrams_results_2017.csv",
            "detailed": "n_grams_files/ngrams_paragraph_frequencies_2017.csv"
        },
        "2018": {
            "title": "N_GRAM/ngrams_results_2018.csv",
            "detailed": "N_GRAM/ngrams_paragraph_frequencies_2018.csv"
        }
    }

    # Loading the data
    dfs_title = {}
    dfs_detailed = {}

    for year, paths in files.items():
        dfs_title[year] = pd.read_csv(paths["title"], encoding='latin1')
        dfs_detailed[year] = pd.read_csv(paths["detailed"], encoding='latin1')

    # Add general summary table to the PDF
    summary_data = []
    for year, df in dfs_detailed.items():
        average_frequency = df['Frequency'].mean()
        max_frequency = df['Frequency'].max()
        most_frequent_ngram = df[df['Frequency'] == max_frequency]['Ngram'].values[0]
        summary_data.append([
            year,
            f"{average_frequency:.2f}",
            most_frequent_ngram,
            f"{max_frequency:.2f}"
        ])

    # Add summary table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=summary_data,
        colLabels=["Year", "Average Frequency", "Most Frequent Ngram", "Max Frequency"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2, 3])
    pdf.savefig(fig)
    plt.close()

    # Variance and Standard Deviation table
    variance_std_data = [
        ["Year", "Variance", "Standard Deviation"],
        ["2017", f"{dfs_detailed['2017']['Frequency'].var():.2f}", f"{dfs_detailed['2017']['Frequency'].std():.2f}"],
        ["2018", f"{dfs_detailed['2018']['Frequency'].var():.2f}", f"{dfs_detailed['2018']['Frequency'].std():.2f}"]
    ]

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=variance_std_data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1, 2])
    pdf.savefig(fig)
    plt.close()

    # Add top Ngrams by Title_Type for 2017 and 2018
    for year in ["2017", "2018"]:
        top_ngrams_by_category = dfs_detailed[year].loc[
            dfs_detailed[year].groupby('Title_Type')['Frequency'].idxmax()
        ][['Title_Type', 'Ngram', 'Frequency']]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=top_ngrams_by_category.values,
            colLabels=top_ngrams_by_category.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])
        ax.set_title(f"Top Ngrams by Title_Type for {year}", fontsize=14)
        pdf.savefig(fig)
        plt.close()

    # 1. Bar plots: Total Rows
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, (year, df) in enumerate(dfs_title.items()):
        axes[i].bar(df['Title_Type'], df['Total_Rows'], color='blue', alpha=0.7)
        axes[i].set_title(f'Total Rows by Title Type ({year})')
        axes[i].set_xlabel('Title Type')
        axes[i].set_ylabel('Total Rows')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 2. Bar plots: Percentage
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, (year, df) in enumerate(dfs_title.items()):
        axes[i].bar(df['Title_Type'], df['Percentage'], color='orange', alpha=0.7)
        axes[i].set_title(f'Percentage by Title Type ({year})')
        axes[i].set_xlabel('Title Type')
        axes[i].set_ylabel('Percentage')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 3. Bar plots without "Paragraph": Total Rows
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, (year, df) in enumerate(dfs_title.items()):
        filtered_df = df[df['Title_Type'] != 'Paragraph']
        axes[i].bar(filtered_df['Title_Type'], filtered_df['Total_Rows'], color='green', alpha=0.7)
        axes[i].set_title(f'Total Rows by Title Type (No Paragraph) ({year})')
        axes[i].set_xlabel('Title Type')
        axes[i].set_ylabel('Total Rows')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 4. Bar plots without "Paragraph": Percentage
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, (year, df) in enumerate(dfs_title.items()):
        filtered_df = df[df['Title_Type'] != 'Paragraph']
        axes[i].bar(filtered_df['Title_Type'], filtered_df['Percentage'], color='purple', alpha=0.7)
        axes[i].set_title(f'Percentage by Title Type (No Paragraph) ({year})')
        axes[i].set_xlabel('Title Type')
        axes[i].set_ylabel('Percentage')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 5. Histograms: Frequency
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i, (year, df) in enumerate(dfs_detailed.items()):
        axes[i].hist(df['Frequency'], bins=10, color='orange', alpha=0.7)
        axes[i].set_title(f'Histogram of NGRAMS ({year})')
        axes[i].set_xlabel('NGRAMS')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 6. Top 10 Ngrams comparison from 2017 in 2018
    top_ngrams_2017 = dfs_detailed["2017"].nlargest(10, 'Frequency')[['Ngram', 'Frequency']]
    ngrams_2017 = top_ngrams_2017['Ngram']
    frequencies_2017 = top_ngrams_2017['Frequency']

    # Values for 2018
    frequencies_2018 = dfs_detailed["2018"][dfs_detailed["2018"]['Ngram'].isin(ngrams_2017)]
    frequencies_2018 = frequencies_2018.set_index('Ngram').reindex(ngrams_2017).fillna(0)['Frequency']

    # Bar plot for 2017
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    axes[0].bar(ngrams_2017, frequencies_2017, color='blue', alpha=0.7, label='2017')
    axes[0].set_title('Top 10 Ngrams by Frequency (2017)')
    axes[0].set_xlabel('Ngram')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    # Bar plot for 2018
    axes[1].bar(ngrams_2017, frequencies_2018, color='orange', alpha=0.7, label='2018')
    axes[1].set_title('Top 10 Ngrams by Frequency (2018 for 2017 Ngrams)')
    axes[1].set_xlabel('Ngram')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # 7. Top 10 Ngrams comparison from 2018 in 2017
    top_ngrams_2018 = dfs_detailed["2018"].nlargest(10, 'Frequency')[['Ngram', 'Frequency']]
    ngrams_2018 = top_ngrams_2018['Ngram']
    frequencies_2018 = top_ngrams_2018['Frequency']

    # Values for 2017
    frequencies_2017_for_2018 = dfs_detailed["2017"][dfs_detailed["2017"]['Ngram'].isin(ngrams_2018)]
    frequencies_2017_for_2018 = frequencies_2017_for_2018.set_index('Ngram').reindex(ngrams_2018).fillna(0)['Frequency']

    # Bar plot for 2018
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    axes[0].bar(ngrams_2018, frequencies_2018, color='blue', alpha=0.7, label='2018')
    axes[0].set_title('Top 10 Ngrams by Frequency (2018)')
    axes[0].set_xlabel('Ngram')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    # Bar plot for 2017
    axes[1].bar(ngrams_2018, frequencies_2017_for_2018, color='orange', alpha=0.7, label='2017')
    axes[1].set_title('Top 10 Ngrams by Frequency in 2018 (found in 2017)')
    axes[1].set_xlabel('Ngram')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
