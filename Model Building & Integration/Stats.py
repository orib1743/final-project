import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Ensure the output directory exists
output_dir = r"C:\Users\אורי בראל\PycharmProjects\Final-Project\Output_Files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the output PDF file
output_pdf_path = os.path.join(output_dir, "Statistics.pdf")
with PdfPages(output_pdf_path) as pdf:

    # Reading the files
    files = {
        "2017": {
            "title": r"C:\Users\אורי בראל\PycharmProjects\Final-Project\N_GRAM\ngrams_results_2017.csv",
            "detailed": r"C:\Users\אורי בראל\PycharmProjects\Final-Project\N_GRAM\cleaned_Paragraph_ngrams_frequencies_2017_v2.xlsx"
        },
        "2018": {
            "title": r"C:\Users\אורי בראל\PycharmProjects\Final-Project\N_GRAM\ngrams_results_2018.csv",
            "detailed": r"C:\Users\אורי בראל\PycharmProjects\Final-Project\N_GRAM\cleaned_Paragraph_ngrams_frequencies_2018_v2.xlsx"
        }
    }

    # Additional Excel files for bigram data
    bigram_files = {
        "2017": r"C:\Users\אורי בראל\PycharmProjects\Final-Project\N_GRAM\Titles_N_GRAM\cleaned_bigram_frequency_2017_v2.xlsx",
        "2018": r"C:\Users\אורי בראל\PycharmProjects\Final-Project\N_GRAM\Titles_N_GRAM\cleaned_bigram_frequency_2018_v2.xlsx"
    }

    # Loading the data
    dfs_title = {}
    dfs_detailed = {}
    dfs_bigram = {}

    for year, paths in files.items():
        dfs_title[year] = pd.read_csv(paths["title"], encoding='latin1')
        dfs_detailed[year] = pd.read_excel(paths["detailed"])

    for year, path in bigram_files.items():
        dfs_bigram[year] = pd.read_excel(path)

    # Standardize column names for detailed and bigram files
    for year in dfs_detailed:
        dfs_detailed[year].columns = ["Ngram", "Frequency", "Cleaned Ngram"]  # Adjusted to three columns

    for year in dfs_bigram:
        dfs_bigram[year].columns = ["Bigram", "Frequency", "Cleaned Bigram"]  # Adjusted to three columns

    # Add general summary table to the PDF
    summary_data = []
    for year, df in dfs_detailed.items():
        avg_freq = df['Frequency'].mean()
        max_freq = df['Frequency'].max()
        top_ngram = df[df['Frequency'] == max_freq]['Ngram'].values[0]
        summary_data.append([year, f"{avg_freq:.2f}", top_ngram, f"{max_freq:.2f}"])

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
    variance_data = []
    for year, df in dfs_detailed.items():
        var = df['Frequency'].var()
        std_dev = df['Frequency'].std()
        variance_data.append([year, f"{var:.2f}", f"{std_dev:.2f}"])

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=variance_data,
        colLabels=["Year", "Variance", "Standard Deviation"],
        loc='center',
        cellLoc='center'
    )
    pdf.savefig(fig)
    plt.close()
    """"
    # Add top Ngrams by Title_Type for each year
    for year, df in dfs_detailed.items():
        top_ngrams = df.nlargest(10, 'Frequency')[['Ngram', 'Frequency']]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=top_ngrams.values,
            colLabels=top_ngrams.columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title(f"Top 10 Ngrams for {year}", fontsize=14)
        pdf.savefig(fig)
        plt.close()
    """

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
    """""
    # Compare top 10 Ngrams: 2017 in 2018
    top_ngrams_2017 = dfs_detailed["2017"].nlargest(10, 'Frequency')[['Ngram', 'Frequency']]
    ngrams_2017 = top_ngrams_2017['Ngram']
    frequencies_2017 = top_ngrams_2017['Frequency']
    frequencies_2018 = dfs_detailed["2018"][dfs_detailed["2018"]['Ngram'].isin(ngrams_2017)]
    frequencies_2018 = frequencies_2018.set_index('Ngram').reindex(ngrams_2017).fillna(0)['Frequency']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].bar(ngrams_2017, frequencies_2017, color='blue', alpha=0.7, label='2017')
    axes[0].set_title('Top 10 Ngrams by Frequency (2017)')
    axes[0].set_xlabel('Ngram')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(ngrams_2017, frequencies_2018, color='orange', alpha=0.7, label='2018')
    axes[1].set_title('Top 10 Ngrams by Frequency (2018 for 2017 Ngrams)')
    axes[1].set_xlabel('Ngram')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # Compare top 10 Ngrams: 2018 in 2017
    top_ngrams_2018 = dfs_detailed["2018"].nlargest(10, 'Frequency')[['Ngram', 'Frequency']]
    ngrams_2018 = top_ngrams_2018['Ngram']
    frequencies_2018 = top_ngrams_2018['Frequency']
    frequencies_2017_for_2018 = dfs_detailed["2017"][dfs_detailed["2017"]['Ngram'].isin(ngrams_2018)]
    frequencies_2017_for_2018 = frequencies_2017_for_2018.set_index('Ngram').reindex(ngrams_2018).fillna(0)['Frequency']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].bar(ngrams_2018, frequencies_2018, color='blue', alpha=0.7, label='2018')
    axes[0].set_title('Top 10 Ngrams by Frequency (2018)')
    axes[0].set_xlabel('Ngram')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(ngrams_2018, frequencies_2017_for_2018, color='orange', alpha=0.7, label='2017')
    axes[1].set_title('Top 10 Ngrams by Frequency in 2018 (found in 2017)')
    axes[1].set_xlabel('Ngram')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    """
    # Compare top 10 Bigrams: 2017 in 2018
    top_bigrams_2017 = dfs_bigram["2017"].nlargest(10, 'Frequency')[['Bigram', 'Frequency']]
    bigrams_2017 = top_bigrams_2017['Bigram']
    frequencies_bigrams_2017 = top_bigrams_2017['Frequency']
    frequencies_bigrams_2018 = dfs_bigram["2018"][dfs_bigram["2018"]['Bigram'].isin(bigrams_2017)]
    frequencies_bigrams_2018 = frequencies_bigrams_2018.set_index('Bigram').reindex(bigrams_2017).fillna(0)['Frequency']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].bar(bigrams_2017, frequencies_bigrams_2017, color='blue', alpha=0.7, label='2017')
    axes[0].set_title('Top 10 Bigrams by Frequency (2017)')
    axes[0].set_xlabel('Bigram')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(bigrams_2017, frequencies_bigrams_2018, color='orange', alpha=0.7, label='2018')
    axes[1].set_title('Top 10 Bigrams by Frequency (2018 for 2017 Bigrams)')
    axes[1].set_xlabel('Bigram')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # Compare top 10 Bigrams: 2018 in 2017
    top_bigrams_2018 = dfs_bigram["2018"].nlargest(10, 'Frequency')[['Bigram', 'Frequency']]
    bigrams_2018 = top_bigrams_2018['Bigram']
    frequencies_bigrams_2018 = top_bigrams_2018['Frequency']
    frequencies_bigrams_2017_for_2018 = dfs_bigram["2017"][dfs_bigram["2017"]['Bigram'].isin(bigrams_2018)]
    frequencies_bigrams_2017_for_2018 = frequencies_bigrams_2017_for_2018.set_index('Bigram').reindex(bigrams_2018).fillna(0)['Frequency']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    axes[0].bar(bigrams_2018, frequencies_bigrams_2018, color='blue', alpha=0.7, label='2018')
    axes[0].set_title('Top 10 Bigrams by Frequency (2018)')
    axes[0].set_xlabel('Bigram')
    axes[0].set_ylabel('Frequency')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(bigrams_2018, frequencies_bigrams_2017_for_2018, color='orange', alpha=0.7, label='2017')
    axes[1].set_title('Top 10 Bigrams by Frequency in 2018 (found in 2017)')
    axes[1].set_xlabel('Bigram')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"PDF saved to: {output_pdf_path}")
