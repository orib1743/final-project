import pandas as pd

# Load the uploaded files
bigram_file_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\N_GRAM\Titles_N_GRAM\Combined_Bigram_Frequencies.xlsx'
trigram_file_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\N_GRAM\Combined_Trigram_Frequencies.xlsx'

bigram_data = pd.ExcelFile(bigram_file_path)
trigram_data = pd.ExcelFile(trigram_file_path)

# Load the sheets into DataFrames
bigram_df = bigram_data.parse('Sheet1')
trigram_df = trigram_data.parse('Sheet1')

# Standardizing column names for both datasets
bigram_df.columns = ['Bigram_2017', 'Frequency_2017', 'Bigram_2018', 'Frequency_2018']
trigram_df.columns = ['Trigram_2017', 'Frequency_2017', 'Trigram_2018', 'Frequency_2018']

# Function to calculate statistics for a given column
def calculate_statistics(data, column_name):
    max_frequency = data[column_name].max()
    most_frequent_ngram = data[data[column_name] == max_frequency].iloc[0, 0]
    avg_frequency = data[column_name].mean()

    stats = {
        'Total': data[column_name].count(),
        'Unique Frequencies': data[column_name].nunique(),
        'Mean': data[column_name].mean(),
        'Median': data[column_name].median(),
        'Average Frequency': avg_frequency,
        'Most Frequent Ngram': most_frequent_ngram,
        'Max Frequency': max_frequency,
        'Standard Deviation': data[column_name].std(),
        'Minimum Frequency': data[column_name].min(),
        'Maximum Frequency': data[column_name].max(),
    }
    return stats

# Calculating statistics for bigram and trigram frequencies for both years
bigram_stats_2017 = calculate_statistics(bigram_df, 'Frequency_2017')
bigram_stats_2018 = calculate_statistics(bigram_df, 'Frequency_2018')
trigram_stats_2017 = calculate_statistics(trigram_df, 'Frequency_2017')
trigram_stats_2018 = calculate_statistics(trigram_df, 'Frequency_2018')

# Combine the results into a DataFrame for easy comparison
stats_summary = pd.DataFrame({
    'Bigram 2017': bigram_stats_2017,
    'Bigram 2018': bigram_stats_2018,
    'Trigram 2017': trigram_stats_2017,
    'Trigram 2018': trigram_stats_2018
})

# Save the summary to an Excel file
output_file_path = r'C:\Users\revit\PycharmProjects\DS_project\Final-Project\N_GRAM\Ngram_Statistics.xlsx'
stats_summary.to_excel(output_file_path, index=True)

print(f"Summary saved to {output_file_path}")
