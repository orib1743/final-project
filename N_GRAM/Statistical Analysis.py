import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
import io


def compute_bigram_statistics(file_path):
    # Load the dataset
    data = pd.read_excel(file_path)

    # Ensure the 'Frequency' column exists
    if 'Frequency' not in data.columns:
        raise ValueError("The dataset does not contain a 'Frequency' column.")

    # Compute statistics
    stats = data['Frequency'].describe()

    # Additional statistics
    total_bigrams = len(data)
    unique_bigrams = data['Frequency'].nunique()
    mode_frequency = data['Frequency'].mode().iloc[0] if not data['Frequency'].mode().empty else None

    # Results dictionary
    results = {
        'Total Bigrams': total_bigrams,
        'Unique Frequencies': unique_bigrams,
        'Mean': stats['mean'],
        'Median': stats['50%'],
        'Standard Deviation': stats['std'],
        'Minimum Frequency': stats['min'],
        'Maximum Frequency': stats['max'],
        'Mode Frequency': mode_frequency,
    }

    return results, data


def create_combined_excel(stats_2017, stats_2018, file_path):
    # Combine statistics into a DataFrame
    combined_stats = pd.DataFrame({
        'Metric': list(stats_2017.keys()),
        '2017': list(stats_2017.values()),
        '2018': list(stats_2018.values())
    })

    # Save to Excel
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        combined_stats.to_excel(writer, index=False, sheet_name='Statistics')
        workbook = writer.book
        sheet = writer.sheets['Statistics']

        # Add graphs to the Excel sheet
        for year, color, data in zip([2017, 2018], ['blue', 'green'], [data_2017, data_2018]):
            # Create histogram
            fig, ax = plt.subplots()
            ax.hist(data['Frequency'], bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'Histogram of Bigram Frequencies ({year})')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Count')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Save the histogram to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)

            # Insert histogram into the Excel sheet
            img = Image(buf)
            sheet.add_image(img, f"E{(year - 2015) * 15}")

            # Create boxplot
            fig, ax = plt.subplots()
            ax.boxplot(data['Frequency'], patch_artist=True, boxprops=dict(facecolor=color))
            ax.set_title(f'Boxplot of Bigram Frequencies ({year})')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Save the boxplot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)

            # Insert boxplot into the Excel sheet
            img = Image(buf)
            sheet.add_image(img, f"E{(year - 2015) * 15 + 20}")


# File paths
file_2017 = 'C:\\Users\\revit\\PycharmProjects\\DS_project\\Final-Project\\N_GRAM\\Titles_N_Gram\\cleaned_bigram_frequency_2017_v2.xlsx'
file_2018 = 'C:\\Users\\revit\\PycharmProjects\\DS_project\\Final-Project\\N_GRAM\\Titles_N_Gram\\cleaned_bigram_frequency_2018_v2.xlsx'
output_combined = 'C:\\Users\\revit\\PycharmProjects\\DS_project\\Final-Project\\N_GRAM\\combined_statistics.xlsx'

# Compute statistics for both datasets
stats_2017, data_2017 = compute_bigram_statistics(file_2017)
stats_2018, data_2018 = compute_bigram_statistics(file_2018)

# Create combined Excel file with statistics and graphs
create_combined_excel(stats_2017, stats_2018, output_combined)

# Display results
print("Combined statistics and graphs saved to:", output_combined)
