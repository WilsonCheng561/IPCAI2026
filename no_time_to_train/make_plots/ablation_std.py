import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
csv_file_path = "ablation-coco-std.csv"  # Change to your file path
df = pd.read_csv(csv_file_path)

# Clean and restructure the data
df_cleaned = df.iloc[2:].copy()  # Skip first two rows
df_cleaned.columns = df.iloc[1]  # Set proper column names from second row

# Extract individual seed columns and compute statistics
# Select columns that are just 'bbox' (for seeds) and exclude 'bbox\nmean' and 'bbox\nstd'
seed_bbox_cols = [col for col in df_cleaned.columns if col == 'bbox']
seed_segm_cols = [col for col in df_cleaned.columns if col == 'segm']

# Create new dataframe with shots and computed statistics
df_stats = pd.DataFrame()

df_stats['shots'] = df_cleaned['Shots']
df_stats['bbox_mean'] = df_cleaned[seed_bbox_cols].astype(float).mean(axis=1)
df_stats['bbox_std'] = df_cleaned[seed_bbox_cols].astype(float).std(axis=1)
df_stats['segm_mean'] = df_cleaned[seed_segm_cols].astype(float).mean(axis=1)
df_stats['segm_std'] = df_cleaned[seed_segm_cols].astype(float).std(axis=1)
df_stats = df_stats.dropna()
df_stats = df_stats.astype(float)

# Print shots and statistics
print("\nNumber of shots found:", df_stats['shots'].tolist())
print("\nStatistics by shot:")
for shot in df_stats['shots'].unique():
    shot_data = df_stats[df_stats['shots'] == shot]
    print(f"\nShot {shot}:")
    print(f"BBox - Mean: {shot_data['bbox_mean'].values[0]:.4f}, Std: {shot_data['bbox_std'].values[0]:.4f}")
    print(f"Segm - Mean: {shot_data['segm_mean'].values[0]:.4f}, Std: {shot_data['segm_std'].values[0]:.4f}")

# Find and print rows with missing values before dropna
missing_rows = df_cleaned[df_cleaned[seed_bbox_cols + seed_segm_cols].isna().any(axis=1)]
if not missing_rows.empty:
    print("\nRows with missing values:")
    print(missing_rows[['Shots'] + seed_bbox_cols + seed_segm_cols])

def create_plot(show_error_bars=True, show_lines=True):
    # Set Seaborn style and color palette
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.35)  # Increased font scale from 1.2 to 1.5
    palette = sns.color_palette("deep")
    bbox_color = palette[0]  # First color from the palette
    segm_color = palette[1]  # Second color from the palette

    # Create a single combined plot
    plt.figure(figsize=(9, 6))

    # Turn off the grid and add vertical lines at shot values
    # plt.grid(False)
    # for shot in df_stats['shots']:
    #     plt.axvline(x=shot, color='gray', linestyle='-', alpha=0.2, zorder=0)

    if show_lines:
        # Plot lines with shaded regions using Seaborn's color palette
        sns.lineplot(data=df_stats, x='shots', y='bbox_mean', color=bbox_color, 
                    label='BBox', marker='o', markersize=7, linewidth=2)
        plt.fill_between(df_stats['shots'], 
                         df_stats['bbox_mean'] - df_stats['bbox_std'],
                         df_stats['bbox_mean'] + df_stats['bbox_std'], 
                         color=bbox_color, alpha=0.2)

        sns.lineplot(data=df_stats, x='shots', y='segm_mean', color=segm_color, 
                    label='Segm', marker='s', markersize=7, linewidth=2)
        plt.fill_between(df_stats['shots'], 
                         df_stats['segm_mean'] - df_stats['segm_std'],
                         df_stats['segm_mean'] + df_stats['segm_std'], 
                         color=segm_color, alpha=0.2)

    if show_error_bars:
        plt.errorbar(df_stats['shots'], df_stats['bbox_mean'], 
                    yerr=df_stats['bbox_std'], fmt='none' if not show_lines else 'none', 
                    color=bbox_color, capsize=5, capthick=1.5, elinewidth=1.5)

        plt.errorbar(df_stats['shots'], df_stats['segm_mean'], 
                    yerr=df_stats['segm_std'], fmt='none' if not show_lines else 'none', 
                    color=segm_color, capsize=5, capthick=1.5, elinewidth=1.5)

    # Set x-ticks to only show the shot values as integers
    plt.xticks(df_stats['shots'], [int(x) for x in df_stats['shots']])

    # Customize the plot
    plt.xlabel("Number of Shots", fontsize=13, fontweight='bold')
    plt.ylabel("mAP", fontsize=13, fontweight='bold')

    # Customize legend
    plt.legend(frameon=True, fancybox=True, shadow=False, fontsize=11)

    # Adjust layout and save
    plt.tight_layout()
    
    # Generate filename based on options
    filename_parts = ["ablation"]
    if not show_error_bars:
        filename_parts.append("no_error_bars")
    if not show_lines:
        filename_parts.append("no_lines")
    filename_parts.append("std")
    filename = "_".join(filename_parts) + ".png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create different versions of the plot
create_plot(show_error_bars=True, show_lines=True)  # Original (ablation_std.png)
create_plot(show_error_bars=False, show_lines=True)  # No error bars (ablation_no_error_bars_std.png)
create_plot(show_error_bars=True, show_lines=False)  # No lines (ablation_no_lines_std.png)
