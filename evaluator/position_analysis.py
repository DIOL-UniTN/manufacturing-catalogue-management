import pandas as pd
import matplotlib.pyplot as plt
import argparse

def read_arguments():
    parser = argparse.ArgumentParser(description="Python to evaluate position of positive item, soft negative and hard negative item in the retriever output.")
    
    parser.add_argument('--input_csv', type=str, required=True, help='Path of the csv')
    parser.add_argument('--method', type=str, required=True, help='Method used for retrieval')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == "__main__":
    args = read_arguments()
    input_csv = args['input_csv']
    method = args['method']
    dataset = args['dataset']

    # read CSV file
    df = pd.read_csv(input_csv)

    positive_position_frequency = {}
    hard_negative_position_frequency = {}
    soft_negative_position_frequency = {}

    # iterate through the DataFrame to collect position frequencies
    for index, row in df.iterrows():
        positive_position = row['positive_position']
        hard_negative_position = row['hard_negative_position']
        soft_negative_position = row['soft_negative_position']

        positive_position_frequency[positive_position] = positive_position_frequency.get(positive_position, 0) + 1
        hard_negative_position_frequency[hard_negative_position] = hard_negative_position_frequency.get(hard_negative_position, 0) + 1
        soft_negative_position_frequency[soft_negative_position] = soft_negative_position_frequency.get(soft_negative_position, 0) + 1

    # Helper function to prepare tick labels
    def format_ticks(position_keys):
        return ['Not Present' if pos == -1 else str(pos) for pos in position_keys]

    def compute_percentage_frequency(position_frequency):
        total = sum(position_frequency.values())
        return {pos: (count / total) * 100 for pos, count in position_frequency.items()}
    
    def annotate_bars(ax, freqs):
        for x, freq in zip(full_positions, freqs):
            if freq >= 0.1:
                ax.text(x, freq + 1, f'{freq:.1f}%', ha='center', va='bottom', fontsize=6)

    # Compute percentage frequencies
    positive_percentage = compute_percentage_frequency(positive_position_frequency)
    hard_negative_percentage = compute_percentage_frequency(hard_negative_position_frequency)
    soft_negative_percentage = compute_percentage_frequency(soft_negative_position_frequency)

    # Define full x-axis range from -1 (Not Present) to 50
    full_positions = [-1] + list(range(51))  # -1 is 'Not Present'

    # Prepare frequency lists, filling missing positions with 0.0
    def prepare_frequencies(freq_dict):
        return [freq_dict.get(pos, 0.0) for pos in full_positions]

    positive_freqs = prepare_frequencies(positive_percentage)
    hard_negative_freqs = prepare_frequencies(hard_negative_percentage)
    soft_negative_freqs = prepare_frequencies(soft_negative_percentage)

    # Create subplots: 3 rows, 1 column, shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"method: {method}, dataset: {dataset}", fontsize=16)

    # Common plot arguments
    bar_width = 1
    xtick_labels = format_ticks(full_positions)

    # Plot 1: Positive Item
    axes[0].bar(full_positions, positive_freqs, color='blue', width=bar_width)
    axes[0].set_title('Positive Item Position (%)')
    axes[0].set_ylabel('Percentage')
    axes[0].set_ylim(0, 100)
    annotate_bars(axes[0], positive_freqs)

    # Plot 2: Hard Negative Item
    axes[1].bar(full_positions, hard_negative_freqs, color='orange', width=bar_width)
    axes[1].set_title('Hard Negative Item Position (%)')
    axes[1].set_ylabel('Percentage')
    axes[1].set_ylim(0, 100)
    annotate_bars(axes[1], hard_negative_freqs)

    # Plot 3: Soft Negative Item
    axes[2].bar(full_positions, soft_negative_freqs, color='green', width=bar_width)
    axes[2].set_title('Soft Negative Item Position (%)')
    axes[2].set_ylabel('Percentage')
    axes[2].set_xlabel('Position')
    axes[2].set_ylim(0, 100)
    annotate_bars(axes[2], soft_negative_freqs)

    # Set x-ticks and labels only on the last plot (shared x-axis)
    axes[2].set_xticks(full_positions)
    axes[2].set_xticklabels(xtick_labels, rotation=90, ha='center')

    plt.tight_layout()
    plt.savefig(f'evaluator/output/position_frequencies_{method}_{dataset}.png')