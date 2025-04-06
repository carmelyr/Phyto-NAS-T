import pandas as pd

# This method is used to rank the evolution results based on validation accuracy
def rank_evolution_results(csv_file='evolution_results.csv', output_file='ranked_evolution_results.csv'):
    df = pd.read_csv(csv_file)

    ranked_df = df.sort_values(by='Validation Accuracy', ascending=False)
    ranked_df.reset_index(drop=True, inplace=True)
    
    ranked_df.insert(0, 'Rank', ranked_df.index + 1)
    
    ranked_df.to_csv(output_file, index=False)
    
    # Print confirmation
    print(f"Ranked results saved to '{output_file}'. Top 5 entries:")
    print(ranked_df[['Rank', 'Run ID', 'Generation', 'Validation Accuracy']].head())
    
    return ranked_df

if __name__ == "__main__":
    rank_evolution_results()