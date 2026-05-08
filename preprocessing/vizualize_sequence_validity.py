import seaborn as sns
import matplotlib.pyplot as plt

def plot_sequence_validity(
    df,
    sequence_col='standard_sequence',
    figsize=(6, 4),
    palette='RdYlGn',
    title='Sequence Validity'
):
    """
    Plots distribution of valid vs invalid sequences with correct color mapping.
    """
    # Create categorical values with controlled order
    validity = df[sequence_col].notna().map({
        True: 'Valid',
        False: 'Invalid'
    }).astype('category').cat.set_categories(['Invalid', 'Valid'])
    
    plt.figure(figsize=figsize)
    ax = sns.countplot(
        x=validity,
        hue=validity,
        palette=palette,
        legend=False,
        order=['Invalid', 'Valid']  # Explicit order
    )
    
    total = len(df)
    for p in ax.patches:
        height = int(p.get_height())
        percentage = height / total * 100
        ax.annotate(
            f"{height}\n({percentage:.0f}%)",
            (p.get_x() + p.get_width()/2., height),
            ha='center', 
            va='center',
            fontsize=12,
            color='black'
        )
    
    plt.title(title)
    plt.xlabel('Validity')
    plt.ylabel('Count')
    plt.xticks(rotation=0)  # Remove rotation for clarity
    sns.despine()
    
    return ax

def plot_invalid_categories(
    df,
    sequence_col='standard_sequence',
    category_col='sequence_category',
    figsize=(8, 6),
    palette='Reds_r',
    title='Invalid Sequences by Category'
):
    """
    Shows distribution of categories for invalid sequences with sequential coloring.
    """
    # Filter only the invalid sequences
    invalid_df = df[df[sequence_col].isna()]
    
    # Skip plotting if there are no invalid sequences
    if invalid_df.empty:
        print("No invalid sequences found for the given category.")
        return None
    
    order = invalid_df[category_col].value_counts().index
    
    plt.figure(figsize=figsize)
    ax = sns.countplot(
        x=category_col,
        hue=category_col,
        data=invalid_df,
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        saturation=1
    )
    
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    
    total = len(invalid_df)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}\n({height/total:.0%})",
                (p.get_x() + p.get_width()/2, height),
                ha='center', 
                va='bottom',
                fontsize=10,
                color='black'
            )
    
    plt.tight_layout()
    plt.show()
    return ax