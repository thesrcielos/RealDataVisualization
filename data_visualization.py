import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    bin_map = {'Yes': 1, 'No': 0}
    df['Depression'] = df['Depression'].astype(float)
    df['Gender'] = df['Gender'].astype(str)
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(bin_map)
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(bin_map)

    return df

def plot_correlation_matrix(df):
    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_cgpa_vs_depression(df):
    fig, ax = plt.subplots(figsize=(6, 4))

    stats = df.groupby('Depression')['CGPA'].agg(['mean', 'std', 'count'])
    stats['error'] = stats['std'] / np.sqrt(stats['count'])
    
    bars = ax.bar(
        x=stats.index.astype(str),
        height=stats['mean'],
        yerr=stats['error'],
        capsize=5,
        color='lightsteelblue',
        edgecolor='royalblue',
        linewidth=1.5
    )
    
    ax.axhline(y=df['CGPA'].mean(), color='firebrick', linestyle='--', label='Global Mean CGPA')
    
    ax.set_title('Average CGPA by Depression Level', pad=15)
    ax.set_xlabel('Depression (0=No, 1=Yes)')
    ax.set_ylabel('CGPA (media ± error estándar)')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_academic_pressure_vs_satisfaction(df):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df['Academic Pressure'],
        df['Study Satisfaction'],
        alpha=0.6,
        color='royalblue',
        edgecolors='w',
        linewidth=0.5
    )

    coeffs = np.polyfit(df['Academic Pressure'], df['Study Satisfaction'], 1)
    trend_line = np.poly1d(coeffs)
    plt.plot(
        df['Academic Pressure'],
        trend_line(df['Academic Pressure']),
        color='crimson',
        linewidth=2,
        label=f'Trend: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}'
    )
    
    plt.title('Academic Pressure vs Study Satisfaction with Regression', pad=20)
    plt.xlabel('Academic Pressure')
    plt.ylabel('Study Satisfaction')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

def plot_depression_by_gender(df):
    print(df[['Gender', 'Depression']].dtypes)
    gender_depression = df.groupby('Gender')['Depression'].mean()
    gender_depression.plot(kind='bar', color='skyblue')
    plt.title('Average Depression by Gender')
    plt.ylabel('Proportion with Depression')
    plt.xlabel('Gender')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()

def plot_suicidal_thoughts_vs_depression(df):
    cross = pd.crosstab(df['Have you ever had suicidal thoughts ?'], df['Depression'])
    cross.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title('Suicidal Thoughts by Depression')
    plt.xlabel('Have you ever had suicidal thoughts? (0=No, 1=Yes)')
    plt.ylabel('Number of Students')
    plt.legend(title='Depression', labels=['No', 'Yes'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_sleep_depression(df):
    sleep_order = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
    
    existing_cats = [cat for cat in sleep_order if cat in df['Sleep Duration'].unique()]

    sleep_avg = df.groupby('Sleep Duration')['Depression'].mean()
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        sleep_avg.index,
        sleep_avg.values,
        color=['skyblue', 'lightblue', 'steelblue', 'navy'][:len(existing_cats)],
        edgecolor='black'
    )
    
    for bar, category in zip(bars, existing_cats):
        height = bar.get_height()
        n = len(df[df['Sleep Duration'] == category])
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.2f}\n(n={n})',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.title('Average Depression by Sleep Duration', pad=15)
    plt.xlabel('Sleep Duration')
    plt.ylabel('Average Depression (0=No, 1=Yes)')
    plt.xticks(rotation=45 if len(existing_cats) > 2 else 0)
    plt.ylim(0, 1)  
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_diet_depression(df):
    diet_map = {
        'Healthy': 'Healthy',
        'Moderate': 'Moderate',
        'Unhealthy': 'Unhealthy'
    }
    df['Diet_Clean'] = df['Dietary Habits'].map(diet_map).dropna()

    diet_stats = df.groupby('Diet_Clean')['Depression'].value_counts(normalize=True).unstack()

    plt.figure(figsize=(8, 5))

    diet_stats.plot(
        kind='barh', 
        stacked=True,
        color=['lightgreen', 'salmon'],
        edgecolor='black',
        width=0.7  
    )

    plt.title('Proportion of Depression by Dietary Habits', pad=15)
    plt.xlabel('Proportion')
    plt.ylabel('Dietary Habits')
    plt.legend(['No Depression', 'Depression'], title='Status', bbox_to_anchor=(1, 1))
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    for i, (idx, row) in enumerate(diet_stats.iterrows()):
        total = 0
        for dep, value in row.items():
            if value > 0:
                plt.text(total + value/2, i, 
                        f'{value:.0%}', 
                        ha='center', 
                        va='center',
                        color='white' if dep == 1 else 'black')
                total += value
    
    plt.tight_layout()
    plt.show()

def plot_academic_pressure_vs_depression(df):
    df['Pressure_Group'] = pd.cut(df['Academic Pressure'],
                                 bins=[0, 1, 2, 3, 4, 5],
                                 labels=['0-1', '2', '3', '4', '5'])
    
    pressure_stats = df.groupby('Academic Pressure')['Depression'].value_counts(normalize=True).unstack()

    plt.figure(figsize=(10, 6))

    pressure_stats.plot(kind='bar',
                      stacked=False,
                      color=['lightgreen', 'salmon'],
                      edgecolor='black',
                      width=0.8)

    plt.title('Relationship between Academic Pressure and Depression', pad=20)
    plt.xlabel('Academic Pressure Level (1-5)')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.legend(['No Depression', 'Depression'], title='Status')
    plt.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(pressure_stats.iterrows()):
        for j, value in enumerate(row):
            plt.text(i, value + 0.02, 
                    f'{value:.0%}', 
                    ha='center',
                    color='black')
    
    plt.tight_layout()
    plt.show()

def generate_graphics():
    df = load_and_prepare_data('./student_depression_dataset.csv')
    plot_correlation_matrix(df)
    plot_cgpa_vs_depression(df)
    plot_academic_pressure_vs_satisfaction(df)
    plot_depression_by_gender(df)
    plot_suicidal_thoughts_vs_depression(df)
    plot_sleep_depression(df)
    plot_diet_depression(df)
    plot_academic_pressure_vs_depression(df)