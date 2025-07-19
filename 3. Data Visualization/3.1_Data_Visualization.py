import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

df = pd.read_csv('cleaned.csv')

# Pie Chart
labels = df['Sentiment'].value_counts().index.to_list()
sizes = df['Sentiment'].value_counts().values
plt.pie(np.array(sizes), labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Overall Percentages of Different Sentiments')
plt.show()

# Bar Plot
plt.bar(x=df['Sentiment'].value_counts().index, height=df['Sentiment'].value_counts().values.tolist(), color=['green','darkblue','red','yellowgreen'])
plt.title('Counts of Each Sentiment')
plt.show()

# Multiple Bar Chart / Count Plot
fig, axs = plt.subplots(1, 1, figsize=(10,6))
sb.countplot(data=df, x='Tool', hue='Sentiment', ax=axs)
axs.set_title('Multiple Barplot / CountPlot', fontsize=12)
axs.tick_params(axis='x', rotation=90)
axs.set_xlabel('Tool', fontsize=10)
axs.set_ylabel('No. Of Instances', fontsize=10)

fig.tight_layout()
plt.show()