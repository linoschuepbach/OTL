import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

data_path = './results/'
plot_path = './plots/'
gendf = pd.read_csv(data_path + 'genData.txt', index_col=0, header=0)
df = pd.read_csv(data_path + 'dataOverview.txt', index_col=0, header=0)
df = pd.concat([df, gendf], axis=1)
print df
df['totPos'] = df['Nr. of pos. Labels grader 1'] + df['Nr. of pos. Labels grader 2']
df['totPosPlusGen'] = df['Nr. of pos. Labels grader 1'] + df['Nr. of pos. Labels grader 2'] + df[
    'Generated Data Grader 1'] + df['Generated Data Grader2']
df['totNeg'] = df['Nr. of neg. Labels grader 1'] + df['Nr. of neg. Labels grader 2']
df['doubleImages'] = df['Nr. of Images'] * 2
print df[['doubleImages', 'Nr. of Images']].sum(), df['totPos'].sum(), df['Nr. of Images'].sum()
# df.groupby('Dataset')
# df['totPos'] = df[[1]] + df[[3]]

ax1 = df[['doubleImages', 'totNeg', 'totPosPlusGen']].sum().plot.bar()
ax1.set_ylabel('#')
ax1.set_title('Overview of Total Number of Labels (augmented)')
plt.savefig(plot_path + 'overNumbersAug.png')

plt.figure()
ax2 = df[['doubleImages', 'totNeg', 'totPos']].sum().plot.bar()
ax2.set_ylabel('#')
ax2.set_title('Overview of Total Number of Labels')
plt.savefig(plot_path + 'overNumbers.png')

ax3 = df[[0, 1, 3, 2, 4]].plot.bar()
ax3.set_ylabel('#')
ax3.set_title('Overview of the Datasets')
plt.savefig(plot_path + 'overDataset.png')

ax4 = df[['Nr. of Images', 'totPos', 'totNeg']].plot.bar()
ax4.set_ylabel('#')
ax4.set_title('Overview over the Distribution of pos. and neg. labels')
plt.savefig(plot_path + 'labelDist.png')

ax5 = df[[5, 6, 7]].plot.box()
ax5.set_ylabel('Dice Score')
ax5.set_title('Overlap between the Graders and between Graders and Ground Truth')
plt.savefig(plot_path + 'dices.png')

ax6 = df[['ratio1', 'ratio2']].plot.box()
ax6.set_ylabel('%')
ax6.set_title('Mean Ratio of Labeled Pixels')
plt.savefig(plot_path + 'pixelRatio.png')

ax7 = df[['Nr. of Images', 'totPosPlusGen', 'totNeg']].plot.bar()
ax7.set_ylabel('#')
ax7.set_title('Overview over the Distribution of pos. and neg. labels (augmented)')
plt.savefig(plot_path + 'labelDistAug.png')

plt.show()
