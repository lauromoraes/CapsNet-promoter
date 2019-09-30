# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:32:01 2018

@author: fnord
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sn_a_mean1  = (.91, .97, .88, .87, .90, .88, .94)
sn_a_std1   = (.01, .02, .07, .04, .02, .02, .01)

sn_a_mean2  = (.94, .95, .91, .90, .90, .88, .97)
sn_a_std2   = (.0, .0, .0, .0, .0, .0, .0)
# =========================================
sp_a_mean1  = (.96, .98, .95, .97, .98, .95, .98)
sp_a_std1   = (.01, .01, .02, .01, .01, .01, .01)

sp_a_mean2  = (.94, .97, .95, .96, .98, .94, .97)
sp_a_std2   = (.0, .0, .0, .0, .0, .0, .0)
# =========================================
mcc_a_mean1 = (.88, .95, .83, .86, .90, .84, .92)
mcc_a_std1  = (.01, .01, .08, .02, .01, .01, .01)

mcc_a_mean2  = (.86, .91, .86, .84, .89, .83, .93)
mcc_a_std2   = (.0, .0, .0, .0, .0, .0, .0)

N = len(sn_a_mean1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, mcc_a_mean2, width, color='#D4D6D3', yerr=mcc_a_std2)
rects2 = ax.bar(ind + width, mcc_a_mean1, width, color='#E5BE83', yerr=mcc_a_std1)

# add some text for labels, title and axes ticks
ax.set_ylabel('Mcc')
ax.set_title(u'Comparação do Coeficiente de Matthews')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Arabidopsis_non_tata', 'Arabidopsis_tata', 'Bacillus', 'Ecoli', 'Human_non_tata', 'Mouse_non_tata', 'Mouse_tata'))

for tick in ax.get_xticklabels():
    tick.set_rotation(90)

ax.legend((rects1[0], rects2[0]), ('M1', 'M2'), loc=3)


#def autolabel(rects):
#    """
#    Attach a text label above each bar displaying its height
#    """
#    for rect in rects:
#        height = rect.get_height()
#        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                '%d' % int(height),
#                ha='center', va='bottom')
#
#autolabel(rects1)
#autolabel(rects2)

plt.tight_layout()
plt.savefig('comp_mcc.eps', format='eps', dpi=3000)
plt.show()