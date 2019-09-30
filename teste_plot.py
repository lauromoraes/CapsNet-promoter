# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:35:08 2018

@author: fnord
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################################################################# MCC

#k = [0.8593737651, 0.8553389745, 0.7784318972, 0.9113220823, 0.8003214083, 0.8503145005, 0.8212621392, 0.8153599144, 0.8392105873, 0.7344867993]
#x = list()
#
#for i in range(0,10,2):
#    x.append( (k[i]+k[i+1])/2 )
#print(x)
#
#df_mcc = pd.DataFrame({
#    'Arabidopsis_non_tata' : np.array([0.8914618592 , 0.8569780601 , 0.8909567193 , 0.9027168258 , 0.8788289546]),
#    'Arabidopsis_tata' : np.array([0.945435053 , 0.965500981 , 0.950218117 , 0.959896776 , 0.955468017]),
#    'Bacillus' : np.array(x),
#    'Ecoli' : np.array([0.83631664, 0.81214853, 0.84394381, 0.79153428, 0.79061714]),
#    'Human_non_tata' : np.array([0.6829905720948718, 0.6806508241752528, 0.6771045795907203, 0.7131432179128229, 0.6889706432976804]),
#    'Mouse_tata' : np.array([0.9262202846, 0.921279092, 0.9362495065, 0.9030511015, 0.8914067925]),
#    'Mouse_non_tata' : np.array([0.8219135468, 0.8070214523, 0.8126171665, 0.8446262532, 0.8433015854]),
#})
#
#df_mcc.boxplot(rot=90)
#
#plt.title("Coeficiente de Matthews")
#plt.ylabel("MCC")
##plt.xlabel("Bases de dados")
#plt.tight_layout()
#
#plt.savefig('mcc.eps', format='eps', dpi=1000)

####################################################################### F1

#k = [0.89743590, 0.89473684, 0.83783784, 0.93506494, 0.85365854, 0.88888889, 0.86746988, 0.86486486, 0.88311688, 0.80000000]
#x = list()
#
#for i in range(0,10,2):
#    x.append( (k[i]+k[i+1])/2 )
#print(x)
#
#df_mcc = pd.DataFrame({
#    'Arabidopsis_non_tata' : np.array([0.92863140, 0.90632506, 0.92845528, 0.93620547, 0.92058347]),
#    'Arabidopsis_tata' : np.array([0.96416938, 0.97719870, 0.96732026, 0.97368421, 0.97068404]),
#    'Bacillus' : np.array(x),
#    'Ecoli' : np.array([0.86956522, 0.85380117, 0.87861272, 0.83798883, 0.83720930]),
#    'Human_non_tata' : np.array([0.8113842173350582, 0.8190294047334449, 0.816718712466971, 0.8360535931790499, 0.8244705882352941]),
#    'Mouse_tata' : np.array([0.94573643, 0.94208494, 0.95312500, 0.92830189, 0.91935484]),
#    'Mouse_non_tata' : np.array([0.89419994, 0.88541973, 0.88648982, 0.90741840, 0.90660321]),
#})
#
#df_mcc.boxplot(rot=90)
#
#plt.title("F-score")
#plt.ylabel("F1")
##plt.xlabel("Bases de dados")
#plt.tight_layout()
#
#plt.savefig('f1.eps', format='eps', dpi=1000)

####################################################################### Sn

#k = [0.94594595, 0.91891892, 0.83783784, 0.97297297, 0.94594595, 0.86486486, 0.97297297, 0.86486486, 0.91891892, 0.75675676]
#x = list()
#
#for i in range(0,10,2):
#    x.append( (k[i]+k[i+1])/2 )
#print(x)
#
#df_mcc = pd.DataFrame({
#    'Arabidopsis_non_tata' : np.array([0.93570220, 0.95769882, 0.96615905, 0.95600677, 0.96108291]),
#    'Arabidopsis_tata' : np.array([0.98666667, 1.00000000, 0.98666667, 0.98666667, 0.99333333]),
#    'Bacillus' : np.array(x),
#    'Ecoli' : np.array([0.83333333, 0.86904762, 0.90476190, 0.89285714, 0.85714286]),
#    'Human_non_tata' : np.array([0.79151943, 0.86471479, 0.85815245, 0.86622918, 0.88440182]),
#    'Mouse_tata' : np.array([0.96825397, 0.96825397, 0.96825397, 0.97619048, 0.90476190]),
#    'Mouse_non_tata' : np.array([0.94226044, 0.92321867, 0.88267813, 0.93918919, 0.93611794]),
#})
#
#df_mcc.boxplot(rot=90)
#
#plt.title("Sensibilidade")
#plt.ylabel("Sn")
##plt.xlabel("Bases de dados")
#plt.tight_layout()
#
#plt.savefig('sn.eps', format='eps', dpi=1000)

####################################################################### Sp

#k = [0.94059406, 0.95049505, 0.94059406, 0.96039604, 0.90099010, 0.97029703, 0.90099010, 0.95049505, 0.94059406, 0.95049505]
#x = list()
#
#for i in range(0,10,2):
#    x.append( (k[i]+k[i+1])/2 )
#print(x)
#
#df_mcc = pd.DataFrame({
#    'Arabidopsis_non_tata' : np.array([0.95898778, 0.91972077, 0.94066318, 0.95549738, 0.93455497]),
#    'Arabidopsis_tata' : np.array([0.96875000, 0.97569444, 0.97222222, 0.97916667, 0.97222222]),
#    'Bacillus' : np.array(x),
#    'Ecoli' : np.array([0.97666667, 0.95333333, 0.95666667, 0.93333333, 0.94666667]),
#    'Human_non_tata' : np.array([0.88608508, 0.82372026, 0.82624369, 0.85291997, 0.81362653]),
#    'Mouse_tata' : np.array([0.97167139, 0.96883853, 0.97733711, 0.95467422, 0.97733711]),
#    'Mouse_non_tata' : np.array([0.89166331, 0.89367700, 0.92871526, 0.91421667, 0.91542489]),
#})
#
#df_mcc.boxplot(rot=90)
#
#plt.title("Especificidade")
#plt.ylabel("Sp")
##plt.xlabel("Bases de dados")
#plt.tight_layout()
#
#
#plt.savefig('sp.eps', format='eps', dpi=1000)

####################################################################### Acc

#k = [0.94202899, 0.94202899, 0.91304348, 0.96376812, 0.91304348, 0.94202899, 0.92028986, 0.92753623, 0.93478261, 0.89855072]
#x = list()
#
#for i in range(0,10,2):
#    x.append( (k[i]+k[i+1])/2 )
#print(x)
#
#df_mcc = pd.DataFrame({
#    'Arabidopsis_non_tata' : np.array([0.95106505, 0.93264249, 0.94933794, 0.95567070, 0.94358089]),
#    'Arabidopsis_tata' : np.array([0.97488584, 0.98401826, 0.97716895, 0.98173516, 0.97945205]),
#    'Bacillus' : np.array(x),
#    'Ecoli' : np.array([0.94531250, 0.93489583, 0.94531250, 0.92447917, 0.92708333]),
#    'Human_non_tata' : np.array([0.84668770, 0.84079916, 0.83953733, 0.85846477, 0.84311251]),
#    'Mouse_tata' : np.array([0.97077244, 0.96868476, 0.97494781, 0.96033403, 0.95824635]),
#    'Mouse_non_tata' : np.array([0.91170032, 0.90537582, 0.91048407, 0.92410606, 0.92361956]),
#})
#
#df_mcc.boxplot(rot=90)
#
#plt.title("Acurácia")
#plt.ylabel("Acc")
##plt.xlabel("Bases de dados")
#plt.tight_layout()
#
#
#plt.savefig('acc.eps', format='eps', dpi=1000)

####################################################################### Prec

#k = [0.85365854, 0.87179487, 0.83783784, 0.90000000, 0.77777778, 0.91428571, 0.78260870, 0.86486486, 0.85000000, 0.84848485]
#x = list()
#
#for i in range(0,10,2):
#    x.append( (k[i]+k[i+1])/2 )
#print(x)
#
#df_mcc = pd.DataFrame({
#    'Arabidopsis_non_tata' : np.array([0.92166667, 0.86018237, 0.89358372, 0.91720779, 0.88335925]),
#    'Arabidopsis_tata' : np.array([0.94267516, 0.95541401, 0.94871795, 0.96103896, 0.94904459]),
#    'Bacillus' : np.array(x),
#    'Ecoli' : np.array([0.90909091, 0.83908046, 0.85393258, 0.78947368, 0.81818182]),
#    'Human_non_tata' : np.array([0.83227176, 0.77792916, 0.77910174, 0.80790960, 0.77214632]),
#    'Mouse_tata' : np.array([0.92424242, 0.91729323, 0.93846154, 0.88489209, 0.93442623]),
#    'Mouse_non_tata' : np.array([0.85080422, 0.85059423, 0.89033457, 0.87772675, 0.87889273]),
#})
#
#df_mcc.boxplot(rot=90)
#
#plt.title("Precisão")
#plt.ylabel("Prec")
##plt.xlabel("Bases de dados")
#plt.tight_layout()
#
#
#plt.savefig('prec.eps', format='eps', dpi=1000)


####################################################################### Comp Mcc
#
#cnn = np.array([0.86, 0.91, 0.86, 0.84, 0.90, 0.83, 0.93])
#caps = np.array([0.88, 0.96, 0.83, 0.81, 0.69, 0.83, 0.92])
#std = np.array([0.02, 0.01, 0.05, 0.02, 0.01, 0.02, 0.02])
#
#ind = np.arange(len(cnn))
#width = 0.2
#
#ax = plt.subplot(111)
#ax.bar(ind, cnn, width, color='#EDC951', label='CNN')
#ax.bar(ind+width, caps, width, yerr=std, color='#EB6841', label='CapsNet')
#
#ax.set_ylabel('MCC')
#plt.xticks(ind, ('Arabidopsis_non_tata', 'Arabidopsis_tata', 'Bacillus', 'Ecoli', 'Human_non_tata', 'Mouse_non_tata', 'Mouse_tata'), rotation='vertical')
##plt.subplots_adjust(top=10.0)
#
##ax.legend(loc='upper right', shadow=True)
#plt.title("Comparação do Coeficiente de Matthews")
##plt.legend(loc='upper left', prop={'size':10}, bbox_to_anchor=(1,1))
#plt.legend(loc='bottom right')
#
#
#
#
#plt.tight_layout()
#plt.savefig('comp_mcc.eps', format='eps', dpi=1000)
#plt.show()

####################################################################### Comp Sn

#cnn = np.array([0.94, 0.95, 0.91, 0.90, 0.90, 0.88, 0.97])
#caps = np.array([0.96, 0.99, 0.90, 0.87, 0.85, 0.92, 0.96])
#std = np.array([0.01, 0.01, 0.07, 0.03, 0.04, 0.02, 0.03])
#
#ind = np.arange(len(cnn))
#width = 0.2
#
#ax = plt.subplot(111)
#ax.bar(ind, cnn, width, color='#EDC951', label='CNN')
#ax.bar(ind+width, caps, width, yerr=std, color='#EB6841', label='CapsNet')
#
#ax.set_ylabel('Sn')
#plt.xticks(ind, ('Arabidopsis_non_tata', 'Arabidopsis_tata', 'Bacillus', 'Ecoli', 'Human_non_tata', 'Mouse_non_tata', 'Mouse_tata'), rotation='vertical')
##plt.subplots_adjust(top=10.0)
#
##ax.legend(loc='upper right', shadow=True)
#plt.title("Comparação da Sensibilidade")
##plt.legend(loc='upper left', prop={'size':10}, bbox_to_anchor=(1,1))
#plt.legend(loc='bottom right')
#
#
#
#
#plt.tight_layout()
#plt.savefig('comp_sn.eps', format='eps', dpi=1000)
#plt.show()

####################################################################### Comp Sp

cnn = np.array([0.94, 0.97, 0.95, 0.96, 0.98, 0.94, 0.97])
caps = np.array([0.94, 0.97, 0.94, 0.95, 0.84, 0.91, 0.97])
std = np.array([0.02, 0.00, 0.02, 0.02, 0.03, 0.02, 0.01])

ind = np.arange(len(cnn))
width = 0.2

ax = plt.subplot(111)
ax.bar(ind, cnn, width, color='#EDC951', label='CNN')
ax.bar(ind+width, caps, width, yerr=std, color='#EB6841', label='CapsNet')

ax.set_ylabel('Sp')
plt.xticks(ind, ('Arabidopsis_non_tata', 'Arabidopsis_tata', 'Bacillus', 'Ecoli', 'Human_non_tata', 'Mouse_non_tata', 'Mouse_tata'), rotation='vertical')
#plt.subplots_adjust(top=10.0)

#ax.legend(loc='upper right', shadow=True)
plt.title("Comparação da Especificidade")
#plt.legend(loc='upper left', prop={'size':10}, bbox_to_anchor=(1,1))
plt.legend(loc='bottom right')




plt.tight_layout()
plt.savefig('comp_sp.eps', format='eps', dpi=1000)
plt.show()


print('END')