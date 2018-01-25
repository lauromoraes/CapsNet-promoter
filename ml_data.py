#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:44:05 2017

@author: fnord
"""

import numpy as np

# ================================================================
# BaseData =======================================================
# ================================================================
class BaseData(object):
    def __init__(self, npath, ppath):
        self.ppath = ppath
        self.npath = npath

#    def get_sequences_from_fasta(self, path):
#        seqs = list()
#        with open(path) as f:
#            for l in f.readlines()[1::2]:
#                seqs.append(l[:-1])
#        return seqs

    def get_sequences_from_fasta(self, path):
        from Bio import SeqIO
        seqs = list()
        for seq_record in SeqIO.parse(path, "fasta"):
                s = str(seq_record.seq.upper())
                if 'N' not in s:
                    seqs.append( s )
        return seqs
        
    def get_kmers(self, seq, k=1, step=1):
        numChunks = ((len(seq)-k)/step)+1
        mers = list()
        for i in range(0, numChunks*step-1, step):
            mers.append(seq[i:i+k])
        return mers
    
    def encode_sequences(self, seqs):
        raise NotImplementedError()
    
    def enconde_positives(self):
        return self.encode_sequences(self.get_sequences_from_fasta(self.ppath))
    
    def enconde_negatives(self):
        return self.encode_sequences(self.get_sequences_from_fasta(self.npath))
    
    def set_XY(self, negdata, posdata):
        from numpy import array
        from numpy import vstack
        Y = array([0 for x in range(negdata.shape[0])] + [1 for x in range(posdata.shape[0])])
        #Y = Y.transpose()
        X = vstack((negdata, posdata))
        assert X.shape[0]==Y.shape[0]
        self.X = X
        self.Y = Y
        return X, Y
    
    def getX(self, frame=0):
        if frame<0 or frame>3:
            return
        elif frame==0:
            return self.X
        else:
            return self.X[:, frame::3]
    
    def getY(self):
        return self.Y

    def get_negative_samples(self):
        samples = self.X[:(self.n_samples_neg)]
        assert samples.shape[0]==self.n_samples_neg
        return samples

    def get_positive_samples(self):
        samples = self.X[self.n_samples_pos:]
        assert samples.shape[0]==self.n_samples_pos
        return samples
    
    def set_data(self):
        posdata = self.enconde_positives()
        negdata = self.enconde_negatives()
        assert negdata.shape[1]==posdata.shape[1]
        self.n_samples_pos = posdata.shape[0]
        self.n_samples_neg = negdata.shape[0]
        self.sample_size = posdata.shape[1]
        self.set_XY(negdata, posdata)

    def set_kmers_encoder(self):
        from itertools import product
        from numpy import array
        from sklearn.preprocessing import LabelEncoder
        nucs = ['G','A','C','T']
        tups = list(product(nucs, repeat=self.k))
        data = array([''.join(x) for x in tups])
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(data)
    
    def value2label_dinuc_converter(self):
        import numpy as np
        tab = np.loadtxt('dinuc_values', delimiter=',', dtype=np.float32)
        conv = dict()
        cnt = 0
        for i in range(len(tab)):
            conv[i] = dict()
            for elem in tab[i]:
                if not elem in conv[i].keys():
                    conv[i][elem] = cnt
                    cnt += 1
        T = list()
        for i in range(len(tab)):
            R = list()
            for elem in tab[i]:
                R.append( conv[i][elem] )
            T.append( R )
        return np.array( T )
        
# ================================================================
# SequenceProteinData ============================================
# ================================================================
class SequenceProteinData(BaseData):
    STANDARD_GENETIC_CODE   =   {
        'TTT':0,    'TTC':0,    'TCT':10,    'TCC':10,
        'TAT':1,    'TAC':1,    'TGT':13,    'TGC':13,
        'TTA':2,    'TCA':10,   'TAA':20,    'TGA':20,
        'TTG':2,    'TCG':10,   'TAG':20,    'TGG':19,
        'CTT':2,    'CTC':2,    'CCT':14,    'CCC':14,
        'CAT':3,    'CAC':3,    'CGT':15,    'CGC':15,
        'CTA':2,    'CTG':2,    'CCA':14,    'CCG':14,
        'CAA':4,    'CAG':4,    'CGA':15,    'CGG':15,
        'ATT':5,    'ATC':5,    'ACT':11,    'ACC':11,
        'AAT':6,    'AAC':6,    'AGT':10,    'AGC':10,
        'ATA':5,    'ACA':11,   'AAA':16,    'AGA':15,
        'ATG':7,    'ACG':11,   'AAG':16,    'AGG':15,
        'GTT':8,    'GTC':8,    'GCT':17,    'GCC':17,
        'GAT':12,   'GAC':12,   'GGT':18,    'GGC':18,
        'GTA':8,    'GTG':8,    'GCA':17,    'GCG':17,
        'GAA':9,    'GAG':9,    'GGA':18,    'GGG':18}
    
    def __init__(self, npath, ppath):
        super(SequenceProteinData, self).__init__(npath, ppath)
        self.set_data()
    
    def transform(self, trimers):
        return [self.STANDARD_GENETIC_CODE[trimer] for trimer in trimers]
        
    def encode_sequences(self, seqs):
        from numpy import vstack
        return vstack([self.transform(self.get_kmers(seq, k=3)) for seq in seqs])

# ================================================================
# SequenceNucsData ===============================================
# ================================================================
class SequenceNucsData(BaseData):
    def __init__(self, npath, ppath, k=1):
        super(SequenceNucsData, self).__init__(npath, ppath)
        self.k = k
        self.set_kmers_encoder()
        self.set_data()
    
    def encode_sequences(self, seqs):
        from numpy import vstack
        return vstack([self.label_encoder.transform(self.get_kmers(seq, k=self.k)) for seq in seqs])

# ================================================================
# SequenceSimpleData =============================================
# ================================================================
class SequenceSimpleData(BaseData):
    def __init__(self, npath, ppath, k=1):
        super(SequenceSimpleData, self).__init__(npath, ppath)
        self.k = k
        self.set_data()
    
    def enconde_seq(self, seq):
        return [0 if x=='A' or x=='T' else 1 for x in seq]
    
    def encode_sequences(self, seqs):
        from numpy import vstack
        return vstack([self.enconde_seq(seq) for seq in seqs])
    
# ================================================================
# SequenceNucHotvector ===========================================
# ================================================================
class SequenceNucHotvector(BaseData):
    
    NUC_HOT_VECTOR = {
            'A' : np.array([[1.], [0.], [0.], [0.]]),
            'C' : np.array([[0.], [1.], [0.], [0.]]),
            'G' : np.array([[0.], [0.], [1.], [0.]]),
            'T' : np.array([[0.], [0.], [0.], [1.]])
    }
    
    def __init__(self, npath, ppath):
        super(SequenceNucHotvector, self).__init__(npath, ppath)
        self.set_data()

    def enconde_seq(self, seq):
        convHot = lambda x : self.NUC_HOT_VECTOR[nuc]
        return np.hstack([ convHot(nuc) for nuc in seq ]).reshape(1, 4, len(seq), 1)
    
    def encode_sequences(self, seqs):
        return np.vstack([self.enconde_seq(seq) for seq in seqs])

# ================================================================
# SequenceMotifHot ===============================================
# ================================================================
class SequenceMotifHot(BaseData):
    
    NUC_HOT_VECTOR = {
            'A' : np.array([[1.], [0.], [0.], [0.]]),
            'C' : np.array([[0.], [1.], [0.], [0.]]),
            'G' : np.array([[0.], [0.], [1.], [0.]]),
            'T' : np.array([[0.], [0.], [0.], [1.]])
    }

    def __init__(self, npath, ppath):
        super(SequenceMotifHot, self).__init__(npath, ppath)
        self.set_motifs(ppath)
        self.set_data()
        
    def enconde_seq(self, seq):
        convHot = lambda x, i : self.mot[x][i] * self.mot[x][i] * 100 * self.NUC_HOT_VECTOR[x]
        return np.hstack([ convHot(seq[i], i) for i in range(len(seq)) ]).reshape(1, 4, len(seq), 1)
    
    def encode_sequences(self, seqs):
        return np.vstack([self.enconde_seq(seq) for seq in seqs])
    
    def set_motifs(self, ppath):
        from Bio import motifs
        seqs = self.get_sequences_from_fasta(ppath)
        mot = motifs.create(seqs)
        self.mot = {'A':mot.pwm.log_odds()['A'], 'C':mot.pwm.log_odds()['C'], 'G':mot.pwm.log_odds()['G'], 'T':mot.pwm.log_odds()['T']}
#        print mot.counts.normalize()
#        mot.weblogo('weblogo.png')
# ===============================================================
        
        
        
        
        
        
        
        
# ================================================================
# SequenceDinucLabelsProperties ==================================
# ================================================================
class SequenceDinucLabelsProperties(BaseData):
    def __init__(self, npath, ppath):
        super(SequenceDinucLabelsProperties, self).__init__(npath, ppath)
        self.convtable2 = self.value2label_dinuc_converter()
        self.k = 2
        self.set_kmers_encoder()
        self.set_data()
    
    def encode_seq(self, seq):
        import numpy as np                
        convProp = lambda x, prop : np.array([ self.convtable2[prop, x[i]] for i in range(len(x)) ])        
        return np.hstack([ convProp(seq, i) for i in range(38) ])
        
    def encode_sequences(self, seqs):
        from numpy import vstack, array
        mat = vstack([self.label_encoder.transform(self.get_kmers(seq, k=self.k)) for seq in seqs])
        return array([ self.encode_seq(seq) for seq in mat ])









# ================================================================
# SequenceDinucProperties ========================================
# ================================================================
class SequenceDinucProperties(BaseData):
    
    def __init__(self, npath, ppath):
        from Bio import motifs
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        super(SequenceDinucProperties, self).__init__(npath, ppath)
        self.k = 2
        self.set_motifs(ppath)
        self.set_kmers_encoder()
        # Setup tables for convert nucleotides to 2d properties matrix - multichannel signals
        self.convtable2 = np.loadtxt('dinuc_values', delimiter=',', dtype=np.float32)
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # MIN-MAX SCALER
        for prop in range(len(self.convtable2)):
            data = np.vstack([ [x] for x in self.convtable2[prop, :] ])
            scaler.fit(data)
            self.convtable2[prop, :] = scaler.transform(data).transpose()[0]
            #print self.convtable2[prop, :]
        
        self.set_data()
#        seqs = self.get_sequences_from_fasta(self.ppath)
#        instances = motifs.create(seqs)
#        instances.weblogo('test.png')
        
    def set_motifs(self, ppath):
        from Bio import motifs
        seqs = self.get_sequences_from_fasta(ppath)
        mot = motifs.create(seqs)
        self.mot = {'A':mot.pwm['A'], 'C':mot.pwm['C'], 'G':mot.pwm['G'], 'T':mot.pwm['T']}
        
            
        
    def encode_seq(self, seq):
        import numpy as np
                
        convProp = lambda x, prop : np.array([ self.convtable2[prop, x[i]] for i in range(len(x)) ])
        
        return np.vstack([ convProp(seq, i) for i in range(38) ]).reshape(38,len(seq),1)
        
#        return convertedseq.reshape(1, 38, len(seq))

    def encode_sequences(self, seqs):
        from numpy import vstack, array
        mat = vstack([self.label_encoder.transform(self.get_kmers(seq, k=self.k)) for seq in seqs])
        return array([ self.encode_seq(seq) for seq in mat ])
        
# ================================================================
# SimpleHistData =================================================
# ================================================================
class SimpleHistData(BaseData):
    
    def __init__(self, npath, ppath, k=1, upto=False):
        super(SimpleHistData, self).__init__(npath, ppath)
        self.k = k
        self.upto = upto
        self.set_data()
    
    def encode_sequences(self, seqs):
        from repDNA.nac import Kmer
        from numpy import vstack
        kmer = Kmer(k=self.k, upto=self.upto, normalize=True)
        return vstack(kmer.make_kmer_vec(seqs))

# ================================================================
# DinucAutoCovarData =============================================
# ================================================================
class DinucAutoCovarData(BaseData):
    
    def __init__(self, npath, ppath, k=1, upto=False):
        super(DinucAutoCovarData, self).__init__(npath, ppath)
        self.k = k
        self.upto = upto
        self.set_data()
    
    def encode_sequences(self, seqs):
        from repDNA.ac import DAC
        from numpy import vstack
        dac = DAC(self.k)
        return vstack(dac.make_dac_vec(seqs, all_property=True))

# ================================================================
# DinucCrossCovarData ============================================
# ================================================================
class DinucCrossCovarData(BaseData):
    
    def __init__(self, npath, ppath, k=1, upto=False):
        super(DinucCrossCovarData, self).__init__(npath, ppath)
        self.k = k
        self.upto = upto
        self.set_data()
    
    def encode_sequences(self, seqs):
        from repDNA.ac import DCC
        from numpy import vstack
        dcc = DCC(self.k)
        return vstack(dcc.make_dcc_vec(seqs, all_property=True))












# ================================================================
# ======================= TEST ===================================
# ================================================================
TEST = 0

if TEST==1:
    #from pandas.tools.plotting import andrews_curves
    from matplotlib.pyplot import *
     
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('png', 'pdf')
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
#    npath = "fasta/Bacillus_non_prom.fa"
#    ppath = "fasta/Bacillus_prom.fa"
    
    npath = "fasta/Ecoli_non_prom.fa"
    ppath = "fasta/Ecoli_prom.fa"
    
    #mldata = SequenceMotifHot(npath, ppath)
    #print mldata.getX()
    
    
    data = SequenceDinucProperties(npath, ppath)
    X = data.getX()
    Y = data.getY()
    fsize=(18,12)
    
    #for i in range(38):
    #    D = pd.DataFrame( data=X[:,i,:,0].reshape( X.shape[0],  X.shape[2]) )
    #    D = D.iloc[:, 45:66]
    #    D[21] = Y
    #    print i
    #    pd.tools.plotting.parallel_coordinates(D, 21)
    #    pd.tools.plotting.andrews_curves(D, 21)
    #    plt.show()
    
    i=1
    D = pd.DataFrame( data=X[:,i,:,0].reshape( X.shape[0],  X.shape[2]) )
    
    F = D
    F[F.shape[1]+1] = Y
    
    #plt.figure(figsize=(20,10))
    #pd.plotting.parallel_coordinates(F, F.shape[1], color=['#FD1999', '#00E6FE'] )
    #plt.savefig('parallel'+'.pdf')
    #plt.show()
    
#    seg = range(25,66)
    seg = range(25,70)
    D = D.iloc[:, seg]
    D[D.shape[1]+1] = Y
    
    neg = D[D[D.shape[1]] == 0]
    pos = D[D[D.shape[1]] == 1]
    
    #plt.figure(figsize=fsize)
    #pd.plotting.parallel_coordinates(pos, pos.shape[1], color='#00E6FE' )
    #pd.plotting.parallel_coordinates(neg, neg.shape[1], color='#FD1999' )
    #plt.show()
    
    plt.figure(figsize=fsize)
    pd.plotting.parallel_coordinates(neg, neg.shape[1], color='#FD1999' )
    pd.plotting.parallel_coordinates(pos, pos.shape[1], color='#00E6FE' )
    plt.show()
    #plt.figure(figsize=(20,10))
    #pd.tools.plotting.parallel_coordinates(D, D.shape[1])
    
    #plt.savefig('parallel'+'.pdf')
    #plt.figure(figsize=(20,10))
    #pd.tools.plotting.andrews_curves(D, D.shape[1], colorbar)
    #plt.savefig('andrews'+'.pdf')
    plt.show()
elif TEST==2:
    npath = "fasta/Bacillus_non_prom.fa"
    ppath = "fasta/Bacillus_prom.fa"
    data = SequenceDinucLabelsProperties(npath, ppath)
    print data.getX()[0, -1, :, 0]

elif TEST==3:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    npath = "fasta/Ecoli_non_prom.fa"
    ppath = "fasta/Ecoli_prom.fa"
    
    data = SequenceDinucProperties(npath, ppath)
    X = data.getX()
    Y = data.getY()
    
    print 'X', X.shape
    X = X.reshape(len(X), -1)
    print 'X', X.shape

            
   # Create the RFE object and rank each pixel
#    svc = SVC(kernel="linear", C=1)
    tree = RandomForestClassifier(n_jobs=-1, max_features='sqrt')
    rfecv = RFECV(estimator=tree, step=0.1, cv=StratifiedKFold(3), scoring='accuracy')
    rfecv.fit(X, Y)
    ranking = rfecv.ranking_.reshape(38, 79)
    
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    # Plot pixel ranking
    plt.figure()
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking of pixels with RFE")
    plt.show()
    
    print ranking
        
        
elif TEST==4:
    npath = "fasta/Bacillus_non_prom.fa"
    ppath = "fasta/Bacillus_prom.fa"
    data = SequenceNucsData(npath, ppath, k=3)
    print data.getX()