import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.ticker as ticker

 #correct_lens=np.load("nq_correct_lens_wiki16.npy")
correct_lens=np.load("correct_lens_wiki_idx.npy")
all_lens=np.load("all_lens.npy")
#lens=np.load("nq_lucene_trigram_train_tuned_k1=1.81,b=0.2295_wiki16.npy")
default_lens=np.load("lens_lucene_bm25_default_bigram.npy")
lens=np.load("retrieved_lens_lucene_bigram_tuned_train.npy")
tfidf_lens=np.load("retrieved_lens_tfidf.npy")
#tfidf_lens=np.load("nq_tfidf_bigram.npy")
#default_lens=np.load("nq_lucene_unigram_default_wiki16.npy")
#lens=np.load("nq_retrieved_lens_lucene_unigram_tuned_train.npy")
#unigram=np.load("nq_tfidf_unigram.npy")
#avg_unigram=np.mean(unigram)
#bigram=np.load("nq_tfidf_bigram.npy")
#avg_bigram=np.mean(bigram)
#print(f"unigram mean len: {avg_unigram} bigram mean len: {avg_bigram}")
fig, ax = plt.subplots()
#tfidf_lens=np.load("retrieved_lens_searchQA_tfidf_bigram.npy")
#default_lens=np.load("retrieved_lens_searchQA_lucene_bigram_default.npy")
#lens=np.load("rretrieved_lens_searchQA_lucene_bigram_tuned.npy")
#unigram=np.load("nq_tfidf_unigram.npy")
#print(f"all mean len: {np.mean(all_lens)}, retrieved mean len: {np.mean(lens)}, correct mean len:{np.mean(correct_lens)}")
shade=True
vertical=False
sns.distplot(np.asarray(lens), hist=False, kde=True, norm_hist=True, vertical=vertical,
             kde_kws={'linewidth': 1,"linestyle":"--",'shade':shade}, label="BM25 tuned")
sns.distplot(np.asarray(default_lens), hist=False, kde=True, norm_hist=True,vertical=vertical,
             kde_kws={'linewidth': 1,"linestyle":"-",'shade':shade}, label="BM25 default")
sns.distplot(np.asarray(tfidf_lens), hist=False, kde=True, norm_hist=True,vertical=vertical,
             kde_kws={'linewidth': 2,"linestyle":"dotted",'shade':shade}, label="TF-IDF")
#print(all_lens)
#print(lens)

sns.distplot(np.asarray(all_lens), hist=False, kde=True, norm_hist=True,
             kde_kws={'linewidth': 1, 'linestyle':'-.','shade':shade}, label="all documents")

sns.distplot(np.asarray(correct_lens), hist=False, kde=True, norm_hist=True,
            kde_kws={'linewidth': 2,'alpha':0.5,'shade':shade}, label="correct documents")

#sns.distplot(np.asarray(unigram), hist=True, kde=True, norm_hist=True,
 #            kde_kws={'linewidth': 1}, label="unigrams")

#sns.distplot(np.asarray(bigram), hist=True, kde=True, norm_hist=True,
 #            kde_kws={'linewidth': 1}, label="unigrams+bigrams")


#  plt.scatter(quest_lens, lens,s=0.05)

plt.rcParams.update({'font.size': 14,"axes.titlesize":14,"axes.labelsize":14})
#sns.set(rc={'figure.figsize':(11.7,8.27),"font.size":20,"axes.titlesize":20,"axes.labelsize":20},style="white")
#sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15})

plt.legend()
plt.xlim(-5000, 135000)
#sns.set_context("paper", font_scale=10.0)

plt.xlabel('Length',  fontsize=14)
plt.ylabel('Density',  fontsize=14)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#

# We change the fontsize of minor ticks label

#ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*10))
#ax.yaxis.set_major_formatter(ticks)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
#sns.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#sns.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#plt.xscale("log")
#plt.yscale("log")

plt.savefig("dist_tfidf_uni_bi.png", dpi=600)
plt.clf()
exit()



sns.kdeplot(np.asarray(lens), bw=1.2,  label="retrieved lens")


sns.kdeplot(np.asarray(all_lens), bw=1.2, label="all lens")

sns.kdeplot(np.asarray(correct_lens), bw=1.2, label="correct lens")
plt.legend()
plt.xlim(-10000, 120000)
plt.xlabel('Length')
plt.ylabel('Density')
plt.savefig("asdfg2.png", dpi=300)