import numpy as np
import sklearn as sk
from scipy.stats import norm
import csv
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from operator import add
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.cm as cm
import matplotlib as mpl

use_wkend, use_wkday= 1,1

#return list of application_version_ids that match a given developer
def find_app_ids(category):
	cat_df = app_info[app_info['category']==category]
	return np.array(cat_df['application_version_id']) 

#returns array with app usage of each unique device

def find_category_usage(category):
	app_id_list = find_app_ids(category)               # list of possible version id's for app
	app_usage_all_arr = []                              # will contain start/stop/run times for *all* app users
	app_usage_indiv_arr = []            
	for indiv_usage in device_app_usage_arr:
		cat_id_list = find_app_ids(category)       
		#app_use = indiv_usage[(indiv_usage['application_version_id'].isin(cat_id_list)) & (indiv_usage['type']==5)]
		app_use = indiv_usage[indiv_usage['application_version_id'].isin(cat_id_list) ]
		if app_use.size!=0:
			app_usage_indiv_arr.append(app_use)
	return app_usage_indiv_arr

def instances_per_day(usergroup):
	sum=0.
	for elem in usergroup:
		sum+=len(elem)
	return( sum/len(usergroup) )

def indiv_group_usage(N_user):
	#returns[[group_name],[number uses]]
	N_uses=[]
	usage_rec = device_app_usage_arr[N_user]

	for cat in category:
		cat_id_list = find_app_ids(cat)   
		app_use = usage_rec[usage_rec['application_version_id'].isin(cat_id_list) ]
		
		N_uses.append(len(app_use))
	return (N_uses)


date_list=['09_01','09_02','09_03','09_04','09_05','09_06','09_07']
wkend_list=['09_05','09_06']
wkday_list=['09_01','09_02','09_03','09_04','09_07']

wkend_arr, wkday_arr, wk_arr=[],[],[]
for str in wkend_list:
	wkend_arr.append('group_usage_'+str+'.dat')
for str in wkday_list:
	wkday_arr.append('group_usage_'+str+'.dat')
for str in date_list:
	wk_arr.append('group_usage_'+str+'.dat')
	
app_categories = pd.read_csv('data/categories_google.csv')
app_categories.columns = ['category', 'category_str']

category=app_categories['category']
###################
category_str=app_categories['category_str'].replace('&','\&')


day_results_arr=[]  #1st dim is all user results in one day

if use_wkend==1 and use_wkday==0:
	multiday_arr=wkend_arr
elif use_wkend==0 and use_wkday==1:
	multiday_arr=wkday_arr
elif use_wkend==1 and use_wkday==1:
	multiday_arr=wk_arr
	
for d in range(len(multiday_arr)):
	day_results = []
	with open(multiday_arr[d]) as inputfile:
		for line in inputfile:
			day_results.append(line.strip().split("  "))
		inputfile.close()
	for m in range(len(day_results)):
		for n in range(len(day_results[m])):
			day_results[m][n]=int(day_results[m][n])
	day_results_arr.append(day_results)

# get list of all unique users over span of days
unique_devices_all = np.asarray(day_results_arr[0]).T.tolist()[0]
for d in range(1,len(multiday_arr)):
	unique_devices_day = np.asarray(day_results_arr[d]).T.tolist()[0]
	new_unique_devices = list(set(unique_devices_day) - set(unique_devices_all))
	for new_device in new_unique_devices:
		unique_devices_all.append(new_device)
		

#check that every row in daily data corresponds to same device_id
compiled_usage=[]    #each row is one user, average uses/day over time period
N_user_cols = len(day_results_arr[0][0])                  

"""
if day_results_arr[0][-1][0]==day_results_arr[-1][-1][0]:
	for N_user in range(len(unique_devices_all)):
		temp_arr= np.zeros(N_user_cols) 
		for d in range(len(day_results_arr)):
			temp_arr = list(map(add, temp_arr, day_results_arr[d][N_user]))
		for cat in range(len(temp_arr)):
			temp_arr[cat]=(1.*temp_arr[cat])/(1.*len(day_results_arr))
		compiled_usage.append(temp_arr)
else:
	print ('Need to sort device IDs: no one-to-one correspondence between rows')
	"""

for N_user in range(len(unique_devices_all)):
	userID=unique_devices_all[N_user]
	N_days_present=0
	temp_arr= np.zeros(N_user_cols) 
			
	for d in range(len(day_results_arr)):        
		day_data=day_results_arr[d]
		for userrow in range(len(day_data)):
			if userID==day_data[userrow][0]:
				N_days_present+=1
				temp_arr= list(map(add, temp_arr, day_results_arr[d][userrow]))
	for col in range(0,len(temp_arr)):
		temp_arr[col]=temp_arr[col]*1./N_days_present
		#temp_arr[0]=int(temp_arr[0])
	compiled_usage.append(temp_arr[1:])   #skip recording device_id, don't need anymore
	#for row in range(0,len(compiled_usage)):
	#    del compiled_usage[row][0]

n_clusters = 4

scaler = preprocessing.StandardScaler()
clf = KMeans(n_clusters)
clf.fit(scaler.fit_transform(compiled_usage))


#calculate and show correlation matrix

N_cats=len(compiled_usage[0])
N_users=len(compiled_usage)
correlation_matrix=np.zeros((N_cats,N_cats))


for i in range(N_cats):
	cat_arr = np.asarray(compiled_usage).T.tolist()[i]
	mean = np.mean(cat_arr)
	std_dev = np.mean(cat_arr)
	for r in range(N_users):
		compiled_usage[r][i]=(compiled_usage[r][i]-mean)
	max_dev = np.max( np.asarray(compiled_usage).T.tolist()[i] )
	#max_dev = np.max( compiled_usage[r] )
	for r in range(N_users):
		if std_dev!=0:
			compiled_usage[r][i]=compiled_usage[r][i]/std_dev
	
		
#this is not really a correlation matrix but whatever
for i in range(0,N_cats):
	N_specific_users=0           # num people with uses>0
	for j in range(0,N_cats):
		correlation=0.
		for k in range(N_users):
			if compiled_usage[k][i]!=0:      #only interested in people who actually use this category
				correlation = correlation + compiled_usage[k][i]*compiled_usage[k][j]
				N_specific_users+=1
		if N_specific_users>0:
			correlation = correlation / N_specific_users
		if i!=j:
			correlation_matrix[i][j] = correlation
		else:
			correlation_matrix[i][j]=0.
			
		#print (i,j,correlation)         

def whatcategory(ind):
	#this is the index in the correlation matrix = google index - 1
	return category_str[ind]

def find_likelevel(used_app, other_app):
	#for now, input apps as integers corresponding to google categories
	corr=correlation_matrix[used_app][other_app]
	#print (corr)
	
	if corr>0.5:
		likelevelstr='really like'
		likelevel=3
	if corr<=0.5 and corr>0.25:
		likelevelstr='like'
		likelevel=2
	if corr<=0.25 and corr>0.05:
		likelevelstr='sort of like'
		likelevel=1
	if corr<=0.05 and corr>-0.05:
		likelevelstr='neither like nor dislike'
		likelevel=0
	if corr<=-0.05 and corr>-0.25:
		likelevelstr='sort of dislike'
		likelevel=-1
	if corr<=-0.25 and corr>-0.5:
		likelevelstr='dislike'
		likelevel=-2
	if corr<=-0.5:
		likelevelstr='really dislike'
		likelevel=-3
	#print('People who use '+whatcategory(used_app)+' apps '+likelevelstr+' '+whatcategory(other_app)+' apps.')
	return likelevel

def normalize_correlations(used_app):
	#returns lognormalized row of correlation matrix
	specific_app_correlations=correlation_matrix[used_app]
	mean=np.mean(specific_app_correlations)
	std=np.std(specific_app_correlations)

	norm_arr=[]
	for corr in specific_app_correlations:
		norm_arr.append( (corr-mean)/std )
	return norm_arr

def best_relations(used_app):
	score3_arr, score2_arr, score1_arr, score0_arr, score_m1_arr, score_m2_arr, score_m3_arr = [],[],[],[],[],[],[]
	for j in range(N_cats):
		if j!=used_app:
			score=find_likelevel(used_app, j)
			if score==3:
				score3_arr.append(whatcategory(j))
			if score==2:
				score2_arr.append(whatcategory(j))
			if score==1:
				score1_arr.append(whatcategory(j))            
			if score==0:
				score0_arr.append(whatcategory(j))
			if score==-1:
				score_m1_arr.append(whatcategory(j))                
			if score==-2:
				score_m2_arr.append(whatcategory(j))    
			if score==-3:
				score_m3_arr.append(whatcategory(j))                    
	big_arr=[score3_arr, score2_arr, score1_arr, score0_arr, score_m1_arr, score_m2_arr, score_m3_arr]
	descriptor_arr=['really like','like','kind of like', 'neither like nor dislike', 'kind of dislike', 'dislike', 'really dislike']
	print('People who often use '+whatcategory(used_app)+' apps...')


	for a in range(len(big_arr)):
		if len(big_arr[a])>1:
			score_arr_str=", ".join(big_arr[a][:-1])
			score_arr_str=score_arr_str+', and '+big_arr[a][-1]
		else:
			score_arr_str=", ".join(big_arr[a])
		if len(big_arr[a])>0:
			print('')
			print ('... '+descriptor_arr[a]+' '+score_arr_str+' apps.')


#print reference guide for app catgories
def print_reference_guide():
	print ('Category        Index')
	print ('---------------------')
	for n in range(N_cats):
		print(repr(n)+'               '+whatcategory(n))

def plot_category(plot_cat_ind, label_threshold=0.3):

	fig=plt.figure()
	ax=fig.add_subplot(111)
	abridged_corr_matrix= np.asarray(correlation_matrix)[plot_cat_ind] 
	x_arr=np.linspace(0,N_cats-1,N_cats,endpoint=True)
	color_range=cm.seismic(np.linspace(0.05, 0.95, 40))

	min_abs= abs(np.min(abridged_corr_matrix))
	norm=np.max( [np.max(abridged_corr_matrix)  , min_abs])
	#print (np.max( np.max(abridged_corr_matrix)  , abs(np.min(abridged_corr_matrix))))


	for i in range(len(abridged_corr_matrix)):
		abridged_corr_matrix[i]=abridged_corr_matrix[i]/norm

	x_arr=np.append(x_arr, 100)
	x_arr=np.append(x_arr, 101)
	abridged_corr_matrix=np.append(abridged_corr_matrix, -1.)
	abridged_corr_matrix=np.append(abridged_corr_matrix, 1.)

	#plt.grid(linewidth=1.)
	plt.plot([-10,60],[0,0],':')
	plt.scatter(x_arr, abridged_corr_matrix, marker='o', s=200, linewidths=4, c=abridged_corr_matrix, cmap=plt.cm.seismic, edgecolors='grey', linewidth=1)

	#ax.scatter(x_arr, abridged_corr_matrix, 'o', color=abridged_corr_matrix, cmap=plt.cm.seismic)
	#ax = cb.ax
	plt.axis([-7, N_cats+5., -1.5, 1.5])

	text = ax.yaxis.label
	font = mpl.font_manager.FontProperties(size=22)
	text.set_font_properties(font)

	plt.title('App usage amongst people who use '+whatcategory(plot_cat_ind)+' apps', fontsize=24)
	plt.ylabel(r'$\leftarrow$ Disfavor~~~~~~~~Favor$\rightarrow$',fontsize=22)

	for x in range(0,len(x_arr)-2):
		if abs(abridged_corr_matrix[x])>label_threshold: 
			y_sign=abridged_corr_matrix[x]/abs(abridged_corr_matrix[x])
			plt.text(x , abridged_corr_matrix[x]+0.2*y_sign,whatcategory(x), fontsize=20, horizontalalignment='center', verticalalignment='center')
	fig.set_size_inches(10,5)