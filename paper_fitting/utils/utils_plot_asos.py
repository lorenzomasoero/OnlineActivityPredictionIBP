import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('utils_folder/')
from utils_plots import make_color_dict

#                          minimum_max_accuracy = 0.9, 
#                          minimum_min_accuracy = 0.1,

def make_accuracy_asos(results,  
                       minimum_max_accuracy, # every method must attain at least this accuracy 
                       minimum_min_accuracy, # at least one method must attain this accuracy
                       extrapolation=-1,
                    ):

    accuracy = {}

    accuracy['NBP'] = []
    accuracy['SSP'] = []
    # TODO ADD ACCURACY SSP_geom
    accuracy['IBP'] = []
    accuracy['BG'] = []


    for experiment in results:

        for treatment_arm in ['C', 'T']:

            true = results[experiment][treatment_arm]['True'][extrapolation]
            nbp = 1 - min(1,abs(results[experiment][treatment_arm]['NBP_regression'][extrapolation] - true)/true)
            ssp = 1 - min(1,abs(results[experiment][treatment_arm]['SSP_regression'][extrapolation] - true)/true)
            ibp = 1 - min(1,abs(results[experiment][treatment_arm]['IBP_regression'][extrapolation] - true)/true)
            bg = 1 - min(1,abs(results[experiment][treatment_arm]['BG'][extrapolation] - true)/true)
            
            if max(nbp, ssp, ibp) > minimum_max_accuracy and min(nbp, ssp, ibp, bg) > minimum_min_accuracy:
                accuracy['NBP'].append(nbp)
                accuracy['SSP'].append(ssp)
                accuracy['IBP'].append(ibp)
                accuracy['BG'].append(bg)
                #print(experiment, treatment_arm, ' keeping... ', nbp, ssp, ibp)
            #else:
                #print(experiment, treatment_arm, ' skipping... ', nbp, ssp, ibp)
    
    return accuracy


def make_accuracy_plots_asos(results, 
                             color_dict, 
                             minimum_max_accuracy = 0.75, # every method must attain at least this accuracy 
                             minimum_min_accuracy = 0.05, # at least one method must attain this accuracy 
                             extrapolation = -1, 
                             vert=True,
                             save = False):
    
    accuracy = make_accuracy_asos(results, 
                                extrapolation = extrapolation, 
                                minimum_max_accuracy = minimum_max_accuracy,
                                minimum_min_accuracy = minimum_min_accuracy)
    black_dict = make_color_dict('k')
    
    NBP = accuracy['NBP']
    SSP = accuracy['SSP']
    accuracy['SSP_geom']  = accuracy['SSP'] # TODO remove this line
    SSP_geom = accuracy['SSP_geom'] 
    IBP = accuracy['IBP']
    BG = accuracy['BG']
    
    print(len(SSP), ' retained after filtering!')
    
    labs = ['NB-SSP',
            'Be-SSP',
            'TG-SSP',
            'IBP', 
            'BG'] 

    colors = [color_dict['NB-SSP'], 
              color_dict['Be-SSP'], 
              color_dict['TG-SSP'],
              color_dict['IBP'],
              color_dict['BG']] 
        
    if vert == False:
        plt.figure(figsize = (14,6.5))
    else:
        plt.figure(figsize = (7,8.5))

    bp = plt.boxplot([np.array(NBP), 
                      np.array(SSP), 
                      np.array(SSP_geom), 
                      np.array(IBP), 
                      np.array(BG)], 
                      labels=labs, 
                      vert = vert, 
                      **black_dict) 
    
    if vert == False:
    
        plt.xticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 30)
        plt.yticks(fontsize = 25)
        plt.xlabel(r'Prediction Accuracy $v_{7}^{(D_{1})}$', fontsize = 30)
        plt.ylabel('Methods', fontsize = 30)
        plt.xlim([.25,1.02])
    else:
        plt.yticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 20)
        plt.xticks(fontsize = 25, rotation = 20)
        plt.title(r'Prediction Accuracy $v_{7}^{(D_{1})}$', fontsize = 30)
        plt.xlabel('Methods', fontsize = 30)
        plt.ylim([.25,1.02])        
        
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
    for cap in bp['medians']: cap.set(linewidth = 2) 
    for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
    for cap in bp['caps']: cap.set(linewidth = 2)
    
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')
    plt.show()