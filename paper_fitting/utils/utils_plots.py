import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    
def make_color_dict(color):
    
    dict_ = {'patch_artist': True,
             'boxprops': dict(color=color, facecolor='w'),
             'capprops': dict(color=color),
             'flierprops': dict(color=color, markeredgecolor=color),
             'medianprops': dict(color='k'),
             'whiskerprops': dict(color=color)}
    
    return dict_

def plot_accumulation_curve(weblab_id, results, color_dict, save=False):
    plt.figure(figsize = (20,4))
    for t, treatment in enumerate(results[weblab_id]):
        
        plt.subplot(1,len(results[weblab_id]), t+1)
        
        plt.plot(results[weblab_id][treatment]['NBP'], color = color_dict['NBP'], label = 'NBP')
        plt.plot(results[weblab_id][treatment]['SSP'], color = color_dict['SSP'], label = 'SSP')
        plt.plot(results[weblab_id][treatment]['IBP'], color = color_dict['IBP'], label = 'IBP')
        plt.plot(results[weblab_id][treatment]['BB'], color = color_dict['BB'],label = 'BB')
        plt.plot(results[weblab_id][treatment]['GT'][0], color = color_dict['GT'], label = 'GT')
        len_exp = len(results[weblab_id][treatment]['True'])//7
        plt.scatter(7*np.arange(1,len_exp+1), results[weblab_id][treatment]['GM'][:len_exp], marker = 'X', s= 200, color = color_dict['GM'], label = 'GM')

        for order in range(results[weblab_id][treatment]['J'].shape[0]):
            plt.plot(results[weblab_id][treatment]['J'][order], color = color_dict['J'][order], label = 'J'+str(int(order+1)))
        
        trues = results[weblab_id][treatment]['True']
        plt.vlines(x = 7, ymin = 0, ymax = trues[-1], color = 'k', lw = 2)   
        plt.plot(trues, color = color_dict['True'], label = 'True', lw = 3)
        
        plt.xlabel('# Days', fontsize = 30)
        plt.ylabel('# Customers', fontsize = 30)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.legend(bbox_to_anchor=(0, -0.15, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
        plt.title(weblab_id+', '+str(treatment), fontsize = 15)
        plt.legend()
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight') 
    plt.show()

def make_accuracy_run_time_dicts_real(results, filter_threshold = -1, include_regression = False):
    
    accuracy_at_end = {}
    run_time = {}
    if include_regression == False:
        keys = ['NBP', 'IBP', 'SSP', 'BB', 'BBG', 'LP', 'J1', 'J2', 'J3', 'J4', 'GT'] #, 'GM']
    else:
        keys = ['NBP', 'IBP', 'IBP_regression', 'SSP', 'SSP_regression', 'BB', 'BBG', 'LP', 'J1', 'J2', 'J3', 'J4', 'GT']
    for key in keys:
        accuracy_at_end[key] = []
        run_time[key] = []
        
    for weblab in results:
        for treatment in results[weblab]:
            if treatment == 'weblab':
                continue
            #print(results[weblab].keys())
            true = results[weblab][treatment]['True'][-1]
            best = 0
            accuracy = {}
            time = {}
            for key in accuracy_at_end.keys():
                if key in ['J1', 'J2', 'J3', 'J4']:
                    order = int(key[-1])
                    accuracy[key] = max(0,1-np.abs((results[weblab][treatment]['J'][order-1,-1]-true))/true)
                    best = max(best, accuracy[key])
                    time[key] = results[weblab][treatment]['J_time'][order-1]
                elif key == 'GT':
                    accuracy[key] = max(0,1-np.abs((results[weblab][treatment]['GT'][0][-1]-true))/true)
                    best = max(best, accuracy[key])
                    time[key] = results[weblab][treatment]['GT_time']

                else:
                    accuracy[key] = max(0,1-np.abs((results[weblab][treatment][key][-1]-true))/true)
                    best = max(best, accuracy[key])
                    time[key] = results[weblab][treatment][key+'_time']

            if best > filter_threshold:
                for key in accuracy.keys():
                    accuracy_at_end[key].append(accuracy[key])
                    run_time[key].append(time[key])
                        
    return accuracy_at_end, run_time

    
def make_accuracy_run_time_dicts(results, filter_threshold = -1, include_regression = False):
    
    accuracy_at_end = {}
    run_time = {}
    if include_regression == False:
        keys = ['SSP', 'IBP', 'NBP', 'BB', 'BBG', 'LP', 'J1', 'J2', 'J3', 'J4', 'GT'] #, 'GM']

    for key in keys:
        accuracy_at_end[key] = []
        run_time[key] = []
        
    for weblab in results:
        for treatment in results[weblab]:
            #print(results[weblab].keys())
            true = results[weblab][treatment]['True'][-1]
            best = 0
            accuracy = {}
            time = {}
            for key in accuracy_at_end.keys():
                if key in ['J1', 'J2', 'J3', 'J4']:
                    order = int(key[-1])
                    accuracy[key] = max(0,1-np.abs((results[weblab][treatment]['J'][order-1,-1]-true))/true)
                    best = max(best, accuracy[key])
                    time[key] = results[weblab][treatment]['J_time'][order-1]
                elif key == 'GT':
                    accuracy[key] = max(0,1-np.abs((results[weblab][treatment]['GT'][0][-1]-true))/true)
                    best = max(best, accuracy[key])
                    time[key] = results[weblab][treatment]['GT_time']

                else:
                    accuracy[key] = max(0,1-np.abs((results[weblab][treatment][key][-1]-true))/true)
                    best = max(best, accuracy[key])
                    time[key] = results[weblab][treatment][key+'_time']

            if best > filter_threshold:
                for key in accuracy.keys():
                    accuracy_at_end[key].append(accuracy[key])
                    run_time[key].append(time[key])
                        
    return accuracy_at_end, run_time

def make_accuracy_plots_real(results, 
                             color_dict, 
                             filter_threshold = -1, 
                             save = False, 
                             include_regression = True):
    
    accuracy_at_end, _ = make_accuracy_run_time_dicts_real(results, 
                                                           filter_threshold = filter_threshold, 
                                                           include_regression = include_regression)
    black_dict = make_color_dict('k')
    
    NBP = accuracy_at_end['NBP']
    SSP = accuracy_at_end['SSP']
    IBP = accuracy_at_end['IBP']
    BB = accuracy_at_end['BB']
    BG = accuracy_at_end['BBG']
    LP = accuracy_at_end['LP']
    GT = accuracy_at_end['GT']
    J1 = accuracy_at_end['J1']
    J2 = accuracy_at_end['J2']
    J3 = accuracy_at_end['J3']
    J4 = accuracy_at_end['J4']
    #GM = accuracy_at_end['GM']
    
    if include_regression == True:
        SSP_reg = accuracy_at_end['SSP_regression']
        IBP_reg = accuracy_at_end['IBP_regression']        
    
    print(len(SSP), ' retained after filtering!')
    
    labs = ['NBP', 'SSP', 'IBP', 'BB', 'BG', 'LP', 'GT', 'J1', 'J2', 'J3', 'J4'] #, 'GM'
    colors = [color_dict['NBP'], color_dict['SSP'], color_dict['IBP'], color_dict['BB'], color_dict['BG'], color_dict['LP'], color_dict['GT'], color_dict['J'][0], color_dict['J'][1], color_dict['J'][2], color_dict['J'][3]]#, color_dict['GM']
    
    if include_regression == True:
        
        labs = ['NBP', 
                'SSP',
                'SSP_r',
                'IBP', 
                'IBP_r',
                'BB', 
                'BG', 
                'LP', 
                'GT', 
                'J1', 
                'J2', 
                'J3', 
                'J4'] #, 'GM'
        colors = [color_dict['NBP'], 
                  color_dict['SSP'], 
                  color_dict['SSP'], 
                  color_dict['IBP'],  
                  color_dict['IBP'],  
                  color_dict['BB'], 
                  color_dict['BG'], 
                  color_dict['LP'], 
                  color_dict['GT'], 
                  color_dict['J'][0], 
                  color_dict['J'][1], 
                  color_dict['J'][2], 
                  color_dict['J'][3]] #, color_dict['GM']
        
        
    plt.figure(figsize = (14,8))
    if include_regression == True:
        bp = plt.boxplot([np.array(NBP), np.array(SSP), np.array(SSP_reg), np.array(IBP), np.array(IBP_reg), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4)], labels=labs, vert = False, **black_dict) #, np.array(GM)
    else: 
        bp = plt.boxplot([np.array(NBP), np.array(SSP), np.array(IBP), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4)], labels=labs, vert = False, **black_dict) #, np.array(GM)
    
    plt.xticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 30)
    plt.yticks(fontsize = 25)
    plt.xlabel('Prediction Accuracy', fontsize = 30)
    plt.ylabel('Methods', fontsize = 30)
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
    for cap in bp['medians']: cap.set(linewidth = 2) 
    for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
    for cap in bp['caps']: cap.set(linewidth = 2)
    plt.xlim([.5,1.02])
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')
    plt.show()
    
def make_runtime_plot(results, color_dict, save = False):
    
    _, run_time = make_accuracy_run_time_dicts(results)
    black_dict = make_color_dict('k')
    NBP = np.log(run_time['NBP'])
    SSP = np.log(run_time['SSP'])
    IBP = np.log(run_time['IBP'])
    BB = np.log(run_time['BB'])
    BG = np.log(run_time['BBG'])
    LP = np.log(run_time['LP'])
    GT = np.log(run_time['GT'])
    J1 = np.log(run_time['J1'])
    J2 = np.log(run_time['J2'])
    J3 = np.log(run_time['J3'])
    J4 = np.log(run_time['J4'])
    GM = np.log(run_time['GM'])

    labs = ['NBP', 'SSP', 'IBP', 'BB', 'BG', 'LP', 'GT', 'J1', 'J2', 'J3', 'J4', 'GM']
    colors = [color_dict['NBP'], color_dict['SSP'], color_dict['IBP'], color_dict['BB'], color_dict['BG'], color_dict['LP'], color_dict['GT'], color_dict['J'][0], color_dict['J'][1], color_dict['J'][2], color_dict['J'][3], color_dict['GM']]
    plt.figure(figsize = (20,6))
    bp = plt.boxplot([np.array(SSP), np.array(IBP), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4), np.array(GM)], labels=labs, vert = False, **black_dict)
    # plt.xlim([0,1.02])
    plt.xticks(fontsize = 30)
    # plt.xticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 15)
    plt.yticks(fontsize = 25)
    plt.xlabel('Runtime (log seconds)', fontsize = 30)
    plt.ylabel('Methods', fontsize = 30)
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
    for cap in bp['medians']: cap.set(linewidth = 2) 
    for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
    for cap in bp['caps']: cap.set(linewidth = 2) 
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')    
    plt.show()
    
def make_accuracy_plots_zipf_paper(results_all, 
                                   color_dict, 
                                   filter_threshold = -1, 
                                   save = False):
    
    plt.figure(figsize = (14,12))
    c=1
    for tau, results in results_all.items():
        plt.subplot(2,2,c)
        accuracy_at_end = make_accuracy_dicts_zipf(results, filter_threshold = filter_threshold)
        black_dict = make_color_dict('k')

        
        NBP = accuracy_at_end['NBP_like']
        SSP = accuracy_at_end['SSP_like']
        IBP = accuracy_at_end['IBP_like']
        BB = accuracy_at_end['BB']
        LP = accuracy_at_end['LP']
        GT = accuracy_at_end['GT']
        J1 = accuracy_at_end['J1']
        J2 = accuracy_at_end['J2']
        J3 = accuracy_at_end['J3']
        J4 = accuracy_at_end['J4']
      

        print(len(SSP), ' retained after filtering!')

        labs = ['NBP', 'SSP', 'IBP', 'BB', 'LP', 'J3', 'J4'] #, 'GM']
        colors = [color_dict['NBP'], 
                  color_dict['SSP'],
                  color_dict['IBP'], 
                  color_dict['BB'],
                  #color_dict['BG'],
                  color_dict['LP'], 
                  color_dict['J'][2], 
                  color_dict['J'][3]]

        
        bp = plt.boxplot([np.array(NBP), 
                          np.array(SSP), 
                          np.array(IBP), 
                          np.array(BB), 
                          np.array(LP), 
                          np.array(J3), 
                          np.array(J4)], labels=labs, vert = True, **black_dict)
#         if c > 2:
        plt.xticks(#np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], 
                       fontsize = 27)
        if c < 3:
            plt.xticks([])
        #if c in [1, 3]:
        plt.yticks(fontsize = 18)
        #else:
        #    plt.yticks([])
        if c in [1, 3]:
            plt.ylabel(r'Prediction Accuracy $v_{D_0}^{(D_1)}$', fontsize = 20)
        if c in [3,4]:
            plt.xlabel('Methods', fontsize = 25)
        for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
        for cap in bp['medians']: cap.set(linewidth = 2) 
        for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
        for cap in bp['caps']: cap.set(linewidth = 2)
#         plt.xlim([.5, 1.02])
        plt.title(r'$\tau = $'+str(tau), fontsize = 20)
        c+=1
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')
    plt.show()    

def make_accuracy_plots_zipf_all(results_all, color_dict, filter_threshold = -1, save = False, include_regression = False):
    
    plt.figure(figsize = (14,12))
    c=1
    for tau, results in results_all.items():
        plt.subplot(2,2,c)
        accuracy_at_end = make_accuracy_dicts_zipf(results, filter_threshold = filter_threshold, include_regression = include_regression)
        black_dict = make_color_dict('k')

        NBP = accuracy_at_end['NBP']
        NBP_like = accuracy_at_end['NBP_like']
        SSP = accuracy_at_end['SSP']
        SSP_like = accuracy_at_end['SSP_like']
        IBP = accuracy_at_end['IBP']
        IBP_like = accuracy_at_end['IBP_like']
        BB = accuracy_at_end['BB']
        BG = accuracy_at_end['BBG']
        LP = accuracy_at_end['LP']
        GT = accuracy_at_end['GT']
        J1 = accuracy_at_end['J1']
        J2 = accuracy_at_end['J2']
        J3 = accuracy_at_end['J3']
        J4 = accuracy_at_end['J4']
      

        print(len(SSP), ' retained after filtering!')

        labs = ['NBP', 'NBP_like', 'SSP', 'SSP_like', 'IBP', 'IBP_like', 'BB', 'BG', 'LP', 'GT', 'J1', 'J2', 'J3', 'J4'] #, 'GM']
        colors = [color_dict['NBP'], color_dict['NBP'], color_dict['SSP'], color_dict['SSP'], color_dict['IBP'], color_dict['IBP'], color_dict['BB'], color_dict['BG'], color_dict['LP'], color_dict['GT'], color_dict['J'][0], color_dict['J'][1], color_dict['J'][2], color_dict['J'][3], color_dict['GM']]

        
        bp = plt.boxplot([np.array(NBP), np.array(NBP_like), np.array(SSP), np.array(SSP_like), np.array(IBP), np.array(IBP_like), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4)], labels=labs, vert = False, **black_dict)
        if c > 2:
            plt.xticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 20)
        if c < 3:
            plt.xticks([])
        plt.yticks(fontsize = 18)
        if c > 2:
            plt.xlabel(r'Prediction Accuracy $v_{D_0}^{(D_1)}$', fontsize = 20)
        if c in [1,3]:
            plt.ylabel('Methods', fontsize = 20)
        for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
        for cap in bp['medians']: cap.set(linewidth = 2) 
        for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
        for cap in bp['caps']: cap.set(linewidth = 2)
        plt.xlim([.5,1.02])
        plt.title(r'$\tau = $'+str(tau), fontsize = 20)
        c+=1
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')
    plt.show()    
    
def make_accuracy_dicts_zipf(results, 
                             filter_threshold = -1):
    
    accuracy_at_end = {}

    keys = ['SSP', 
            'SSP_like', 
            'IBP', 
            'IBP_like', 
            'NBP', 
            'NBP_like', 
            'BB', 
            'BBG',
            'BG',
            'LP', 
            'J1', 
            'J2', 
            'J3', 
            'J4', 
            'GT'] 

    for key in keys:
        accuracy_at_end[key] = []
        
    for seed in results:

        true = results[seed]['True'][-1]
        best = 0
        accuracy = {}
        time = {}
        for key in accuracy_at_end.keys():
            if key in ['J1', 'J2', 'J3', 'J4']:
                order = int(key[-1])
                accuracy[key] = max(0,1-np.abs((results[seed]['J'][order-1,-1]-true))/true)
                best = max(best, accuracy[key])
            elif key == 'GT':
                accuracy[key] = max(0,1-np.abs((results[seed]['GT'][0][-1]-true))/true)
                best = max(best, accuracy[key])
            else:
                accuracy[key] = max(0,1-np.abs((results[seed][key][-1]-true))/true)
                best = max(best, accuracy[key])

        if best > filter_threshold:
            for key in accuracy.keys():
                accuracy_at_end[key].append(accuracy[key])

    return accuracy_at_end


def make_accuracy_plots_zipf(results_all, color_dict, filter_threshold = -1, save = False, include_regression = False):
    
    plt.figure(figsize = (14,12))
    c=1
    for tau, results in results_all.items():
        plt.subplot(2,2,c)
        accuracy_at_end, _ = make_accuracy_run_time_dicts(results, filter_threshold = filter_threshold, include_regression = include_regression)
        black_dict = make_color_dict('k')

        NBP = accuracy_at_end['NBP']
        SSP = accuracy_at_end['SSP']
        IBP = accuracy_at_end['IBP']
        BB = accuracy_at_end['BB']
        BG = accuracy_at_end['BBG']
        LP = accuracy_at_end['LP']
        GT = accuracy_at_end['GT']
        J1 = accuracy_at_end['J1']
        J2 = accuracy_at_end['J2']
        J3 = accuracy_at_end['J3']
        J4 = accuracy_at_end['J4']
        #GM = accuracy_at_end['GM']

        if include_regression == True:
            SSP_reg = accuracy_at_end['SSP_regression']
            IBP_reg = accuracy_at_end['IBP_regression']        

        print(len(SSP), ' retained after filtering!')

        labs = ['NBP', 'SSP', 'IBP', 'BB', 'BG', 'LP', 'GT', 'J1', 'J2', 'J3', 'J4'] #, 'GM']
        colors = [color_dict['NBP'], color_dict['SSP'], color_dict['IBP'], color_dict['BB'], color_dict['BG'], color_dict['LP'], color_dict['GT'], color_dict['J'][0], color_dict['J'][1], color_dict['J'][2], color_dict['J'][3], color_dict['GM']]

        if include_regression == True:

            labs = ['SSP', 'SSP_reg', 'IBP', 'IBP_reg', 'BB', 'BG', 'LP', 'GT', 'J1', 'J2', 'J3', 'J4'] #, 'GM']
            colors = [color_dict['SSP'], color_dict['SSP'], color_dict['IBP'], color_dict['IBP'], color_dict['BB'], color_dict['BG'], color_dict['LP'], color_dict['GT'], color_dict['J'][0], color_dict['J'][1], color_dict['J'][2], color_dict['J'][3]] #, color_dict['GM']]


        
        if include_regression == True:
            bp = plt.boxplot([np.array(SSP), np.array(SSP_reg), np.array(IBP), np.array(IBP_reg), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4)], labels=labs, vert = False, **black_dict)
        else: 
            bp = plt.boxplot([np.array(NBP), np.array(SSP), np.array(IBP), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4)], labels=labs, vert = False, **black_dict)
        if c > 2:
            plt.xticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 20)
        if c < 3:
            plt.xticks([])
        plt.yticks(fontsize = 18)
        if c > 2:
            plt.xlabel(r'Prediction Accuracy $v_{D_0}^{(D_1)}$', fontsize = 20)
        if c in [1,3]:
            plt.ylabel('Methods', fontsize = 20)
        for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
        for cap in bp['medians']: cap.set(linewidth = 2) 
        for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
        for cap in bp['caps']: cap.set(linewidth = 2)
        plt.xlim([.5,1.02])
        plt.title(r'$\tau = $'+str(tau), fontsize = 20)
        c+=1
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')
    plt.show()
    
def make_runtime_plot(results, color_dict, save = False):
    
    _, run_time = make_accuracy_run_time_dicts(results)
    black_dict = make_color_dict('k')
    NBP = np.log(run_time['NBP'])
    SSP = np.log(run_time['SSP'])
    IBP = np.log(run_time['IBP'])
    BB = np.log(run_time['BB'])
    BG = np.log(run_time['BBG'])
    LP = np.log(run_time['LP'])
    GT = np.log(run_time['GT'])
    J1 = np.log(run_time['J1'])
    J2 = np.log(run_time['J2'])
    J3 = np.log(run_time['J3'])
    J4 = np.log(run_time['J4'])
    GM = np.log(run_time['GM'])

    labs = ['NBP', 'SSP', 'IBP', 'BB', 'BG', 'LP', 'GT', 'J1', 'J2', 'J3', 'J4', 'GM']
    colors = [color_dict['NBP'], color_dict['SSP'], color_dict['IBP'], color_dict['BB'], color_dict['BG'], color_dict['LP'], color_dict['GT'], color_dict['J'][0], color_dict['J'][1], color_dict['J'][2], color_dict['J'][3], color_dict['GM']]
    plt.figure(figsize = (20,6))
    bp = plt.boxplot([np.array(SSP), np.array(IBP), np.array(BB), np.array(BG), np.array(LP), np.array(GT), np.array(J1), np.array(J2), np.array(J3), np.array(J4), np.array(GM)], labels=labs, vert = False, **black_dict)
    # plt.xlim([0,1.02])
    plt.xticks(fontsize = 30)
    # plt.xticks(np.linspace(0,1,6), [str(i) + str('%') for i in np.linspace(0,100,6,dtype = int)], fontsize = 15)
    plt.yticks(fontsize = 25)
    plt.xlabel('Runtime (log seconds)', fontsize = 30)
    plt.ylabel('Methods', fontsize = 30)
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color) 
    for cap in bp['medians']: cap.set(linewidth = 2) 
    for whisker in bp['whiskers']: whisker.set(linewidth = 1.5)
    for cap in bp['caps']: cap.set(linewidth = 2) 
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')    
    plt.show()
    
def join_results(results, geom_results):
    for weblab_id in results:
        results[weblab_id].pop('weblab')
        geom_results[weblab_id].pop('weblab')
        for treatment in results[weblab_id]:

            results[weblab_id][treatment]['GM'] = geom_results[weblab_id][treatment]['GM']
            results[weblab_id][treatment]['GM_time'] = geom_results[weblab_id][treatment]['GM_time']
       
            
    return results

def join_results_nbp(results, nbp_results):
    for weblab_id in results:

        for treatment in results[weblab_id]:

            results[weblab_id][treatment]['NBP'] = nbp_results[weblab_id][treatment]['NBP_regression']
            results[weblab_id][treatment]['NBP_time'] = nbp_results[weblab_id][treatment]['NBP_regression_time']
       
            
    return results

def survival_plot(results, color_dict, save = False):

    lw = 3
    bandwith=.1
    kernel = 'gaussian'
    lo_retain = 100
    hi_retain = -100
    X_plot = np.linspace(0,1,100).reshape(-1,1)
    lns = (0, (4, .5, 1, 1))

    plt.figure(figsize = (20,4.5))
    
    trues_list = [[results[weblab_id][treatment]['True'][-1] for treatment in results[weblab_id]] for weblab_id in results]
    trues = np.array([item for sublist in trues_list for item in sublist])

    NBP_list = [[results[weblab_id][treatment]['NBP'][-1] for treatment in results[weblab_id]] for weblab_id in results]
    NBP = np.array([item for sublist in NBP_list for item in sublist])
    NBP_precision = 1-np.sort(np.abs(NBP-trues)/trues)
    
    SSP_list = [[results[weblab_id][treatment]['SSP'][-1] for treatment in results[weblab_id]] for weblab_id in results]
    SSP = np.array([item for sublist in SSP_list for item in sublist])
    SSP_precision = 1-np.sort(np.abs(SSP-trues)/trues)
    
    kde = KernelDensity(kernel=kernel, bandwidth=bandwith).fit(SSP_precision.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    dens = np.exp(log_dens)
    dens = dens/dens.sum()
    plt.plot(X_plot[:, 0], 1-dens.cumsum(), color=color_dict['SSP'], lw=lw,
            linestyle='-',  label = 'SSP')
    plt.scatter(X_plot[:, 0][::8], 1-dens.cumsum()[::8], color=color_dict['SSP'], lw=lw,
    marker='X', s= 200)
    
    IBP_list = [[results[weblab_id][treatment]['IBP'][-1] for treatment in results[weblab_id]] for weblab_id in results]
    IBP = np.array([item for sublist in IBP_list for item in sublist])
    IBP_precision = 1-np.sort(np.abs(IBP-trues)/trues)

    kde = KernelDensity(kernel=kernel, bandwidth=bandwith).fit(IBP_precision.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    dens = np.exp(log_dens)
    dens = dens/dens.sum()
    plt.plot(X_plot[:, 0], 1-dens.cumsum(), color=color_dict['IBP'], lw=lw,
            linestyle='-',  label = 'IBP')
    plt.scatter(X_plot[:, 0][::8], 1-dens.cumsum()[::8], color=color_dict['IBP'], lw=lw,
    marker='X', s= 200)
    
    GM_list = [[results[weblab_id][treatment]['GM'][-1] for treatment in results[weblab_id]] for weblab_id in results]
    GM = np.array([item for sublist in GM_list for item in sublist])
    GM_precision = 1-np.sort(np.abs(GM-trues)/trues)
    
    kde = KernelDensity(kernel=kernel, bandwidth=bandwith).fit(GM_precision.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    dens = np.exp(log_dens)
    dens = dens/dens.sum()
    plt.plot(X_plot[:, 0], 1-dens.cumsum(), color=color_dict['GM'], lw=lw,
            linestyle='-',  label = 'GM')
    plt.scatter(X_plot[:, 0][::8], 1-dens.cumsum()[::8], color=color_dict['GM'], lw=lw,
    marker='X', s= 200)
    
    J4_list = [[results[weblab_id][treatment]['J'][-1,-1] for treatment in results[weblab_id]] for weblab_id in results]
    J4 = np.array([item for sublist in J4_list for item in sublist])
    J4_precision = 1-np.sort(np.abs(J4-trues)/trues)
    
    kde = KernelDensity(kernel=kernel, bandwidth=bandwith).fit(J4_precision.reshape(-1,1))
    log_dens = kde.score_samples(X_plot)
    dens = np.exp(log_dens)
    dens = dens/dens.sum()
    plt.plot(X_plot[:, 0], 1-dens.cumsum(), color=color_dict['J'][-1], lw=lw,
            linestyle='-',  label = 'J4')
    plt.scatter(X_plot[:, 0][::8], 1-dens.cumsum()[::8], color=color_dict['J'][-1], lw=lw,
    marker='X', s= 200)
    
    plt.xlim([.5,1])
    plt.ylabel(r'$\mathbb{P}(\eta_{T}^{(T_2)} \geq v)$', fontsize = 20)
    plt.xlabel(r'$v$', fontsize = 20)
    plt.title('Prediction accuracy', fontsize = 20)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.legend(loc='upper center', bbox_to_anchor=(.2, -.2),fancybox=True, shadow=True, ncol=5, fontsize = 20)
    plt.tight_layout()

    if save !=  False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight') 
    plt.show()
    
    
def survival_plot_zipf_synthetic(results_sums, save = False):
    
    accuracy = {}
    for sigma in results_sums:
        accuracy[sigma] = []
        for it in range(len(results_sums[sigma])):
            error = abs(results_sums[sigma][it]['fitted_total_sum_news'] - results_sums[sigma][it]['true_total_sum_news'])/\
                    results_sums[sigma][it]['true_total_sum_news']
            accuracy[sigma].append(1-min(error,1)) 

    lw = 3
    bandwith=.1
    kernel = 'gaussian'
    lo_retain = 100
    hi_retain = -100
    X_plot = np.linspace(0,1,100).reshape(-1,1)
    lns = (0, (4, .5, 1, 1))

    plt.figure(figsize = (10,6))
    color_ls = ['r', 'g', 'b', 'orange']
    linestyles = [':', '--', '-.', '-']
    for t, tau in enumerate(accuracy.keys()):
    
        kde = KernelDensity(kernel=kernel, bandwidth=bandwith).fit(np.array(accuracy[tau]).reshape(-1,1))
        log_dens = kde.score_samples(X_plot)
        dens = np.exp(log_dens)
        dens = dens/dens.sum()
        plt.plot(X_plot[:, 0], 1-dens.cumsum(), color=color_ls[t], lw=lw,
                linestyle=linestyles[t],  label = r'$\tau=$'+str(tau))
        plt.scatter(X_plot[:, 0][::8], 1-dens.cumsum()[::8], color=color_ls[t], lw=lw,
                    marker='X', s= 200)

    plt.xlim([.5,1])
    plt.ylabel(r'$\hat{\mathbb{P}}(\tilde{v}_{D_0}^{(D_2)} \geq v)$', fontsize = 20)
    plt.xlabel(r'$v$', fontsize = 20)
    plt.title(r'Prediction accuracy $T_{D_0}^{(D_1)}$', fontsize = 20)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -.2),fancybox=True, shadow=True, ncol=5, fontsize = 20)
    plt.tight_layout()

    if save !=  False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight') 
    plt.show()

    
def make_ranking(results, filter_threshold = -1):
    accuracy_at_end, _ = make_accuracy_run_time_dicts(results, filter_threshold = filter_threshold)
    ranking = {}
    keys = list(accuracy_at_end.keys())
    for key in keys:
        ranking[key] = []
    number_weblabs = len(accuracy_at_end[keys[0]])
    
    for w in range(number_weblabs):
        vals = np.array([-accuracy_at_end[key][w] for key in keys])
        
        ordering = 1+np.argsort(vals)
        for k, key in enumerate(keys):
            ranking[key].append(ordering[k])

    return ranking

def make_ranking_plot(results, color_dict, filter_threshold = -1, save = False):

    ordering = make_ranking(results, filter_threshold=filter_threshold)
    ordering['BG'] = ordering['BBG']
    ordering.pop('BBG')
    top_k = {}
    for key in ordering.keys():
        top_k[key] = [np.sum(ordering[key]<= k)/len(ordering[key]) for k in np.arange(1,4)]

    plt.figure(figsize = (20,4.5))
    for place in np.arange(1,4):
        plt.subplot(1, 3, place)
        for k, key in enumerate(top_k.keys()):
            if key[0] == 'J':
                color_ = color_dict['J'][int(key[1])-1]
            else:
                color_ = color_dict[key]
            plt.hlines(xmin = 0, xmax = 100*top_k[key][place-1], y = k, color = color_)
        plt.title(r'% Rank $\leq$'+str(place), fontsize = 20)
        plt.xticks(np.linspace(0,60,7), [str(i) + str('%') for i in np.linspace(0,60,7,dtype = int)], fontsize = 20)
        plt.yticks(np.arange(len(top_k.keys())), list(top_k.keys()), fontsize = 20)
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight') 
    plt.show()