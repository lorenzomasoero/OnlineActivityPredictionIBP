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
        NBP_r = accuracy_at_end['NBP_fixed_r']
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
                  color_dict['LP'], 
                  color_dict['J'][2], 
                  color_dict['J'][3]]

        
        bp = plt.boxplot([np.array(NBP_r), 
                          np.array(SSP), 
                          np.array(IBP), 
                          np.array(BB), 
                          np.array(LP), 
                          np.array(J3), 
                          np.array(J4)], labels=labs, vert = True, **black_dict)

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
        plt.title(r'$\tau = $'+str(tau), fontsize = 20)
        c+=1
    plt.tight_layout()
    if save != False:
        plt.savefig(save+'.pdf', bbox_inches = 'tight')
    plt.show()    

def make_accuracy_plots_zipf_all(results_all, 
                                 color_dict, 
                                 filter_threshold = -1, 
                                 save = False, 
                                 include_regression = False):
    
    plt.figure(figsize = (14,12))
    c=1
    for tau, results in results_all.items():
        plt.subplot(2,2,c)
        accuracy_at_end = make_accuracy_dicts_zipf(results, 
                                                   filter_threshold = filter_threshold)
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

        labs = ['NBP', 
                'NBP_like', 
                'SSP', 
                'SSP_like', 
                'IBP', 
                'IBP_like', 
                'BB', 
                'BG', 
                'LP', 
                'GT', 
                'J1', 
                'J2', 
                'J3', 
                'J4'] 
        colors = [color_dict['NBP'], 
                  color_dict['NBP'], 
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
                  color_dict['J'][3], 
                  color_dict['GM']]

        
        bp = plt.boxplot([np.array(NBP), 
                          np.array(NBP_like), 
                          np.array(SSP), 
                          np.array(SSP_like), 
                          np.array(IBP), 
                          np.array(IBP_like), 
                          np.array(BB), 
                          np.array(BG), 
                          np.array(LP), 
                          np.array(GT), 
                          np.array(J1), 
                          np.array(J2), 
                          np.array(J3), 
                          np.array(J4)], 
                          labels=labs, vert = False, **black_dict)
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
            'NBP_fixed_r',
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

    labs = ['NBP', 
            'SSP', 
            'IBP', 
            'BB', 
            'BG', 
            'LP', 
            'GT', 
            'J1', 
            'J2', 
            'J3', 
            'J4', 
            'GM']
    colors = [color_dict['NBP'], 
              color_dict['SSP'], 
              color_dict['IBP'], 
              color_dict['BB'], 
              color_dict['BG'], 
              color_dict['LP'], 
              color_dict['GT'], 
              color_dict['J'][0], 
              color_dict['J'][1], 
              color_dict['J'][2], 
              color_dict['J'][3], 
              color_dict['GM']]
    plt.figure(figsize = (20,6))
    bp = plt.boxplot([np.array(SSP),
                      np.array(IBP), 
                      np.array(BB), 
                      np.array(BG), 
                      np.array(LP), 
                      np.array(GT), 
                      np.array(J1), 
                      np.array(J2),
                      np.array(J3), 
                      np.array(J4), 
                      np.array(GM)], 
                     labels=labs, vert = False, **black_dict)
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
    
    
def survival_plot_zipf(results_sums, save = False):
    
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