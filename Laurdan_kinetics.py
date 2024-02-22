# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:45:33 2023

@author: sacco004
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import numpy as np
import easygui as egui
import os

def GP_calculator(x,y_blue,y_red,lower_xbound,upper_xbound):
    idx = np.where((np.array(x)>=lower_xbound) & (np.array(x)<=upper_xbound))[0] #define x range
    integrated_value_blue = sp.integrate.simpson(x=np.array(x)[idx],y=np.array(y_blue)[idx])
    integrated_value_red = sp.integrate.simpson(x=np.array(x)[idx],y=np.array(y_red)[idx])
    GP = (integrated_value_blue-integrated_value_red)/(integrated_value_blue+integrated_value_red)
    return integrated_value_blue, integrated_value_red, GP

#Double Exponential function for tail fitting
def exp_decay(x,a1,a2,k1,k2,b):
    return a1*(1-np.exp(-k1*x)) + a2*(1-np.exp(-k2*x)) + b

#Double Exponential function for tail fitting
def mono_exp_decay(x,a,k1):
    return a*(np.exp(-k1*x))

#Loop for fitting the data
def fit_data(x,y, p0):
    
    # Perform fitting
    popt, pcov = curve_fit(exp_decay, x, y, p0=p0)

    # Define array containing the fitting results
    y_fit = exp_decay(x, *popt)

    Res = np.array([x, y, y_fit, y - y_fit])

    std = get_stderror(popt,pcov)
    adjusted_R2 = calc_adjusted_R_sq(np.vstack([x.values, y.values]).T,exp_decay,popt)
    avg_gp = (popt[0]*(popt[1]**2)+popt[2]*(popt[3]**2))/(popt[0]*popt[1]+popt[2]*popt[3])

    #Create a DataFrame with the array containing fit values
    df_Res = pd.DataFrame(Res,index=None,columns=None)
    df_Res = df_Res.T
    df_Res.columns=['PAR','Data','Fit','Residuals']
    
    #Create a new dataframe containing fit parameters and errors
    df_params = pd.DataFrame(popt,columns=['Parameters(a,k,b)'])
    df_params['Standard Dev'] = std
    df_params['Adjusted R2'] = adjusted_R2
    df_params['Avg time-resolved GP'] = avg_gp

    #Add them to the original dataframe
    df_Res = pd.concat([df_Res,df_params],axis=1,ignore_index=False)
    df_Res.set_index('PAR',inplace=True) 

    return df_Res

# Get standard errors
def get_stderror(popt,pcov):
    error = []
    for ii in range(len(popt)):
        try:
            error.append(np.absolute(pcov[ii][ii])**0.5)
        except:
            error.append(0.)
    return error

# Get corrected R2
def calc_adjusted_R_sq(data_to_fit, fit_model, popt_now):
    residuals = data_to_fit[:,1] - fit_model(data_to_fit[:,0], *popt_now)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data_to_fit[:,1] - np.mean(data_to_fit[:,1]))**2)
    R_sq = 1 - (ss_res/ss_tot)
    adjusted_R_sq = 1 - (((1 - R_sq**2) * (len(data_to_fit[:,1]) - 1)) / (len(data_to_fit[:,1]) - len(popt_now) - 1))
    return adjusted_R_sq

path = str(egui.fileopenbox("Select Files to Import"))
current_dir = str(os.path.dirname('/'.join(path.split('\\'))))
os.chdir(current_dir)
files = filter(os.path.isfile, os.listdir( os.curdir))
files = [ff for ff in os.listdir('.') if os.path.isfile(ff)]
data_files = [x for x in files if (x.endswith(".csv")) and ('TRES' in x)]
df={}
for i in data_files:
    name = i.split(' ')[1].split('.')[0]
    df[name]=pd.read_csv(i)
    df[name].dropna(axis=1,inplace=True)
    # type(df[name][420.0].iloc[15])

    df[name].columns = ['Time (ns)'] + list(np.linspace(420,600,19))
    # df[name].apply(pd.to_numeric, 'coerce')
    df[name]['avg_blue']=df[name].loc[:,420.0:490.0].mean(axis=1)
    df[name]['avg_red']=df[name].loc[:,510.0:580.0].mean(axis=1)
    df[name]['avg_all']=df[name].loc[:,420.0:580.0].mean(axis=1)
    df[name]['Time-resolved GP']=(df[name]['avg_blue']-df[name]['avg_red'])/df[name]['avg_all']

x_range=[0,50]

fig, ax = plt.subplots(2,3,figsize=(10,6),layout='tight')
ax[0][1].set_xlabel('Time (ns)') 
ax[0][0].set_ylabel('Photon counts') 
ax[1][0].set_ylabel('Time (ns)') 
ax[1][1].set_xlabel('Wavelength (nm)')
# ax[1][2].set_ylabel('Photon counts') 
# ax[2][2].set_xlabel('Wavelength (nm)')

for i,sample in enumerate(df):
    #Plot average decays
    
    ax[0][i].spines[['right', 'top']].set_visible(False)
    ax[0][i].plot(df[sample]['Time (ns)'],df[sample]['avg_blue'],color='blue', label='Blue-shifted emission')
    ax[0][i].plot(df[sample]['Time (ns)'],df[sample]['avg_red'],color='red', label='Red-shifted emission')
    # ax[0][i].plot(df[sample]['Time (ns)'],df[sample]['avg_all'],color='k', label='Mean \'total\'')
    ax[0][0].legend(frameon=False)
    ax[0][i].set_yscale('log')
    ax[0][i].set_ylim(0.6,2000)
    ax[0][i].set_xlim(0,50)

    
    #Plot contour plots
    ax[0][i].set_title(f"{sample.strip('EmMap'+'deg')}$^\circ$C", weight='bold',fontsize=14)
    lvls = np.logspace(0,900,9)
    CF = ax[1][i].contourf(np.linspace(430,600,18),df[sample]['Time (ns)'],df[sample].loc[:,430.0:600.0],
                     levels=100, cmap="RdBu_r",vmin=50,vmax=900)
    ax[1][i].contour(np.linspace(430,600,18),df[sample]['Time (ns)'],df[sample].loc[:,430.0:600.0],
                     levels=10,colors='k',linewidths=0.5, alpha=0.7)
    
    ax[1][i].set_yscale('linear')
    ax[1][i].set_ylim(1,12)
    
    
#     #Plot TRES
#     time_stamps = [0,1.5,3,4.5,5,6]
#     for time in time_stamps:

#         loc_time=df[sample].loc[df[sample]['Time (ns)'] == time+3.3].index[0]
#         y= df[sample].loc[loc_time,420.0:600.0]/np.max(df[sample].loc[loc_time,450.0:600.0])
#         ax[i][2].plot(np.linspace(420,600,19),y,'-o',markersize=3,label=f"{time} ns")
#         ax[i][2].set_ylim(0,1.1)
#         ax[i][2].spines[['right', 'top']].set_visible(False)
#         ax[i][2].legend(fontsize=10,loc = 'upper right', bbox_to_anchor= (1.35,1),frameon=False)
plt.colorbar(CF)
plt.savefig('Laurdan_lifetimes_contours.svg')
plt.savefig('Laurdan_lifetimes_contours.png',dpi=1200)
plt.show()

#Calculate GP values
GPs = []
for i,sample in enumerate(df):
    x= df[sample]['Time (ns)']
    y_blue=df[sample]['avg_blue']
    y_red=df[sample]['avg_red']
    lower_xbound=x_range[0]
    upper_xbound=x_range[1]
    integrated_value_blue, integrated_value_red, GP = GP_calculator(x,y_blue,y_red,lower_xbound,upper_xbound)
    GPs.append(GP)
    
fig, ax = plt.subplots(figsize=(4,3),layout='tight')
ax.plot([4,25,37],[0.61,0.38,0.33],'-o',color = 'b',label='Steady-state GP (from literature)')
ax.plot([4,20,45],GPs,'-o',color = 'k',label='Time-resolved GP (calculated)')

ax.legend(loc=(0.05,0.29),frameon=False,fontsize=10)
ax.set_xlabel('Temperature ($^\circ$C)') 
ax.set_ylabel('GP value') 
plt.savefig('Laurdan_SS_GPs.svg')
plt.savefig('Laurdan_SS_GPs.png',dpi=1200)
plt.show()

#Time-resolved GPs
fig, ax = plt.subplots(3,2,figsize=(8,10),layout='tight')
ax[1][0].set_ylabel('Time-resolved GP')
ax[2][0].set_xlabel('Time (ns)')
ax[2][1].set_xlabel('Time (ns)')
df_fit={}
for i,sample in enumerate(df):
    p0=[1,1,0.2,1.5,1]
    x=df[sample].loc[:1360,'Time (ns)']
    y=df[sample].loc[277:1637,'Time-resolved GP'].reset_index()
    del y['index']
    y=y['Time-resolved GP']
    df_fit[sample]=fit_data(x,y, p0)

    ax[i][0].set_xlim(0,20)
    ax[i][0].set_ylim(-1.5,1.5)
    ax[i][0].text(0.75,0.75,f"{sample.strip('EmMap'+'deg')}$^\circ$C", transform=ax[i][0].transAxes, weight='bold',fontsize=14)

    ax[i][0].plot(df[sample]['Time (ns)'],df[sample]['Time-resolved GP'],color='k', label='Time-resolved GP',alpha=0.5)

    ax[i][0].plot(df[sample].loc[277:1637,'Time (ns)'],df_fit[sample]['Fit'],color='k')  
    
    ax[i][1].plot(x,mono_exp_decay(x, df_fit[sample].loc[0.01,'Parameters(a,k,b)'], df_fit[sample].loc[0.04,'Parameters(a,k,b)']))
    ax[i][1].plot(x,mono_exp_decay(x, df_fit[sample].loc[0.02,'Parameters(a,k,b)'], df_fit[sample].loc[0.05,'Parameters(a,k,b)']))
    
    ax[i][1].set_ylim(-1.25,0.05)
    
plt.savefig('Laurdan_TR_GPs.svg')
plt.show()

#Time-resolved GPs - components
y_WT = [x for x in df]
y_fad = [x for x in df if 'fad' in x]

fig, ax = plt.subplots(3,2,figsize=(8,10),layout='tight')
ax[1][0].set_ylabel('Time-resolved GP')
ax[2][0].set_xlabel('Time (ns)')
ax[2][1].set_xlabel('Time (ns)')
df_fit_WT={}
df_fit_fad={}

for i,sample in enumerate(y_WT):
    p0=[1,1,0.2,1.5,1]
    x=df[sample].loc[:1300,'Time (ns)']
    y=df[sample].loc[287:1587,'Time-resolved GP'].reset_index()
    del y['index']
    y=y['Time-resolved GP']
    df_fit_WT[sample]=fit_data(x,y, p0)
    df_fit_WT[sample]=df_fit_WT[sample].reset_index()
    ax[i][0].set_xlim(0,20)
    # ax[i][0].set_ylim(0,1.5)
    ax[i][0].text(0.75,0.75,f"{sample.split('_')[0].strip('EmMap'+'deg')}$^\circ$C", transform=ax[i][0].transAxes, weight='bold',fontsize=14)
    ax[i][0].plot(df[sample]['Time (ns)'],df[sample]['Time-resolved GP'],color='k', label='Time-resolved GP',alpha=0.5)
    ax[i][0].plot(df[sample].loc[287:1587,'Time (ns)'],df_fit_WT[sample]['Fit'],color='k')  
    ax[i][1].plot(x,-mono_exp_decay(x, df_fit_WT[sample].loc[0,'Parameters(a,k,b)'], df_fit_WT[sample].loc[2,'Parameters(a,k,b)']))
    ax[i][1].plot(x,-mono_exp_decay(x, df_fit_WT[sample].loc[1,'Parameters(a,k,b)'], df_fit_WT[sample].loc[3,'Parameters(a,k,b)']))
    # ax[i][1].set_ylim(-1.25,0.05)
plt.savefig('Laurdan_TR_GPs_final.svg')
plt.show()

# for i,sample in enumerate(y_fad):
#     p0=[1,1,0.2,1.5,1]
#     x=df[sample].loc[:1300,'Time (ns)']
#     y=df[sample].loc[337:1637,'Time-resolved GP'].reset_index()
#     del y['index']
#     y=y['Time-resolved GP']
#     df_fit_fad[sample]=fit_data(x,y, p0)
#     df_fit_fad[sample]=df_fit_fad[sample].reset_index()
#     ax[i][0].set_xlim(0,20)
#     ax[i][0].set_ylim(0,1.5)
#     ax[i][0].text(0.75,0.75,f"{sample.split('_')[0].strip('EmMap'+'deg')}$^\circ$C", transform=ax[i][0].transAxes, weight='bold',fontsize=14)
#     ax[i][0].plot(df[sample]['Time (ns)'],df[sample]['Time-resolved GP'],color='k', label='Time-resolved GP',alpha=0.5)
#     ax[i][0].plot(df[sample].loc[337:1637,'Time (ns)'],df_fit_fad[sample]['Fit'],color='k')  
#     ax[i][1].plot(x,mono_exp_decay(x, df_fit_fad[sample].loc[0,'Parameters(a,k,b)'], df_fit_fad[sample].loc[2,'Parameters(a,k,b)']))
#     ax[i][1].plot(x,mono_exp_decay(x, df_fit_fad[sample].loc[1,'Parameters(a,k,b)'], df_fit_fad[sample].loc[3,'Parameters(a,k,b)']))
#     ax[i][1].set_ylim(-1.25,0.05)
    
# plt.savefig('Laurdan_TR_GPs_fad.svg')
# plt.show()

#Time-resolved GPs - final graph
GPs_WT = []
GPs_fad = []
for i,sample in enumerate(y_WT):
    GPs_WT.append(df_fit_WT[sample].loc[0,'Avg time-resolved GP'])
# for i,sample in enumerate(y_fad):
#     GPs_fad.append(df_fit_fad[sample].loc[0,'Avg time-resolved GP'])
    
fig, ax = plt.subplots(figsize=(4,3),layout='tight')
ax.scatter([4,20,45],GPs_WT,color='k')
# ax.scatter([4,20,45],GPs_fad,color='r')
ax.set_xlabel('Temperature$^\circ$C') 
ax.set_ylabel('Avg GP T') 
plt.savefig('Laurdan_GPs_final.svg')
plt.show()

