from scipy.io import loadmat 
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import pyPartAnalysis.plot as ppa_plt

class ProfmonImage:
    
    def __init__(self,file_name: str,center_method: str='weighted mean'):
        #Read in profmon matlab structure
        image_data = loadmat(file_name=file_name)
        field_names = image_data['data'][0,0].dtype.names
    
        data_dict = {field:np.squeeze(dat) for dat,field in zip(image_data['data'][0,0],field_names)} 
        
        self.data = data_dict
        self.center_method = center_method.lower()
        _ = self.extent()
        
    def save(self,file_name: str):
        im = Image.fromarray(self.data['img'])
        im.save(file_name)
    
    def extent(self):
        # Gets x,y extents of image from CXLS profmon image struct.
        #
        # 0 intercept of image can be set either by the weighted 
        # mean or the peak value of the summed profiles.

        max_x = self.data['roiXN']*self.data['res'][0]*1e-6
        max_y = self.data['roiYN']*self.data['res'][1]*1e-6
        min_y = 0
        min_x = 0

        sum_y = np.sum(self.data['img'], axis=0)
        sum_x = np.sum(self.data['img'], axis=1)

        x_val = np.linspace(min_x,max_x,self.data['roiXN'])
        y_val = np.linspace(max_y,min_y,self.data['roiYN'])

        if(self.center_method=='weighted mean'):
            shift_x = np.average(x_val, weights=sum_y)
            shift_y = np.average(y_val, weights=sum_x)
        elif(self.center_method.lower()=='peak'):
            shift_x = x_val[np.argmax(sum_y)]
            shift_y = y_val[np.argmax(sum_x)]

        self.max_x = max_x - shift_x
        self.min_x = min_x - shift_x
        self.max_y = max_y - shift_y
        self.min_y = min_y - shift_y
        
        return [[self.min_x,self.max_x],[self.min_y,self.max_y]]
    
    def linspace_x(self):
        return np.linspace(self.min_x,self.max_x,self.data['roiXN'])
    
    def linspace_y(self):
        return np.linspace(self.min_y,self.max_y,self.data['roiYN'])
    
    def profile_x(self):
        return np.sum(self.data['img'], axis=0)

    def profile_y(self):
        return np.sum(self.data['img'], axis=1)
    
    def plot_profile(self,dim: str, ax=None, **kwargs):
        # plots x or y profile. dim is either 'x' or 'y'.
        
        if ax is None:
            ax = plt.gca()
            
        scale_info = ppa_plt.det_plot_scale(pd.DataFrame({'x':[self.min_x,self.max_x],
                                                          'y':[self.min_y,self.max_y]}))
        
        if dim == 'x':
            ax.set(xlabel=ppa_plt.make_phase_space_axis_labels('x',scale_info['x']),
                   ylabel='Counts (arb. units)')  
            ax.plot(self.linspace_x()*10**-scale_info['x'],self.profile_x())
        elif dim == 'y':
            ax.set(xlabel=ppa_plt.make_phase_space_axis_labels('y',scale_info['y']),
                   ylabel='Counts (arb. units)') 
            ax.plot(self.linspace_y()*10**-scale_info['y'],self.profile_y())
            
        return ax
            
    
    def plot(self,ax=None,**kwargs):
        if ax is None:
            ax = plt.gca()
        scale_info = ppa_plt.det_plot_scale(pd.DataFrame({'x':[self.min_x,self.max_x],
                                                          'y':[self.min_y,self.max_y]}))
        
        ax.set(xlabel=ppa_plt.make_phase_space_axis_labels('x',scale_info['x']),
               ylabel=ppa_plt.make_phase_space_axis_labels('y',scale_info['y']))  
        
        extent = [self.min_x*10**-scale_info['x'],self.max_x*10**-scale_info['x'],
                  self.min_y*10**-scale_info['y'],self.max_y*10**-scale_info['y']]
        return ax.imshow(self.data['img'],extent=extent,**kwargs)
