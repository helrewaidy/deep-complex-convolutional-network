
import torch
from torch.nn.parameter import Parameter

import numpy as np
import torch.nn as nn


def polarToCylindricalConversion(input):
    real = input[:,:,:,:,0]*torch.cos(input[:,:,:,:,1])
    imag = input[:,:,:,:,0]*torch.sin(input[:,:,:,:,1])
    return torch.stack([real, imag],dim=4)

def cylindricalToPolarConversion(input):
    mag = (input[:,:,:,:,0]**2 + input[:,:,:,:,1]**2)**(0.5)
    phase = torch.atan(input[:,:,:,:,1] / input[:,:,:,:,0])
    phase[phase.ne(phase)] = 0.0 #remove NANs
    return torch.stack([mag, phase], dim=4)


def magnitudeBatchStandardize(input, epsilon=1e-4, polar=False):
    if not polar:
        input = cylindricalToPolarConversion(input)
    
    real_input = input[:,:,:,:,0].unsqueeze(-1);
    nn.BatchNorm2d(real_input)
    
    ndims= real_input.ndimension()
    real_axes = [ndims-3,ndims-2] #2,3
    mu = torch.mean(torch.mean(real_input, real_axes[0], True), real_axes[1], True)
    real_input_centered = real_input - mu
    real_input_centered_squared = real_input_centered ** 2
    
    Vrr = torch.mean(torch.mean(
            input_centered_squared[:,:,:,:,0], input_axes[0] ,True),
            input_axes[1] ,True) + epsilon
    Vii = torch.mean(torch.mean(
            input_centered_squared[:,:,:,:,1], input_axes[0] ,True),
            input_axes[1] ,True) + epsilon
    Vri = torch.mean(torch.mean(
            input_centered[:,:,:,:,0] * input_centered[:,:,:,:,1],
            input_axes[0] ,True), input_axes[1] ,True)
    
    tau = Vrr + Vii
    delta = (Vrr * Vii) - (Vri ** 2)
    
    s = torch.sqrt(delta)
    t = torch.sqrt(tau + 2 * s) 
    
    st_invrs = 1.0 / (s * t)
    Wrr = (Vii + s) * st_invrs 
    Wii = (Vrr + s) * st_invrs 
    Wri = -Vri * st_invrs 
    
    brdcst_Wrr = Wrr.expand_as(input[:,:,:,:,1].squeeze(-1))
    brdcst_Wii = Wii.expand_as(input[:,:,:,:,1].squeeze(-1))
    brdcst_Wri = Wri.expand_as(input[:,:,:,:,1].squeeze(-1))
    
    rolled_input = torch.stack([input_centered[:,:,:,:,1], input_centered[:,:,:,:,0]], dim=ndims-1)
    
    stndrzd_output = torch.stack([brdcst_Wrr,brdcst_Wii],dim=ndims-1) * input_centered + torch.stack([brdcst_Wri,brdcst_Wri],dim=ndims-1) * rolled_input
    
    if not polar:
        stndrzd_output = polarToCylindricalConversion(stndrzd_output) 
    
    return stndrzd_output       

class MagnitudeBatchNormalize(nn.Module):
    
    def __init__(self, num_features ,epsilon=1e-4, momentum=0.1, center=True, scale=True, track_running_stats=True, polar=False):
        super(MagnitudeBatchNormalize, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.center = center
        self.scale = scale
        self.track_running_stats = track_running_stats
        self.polar = polar
        
        if self.center:
            self.register_buffer('beta', torch.zeros(self.num_features))
            if self.track_running_stats:
                self.register_buffer('running_mean', torch.zeros(self.num_features))
        else: 
            self.register_buffer('beta', None)
            if self.track_running_stats:
                self.register_buffer('running_mean', None)
            
        if self.scale:
            self.register_buffer('gamma_rr', torch.zeros(self.num_features))
            self.register_buffer('gamma_ii', torch.zeros(self.num_features))
            self.register_buffer('gamma_ri', torch.zeros(self.num_features))
            if self.track_running_stats:
                self.register_buffer('running_Vrr', torch.zeros(self.num_features))
                self.register_buffer('running_Vii', torch.zeros(self.num_features))
                self.register_buffer('running_Vri', torch.zeros(self.num_features))

        else:
            self.register_parameter('gamma_rr', None)
            self.register_parameter('gamma_ii', None)
            self.register_parameter('gamma_ri', None)
            if self.track_running_stats:
                self.register_parameter('running_Vrr', None)
                self.register_parameter('running_Vii', None)
                self.register_parameter('running_Vri', None)
                        
        self.reset_parameters()
     
    def reset_parameters(self):
        if self.center:
            self.beta.zero_()
            if self.track_running_stats:
                self.running_mean.zero_()
            
        if self.scale:        
            self.gamma_rr.fill_(1.0/np.sqrt(2))
            self.gamma_ii.fill_(1.0/np.sqrt(2))
            self.gamma_ri.fill_(1.0/np.sqrt(2))
            if self.track_running_stats:
                self.running_Vrr.fill_(1.0/np.sqrt(2))
                self.running_Vii.fill_(1.0/np.sqrt(2))
                self.running_Vri.zero_()
           
    def normalizeComplexBatch(self, input):
        ''' normalize each slice alone for 5D input only'''
        if not self.polar:
            input = cylindricalToPolarConversion(input)
            
        ndims= input.ndimension()
        input_axes = [ndims-3,ndims-2] #2,3
        mu = torch.mean(torch.mean(input, input_axes[0], True), input_axes[1], True)
        input_centered = input - mu
        
        input_centered_squared = input_centered ** 2
    
        Vrr = torch.mean(torch.mean(
                input_centered_squared[:,:,:,:,0], input_axes[0] ,True),
                input_axes[1] ,True) + self.epsilon
        Vii = torch.mean(torch.mean(
                input_centered_squared[:,:,:,:,1], input_axes[0] ,True),
                input_axes[1] ,True) + self.epsilon
        Vri = torch.mean(torch.mean(
                input_centered[:,:,:,:,0] * input_centered[:,:,:,:,1],
                input_axes[0] ,True), input_axes[1] ,True)
        
        if self.scale:
            stndrzd_output = magnitudeBatchStandardize(input, self.epsilon, not self.polar)
            
            brdcst_gamma_rr = self.gamma_rr.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(Vrr)
            brdcst_gamma_ii = self.gamma_ii.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(Vrr)
            brdcst_gamma_ri = self.gamma_ri.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(Vrr)
            
            rolled_stndrzd_output = torch.stack([stndrzd_output[:,:,:,:,1], stndrzd_output[:,:,:,:,0]], dim=ndims-1)
            
            if self.center:
                brdcst_beta = self.beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(mu)
                output = torch.stack([brdcst_gamma_rr,brdcst_gamma_ii],dim=ndims-1) * stndrzd_output + torch.stack([brdcst_gamma_ri,brdcst_gamma_ri],dim=ndims-1) * rolled_stndrzd_output + brdcst_beta 
            else:
                output = torch.stack([brdcst_gamma_rr,brdcst_gamma_ii],dim=ndims-1) * stndrzd_output + torch.stack([brdcst_gamma_ri,brdcst_gamma_ri],dim=ndims-1) * rolled_stndrzd_output
        else:
            if self.center:
                brdcst_beta = self.beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(mu)
                output = input_centered + brdcst_beta
            else:
                output = input_centered
        
        if self.training and self.track_running_stats:
            def update_running_average(current_avg, running_avg, momentum=self.momentum):
                return (1 - momentum) * running_avg + momentum * current_avg
            model_dict = self.state_dict()
            model_dict['running_mean'] = update_running_average(mu[0,:,0,0,0], model_dict['running_mean'])
            model_dict['running_Vrr'] = update_running_average(Vrr[0,:,0,0], model_dict['running_Vrr'])
            model_dict['running_Vii'] = update_running_average(Vii[0,:,0,0], model_dict['running_Vii'])
            model_dict['running_Vri'] = update_running_average(Vri[0,:,0,0], model_dict['running_Vri'])
            self.load_state_dict(model_dict)
        
        if not self.polar:
            output = cylindricalToPolarConversion(output)
            
        return output 

    def forward(self, input):
        return self.normalizeComplexBatch(input)
    
    
    

    
    
    
