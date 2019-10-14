# For 2 legged monopod on board
from pyomo.environ import*
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from IPython.display import HTML
import numpy as np

from scipy.interpolate import interp1d
import time
import winsound

def define_EOMs(m,BW,lambEOMx_board,lambEOMy_board,lambEOMth_board,lambEOMxp_biped,lambEOMyp_biped,lambEOMthp_biped,
        lambEOMthb_biped,lambEOMthll_biped,lambEOMthlr_biped,lambEOMrl_biped,lambEOMrr_biped,lambEOMthal_biped,lambEOMthar_biped):
    # EOMS OF SKATEBOARD -------------------------------------------------------------------------------------------------------
    def EOMx_board(m,n):
        BFx_in = BW*m.Fb_b_total[n,'BF','x_b']
        BFy_in = BW*m.Fb_b_total[n,'BF','y_b']
        FFx_in = BW*m.Fb_b_total[n,'FF','x_b']
        FFy_in = BW*m.Fb_b_total[n,'FF','y_b']
        GBW_in = BW*m.GRF[n,'BW']
        GFW_in = BW*m.GRF[n,'FW']

        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g, m.mbd, m.lbd, m.lbrd, m.Inbd, m.hbd, m.q[n,'x'], m.q[n,'y'], m.q[n,'th'],
                   m.dq[n,'x'], m.dq[n,'y'], m.dq[n,'th'], m.ddq[n,'x'], m.ddq[n,'y'], m.ddq[n,'th'],
                   m.rF[n,'BF'], m.rF[n,'FF'],
                   BFx_in, BFy_in, FFx_in, FFy_in, 
                   GBW_in, GFW_in]
        return lambEOMx_board(*var_list) == 0
    m.EOMx_board = Constraint(m.N, rule = EOMx_board)

    def EOMy_board(m,n):
        BFx_in = BW*m.Fb_b_total[n,'BF','x_b']
        BFy_in = BW*m.Fb_b_total[n,'BF','y_b']
        FFx_in = BW*m.Fb_b_total[n,'FF','x_b']
        FFy_in = BW*m.Fb_b_total[n,'FF','y_b']
        GBW_in = BW*m.GRF[n,'BW']
        GFW_in = BW*m.GRF[n,'FW']

        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g, m.mbd, m.lbd, m.lbrd, m.Inbd, m.hbd, m.q[n,'x'], m.q[n,'y'], m.q[n,'th'],
                   m.dq[n,'x'], m.dq[n,'y'], m.dq[n,'th'], m.ddq[n,'x'], m.ddq[n,'y'], m.ddq[n,'th'],
                   m.rF[n,'BF'], m.rF[n,'FF'],
                   BFx_in, BFy_in, FFx_in, FFy_in, 
                   GBW_in, GFW_in]
        return lambEOMy_board(*var_list) == 0
    m.EOMy_board = Constraint(m.N, rule = EOMy_board)

    def EOMth_board(m,n):
        BFx_in = BW*m.Fb_b_total[n,'BF','x_b']
        BFy_in = BW*m.Fb_b_total[n,'BF','y_b']
        FFx_in = BW*m.Fb_b_total[n,'FF','x_b']
        FFy_in = BW*m.Fb_b_total[n,'FF','y_b']
        GBW_in = BW*m.GRF[n,'BW']
        GFW_in = BW*m.GRF[n,'FW']

        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g, m.mbd, m.lbd, m.lbrd, m.Inbd, m.hbd, m.q[n,'x'], m.q[n,'y'], m.q[n,'th'],
                   m.dq[n,'x'], m.dq[n,'y'], m.dq[n,'th'], m.ddq[n,'x'], m.ddq[n,'y'], m.ddq[n,'th'],
                   m.rF[n,'BF'], m.rF[n,'FF'],
                   BFx_in, BFy_in, FFx_in, FFy_in, 
                   GBW_in, GFW_in]
        return lambEOMth_board(*var_list) == 0
    m.EOMth_board = Constraint(m.N, rule = EOMth_board)

    # EOMs OF BIPED ----------------------------------------------------------------------------------------------------------------
    def EOMxp_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMxp_biped(*var_list) == 0
    m.EOMxp_biped = Constraint(m.N, rule = EOMxp_biped)

    def EOMyp_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMyp_biped(*var_list) == 0
    m.EOMyp_biped = Constraint(m.N, rule = EOMyp_biped)
    
    def EOMthp_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMthp_biped(*var_list) == 0
    m.EOMthp_biped = Constraint(m.N, rule = EOMthp_biped)
    
    def EOMthb_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMthb_biped(*var_list) == 0
    m.EOMthb_biped = Constraint(m.N, rule = EOMthb_biped)
    
    def EOMthal_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMthal_biped(*var_list) == 0
    m.EOMthal_biped = Constraint(m.N, rule = EOMthal_biped)
    
    def EOMthar_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMthar_biped(*var_list) == 0
    m.EOMthar_biped = Constraint(m.N, rule = EOMthar_biped)
    
    def EOMthll_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMthll_biped(*var_list) == 0
    m.EOMthll_biped = Constraint(m.N, rule = EOMthll_biped)

    def EOMrl_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMrl_biped(*var_list) == 0
    m.EOMrl_biped = Constraint(m.N, rule = EOMrl_biped)

    def EOMthlr_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMthlr_biped(*var_list) == 0
    m.EOMthlr_biped = Constraint(m.N, rule = EOMthlr_biped)

    def EOMrr_biped(m,n):
        GRFx_l_in = BW*m.GRFbf[n,'BF','X']
        GRFx_r_in = BW*m.GRFbf[n,'FF','X']
        GRFy_l_in = BW*m.GRFbf[n,'BF','Y']
        GRFy_r_in = BW*m.GRFbf[n,'FF','Y']
        F_a_l_in = BW*(m.F_a[n,'BL'] + m.F_r[n,'BL','ps'] - m.F_r[n,'BL','ng'])
        F_a_r_in = BW*(m.F_a[n,'FL'] + m.F_r[n,'FL','ps'] - m.F_r[n,'FL','ng'])
        tau_a_b_in = BW*m.tau_b[n]
        tau_a_l_in = BW*m.tau_a[n,'BL']
        tau_a_r_in = BW*m.tau_a[n,'FL']
        tau_arm_l_in = BW*m.tau_arm[n,'BA']
        tau_arm_r_in = BW*m.tau_arm[n,'FA']
            
        # list the model versions of all quantities in the same order as sym_list
        var_list = [m.g,m.m[('body',1)],m.m[('left_arm',1)],m.m[('right_arm',1)],m.m[('pelvis',1)],m.m[('left_leg',1)],
                    m.m[('left_leg',2)],m.m[('right_leg',1)],m.m[('right_leg',2)],m.len[('body',1)],m.len[('left_arm',1)],m.lp2a,
                    m.len[('right_arm',1)],m.len[('pelvis',1)],m.len[('left_leg',1)],m.len[('left_leg',2)],m.len[('right_leg',1)],
                    m.len[('right_leg',2)],m.In[('body',1)],m.In[('left_arm',1)],m.In[('right_arm',1)],m.In[('pelvis',1)],
                    m.In[('left_leg',1)],m.In[('left_leg',2)],m.In[('right_leg',1)],m.In[('right_leg',2)],m.q[n,'xp'],m.q[n,'yp'],
                    m.q[n,'thp'],m.q[n,'thb'],m.q[n,'tha_l'],m.q[n,'tha_r'],m.q[n,'thl_l'],m.q[n,'thl_r'],m.q[n,'r_l'],m.q[n,'r_r'],
                    m.dq[n,'xp'],m.dq[n,'yp'],m.dq[n,'thp'],m.dq[n,'thb'],m.dq[n,'tha_l'],m.dq[n,'tha_r'],m.dq[n,'thl_l'],m.dq[n,'thl_r'],
                    m.dq[n,'r_l'],m.dq[n,'r_r'],m.ddq[n,'xp'],m.ddq[n,'yp'],m.ddq[n,'thp'],m.ddq[n,'thb'],m.ddq[n,'tha_l'],m.ddq[n,'tha_r'],
                    m.ddq[n,'thl_l'],m.ddq[n,'thl_r'],m.ddq[n,'r_l'],m.ddq[n,'r_r'], 
                    F_a_l_in, F_a_r_in, tau_a_b_in, tau_arm_l_in, tau_arm_r_in, tau_a_l_in, tau_a_r_in, GRFx_l_in, GRFy_l_in, GRFx_r_in, GRFy_r_in]
        return lambEOMrr_biped(*var_list) == 0
    m.EOMrr_biped = Constraint(m.N, rule = EOMrr_biped)

def solve(m, display_results, expect_infeasible):
    # Solve -----------------------------------------------------------------------------------------------------------------------
#    opt = SolverFactory('ipopt') # standard issue, garden variety ipopt
    
    opt = SolverFactory('ipopt',executable = 'C:/cygwin64/home/Nick/CoinIpopt/build/bin/ipopt.exe')
    opt.options["linear_solver"] = 'ma86'
    
    # solver options
    opt.options["expect_infeasible_problem"] = expect_infeasible # I changed this
    #opt.options["linear_system_scaling"] = 'none'
    #opt.options["mu_strategy"] = "adaptive"
    opt.options["halt_on_ampl_error"] = "yes"
    opt.options["print_level"] = 5 # prints a log with each iteration
    opt.options["max_iter"] = 100000 # maximum number of iterations
    opt.options["max_cpu_time"] = 30*60 # maximum cpu time in seconds
    opt.options["Tol"] = 1e-6 # the tolerance for feasibility. Considers constraints satisfied when they're within this margin.

    results = opt.solve(m, tee=display_results) 
    
    print(results.solver.status) # tells you if the solver had any errors/ warnings
    print(results.solver.termination_condition) # tells you if the solution was (locally) optimal, feasible, or neither.
    
    return results

def print_results(m,N,hm,ground_constraints, contact_constraints, joint_constraints, joints, Fs, legs):
    penalty_sum_ground = 0
    penalty_sum_contact = 0
    penalty_sum_joint = 0
    force_sum = 0
    torque_sum = 0
    for n in range(1,N+1):
        for gc in ground_constraints:
            penalty_sum_ground += m.ground_penalty[n,gc].value
        for cc in contact_constraints:
            for fs in Fs:
                penalty_sum_contact += m.contact_penalty[n,fs,cc].value 
        for j in joints:
            for leg in legs:
                for jc in joint_constraints:
                    penalty_sum_joint += m.joint_penalty[n,leg,j,jc].value 
        force_sum  += (m.F_a[n,'BL'].value**2 + m.F_a[n,'FL'].value**2)*hm*m.h[n].value
        torque_sum += (m.tau_a[n,'BL'].value**2 + m.tau_a[n,'FL'].value**2 + m.tau_arm[n,'BA'].value**2 + m.tau_arm[n,'FA'].value**2+m.tau_b[n].value**2)*hm*m.h[n].value
                
    print("--------------------------------")
    print("GROUND:  ", penalty_sum_ground)
    print("CONTACTS:", penalty_sum_contact)
    print("JOINT:   ", penalty_sum_joint)
    print("FORCE:   ", force_sum)
    print("TORQUE:  ", torque_sum)
    print("--------------------------------")
    
def get_max_values(m, N, ground_constraints, contact_constraints, joint_constraints, joints, WDOFs, DOF_b, Fs, signs, GRFs, legs):
    maxForce = 0
    maxGRF = 0
    maxGRFbf = 0
    maxF_a = 0

    for i in range(1,N+1):
        for fs in Fs:
            for dof in DOF_b:
                for sgn in signs:
                    if m.Fb_b[i,fs,dof,sgn].value is not None:
                        if abs(m.Fb_b[i,fs,dof,sgn].value)>maxForce:
                            maxForce = abs(m.Fb_b[i,fs,dof,sgn].value)
        for grf in GRFs:
            if m.GRF[i,grf].value is not None: 
                if m.GRF[i,grf].value>maxGRF:
                     maxGRF = m.GRF[i,grf].value
        for grf in WDOFs:
            for fs in Fs:
                if m.GRFbf[i,fs,grf].value is not None: 
                    if abs(m.GRFbf[i,fs,grf].value)>maxGRFbf:
                         maxGRFbf = abs(m.GRFbf[i,fs,grf].value)
        for leg in legs:
            if abs(m.F_a[i,leg].value)>maxF_a:
                maxF_a = abs(m.F_a[i,leg].value)
            
    return [maxForce, maxGRF, maxGRFbf, maxF_a]

def make_animation(m, Ns, scenario, X_step, Y_step, A_step, Y_obs, maxForce, maxGRF, maxGRFbf, maxF_a):
    
    N = Ns[0]
    XMIN = -1.0
    XMAX = 1.0
    YMIN = -0.25
    YMAX = 2.0
    
    GNDx = np.linspace(XMIN,XMAX,100)
    
    if scenario == "OU": 
        GNDy = Y_step*(1/2+1/2*np.tanh(500.0*(GNDx-X_step)))
        OBSx = np.linspace(XMIN,XMAX,100)
        OBSy = -1000.0*(OBSx-X_step)**4 + Y_obs
        print("OU")
    elif scenario == "OD":
        GNDy = Y_step*(1/2+1/2*np.tanh(-500.0*(GNDx-X_step)))
        OBSx = np.linspace(XMIN,XMAX,100)
        OBSy = -1000.0*(OBSx-X_step)**4 + Y_obs
        print("OD")
    else:
        GNDy = 0.0*GNDx
        OBSx = 0
        OBSy = 0
    
    #animate it
    fig1, ax1 = plt.subplots(1,1) #create axes
    ax1.set_aspect('equal')
    
    def plot_board(i,m,ax): #update function for animation
        MARKER_SIZE = 6

        ax.clear()
        ax.set_xlim([XMIN,XMAX])
        ax.set_ylim([YMIN,YMAX])
        ax.axis('off')

        #plot ground
        groundLx = XMIN
        groundLy = 0
        groundRx = XMAX
        groundRy = 0
        if i in Ns:
            ax.plot(GNDx,GNDy,color='xkcd:red')
            ax.plot(OBSx,OBSy,color='xkcd:red')
        else:
            ax.plot(GNDx,GNDy,color='xkcd:black')
            ax.plot(OBSx,OBSy,color='xkcd:green')

        #plot skateboard
        boardLx = m.ptail[i,'X'].value
        boardLy = m.ptail[i,'Y'].value
        boardRx = m.pnose[i,'X'].value
        boardRy = m.pnose[i,'Y'].value
        ax.plot([boardLx,boardRx],[boardLy,boardRy],color='xkcd:black')

        #plot left wheel
        leftwheelTopx = m.q[i,'x'].value-0.5*m.lbrd*np.cos(m.q[i,'th'].value)
        leftwheelTopy = m.q[i,'y'].value-0.5*m.lbrd*np.sin(m.q[i,'th'].value)
        leftwheelBottomx = m.pwheel[i,'BW','X'].value
        leftwheelBottomy = m.pwheel[i,'BW','Y'].value
        ax.plot([leftwheelTopx,leftwheelBottomx],[leftwheelTopy,leftwheelBottomy],color='xkcd:black')

        #plot left wheel-wheel
        leftwheelx = m.q[i,'x'].value-0.5*m.lbrd*cos(m.q[i,'th'].value)+0.6*m.hbd*sin(m.q[i,'th'].value)
        leftwheely = m.q[i,'y'].value-0.5*m.lbrd*sin(m.q[i,'th'].value)-0.6*m.hbd*cos(m.q[i,'th'].value)
        ax.plot(leftwheelx,leftwheely,color='xkcd:black',marker = 'o',markersize=MARKER_SIZE)

        #plot right wheel
        rightwheelTopx = m.q[i,'x'].value+0.5*m.lbrd*np.cos(m.q[i,'th'].value)
        rightwheelTopy = m.q[i,'y'].value+0.5*m.lbrd*np.sin(m.q[i,'th'].value)
        rightwheelBottomx = m.pwheel[i,'FW','X'].value
        rightwheelBottomy = m.pwheel[i,'FW','Y'].value
        ax.plot([rightwheelTopx,rightwheelBottomx],[rightwheelTopy,rightwheelBottomy],color='xkcd:black')

        #plot right wheel-wheel
        rightwheelx = m.q[i,'x'].value+0.5*m.lbrd*cos(m.q[i,'th'].value)+0.6*m.hbd*sin(m.q[i,'th'].value)
        rightwheely = m.q[i,'y'].value+0.5*m.lbrd*sin(m.q[i,'th'].value)-0.6*m.hbd*cos(m.q[i,'th'].value)
        ax.plot(rightwheelx,rightwheely,color='xkcd:black',marker = 'o',markersize=MARKER_SIZE)

        #plot forces
        backforcex = m.q[i,'x'].value - m.rF[i,'BF'].value*cos(m.q[i,'th'].value)
        backforcey = m.q[i,'y'].value - m.rF[i,'BF'].value*sin(m.q[i,'th'].value)
        frontforcex = m.q[i,'x'].value + m.rF[i,'FF'].value*cos(m.q[i,'th'].value)
        frontforcey = m.q[i,'y'].value + m.rF[i,'FF'].value*sin(m.q[i,'th'].value)

        if maxForce!=0:
            magforceBFx = m.Fb_b_total[i,'BF','x_b'].value/maxForce
            magforceBFy = m.Fb_b_total[i,'BF','y_b'].value/maxForce
            magforceFFx = m.Fb_b_total[i,'FF','x_b'].value/maxForce
            magforceFFy = m.Fb_b_total[i,'FF','y_b'].value/maxForce 

            # Back foot x force
            pBFxx = backforcex - 0.5*magforceBFx*np.cos(m.q[i,'th'].value)
            dpBFxx = 0.5*magforceBFx*np.cos(m.q[i,'th'].value)
            pBFxy = backforcey - 0.5*magforceBFx*np.sin(m.q[i,'th'].value)
            dpBFxy = 0.5*magforceBFx*np.sin(m.q[i,'th'].value)

            # back foot y force
            pBFyx = backforcex - 0.5*magforceBFy*np.sin(m.q[i,'th'].value)
            dpBFyx = 0.5*magforceBFy*np.sin(m.q[i,'th'].value)
            pBFyy = backforcey + 0.5*magforceBFy*np.cos(m.q[i,'th'].value)
            dpBFyy = -0.5*magforceBFy*np.cos(m.q[i,'th'].value)

#             ax.arrow(pBFxx, pBFxy, dpBFxx, dpBFxy, length_includes_head=True,head_width=abs(magforceBFx)*0.05,color='y')
#             ax.arrow(pBFyx, pBFyy, dpBFyx, dpBFyy, length_includes_head=True,head_width=abs(magforceBFy)*0.05,color='y')
            
            # front foot x force
            pFFxx = frontforcex - 0.5*magforceFFx*np.cos(m.q[i,'th'].value)
            dpFFxx = 0.5*magforceFFx*np.cos(m.q[i,'th'].value)
            pFFxy = frontforcey - 0.5*magforceFFx*np.sin(m.q[i,'th'].value)
            dpFFxy = 0.5*magforceFFx*np.sin(m.q[i,'th'].value)

            # front foot y force
            pFFyx = frontforcex - 0.5*magforceFFy*np.sin(m.q[i,'th'].value)
            dpFFyx = 0.5*magforceFFy*np.sin(m.q[i,'th'].value)
            pFFyy = frontforcey + 0.5*magforceFFy*np.cos(m.q[i,'th'].value)
            dpFFyy = -0.5*magforceFFy*np.cos(m.q[i,'th'].value)
            
#             ax.arrow(pFFxx, pFFxy, dpFFxx, dpFFxy, length_includes_head=True,head_width=abs(magforceFFx)*0.05,color='y')
#             ax.arrow(pFFyx, pFFyy, dpFFyx, dpFFyy, length_includes_head=True,head_width=abs(magforceFFy)*0.05,color='y')
            
        #plot GRF's
        if (m.GRF[i,'BW'].value is not None) and (m.GRF[i,'BW'].value!=0.0):
            magGRFBW = m.GRF[i,'BW'].value/maxGRF
        else: 
            magGRFBW = 0
        if (m.GRF[i,'FW'].value is not None) and (m.GRF[i,'FW'].value!=0.0):
            magGRFFW = m.GRF[i,'FW'].value/maxGRF
        else: 
            magGRFFW = 0

        backGRFx = leftwheelBottomx
        backGRFy = leftwheelBottomy
        frontGRFx = rightwheelBottomx
        frontGRFy = rightwheelBottomy        

#         ax.arrow(backGRFx, backGRFy-magGRFBW*0.5,0,magGRFBW*0.5, length_includes_head=True,head_width=magGRFBW*0.05,color='blue')
#         ax.arrow(frontGRFx, frontGRFy-magGRFFW*0.5,0,magGRFFW*0.5, length_includes_head=True,head_width=magGRFFW*0.05,color='blue')

        #plot biped with arms
        #plot body
        body_xb = m.q[i,'xp'].value
        body_yb = m.q[i,'yp'].value
        body_xt = m.q[i,'xp'].value + m.len[('body',1)]*cos(m.q[i,'thp'].value+m.q[i,'thb'].value)
        body_yt = m.q[i,'yp'].value + m.len[('body',1)]*sin(m.q[i,'thp'].value+m.q[i,'thb'].value)  
        ax.plot([body_xb,body_xt],[body_yb,body_yt],color='k')
        
        #Plot arms
        #plot left arm
        arm_lxb = body_xb+m.lp2a*np.cos(m.q[i,'thp'].value+m.q[i,'thb'].value)
        arm_lyb = body_yb+m.lp2a*np.sin(m.q[i,'thp'].value+m.q[i,'thb'].value)
        arm_lxt = arm_lxb - m.len[('left_arm',1)]*np.sin(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_l'].value)
        arm_lyt = arm_lyb + m.len[('left_arm',1)]*np.cos(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_l'].value) 
        ax.plot([arm_lxb,arm_lxt],[arm_lyb,arm_lyt],color='k')
        
        #plot right arm
        arm_rxb = arm_lxb
        arm_ryb = arm_lyb
        arm_rxt = arm_rxb + m.len[('right_arm',1)]*np.sin(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_r'].value)
        arm_ryt = arm_ryb - m.len[('right_arm',1)]*np.cos(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_r'].value) 
        ax.plot([arm_rxb,arm_rxt],[arm_ryb,arm_ryt],color='k')
        
        #plot pelvis
        pelvis_xb = m.q[i,'xp'].value - 0.5*m.len[('pelvis',1)]*cos(m.q[i,'thp'].value)
        pelvis_yb = m.q[i,'yp'].value - 0.5*m.len[('pelvis',1)]*sin(m.q[i,'thp'].value)
        pelvis_xf = m.q[i,'xp'].value + 0.5*m.len[('pelvis',1)]*cos(m.q[i,'thp'].value)
        pelvis_yf = m.q[i,'yp'].value + 0.5*m.len[('pelvis',1)]*sin(m.q[i,'thp'].value)  
        ax.plot([pelvis_xb,pelvis_xf],[pelvis_yb,pelvis_yf],color='k')
        
        #Left Leg
        #plot leg 1
        leg1_xt = pelvis_xb
        leg1_yt = pelvis_yb
        leg1_xb = pelvis_xb + m.len[('left_leg',1)]*sin(m.thA[i,'BL'].value)
        leg1_yb = pelvis_yb - m.len[('left_leg',1)]*cos(m.thA[i,'BL'].value)
        ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

        #plot leg 2
        Lt = 0.5*m.len[('left_leg',1)] + m.q[i,'r_l'].value - 0.5*m.len[('left_leg',2)]
        Lb = 0.5*m.len[('left_leg',1)] + m.q[i,'r_l'].value + 0.5*m.len[('left_leg',2)]
        leg2_xt = pelvis_xb + Lt*sin(m.thA[i,'BL'].value)
        leg2_yt = pelvis_yb - Lt*cos(m.thA[i,'BL'].value)
        leg2_xb = m.pfoot[i,'BF','X'].value
        leg2_yb = m.pfoot[i,'BF','Y'].value
        ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        
        if m.pfoot_b[i,'BF','y_b'].value<1e-5:
            ax.plot(leg2_xb,leg2_yb,color='r',marker = '.')
        else:
            ax.plot(leg2_xb,leg2_yb,color='gray',marker = '.')
            
        #Right Leg
        #plot leg 1
        leg1_xt = pelvis_xf
        leg1_yt = pelvis_yf
        leg1_xb = pelvis_xf + m.len[('right_leg',1)]*sin(m.thA[i,'FL'].value)
        leg1_yb = pelvis_yf - m.len[('right_leg',1)]*cos(m.thA[i,'FL'].value)
        ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

        #plot leg 2
        Lt = 0.5*m.len[('right_leg',1)] + m.q[i,'r_r'].value - 0.5*m.len[('right_leg',2)]
        Lb = 0.5*m.len[('right_leg',1)] + m.q[i,'r_r'].value + 0.5*m.len[('right_leg',2)]
        leg2_xt = pelvis_xf + Lt*sin(m.thA[i,'FL'].value)
        leg2_yt = pelvis_yf - Lt*cos(m.thA[i,'FL'].value)
        leg2_xb = m.pfoot[i,'FF','X'].value
        leg2_yb = m.pfoot[i,'FF','Y'].value
        ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        
        if m.pfoot_b[i,'FF','y_b'].value<1e-5:
            ax.plot(leg2_xb,leg2_yb,color='r',marker = '.')
        else:
            ax.plot(leg2_xb,leg2_yb,color='gray',marker = '.')
        
    update = lambda i: plot_board(i,m,ax1) #lambdify update function

    animate = ani.FuncAnimation(fig1, update, frames=range(1,N+1), interval=100, repeat=True)
    plt.close(animate._fig)
    
    return animate

def make_discrete_plots(m, i):
    XMIN = -1.0
    XMAX = 1.0
    YMIN = -0.25
    YMAX = 2.0
    
    GNDx = np.linspace(XMIN,XMAX,100)
    GNDy = 0.0*GNDx
    
    #plot it
    fig, ax = plt.subplots(1,1) #create axes
    ax.set_aspect('equal')
    
    MARKER_SIZE = 6

    ax.clear()
    ax.set_xlim([XMIN,XMAX])
    ax.set_ylim([YMIN,YMAX])
    ax.axis('off')

    #plot ground
    groundLx = XMIN
    groundLy = 0
    groundRx = XMAX
    groundRy = 0
    ax.plot(GNDx,GNDy,color='xkcd:black')

    #plot skateboard
    boardLx = m.ptail[i,'X'].value
    boardLy = m.ptail[i,'Y'].value
    boardRx = m.pnose[i,'X'].value
    boardRy = m.pnose[i,'Y'].value
    ax.plot([boardLx,boardRx],[boardLy,boardRy],color='xkcd:black')

    #plot left wheel
    leftwheelTopx = m.q[i,'x'].value-0.5*m.lbrd*np.cos(m.q[i,'th'].value)
    leftwheelTopy = m.q[i,'y'].value-0.5*m.lbrd*np.sin(m.q[i,'th'].value)
    leftwheelBottomx = m.pwheel[i,'BW','X'].value
    leftwheelBottomy = m.pwheel[i,'BW','Y'].value
    ax.plot([leftwheelTopx,leftwheelBottomx],[leftwheelTopy,leftwheelBottomy],color='xkcd:black')

    #plot left wheel-wheel
    leftwheelx = m.q[i,'x'].value-0.5*m.lbrd*cos(m.q[i,'th'].value)+0.6*m.hbd*sin(m.q[i,'th'].value)
    leftwheely = m.q[i,'y'].value-0.5*m.lbrd*sin(m.q[i,'th'].value)-0.6*m.hbd*cos(m.q[i,'th'].value)
    ax.plot(leftwheelx,leftwheely,color='xkcd:black',marker = 'o',markersize=MARKER_SIZE)

    #plot right wheel
    rightwheelTopx = m.q[i,'x'].value+0.5*m.lbrd*np.cos(m.q[i,'th'].value)
    rightwheelTopy = m.q[i,'y'].value+0.5*m.lbrd*np.sin(m.q[i,'th'].value)
    rightwheelBottomx = m.pwheel[i,'FW','X'].value
    rightwheelBottomy = m.pwheel[i,'FW','Y'].value
    ax.plot([rightwheelTopx,rightwheelBottomx],[rightwheelTopy,rightwheelBottomy],color='xkcd:black')

    #plot right wheel-wheel
    rightwheelx = m.q[i,'x'].value+0.5*m.lbrd*cos(m.q[i,'th'].value)+0.6*m.hbd*sin(m.q[i,'th'].value)
    rightwheely = m.q[i,'y'].value+0.5*m.lbrd*sin(m.q[i,'th'].value)-0.6*m.hbd*cos(m.q[i,'th'].value)
    ax.plot(rightwheelx,rightwheely,color='xkcd:black',marker = 'o',markersize=MARKER_SIZE)
            
    #plot biped with arms
    #plot body
    body_xb = m.q[i,'xp'].value
    body_yb = m.q[i,'yp'].value
    body_xt = m.q[i,'xp'].value + m.len[('body',1)]*cos(m.q[i,'thp'].value+m.q[i,'thb'].value)
    body_yt = m.q[i,'yp'].value + m.len[('body',1)]*sin(m.q[i,'thp'].value+m.q[i,'thb'].value)  
    ax.plot([body_xb,body_xt],[body_yb,body_yt],color='k')
        
    #Plot arms
    #plot left arm
    arm_lxb = body_xb+m.lp2a*np.cos(m.q[i,'thp'].value+m.q[i,'thb'].value)
    arm_lyb = body_yb+m.lp2a*np.sin(m.q[i,'thp'].value+m.q[i,'thb'].value)
    arm_lxt = arm_lxb - m.len[('left_arm',1)]*np.sin(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_l'].value)
    arm_lyt = arm_lyb + m.len[('left_arm',1)]*np.cos(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_l'].value) 
    ax.plot([arm_lxb,arm_lxt],[arm_lyb,arm_lyt],color='k')
        
    #plot right arm
    arm_rxb = arm_lxb
    arm_ryb = arm_lyb
    arm_rxt = arm_rxb + m.len[('right_arm',1)]*np.sin(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_r'].value)
    arm_ryt = arm_ryb - m.len[('right_arm',1)]*np.cos(m.q[i,'thp'].value+m.q[i,'thb'].value+m.q[i,'tha_r'].value) 
    ax.plot([arm_rxb,arm_rxt],[arm_ryb,arm_ryt],color='k')
        
    #plot pelvis
    pelvis_xb = m.q[i,'xp'].value - 0.5*m.len[('pelvis',1)]*cos(m.q[i,'thp'].value)
    pelvis_yb = m.q[i,'yp'].value - 0.5*m.len[('pelvis',1)]*sin(m.q[i,'thp'].value)
    pelvis_xf = m.q[i,'xp'].value + 0.5*m.len[('pelvis',1)]*cos(m.q[i,'thp'].value)
    pelvis_yf = m.q[i,'yp'].value + 0.5*m.len[('pelvis',1)]*sin(m.q[i,'thp'].value)  
    ax.plot([pelvis_xb,pelvis_xf],[pelvis_yb,pelvis_yf],color='k')
        
    #Left Leg
    #plot leg 1
    leg1_xt = pelvis_xb
    leg1_yt = pelvis_yb
    leg1_xb = pelvis_xb + m.len[('left_leg',1)]*sin(m.thA[i,'BL'].value)
    leg1_yb = pelvis_yb - m.len[('left_leg',1)]*cos(m.thA[i,'BL'].value)
    ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

    #plot leg 2
    Lt = 0.5*m.len[('left_leg',1)] + m.q[i,'r_l'].value - 0.5*m.len[('left_leg',2)]
    Lb = 0.5*m.len[('left_leg',1)] + m.q[i,'r_l'].value + 0.5*m.len[('left_leg',2)]
    leg2_xt = pelvis_xb + Lt*sin(m.thA[i,'BL'].value)
    leg2_yt = pelvis_yb - Lt*cos(m.thA[i,'BL'].value)
    leg2_xb = m.pfoot[i,'BF','X'].value
    leg2_yb = m.pfoot[i,'BF','Y'].value
    ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        
    if m.pfoot_b[i,'BF','y_b'].value<1e-5:
        ax.plot(leg2_xb,leg2_yb,color='r',marker = '.')
    else:
        ax.plot(leg2_xb,leg2_yb,color='gray',marker = '.')
            
    #Right Leg
    #plot leg 1
    leg1_xt = pelvis_xf
    leg1_yt = pelvis_yf
    leg1_xb = pelvis_xf + m.len[('right_leg',1)]*sin(m.thA[i,'FL'].value)
    leg1_yb = pelvis_yf - m.len[('right_leg',1)]*cos(m.thA[i,'FL'].value)
    ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

    #plot leg 2
    Lt = 0.5*m.len[('right_leg',1)] + m.q[i,'r_r'].value - 0.5*m.len[('right_leg',2)]
    Lb = 0.5*m.len[('right_leg',1)] + m.q[i,'r_r'].value + 0.5*m.len[('right_leg',2)]
    leg2_xt = pelvis_xf + Lt*sin(m.thA[i,'FL'].value)
    leg2_yt = pelvis_yf - Lt*cos(m.thA[i,'FL'].value)
    leg2_xb = m.pfoot[i,'FF','X'].value
    leg2_yb = m.pfoot[i,'FF','Y'].value
    ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        
    if m.pfoot_b[i,'FF','y_b'].value<1e-5:
        ax.plot(leg2_xb,leg2_yb,color='r',marker = '.')
    else:
        ax.plot(leg2_xb,leg2_yb,color='gray',marker = '.')
    return fig

def make_realtime_animation(m, N, hm):
    # store all variables for extrapolation
    #initialize time dependent arrays
    q_x_ar = np.zeros(N)
    q_y_ar = np.zeros(N)
    q_th_ar = np.zeros(N)
    q_xp_ar = np.zeros(N)
    q_yp_ar = np.zeros(N)
    q_thp_ar = np.zeros(N)
    q_thb_ar = np.zeros(N)
    q_thl_l_ar = np.zeros(N)
    q_r_l_ar = np.zeros(N)
    q_thl_r_ar = np.zeros(N)
    q_r_r_ar = np.zeros(N)

    ptail_x_ar = np.zeros(N)
    ptail_y_ar = np.zeros(N)
    pnose_x_ar = np.zeros(N)
    pnose_y_ar = np.zeros(N)

    pbackwheel_x_ar = np.zeros(N)
    pbackwheel_y_ar = np.zeros(N)
    pfrontwheel_x_ar = np.zeros(N)
    pfrontwheel_y_ar = np.zeros(N)

    h_ar = np.zeros(N)

    # fill arrays with solver values
    for i in range (1,N+1):
        q_x_ar[i-1]     = m.q[i,'x'].value
        q_y_ar[i-1]     = m.q[i,'y'].value
        q_th_ar[i-1]    = m.q[i,'th'].value
        q_xp_ar[i-1]    = m.q[i,'xp'].value
        q_yp_ar[i-1]    = m.q[i,'yp'].value
        q_thp_ar[i-1]   = m.q[i,'thp'].value
        q_thb_ar[i-1]   = m.q[i,'thb'].value
        q_thl_l_ar[i-1] = m.q[i,'thl_l'].value
        q_r_l_ar[i-1]   = m.q[i,'r_l'].value
        q_thl_r_ar[i-1] = m.q[i,'thl_r'].value
        q_r_r_ar[i-1]   = m.q[i,'r_r'].value

        ptail_x_ar[i-1] = m.ptail[i,'X'].value
        ptail_y_ar[i-1] = m.ptail[i,'Y'].value
        pnose_x_ar[i-1] = m.pnose[i,'X'].value
        pnose_y_ar[i-1] = m.pnose[i,'Y'].value

        pbackwheel_x_ar[i-1] = m.pwheel[i,'BW','X'].value
        pbackwheel_y_ar[i-1] = m.pwheel[i,'BW','Y'].value
        pfrontwheel_x_ar[i-1] = m.pwheel[i,'FW','X'].value
        pfrontwheel_y_ar[i-1] = m.pwheel[i,'FW','Y'].value

        h_ar[i-1] = m.h[i].value*hm

    # calculate the time array. 
    t_ar = np.cumsum(h_ar)

    # caculate the interpolation functions.
    f_x_interp = interp1d(t_ar, q_x_ar, kind='cubic')
    f_y_interp = interp1d(t_ar, q_y_ar, kind='cubic')
    f_th_interp = interp1d(t_ar, q_th_ar, kind='cubic')
    f_xp_interp = interp1d(t_ar, q_xp_ar, kind='cubic')
    f_yp_interp = interp1d(t_ar, q_yp_ar, kind='cubic')
    f_thp_interp = interp1d(t_ar, q_thp_ar, kind='cubic')
    f_thb_interp = interp1d(t_ar, q_thb_ar, kind='cubic')
    f_thl_l_interp = interp1d(t_ar, q_thl_l_ar, kind='cubic')
    f_r_l_interp = interp1d(t_ar, q_r_l_ar, kind='cubic')
    f_thl_r_interp = interp1d(t_ar, q_thl_r_ar, kind='cubic')
    f_r_r_interp = interp1d(t_ar, q_r_r_ar, kind='cubic')

    f_ptail_x_interp = interp1d(t_ar, ptail_x_ar, kind='cubic')
    f_ptail_y_interp = interp1d(t_ar, ptail_y_ar, kind='cubic')
    f_pnose_x_interp = interp1d(t_ar, pnose_x_ar, kind='cubic')
    f_pnose_y_interp = interp1d(t_ar, pnose_y_ar, kind='cubic')

    f_pbackwheel_x_interp = interp1d(t_ar, pbackwheel_x_ar, kind='cubic')
    f_pbackwheel_y_interp = interp1d(t_ar, pbackwheel_y_ar, kind='cubic')
    f_pfrontwheel_x_interp = interp1d(t_ar, pfrontwheel_x_ar, kind='cubic')
    f_pfrontwheel_y_interp = interp1d(t_ar, pfrontwheel_y_ar, kind='cubic')

    # the desired time steps of 0.01s (100Hz)
    Ts = hm
    t_d = np.arange(hm, t_ar[-1], Ts)

    # sample the interpolation functions at 50Hz.
    q_x_interp = f_x_interp(t_d)
    q_y_interp = f_y_interp(t_d)
    q_th_interp = f_th_interp(t_d)
    q_xp_interp = f_xp_interp(t_d)
    q_yp_interp = f_yp_interp(t_d)
    q_thp_interp = f_thp_interp(t_d)
    q_thb_interp = f_thb_interp(t_d)
    q_thl_l_interp = f_thl_l_interp(t_d)
    q_r_l_interp = f_r_l_interp(t_d)
    q_thl_r_interp = f_thl_r_interp(t_d)
    q_r_r_interp = f_r_r_interp(t_d)

    ptail_x_interp = f_ptail_x_interp(t_d)
    ptail_y_interp = f_ptail_y_interp(t_d)
    pnose_x_interp = f_pnose_x_interp(t_d)
    pnose_y_interp = f_pnose_y_interp(t_d)

    pbackwheel_x_interp = f_pbackwheel_x_interp(t_d)
    pbackwheel_y_interp = f_pbackwheel_y_interp(t_d)
    pfrontwheel_x_interp = f_pfrontwheel_x_interp(t_d)
    pfrontwheel_y_interp = f_pfrontwheel_y_interp(t_d)

    # number of samples
    N_interp = len(t_d)
    
    #animate it
    fig1, ax1 = plt.subplots(1,1) #create axes
    ax1.set_aspect('equal')
    
    def plot_board(i,m,ax): #update function for animation
        ax.clear()
        ax.set_title('Real Time')
        ax.set_xlim([-1,4])
        ax.set_ylim([-0.5,2.0])

        MARKER_SIZE = 6

        #plot ground
        groundLx = -1
        groundLy = 0
        groundRx = 2
        groundRy = 0
        ax.plot([groundLx,groundRx],[groundLy,groundRy],color='xkcd:black')

        #plot skateboard
        boardLx = ptail_x_interp[i]
        boardLy = ptail_y_interp[i]
        boardRx = pnose_x_interp[i]
        boardRy = pnose_y_interp[i]
        ax.plot([boardLx,boardRx],[boardLy,boardRy],color='xkcd:black')

        #plot left wheel
        leftwheelTopx = q_x_interp[i]-0.5*m.lbrd*np.cos(q_th_interp[i])
        leftwheelTopy = q_y_interp[i]-0.5*m.lbrd*np.sin(q_th_interp[i])
        leftwheelBottomx = pbackwheel_x_interp[i]
        leftwheelBottomy = pbackwheel_y_interp[i]
        ax.plot([leftwheelTopx,leftwheelBottomx],[leftwheelTopy,leftwheelBottomy],color='xkcd:black')

        #plot left wheel-wheel
        leftwheelx = q_x_interp[i]-0.5*m.lbrd*cos(q_th_interp[i])+0.6*m.hbd*sin(q_th_interp[i])
        leftwheely = q_y_interp[i]-0.5*m.lbrd*sin(q_th_interp[i])-0.6*m.hbd*cos(q_th_interp[i])
        ax.plot(leftwheelx,leftwheely,color='xkcd:black',marker = 'o',markersize=MARKER_SIZE)

        #plot right wheel
        rightwheelTopx = q_x_interp[i]+0.5*m.lbrd*np.cos(q_th_interp[i])
        rightwheelTopy = q_y_interp[i]+0.5*m.lbrd*np.sin(q_th_interp[i])
        rightwheelBottomx = pfrontwheel_x_interp[i]
        rightwheelBottomy = pfrontwheel_y_interp[i]
        ax.plot([rightwheelTopx,rightwheelBottomx],[rightwheelTopy,rightwheelBottomy],color='xkcd:black')

        #plot right wheel-wheel
        rightwheelx = q_x_interp[i]+0.5*m.lbrd*cos(q_th_interp[i])+0.6*m.hbd*sin(q_th_interp[i])
        rightwheely = q_y_interp[i]+0.5*m.lbrd*sin(q_th_interp[i])-0.6*m.hbd*cos(q_th_interp[i])
        ax.plot(rightwheelx,rightwheely,color='xkcd:black',marker = 'o',markersize=MARKER_SIZE)
        
        #plot biped
        #plot body
        body_xb = q_xp_interp[i]
        body_yb = q_yp_interp[i]
        body_xf = q_xp_interp[i] + m.len[('body',1)]*cos(q_thp_interp[i]+q_thb_interp[i])
        body_yf = q_yp_interp[i] + m.len[('body',1)]*sin(q_thp_interp[i]+q_thb_interp[i])  
        ax.plot([body_xb,body_xf],[body_yb,body_yf],color='k')
        
        #plot pelvis
        pelvis_xb = q_xp_interp[i] - 0.5*m.len[('pelvis',1)]*cos(q_thp_interp[i])
        pelvis_yb = q_yp_interp[i] - 0.5*m.len[('pelvis',1)]*sin(q_thp_interp[i])
        pelvis_xf = q_xp_interp[i] + 0.5*m.len[('pelvis',1)]*cos(q_thp_interp[i])
        pelvis_yf = q_yp_interp[i] + 0.5*m.len[('pelvis',1)]*sin(q_thp_interp[i])  
        ax.plot([pelvis_xb,pelvis_xf],[pelvis_yb,pelvis_yf],color='k')
        
        #Left Leg
        #plot leg 1
        leg1_xt = pelvis_xb
        leg1_yt = pelvis_yb
        leg1_xb = pelvis_xb + m.len[('left_leg',1)]*sin(q_thp_interp[i]+q_thl_l_interp[i])
        leg1_yb = pelvis_yb - m.len[('left_leg',1)]*cos(q_thp_interp[i]+q_thl_l_interp[i])
        ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

        #plot leg 2
        Lt = 0.5*m.len[('left_leg',1)] + q_r_l_interp[i] - 0.5*m.len[('left_leg',2)]
        Lb = 0.5*m.len[('left_leg',1)] + q_r_l_interp[i] + 0.5*m.len[('left_leg',2)]
        leg2_xt = pelvis_xb + Lt*sin(q_thp_interp[i]+q_thl_l_interp[i])
        leg2_yt = pelvis_yb - Lt*cos(q_thp_interp[i]+q_thl_l_interp[i])
        leg2_xb = pelvis_xb + Lb*sin(q_thp_interp[i]+q_thl_l_interp[i])
        leg2_yb = pelvis_yb - Lb*cos(q_thp_interp[i]+q_thl_l_interp[i])
        ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        ax.plot(leg2_xb,leg2_yb,color='m',marker = '.')
            
        #Right Leg
        #plot leg 1
        leg1_xt = pelvis_xf
        leg1_yt = pelvis_yf
        leg1_xb = pelvis_xf + m.len[('right_leg',1)]*sin(q_thp_interp[i]+q_thl_r_interp[i])
        leg1_yb = pelvis_yf - m.len[('right_leg',1)]*cos(q_thp_interp[i]+q_thl_r_interp[i])
        ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

        #plot leg 2
        Lt = 0.5*m.len[('right_leg',1)] + q_r_r_interp[i] - 0.5*m.len[('right_leg',2)]
        Lb = 0.5*m.len[('right_leg',1)] + q_r_r_interp[i] + 0.5*m.len[('right_leg',2)]
        leg2_xt = pelvis_xf + Lt*sin(q_thp_interp[i]+q_thl_r_interp[i])
        leg2_yt = pelvis_yf - Lt*cos(q_thp_interp[i]+q_thl_r_interp[i])
        leg2_xb = pelvis_xf + Lb*sin(q_thp_interp[i]+q_thl_r_interp[i])
        leg2_yb = pelvis_yf - Lb*cos(q_thp_interp[i]+q_thl_r_interp[i])
        ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        ax.plot(leg2_xb,leg2_yb,color='m',marker = '.')

    update = lambda i: plot_board(i,m,ax1) #lambdify update function

    # animate it - interval is in ms
    animate = ani.FuncAnimation(fig1,update,frames=range(0,N_interp),interval=Ts*1000,repeat=True)
    plt.close(animate._fig)
    
    return animate
    
    