# import libraries
import pickle as pkl
import sympy as sym
import numpy as np

from pyomo.environ import*
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

# DERIVE EOMS OF BOARD --------------------------------------------------------------------------------------------------------

def get_EOMs_board():
    # system parameters
    g = sym.symbols('g')
    mb = sym.symbols('m_{board}') # mass
    lb = sym.symbols('l_{board}') # length
    lbr = sym.symbols('l_{wheels}') # length to reaction forces
    hb = sym.symbols('h_{board}') # height - board clearance
    Inb = sym.symbols('In_{board}') # moment of intertia
    rF_FF, rF_BF = sym.symbols(['r_{F_{FF}}','r_{F_{BF}}']) # distance of feet from COM. 

    # generalized coordinates
    x,y,thb = sym.symbols(['x','y','\\theta_{board}']) 
    dx,dy,dthb = sym.symbols(['\dot{x}','\dot{y}','\dot{\\theta}_{board}']) 
    ddx,ddy,ddthb = sym.symbols(['\ddot{x}','\ddot{y}','\ddot{\\theta}_{board}']) 

    q = sym.Matrix([[x],[y],[thb]])
    dq = sym.Matrix([[dx],[dy],[dthb]])
    ddq = sym.Matrix([[ddx],[ddy],[ddthb]])

    # forces
    GRF_BW,GRF_FW = sym.symbols(['GRF_{BW}','GRF_{FW}']) # ground reaction forces
    F_BFx_sb,F_BFy_sb = sym.symbols(['F_{BFx_{sb}}','F_{BFy_{sb}}']) # back foot applied forces
    F_FFx_sb,F_FFy_sb = sym.symbols(['F_{FFx_{sb}}','F_{FFy_{sb}}']) # front foot applied forces

    # Unpickle stored EOMs
    infile = open('skateboard_EOMs','rb')
    data = pkl.load(infile)
    infile.close()

    EOMs_board = data['EOMs_Fsb']
    
    func_map = {'sin':sin, 'cos':cos} 

    sym_list = [g,mb,lb,lbr,Inb,hb,
                x,y,thb,
                dx,dy,dthb,
                ddx,ddy,ddthb,
                rF_BF,rF_FF,
                F_BFx_sb,F_BFy_sb,F_FFx_sb,F_FFy_sb,
                GRF_BW,GRF_FW]
            
    lambEOMx_board   = sym.lambdify(sym_list,EOMs_board[0],modules = [func_map])
    lambEOMy_board   = sym.lambdify(sym_list,EOMs_board[1],modules = [func_map])
    lambEOMth_board  = sym.lambdify(sym_list,EOMs_board[2],modules = [func_map])
    
    return [EOMs_board,[lambEOMx_board, lambEOMy_board, lambEOMth_board]]

def get_EOMs_biped_with_arms():
    # system parameters
    g = sym.symbols('g')
    mb,ma_l,ma_r,mp,ml1_l,ml2_l,ml1_r,ml2_r = sym.symbols(['m_{body}','m_{arm_{left}}','m_{arm_{right}}','m_{pelvis}',
                                                           'm_{leg1_{left}}','m_{leg2_{left}}','m_{leg1_{right}}',
                                                           'm_{leg2_{right}}']) # mass

    lb,la_l,la_r,lp,ll1_l,ll2_l,ll1_r,ll2_r,l_p2s = sym.symbols(['l_{body}','l_{arm_{left}}','l_{arm_{right}}','l_{pelvis}',
                                                           'l_{leg1_{left}}','l_{leg2_{left}}','l_{leg1_{right}}',
                                                           'l_{leg2_{right}}','l_{p2s}']) # length

    Inb,Ina_l,Ina_r,Inp,Inl1_l,Inl2_l,Inl1_r,Inl2_r = sym.symbols(['I_{body}','I_{arm_{left}}','I_{arm_{right}}','I_{pelvis}',
                                                                   'I_{leg1_{left}}','I_{leg2_{left}}','I_{leg1_{right}}',
                                                                   'I_{leg2_{right}}']) # moment of intertia

    # generalized coordinates
    xp,yp,thp,thb,tha_l,tha_r,thl_l,thl_r,r_l,r_r = sym.symbols(['x_{pelvis}','y_{pelvis}','\\theta_{pelvis}','\\theta_{body}',
                                                                 '\\theta_{arm_{left}}','\\theta_{arm_{right}}',
                                                                 '\\theta_{leg_{left}}','\\theta_{leg_{right}}',
                                                                 'r_{left}','r_{right}']) 

    dxp,dyp,dthp,dthb,dtha_l,dtha_r,dthl_l,dthl_r,dr_l,dr_r = sym.symbols(['\dot{x}_{pelvis}','\dot{y}_{pelvis}','\dot{\\theta}_{pelvis}','\dot{\\theta}_{body}',
                                '\dot{\\theta}_{arm_{left}}','\dot{\\theta}_{arm_{right}}',
                                '\dot{\\theta}_{leg_{left}}','\dot{\\theta}_{leg_{right}}',
                                '\dot{r}_{left}','\dot{r}_{right}']) 

    ddxp,ddyp,ddthp,ddthb,ddtha_l,ddtha_r,ddthl_l,ddthl_r,ddr_l,ddr_r = sym.symbols(['\ddot{x}_{pelvis}','\ddot{y}_{pelvis}','\ddot{\\theta}_{pelvis}','\ddot{\\theta}_{body}',
                                '\ddot{\\theta}_{arm_{left}}','\ddot{\\theta}_{arm_{right}}',
                                '\ddot{\\theta}_{leg_{left}}','\ddot{\\theta}_{leg_{right}}',
                                '\ddot{r}_{left}','\ddot{r}_{right}']) 

    # forces
    # total joint action = actuator + rebound, but that will be dealt with elsewhere
    F_a_l,F_a_r,GRFx_l,GRFy_l,GRFx_r,GRFy_r = sym.symbols(['F_{left}','F_{right}','G_{x_{left}}','G_{y_{left}}','G_{x_{right}}','G_{y_{right}}']) 
    tau_b,tau_a_l,tau_a_r,tau_l_l,tau_l_r = sym.symbols(['\\tau_{body}','\\tau_{arm_{left}}','\\tau_{arm_{right}}','\\tau_{leg_{left}}','\\tau_{leg_{right}}'])


    # Unpickle stored EOMs
    infile = open('biped_with_arms_EOMs','rb')
    data = pkl.load(infile)
    infile.close()

    EOMs_biped = data['EOMs']
    
    func_map = {'sin':sin, 'cos':cos} 

    sym_list = [g,mb,ma_l,ma_r,mp,ml1_l,ml2_l,ml1_r,ml2_r,
                lb,la_l,la_r,lp,ll1_l,ll2_l,ll1_r,ll2_r,l_p2s,
                Inb,Ina_l,Ina_r,Inp,Inl1_l,Inl2_l,Inl1_r,Inl2_r,
                xp,yp,thp,thb,tha_l,tha_r,thl_l,thl_r,r_l,r_r,
                dxp,dyp,dthp,dthb,dtha_l,dtha_r,dthl_l,dthl_r,dr_l,dr_r,
                ddxp,ddyp,ddthp,ddthb,ddtha_l,ddtha_r,ddthl_l,ddthl_r,ddr_l,ddr_r,
                F_a_l,F_a_r,tau_b,tau_a_l,tau_a_r,tau_l_l,tau_l_r,GRFx_l,GRFy_l,GRFx_r,GRFy_r]
            
    lambEOMxp_biped   = sym.lambdify(sym_list,EOMs_biped[0],modules = [func_map])
    lambEOMyp_biped   = sym.lambdify(sym_list,EOMs_biped[1],modules = [func_map])
    lambEOMthp_biped  = sym.lambdify(sym_list,EOMs_biped[2],modules = [func_map])
    lambEOMthb_biped  = sym.lambdify(sym_list,EOMs_biped[3],modules = [func_map])
    lambEOMtha_l_biped = sym.lambdify(sym_list,EOMs_biped[4],modules = [func_map])
    lambEOMtha_r_biped = sym.lambdify(sym_list,EOMs_biped[5],modules = [func_map])
    lambEOMthl_l_biped = sym.lambdify(sym_list,EOMs_biped[6],modules = [func_map])
    lambEOMthl_r_biped = sym.lambdify(sym_list,EOMs_biped[7],modules = [func_map])
    lambEOMr_l_biped   = sym.lambdify(sym_list,EOMs_biped[8],modules = [func_map])
    lambEOMr_r_biped   = sym.lambdify(sym_list,EOMs_biped[9],modules = [func_map])
    
    return [EOMs_biped,[lambEOMxp_biped, lambEOMyp_biped, lambEOMthp_biped, lambEOMthb_biped, lambEOMthl_l_biped, lambEOMr_l_biped, lambEOMthl_r_biped, lambEOMr_r_biped, lambEOMtha_l_biped, lambEOMtha_r_biped]]

def get_foot_v_and_p():
    # foot velocity and position in board and global frame
    
    # system variables
    xm,ym,thm,thl,r = sym.symbols(['x_{mono}','y_{mono}','\\theta_{mono}','\\theta_{link1}','r_{mono}']) # monopod in global frame
    dxm,dym,dthm,dthl,dr = sym.symbols(['\dot{x}_{mono}','\dot{y}_{mono}','\dot{\\theta}_{mono}','\dot{\\theta}_{link1}','\dot{r}_{mono}'])
    lb,l1,l2 = sym.symbols(['l_{board}','l_{link1}','l_{link2}']) # lenghts of links
    xb,yb,thb = sym.symbols(['x_{board}','y_{board}','\\theta_{board}']) # board in global frame
    dxb,dyb,dthb = sym.symbols(['\dot{x}_{board}','\dot{y}_{board}','\dot{\\theta}_{board}'])

    # Unpickle stored EOMs
    infile = open('biped_with_arms_EOMs','rb')
    data = pkl.load(infile)
    infile.close()

    pfoot_l = data['pfoot_left']
    pfoot_l_board = data['pfoot_left_board']
    vfoot_l = None
    vfootx_l_board = data['vfootx_left_board']
    pfoot_r = data['pfoot_right']
    pfoot_r_board = data['pfoot_right_board']
    vfoot_r = None
    vfootx_r_board = data['vfootx_right_board']

    # Lambdify pfoot_board
    func_map = {'sin':sin, 'cos':cos, 'atan':atan} 
    sym_list = [lb, l1, l2, xb, yb, thb, xm, ym, thm, thl, r]
    sym_list2 = [lb, l1, l2, xb, yb, thb, xm, ym, thm, thl, r, dxb, dyb, dthb, dxm, dym, dthm, dthl, dr]

    lambfootx_l = sym.lambdify(sym_list,pfoot_l[0],modules = [func_map])
    lambfooty_l = sym.lambdify(sym_list,pfoot_l[1],modules = [func_map])
    lambfootx_l_board = sym.lambdify(sym_list,pfoot_l_board[0],modules = [func_map])
    lambfooty_l_board = sym.lambdify(sym_list,pfoot_l_board[1],modules = [func_map])
    lambvfootx_l_board = sym.lambdify(sym_list2,vfootx_l_board,modules = [func_map])
    lambfootx_r = sym.lambdify(sym_list,pfoot_r[0],modules = [func_map])
    lambfooty_r = sym.lambdify(sym_list,pfoot_r[1],modules = [func_map])
    lambfootx_r_board = sym.lambdify(sym_list,pfoot_r_board[0],modules = [func_map])
    lambfooty_r_board = sym.lambdify(sym_list,pfoot_r_board[1],modules = [func_map])
    lambvfootx_r_board = sym.lambdify(sym_list2,vfootx_r_board,modules = [func_map])

    return [[[pfoot_l, pfoot_l_board, vfoot_l, vfootx_l_board],
             [pfoot_r, pfoot_r_board, vfoot_r, vfootx_r_board]],
            [[lambfootx_l, lambfooty_l, lambfootx_l_board, lambfooty_l_board, lambvfootx_l_board],
             [lambfootx_r, lambfooty_r, lambfootx_r_board, lambfooty_r_board, lambvfootx_r_board]]]