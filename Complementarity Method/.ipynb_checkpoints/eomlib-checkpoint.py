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

def get_EOMs_bipod():
    # system parameters
    g = sym.symbols('g')
    mb,mll1,mll2,mlr1,mlr2 = sym.symbols(['m_{body}','m_{leg1left}','m_{leg2left}','m_{leg1right}','m_{leg2right}']) # mass
    lb,ll1,ll2,lr1,lr2 = sym.symbols(['l_{body}','l_{leg1left}','l_{leg2left}','l_{leg1right}','l_{leg2right}']) # length
    Inb,Inll1,Inll2,Inlr1,Inlr2 = sym.symbols(['I_{body}','I_{leg1left}','I_{leg2left}','I_{leg1right}','I_{leg2right}']) # moment of intertia

    # generalized coordinates
    x,y,thb,thll,thlr,rl,rr = sym.symbols(['x','y','\\theta_{body}','\\theta_{legleft}','\\theta_{legright}','r_{left}','r_{right}']) 
    dx,dy,dthb,dthll,dthlr,drl,drr = sym.symbols(['\dot{x}','\dot{y}','\dot{\\theta}_{body}','\dot{\\theta}_{legleft}','\dot{\\theta}_{legright}','\dot{r}_{left}','\dot{r}_{right}']) 
    ddx,ddy,ddthb,ddthll,ddthlr,ddrl,ddrr = sym.symbols(['\ddot{x}','\ddot{y}','\ddot{\\theta}_{body}','\ddot{\\theta}_{legleft}','\ddot{\\theta}_{legright}','\ddot{r}_{left}','\ddot{r}_{right}']) 

    q = sym.Matrix([[x],[y],[thb],[thll],[thlr],[rl],[rr]])
    dq = sym.Matrix([[dx],[dy],[dthb],[dthll],[dthlr],[drl],[drr]])
    ddq = sym.Matrix([[ddx],[ddy],[ddthb],[ddthll],[ddthlr],[ddrl],[ddrr]])

    # forces
    F_a_l,F_a_r,tau_l,tau_r,GRFx_l,GRFy_l,GRFx_r,GRFy_r = sym.symbols(['F_{left}','F_{right}','\\tau_{left}','\\tau_{right}','G_{x_{left}}','G_{y_{left}}','G_{x_{right}}','G_{y_{right}}']) 


    # Unpickle stored EOMs
    infile = open('bipod_EOMs','rb')
    data = pkl.load(infile)
    infile.close()

    EOMs_bipod = data['EOMs']
    
    func_map = {'sin':sin, 'cos':cos} 

    sym_list = [g,mb,mll1,mll2,mlr1,mlr2,lb,ll1,ll2,lr1,lr2,Inb,Inll1,Inll2,Inlr1,Inlr2,
                x,y,thb,thll,rl,thlr,rr,
                dx,dy,dthb,dthll,drl,dthlr,drr,
                ddx,ddy,ddthb,ddthll,ddrl,ddthlr,ddrr,
                F_a_l,F_a_r,tau_l,tau_r,GRFx_l,GRFy_l,GRFx_r,GRFy_r]
            
    lambEOMxb_bipod   = sym.lambdify(sym_list,EOMs_bipod[0],modules = [func_map])
    lambEOMyb_bipod   = sym.lambdify(sym_list,EOMs_bipod[1],modules = [func_map])
    lambEOMthb_bipod  = sym.lambdify(sym_list,EOMs_bipod[2],modules = [func_map])
    lambEOMthll_bipod = sym.lambdify(sym_list,EOMs_bipod[3],modules = [func_map])
    lambEOMthlr_bipod = sym.lambdify(sym_list,EOMs_bipod[4],modules = [func_map])
    lambEOMrl_bipod   = sym.lambdify(sym_list,EOMs_bipod[5],modules = [func_map])
    lambEOMrr_bipod   = sym.lambdify(sym_list,EOMs_bipod[6],modules = [func_map])
    
    return [EOMs_bipod,[lambEOMxb_bipod, lambEOMyb_bipod, lambEOMthb_bipod, lambEOMthll_bipod, lambEOMrl_bipod, lambEOMthlr_bipod, lambEOMrr_bipod]]

def get_foot_v_and_p():
    # foot velocity and position in board and global frame
    
    # system variables
    xm,ym,thm,thl,r = sym.symbols(['x_{mono}','y_{mono}','\\theta_{mono}','\\theta_{link1}','r_{mono}']) # monopod in global frame
    dxm,dym,dthm,dthl,dr = sym.symbols(['\dot{x}_{mono}','\dot{y}_{mono}','\dot{\\theta}_{mono}','\dot{\\theta}_{link1}','\dot{r}_{mono}'])
    lb,l1,l2 = sym.symbols(['l_{board}','l_{link1}','l_{link2}']) # lenghts of links
    xb,yb,thb = sym.symbols(['x_{board}','y_{board}','\\theta_{board}']) # board in global frame
    dxb,dyb,dthb = sym.symbols(['\dot{x}_{board}','\dot{y}_{board}','\dot{\\theta}_{board}'])

    # Unpickle stored EOMs
    infile = open('bipod_EOMs','rb')
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