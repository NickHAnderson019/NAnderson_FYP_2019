# For 1 legged monopod on skateboard
from pyomo.environ import*
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from IPython.display import HTML
import numpy as np

from scipy.interpolate import interp1d

def solve(m, display_results):
    # Solve -----------------------------------------------------------------------------------------------------------------------
    opt = SolverFactory('ipopt') # standard issue, garden variety ipopt

    # solver options
    opt.options["expect_infeasible_problem"] = 'yes'
    opt.options["print_level"] = 5 # prints a log with each iteration (you want to this - it's the only way to see progress.)
    opt.options["max_iter"] = 50000 # maximum number of iterations
    opt.options["max_cpu_time"] = 10*60 # maximum cpu time in seconds
    opt.options["Tol"] = 1e-6 # the tolerance for feasibility. Considers constraints satisfied when they're within this margin.

    results = opt.solve(m, tee=display_results) 
    
    if results.solver.termination_condition == TerminationCondition.optimal: # doctest: +SKIP
        m.solutions.load_from(results) # doctest: +SKIP
    
    print(results.solver.status) # tells you if the solver had any errors/ warnings
    print(results.solver.termination_condition) # tells you if the solution was (locally) optimal, feasible, or neither.
    
    return results

def print_results(m,N, ground_constraints, contact_constraints, joint_constraints, joints):
    penalty_sum_ground = 0
    penalty_sum_contact = 0
    penalty_sum_joint = 0
    force_sum = 0
    torque_sum = 0
    for n in range(1,N+1):
        for gc in ground_constraints:
            penalty_sum_ground += m.ground_penalty[n,gc].value
        for cc in contact_constraints:
            penalty_sum_contact += m.contact_penalty[n,cc].value 
        for j in joints:
            for jc in joint_constraints:
                penalty_sum_joint += m.joint_penalty[n,j,jc].value 
        force_sum  += m.F_a[n].value**2 + m.Fb_b_total[n,'FF','y_b'].value**2+m.Fb_b_total[n,'FF','x_b'].value**2
        torque_sum += m.tau_a[n].value**2
                
    print("--------------------------------")
    print("GROUND:  ", penalty_sum_ground)
    print("CONTACTS:", penalty_sum_contact)
    print("JOINT:   ", penalty_sum_joint)
    print("FORCE:   ", force_sum)
    print("TORQUE:  ", torque_sum)
    print("--------------------------------")
    
def get_max_values(m, N, ground_constraints, contact_constraints, joint_constraints, joints, WDOFs, DOF_b, Fs, signs, GRFs):
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
            if m.GRFbf[i,grf].value is not None: 
                if abs(m.GRFbf[i,grf].value)>maxGRFbf:
                     maxGRFbf = abs(m.GRFbf[i,grf].value)
        if abs(m.F_a[i].value)>maxF_a:
            maxF_a = abs(m.F_a[i].value)
            
    return [maxForce, maxGRF, maxGRFbf, maxF_a]

def make_animation(m, N, N1, Nhw, N2, maxForce, maxGRF, maxGRFbf, maxF_a):
    #animate it
    fig1, ax1 = plt.subplots(1,1) #create axes
    ax1.set_aspect('equal')

    def plot_board(i,m,ax): #update function for animation
        XMIN = -2.0
        XMAX = 2.0
        YMIN = -0.5
        YMAX = 1.5
        MARKER_SIZE = 6

        ax.clear()
        ax.set_xlim([XMIN,XMAX])
        ax.set_ylim([YMIN,YMAX])

        #plot ground
        groundLx = XMIN
        groundLy = 0
        groundRx = XMAX
        groundRy = 0
        if i == N1 or i == Nhw or i == N2:
            ax.plot([groundLx,groundRx],[groundLy,groundRy],color='xkcd:red')
        else:
            ax.plot([groundLx,groundRx],[groundLy,groundRy],color='xkcd:black')

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

#             magforceBFx = -1
#             magforceBFy = 1

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

            ax.arrow(pBFxx, pBFxy, dpBFxx, dpBFxy, length_includes_head=True,head_width=abs(magforceBFx)*0.05,color='y')
            ax.arrow(pBFyx, pBFyy, dpBFyx, dpBFyy, length_includes_head=True,head_width=abs(magforceBFy)*0.05,color='y')
            
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
            
            ax.arrow(pFFxx, pFFxy, dpFFxx, dpFFxy, length_includes_head=True,head_width=abs(magforceFFx)*0.05,color='y')
            ax.arrow(pFFyx, pFFyy, dpFFyx, dpFFyy, length_includes_head=True,head_width=abs(magforceFFy)*0.05,color='y')
            
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

        ax.arrow(backGRFx, backGRFy-magGRFBW*0.5,0,magGRFBW*0.5, length_includes_head=True,head_width=magGRFBW*0.05,color='blue')
        ax.arrow(frontGRFx, frontGRFy-magGRFFW*0.5,0,magGRFFW*0.5, length_includes_head=True,head_width=magGRFFW*0.05,color='blue')

        #plot monopod
        #plot body
        body_xb = m.q[i,'xb'].value - 0.5*m.len[('body',1)]*cos(m.q[i,'thb'].value)
        body_yb = m.q[i,'yb'].value - 0.5*m.len[('body',1)]*sin(m.q[i,'thb'].value)
        body_xf = m.q[i,'xb'].value + 0.5*m.len[('body',1)]*cos(m.q[i,'thb'].value)
        body_yf = m.q[i,'yb'].value + 0.5*m.len[('body',1)]*sin(m.q[i,'thb'].value)  
        ax.plot([body_xb,body_xf],[body_yb,body_yf],color='k')

        #plot leg 1
        leg1_xt = m.q[i,'xb'].value
        leg1_yt = m.q[i,'yb'].value
        leg1_xb = m.q[i,'xb'].value + m.len[('leg',1)]*sin(m.thA[i].value)
        leg1_yb = m.q[i,'yb'].value - m.len[('leg',1)]*cos(m.thA[i].value)
        ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

        #plot leg 2
        Lt = 0.5*m.len[('leg',1)] + m.q[i,'r'].value - 0.5*m.len[('leg',2)]
        Lb = 0.5*m.len[('leg',1)] + m.q[i,'r'].value + 0.5*m.len[('leg',2)]
        leg2_xt = m.q[i,'xb'].value + Lt*sin(m.thA[i].value)
        leg2_yt = m.q[i,'yb'].value - Lt*cos(m.thA[i].value)
        leg2_xb = m.pfoot[i,'X'].value
        leg2_yb = m.pfoot[i,'Y'].value
        ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        
        if m.pfoot_b[i,'y_b'].value<1e-5:
            ax.plot(leg2_xb,leg2_yb,color='r',marker = '.')
        else:
            ax.plot(leg2_xb,leg2_yb,color='gray',marker = '.')

    update = lambda i: plot_board(i,m,ax1) #lambdify update function

    animate = ani.FuncAnimation(fig1, update, frames=range(1,N+1), interval=100, repeat=True)
    plt.close(animate._fig)
    
    return animate

def make_realtime_animation(m, N, hm):
    # store all variables for extrapolation
    #initialize time dependent arrays
    q_x_ar = np.zeros(N)
    q_y_ar = np.zeros(N)
    q_th_ar = np.zeros(N)
    q_xb_ar = np.zeros(N)
    q_yb_ar = np.zeros(N)
    q_thb_ar = np.zeros(N)
    q_thl_ar = np.zeros(N)
    q_r_ar = np.zeros(N)

    ptail_x_ar = np.zeros(N)
    ptail_y_ar = np.zeros(N)
    pnose_x_ar = np.zeros(N)
    pnose_y_ar = np.zeros(N)

    pbackwheel_x_ar = np.zeros(N)
    pbackwheel_y_ar = np.zeros(N)
    pfrontwheel_x_ar = np.zeros(N)
    pfrontwheel_y_ar = np.zeros(N)

#     F_BFx_ar = np.zeros(N)
#     F_BFy_ar = np.zeros(N)
#     F_FFx_ar = np.zeros(N)
#     F_FFy_ar = np.zeros(N)

#     rF_BF_ar = np.zeros(N)
#     rF_FF_ar = np.zeros(N)

#     GRF_FW_ar = np.zeros(N)
#     GRF_BW_ar = np.zeros(N)
    h_ar = np.zeros(N)

    # fill arrays with solver values
    for i in range (1,N+1):
        q_x_ar[i-1]    = m.q[i,'x'].value
        q_y_ar[i-1]    = m.q[i,'y'].value
        q_th_ar[i-1]  = m.q[i,'th'].value
        q_xb_ar[i-1]    = m.q[i,'xb'].value
        q_yb_ar[i-1]    = m.q[i,'yb'].value
        q_thb_ar[i-1]  = m.q[i,'thb'].value
        q_thl_ar[i-1]  = m.q[i,'thl'].value
        q_r_ar[i-1]  = m.q[i,'r'].value

        ptail_x_ar[i-1] = m.ptail[i,'X'].value
        ptail_y_ar[i-1] = m.ptail[i,'Y'].value
        pnose_x_ar[i-1] = m.pnose[i,'X'].value
        pnose_y_ar[i-1] = m.pnose[i,'Y'].value

        pbackwheel_x_ar[i-1] = m.pwheel[i,'BW','X'].value
        pbackwheel_y_ar[i-1] = m.pwheel[i,'BW','Y'].value
        pfrontwheel_x_ar[i-1] = m.pwheel[i,'FW','X'].value
        pfrontwheel_y_ar[i-1] = m.pwheel[i,'FW','Y'].value

#         F_BFx_ar[i-1] = m.Fb_b_total[i,'BF','x_b'].value
#         F_BFy_ar[i-1] = m.Fb_b_total[i,'BF','y_b'].value
#         F_FFx_ar[i-1] = m.Fb_b_total[i,'FF','x_b'].value
#         F_FFy_ar[i-1] = m.Fb_b_total[i,'FF','y_b'].value

#         rF_BF_ar[i-1]= m.rF[i,'BF'].value
#         rF_FF_ar[i-1]= m.rF[i,'FF'].value

#         GRF_FW_ar[i-1] = m.GRF[i,'FW'].value
#         GRF_BW_ar[i-1] = m.GRF[i,'BW'].value

        h_ar[i-1] = m.h[i].value*hm

    # calculate the time array. 
    t_ar = np.cumsum(h_ar)

    # caculate the interpolation functions.
    f_x_interp = interp1d(t_ar, q_x_ar, kind='cubic')
    f_y_interp = interp1d(t_ar, q_y_ar, kind='cubic')
    f_th_interp = interp1d(t_ar, q_th_ar, kind='cubic')
    f_xb_interp = interp1d(t_ar, q_xb_ar, kind='cubic')
    f_yb_interp = interp1d(t_ar, q_yb_ar, kind='cubic')
    f_thb_interp = interp1d(t_ar, q_thb_ar, kind='cubic')
    f_thl_interp = interp1d(t_ar, q_thl_ar, kind='cubic')
    f_r_interp = interp1d(t_ar, q_r_ar, kind='cubic')

    f_ptail_x_interp = interp1d(t_ar, ptail_x_ar, kind='cubic')
    f_ptail_y_interp = interp1d(t_ar, ptail_y_ar, kind='cubic')
    f_pnose_x_interp = interp1d(t_ar, pnose_x_ar, kind='cubic')
    f_pnose_y_interp = interp1d(t_ar, pnose_y_ar, kind='cubic')

    f_pbackwheel_x_interp = interp1d(t_ar, pbackwheel_x_ar, kind='cubic')
    f_pbackwheel_y_interp = interp1d(t_ar, pbackwheel_y_ar, kind='cubic')
    f_pfrontwheel_x_interp = interp1d(t_ar, pfrontwheel_x_ar, kind='cubic')
    f_pfrontwheel_y_interp = interp1d(t_ar, pfrontwheel_y_ar, kind='cubic')

#     f_F_BFx_interp = interp1d(t_ar, F_BFx_ar, kind='next')
#     f_F_BFy_interp = interp1d(t_ar, F_BFy_ar, kind='next')
#     f_F_FFx_interp = interp1d(t_ar, F_FFx_ar, kind='next')
#     f_F_FFy_interp = interp1d(t_ar, F_FFy_ar, kind='next')

#     f_rF_BF_interp = interp1d(t_ar, rF_BF_ar, kind='next')
#     f_rF_FF_interp = interp1d(t_ar, rF_FF_ar, kind='next')

#     f_GRF_FW_interp = interp1d(t_ar, GRF_FW_ar, kind='next')
#     f_GRF_BW_interp = interp1d(t_ar, GRF_BW_ar, kind='next')

    # the desired time steps of 0.01s (100Hz)
    Ts = hm
    t_d = np.arange(hm, t_ar[-1], Ts)

    # sample the interpolation functions at 50Hz.
    q_x_interp = f_x_interp(t_d)
    q_y_interp = f_y_interp(t_d)
    q_th_interp = f_th_interp(t_d)
    q_xb_interp = f_xb_interp(t_d)
    q_yb_interp = f_yb_interp(t_d)
    q_thb_interp = f_thb_interp(t_d)
    q_thl_interp = f_thl_interp(t_d)
    q_r_interp = f_r_interp(t_d)

    ptail_x_interp = f_ptail_x_interp(t_d)
    ptail_y_interp = f_ptail_y_interp(t_d)
    pnose_x_interp = f_pnose_x_interp(t_d)
    pnose_y_interp = f_pnose_y_interp(t_d)

    pbackwheel_x_interp = f_pbackwheel_x_interp(t_d)
    pbackwheel_y_interp = f_pbackwheel_y_interp(t_d)
    pfrontwheel_x_interp = f_pfrontwheel_x_interp(t_d)
    pfrontwheel_y_interp = f_pfrontwheel_y_interp(t_d)

#     F_BFx_interp = f_F_BFx_interp(t_d)
#     F_BFy_interp = f_F_BFy_interp(t_d)
#     F_FFx_interp = f_F_FFx_interp(t_d)
#     F_FFy_interp = f_F_FFy_interp(t_d)

#     rF_BF_interp = f_rF_BF_interp(t_d)
#     rF_FF_interp = f_rF_FF_interp(t_d)

#     GRF_FW_interp = f_GRF_FW_interp(t_d)
#     GRF_BW_interp = f_GRF_BW_interp(t_d)

    # number of samples
    N_interp = len(t_d)
    
    #animate it
    fig1, ax1 = plt.subplots(1,1) #create axes
    ax1.set_aspect('equal')
    
    def plot_board(i,m,ax): #update function for animation
        ax.clear()
        ax.set_title('Real Time')
        ax.set_xlim([-1,2])
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
        
        #plot monopod
        #plot body
        body_xb = q_xb_interp[i] - 0.5*m.len[('body',1)]*cos(q_thb_interp[i])
        body_yb = q_yb_interp[i] - 0.5*m.len[('body',1)]*sin(q_thb_interp[i])
        body_xf = q_xb_interp[i] + 0.5*m.len[('body',1)]*cos(q_thb_interp[i])
        body_yf = q_yb_interp[i] + 0.5*m.len[('body',1)]*sin(q_thb_interp[i])  
        ax.plot([body_xb,body_xf],[body_yb,body_yf],color='k')

        #plot leg 1
        leg1_xt = q_xb_interp[i]
        leg1_yt = q_yb_interp[i]
        leg1_xb = q_xb_interp[i] + m.len[('leg',1)]*sin(q_thb_interp[i]+q_thl_interp[i])
        leg1_yb = q_yb_interp[i] - m.len[('leg',1)]*cos(q_thb_interp[i]+q_thl_interp[i])
        ax.plot([leg1_xt,leg1_xb],[leg1_yt,leg1_yb],color='k')

        #plot leg 2
        Lt = 0.5*m.len[('leg',1)] + q_r_interp[i] - 0.5*m.len[('leg',2)]
        Lb = 0.5*m.len[('leg',1)] + q_r_interp[i] + 0.5*m.len[('leg',2)]
        leg2_xt = q_xb_interp[i] + Lt*sin(q_thb_interp[i]+q_thl_interp[i])
        leg2_yt = q_yb_interp[i] - Lt*cos(q_thb_interp[i]+q_thl_interp[i])
        leg2_xb = q_xb_interp[i] + Lb*sin(q_thb_interp[i]+q_thl_interp[i])
        leg2_yb = q_yb_interp[i] - Lb*cos(q_thb_interp[i]+q_thl_interp[i])
        ax.plot([leg2_xt,leg2_xb],[leg2_yt,leg2_yb],color='k')
        ax.plot(leg2_xb,leg2_yb,color='gray',marker = '.')

    update = lambda i: plot_board(i,m,ax1) #lambdify update function

    # animate it - interval is in ms
    animate = ani.FuncAnimation(fig1,update,frames=range(0,N_interp),interval=Ts*1000,repeat=True)
    plt.close(animate._fig)
    
    return animate
    
    