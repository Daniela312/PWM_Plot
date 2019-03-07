import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal

t = np.linspace(2/50, 3/50, 7000)
    
def Graphing(sig_va, sig_vb, sig_vc, car_upper, car_lower):

      
    carwave_upper = car_upper
    carwave_lower = car_lower
    
    # sigwave = np.sin(1 * np.pi * 50 * t)
    sigwave_Va = sig_va
    sigwave_Vb = sig_vb
    sigwave_Vc = sig_vc

    pwmwave_Va = np.ones(t.shape[0])
    pwmwave_Vb = np.ones(t.shape[0])
    pwmwave_Vc = np.ones(t.shape[0])


    ar_Va = np.where(sigwave_Va > carwave_upper, 1, 
                    (np.where((sigwave_Va < carwave_upper) & (sigwave_Va > carwave_lower), 0, -1)))
    pwmwave_Va = ar_Va

    ar_Vb = np.where(sigwave_Vb > carwave_upper, 1, 
                    (np.where((sigwave_Vb < carwave_upper) & (sigwave_Vb > carwave_lower), 0, -1)))
    pwmwave_Vb = ar_Vb

    ar_Vc = np.where(sigwave_Vc > carwave_upper, 1, 
                    (np.where((sigwave_Vc < carwave_upper) & (sigwave_Vc > carwave_lower), 0, -1)))
    pwmwave_Vc = ar_Vc

    #t = np.linspace(-4*np.pi,4*np.pi,100)
    y1_upper = carwave_upper
    y1_lower = carwave_lower

    y2_Va = sigwave_Va
    y3_Va = pwmwave_Va

    y2_Vb = sigwave_Vb
    y3_Vb = pwmwave_Vb

    y2_Vc = sigwave_Vc
    y3_Vc = pwmwave_Vc

    # colors of each line and label
    car_color,Va,Vb,Vc= "grey","red","blue","green"

    # labels of each line
    l3_Va,l3_Vb,l3_Vc = "Va","Vb","Vc" 
    o1,o2,o3_Va,o3_Vb,o3_Vc = 10,10,5,5,5 # offsets of each line
    labels1 = [-2,-1,0,1,2]
    labels2 = [-2,-1,0,1,2]
    labels3 = [-1,0,1]
    yticks1 = [la+o1 for la in labels1]
    yticks3 = [la+o3_Va for la in labels3]
    

    ytls = labels1+labels3
    ytks = yticks1+yticks3

    plt.figure(figsize=(13,10),facecolor="w")

    # plot each line
    plt.plot(t*10**3,y3_Va+o3_Va,color=Va,linewidth=0.5)
    plt.plot(t*10**3,y2_Va+o2,color=Va,label=l3_Va, linewidth=0.7)

    plt.plot(t*10**3,y3_Vb+o3_Vb,color=Vb,linewidth=0.5)
    plt.plot(t*10**3,y2_Vb+o2,color=Vb,label=l3_Vb, linewidth=0.7)

    plt.plot(t*10**3,y3_Vc+o3_Vc,color=Vc,linewidth=0.5)
    plt.plot(t*10**3,y2_Vc+o2,color=Vc,label=l3_Vc, linewidth=0.7)

    plt.plot(t*10**3,y1_upper+o1,color=car_color, linewidth=0.5)
    plt.plot(t*10**3,y1_lower+o1,color=car_color, linewidth=0.5)
    plt.plot(t*10**3,y1_lower+o1,color=car_color, linewidth=0.5)

    # plot zero level for each line
    plt.plot([t[0]*10**3,t[-1]*10**3],[o1,o1],color=car_color,lw=0.1,label="")
    plt.plot([t[0]*10**3,t[-1]*10**3],[o2,o2],color="black",lw=0.1,label="")
    plt.plot([t[0]*10**3,t[-1]*10**3],[o3_Va,o3_Va],color="black",lw=0.1,label="")
    plt.plot([t[0]*10**3,t[-1]*10**3],[o3_Vb,o3_Vb],color="black",lw=0.1,label="")
    plt.plot([t[0]*10**3,t[-1]*10**3],[o3_Vc,o3_Vc],color="black",lw=0.1,label="")


    plt.ylim(o3_Va-1.5,o1+1.5)

    plt.yticks(ytks,ytls)
    plt.legend(loc="upper right",fontsize=10)
    plt.xlabel("Time [ms]")
    labs = plt.yticks()[1]
    for i in range(len(labs)):
        # if i < len(labels3):
        #     labs[i].set_color(car_color)
        # elif i < len(labels3+labels2):
        #     labs[i].set_color(Vc)
        # elif i < len(labels3+labels2+labels1):
        #     labs[i].set_color(Vb)
        # else:
        #     labs[i].set_color(Va)

        labs[i].set_color('black')


    # plt.savefig("./pwm_in_one_axes_plot.png",dpi=250,bbox_inches="tight",pad_inches=0.02)

    plt.show()


def graphing_first():  #초기기준전압 그래핑

    # 초기기준전압
    sigwave_Va = np.sin(1 * np.pi * 40 * t)
    sigwave_Vb = np.sin(1 * np.pi * 40 * t - 2/3*np.pi)
    sigwave_Vc = np.sin(1 * np.pi * 40 * t + 2/3*np.pi)

    carwave_upper = 1/2 * signal.sawtooth(np.pi * 1000 * t, width=0.5) + 1/2
    carwave_lower = 1/2 * signal.sawtooth(np.pi * 1000 * t, width=0.5) - 1/2

    Graphing(sigwave_Va, sigwave_Vb, sigwave_Vc,carwave_upper,carwave_lower)


def graphing_sec():  #수정기준전압 그래핑

    # 초기기준전압
    sigwave_Va = np.sin(1 * np.pi * 40 * t)
    sigwave_Vb = np.sin(1 * np.pi * 40 * t - 2/3*np.pi)
    sigwave_Vc = np.sin(1 * np.pi * 40 * t + 2/3*np.pi)
    
    #수정기준전압
    # sigwave_Va_mod = 0 * np.sin(1 * np.pi * 50 * t + 1/6 * np.pi)
    # sigwave_Vb_mod = -1 * 1/np.sqrt(3) * np.sin(1 * np.pi * 50 * t + 1/6 * np.pi)
    # sigwave_Vc_mod = -1 * 1/np.sqrt(3) * np.sin(1 * np.pi * 50 * t + 5/6 * np.pi)

    sigwave_Va_mod = sigwave_Va - sigwave_Va
    sigwave_Vb_mod = sigwave_Vb - sigwave_Va
    sigwave_Vc_mod = sigwave_Vc - sigwave_Va

    carwave_upper = np.sqrt(3)/2 * signal.sawtooth(np.pi * 1000 * t, width=0.5) + np.sqrt(3)/2
    carwave_lower = np.sqrt(3)/2 * signal.sawtooth(np.pi * 1000 * t, width=0.5) - np.sqrt(3)/2

    Graphing(sigwave_Va_mod, sigwave_Vb_mod, sigwave_Vc_mod,carwave_upper,carwave_lower)


def graphing_last():  #최종기준전압 그래핑

    # 초기기준전압
    sigwave_Va = np.sin(1 * np.pi * 40 * t)
    sigwave_Vb = np.sin(1 * np.pi * 40 * t - 2/3*np.pi)
    sigwave_Vc = np.sin(1 * np.pi * 40 * t + 2/3*np.pi)

    #최종기준전압 - 제어가능영역
    sigwave_Va_final = sigwave_Va - sigwave_Va - 1 * np.sqrt(3)
    sigwave_Vb_final = sigwave_Vb - sigwave_Va - 1 * np.sqrt(3)
    sigwave_Vc_final = sigwave_Vc - sigwave_Va - 1 * np.sqrt(3)


    carwave_upper = np.sqrt(3)/2 * signal.sawtooth(np.pi * 1000 * t, width=0.5) + np.sqrt(3)/2
    carwave_lower = np.sqrt(3)/2 * signal.sawtooth(np.pi * 1000 * t, width=0.5) - np.sqrt(3)/2

    Graphing(sigwave_Va_final, sigwave_Vb_final, sigwave_Vc_final, carwave_upper, carwave_lower)




graphing_last()






