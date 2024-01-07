"""

"""







from statsmodels.tsa import stattools as ts
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def optimize_arima(data,max_p=12,max_q=12,calculate_diff=False,print_summary=False,print_aic=False,test_run=False):


    """
    This function optimizes ARIMA model for given dataset and returns the params of the best model
    """

    if test_run:
        max_q = 2
        max_p = 2

    if calculate_diff:
        d = 1
    else:
        d = 0

    best_aic = np.inf
    best_p,best_q = None,None




    total_iterations = (max_p+1) * (max_q+1)
    i = 1

    for p in range(max_p+1):
        for q in range(max_q+1):

            print(p,q)

            print("-"*40,"ARIMA ",round(i/total_iterations*100,2),"%","-"*40)
            i+=1


            if q==0 and p == 0:
                continue

            order = (p,d,q)
            
            model = ARIMA(data,order=order)
            model = model.fit()


            aic = model.aic

            if print_aic:
                print("\nAIC: ",aic,"\n")
                
            if print_summary:
                print(model.summary())

            if aic < best_aic:
                best_aic = aic
                best_p,best_q = p,q
    
    model_parameters = {"parameter":{"arima_p":best_p,'arima_d':d,'arima_q':best_q,
                        "aic":best_aic}}
    
    return model_parameters



def optimize_garch(data,means = ['Constant', 'Zero', 'LS', 'AR', 'ARX', 'HAR', 'HARX', 'constant', 'zero'],
                   vols = ['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'HARCH', 'FIGARCH'],
                   distributions = ['normal', 'gaussian', 't', 'studentst', 'skewstudent', 'skewt', 'ged', 'generalized error'],
                   max_p=12,max_q=12,print_summary=False,print_aic=False,test_run=False):

    """
    This function optimizes garch model for given dataset and returns the params of the best model

    """



    total_iterations = len(means) * len(vols) * len(distributions) * (max_q+1) * (max_p+1)
    best_mean,best_vol,best_dist,best_p,best_q = None,None,None,None,None
    best_aic = np.inf



    if test_run:
        means = means[:1]
        vols = vols[:1]
        distributions = distributions[:1]
        max_p = 2
        max_q = 2
    
    o = 0
    i = 1
    for mean in means:
        for vol in vols:
            for distribution in distributions:
                for p in range(max_p+1):
                    for q in range(max_q+1):
                        
                        print("-"*40,"GARCH ",round(i/total_iterations*100,5),"%","-"*40)
                        i += 1
                        if p == 0 and q == 0:
                            continue

                        
                        if p == 0:
                            o = 1


                        print(q,p)
                        try:
                            model = arch_model(data,vol = vol, p = p, q = q,o=o, mean = mean, dist = distribution)
                        except:
                            continue
                        result = model.fit()

                        if print_summary:
                            print(result.summary())

                        aic = result.aic
                        if print_aic:

                            print("AIC: ",aic)

                        if aic < best_aic:
                            best_aic = aic
                            best_mean,best_vol,best_dist,best_p,best_q = mean,vol,distribution,p,q

      
      
      
      
    best_mean,best_vol,best_dist,best_p,best_q,o = "AR","GARCH","normal",1,1,0

    
    model_parameters = {"parameter":{"mean":best_mean,"volatility":best_vol,"distribution":best_dist,"p":best_p,"q":best_q,"o":o,
                        "aic":best_aic}}
    
    return model_parameters






def optimize_var():

    """
    This function optimizes VAR model for given dataset and returns the params of the best model
    """

