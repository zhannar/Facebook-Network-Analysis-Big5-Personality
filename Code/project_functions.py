def label_polynomial_features(input_df,power,bias):
    '''Basically this is a cover for the sklearn preprocessing function. 
    The problem with that function is if you give it a labeled dataframe, it ouputs an unlabeled dataframe with potentially
    a whole bunch of unlabeled columns. 
    
    Inputs:
    input_df = Your labeled pandas dataframe (list of x's not raised to any power) 
    power = what order polynomial you want variables up to. (use the same power as you want entered into pp.PolynomialFeatures(power) directly)
    bias = whether or not to include a bias/constant term in the begining
    
    Ouput:
    Output: This function relies on the powers_ matrix which is one of the preprocessing function's outputs to create logical labels and 
    outputs a labeled pandas dataframe   
    '''
    poly = pp.PolynomialFeatures(power,include_bias=bias)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    #print input_feature_names
    
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            #print "This is the input:", input_feature_names[i]
            if feature_distillation[i] == 0:
                continue            
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s^%d" % (variable,power)
                
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                    #final_label = final_label.replace("^1^1","^1")                    
                else:
                    final_label = final_label + " x " + intermediary_label
                    #final_label = final_label.replace("^1^1","^1")
            #print "This is the output:", final_label

        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df

def model_to_dictionary(model,X_test,y_test):
    model_dict = {}
    '''This function takes as an input a "model" which is the outcome of sm.OLS(y_train,X_train)
    And outputs a dictionary which extracts out certain key metrics of that model.'''

    # Step 1: Create select keys before/independent of training model
    model_dict["Y Name"] = model.endog_names
    model_dict["Feature Names"] = model.exog_names
    model_dict["Number Residuals"] = model.df_resid
    model_dict["Number Regressors"] = model.df_model    #Model degress of freedom. The number of regressors p. Does not include the constant if one is present

    # Step 2: Create select keys created from training the model
    results = model.fit()
    model_dict["TRAIN AIC"] = results.aic
    model_dict["TRAIN BIC"] = results.bic
    model_dict["TRAIN RSqAdj"] = results.rsquared_adj
    model_dict["TRAIN SSR"] = results.ssr      # Equal to sum(y_train - results.fittedvalues) **2)
    model_dict["TRAIN Confidence Interval"] = results.conf_int()
    model_dict["TRAIN PValues"] = results.pvalues
    model_dict["Estimated Coefficients"] = results.params
       
    # Step 3: Create select keys created from training the model
    y_hat = results.predict(X_test)
    model_dict["Test SSE"] = sum((y_hat - y_test)**2)
    
    # This is Jeremy's formula
    model_dict["Test RSqAdj"] = 1 - ((sum((y_test-y_hat)**2.0)/(len(X_test) - len(results.params) - 1))
                /(sum((y_test - np.mean(y_test))**2.0)/(len(X_test) - 1)))

    #model_dict["J_Test sse"] = (sum((y_test-y_hat)**2.0)
    #model_dict["J_Test sst"] = (sum((y_test - np.mean(y_test))**2.0)                         
    
    # This is my formula from here: http://onlinestatbook.com/2/effect_size/images/adjusted_rsquared.gif
    n = len(X_test)
    p = len(results.params)
    test_ssr =  model_dict["Test SSE"]
    z_test_sst =  sum((y_test - np.mean(y_test))**2.0)
                                #* ((n - 1))

    model_dict["z_test_sst"] = z_test_sst
    
    test_rsq = 1 - (test_ssr/z_test_sst)
    test_rsq_adj = 1 - (((1 - test_rsq) * (n-1)) / (n-p-1))
    model_dict["Z_Test RSqAdj"] = test_rsq_adj
    
    ###Is there something wrong with how R sq. adjusted is being calculated? why am i getting negative numbers?
    return model_dict 
    

def results_summary_to_dataframe(results):
    '''This takes the result of an statsmodel results table and transforms it into a dataframe'''
    import pandas as pd
    results_df = pd.DataFrame()    
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]
    
    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })
    
    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df