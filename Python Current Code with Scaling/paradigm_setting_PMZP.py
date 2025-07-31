import numpy as np 

def paradigm_setting(paradigm, cond1, cond2):
    #This function sets up the paradigm configuration for the simulation to match
    #That of the empirical data and returns j which sets
    #up the paradigm trial orders in "j" and "ind" variables which return the
    #Trial type and and presentation indices

    if paradigm == 'face':
        reset_after = 2
        j= [cond1, cond1, cond2, cond2]

        condition = np.tile([1, 1, 0, 0], (1, 49))
        presentation = np.tile([1, 2], (1, 98))
        cond1_p1 = np.where((condition == 1) & (presentation == 1))[1]
        cond1_p2 = np.where((condition == 1) & (presentation == 2))[1]
        cond2_p1 = np.where((condition == 0) & (presentation == 1))[1]
        cond2_p2 = np.where((condition == 0) & (presentation == 2))[1]

        # Returning indices in a dictionary
        ind = {
            'cond1_p1': cond1_p1,
            'cond1_p2': cond1_p2,
            'cond2_p1': cond2_p1,
            'cond2_p2': cond2_p2
        }

        #A combination of optimal parameters that will fit the face data
        #(Found using a grid search analysses for a wide range of parameters)
        winning_params = {
            'sigma' : 0.2,
            'a' : 0.7,
            'b' : 0.2
        }
    elif paradigm == 'grating':
        reset_after = 6
        j=[ cond1, cond2, cond1, cond2, cond1, cond2, 
        cond2, cond1, cond2, cond1, cond2, cond1
        ]

        condition = np.tile([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], (1,4))
        presentation = np.tile([1, 1, 2, 2, 3, 3], (1,8))
        # print(np.where((condition==1) & (presentation == 1)))
        cond1_p1 = np.where((condition==1) & (presentation == 1))[1]
        cond1_p2 = np.where((condition==1) & (presentation == 2))[1]
        cond1_p3 = np.where((condition==1) & (presentation == 3))[1]
        cond2_p1 = np.where((condition==0) & (presentation == 1))[1]
        cond2_p2 = np.where((condition==0) & (presentation == 2))[1]
        cond2_p3 = np.where((condition==0) & (presentation == 3))[1]

        ind = {
            'cond1_p1': cond1_p1,
            'cond1_p2': cond1_p2,
            'cond1_p3': cond1_p3,
            'cond2_p1': cond2_p1,
            'cond2_p2': cond2_p2,
            'cond2_p3': cond2_p3
        }

        winning_params = {
            'sigma' : 0.4,
            'a' : 0.8,
            'b' : 0.4
        }
    else:
        raise Exception("Unknown Paradigm! Please set paradigm to either 'face' or 'grating'")


    return j, ind, reset_after, winning_params