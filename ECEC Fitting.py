"""
Updated January 2025
@author: Matthew Liu, Stanford University

This script takes several user-defined inputs:
    1. data_file: The name of the .mpt text file (output from Biologic potentiostat) containing CV data. The file should be in the same directory as this script
    2. output_path and output_name: where outputs will be saved/named
    3. Physicochemical properties of the experiment, such electrolyte pH, electrolyte temp, scan rate, nitrate concentration, the CoII/I reduction peak potential measured in the absence of nitrate, 
    and the half-wave potential of the CoII/I couple measured in the absence of nitrate
    4. lower and upper potential limits over which a linear regression is performed on the CV (lower bound typically ~ 200 mV positive of E1; upper bound 100 mV positive of the lower bound). 
       The resulting linear fit will be used for capacitance subtraction. 
    
The script will also prompt you for how many cycles you'd like data from, in case multiple cycles are of interest. This text file contains CVs collected over 3 cycles. The 3rd cycle was analysis in the manuscript.
Try the script by specifying you would like data from 3 cycles, and enter 1, 2, 3. 

The script yields several outputs:
    1. 3 png files of: FOWA, the overall ECEC model fit, and the raw data vs. capacitance-subtracted data 
    2. 3 excel files: FOWA, ECEC model fit, and parameter values 
    
You will need to change output_path in Line 31 before running
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import r2_score

# User inputs 
data_file = "CVA-0p1M-KBr.mpt" # sample file to run the script on. It is one of five replicate CVs performed at 0.1 M KBr used to estimate the catalytic parameters in Table 1 of the main text. 
output_path = file-path-goes-here #e.g., "C:/Users/.../" (make sure to end with forward slash)
output_name = "0p1M-KBr-Rep1" 
pH = 5.0
Ip0 = -0.0799 # mA
E1 = -1.002 # V vs. Ag/AgCl --> estimated from noncatalytic voltammetry. Start with the average value and vary +/- by experimental error and RE drift as needed. 
Capacitive_window_left = -0.80 # V vs. Ag/AgCl, the lower bound over which we'll do a linear regression 
Capacitive_window_right = -0.70 # V vs. Ag/AgCl, the upper bound over which we'll do a linear regression 
Temperature = 295.15 # Kelvin 
Faraday = 96485 # C / mol 
GasConstant = 8.314 # J / mol / K
f = Faraday / GasConstant / Temperature # V-1 
v = 1.00 # V s-1 (scan rate) 
nprime = 4 # catalyst equivalents used in turnover in NH3. In theory, could be between 1-4 for CoDIM-catalyzed NO3RR. To avoid overestimating kinetics, use n'=4 
NO3 = 0.08 # M 
if pH < 7.0:
    HA = 10**-pH 
else: 
    HA = 55 # assume water is proton donor
    

# Create empty arrays
cycle_criteria = []
Ecat_half_array = []
Iplat_array = []
E_array = []
cap_array = []
k1_prime_array = []
k2_array = []
r2_array = []
TOFmax_array = []


# Ask user for which cycles they want to extract data from
number_ofCycles = int(input("Enter number cycles you want data from: "))
for i in range(0, number_ofCycles):
    cycle_criterion = float(input("what cycle do you want data from?  ")) # enter in increasing order (e.g., 2 then 5, not 5 then 2) 
    cycle_criteria.append(cycle_criterion)
splitData = [[] for _ in range(len(cycle_criteria))] # A 2-d array with this structure: [ [E1 I1] [E2 I2] ... [En Im] ] , where the subscript refers to the jth unique cycle


# helper function to identify the index of the element closest to some target "value" specified by the user 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

    
# This takes the raw .mpt file of any format and extracts only the columns containing EWE, I, and cycle number
def readBioLogic(filename):
    """ function for reading  .mpt files from EC-Lab

        Parameters
        ----------
        filename: string
            Filename of .mpt file to extract CV data from

        Returns
        -------

        PROVIDE INFO HERE


        """

    with open(filename, 'r', encoding="latin-1") as input_file:
        lines = input_file.readlines()

    header_line = lines[1]

    # mpt data format has variable number of header lines
    number_header_lines = int(header_line.split(":")[1])

    # find the columns for E_WE (working electrode potential), I (current), and cycle number
    headers = lines[number_header_lines-1].split('\t')

    EWE_cols = [o for o, h in enumerate(headers) if h == 'Ewe/V']
    I_cols = [o for o, h in enumerate(headers) if h == '<I>/mA']
    Cycle_cols = [o for o, h in enumerate(headers) if h == 'cycle number']

    col_heads = ['Ewe/V', '<I>/mA', 'cycle number']

    for cols, ch in zip([EWE_cols, I_cols, Cycle_cols], col_heads):
        assert len(cols) > 0, f'"{ch}" not found in column headers'

    EWE_col = EWE_cols[0] # Column number that E_WE values reside
    I_col = I_cols[0]
    Cycle_col = Cycle_cols[0]

    raw_data = lines[number_header_lines:]
    E, I, Cycles = [], [], []

    for line in raw_data:
        each = line.split('\t')
        currentCycle = float(each[Cycle_col])
        if currentCycle in cycle_criteria:
            E.append(float(each[EWE_col]))
            I.append(float(each[I_col]))
            Cycles.append(float(each[Cycle_col]))
    I_mA = np.array(I)
    # I_uA = 1000*I_mA # Convert from milliamps to microamps

    fullDataset = np.vstack([np.array(E),I_mA,np.array(Cycles)]) # fullDataset[0] is column of all E_WE values, [1] is column of all I values
    return fullDataset


dataset = readBioLogic(data_file) # At this point, dataset is structured as: [ [all E values] [all I values] [all cycle numbers] ]

pairPosition = 0 

for currentTargetCycle in cycle_criteria:
    i = 0
    currentCycle = dataset[2][0]  # This updates with each new line of data we traverse
    while currentTargetCycle != currentCycle:
        i=i+1
        currentCycle = dataset[2][i]
    while currentTargetCycle == currentCycle:
        if i < len(dataset[0])-1:
            splitData[pairPosition].append([dataset[0][i], dataset[1][i], dataset[2][i]])
            i = i+1
            currentCycle = dataset[2][i]
        else:
            break
    pairPosition = pairPosition + 1
    
# Assume you chose 3 cycles, cycles i, j, and k. Then, splitData currently looks like:
# [ [ [E_1,i I_1,i cycle number=i] [E_2,i I_2,i cycle number=i] ...  [E_1,j I_1,j cycle number=j] [E_2,j I_2,j cycle number=j] ...  [E_1,k I_1,k cycle number=k] [E_2,k I_2,k cycle number=k] ...] ] 


# Iterate through each CV 
for i in range(len(splitData)): # splitData[0][0] gives you: [E_1,i I_1,i cycle number=i]
    E_CV_Full = np.asarray([x[0] for x in splitData[i]]) # Obtain array of E for the current CV 
    I_CV_Full = np.asarray([x[1] for x in splitData[i]]) # Obtain array of I for the current CV 

    
    # The 5 lines below simply take our CV and extract the cathodic sweep 
    #mymin = np.min(E_CV_Full) # find the minimum potential in array E
    mymin = E_CV_Full[find_nearest(E_CV_Full, value=-1.2)]
    min_position_array = [j for j, x in enumerate(E_CV_Full) if x == mymin] # get the index of the minimum potential 
    min_position = min_position_array[0]
    E_LSV = E_CV_Full[:min_position]
    I_LSV = I_CV_Full[:min_position]


    # Next, we want to do a linear regression to approximate capacitance. The below 5 lines give us the capacitance line  
    cap_left = find_nearest(E_LSV, value=Capacitive_window_left) # get the index of left cutoff for linear regression
    cap_right = find_nearest(E_LSV, value=Capacitive_window_right) # get the index of left cutoff for linear regression
    E_cap_regression = E_LSV[cap_right:cap_left]
    I_cap_regression = I_LSV[cap_right:cap_left]
    m,b = np.polyfit(E_cap_regression, I_cap_regression, 1) # I = m*E+b
    
    # subtract off the capacitance from I_LSV. First, we  truncate the cathodic sweep to obtain the catalytic wave, with approximate domain [-1.2 V, -0.8 V] vs. Ag/AgCl
    trunc = find_nearest(E_LSV, value=-0.80)
    E_wave = E_LSV[trunc:]
    I_wave = I_LSV[trunc:]

    # (for plotting later)
    trunc_forPlot = find_nearest(E_LSV, value=-0.7)
    E_wave_forPlot = E_LSV[trunc_forPlot:]
    I_wave_forPlot = I_LSV[trunc_forPlot:]

    # subtract capacitive current 
    I_wave_Subtracted = []
    for j in range(len(I_wave)):
        capCurrent = m*E_wave[j]+b
        I_wave_Subtracted.append(I_wave[j]-capCurrent)
        if i == len(splitData)-1:
            cap_array.append(capCurrent)
            
    # Normalize the capacitance-subtracted catalytic wave by I0
    a_wave = np.asarray(I_wave_Subtracted)/-Ip0 
    
    # with our capacitance-subtracted, normalized catalytic wave (E_wave, a_wave), let's do FOWA to find k1', as well as the R2 in fitting the FOW
    
    FOWA_domain_raw = 1/(1+np.exp(f*(E_wave-E1)))
    FOWA_current_raw = -1 * a_wave
    
    FOWA_domain = 1/(1+np.exp(f*(E_wave-E1)))
    FOWA_current = -1 * a_wave
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(FOWA_domain, FOWA_current)
    k1 = (slope/4.481)**2 * (f*v/(nprime*NO3*HA**2))
    k1_prime = k1*NO3*HA**2
    k1_prime_array.append(k1_prime)
    r_squared = r_value**2
    roundedR2 = np.round(r_squared, 3)
    r2_array.append(roundedR2)
    FOWA_fit = slope*FOWA_domain + intercept 
    
    # Lastly, with the plateau current, find k2. also record I_plat, Ecat/2, and calculate TOF_max
    I_plateau = np.min(I_wave_Subtracted)
    Iplat_array.append(I_plateau)
    height = I_wave_Subtracted[0]-I_plateau
    I_half = I_wave_Subtracted[0]-(height/2)
    half_index = find_nearest(I_wave_Subtracted, value=I_half)
    Ecat_half = E_wave[half_index]
    Ecat_half_array.append(Ecat_half)
    
    k2 = ( (4.481 * np.sqrt(nprime/(f*v)) * (Ip0/I_plateau)) - (1/np.sqrt(k1_prime)) )**-2
    k2_array.append(k2)
    TOFmax = (k1_prime*k2)/(k1_prime+k2)
    TOFmax_array.append(TOFmax)
    

# Calculate the current predicted by the ECEC model using k1' and k2 
E_sim = np.linspace(-0.8, -1.20, len(a_wave))
I_normalized_numerator = -4.481*np.sqrt(nprime/(f*v))
I_normalized_denominator = (1/np.sqrt(k2))+(1/np.sqrt(k1_prime))*(1+np.exp(f*(E_sim-E1)))
I_sim = I_normalized_numerator / I_normalized_denominator




r2 = r2_score(a_wave, I_sim)
print("R2 for model current vs. experimental current (I/|Ip0|):")
print(r2)



# Export arrays to excel
df1 = pd.DataFrame(Ecat_half_array).T
df1.name = "E_cat/2 (V vs. 3.0 M KCl Ag/AgCl)"
df2 = pd.DataFrame(k1_prime_array).T
df2.name = "k1' (s-1)"
df3 = pd.DataFrame(r2_array).T
df3.name = "R2 from the linear regression of FOW used to derive k1"
df4 = pd.DataFrame(k2_array).T
df4.name = "k2 (s-1)"
df5 = pd.DataFrame(Iplat_array).T
df5.name = "I_plateau (mA) used to derive k2"
df6 = pd.DataFrame(TOFmax_array).T
df6.name = "TOFmax (s-1)"


writer = pd.ExcelWriter(output_path + 'kineticAnalysis_' + output_name + ".xlsx",engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet(output_name)
writer.sheets[output_name] = worksheet
worksheet.write_string(0, 0, df1.name)
df1.to_excel(writer,sheet_name=output_name,startrow=1 , startcol=0)
worksheet.write_string(df1.shape[0] + 3, 0, df2.name)
df2.to_excel(writer,sheet_name=output_name,startrow=df1.shape[0] + 4, startcol=0)
worksheet.write_string(df1.shape[0] + 7, 0, df3.name)
df3.to_excel(writer,sheet_name=output_name,startrow=df2.shape[0] + 8, startcol=0)
worksheet.write_string(df1.shape[0] + 11, 0, df4.name)
df4.to_excel(writer,sheet_name=output_name,startrow=df3.shape[0] + 12, startcol=0)
worksheet.write_string(df1.shape[0] + 15, 0, df5.name)
df5.to_excel(writer,sheet_name=output_name,startrow=df4.shape[0] + 16, startcol=0)
worksheet.write_string(df1.shape[0] + 19, 0, df6.name)
df6.to_excel(writer,sheet_name=output_name,startrow=df5.shape[0] + 20, startcol=0)
writer.save()




# Write ECEC model vs. experiment to excel file 
plots = [E_sim, I_sim, E_wave, a_wave]
df = pd.DataFrame(plots).T
df.to_excel(excel_writer = output_path + "model_vs_exp_" + output_name + ".xlsx")

# Write FOWA to excel file 
plots_FOWA = [FOWA_domain, FOWA_current, FOWA_fit]
df = pd.DataFrame(plots_FOWA).T
df.to_excel(excel_writer = output_path + "FOWA_" + output_name + ".xlsx")


# plot ECEC model vs. experiment for the last cycle you "want data from" in user input 
fig1 = plt.figure()
plt.plot(E_sim, I_sim, color="#EE82EE", linestyle='dashed', label='ECEC Model')
plt.plot(E_wave, a_wave, 'k', marker='o', markersize=1.5, linestyle='none', label='Experiment')
plt.title("pH " + str(pH) + ": Scan rate " + str(v) + "V/sec")
plt.xlabel("Applied potential (V vs. Ag/AgCl)")
plt.ylabel("I / |Ip0|")
plt.legend()
fig1.savefig('model_vs_exp_' + output_name + '.png')




# Plot the raw CV, the portion over which to do capacitive linear extrapolation, the linearly extracted line, and the background subtracted trace for the last cycle you "want data from" in user input 
fig2 = plt.figure()
plt.plot(E_wave_forPlot, I_wave_forPlot, '-k', label='Raw CV', linestyle='none', marker='o', markersize=1.5)
plt.plot(E_cap_regression, I_cap_regression, '-r', linestyle='dashed', label='portion for linear fit')
plt.plot(E_wave, I_wave_Subtracted, label='cap-substracted CV', linestyle='none', marker='o', markersize=1.5)
plt.plot(E_wave, cap_array, '-g', label='capacitive line', linestyle='dashed')
plt.title("pH " + str(pH) + ": Raw CV and Capacitance-Subtracted CV")
plt.xlabel("Applied potential (V vs. Ag/AgCl)")
plt.ylabel("Current (mA)")
plt.legend()
fig2.savefig('LSVCap_' + output_name + '.png')



# plot FOWA for the last cycle you "want data from" in user input 
fig3 = plt.figure()
plt.plot(FOWA_domain, FOWA_current, 'k', marker='o', markersize=1.5, linestyle='none', label='Capacitance-subtracted data')
plt.plot(FOWA_domain, FOWA_fit, 'r', label='Linear fit')
plt.xlabel("1/1+exp(f*(E-E1))")
plt.ylabel("I / Ip0")
plt.title("pH " + str(pH) + ": Foot of the Wave Analysis")
plt.legend()
fig3.savefig('FOWA_' + output_name + '.png')


