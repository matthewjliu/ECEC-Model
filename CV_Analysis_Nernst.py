"""
Updated January 2025
@author: Matthew Liu, Stanford University 

This script applies an additional 15% iR correction, plots/saves the data, and exports to Excel (columns A-B are the original data and C-D are the modified data)

For a CV with only one global min/max, the script will print the potential of the cathodic/anodic peaks. 

The script will also prompt you for how many cycles you'd like data from, please just type "1" and select which cycle you'd like from the text file (e.g., if 3 cycles were collected, just choose one of the three)
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# User inputs 
my_text_file = "CVA-0p1M-KBr-nonCat-CoIII-II.mpt"
KBr_string = "0p1M-KBr-nonCat"
Ru = 142.8 # as determined from PEIS measurement 
r = 1.5/10 # cm of the glassy carbon working electrode used to collect CVs 
area = np.pi * r**2  

# Create empty arrays
cycle_criteria = []
Ecat_half_array = []
Iplat_array = []
E_array = []
TOFmax_array = []
E_CV_Corrected = []
j_array = []

# Ask user for which cycles they want to extract data from
number_ofCycles = int(input("Enter number cycles you want data from: "))
for i in range(0, number_ofCycles):
    cycle_criterion = float(input("what cycle do you want data from?  ")) # enter in increasing order (e.g., 2 then 5, not 5 then 2)
    cycle_criteria.append(cycle_criterion)
splitData = [[] for _ in range(len(cycle_criteria))] # A 2-d array with this structure: [ [E1 I1] [E2 I2] ... [En Im] ] , where the subscript refers to the jth unique cycle


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


dataset = readBioLogic(my_text_file) # At this point, dataset is structured as: [ [all E values] [all I values] [all cycle numbers] ]

pairPosition = 0 
           
for currentTargetCycle in cycle_criteria:
    i = 0
    currentCycle = dataset[2][0]  
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

# iterate through each CV
for i in range(len(splitData)): # Iterating through each CV. splitData[0][0] gives you: [E_1,i I_1,i cycle number=i]
    E_CV_Full = np.asarray([x[0] for x in splitData[i]]) # Obtain array of E for the current CV 
    I_CV_Full = np.asarray([x[1] for x in splitData[i]]) # Obtain array of I for the current CV 

    # Find Ep for both the cathodic and anodic peaks 
    mymin_I = np.min(I_CV_Full) # find the cathodic peak current 
    mymax_I = np.max(I_CV_Full) # find the anodic peak current
    min_I_array =  [j for j, x in enumerate(I_CV_Full) if x == mymin_I] # get the index of the cathodic peak
    min_I_position = min_I_array[0]
    cathodic_peak_potential = E_CV_Full[min_I_position]
    max_I_array =  [j for j, x in enumerate(I_CV_Full) if x == mymax_I] # get the index of the cathodic peak
    max_I_position = max_I_array[0]
    anodic_peak_potential = E_CV_Full[max_I_position]
    
    for k in range(len(E_CV_Full)):
        E_CV_Corrected.append(E_CV_Full[k]-(I_CV_Full[k]/1000)*Ru*0.15)
        j_array.append(I_CV_Full[k]/area)
    
    cathodic_peak_potential_corrected = E_CV_Corrected[min_I_position]
    anodic_peak_potential_corrected = E_CV_Corrected[max_I_position]
   
    
#fig1 = plt.figure()
plt.plot(E_CV_Full, j_array, color="#EE82EE", label='85% iR compensation')
plt.plot(E_CV_Corrected, j_array, color="k", label='100% iR compensation')
#plt.title("Cycle 5, pH " + str(pH) + ": Scan rate " + str(v) + "V/sec")
plt.xlabel("Potential (V vs. Ag/AgCl)")
plt.ylabel("Current density (mA/cm2)")
plt.legend()
plt.savefig(KBr_string, dpi=300)
plt.title(KBr_string)
print("cathodic peak potential: "+ str(cathodic_peak_potential) + " V")
print("anodic peak potential: "+ str(anodic_peak_potential) + " V")

print("corrected cathodic peak potential: "+ str(cathodic_peak_potential_corrected) + " V")
print("corrected anodic peak potential: "+ str(anodic_peak_potential_corrected) + " V")




# Write ECEC model vs. experiment to excel file 
plots = [E_CV_Full, j_array, E_CV_Corrected, j_array]
df = pd.DataFrame(plots).T
df.to_excel(excel_writer = KBr_string + ".xlsx")

