# ECEC-Model

Catalysis
---------
"ECEC Fitting.py" script was written to analyze .mpt output files of cyclic voltammograms (CVs) collected in EC-Lab® (BioLogic). Please read the comments at the beginning of the script for instructions and further context. An example .mpt file "CVA-0p1M-KBr.mpt" is provided to run the code.

Non-catalytic
---------------
"CV_Analysis_Nernst.py" script was written to apply an additional 15% IR correction to CVs collected with 85% IR correction. The resulting 100% IR corrected CV can be used to calculate anodic/cathodic peak potentials, from which half-wave potentials can be calculated. This is helpful for fitting data to the Nernst equation. An example .mpt file "CVA-0p1M-KBr-nonCat-CoIII-II" is provided to run the code. 

