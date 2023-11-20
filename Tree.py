from core_functions import *


TID = 10**4 # total initial cell density
Nc = 50 # number of species

# Method1: the distribution of cell numbers for each species as a list 
PopDist = 1 / Nc * np.ones(Nc)
nCell0 = TID*PopDist # where cellRatio = PopDist in the simulation 

# Method2: the distribution of cell numbers for each species as a table
PopDist = 1 / Nc
label = ['Parameter', 'Change', 'Number','Length']
table0 = []
for i in range(Nc):
    table0.append(pd.DataFrame([[1, 0, TID*PopDist,0]], columns= label))

nCell0_table = [table0[_]['Number'][0] for _ in range(len(table0))]

# initial state: chemicals
Nm = 7
c0 = np.zeros(Nm)

# initial and final time with timestep 
tau0 = 0
tauf = 250
T = 22 # maturation time to reach dilution threshold (21.75 last, 23.12 largest)
dtau = 0.01
 
kSat = 1e4 # % interaction strength saturation level of each population %why saturation  -phrasing
ExtTh = 0.1 # % population extinction threshold %high threshold to maintain (try lower); try number of individuals. (not part of pop)
#   Jamila: Extinction threshold is ridiculously high. Is this from Babak?
DilTh = 1e7 # % coculture dilution threshold

nGen = 1000; # number of generations 
GenPerRound = np.log(DilTh/TID)/np.log(2)
Nr = round(nGen/GenPerRound); #% number of rounds of propagation

# locally simulated parameters, where seed 1343 achieves species coexistence 
#file_2075 = '/Users/indraina/Documents/University/Modeling Microbiome/Microbial Coexistence/Coexistence_data/params_coex/params_coex_50Nc2075'
#seed_2075 = open_directory(file_2075)
#params_2075 = open_file_(seed_2075)
params_2075 = open_file_('params_coex_2075')

params1 = params_tree(params_2075,ExtTh)

nCell, cMed, biomass_table = dilution_table(1,Nc,c0,table0,params1,TID,0,22,dtau,'name_')
