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

# initial state: chemicals
Nm = 7
c0 = np.zeros(Nm)

# initial biomass and chemicals
y0 = np.concatenate((nCell0,c0))

# initial and final time with timestep 
tau0 = 0
tauf = 250
T = 22 # maturation time to reach dilution threshold (21.75 last, 23.12 largest)
dtau = 0.01
 
kSat = 1e4 # % interaction strength saturation level of each population %why saturation  -phrasing
#ExtTh = 0.1 # % population extinction threshold %high threshold to maintain (try lower); try number of individuals. (not part of pop)
ExtTh = 0.001 # Jamila: Lowering extinction threshold does not promote coexistence
DilTh = 1e7 # % coculture dilution threshold

nGen = 1000; # number of generations 
GenPerRound = np.log(DilTh/TID)/np.log(2)
Nr = round(nGen/GenPerRound); #% number of rounds of propagation


params_2075 = open_file_('params_coex_2075') # open parameters saved from coexistence.py
 # species, chemicals, time 

s2075,c2075 = dilution(Nr,Nc,c0,params_2075,TID,DilTh,ExtTh,tauf,'dynamics_events') # here each round is terminated as soon as species biomass reaches DilTh
# ds2075,dc2075,dt2075 = dilution_time(Nr,Nc,c0,params_2075,TID,DilTh,ExtTh,22,dtau,'dynamics2075_time') # each round terminated after each maturation time 22
