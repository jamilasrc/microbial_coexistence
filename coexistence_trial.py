from core_functions import *


Ns = 10000 # the number of communities or seeds to simulate  
### To trial various seeds and their surviving number of species, write "for i in range(Ns)",
### To see the parameters of a particular seed, write " for i in [seed]", where seed is the random seed which represents a community 

# for i in range(Ns):
# for i in [310,557,774,1547,1575]:
# for i in[1343]:
for i in [1668, 2075, 5485]: # here we are looking at 3 communities 
    np.random.seed(i)
    Nc = 50;  # of cell types in the initial pool
    Nm = 7;  # of mediators
    TID = 1e4; # total initial cell density
    kSat = 1e4; # interaction strength saturation level of each population %why saturation  -phrasing
    ExtTh = 0.1; # population extinction threshold %high threshold to maintain (try lower); try number of individuals. (not part of pop)
    DilTh = 1e7; # coculture dilution threshold

    min_gr = 0.1; #min value for any growth rate (1/hr)
    range_gr = 0.1; #range of variation of growth rate (0.1 - 0.2 currently)

    ri0 = 0.2; # maximum interaction strength, 1/hr
    fpi = 0.1; # fraction of interactions that are positive
    tau0 = 0; # in hours
    tauf = 250; # in hours
    dtau = 0.01; # in hours, cell growth update and uptake timescale
    at = 0.1; # avg. consumption values (fmole per cell); alpha_ij: population i, resource j
    bt = 0.2; # avg. production rates (fmole per cell per hour); beta_ij: population i, resource j
    qp = 0.5; # probability of production link per population
    qc = 0.3; # probability of influence link per population

    #previously I had 200 = nGen
    nGen = 1000; # total number of generations of community growth simulated
    GenPerRound = np.log(DilTh/TID)/np.log(2);
    Nr = round(nGen/GenPerRound); # number of rounds of propagation
    r0 = min_gr +range_gr*np.random.rand(Nc,1); # population reproduction rates, per hour
    kSatVector = kSat * (0.5 + np.random.rand(Nm, 1)); # population levels for influence saturation

    R = NetworkConfig_Binomial(Nc,Nm,qc)
    P = NetworkConfig_Binomial(Nc,Nm,qp); #chance of qp present or absent (zero), 1 means all links are present 
    # interaction matrix
    alpha = at * (0.5+np.random.rand(Nc,Nm)); # consumption rates
    beta = bt * (0.5+np.random.rand(Nc,Nm)); # mediator release rates
    A = (R*alpha).T
    B = (P*beta).T

    rIntMatA = R * DistInteractionStrengthMT_PA(Nc, Nm, ri0); # matrix of interaction coefficients, 50/50
    rIntMatB = R * DistInteractionStrengthMT_PB(Nc, Nm, ri0, fpi); # matrix of interaction coefficients, more negative
    rIntMatE = R * DistInteractionStrengthMT_PB(Nc, Nm, ri0, 1-fpi); # matrix of interaction coefficients, more positive

    # initial distribution for well-mixed communities
    PopDist = 1 / Nc * np.ones(Nc) # cell distribution; population ratios

    params1 = {}
    params1["num_spec"] = Nc
    params1["Nm"] = Nm
    params1["alpha"]  = A
    params1["beta"]  = B
    params1["K"]  = np.tile(kSatVector, Nc).T # chemical saturation is the same for all 7 chemicals for 20 species 
    params1["r0"] = r0
    params1["rho_plus"] ,params1["rho_minus"] = split_rho(rIntMatA) # 50% facilitative 50% inhibiting interaction 
    params1["rho_plusB"] ,params1["rho_minusB"] = split_rho(rIntMatB) 
    params1["rho_plusE"] ,params1["rho_minusE"] = split_rho(rIntMatE)
    params1["kSat"] = kSat

    n = coexistence(Nr,Nc,params1,TID,DilTh,ExtTh,tauf)
    result_save_(f'params_coex_{i}',params1) # saving the parameters which achieved coexistence 
    result_save_(f'seed_{i}',n) # where this will save the seeds, which contains the surviving number of species of each random seed at the end of 100 rounds 


i = 1668 # random seed we are looking at 

### If we had a collection of communities say, 100 seeds, this "seed_finder" counts surviving number of species for each 100 seed. "i" represents the seed or community 
# success = seed_finder(Ns,f'seed_{i}',6) # searching for the number of surviving species in this case 6 species after 100 rounds of dilution
# result_save_(f'success_seed_{i}',success)
