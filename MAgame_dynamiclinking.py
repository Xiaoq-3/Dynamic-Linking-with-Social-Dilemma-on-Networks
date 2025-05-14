# partner selection, PD
# Q, knowing every opponent past action
import os
import pickle
import numpy as np
import networkx as nx
import random

from Environment02 import Environment02
from Agent36b import Agent36b

def mainTrain(gameName, Nact, roundsG, algoName, lr, Nagent, simStart, Nsim, tStart, T, Prob0, Prob1):

    Npast = 1
    dirName1 = 'result_%s_%s_lr%.2f_Npast%d_Nagent%d_R%d_Prob0%.2f_Prob1%.4f_sep_random'%(gameName, algoName, lr, Npast, Nagent, roundsG, Prob0, Prob1)
    print(dirName1)
    if not os.path.exists(dirName1):
        os.makedirs(dirName1)

    reward1s = {}
    reward2s = {}
    # PD2
    reward1s['PD2'] = np.array([[3, -1], [5, 0]])
    reward2s['PD2'] = np.array([[3, 5], [-1, 0]])

    env = Environment02(Nact, reward1s[gameName], reward2s[gameName])

    np.random.seed(42)

    for sim in range(simStart, Nsim+1):
        dirName = '%s/sim%04d'%(dirName1, sim)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        Nelement1 = (Npast + 1) * 4 + (Npast+1) * Nact
        StatQmeanT = np.zeros((T + 1, Nelement1), dtype = np.float32)
        StatQvarT = np.zeros((T + 1, Nelement1), dtype = np.float32)
        StatPmeanT = np.zeros((T + 1, Nelement1), dtype = np.float32)
        StatPvarT = np.zeros((T + 1, Nelement1), dtype = np.float32)

        CountLinkT = np.zeros((T, 2), dtype=int) # 0:N; 1:Y (Link)
        CountBreakT = np.zeros((T, 2), dtype=int) # 0:N; 1:Y (Break)
        CountOutcomeT = np.zeros((T, 4), dtype=int) # 0:(C,C); 1:(C,D); 2:(D,C); 3:(D,D)
        CountActualOutcomeT1 = np.zeros((T, 8), dtype=int)      #  0:NotLink(C,C); 1:NotLink(C,D); 2:NotLink(D,C) 3:NotLink(D,D)
                                                                #  4:Link(C,C) 5:Link(C,D) 6:Link(D,C) 7:Link(D,D) 
        CountActualOutcomeT2 = np.zeros((T, 8), dtype=int)      #  0:NotBreak(C,C); 1:NotBreak(C,D); 2:NotBreak(D,C); 3:NotBreak(D,D)
                                                                #  4:Break(C,C); 5:Break(C,D); 6:Break(D,C); 7:Break(D,D)
        AgentRewardSumT = np.zeros((T, Nagent), dtype=int)

        AgentPolicyLkTypeT = np.zeros((T, Nagent), dtype=int)  # 0; 1; 2; 3
        AgentPolicyBkTypeT = np.zeros((T, Nagent), dtype=int)  # 0; 1; 2; 3
        AgentPolicyPDTypeT = np.zeros((T, Nagent), dtype=int) # 0; 1; 2; 3
        CountPolicyLkTypeT = np.zeros((T, 4), dtype=int)  # N|oC,N|oD; N|oC,Y|oD; Y|oC,N|oD; Y|oC,Y|oD
        CountPolicyBkTypeT = np.zeros((T, 4), dtype=int)  # N|oC,N|oD; N|oC,Y|oD; Y|oC,N|oD; Y|oC,Y|oD
        CountPolicyPDTypeT = np.zeros((T, 4), dtype=int)  # C|oC,C|oD; C|oC,D|oD; D|oC,C|oD; D|oC,D|oD

        AgentCountLkT = np.zeros((T, Nagent), dtype=int)  # 0-roundG, number of Y (Link)
        AgentCountBkT = np.zeros((T, Nagent), dtype=int)  # 0-roundG, number of Y (Break)
        AgentCountDT = np.zeros((T, Nagent), dtype=int)  # 0-roundG, number of D

        CountLink_oAT = np.zeros((T, 4), dtype=int) # 0:N|oC; 1:Y|oC; 2:N|oD; 3:Y|oD;
        CountBreak_oAT = np.zeros((T, 4), dtype=int) # 0:N|oC; 1:Y|oC; 2:N|oD; 3:Y|oD;
        CountCD_oAT = np.zeros((T, 4), dtype=int) # 0:C|oC; 1:D|oC; 2:C|oD; 3:D|oD;
        
        CountPDT = np.zeros((T, 4), dtype=int) # 0:(C,C); 1:(C,D); 2:(D,C); 3:(D,D)

        if tStart != 0:
            StatQmeanTinit = np.loadtxt(('%s/StatQmeanT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatQvarTinit = np.loadtxt(('%s/StatQvarT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatPmeanTinit = np.loadtxt(('%s/StatPmeanT-sim%04d.txt')%(dirName, sim), delimiter=',')
            StatPvarTinit = np.loadtxt(('%s/StatPvarT-sim%04d.txt')%(dirName, sim), delimiter=',')
            
            StatQmeanT[:tStart+1] = StatQmeanTinit[:tStart+1]
            StatQvarT[:tStart+1] = StatQvarTinit[:tStart+1]
            StatPmeanT[:tStart+1] = StatPmeanTinit[:tStart+1]
            StatPvarT[:tStart+1] = StatPvarTinit[:tStart+1]
            
            CountLinkTinit = np.loadtxt(('%s/CountLinkT-sim%04d.txt')%(dirName, sim), delimiter=',')
            CountBreakTinit = np.loadtxt(('%s/CountBreakT-sim%04d.txt')%(dirName, sim), delimiter=',')
            CountOutcomeTinit = np.loadtxt(('%s/CountOutcomeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            
            AgentRewardSumTinit = np.loadtxt(('%s/AgentRewardSumT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountLinkT[:tStart] = CountLinkTinit[:tStart]
            CountBreakT[:tStart] = CountBreakTinit[:tStart]
            CountOutcomeT[:tStart] = CountOutcomeTinit[:tStart]
            
            AgentRewardSumT[:tStart] = AgentRewardSumTinit[:tStart]
            CountActualOutcomeTinit1 = np.loadtxt(('%s/CountActualOutcomeT1-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountActualOutcomeT1[:tStart] = CountActualOutcomeTinit1[:tStart]
            CountActualOutcomeTinit2 = np.loadtxt(('%s/CountActualOutcomeT2-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountActualOutcomeT2[:tStart] = CountActualOutcomeTinit2[:tStart]
            
            AgentPolicyLkTypeTinit = np.loadtxt(('%s/AgentPolicyLkTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentPolicyBkTypeTinit = np.loadtxt(('%s/AgentPolicyBkTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountPolicyLkTypeTinit = np.loadtxt(('%s/CountPolicyLkTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountPolicyBkTypeTinit = np.loadtxt(('%s/CountPolicyBkTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentPolicyLkTypeT[:tStart] = AgentPolicyLkTypeTinit[:tStart]
            AgentPolicyBkTypeT[:tStart] = AgentPolicyBkTypeTinit[:tStart]
            CountPolicyLkTypeT[:tStart] = CountPolicyLkTypeTinit[:tStart]
            CountPolicyBkTypeT[:tStart] = CountPolicyBkTypeTinit[:tStart]
            AgentPolicyPDTypeTinit = np.loadtxt(('%s/AgentPolicyPDTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountPolicyPDTypeTinit = np.loadtxt(('%s/CountPolicyPDTypeT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentPolicyPDTypeT[:tStart] = AgentPolicyPDTypeTinit[:tStart]
            CountPolicyPDTypeT[:tStart] = CountPolicyPDTypeTinit[:tStart]
            
            AgentCountLkTinit = np.loadtxt(('%s/AgentCountLkT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentCountBkTinit = np.loadtxt(('%s/AgentCountBkT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentCountDTinit = np.loadtxt(('%s/AgentCountDT-sim%04d.txt') % (dirName, sim), delimiter=',')
            AgentCountLkT[:tStart] = AgentCountLkTinit[:tStart]
            AgentCountBkT[:tStart] = AgentCountBkTinit[:tStart]
            AgentCountDT[:tStart] = AgentCountDTinit[:tStart]
            CountLink_oATinit = np.loadtxt(('%s/CountLink_oAT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountBreak_oATinit = np.loadtxt(('%s/CountBreak_oAT-sim%04d.txt') % (dirName, sim), delimiter=',')

            CountCD_oATinit = np.loadtxt(('%s/CountCD_oAT-sim%04d.txt') % (dirName, sim), delimiter=',')
            CountLink_oAT[:tStart] = CountLink_oATinit[:tStart]
            CountBreak_oAT[:tStart] = CountBreak_oATinit[:tStart]

            CountCD_oAT[:tStart] = CountCD_oATinit[:tStart]
        else:
            StatPmeanT[0, :] = 1/2

        # initialize
        if tStart != 0:
            filePath1 = '%s/Agents-sim%04d_T%d.pickle' % (dirName, sim, tStart)
            f01 = open(filePath1, 'rb')
            agents = pickle.load(f01)
            f01.close()
            filePath1 = '%s/partners-sim%04d_T%d.pickle' % (dirName, sim, tStart)
            f01 = open(filePath1, 'rb')
            partners = pickle.load(f01)
            f01.close()
            filePath1 = '%s/ActionsPD-sim%04d_T%d.pickle' % (dirName, sim, tStart)
            f01 = open(filePath1, 'rb')
            actionsPD = pickle.load(f01)
            f01.close()
        else:
            agents = []
            for i in range(Nagent):
                agent = Agent36b(env, lr=lr, name='%s_%d'%(algoName, i))
                agents.append(agent)
            partners = [[] for _ in range(Nagent)]
            random_graph = nx.random_regular_graph(Nagent - 1, Nagent)
            for i in range(Nagent):
                partners[i] = list(random_graph.neighbors(i))
            actionsPD = random.choices([0, 1], weights=[Prob0, 1-Prob0], k = 100)
            #print(actionsPD)
            #actionsPD = np.random.choice(Nact, Nagent)
            filePath1 = '%s/Agents-sim%04d_T%d.pickle' % (dirName, sim, 0)
            f01 = open(filePath1, 'wb')
            pickle.dump(agents, f01)
            f01.close()
            filePath1 = '%s/partners-sim%04d_T%d.pickle' % (dirName, sim, 0)
            f01 = open(filePath1, 'wb')
            pickle.dump(partners, f01)
            f01.close()
            filePath1 = '%s/ActionsPD-sim%04d_T%d.pickle' % (dirName, sim, 0)
            f01 = open(filePath1, 'wb')
            pickle.dump(actionsPD, f01)
            f01.close()
            list1 = [0] * int(roundsG * Prob1)
            list2 = [1] * roundsG
            listall = list1 + list2
            l = len(listall)
            print(l)

        for t in range(tStart+1, T+1):
            CountLink = np.zeros(2, dtype=int)
            CountBreak = np.zeros(2, dtype=int)
            CountOutcome = np.zeros(4, dtype=int)
            CountActualOutcome1 = np.zeros(8, dtype=int)
            CountActualOutcome2 = np.zeros(8, dtype=int)
            AgentRewardSum = np.zeros(Nagent, dtype=int)
            AgentCountLk = np.zeros(Nagent, dtype=int)
            AgentCountBk = np.zeros(Nagent, dtype=int)
            AgentCountD = np.zeros(Nagent, dtype=int)
            CountLink_oA = np.zeros(4, dtype=int)
            CountBreak_oA = np.zeros(4, dtype=int)
            CountCD_oA = np.zeros(4, dtype=int)
            CountPD = np.zeros(4, dtype=int)
            
            random.shuffle(listall)
            #print(listall)
            

            for round1 in range(l):
                
                # PS
                isLink = np.zeros(Nagent, dtype=int)
                isBreak = np.zeros(Nagent, dtype=int)
                
                # choose edge
                i, j = random.sample(range(Nagent), 2)
                
                if listall[round1] == 0:
                    #print('link')
                # link or not
                    if i not in partners[j]:
                        s = [actionsPD[i], actionsPD[j]]
                        isLink1 = [agents[i].getAction(s[0]), agents[j].getAction(s[1])]
                        agents[i].storeMemory(s[0], isLink1[0], 0)
                        agents[j].storeMemory(s[1], isLink1[1], 0)
                        CountLink[isLink1[0]] += 1
                        CountLink[isLink1[1]] += 1
                        CountLink_oA[s[0] * 2 + isLink1[0]] += 1
                        CountLink_oA[s[1] * 2 + isLink1[1]] += 1
                        AgentCountLk[i] += isLink1[0]
                        AgentCountLk[j] += isLink1[1]
                        
                        if isLink1[0] == 1 and isLink1[1] == 1:
                        # If both decide to link
                        
                            partners[i] += [j]
                            partners[j] += [i]
                            isLink[i] = 1
                            isLink[j] = 1
                              
                            
                    # break or not
                    else:
                        s = [100 + actionsPD[j], 100 + actionsPD[i]]
                        isBreak1 = [agents[i].getAction(s[0]), agents[j].getAction(s[1])]
                        agents[i].storeMemory(s[0], isBreak1[0], 0)
                        agents[j].storeMemory(s[1], isBreak1[1], 0)
                        CountBreak[isBreak1[0]] += 1
                        CountBreak[isBreak1[1]] += 1
                        CountBreak_oA[(s[0]-100) * 2 + isBreak1[0]] += 1
                        CountBreak_oA[(s[1]-100) * 2 + isBreak1[1]] += 1
                        AgentCountBk[i] += isBreak1[0]
                        AgentCountBk[j] += isBreak1[1]
                    
                        if isBreak1[0] == 1 or isBreak1[1] == 1:
                                
                            isBreak[i] = 1
                            isBreak[j] = 1
                            
                            if all(len(partners[k]) > 1 for k in (i, j)):
                                partners[i].remove(j)
                                partners[j].remove(i)
                    
                else:
                    #print('game')
                    i = random.choice(range(Nagent))
                    j = random.choice(partners[i])
                    s = [200 + actionsPD[j], 200 + actionsPD[i]]
                    actionsPD1 = [agents[i].getAction(s[0]), agents[j].getAction(s[1])]
                    rewards1 = env.getRewards(actionsPD1)
                    agents[i].storeMemory(s[0], actionsPD1[0], rewards1[0])
                    agents[j].storeMemory(s[1], actionsPD1[1], rewards1[1])
                    CountPD[2 * actionsPD1[0] + actionsPD1[1]] += 1
                    #CountActualOutcome1[4*isLink[i] + 2*actionsPD1[0] + actionsPD1[1]] += 1
                    AgentRewardSum[i] += rewards1[0]
                    AgentRewardSum[j] += rewards1[1]
                    AgentCountD[i] += actionsPD1[0]
                    AgentCountD[j] += actionsPD1[1]
    
                    actionsPD[i] = actionsPD1[0]
                    actionsPD[j] = actionsPD1[1]    
                            
                
            for i in range(Nagent):
                agents[i].train()
                
            # record
            Qsum = np.zeros(Nelement1, dtype=np.float32)
            Qsqsum = np.zeros(Nelement1, dtype=np.float32)
            Psum = np.zeros(Nelement1, dtype=np.float32)
            Psqsum = np.zeros(Nelement1, dtype=np.float32)
            AgentPolicyLkType = np.zeros(Nagent, dtype=int)
            AgentPolicyBkType = np.zeros(Nagent, dtype=int)
            CountPolicyLkType = np.zeros(4, dtype=int)
            CountPolicyBkType = np.zeros(4, dtype=int)
            AgentPolicyPDType = np.zeros(Nagent, dtype=int)
            CountPolicyPDType = np.zeros(4, dtype=int)
            
            for i in range(len(agents)):
                
                Qsum += agents[i].getQall()
                Qsqsum += agents[i].getQall() ** 2
                Psum += agents[i].getPolicyAll()
                Psqsum += agents[i].getPolicyAll() ** 2

                Q_oC = agents[i].Q[0]
                Q_oD = agents[i].Q[1]
                if Q_oC[0] > Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyLkType[0] += 1
                    AgentPolicyLkType[i] = 0
                elif Q_oC[0] > Q_oC[1] and Q_oD[0] <= Q_oD[1]:
                    CountPolicyLkType[1] += 1
                    AgentPolicyLkType[i] = 1
                elif Q_oC[0] <= Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyLkType[2] += 1
                    AgentPolicyLkType[i] = 2
                else:
                    CountPolicyLkType[3] += 1
                    AgentPolicyLkType[i] = 3

                Q_oC = agents[i].Q[100]
                Q_oD = agents[i].Q[101]
                if Q_oC[0] > Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyBkType[0] += 1
                    AgentPolicyBkType[i] = 0
                elif Q_oC[0] > Q_oC[1] and Q_oD[0] <= Q_oD[1]:
                    CountPolicyBkType[1] += 1
                    AgentPolicyBkType[i] = 1
                elif Q_oC[0] <= Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyBkType[2] += 1
                    AgentPolicyBkType[i] = 2
                else:
                    CountPolicyBkType[3] += 1
                    AgentPolicyBkType[i] = 3
                    
                Q_oC = agents[i].Q[200]
                Q_oD = agents[i].Q[201]
                if Q_oC[0] > Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyPDType[0] += 1
                    AgentPolicyPDType[i] = 0
                elif Q_oC[0] > Q_oC[1] and Q_oD[0] <= Q_oD[1]:
                    CountPolicyPDType[1] += 1
                    AgentPolicyPDType[i] = 1
                elif Q_oC[0] <= Q_oC[1] and Q_oD[0] > Q_oD[1]:
                    CountPolicyPDType[2] += 1
                    AgentPolicyPDType[i] = 2
                else:
                    CountPolicyPDType[3] += 1
                    AgentPolicyPDType[i] = 3
                    
            StatQmeanT[t] = Qsum / Nagent
            StatQvarT[t] = Qsqsum / Nagent - (Qsum / Nagent) ** 2
            StatPmeanT[t] = Psum / Nagent
            StatPvarT[t] = Psqsum / Nagent - (Psum / Nagent) ** 2
            CountLinkT[t - 1] = CountLink
            CountBreakT[t - 1] = CountBreak
            CountOutcomeT[t - 1] = CountOutcome
            CountActualOutcomeT1[t - 1] = CountActualOutcome1
            CountActualOutcomeT2[t - 1] = CountActualOutcome2
            AgentRewardSumT[t - 1] = AgentRewardSum
            AgentCountLkT[t - 1] = AgentCountLk
            AgentCountBkT[t - 1] = AgentCountBk
            AgentCountDT[t - 1] = AgentCountD

            AgentPolicyLkTypeT[t - 1] = AgentPolicyLkType
            AgentPolicyBkTypeT[t - 1] = AgentPolicyBkType
            CountPolicyLkTypeT[t - 1] = CountPolicyLkType
            CountPolicyBkTypeT[t - 1] = CountPolicyBkType
            AgentPolicyPDTypeT[t - 1] = AgentPolicyPDType
            CountPolicyPDTypeT[t - 1] = CountPolicyPDType
            CountLink_oAT[t-1] = CountLink_oA
            CountBreak_oAT[t-1] = CountBreak_oA
            CountCD_oAT[t-1] = CountCD_oA
            CountPDT[t-1] = CountPD

            if t % 100 == 0 or t==T:
                filePath1 = '%s/Agents-sim%04d_T%d.pickle' % (dirName, sim, t)
                f01 = open(filePath1, 'wb')
                pickle.dump(agents, f01)
                f01.close()
                filePath1 = '%s/partners-sim%04d_T%d.pickle' % (dirName, sim, t)
                f01 = open(filePath1, 'wb')
                pickle.dump(partners, f01)
                f01.close()
                filePath1 = '%s/ActionsPD-sim%04d_T%d.pickle' % (dirName, sim, t)
                f01 = open(filePath1, 'wb')
                pickle.dump(actionsPD, f01)
                f01.close()

                np.savetxt(('%s/StatQmeanT-sim%04d.txt') % (dirName, sim), StatQmeanT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/StatQvarT-sim%04d.txt') % (dirName, sim), StatQvarT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/StatPmeanT-sim%04d.txt') % (dirName, sim), StatPmeanT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/StatPvarT-sim%04d.txt') % (dirName, sim), StatPvarT, fmt='%.6f', delimiter=',')
                np.savetxt(('%s/CountLinkT-sim%04d.txt') % (dirName, sim), CountLinkT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountOutcomeT-sim%04d.txt') % (dirName, sim), CountOutcomeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountActualOutcomeT1-sim%04d.txt') % (dirName, sim), CountActualOutcomeT1, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountActualOutcomeT2-sim%04d.txt') % (dirName, sim), CountActualOutcomeT2, fmt='%d', delimiter=',')

                np.savetxt(('%s/AgentRewardSumT-sim%04d.txt') % (dirName, sim), AgentRewardSumT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentCountLkT-sim%04d.txt') % (dirName, sim), AgentCountLkT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentCountBkT-sim%04d.txt') % (dirName, sim), AgentCountBkT, fmt='%d', delimiter=',')

                np.savetxt(('%s/AgentCountDT-sim%04d.txt') % (dirName, sim), AgentCountDT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentPolicyLkTypeT-sim%04d.txt') % (dirName, sim), AgentPolicyLkTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/AgentPolicyBkTypeT-sim%04d.txt') % (dirName, sim), AgentPolicyBkTypeT, fmt='%d', delimiter=',')

                np.savetxt(('%s/CountPolicyLkTypeT-sim%04d.txt') % (dirName, sim), CountPolicyLkTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountPolicyBkTypeT-sim%04d.txt') % (dirName, sim), CountPolicyBkTypeT, fmt='%d', delimiter=',')

                np.savetxt(('%s/AgentPolicyPDTypeT-sim%04d.txt') % (dirName, sim), AgentPolicyPDTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountPolicyPDTypeT-sim%04d.txt') % (dirName, sim), CountPolicyPDTypeT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountLink_oAT-sim%04d.txt') % (dirName, sim), CountLink_oAT, fmt='%d', delimiter=',')
                np.savetxt(('%s/CountBreak_oAT-sim%04d.txt') % (dirName, sim), CountBreak_oAT, fmt='%d', delimiter=',')

                np.savetxt(('%s/CountCD_oAT-sim%04d.txt') % (dirName, sim), CountCD_oAT, fmt='%d', delimiter=',')
                
                np.savetxt(('%s/CountPDT-sim%04d.txt') % (dirName, sim), CountPDT, fmt='%d', delimiter=',')

                print('sim %d time %d' % (sim, t), CountPolicyLkTypeT[t-1], CountPolicyBkTypeT[t-1], CountPolicyPDTypeT[t-1], CountActualOutcomeT1[t-1], CountActualOutcomeT2[t-1], CountPDT[t-1])
        print('sim %d completed'%(sim))

def softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


if __name__ == '__main__':
    simStart = 1
    Nsim = 100

    tStart = 0
    T = 200000

    lr = 0.05
        
    algoName = 'Qpast1-b'

    Nact = 2
    gameName = 'PD2'

    Nagent = 100
    roundsG = 1500
    
    Prob0 = 0.5
    Prob1 = 0.1
    
    mainTrain(gameName, Nact, roundsG, algoName, lr, Nagent, simStart, Nsim, tStart, T, Prob0, Prob1)




