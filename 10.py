!pip install mchmm
import mchmm as mc
import pandas as pd
obs_seq = 'SWSCRWCSRSRSRSSWWWWCSWWCRW'
sts_seq = '00000000111111100000000000'
print(obs_seq)
print("\nS-Sunny\nW-Windy\nR-Rainy\nC-Cloudy\n")
print(sts_seq)
print("\n0-Sad\n1-Happy")
a = mc.HiddenMarkovModel().from_seq(obs_seq, sts_seq)
print("\nHidden States:\n",a.states)
print("\nObservations:\n",a.observations)
print("\nTransition Probability Matrix:\n",a.tp)
print("\nEmission Probability Matrix:\n")
pd.DataFrame(a.ep, index=a.states, columns=a.observations)
new_obs = "WRRWWWCRSWSSWSRW"
vs, vsi = a.viterbi(new_obs)
print("Predicted States: ", "".join(vs))