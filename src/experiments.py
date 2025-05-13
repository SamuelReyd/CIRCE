from simulation import *
from CIRCE import *
import numpy as np, matplotlib.pyplot as plt, re, time, joblib, json, os
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, plot_tree

# General
def generate_observations(transition, hp, get_dimension_info, get_initial_parameters, seed=42, verbose=0):
  # Generate measures
  np.random.seed(42)

  dim_labels, discrete_dims = get_dimension_info(hp)
  state = get_initial_parameters(hp, dim_labels)

  states = []
  iterator = tqdm(range(hp["n_observation"])) if verbose else range(hp["n_observation"])
  for _ in iterator:
    # print("t=",_, end=" ")
    state = transition(state, dim_labels, hp)
    # print()
    states.append(state)
  return [extract_measures(state, dim_labels) for state in states]
  
def generate_contexts(M, start_t, end_t, rules, measure_labels, dim_labels, hp, verbose=1):
  contexts = []
  if verbose: iterator =tqdm(range(start_t+len(rules),end_t))
  else: iterator = range(start_t+len(rules),end_t)
  for t in iterator:
    all_states = [infer_true_inside_T(state, measure_labels)[0] for state in M[t-len(rules)+1:t+1]]
    all_fs = [dict(zip(dim_labels, state)) for state in all_states]
    # all_fs = [{dim_label: dim_value for dim_label, dim_value in zip(dim_labels, state)} for dt, state in enumerate(all_states)]
    for k in range(hp['nb_houses']):
      house_key = f"house_{k}_"
      all_s = [{key[len(house_key):]: value for key, value in fs.items() if house_key in key} | {"outside_T": fs["outside_T"]} for fs in all_fs]
      for i in range(hp['flat_length']):
        for j in range(hp['flat_length']):
          # print(f"house {k}, room {i},{j}->{rules[0](all_s[0], {}, i, j, hp)}")
          if all([rule(s, {}, i, j, hp) for s,rule in zip(all_s, rules)]):
            contexts.append((k, i, j, t))
  return contexts
  
def get_state_for_predicate(s, measure_labels, dim_labels):
  ref_label = re.sub("house_([0-9]+)_", "", dim_labels[1])
  state = [s[ref_label]]
  for label in measure_labels[1:]:
    label = re.sub("house_([0-9]+)_", "", label)
    if label == ref_label:
      return state
    state.append(s[label])
  return state
  
def R_test_Q1(s, ns, i, j, hp):
  return s[f"room_{i}_{j}_measured_inside_T"] < 15 and R_HS1(s, ns, i, j, hp)

def R_test_Q2(s, ns, i, j, hp):
  return R_IT3(s, ns, i, j, hp) and ns[f"room_{i}_{j}_true_inside_T"] < 15

def R_test_Q31(s, ns, i, j, hp):
  return s[f"room_{i}_{j}_window_state"] and R_WS1(s, ns, i, j, hp)

def R_test_Q32(s, ns, i, j, hp):
  return not s[f"room_{i}_{j}_window_state"] and s[f"room_{i}_{j}_nb_users_in_room"] >= 1 and not R_WS2(s, ns, i, j, hp)

def R_predicate(s, ns, i, j, hp, P, measure_labels, dim_labels):
  state = get_state_for_predicate(s, measure_labels, dim_labels)
  return evaluate_predicate(state, P, measure_labels)
  
def generate_quantitative_experiment_timesteps(M, measure_labels, dim_labels, targets, hp, seed=42):
  np.random.seed(seed)
  R = {
      "Q1":[R_IT1, R_test_Q1],
      "Q2":[R_test_Q2],
      "Q3":[R_test_Q31, R_test_Q32],
      "Q4":[R_ITstar],
      }
  timesteps = {}
  for q, r in R.items():
    dim, comp, th = targets[q]
    r += [lambda s, ns, i, j, hp: R_predicate(s, ns, i, j, hp, [(dim.format(k=0,i=i,j=j), comp, th)], measure_labels, dim_labels)]
    contexts = generate_contexts(M, 10, 70, r, measure_labels, dim_labels, hp)
    contexts = [c[1:] for c in contexts if c[0]==0]
    timesteps[q] = contexts[np.random.randint(len(contexts))]

  return timesteps
  
def match_clauses(c1, c2):
  return c1[0] == c2[0] and c1[1] == c2[1] and abs(c1[2] - c2[2])/abs(c1[2]) < .35

def get_predicate_set(P):
  predicate_set = []
  for c in P:
    for other in predicate_set:
      if match_clauses(c, other):
        break
    else:
      predicate_set.append(c)
  return predicate_set

def evaluate_generated_predicate(true, pred):
  nb_success = 0
  for t in true:
    for p in pred:
      if match_clauses(t, p):
          nb_success += 1
  if not len(pred): r = 0
  else: r = nb_success / len(pred)
  if not len(true): p = 0
  else: p = nb_success / len(true)
  if p == 0  and r == 0: f1 = 0
  else: f1 = 2*p*r/(p+r)
  return r, p, f1
  
def get_expected_pred(q, it, k,i,j,hp):
  if q == "Q1" and it == 0:
    return [(f"house_{k}_room_{i}_{j}_measured_inside_T", "<=", hp["heater_goal_T"])]
  if q == "Q1" and it == 1:
    return [(f"house_{k}_room_{i}_{j}_window_state", ">", 0.50), ("outside_T", "<=", hp["heater_goal_T"]-1)]
  if q == "Q3" and it == 0:
    return [(f"house_{k}_room_{i}_{j}_window_state", "<=", 0.50), (f"house_{k}_room_{i}_{j}_measured_inside_T", "<=", hp["user_max_T"])]
  if q == "Q3" and it == 1:
    return [(f"house_{k}_room_{i}_{j}_nb_users_in_room", ">", 0.50), (f"house_{k}_room_{i}_{j}_measured_inside_T", "<=", hp["user_min_T"])]
  raise Exception(f"Q={q} or it={it} not recognized, acceptable values are Q=('Q1','Q2') and it=(0,1)")
  
