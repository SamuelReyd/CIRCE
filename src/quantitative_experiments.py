"""
This file implements functions for the quantitative experiments. It creates subfolders in the ‘experiments` folder.
When using `python src/quantitative_experiments.py`, make sure to be at the root of the project.
"""

from simulation import *
from CIRCE import *
from rendering import *
from experiments import *
import numpy as np, matplotlib.pyplot as plt, re, time, joblib, json, os
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.tree import DecisionTreeClassifier, plot_tree

# General
def get_paths(family, param, value):
  # return paths for (simulation, context, dataset, trees, outputs, performance)
  paths = (
        "experiments/base/simulation.npy",
        "experiments/base/contexts__{q}.npy",
        "experiments/base/datasets__{q}.npy",
        "experiments/base/trees__{q}.joblib",
        "experiments/base/outputs.json",
        "experiments/base/performance.json"
    )
  if family == None:
    return paths
  if family == "simulation":
    return (
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/simulation.npy",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/contexts__{'{q}'}.npy",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/datasets__{'{q}'}.npy",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/trees__{'{q}'}.joblib",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/outputs.json",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/performance.json"
    )
  if family == "dataset":
    return (
        paths[0],
        paths[1],
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/datasets__{'{q}'}.npy",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/trees__{'{q}'}.joblib",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/outputs.json",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/performance.json"
    )
  if family == "DTC":
    return (
        paths[0],
        paths[1],
        paths[2],
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/trees__{'{q}'}.joblib",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/outputs.json",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/performance.json"
    )
  if family == "output":
    return (
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/outputs.json",
        f"experiments/{family}/{param}/{str(value).replace('.','_')}/performance.json"
    )
  raise Exception(f"Family {family} not recognized, acceptable values are (None, 'simulation', 'dataset', 'DTC', 'output')")
  
def load_json(path):
  folders = path.split("/")
  for i in range(1,len(folders)):
    if not os.path.isdir("/".join(folders[:i])):
      os.mkdir("/".join(folders[:i]))
  if not os.path.isfile(path):
    with open(path, 'w') as file:
      file.write('{}')
    return {}
  with open(path) as file:
    return json.load(file)
    
def update_json(path, D):
  folders = path.split("/")
  for i in range(1,len(folders)):
    if not os.path.isdir("/".join(folders[:i])):
      os.mkdir("/".join(folders[:i]))
  if os.path.isfile(path):
    with open(path) as file:
      base = json.load(file)
  else:
    base = {}
  with open(path, 'w') as file:
    file.write(json.dumps(base | D, indent=2))
    
def get_hp(family, param, value):
  hp = load_json("experiments/hyperparameters.json")

  if family is None: return hp
  if family == "DTC" and param in hp["DTC"]:
    hp[family][param] = value
    return hp
  hp[param] = value
  return hp
  
def base_q(Q):
  return Q.split("_")[0]
  
questions = (
    ("Q1_1", ("house_{k}_room_{i}_{j}_heater_state", ">", .5)),
    ("Q1_2", ("house_{k}_room_{i}_{j}_measured_inside_T", "<=", 19.)),
    ("Q3_1", ("house_{k}_room_{i}_{j}_window_state", "<=", .5)),
    ("Q3_2", ("house_{k}_room_{i}_{j}_window_state", "<=", .5))
)

def show_different_trees(Q, question, modifier1, modifier2, i=3):
  base_hp = load_json("experiments/hyperparameters.json")
  dim_labels, discrete_dims = get_dimension_info(base_hp)
  measure_labels = [label for label in dim_labels if "true" not in label]

  _, context_path_a, _, trees_path_a, _, _ = get_paths(*modifier1)
  _, context_path_b, _, trees_path_b, _, _ = get_paths(*modifier2)

  trees_ = joblib.load(f"experiments/base/trees__{Q}.joblib")
  trees_a = joblib.load(trees_path_a.format(q=Q))
  trees_b = joblib.load(trees_path_b.format(q=Q))
  contexts = np.load(f"experiments/base/contexts__{base_q(Q)}.npy")

  tree_ = trees_[i]
  tree_a = trees_a[i]
  tree_b = trees_b[i]
  k,i,j,t0 = contexts[i]
  dim, comp, th = question
  T = [(dim.format(k=k,i=i,j=j), comp, th)]

  print("Base Tree")
  plot_surrogate(tree_, measure_labels, T, save=True, name="images/Base Tree.pdf", show=True)
  print(f"Tree for {modifier1[1]}={modifier1[2]}")
  plot_surrogate(tree_a, measure_labels, T, save=True, name=f"images/Tree for {modifier1[1]}={str(modifier1[2]).replace('.', '_')}.pdf", show=True)
  print(f"Tree for {modifier2[1]}={modifier2[2]}")
  plot_surrogate(tree_b, measure_labels, T, save=True, name=f"images/Tree for {modifier2[1]}={str(modifier2[2]).replace('.', '_')}.pdf", show=True)
  
def generate_experiment(family, param, value, sequential=False, verbose=0):
  if verbose: print("Make simulation")
  make_simulation((family, param, value), verbose=verbose)
  if verbose: print("Make contexts")
  make_contexts("Q1", ("house_{k}_room_{i}_{j}_heater_state", ">", .5), [R_IT1, R_test_Q1], (family, param, value), verbose=verbose)
  make_contexts("Q3", ("house_{k}_room_{i}_{j}_window_state", "<=", .5), [R_test_Q31, R_test_Q32], (family, param, value), verbose=verbose)

  if verbose: print("Make datasets")
  make_datasets("Q1_1", ("house_{k}_room_{i}_{j}_heater_state", ">", .5), (family, param, value), sequential=sequential, verbose=verbose)
  make_datasets("Q1_2", ("house_{k}_room_{i}_{j}_measured_inside_T", "<=", 20.), (family, param, value), sequential=sequential, verbose=verbose)
  make_datasets("Q3_1", ("house_{k}_room_{i}_{j}_window_state", "<=", .5), (family, param, value), sequential=sequential, verbose=verbose)
  make_datasets("Q3_2", ("house_{k}_room_{i}_{j}_window_state", "<=", .5), (family, param, value), sequential=sequential, verbose=verbose)

  if verbose: print("Make trees")
  make_tree("Q1_1", (family, param, value), verbose=verbose)
  make_tree("Q1_2", (family, param, value), verbose=verbose)
  make_tree("Q3_1", (family, param, value), verbose=verbose)
  make_tree("Q3_2", (family, param, value), verbose=verbose)

  if verbose: print("Make outputs")
  make_outputs("Q1_1", (family, param, value))
  make_outputs("Q1_2", (family, param, value))
  make_outputs("Q3_1", (family, param, value))
  make_outputs("Q3_2", (family, param, value))

  if verbose: print("Make evaluation")
  make_evaluation((family, param, value))
  
def init_dict(D, keys):
  for key in keys:
    D["mean_" + key] = []
    D["std_" + key] = []

def update_dict_one_value(D1, D2):
  for key, value in D2.items():
    if f"mean_"+ key in D2:
      del D1[f"mean_"+ key]
      del D1[f"std_" + key]
      continue
    D1[f"mean_"+ key].append(np.mean(value))
    D1[f"std_" + key].append(np.std(value))

def update_dict_all_values(D1, D2):
  ids = np.argsort(D2["values"])
  for i in ids:
    for key, value in D2.items():
      D1[key].append(value[i])


def gather_performances():
  test_params = load_json("experiments/test_parameters_values.json")
  all_performances = {}
  base_perfs = load_json("experiments/base/performance.json")
  base_hp = load_json("experiments/hyperparameters.json")
  for family, param_values in test_params.items():
    for param, values in param_values.items():
      # Init shuffled values
      D = {"values": []}
      init_dict(D, base_perfs.keys())
      # Fill shuffled values
      for value in values:
        path = f"experiments/{family}/{param}/{str(value).replace('.','_')}/performance.json"
        if os.path.isfile(path):
          performance = load_json(path)
          D["values"].append(value if value is not None else -1)
          update_dict_one_value(D, performance)
      # Fill base values
      if family == "DTC" and param in base_hp["DTC"]: value = base_hp["DTC"][param]
      else: value = base_hp[param]
      D["values"].append(value if value is not None else -1)
      update_dict_one_value(D, base_perfs)

      # Init sorted values
      all_performances[param] = {"values": []}
      init_dict(all_performances[param], performance.keys())
      # Fill sorted values
      D = {key: value for key, value in D.items() if key in all_performances[param]}
      update_dict_all_values(all_performances[param], D)

  return {key: value for key, value in all_performances.items() if len(value['values'])>1}


def plot_performances(all_performances, save=True):
  n_plot = len(all_performances)
  n_col = int(np.ceil(n_plot/3))
  _,axes = plt.subplots(3, n_col, figsize=(n_col*5,9))
  for ax, (param, perfs) in zip(axes.flatten(),all_performances.items()):
    x = np.arange(len(perfs['values']))
    ax.set_title(param)
    ax.bar(x-.3, perfs["mean_f1__Q1_1"], width=.2, yerr=perfs["std_f1__Q1_1"])
    ax.bar(x-.1, perfs["mean_f1__Q1_2"], width=.2, yerr=perfs["std_f1__Q1_2"])
    ax.bar(x+.1, perfs["mean_f1__Q3_1"], width=.2, yerr=perfs["std_f1__Q3_1"])
    ax.bar(x+.3, perfs["mean_f1__Q3_2"], width=.2, yerr=perfs["std_f1__Q3_2"])
    ax.set_xticks(x, [str(v) for v in perfs["values"]])
    ax.tick_params('x', labelrotation=45)
    ax.grid(axis="y")
  for ax in axes.flatten()[n_plot:]: ax.set_visible(False)
  plt.tight_layout()
  if save: plt.savefig("perfomance.pdf")
  plt.show()
  
  
# Makers
def make_simulation(modifier, verbose=1):
  seed = 42
  sim_path, context_path, dataset_path, trees_path, outputs_path, perf_path = get_paths(*modifier)
  if os.path.exists(sim_path): return
  
  hp = get_hp(*modifier)
  #print(get_hp.__globals__.__name__)
  measures = generate_observations(
      transition_multi_house,
      hp,
      get_dimension_info,
      get_initial_state,
      seed=seed,
      verbose=verbose)

  M = np.array(measures)

  np.save(sim_path, M)
  
def make_contexts(q, question, rules, modifier, verbose=1):
  seed=42
  np.random.seed(seed)
  sim_path, context_path, dataset_path, trees_path, outputs_path, perf_path = get_paths(*modifier)
  if os.path.exists(context_path.format(q=q)): return
  hp = get_hp(*modifier)

  S = np.load(sim_path)

  dim_labels, discrete_dims = get_dimension_info(hp)
  measure_labels = [label for label in dim_labels if "true" not in label]

  dim, comp, th = question
  rules += [lambda s, ns, i, j, hp: R_predicate(s, ns, i, j, hp, [(dim.format(k=0,i=i,j=j), comp, th)], measure_labels, dim_labels)]
  context = generate_contexts(S, 10, S.shape[0] - 10, rules, measure_labels, dim_labels, hp, verbose=verbose)
  np.random.shuffle(context)
  np.save(context_path.format(q=q), context)
  
def make_dataset(q, question, contexts, S, measure_labels, hp, seed=42, verbose=1):
  datasets = []
  times = []
  n_queries = []
  if "_" in q:
    _, it = q.split("_")
  else:
    it = 1
  if verbose: iterator = tqdm(contexts[:hp["n_run_evaluation"]])
  else: iterator = contexts[:hp["n_run_evaluation"]]
  for k,i,j,t0 in iterator:
    t_init = time.perf_counter()
    ps = S[t0-int(it)]
    sample_interventions = intervention_samplings[hp["balance"]]
    dim, comp, th = question
    T = [(dim.format(k=k,i=i,j=j), comp, th)]
    X_train, y_train, n_query = sample_interventions(S, ps, transition_multi_house, T, measure_labels, hp, seed=seed)
    t_sample = time.perf_counter() - t_init
    datasets.append(np.concatenate([X_train,y_train.reshape(-1,1)], axis=1))
    times.append(t_sample)
    n_queries.append(n_query)
  return np.stack(datasets), times, n_queries
 
def make_parallel_dataset(q, question, contexts, S, measure_labels, hp, seed=42, verbose=1):
  if "_" in q:
    _, it = q.split("_")
  else:
    it = 1
  inputs = [(S, question, measure_labels, k,i,j,t0,it, hp, seed) for k,i,j,t0 in contexts[:hp["n_run_evaluation"]]]
  with Pool(20) as p:
    if verbose: outputs = list(tqdm(p.imap(make_parallel_dataset_function, inputs), total=len(inputs)))
    else: outputs = list(p.imap(make_parallel_dataset_function, inputs))
  datasets = []
  times = []
  n_queries = []
  for X_train, y_train, n_query, t_sample in outputs:
    datasets.append(np.concatenate([X_train,y_train.reshape(-1,1)], axis=1))
    times.append(t_sample)
    n_queries.append(n_query)
  return np.stack(datasets), times, n_queries

def make_parallel_dataset_function(inputs):
    S, question, measure_labels, k,i,j,t0,it, hp, seed = inputs
    t_init = time.perf_counter()
    ps = S[t0-int(it)]
    sample_interventions = intervention_samplings[hp["balance"]]
    dim, comp, th = question
    T = [(dim.format(k=k,i=i,j=j), comp, th)]
    X_train, y_train, n_query = sample_interventions(S, ps, transition_multi_house, T, measure_labels, hp, seed=seed)
    t_sample = time.perf_counter() - t_init
    return X_train, y_train, n_query, t_sample

def make_datasets(q, question, modifier, sequential=False, verbose=1):
  seed = 42

  sim_path, context_path, dataset_path, trees_path, outputs_path, perf_path = get_paths(*modifier)
  if os.path.exists(dataset_path.format(q=q)): return
  hp = get_hp(*modifier)

  S = np.load(sim_path)

  dim_labels, discrete_dims = get_dimension_info(hp)
  measure_labels = [label for label in dim_labels if "true" not in label]

  contexts = np.load(context_path.format(q=base_q(q)))
  if sequential:
    datasets, times, n_queries = make_dataset(q, question, contexts, S, measure_labels, hp, seed=seed, verbose=verbose)
  else:
    datasets, times, n_queries = make_parallel_dataset(q, question, contexts, S, measure_labels, hp, seed=seed, verbose=verbose)

  np.save(dataset_path.format(q=q), datasets)

  update_json(perf_path, {f"dataset_time__{q}": times, f"dataset_n_query__{q}":n_queries})
  
def make_tree(q, modifier, verbose=1):
  seed = 42

  sim_path, context_path, dataset_path, trees_path, outputs_path, perf_path = get_paths(*modifier)
  if os.path.exists(trees_path.format(q=q)): return
  hp = get_hp(*modifier)

  S = np.load(sim_path)
  dim_labels, discrete_dims = get_dimension_info(hp)
  measure_labels = [label for label in dim_labels if "true" not in label]

  contexts = np.load(context_path.format(q=base_q(q)))
  datasets = np.load(dataset_path.format(q=q))

  trees = []
  times = []
  if verbose: iterator = tqdm(list(zip(datasets,contexts[:,-1])))
  else: iterator = zip(datasets,contexts[:,-1])
  for dataset, t0 in iterator:
    t_init = time.perf_counter()
    tree = train_tree(dataset[:,:-1], dataset[:,-1], S[t0-1], hp, seed=seed)
    t_train = time.perf_counter() - t_init
    trees.append(tree)
    times.append(t_train)

  update_json(perf_path, {f"training__{q}": times})

  joblib.dump(trees, trees_path.format(q=q))
  
def make_outputs(q, modifier):
  seed = 42

  sim_path, context_path, dataset_path, trees_path, outputs_path, perf_path = get_paths(*modifier)
  hp = get_hp(*modifier)

  S = np.load(sim_path)
  dim_labels, discrete_dims = get_dimension_info(hp)
  measure_labels = [label for label in dim_labels if "true" not in label]

  contexts = np.load(context_path.format(q=base_q(q)))
  trees =  joblib.load(trees_path.format(q=q))

  outputs = {"C_BF":[],"C_SBF":[]}
  times = []
  for tree, t0 in zip(trees,contexts[:,-1]):
    t_init = time.perf_counter()
    C_BF, C_SBF = get_causes(tree, S[t0-1], measure_labels, hp)
    t_generate = time.perf_counter() - t_init
    outputs["C_BF"].append(C_BF)
    outputs["C_SBF"].append(C_SBF)
    times.append(t_generate)

  update_json(perf_path, {f"generating__{q}": times})
  update_json(outputs_path, {q : outputs})
  
def make_evaluation(modifier):

  sim_path, context_path, dataset_path, trees_path, outputs_path, perf_path = get_paths(*modifier)
  hp = get_hp(*modifier)

  outputs = load_json(outputs_path)
  for q, output in outputs.items():
    if "_" in q:
      Q, it = q.split("_")
      it = int(it) - 1
    else: Q, it = q, 0
    contexts = np.load(context_path.format(q=base_q(q)))
    precisions, recalls, f1s = [], [], []
    for (k,i,j,t0), C_SBF in zip(contexts, output["C_SBF"]):
      r,p,f1 = evaluate_generated_predicate(get_expected_pred(Q, it, k, i, j, hp), C_SBF)
      precisions.append(p)
      recalls.append(r)
      f1s.append(f1)
    update_json(perf_path, {f"precision__{q}": precisions, f"recall__{q}": recalls, f"f1__{q}": f1s})
  
def generate_all_sequential(small=False, full_sequential=False):
    if small:
      test_params = load_json("experiments/test_parameters_values_small.json")
    else:
      test_params = load_json("experiments/test_parameters_values.json")
    for family in ("output", "DTC", "dataset", "simulation"):
      for param, values in tqdm(test_params[family].items(), desc=f"{family:^50}"):
        for value in tqdm(test_params[family][param],leave=False, desc=f"{param:^50}"):
          os.makedirs(f"experiments/{family}/{param}/{str(value).replace('.','_')}", exist_ok=True)
          generate_experiment(family, param, value, sequential=full_sequential, verbose=0)


# Main
if __name__ == "__main__":
    generate_experiment(None, None, None, verbose=1)
    generate_all_sequential()
    all_performances = gather_performances()
    with open("all_performances.json","w") as file:
        file.write(json.dumps(all_performances, indent=2))
    plot_performances(all_performances)
