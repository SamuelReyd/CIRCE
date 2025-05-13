from simulation import *
import numpy as np, matplotlib.pyplot as plt, re, time, joblib, json
from tqdm.notebook import tqdm
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Utils
def balance_training_dataset(X, y, hp):
  pos_ids = np.where(y == 1)[0]
  neg_ids = np.where(y == 0)[0]
  if len(pos_ids) > len(neg_ids):
    pos_ids = np.random.choice(pos_ids, size=len(neg_ids), replace=False)
  else:
    neg_ids = np.random.choice(neg_ids, size=len(pos_ids), replace=False)
  all_ids = np.concatenate([pos_ids,neg_ids])
  np.random.shuffle(all_ids)
  return X[all_ids], y[all_ids]
  
def get_sibling_id(tree, parent_id, node_id):
  if tree.tree_.children_left[parent_id] == node_id:
    return tree.tree_.children_right[parent_id]
  else:
    return tree.tree_.children_left[parent_id]
    
def get_childen_ids(tree, node_id, path):
  child_left_id = tree.tree_.children_left[node_id]
  child_right_id = tree.tree_.children_right[node_id]
  if child_left_id in path:
    return child_left_id, child_right_id
  elif child_right_id in path:
    return child_right_id, child_left_id
  else:
    return None

def check_BF_end(tree, node_id, sibling_id, next_node_id, next_sibling_id, hp):
  return (
      tree.tree_.impurity[node_id] < hp["min_impurity_threshold"] and tree.classes[node_id] == 1 and
      tree.tree_.impurity[sibling_id] < hp["min_impurity_threshold"] and tree.classes[sibling_id] == 0
  )
  
def check_CBF_end(tree, node_id, sibling_id, next_node_id, next_sibling_id, hp):
  return (
      tree.tree_.impurity[node_id] < hp["min_impurity_threshold"] and tree.classes[node_id] == 1 and
      tree.tree_.impurity[next_node_id] < hp["min_impurity_threshold"] and tree.classes[next_node_id] == 1 and
      (tree.classes[next_sibling_id] == 1 or tree.tree_.impurity[next_sibling_id] > hp["max_impurity_threshold"])
  )
  
def show_clause(dimension, comparator, threshold):
  return " ".join([dimension, comparator, f"{threshold:.2f}"])

def evaluate_clause(s, dim_labels, dimension, comparator, threshold):
  if comparator == ">":
    return s[dim_labels.index(dimension)] > threshold
  elif comparator == "<=":
    return s[dim_labels.index(dimension)] <= threshold
  else:
    raise Exception("Comparator invalid")

def evaluate_predicate(s, P, dim_labels):
  return np.all([evaluate_clause(s, dim_labels, *clause) for clause in P])

def show_predicate(P):
  return " and ".join([show_clause(*clause) for clause in P])
  
def add_perturbation_with_balance(x, y, s, X_train, y_train, weight, pos_w, neg_w, half_w, hp):
  if y and pos_w <= half_w:
    X_train.append(x)
    y_train.append(y)
    pos_w += weight(x, s)
  elif (not y) and neg_w <= half_w:
    X_train.append(x)
    y_train.append(y)
    neg_w += weight(x, s)
  return X_train, y_train, pos_w, neg_w
  
def remove_perturbation_for_weight_balance(X_train, y_train, weights, hp):
  W = weights.sum()
  filter = np.ones_like(y_train)
  while weights[np.logical_and(y_train == 1,filter)].sum() / W > .50:
    del_id = np.random.choice(np.where(y_train == 1)[0])
    filter[del_id] = 0
  while weights[np.logical_and(y_train == 0,filter)].sum() / W > .50:
    del_id = np.random.choice(np.where(y_train == 0)[0])
    filter[del_id] = 0
  return X_train[~filter], y_train[~filter], weights[~filter]
  
def get_distr(n,k):
  return np.abs(-2/n*np.arange(n+1)+1)**k/sum(np.abs(-2/n*np.arange(n+1)+1)**k)
  
def perturbate(s, S, hp):
  while True:
    random_obs = S[np.random.randint(len(S))]
    # nb_perturb = np.random.randint(len(s))
    p = get_distr(s.size-1, hp["perturbation_distribution"])
    nb_perturb = np.random.choice(np.arange(s.size), p=p)
    perturb_dims = np.random.choice(np.arange(len(s)), size=nb_perturb, replace=False)
    perturbation = s.copy()

    perturbation[perturb_dims] = random_obs[perturb_dims]

    if np.max(np.abs(s - perturbation)) > hp["perturbation_min_eps"]:
      break
  return perturbation
  
def evaluate_next_state(f, x, dim_labels, T, hp):
  state, full_dim_labels = infer_true_inside_T(x, dim_labels)
  next_state = f(state, full_dim_labels, hp)
  return evaluate_predicate(extract_measures(next_state, full_dim_labels), T, dim_labels)
  
def imp(tree, node_id):
  return tree.tree_.impurity[node_id]

def prob(tree, node_id):
  return tree.tree_.value[node_id,0,1]/np.sum(tree.tree_.value[node_id])
  
# HP
def L2_distance(s, X): return np.linalg.norm(s - X, axis=1)

def L1_distance(s, X): return np.sum(np.abs(s-X), axis=1)

def L0_distance(s, X): return np.sum(np.abs(s-X) > 1e-5, axis=1)

def inverse_kernel(d): return 1/(d+1e-5)
def squared_inverse_kernel(d): return 1/(d**2+1e-5)
def inverse_sqrt_kernel(d): return 1/(np.sqrt(d)+1e-5)
def inverse_log_kernel(d): return 1/(np.log(d)+1e-5)
def inverse_exp_kernel(d): return np.exp(-d)
def constant_kernel(d): return np.ones_like(d)
def linear_kernel(d): return d

distances = {"L2": L2_distance, "L1": L1_distance, "L0": L0_distance}
kernels = {"inverse": inverse_kernel, "squared_inverse": squared_inverse_kernel,
           "inverse_sqrt":inverse_sqrt_kernel, "inverse_log": inverse_log_kernel,
           "inverse_exp":inverse_exp_kernel, "constant":constant_kernel,
           "linear":linear_kernel}
           
           
# Algorithm
def CIRCE(T, t0, S, f, hp, measure_labels, verbose=1, seed=42, save=False):
  np.random.seed(seed)
  t_init = time.perf_counter()
  ps = S[t0-1]
  sample_interventions = intervention_samplings[hp["balance"]]
  X_train, y_train, N = sample_interventions(S, ps, f, T, measure_labels, hp)
  t_sample = time.perf_counter() - t_init

  tree = train_tree(X_train, y_train, ps, hp)
  t_train = time.perf_counter() - t_init - t_sample

  C_BF, C_SBF = get_causes(tree, ps, measure_labels, hp, verbose=verbose)
  t_generate = time.perf_counter() - t_init - t_sample - t_train

  if verbose: plot_surrogate(tree, measure_labels, T, save=True)
  return tree, C_BF, C_SBF, (t_sample, t_train, t_generate, N)
  
def sample_interventions_no_balance(S, s, f, T, dim_labels, hp, seed=None):
  if seed is not None: np.random.seed(seed)
  X_train = []
  y_train = []
  N = 0
  for _ in range(hp["n_sample"]):
    x = perturbate(s, S, hp)
    X_train.append(x)
    y_train.append(evaluate_next_state(f, x, dim_labels, T, hp))
  return np.array(X_train), np.array(y_train), hp["n_sample"]

def sample_interventions_sample_balance(S, s, f, T, dim_labels, hp, seed=None):
  if seed is not None: np.random.seed(seed)
  X_train = []
  y_train = []
  pos_w = 0
  neg_w = 0
  N = 0
  while len(X_train) < hp["n_sample"]:
    x = perturbate(s, S, hp)
    y = evaluate_next_state(f, x, dim_labels, T, hp)
    X_train, y_train, pos_w, neg_w = add_perturbation_with_balance(x, y, s, X_train, y_train, lambda x, s: 1, pos_w, neg_w, hp["n_sample"]/2, hp)
    N += 1
  return np.array(X_train), np.array(y_train), N

def sample_interventions_weight_balance(S, s, f, T, dim_labels, hp, seed=None):
  if seed is not None: np.random.seed(seed)
  X_train = []
  y_train = []
  for _ in range(hp["n_sample"]):
    x = perturbate(s, S, hp)
    X_train.append(x)
    y_train.append(evaluate_next_state(f, x, dim_labels, T, hp))
  X_train, y_train = np.array(X_train), np.array(y_train)
  kernel = kernels[hp["kernel"]]
  distance = distances[hp["distance"]]
  weights = kernel(distance(s, X_train))
  X_train, y_train, weights = remove_perturbation_for_weight_balance(X_train, y_train, weights, hp)
  pos_w = weights[y_train == 1].sum()
  neg_w = weights[y_train == 0].sum()
  half_w = max(pos_w, neg_w)
  weight = lambda x, y: kernel(distance(s, x.reshape(1,-1))).item()
  X_train, y_train = X_train.tolist(), y_train.tolist()
  N = hp["n_sample"]
  while len(X_train) < hp["n_sample"]:
    x = perturbate(s, S, hp)
    y = evaluate_next_state(f, x, dim_labels, T, hp)
    X_train, y_train, pos_w, neg_w = add_perturbation_with_balance(x, y, s, X_train, y_train, weight, pos_w, neg_w, half_w, hp)
    N += 1
  return np.array(X_train), np.array(y_train), N

intervention_samplings = {
    "none": sample_interventions_no_balance,
    "sample": sample_interventions_sample_balance,
    "weight": sample_interventions_weight_balance,
}

def train_tree(X, y, s, hp, seed=None):
  if seed is not None: np.random.seed(seed)
  d = distances[hp["distance"]](s, X)
  similarities = kernels[hp["kernel"]](d)
  similarities = np.array(similarities)/np.sum(similarities)

  nan_filter = np.logical_or(np.isnan(X).any(axis=1), np.isnan(similarities))
  nan_filter = np.logical_or(nan_filter, np.isnan(y))
  X = X[~nan_filter]
  y = y[~nan_filter]
  similarities = similarities[~nan_filter]
  clf = DecisionTreeClassifier(random_state=0, **hp["DTC"])
  clf.fit(X, y, sample_weight=similarities)

  return clf
  
def get_causes(tree, s, measure_labels, hp, verbose=0):
  path = np.where(tree.tree_.decision_path(s.reshape(1,-1).astype('float32')).toarray().reshape(-1,1))[0]
  if len(path) == 1: return [], []
  first_comparator = "<=" if tree.tree_.children_left[0] == path[1] else ">"
  BF = [(measure_labels[tree.tree_.feature[0]], first_comparator, tree.tree_.threshold[0])]
  do_BF = True
  CBF = [(measure_labels[tree.tree_.feature[0]], first_comparator, tree.tree_.threshold[0])]
  do_CBF = True
  tree.classes = np.argmax(np.squeeze(tree.tree_.value), axis=1)

  for parent_id, node_id in zip(path[:-1], path[1:]):
    comparator = "<=" if s[tree.tree_.feature[node_id]] <= tree.tree_.threshold[node_id] else ">"
    sibling_id = get_sibling_id(tree, parent_id, node_id)
    children_ids = get_childen_ids(tree, node_id, path)
    if children_ids is None:
      return BF, CBF
    next_node_id, next_sibling_id = children_ids
    if verbose: print(f"node ({node_id},{measure_labels[tree.tree_.feature[parent_id]]} {comparator} {tree.tree_.threshold[parent_id]:.2f}): {imp(tree, node_id):.2f} / sibling: {imp(tree, sibling_id):.2f} / child ({next_node_id}): {imp(tree, next_node_id):.2f} / child sibling: {imp(tree, next_sibling_id):.2f}")
    if do_BF:
      if check_BF_end(tree, node_id, sibling_id, next_node_id, next_sibling_id, hp):
        do_BF = False
      else:
        BF.append((measure_labels[tree.tree_.feature[node_id]], comparator, tree.tree_.threshold[node_id]))
    if check_CBF_end(tree, node_id, sibling_id, next_node_id, next_sibling_id, hp):
      return BF, CBF
    CBF.append((measure_labels[tree.tree_.feature[node_id]], comparator, tree.tree_.threshold[node_id]))
    
    
def plot_surrogate(clf, dim_labels, T, save=False, name=None, show=False):
  pred_label = ' and '.join([show_clause(*clause) for clause in T])
  plot_tree(clf, feature_names=dim_labels, class_names=[f"not({pred_label})", pred_label], filled=True)
  if save: plt.savefig(name if name is not None else "surrogate.pdf")
  if show: plt.show()