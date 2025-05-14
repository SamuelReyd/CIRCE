import numpy as np, matplotlib.pyplot as plt, re, time
from tqdm import tqdm

# Utils
def ouside_T_base_dynamic(t, T=24, T1=5, T2=20):
  y_0 = (T1+T2) / 2
  A = abs(T1-T2) / 2
  return y_0 + A * np.sin(2 * np.pi * t / T + T)
  
def get_room_neighbours(i,j, flat_length):
  nei_ids = []
  for dx in (-1,0,1):
    for dy in (-1,0,1):
      if 0 <= i + dx < flat_length and 0 <= j + dy < flat_length and not (dx == 0 and dy == 0):
        nei_ids.append((i+dx, j+dy))

  return nei_ids
  
# Rules
def R_IT1(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_window_state"]:
    ns[f"room_{i}_{j}_true_inside_T"] = np.random.normal(s["outside_T"], hp["noises"]["window"])
    return True
  return False

def R_IT2(s, ns, i, j, hp):
  if not s[f"room_{i}_{j}_window_state"] and s[f"room_{i}_{j}_heater_state"]:
    ns[f"room_{i}_{j}_true_inside_T"] = np.random.normal(hp["heater_goal_T"], hp["noises"]["heater"])
    return True
  return False

def R_IT3(s, ns, i, j, hp):
  if not s[f"room_{i}_{j}_window_state"] and not s[f"room_{i}_{j}_heater_state"]:
    neighbours_temperatures = [s[f"room_{i}_{j}_true_inside_T"]]
    for id in get_room_neighbours(i,j, hp["flat_length"]):
      neighbours_temperatures.append(s[f"room_{id[0]}_{id[1]}_true_inside_T"])
    ns[f"room_{i}_{j}_true_inside_T"] = np.random.normal(np.mean(neighbours_temperatures), hp["noises"]["th"])
    return True
  return False

def R_ITstar(s, ns, i, j, hp):
  th_pos = np.array((s[f"room_{i}_{j}_th_x"], s[f"room_{i}_{j}_th_y"]))
  ligh_pos = np.array((s[f"room_{i}_{j}_light_x"], s[f"room_{i}_{j}_light_y"]))
  if np.linalg.norm(th_pos - ligh_pos) < hp["light_range"] and s[f"room_{i}_{j}_light_state"] == 1:
    ns[f"room_{i}_{j}_measured_inside_T"] = hp["light_close_T"]
    return True
  if ns: ns[f"room_{i}_{j}_measured_inside_T"] = np.random.normal(ns[f"room_{i}_{j}_true_inside_T"], hp["noises"]["measure"])
  return False

def R_HS1(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_measured_inside_T"] < hp["heater_goal_T"]:
    # print("R_HS1", end=" ")
    ns[f"room_{i}_{j}_heater_state"] = 1
    return True
  return False

def R_HS2(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_measured_inside_T"] >= hp["heater_goal_T"]:
    # print("R_HS2", end=" ")
    ns[f"room_{i}_{j}_heater_state"] = 0
    return True
  return False

def R_LS1(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_nb_users_in_room"] > 0:
    ns[f"room_{i}_{j}_light_state"] = 1
    return True
  return False

def R_LS2(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_nb_users_in_room"] <= 0:
    ns[f"room_{i}_{j}_light_state"] = 0
    return True
  return False

def R_WS1(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_nb_users_in_room"] > 0 and s[f"room_{i}_{j}_true_inside_T"] < hp["user_min_T"]:
    ns[f"room_{i}_{j}_window_state"] = 0
    return True
  return False

def R_WS2(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_nb_users_in_room"] > 0 and s[f"room_{i}_{j}_true_inside_T"] > hp["user_max_T"]:
    ns[f"room_{i}_{j}_window_state"] = 1
    return True
  return False

def R_WSstar(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_nb_users_in_room"] <= 0 or hp["user_min_T"] <= s[f"room_{i}_{j}_true_inside_T"] <= hp["user_max_T"]:
    ns[f"room_{i}_{j}_window_state"] = s[f"room_{i}_{j}_window_state"]
    return True
  return False

def R_HS3(s, ns, i, j, hp):
  if s[f"room_{i}_{j}_window_state"]:
    # print("R_HS3", end=" ")
    ns[f"room_{i}_{j}_heater_state"] = 0
    return True
  return False

def R_HS4(s, ns, i, j, hp):
  neighbours_heater_states = []
  for id in get_room_neighbours(i,j, hp["flat_length"]):
    neighbours_heater_states.append(s[f"room_{id[0]}_{id[1]}_heater_state"])
  if all(neighbours_heater_states):
    # print("R_HS4", end=" ")
    ns[f"room_{i}_{j}_heater_state"] = 0
    return True
  return False

def R_WS3(s, ns, i, j, hp):
  neighbours_window_states = []
  for id in get_room_neighbours(i,j, hp["flat_length"]):
    neighbours_window_states.append(s[f"room_{id[0]}_{id[1]}_window_state"])
  if all(neighbours_window_states):
    ns[f"room_{i}_{j}_window_state"] = 0
    return True
  return False

def R_HS5(s, ns, i, j, hp):
  neighbours_nei_users = [s[f"room_{i}_{j}_nb_users_in_room"]]
  for id in get_room_neighbours(i,j, hp["flat_length"]):
    neighbours_nei_users.append(s[f"room_{id[0]}_{id[1]}_nb_users_in_room"])
  if sum(neighbours_nei_users) == 0:
    ns[f"room_{i}_{j}_heater_state"] = 0
    return True
  return False
  
RULES = {"R_IT1": R_IT1, "R_IT2": R_IT2, "R_IT3": R_IT3, "R_ITstar": R_ITstar, "R_HS1": R_HS1, "R_HS2":R_HS2, "R_LS1": R_LS1,
         "R_LS2":R_LS2, "R_WS1": R_WS1, "R_WS2":R_WS2, "R_WSstar": R_WSstar, "R_HS3":R_HS3, "R_HS4":R_HS4, "R_WS3":R_WS3, "R_HS5":R_HS5}

all_rules = ["R_IT1", "R_IT2", "R_IT3", "R_ITstar", "R_HS1", "R_HS2", "R_LS1", "R_LS2", "R_WS1", "R_WS2", "R_WSstar", "R_HS3", "R_HS4", "R_WS3", "R_HS5"]
base_rules = all_rules[:11]

# User interactions
def simulate_user_window(s, ns, i, j, hp):
  if np.random.rand() < hp["opening_window_proba"]:
    ns[f"room_{i}_{j}_window_state"] = 1 - s[f"room_{i}_{j}_window_state"]
    return True
  return False

def simulate_user_movement(s, ns, i, j, hp):
  user_moved = False
  nei_ids = get_room_neighbours(i,j, hp["flat_length"])
  n_user_key = "room_{}_{}_nb_users_in_room"

  if n_user_key.format(i, j) not in ns:
    ns[n_user_key.format(i, j)] = s[n_user_key.format(i, j)]

  for u in range(int(s[n_user_key.format(i, j)])):
    if np.random.rand() < hp["moving_user_proba"]:
      ns[n_user_key.format(i, j)] -= 1
      next_user_id_id = np.random.randint(len(nei_ids))
      ni, nj = nei_ids[next_user_id_id]
      if n_user_key.format(ni, nj) in ns:
        ns[n_user_key.format(ni, nj)] += 1
      else:
        ns[n_user_key.format(ni, nj)] = s[n_user_key.format(ni, nj)] + 1
      user_moved = True
  return user_moved

def simulate_user_object(s, ns, i, j, hp, obj):
  if np.random.rand() < hp["moving_object_proba"]:
    ns[f"room_{i}_{j}_{obj}_x"], ns[f"room_{i}_{j}_{obj}_y"] = np.random.uniform(0,hp["room_length"], size=2)
    return True
  else:
    ns[f"room_{i}_{j}_{obj}_x"], ns[f"room_{i}_{j}_{obj}_y"] = s[f"room_{i}_{j}_{obj}_x"], s[f"room_{i}_{j}_{obj}_y"]
    return False
    
# Infer to/from measure
def infer_true_inside_T(state, dim_labels):
  if any(["true" in label for label in dim_labels]):
    return state, dim_labels
  infered_state = []
  infered_dim_labels = []
  for value, dim in zip(state,dim_labels):
    if "inside_T" in dim:
      infered_state.append(value)
      infered_dim_labels.append(dim.replace("measured", "true"))
    infered_state.append(value)
    infered_dim_labels.append(dim)
  return np.array(infered_state), np.array(infered_dim_labels)

def extract_measures(state, dim_labels):
  return [state[i] for i,dim in enumerate(dim_labels) if "true" not in dim]
  
# Main functions
def transition_multi_house(state, dim_labels, hp, verbose=False):
  """
  state:
    t,
    house_k_room_i_j_[true_inside_T, measured_inside_T, thermo_x, thermo_y, heater_state, light_state, light_x, light_y, window_state, nb_user_in_room],
    outside_T
  hyper_parameters:
    flat_length, noises, min_outside_T, max_outside_T, heater_goal_T, light_close_T, user_min_T, user_max_T, room_length, light_range,
    ouside_T_base_dynamic, moving_object_proba, moving_user_proba, opening_window_proba
  """

  fs = {dim_label: dim_value for dim_label, dim_value in zip(dim_labels, state)}
  fns = {}

  # Time
  fns["t"] = fs["t"] + 1

  # House variables
  for k in range(hp["nb_houses"]):
    house_key = f"house_{k}_"
    s = {key[len(house_key):]: value for key, value in fs.items() if house_key in key} | {"outside_T": fs["outside_T"]}
    ns = {}
    for i in range(hp["flat_length"]):
      for j in range(hp["flat_length"]):
        for r in range(hp["rules"]):
          RULES[all_rules[r]](s, ns, i, j, hp)
        simulate_user_window(s, ns, i, j, hp)
        simulate_user_movement(s, ns, i, j, hp)
        simulate_user_object(s, ns, i, j, hp, "light")
        simulate_user_object(s, ns, i, j, hp, "th")
    fns |= {house_key + key:value for key,value in ns.items()}

  # Outside temperature
  fns["outside_T"] = np.random.normal(ouside_T_base_dynamic(fs["t"]),hp["noises"]["th"])
  return [fns[dim_label] for dim_label in dim_labels]
  
def get_dimension_info(hp):
  dim_labels = ["t"]
  discrete_dims = [False]

  room_dim_labels = ["true_inside_T", "measured_inside_T", "th_x", "th_y", "heater_state", "light_state", "light_x", "light_y", "window_state", "nb_users_in_room"]
  room_discrete_dims = [False, False, False, False, True, True, False, False, True, True]
  for k in range(hp["nb_houses"]):
    for j in range(hp["flat_length"]):
      for i in range(hp["flat_length"]):
        dim_labels += [f"house_{k}_room_{i}_{j}_"+dim_label for dim_label in room_dim_labels]
        discrete_dims += room_discrete_dims

  dim_labels.append("outside_T")
  discrete_dims += [False]

  return dim_labels, discrete_dims
  
def get_initial_state(hp, dim_labels):
  user_initial_rooms = np.random.randint(hp["flat_length"], size=(hp["nb_users"],2))
  s = {"t": -1}
  for k in range(hp["nb_houses"]):
    for j in range(hp["flat_length"]):
      for i in range(hp["flat_length"]):
        room_state = {
            "true_inside_T": hp["heater_goal_T"],
            "measured_inside_T": hp["heater_goal_T"],
            "nb_users_in_room": sum([(user_pos==(i,j)).all() for user_pos in user_initial_rooms])
        }
        room_state |= {"_".join([a,b]): np.random.uniform(0,hp["room_length"]) for a in ("light", "th") for b in ("x", "y")}
        room_state |= {key: 0 for key in ("window_state", "light_state", "heater_state", )}
        s |= {f"house_{k}_room_{i}_{j}_"+key:value for key, value in room_state.items()}

  s["outside_T"] = hp["min_outside_T"]
  initial_state = [s[key] for key in dim_labels]
  return initial_state
