"""
This file implements functions to render figures.
"""

from CIRCE import *
import numpy as np, matplotlib.pyplot as plt, re
from sklearn.tree import DecisionTreeClassifier, plot_tree

# This function renders the measures in 9 rooms in one house.
def plot_behavior(house_k, M, dim_labels, discrete_dims, hyper_parameters, timesteps={}, save=False, start_id=0,end_id=80):
  grid_step = 10
  _, axes = plt.subplots(4, 3, figsize=(22,17), sharex=True)
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  x = np.arange(start_id, end_id)
  for j in range(hyper_parameters["flat_length"]):
    for i in range(hyper_parameters["flat_length"]):
      ax1 = axes.flatten()[j + i * hyper_parameters["flat_length"]+3]

      ax2 = ax1.twinx()
      ax2.set_ylim(-1,12)
      ax2.set_yticks(range(-1,12),labels=["", "0", "1", "2", "3", "", "Off", "On", "Close", "Open", "Off", "On", ""], fontsize=16)
      lns = []
      lns += ax1.plot(x, M[start_id:end_id,dim_labels.index(f"house_{house_k}_room_{i}_{j}_measured_inside_T")], c=colors[0])
      lns += ax1.plot(x, M[start_id:end_id,dim_labels.index(f"outside_T")], c=colors[1])
      lns += ax2.step(x, 5 + M[start_id:end_id,dim_labels.index(f"house_{house_k}_room_{i}_{j}_heater_state")], c=colors[2])
      lns += ax2.step(x, 7 + M[start_id:end_id,dim_labels.index(f"house_{house_k}_room_{i}_{j}_window_state")], c=colors[3])
      lns += ax2.step(x, 9 + M[start_id:end_id,dim_labels.index(f"house_{house_k}_room_{i}_{j}_light_state")], c=colors[4])
      lns += ax2.step(x, M[start_id:end_id,dim_labels.index(f"house_{house_k}_room_{i}_{j}_nb_users_in_room")], c=colors[5])

      line_styles = iter(['-', '--', '-.', ':'])
      for q, (ii,jj,t) in timesteps.items():
        if (i,j) == (ii,jj):
          lns += [ax1.axvline(x=t, c="black", linestyle=next(line_styles))]
          ax2.text(t + 0.1, 10.5, q, fontsize=22)
      ax1.set_xticks(np.arange(start_id, end_id, grid_step))
      ax1.tick_params(axis='both', labelsize=16)
      ax1.grid(axis="x")

      ax1.set_title(f"Room ({i},{j})", fontsize=22)

  axes[0,0].clear()
  axes[0,0].axis('off')
  axes[0,2].clear()
  axes[0,2].axis('off')
  legend_ax = axes[0,1]
  legend_ax.axis('off')
  labs = ["inside_T", "outside_T", "heater_state", "window_state", "light_state", "nb_users_in_room"]
  for lab in labs:
    legend_ax.plot([], [], label=lab)
    legend_ax.legend(loc='center', ncol=2, fontsize=22)
  plt.tight_layout()
  if save: plt.savefig("System behavior over time.pdf", bbox_inches='tight')
  plt.show()

# This function renders one decision tree
def plot_surrogate(clf, dim_labels, T, save=False, name=None, show=False):
  pred_label = ' and '.join([show_clause(*clause) for clause in T])
  plot_tree(clf, feature_names=dim_labels, class_names=[f"not({pred_label})", pred_label], filled=True)
  if save: plt.savefig(name if name is not None else "surrogate.pdf")
  if show: plt.show()

# This function renders the measure for one room
def plot_single_room(M, house_k, i, j, start_id, end_id, dims, ddims, timesteps):
  grid_step = 10
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  x = np.arange(start_id, end_id)
  _, (ax1, legend_ax) = plt.subplots(1,2, figsize=(14,5))
  ax2 = ax1.twinx()
  ax2.set_ylim(-1,12)
  ax2.set_yticks(range(-1,12),labels=["", "0", "1", "2", "3", "", "Off", "On", "Close", "Open", "Off", "On", ""], fontsize=16)
  lns = []

  lns += ax1.plot(x, M[start_id:end_id,dims.index(f"house_{house_k}_room_{i}_{j}_measured_inside_T")], c=colors[0])
  lns += ax1.plot(x, M[start_id:end_id,dims.index(f"outside_T")], c=colors[1])

  lns += ax2.step(x, 5 + M[start_id:end_id,dims.index(f"house_{house_k}_room_{i}_{j}_heater_state")], c=colors[2])
  lns += ax2.step(x, 7 + M[start_id:end_id,dims.index(f"house_{house_k}_room_{i}_{j}_window_state")], c=colors[3])
  lns += ax2.step(x, 9 + M[start_id:end_id,dims.index(f"house_{house_k}_room_{i}_{j}_light_state")], c=colors[4])
  lns += ax2.step(x, M[start_id:end_id,dims.index(f"house_{house_k}_room_{i}_{j}_nb_users_in_room")], c=colors[5])

  line_styles = iter(['-', '--', '-.', ':'])
  for q, (ii,jj,t) in timesteps.items():
    if (i,j) == (ii,jj):
      lns += [ax1.axvline(x=t, c="black", linestyle=next(line_styles))]
      ax2.text(t + 0.1, 10.5, q, fontsize=20)
  ax1.set_xticks(np.arange(start_id, end_id, grid_step))
  ax1.tick_params(axis='both', labelsize=16)
  ax1.grid(axis="x")
  ax1.set_xlabel("Time", fontsize=15)
  ax1.set_ylabel("Temperature", fontsize=15)
  ax2.set_ylabel("Nb. users / States", fontsize=15)
  legend_ax.set_axis_off()

  labs = ["inside_T", "outside_T", "heater_state", "window_state", "light_state", "nb_users_in_room"]
  for lab in labs:
    legend_ax.plot([], [], label=lab)
  legend_ax.axis('off')
  legend_ax.legend(loc='center', ncol=1, fontsize=20)

  ax1.set_title(f"Room ({i},{j})", fontsize=22)
  plt.tight_layout()
  plt.savefig(f"room_{i}_{j}.pdf", bbox_inches='tight')
  plt.show()