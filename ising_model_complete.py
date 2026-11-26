import math
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from numba import njit
from scipy.signal import convolve2d
import sys

# Constants
pairc = 1
k = 1
critical_temperature = 2.269

#numba code needs separate function since classes are not supported yet
@njit
def compiled_sweep(grid, size, temp, field):
  flips = 0
  energy_change = 0.0
  beta = 1.0 / (k * temp)

  for _ in range(size * size):
    x = np.random.randint(0, size)
    y = np.random.randint(0, size)
    spin = grid[x, y]

    # Periodic boundaries via modulo
    neighbors = (
      grid[(x - 1) % size, y]
      + grid[(x + 1) % size, y]
      + grid[x, (y - 1) % size]
      + grid[x, (y + 1) % size]
    )
    dE = 2 * pairc * spin * neighbors + 2 * field[x][y] * spin

    if dE <= 0 or np.random.random() < math.exp(-beta * dE):
      grid[x, y] = -spin
      energy_change += dE
      flips += 1

  return energy_change, flips


# a lot of these functions aren't actively being used, but they were used for 
# some analysis at some time
class Ising:
  def __init__(self, size, temp=2.0):
    self.size = size
    self.temp = temp
    self.grid = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))
    self.rng = np.random.default_rng()
    self.field = np.zeros((size, size))

  def __repr__(self):
    result = ""
    for row in self.grid:
      for x in row:
        result += " + " if x == 1 else " - "
      result += "\n"
    return result

  def reset(self):
    self.grid = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(self.size, self.size))
    self.field = np.zeros((self.size, self.size))

  def energy(self):
    #really the only use of scipy
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    x = (convolve2d(self.grid, kernel, mode="same", boundary="wrap")) * self.grid
    return (-pairc * x.sum() / 2) - (self.field * self.grid).sum()

  def magnetism(self):
    return self.grid.sum() / (self.size * self.size)

  def deltaE(self, x, y):
    d = (
      self.grid[x, (y + 1) % self.size]
      + self.grid[x, (y - 1) % self.size]
      + self.grid[(x + 1) % self.size, y]
      + self.grid[(x - 1) % self.size, y]
    )
    return 2 * pairc * d * self.grid[x, y] + 2 * self.grid[x, y] * self.field[x][y]

  def numba_sweep(self):
    return compiled_sweep(self.grid, self.size, self.temp, self.field)

  # NOTE: Does not take into account external field
  # I could do the logic but i dont think it really matters
  # since this code is not really being used
  def wolf_step(self):
    x = self.rng.integers(self.size)
    y = self.rng.integers(self.size)
    seed = (x, y)
    target_spin = self.grid[x, y]

    cluster = {seed}
    stack = [seed]

    beta = 1.0 / (k * self.temp)
    p_add = 1.0 - np.exp(-2.0 * beta * pairc)

    while stack:
      cx, cy = stack.pop()
      neighbors = [
        ((cx - 1) % self.size, cy),
        ((cx + 1) % self.size, cy),
        (cx, (cy - 1) % self.size),
        (cx, (cy + 1) % self.size),
      ]

      for npos in neighbors:
        nx, ny = npos
        if (
          npos not in cluster
          and self.grid[nx, ny] == target_spin
          and self.rng.random() < p_add
        ):
          cluster.add(npos)
          stack.append(npos)

    dE = 0
    for cx, cy in cluster:
      neighbors = [
        ((cx - 1) % self.size, cy),
        ((cx + 1) % self.size, cy),
        (cx, (cy - 1) % self.size),
        (cx, (cy + 1) % self.size),
      ]
      for nx, ny in neighbors:
        if (nx, ny) not in cluster:
          dE += int(2 * pairc * target_spin * self.grid[nx, ny])

    rows, cols = zip(*cluster)
    self.grid[rows, cols] *= -1

    return dE, len(cluster)

  # sets an external field 
  def set_rotating_field(self, angle_deg, strength, radius_ratio=0.4, spread=0.2):
    cx = self.size / 2
    cy = self.size / 2
    R = self.size * radius_ratio
    
    theta = np.radians(angle_deg)
    self.sy = cy + R * np.cos(theta)  # Column (horizontal)
    self.sx = cx - R * np.sin(theta)  # Row (vertical, inverted for standard angles)
  
    x, y = np.indices((self.size, self.size))
    dist_sq = (x - self.sx)**2 + (y - self.sy)**2
  
    sigma_sq = (self.size * spread)**2
    self.field = strength * np.exp(-dist_sq / (2 * sigma_sq))

#thank you ai for helping me here
#matplotlib is actually pretty handy
def draw_animation(N=81, temp=2.0):
  x = Ising(N, temp)
  x.set_rotating_field(angle_deg=0, strength=0)

  fig, ax = plt.subplots(figsize=(6, 8))
  plt.subplots_adjust(bottom=0.25)

  img = ax.imshow(x.grid, cmap="cool", vmin=-1, vmax=1, interpolation="none")
  ax.axis("off")

  magnet_arrow = patches.FancyArrowPatch(
      (0, 0), (0, 0), 
      color="red", 
      arrowstyle='fancy,head_length=15,head_width=15,tail_width=8', 
      alpha=0.0,  # Starts invisible since strength is 0
      zorder=1000,  # Ensures it draws on top of the grid. heh zorder
      clip_on = False
  )
  ax.add_patch(magnet_arrow)

  fps_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, color="yellow",
    fontsize=12, fontweight="bold", va="top",
    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
  )
  last_time = [time.perf_counter()]

  #axes
  ax_temp = fig.add_axes((0.2, 0.15, 0.5, 0.03))  
  ax_angle = fig.add_axes((0.2, 0.10, 0.5, 0.03)) 
  ax_field = fig.add_axes((0.2, 0.05, 0.5, 0.03)) 
  
  ax_button = fig.add_axes((0.75, 0.15, 0.15, 0.03))
  ax_button_reset = fig.add_axes((0.75, 0.10, 0.15, 0.03))
  ax_button_restart = fig.add_axes((0.75, 0.05, 0.15, 0.03))


  # widgets
  temp_slider = Slider(ax_temp, "Temp", 0.1, 5.0, valinit=critical_temperature, color="red")
  angle_slider = Slider(ax_angle, "Field Angle", 0, 360, valinit=0, color="blue")
  field_slider = Slider(ax_field, "Strength", -2.0, 2.0, valinit=0, color="purple")
  
  play_button = Button(ax=ax_button, label="Pause")
  crit_button = Button(ax=ax_button_reset, label="Reset")
  restart_button = Button(ax=ax_button_restart, label="Restart")


  # 3. callbacks
  def update_params(val):
    x.temp = temp_slider.val
    x.set_rotating_field(angle_slider.val, field_slider.val)

  temp_slider.on_changed(update_params)
  angle_slider.on_changed(update_params)
  field_slider.on_changed(update_params)

  def on_key_press(event):
    step_size = 0.05
    current_val = temp_slider.val
    if event.key in ['right', 'up']:
        temp_slider.set_val(min(temp_slider.valmax, current_val + step_size))
    elif event.key in ['left', 'down']:
        temp_slider.set_val(max(temp_slider.valmin, current_val - step_size))

  fig.canvas.mpl_connect('key_press_event', on_key_press)

  is_playing = [True]
  def toggle_play(event):
    if is_playing[0]:
      ani.pause()
      play_button.label.set_text("Play")
    else:
      ani.resume()
      play_button.label.set_text("Pause")
      last_time[0] = time.perf_counter()
    is_playing[0] = not is_playing[0]
    fig.canvas.draw_idle()

  play_button.on_clicked(toggle_play)

  def crit_callback(event):
    temp_slider.set_val(critical_temperature)
    angle_slider.set_val(0)
    field_slider.set_val(0)

  crit_button.on_clicked(crit_callback)

  def restart_callback(event):
    x.reset()
    crit_callback(event)

  restart_button.on_clicked(restart_callback)

  def update(frame):
    x.numba_sweep()
    img.set_data(x.grid)

    strength = field_slider.val
    if strength == 0:
        magnet_arrow.set_alpha(0.0)
    else:
        magnet_arrow.set_alpha(0.9)
        cx, cy = x.size / 2, x.size / 2
        
        dx = cx - x.sx
        dy = cy - x.sy
        norm = math.hypot(dx, dy)
        
        base_length = x.size * 0.2
        length = base_length + (x.size * 0.04 * abs(strength))
        
        dc_norm = (dx / norm) * (length / 2)
        dr_norm = (dy / norm) * (length / 2)
        
        scale_factor = max(10, 5 + 6 * abs(strength)) / 10
        magnet_arrow.set_mutation_scale(scale_factor)
        
        if strength > 0:
            magnet_arrow.set_color("red")
            magnet_arrow.set_positions(
                (x.sy - dc_norm, x.sx - dr_norm), # Tail 
                (x.sy + dc_norm, x.sx + dr_norm)  # Head 
            )
        else:
            magnet_arrow.set_color("blue")
            magnet_arrow.set_positions(
                (x.sy + dc_norm, x.sx + dr_norm), # Tail 
                (x.sy - dc_norm, x.sx - dr_norm)  # Head 
            )

    #fps calc
    current_time = time.perf_counter()
    dt = current_time - last_time[0]
    last_time[0] = current_time

    if dt > 0:
      fps_text.set_text(f"FPS: {1.0 / dt:.1f}")

    return [img, fps_text, magnet_arrow]

  ani = animation.FuncAnimation(fig, update, frames=500, interval=0, blit=True)
  
  plt.show()

#not using this now but it was useful for deciding the algo to use
#even though numba ends up changing everything
def benchmark_comparison(N=250, temp=2.0):
  board_metro = Ising(N, temp)
  board_wolff = copy.deepcopy(board_metro)

  metro_energy = board_metro.energy()
  metro_history = []
  metro_total_flips = 0
  metro_sweeps = 5000

  print("Running Accelerated Metropolis (Numba)...")
  start_metro = time.perf_counter()
  for _ in range(metro_sweeps):
    dE, flips = board_metro.numba_sweep()
    metro_energy += dE
    metro_total_flips += flips
    metro_history.append(
      (time.perf_counter() - start_metro, metro_energy, metro_total_flips)
    )

  # --- WOLFF BENCHMARK ---
  wolff_energy = board_wolff.energy()
  wolff_history = []
  wolff_total_flips = 0
  wolff_steps = 2000

  print("Running Wolff...")
  start_wolff = time.perf_counter()
  for _ in range(wolff_steps):
    dE, flips = board_wolff.wolf_step()
    wolff_energy += dE
    wolff_total_flips += flips
    wolff_history.append(
      (time.perf_counter() - start_wolff, wolff_energy, wolff_total_flips)
    )

  # --- PLOTTING ---
  time_m, energy_m, flips_m = zip(*metro_history)
  time_w, energy_w, flips_w = zip(*wolff_history)

  print("\n--- Results ---")
  print(
    f"Metropolis: {time_m[-1] / flips_m[-1] * 1e6:.2f} microseconds per flipped spin"
  )
  print(
    f"Wolff:      {time_w[-1] / flips_w[-1] * 1e6:.2f} microseconds per flipped spin"
  )

  plt.figure(figsize=(10, 5))
  plt.plot(flips_m, energy_m, color="blue", label="Metropolis (Numba)", alpha=0.8)
  plt.plot(flips_w, energy_w, color="red", label="Wolff", alpha=0.8)

  plt.title(f"Ising Model Equilibration (Grid: {N}x{N}, T={temp})")
  plt.xlabel("Cumulative Spins Flipped")
  plt.ylabel("Total System Energy")
  plt.legend()
  plt.grid(True)
  plt.show()

#curie point stats
def stats():
  N = 30
  runs = 1000
  sweeps = 5000
  temps = [x / 100 for x in range(10, 300, 10)]

  results = {}
  
  for t in temps:
    results[t] = []
    for r in range(runs):
      x = Ising(N, t)
      for _ in range(sweeps):
        _ = x.numba_sweep()
      mag = x.magnetism()
      results[t].append(mag * mag)

  data = list(results.values())
  data = [np.mean(x) for x in data]

  plt.errorbar(temps, data, yerr=0, capsize=5, ecolor='red', elinewidth=1.5)
  plt.title("Squared Magnetism vs Temperature")
  plt.ylabel("Squared Magnetism")
  plt.xlabel("Temperature")
  plt.show()

def curie_law_stats():
  N = 30
  runs = 500    
  sweeps = 4000
  
  temp = 3.5     
  
  fields = np.linspace(-1.0, 1.0, 41)

  results = {}
  
  for h in fields:
      results[h] = []
      for r in range(runs):
          x = Ising(N, temp)
          
          x.field = np.full((N, N), h)
          
          for _ in range(sweeps):
              _ = x.numba_sweep()
              
          mag = x.magnetism()
          results[h].append(mag)
          print(f"completed run {r} on field {h}")

  data = list(results.values()) 
  mean_mag = [np.mean(val) for val in data]
  std_mag = [np.std(val) for val in data]

  small_h_mask = np.abs(fields) <= 0.2
  small_fields = fields[small_h_mask]
  small_mag = np.array(mean_mag)[small_h_mask]

  chi = np.sum(small_fields * small_mag) / np.sum(small_fields**2)

  curie_line = chi * fields

  # --- PLOTTING ---
  plt.figure(figsize=(8, 6))

  #  data
  plt.errorbar(fields, mean_mag, yerr=std_mag, capsize=3, ecolor='red', 
               elinewidth=1.5, fmt='-o', markersize=4, label="Simulated Magnetization")

  # Curie's Law line
  plt.plot(fields, curie_line, '--', color='green', linewidth=2, 
           label=f"Curie's Law Limit ($chi \\approx {chi:.2f}$)")

  plt.title(f"Magnetization vs External Field (Paramagnetic Phase, T={temp})")
  plt.ylabel("Average Magnetization (M)")
  plt.xlabel("Uniform External Magnetic Field (H)")

  plt.ylim(max(-1.1, min(curie_line)), min(1.1, max(curie_line)))

  plt.grid(True)
  plt.axhline(0, color='black', linewidth=1)
  plt.axvline(0, color='black', linewidth=1)
  plt.legend()
  plt.show()

if __name__ == "__main__":
  #uncommnet below if you want to try other stuff
  # benchmark_comparison(N=250, temp=2.0)
  # draw_animation(N=400, temp=critical_temperature)
  # stats()  
  # curie_law_stats()
  #
  print(sys.argv)
  if len(sys.argv) != 2:
    N = 1000
  else:
    N = int(sys.argv[1])

  draw_animation(N, temp = critical_temperature)
  
