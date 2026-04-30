import matplotlib.pyplot as plt
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

results = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="pl_name, discoverymethod, pl_bmassj, pl_orbsmax",
    where="discoverymethod in ('Transit', 'Radial Velocity', 'Imaging', "
          "'Transit Timing Variations', 'Microlensing') "
          "and pl_bmassj > 0 and pl_orbsmax > 0"
)

solar_system = {'Earth_massj': 0.00314,
                'Earth_smax': 1,
                'Jupiter_massj': 1,
                'Jupiter_smax': 5.2038,
                'Neptune_massj': 0.0539,
                'Neptune_smax': 30.07,}

solar_system_mass = [1.739e-4, 0.00256, 0.00314, 3.38e-4, 1, 0.299, 0.0457, 0.0539]
solar_system_smax = [0.387, 0.723, 1, 1.523, 5.2038, 9.536, 19.189, 30.07]

# Color each detection method differently
colors = {
    "Transit": "steelblue",
    "Radial Velocity": "tomato",
    "Imaging": "mediumseagreen",
    "Transit Timing Variations": "orchid",
    "Microlensing": "orange",
}

fig, ax = plt.subplots(figsize=(10, 7))

for method, color in colors.items():
    mask = results["discoverymethod"] == method
    subset = results[mask]
    ax.scatter(
        subset["pl_orbsmax"],
        subset["pl_bmassj"],
        c=color,
        label=method,
        alpha=0.6,
        edgecolors="none",
        s=50,
    )
ax.scatter(solar_system_smax, solar_system_mass, c = "magenta", label = 'solar system', alpha = 1, s = 100)

ax.set_xscale("log")
ax.set_yscale("log") 
ax.set_xlabel("Semi-Major Axis (AU)", fontsize=16)
ax.set_ylabel("Planet Mass (Jupiter Masses)", fontsize=16)
ax.set_title("Exoplanet Mass vs Semi-Major Axis", fontsize=20)
ax.legend(fontsize=14)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.tick_params(axis="both", which="both", labelsize=16)

plt.tight_layout()
plt.show()