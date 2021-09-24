import sys

"""
argv idxs
1 - slim script
2 - pop to sample
3 - number of chroms to sample at each point
4: - list of gens to sample after CHB separation (21250 gens before modern)
"""
slim_file = sys.argv[1]
pop = sys.argv[2]
samps = sys.argv[3]
gens = sys.argv[4:]
num_gens = len(gens)

with open(slim_file, "r") as infile:
    # raw_lines = [line.strip() for line in infile.readlines()]
    raw_lines = infile.readlines()

samp_eps_line = [i for i in raw_lines if "sampling_episodes" in i]
samp_eps_start = raw_lines.index(
    samp_eps_line
)  # Start of sampling_episodes constant idx
samp_eps_end = (
    raw_lines[samp_eps_start:].index(";") + samp_eps_start
)  # End of definition idx, will be end of sampling ep array

new_lines = []
new_lines.extend(raw_lines[samp_eps_start : samp_eps_start + 1])


for gen in range(num_gens):
    new_lines.append("\t\t" + f"c({pop}, {samps}, {gen}),")
new_lines.append("\t" + f"), c(3, {num_gens})));")

finished_lines = raw_lines[:samp_eps_start]
finished_lines.extend(new_lines)
finished_lines.extend(raw_lines[samp_eps_end + 1 :])

new_file_name = "modded_" + slim_file
with open(new_file_name, "w") as outfile:
    outfile.writelines(finished_lines)
