# %%
import os
import itertools

# %% Input Specs
OUTPUT_FILENAME = 'TD3_Commands.txt'
PYTHON_FILENAME = 'main_PCSE.py'

# Used for Estimating Time Durations
TIME_PER_RUN = 180 # minutes
NO_OF_PROCESSES = 4 # simultaneous processes

PARAMETERS = {
    '--alphabeta': [
                        [0.001,   0.001],
                        [0.0001,  0.0001],
                        [0.01,    0.01],
                        [0.001,   0.01],
                        [0.0001,  0.001],
                        [0.01,    0.1],
                    ], # Divides into --alpha, --beta
    '--tau':            [0.05, 0.005, 0.0005],
    '--update_act_int': [2],
    '--batch_size':     [100, 200, 300],
    '--layer12_size':   [[400, 300]], # Divides into --layer1_size, --layer2_size
    '--n_games':        [10000],
    '--patience':       [1000],
    '--TB_note':        ['""'],
}

# %%
def print_command(key_list, value_list):
    assert len(key_list) == len(value_list)
    output = []
    for key, value in zip(key_list, value_list):
        if type(value) == list:
            if key == '--alphabeta':
                item_string = '--alpha ' + str(value[0]) + ' --beta ' + str(value[1])
            elif key == '--layer12_size':
                item_string = '--layer1_size ' + str(value[0]) + ' --layer2_size ' + str(value[1])
        else:
            item_string = str(key) + " " + str(value)
        output.append(item_string)
    return  "\t".join(output)

print_command(['--batch_size', '--gamma', '--theta'], [64, 0.99, 0.982]) # Use example

# %%
keys = tuple(PARAMETERS.keys())
combos = list(itertools.product(*PARAMETERS.values()))

print("Total Number of Combinations: ", len(combos))
print("Estimated Run Time: {} hours".format(len(combos) * TIME_PER_RUN / 60 / NO_OF_PROCESSES))

# %% Write to file
if os.path.isfile(OUTPUT_FILENAME):
    print("File already exists! Overwriting...")

with open(OUTPUT_FILENAME, 'w') as f:
    for item in combos:
        line = 'python ' + PYTHON_FILENAME + ' ' + print_command(keys, item)
        print(line)
        f.write(line +'\n')
