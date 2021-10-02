# %%
import os
import itertools

# %% Input Specs
OUTPUT_FILENAME = 'Commands.txt'
PYTHON_FILENAME = 'PCSE.py'

TIME_PER_RUN = 15 # minutes
NO_OF_PROCESSES = 4 # cores

PARAMETERS = {
    '--alphabeta':    [[0.01, 0.1], [0.001, 0.01], [0.0001, 0.001], [0.00001, 0.0001], [0.000001, 0.00001]], # Divides into --alpha, --beta
    '--tau':          [0.01, 0.001, 0.0001],
    '--gamma':        [0.95, 0.99],
    '--batch_size':   [32, 64, 128],
    '--layer12_size': [[300, 200], [400, 300], [500, 400], [600, 500], [700, 600], [800, 700]], # Divides into --layer1_size, --layer2_size
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
