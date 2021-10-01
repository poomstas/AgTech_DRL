import argparse

parser = argparse.ArgumentParser(description='Practice parsing arguments with this')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate (float)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size (int)')
parser.add_argument('--nondefault_param', type=int, help="This one doesn't have a default value (int)")

args = parser.parse_args()

print(args.learning_rate)
print(args.batch_size)
print(args.nondefault_parameter)

# python argparse_practice --nondefault_param 2

# (spacewalk) Brian (master *) AgTech $ python argparse_practice.py --nondefault_parameter 12312
# 0.001
# 32
# 12312