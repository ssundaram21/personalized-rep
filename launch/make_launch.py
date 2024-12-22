import argparse
import yaml
import os
from itertools import product

def cross_product(input_dict):
    """
    Generate the cross product of the input dictionary values.

    Args:
        input_dict (dict): Dictionary with lists as values.

    Returns:
        list: List of dictionaries representing the cross product.
    """
    keys = input_dict.keys()
    values = input_dict.values()
    product_list = list(product(*values))
    result = [dict(zip(keys, combination)) for combination in product_list]
    return result


def process(v):
    """
    Process the input value to format it as a string if necessary.

    Args:
        v: The value to process.

    Returns:
        str: The processed value.
    """
    if isinstance(v, str):
        return "'" + v + "'"
    else:
        return v


def generate_command(config_file):
    """
    Generate command-line arguments from a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        list: List of generated command strings.
    """
    # Load YAML configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    commands = []
    # Get the filename from the config
    filename = config.pop('filename', None)

    # Check for sweep arguments 
    if 'sweep_args' in config and isinstance(config['sweep_args'], dict):
        sweep_args_dict = config['sweep_args']
        sweep_args_dictlist = cross_product(sweep_args_dict)

        # Sweep over all sweep args:
        for sweep_args_i in sweep_args_dictlist:
            parser = argparse.ArgumentParser()
            
            for key, value in sweep_args_i.items():
                arg_name = key.replace('_', '-')  # Convert underscores to dashes
                parser.add_argument(f'--{arg_name}', default=value)
            
            for key, value in config['arguments'].items():
                arg_name = key.replace('_', '-')  # Convert underscores to dashes
                if isinstance(value, bool):
                    parser.add_argument(f'--{arg_name}', action="store_true")
                else:
                    parser.add_argument(f'--{arg_name}', default=value)
            
            sweep_args_list = [f'--{key.replace("_", "-")}={value}' for key, value in sweep_args_i.items()]
            args_list = [f'--{key.replace("_", "-")}={value}' for key, value in config['arguments'].items() if not isinstance(value, bool)]
            args_list += [f'--{key.replace("_", "-")}' for key, value in config['arguments'].items() if isinstance(value, bool)]
            all_args_list = sweep_args_list + args_list

            args = parser.parse_args(all_args_list)
            command = f'python {filename} ' + ' '.join([f'--{k}={process(v)}' for k, v in vars(args).items() if not isinstance(v, bool)])
            command += ' ' + ' '.join([f'--{k}' for k, v in vars(args).items() if isinstance(v, bool)])
            commands.append(command)
    else:
        parser = argparse.ArgumentParser()

        for key, value in config['arguments'].items():
            arg_name = key.replace('_', '-')  # Convert underscores to dashes
            if isinstance(value, bool):
                parser.add_argument(f'--{arg_name}', action="store_true")
            else:
                parser.add_argument(f'--{arg_name}', default=value)

        args_list = [f'--{key.replace("_", "-")}={value}' for key, value in config['arguments'].items() if not isinstance(value, bool)]
        args_list += [f'--{key.replace("_", "-")}' for key, value in config['arguments'].items() if isinstance(value, bool)]

        args = parser.parse_args(args_list)
        command = f'python {filename} ' + ' '.join([f'--{k}={process(v)}' for k, v in vars(args).items() if not isinstance(v, bool)])
        command += ' ' + ' '.join([f'--{k}' for k, v in vars(args).items() if isinstance(v, bool)])
        commands.append(command)

    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate command-line arguments from yaml file')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--launchfile_name', type=str, required=False, default=None, help='Path to output')
    parser.add_argument('--quiet', action="store_true")
    args = parser.parse_args()

    commands = generate_command(args.config_file)
    if not args.quiet:
        for cmd in commands:
            print(cmd)

    if args.launchfile_name is not None:
        os.makedirs("launch/launchfiles", exist_ok=True)
        with open(os.path.join("launch/launchfiles", f"{args.launchfile_name}.sh"), "w") as f:
            for i, command in enumerate(commands):
                if i != 0:
                    f.write("\n")
                f.write(command)
