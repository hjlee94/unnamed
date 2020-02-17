
def read_config(conf_path):
    params = dict()

    fd = open(conf_path)

    for line in fd:
        line = line.strip()

        if len(line) < 1 or line[0] == '#':
            continue

        key, value = line.split('=')

        key = key.strip()
        value = value.strip()

        params[key] = value

    return params

