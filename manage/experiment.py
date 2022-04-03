from typing import Dict


def experiment(args: Dict):
    new_args = args.copy()
    const_args = {}
    for kwarg in args:
        if not isinstance(args[kwarg], list) == 1:
            const_args[kwarg] = args[kwarg]
            del new_args[kwarg]
        elif len(args[kwarg]) == 1:
            const_args[kwarg] = args[kwarg][0]
            del new_args[kwarg]

    all_len = 1
    for values in new_args.values():
        all_len *= len(values)
    kwargs = [{} for _ in range(all_len)]

    last_period = 1
    for arg in new_args:
        values = new_args[arg]
        period_len = all_len // len(values) // last_period
        for period_num in range(len(values) * last_period):
            range_start = period_len * period_num
            range_end = period_len * (period_num + 1)
            for exp_num in range(range_start, range_end):
                current_kwarg = kwargs[exp_num]
                current_kwarg[arg] = values[period_num % len(values)]
        last_period *= len(values)

    for kwarg in kwargs:
        str_kwargs = [str(item) for item in kwarg.values()]
        project_name = '_'.join(str_kwargs)
        project_name = project_name.replace('(', '') \
                                   .replace(')', '') \
                                   .replace(', ', '-')
        prefix = const_args['project'] if 'project' in const_args else 'Project'
        project_name = prefix + '_' + project_name
        kwarg.update(const_args)
        kwarg['project'] = project_name
    return kwargs
