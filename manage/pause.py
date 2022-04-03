import os
import json


def get_last_project(last_img_num):
    projects = os.listdir('projects')
    for project in projects:
        if '_' not in project:
            continue
        project_path = os.path.join('projects', project)
        last_img = os.path.join(project_path, 'images', f'{last_img_num}.png')
        if not os.path.exists(last_img):
            return project_path


def get_config(project_path):
    config_path = os.path.join(project_path, 'config.json')
    with open(config_path) as f:
        return json.load(f)


def save_config(project_path, config):
    config_path = os.path.join(project_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)


def pause(last_img_num):
    project = get_last_project(last_img_num)
    config = get_config(project)
    config['pause'] = True
    save_config(project, config)


def unpause(last_img_num):
    project = get_last_project(last_img_num)
    config = get_config(project)
    config['pause'] = False
    save_config(project, config)


def continue_experiment(experiments, last_img_num):
    for i, kwargs in enumerate(experiments):
        project = kwargs['project']
        project_path = os.path.join('projects', project)
        last_img = os.path.join(project_path, 'images', f'{last_img_num}.png')
        if not os.path.exists(last_img):
            return i
        print('Found project:', project)
    print('EXPERIMENTS COMPLETED!!!')