import os
import sys
import subprocess

s_path = os.path.dirname(os.getcwd())
sys.path.insert(0, s_path)
from manage.pause import continue_experiment, pause, unpause
from manage.experiment import experiment

if __name__ == '__main__':
    args = {
        "learning_rate": [0.0005, 0.0002, 0.00005],
        "noise_dim": [50, 200],
        "start_dense": [(2, 2, 512), (2, 2, 128)],
        "strides": [True, False],
        "transp_conv": [True, False],
        "project": "diamonds64",
        "img_shape": (64, 64, 3),
        "packs": 4,
        "batch_size": 24,
        "epochs": 300,
        "fit": True,
        "load": False,
        "img_save_interval": 100,
        "gpu": True
    }
    experiments = experiment(args)
    start_from = continue_experiment(experiments, 65400)
    for i in range(start_from, len(experiments)):
        project_name = experiments[i]['project']
        print(f'STARTING SUBPROCESS #{i}: {project_name}')
        subprocess.call([r'C:\Users\Admin\AppData\Local\Programs\Python\Python36\python.exe', 'wgangp.py', str(i)])