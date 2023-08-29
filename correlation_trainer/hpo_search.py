import optuna
import subprocess

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64, 128])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024, 2048])
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4, 5])
    epochs = trial.suggest_categorical('epochs', [50, 100, 150, 200, 250])
    num_hops = trial.suggest_categorical('num_hops', [2, 3, 4, 5, 6])
    num_mlp_layers = trial.suggest_categorical('num_mlp_layers', [2, 3, 4, 5])
    lr = trial.suggest_categorical('lr', [0.005, 0.001, 0.0005, 0.0001])
    trial_num = trial.number

    cmd = f"python main.py --seed 42 --batch_size {batch_size} --hidden_size {hidden_size} --num_layers {num_layers} --epochs {epochs} --num_hops {num_hops} --num_mlp_layers {num_mlp_layers} --lr {lr} --space nb101 --representation adjgin_zcp --test_tagates --loss_type pwl --id {trial_num}"
    # Write cmd and trial_num to a file
    with open(f"./trial_info.txt", "a") as file:
        file.write(cmd)

    proc = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    with open(f"./trial/trial_{trial_num}.txt", "r") as file:
        reward = float(file.read())

    return reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
