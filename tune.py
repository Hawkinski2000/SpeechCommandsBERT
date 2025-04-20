import optuna

from config import EncoderConfig
from model import build_model


class Tuner:
    def __init__(self, ddp_config, trainer):
        self.ddp_config = ddp_config
        self.master_process = ddp_config['master_process']
        self.trainer = trainer

    def objective(self, single_trial):
        trial = optuna.integration.TorchDistributedTrial(single_trial)

        n_layer = trial.suggest_categorical('n_layer', [1, 2, 4, 8])
        n_head = trial.suggest_categorical('n_head', [1, 2, 4, 8, 16])
        n_embd = trial.suggest_categorical('n_embd', [256, 512, 1024, 2048])
        B = trial.suggest_categorical('B', [16, 32, 64, 128])
        attn_pdrop = trial.suggest_float('attn_pdrop', 0, 0.5)
        resid_pdrop = trial.suggest_float('resid_pdrop', 0, 0.5)
        mlp_pdrop = trial.suggest_float('mlp_pdrop', 0, 0.5)
        max_lr = trial.suggest_float('max_lr', 3e-4, 3e-3)
        weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1)

        config = EncoderConfig(n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            B=B,
                            attn_pdrop=attn_pdrop,
                            resid_pdrop=resid_pdrop,
                            mlp_pdrop=mlp_pdrop,
                            max_lr=max_lr,
                            weight_decay=weight_decay)
        
        model, raw_model = build_model(self.ddp_config, config)

        val_loss = self.trainer.train(model, raw_model, self.ddp_config)

        return val_loss

    def tune(self):
        study = None
        n_trials = 10
        if self.master_process:
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=n_trials)
            
            print("Best hyperparameters:", study.best_params)

        else:
            for _ in range(n_trials):
                try:
                    self.objective(None)
                except optuna.TrialPruned:
                    pass