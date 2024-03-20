import optuna
import lightgbm as lgb
import pandas as pd  # Ensure pandas is imported
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# loading file from path
file1_path = '/content/drive/My Drive/mmc/mmc_final.feather'

# read with feather
df = pd.read_feather(file1_path)

# to keep only cg-site columns
columns_to_keep = [col for col in df.columns if col.startswith("cg")]
cg = df.loc[:, columns_to_keep]

# data
X = cg

# target data
y = df["double_neg_log_relative_age"]

plt.hist(y, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.grid(True)
plt.show()

# define the objective function
def objective(trial, data, target):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42) # built in function to split into k-folds
    rmses = []

    for train_idx, val_idx in kf.split(data): # performing k-fold cross validation
        train_x, val_x = data.iloc[train_idx], data.iloc[val_idx]
        train_y, val_y = target.iloc[train_idx], target.iloc[val_idx]

        trn_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(val_x, label=val_y)

        model = lgb.train(params,
                          trn_data,
                          num_boost_round=200,
                          valid_sets=[trn_data, val_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(20)])

        preds = model.predict(val_x, num_iteration=model.best_iteration)
        rmse = mean_squared_error(val_y, preds, squared=False)
        rmses.append(rmse)

    return np.mean(rmses)

# splitting data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# creating and optimizing the study
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, train_x, train_y), n_trials=20)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# training the final model with the best parameters found
trn_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(test_x, label=test_y)

model = lgb.train(study.best_trial.params,
                  trn_data,
                  num_boost_round=1000,  
                  valid_sets=[trn_data, val_data],
                  early_stopping_rounds=50,  
                  verbose_eval=100)

# evaluating the final model
preds = model.predict(test_x, num_iteration=model.best_iteration)
final_rmse = mean_squared_error(test_y, preds, squared=False)
print('Final RMSE:', final_rmse)
