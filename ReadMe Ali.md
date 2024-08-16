
# Ali TODOs
1. Understanding LocalSGD 
2. Understanding SlowMo and the difference with LocalSGD


# Project stages step-by-step
1.  A. Defining class for handling CIFAR100
        def __init__(self, batch_size=64)
        def compute_mean_std(self, loader)
        def download_data(self)
        def split_data(self, original_train_set, validation_ratio=0.2)
        def compute_statistics(self, train_set)
        def apply_transforms(self, train_mean, train_std, is_validation_set_available = False)
        def save_data(self, data_loader, file_name: str)
        def load_data(self, file_name: str)
        def create_and_save_data_loaders(self, train_set, test_set, train_name: str, test_name: str, validation_set=None)
        def prepare_data(self, validation_ratio = None)
        def train_valid_test(self, validation_ratio=0.2)
        def train_test(self)
        def iid_shards(self, num_shards=2)
    B. Load Kardane train_set, validation_set, test_set, ...

2.  A. Tarif e Model Architecture: Similar to LeNet-5
    B. Tarif e Loss Function (CrossEntropy)
    C. Tarif e function train
    D. Tarif e function test
    E. Tarif e save_checkpoint va load_checkpoint (ba in RegEx./train/{optimizer}/{batch_size}/{learning_rate}/{weight_decay}/{epoch}.pt)
    F. Tarif e function run_training (baraye handle kardane hyperparams (run_training( num_epochs, model, trainloader, validationloader, testloader, optimizer, scheduler, loss_fn, device, optimizer_name: str, accumulation_steps=1, hyperparameters=None, is_wandb = False, n_epochs_stop = None )

3. Centeralized Baseline
    A. SGDM (Stochastic Gradient Descent with Momentum)
        i. Hyperparameter tuning
        ii. Train with the best haperparameters set
    B. AdamW (Adam with Weight Decay)
        i. Hyperparameter tuning
        ii. Train with the best haperparameters set

4. Large Batch Normalization
    A. Define LARS
        i. Define class for LARS inheriting from torch.optimizer
    B. Define LAMB
        i. Define class for LARS inheriting from torch.optimizer
    C. Train with LARS
        i. Hyperparameter tuning only for lr (wd = 1e-03)
        ii. Train with the best haperparameters set lr = 1.5 Check paper 18 to find efficient lr (but wd is also has been set)
        iii. Train using batch size (Lower 4096 = batch size, Greater than 4096 use  accumulation_steps = batch_size // 4096)
    D. Train with LAMB
        i. Hyperparameter tuning only for lr (wd = 4e-04) 
        ii. Train with the best haperparameters set (Why lr = 4.8/(2**5 *1e02) # approximately 15e-04 wd = 4e-04)
        iii. Train using batch size (Lower 4096 = batch size, Greater than 4096 use  accumulation_steps = batch_size // 4096)

5. Switch to Local Methods
    A. LocalSGD (paper 19)
        i. Load LeNet-5 with initial state dict.
        ii. synchrorize function (why?)
        iii. Implement local_SGD function.
        iv. train using different iid_shards

6. Try other distributed optimizers (SlowMo) Paper 21
    A. SlowMo
        i. Define local_SGD_SlowMo
        ii. Apply on different shards

