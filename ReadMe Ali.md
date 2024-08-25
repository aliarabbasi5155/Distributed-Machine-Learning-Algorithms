
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




Points: 
Bazi vaghta mibinim SGD for Large Batch, dar asl SGD yani Batch Size = 1 yani SGD ba Large Batch bi mani mishe, vali khob in SGD be jaaye Mini-batch Gradient Descent estefade mishe.

Problems:
paragraph 2 paper e SHAT ro negah kon. toosh oumade algorithm haye PS-based approach ro tozih dade. ye nokte riz dare: tooye PS server (Hamoon outer loop) ma miangine model ha ro nemigirim balke miangine gradient ha ro az model e feli kam mikonim. pas tebghe in tooye worker ha ham NABAYAD model ro hesab konim va badesh ham model ro befrestim baraye PS-server, BALKE bayad gradient ha ro befrestim baraye server va oun az avg begire va az global model kam kone. Baraye etelaate bishtar papere SlowMo daghighan zire pseudo code e Algorithm 1 gofte bayad chejoori sum begirim (ta jaii ke yadam miad) vase hamine ke lr bayad baraye outer loop bashe, choon aslan tooye iuter loop bayad global model ba global_model = global_model - lr*gradient update beshe. 

Tooye SlowMo na, vali tooye LocalSGD fekr konam model weights average gerefte mishe. Paragraphe yeki moonde be akhare Paper 26 ro bekhoon.

Tebghe Paragraphe LocalSGD safhe 4 paper 26, avalnworker ha 10 ta iteration ba SGD mizanan va bad mifrestan baraye PS ta LocalSGD emal beshe (Hamoon average gereftan).

LocalSGD ro harki ye joor tozih dade, vali in code ham manteghie,
Algorithm: LocalSGD
Input: Initial model \( w_0 \), learning rate \( \eta \), number of local steps \( H \), number of global synchronization steps \( T \), number of workers \( K \)
1. Initialize \( w_k^{(0)} = w_0 \) for all workers \( k = 1, 2, \dots, K \)
2. for each global step \( t = 0, 1, \dots, T-1 \) do:
3.     for each worker \( k = 1, 2, \dots, K \) in parallel do:
4.         for each local step \( h = 1, 2, \dots, H \) do:
5.             Compute the local gradient \( g_k^{(t,h)} \) using the current model \( w_k^{(t,h-1)} \)
6.             Update the local model: \( w_k^{(t,h)} = w_k^{(t,h-1)} - \eta g_k^{(t,h)} \)
7.         end for
8.         Compute the model difference \( \Delta w_k^{(t)} = w_k^{(t,H)} - w_k^{(t,0)} \)
9.     end for
10.    Aggregate the updates across all workers: \( w_{t+1} = w_t - \frac{1}{K} \sum_{k=1}^K \Delta w_k^{(t)} \)
11. end for
Output: Final model \( w_T \)

Aya tooye har worker 1 minibatch gharar migire?


Problem: Fek konam tooye LocalSGD va SlowMo, learning rate tooye outer loop (yani tooye marhale synchronize kardan) update mishe. tooye peper slowmo neveshte γt, yani baraye har outer loop sabete.

Problem: Tooye paper SHAT ci bayad 1 beshe na 0, dalilesh ham baghalesh neveshtam

Problem: Tooye LocalSGD inke gradinet begirim va miangine gradient begirim va bad oun ro emal konim mesle miangin gereftan az model ha mimoone. va dar zemn tooye hame other paper ha neveshtan ke synchronize stage tooye localsgd miangin model hast (makhsoosan tooye algorithm ha)

Problem: Tooye local SGD tebghe algorithm khodesh do bar dare LAMBDA*LR hesab mishe yebar tooye worker local update yebar ham tooye server global update