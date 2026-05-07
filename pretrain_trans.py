#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm


from data_structs import MolData, Vocabulary,MolData_v
from model_trans import Transformer_
from utils import decrease_learning_rate
from tdc.generation import MolGen
import os
import shutil
import time
rdBase.DisableLog('rdApp.error')

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# import setproctitle
# setproctitle.setproctitle("reposition@ft")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = MolGen(name = 'ZINC')
# data = MolGen(name = 'ChEMBL')
def pretrain(voc_file, smiles_file, save_dir, ckpt_name=None,
             plot_name=None, device=None,
             batch_size=256, lr=1e-3, epochs=12,
             restore_from=None):
    """Trains the Prior transformer"""

    logger_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())    

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = os.path.join(save_dir, logger_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)

    # data1 = MolGen(name='ZINC')
    # data2= MolGen(name='ChEMBL')
    # data1=list(data1.smiles_lst.values)
    # data2=[]
    # data_all=data1+data2
    # shape_arr = []
    # out_list=['i','I','.','P','[Se]','[2H]','B','9','[N@+]','[3H]','[C-]','[O]','[18F]']
    # for inter in data_all:
    #     pass_=False
    #     if type(inter)==float:
    #         aaaaa=1
    #     else:
    #         for inter1 in out_list:
    #             if inter1 in inter:
    #                 pass_=True
    #         if pass_:
    #             aaa=1
    #         else:
    #             a1='i' in inter
    #             a2='I' in inter
    #             a3='.' in inter
    #             if a1 or a2:
    #                 aaa = 1
    #             elif a3:
    #                 aaaa=1
    # 
    #             else:
    #                 shape_arr.append(inter)


    print('# Create a Dataset from a SMILES file')
    moldata = MolData(smiles_file, voc)
    # moldata = MolData("data/mols_filtered.smi", voc)
    # moldata=MolData_v(shape_arr, voc)
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16,
                      collate_fn=MolData.collate_fn)
    print('# Build DataLoader')

    Prior = Transformer_(voc)
    # toLoad=False 
    
    # Can restore from a saved RNN
    if restore_from:
        Prior.transformer.load_state_dict(torch.load(restore_from))
        print("# Loaded", restore_from)
    # if toLoad:
    #     # Loading the checkpoint
    #     checkpoint_path = "data/Prior_transformer-epoch67-valid-86.71875.ckpt"#"data/Prior_transformer-epoch5-valid-54.6875.ckpt"
    #     Prior.transformer.load_state_dict(torch.load(checkpoint_path))
    #     print("loaded",checkpoint_path)
    print("# Build Transformer")
    optimizer = torch.optim.Adam(Prior.transformer.parameters(), lr=lr)
    train_losses = []

    print("# Begin to learn")
    for epoch in range(1, epochs + 1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        total_loss = 0

        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate los
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()
            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            # print("loss:",loss)
            optimizer.step()

            total_loss += loss.item()
            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                with torch.no_grad():
                # tqdm.write("*" * 50)
                # tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                    seqs, likelihood, _ = Prior.sample(128)
                    valid = 0
                    for i, seq in enumerate(seqs.cpu().numpy()):
                        smile = voc.decode(seq)
                        if Chem.MolFromSmiles(smile):
                            valid += 1
                        if i < 5:
                            tqdm.write(smile)
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("*" * 50 + "\n")
                # torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")
            #torch.cuda.empty_cache()

        avg_loss = total_loss / len(data)
        train_losses.append(avg_loss)
        tqdm.write(f"# [Pretrain] Epoch {epoch} average loss: {avg_loss:.6f}")

        # Save the Prior
        # torch.save(Prior.transformer.state_dict(), f"data/Prior_transformer-epoch{epoch}-valid-{100 * valid / len(seqs)}.ckpt")
        if ckpt_name:
            save_ckpt = os.path.join(save_dir, ckpt_name)
        else:
            save_ckpt = os.path.join(save_dir,
                                     f"Prior_transformer-epoch{epoch}-valid-{100 * valid / len(seqs)}.ckpt")
        torch.save(Prior.transformer.state_dict(), save_ckpt)
        print(f"# [Pretrain] Epoch {epoch}'s model saved to {save_ckpt}")

    # 可视化训练曲线
    plt.plot(train_losses, label="Pretrain Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretraining Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, plot_name))
    plt.close()

    shutil.copy("./pretrain_trans.py", save_dir)

def finetune(voc_file, smiles_file, save_dir, ckpt_name=None,
             plot_name=None, device=None,
             batch_size=256, lr=1e-3, epochs=12, patience=3,
             freeze_embeddings=True, restore_from=None):
    """Finetunes the Prior transformer"""

    # This function is similar to pretrain but can be used for finetuning
    # the model on a specific dataset. The implementation is left as an exercise.
    logger_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = os.path.join(save_dir, logger_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=voc_file)
    
    print('# Create a Dataset from a SMILES file')
    moldata = MolData(smiles_file, voc)

    print('# Build DataLoader')
    data = DataLoader(moldata, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=16,
                      collate_fn=MolData.collate_fn)

    Prior = Transformer_(voc)

    # Can restore from a saved RNN
    if restore_from:
        Prior.transformer.load_state_dict(torch.load(restore_from))
        print("# Loading", restore_from)
    print("# Build Transformer")

    if freeze_embeddings:
        for param in Prior.transformer.embedding1.parameters():
            param.requires_grad = False
    
    optimizer = torch.optim.Adam(Prior.transformer.parameters(), lr=lr)
    best_loss = float('inf')
    no_improve_epochs = 0
    train_losses = []

    print("# Begin to learn")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for step, batch in tqdm(enumerate(data), total=len(data)):
            print("# step:",step)

            # Sample from DataLoader
            seqs = batch.long()
            # Calculate los
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()
            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            # print("loss:",loss)
            optimizer.step()

            total_loss += loss.item()
            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                with torch.no_grad():
                # tqdm.write("*" * 50)
                # tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data[0]))
                    seqs, likelihood, _ = Prior.sample(128)
                    valid = 0
                    for i, seq in enumerate(seqs.cpu().numpy()):
                        smile = voc.decode(seq)
                        if Chem.MolFromSmiles(smile):
                            valid += 1
                        if i < 5:
                            tqdm.write(smile)
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("*" * 50 + "\n")
                # torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")
            #torch.cuda.empty_cache()

        if step < 500:
            with torch.no_grad():
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
        avg_loss = total_loss / len(data)
        train_losses.append(avg_loss)
        tqdm.write(f"# [Finetune] Epoch {epoch} average loss: {avg_loss:.6f}")

        # Save the Prior
        if ckpt_name:
            save_ckpt = os.path.join(save_dir, ckpt_name)
        else:
            try:
                print("len(seqs):",len(seqs))
                print("valid:",valid)
                print("100 * valid / len(seqs):",100 * valid / len(seqs))
                save_ckpt = os.path.join(save_dir,
                                         f"Prior_finetune_transformer-epoch{epoch}-valid-{100 * valid / len(seqs)}.ckpt")
            except:
                save_ckpt = os.path.join(save_dir,
                                         f"Prior_finetune_transformer-epoch{epoch}-valid-NaN.ckpt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
            torch.save(Prior.transformer.state_dict(), save_ckpt)
            print(f"# [Finetune] Epoch {epoch}'s model saved to {save_ckpt}")
        else:
            no_improve_epochs += 1
            tqdm.write(f"# No improvement for {no_improve_epochs} epochs.")
            save_ckpt = os.path.join(save_dir,
                                     f"Prior_finetune_transformer-epoch{epoch}|no_improve-valid-{100 * valid / len(seqs)}.ckpt")
            torch.save(Prior.transformer.state_dict(), save_ckpt)
            print(f"# [Finetune] Epoch {epoch}'s model saved to {save_ckpt}")
            if no_improve_epochs >= patience:
                tqdm.write("⏹️ Early stopping triggered.")
                break

    # 可视化训练曲线
    plt.plot(train_losses, label="Finetune Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Finetuning Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, plot_name))
    plt.close()



if __name__ == "__main__":
    # pretrain()
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    torch.multiprocessing.set_start_method('spawn')
    
    # Example usage, relative paths
    # pretrain(
    #     voc_file="data/Voc-payrus_H_M_L_patent",
    #     smiles_file="data/payrus_H_M_L_filtered.smi",
    #     save_dir="data/payrus_H_M_L",
    #     ckpt_name=None,
    #     plot_name="pretrain_loss_payrus_H_M_L.png",
    #     batch_size=128,
    #     lr=1e-3,
    #     epochs=50,
    #     )

    # finetune(
    #     restore_from="data/payrus_H_M_L/2025-11-02-22_27_14/Prior_transformer-epoch33-valid-79.6875.ckpt",
    #     voc_file="data/Voc-payrus_H_M_L_patent",
    #     smiles_file="data/patent_filtered.smi",
    #     patience=3,
    #     save_dir="data/finetune/payrus_H_M_L",
    #     ckpt_name=None,
    #     plot_name="finetune_loss_patent-payrus_H_M_L.png",
    #     freeze_embeddings=True,
    #     lr=5e-5,
    #     batch_size=128,
    #     epochs=20
    #     )

    #-----------------------------------------------------
    # Kinase-patent pretraining and finetuning
    #-----------------------------------------------------

    #pretrain(
    #    voc_file="data/Voc-payrus_H_M_L_patent_kinase",
    #    smiles_file="data/payrus_H_filtered.smi",
    #    save_dir="data/payrus_H/kinase",
    #    ckpt_name=None,
    #    plot_name="pretrain_loss_payrus_H.png",
    #    batch_size=128,    
    #    lr=1e-3,
    #    epochs=50,
    #    )
    
    # Single-kinase finetuning
    KINASES = [
    "ALK_TYROSINE_KINASE_RECEPTOR",
    "CYCLIN-DEPENDENT_KINASE_2",
    "CYCLIN-DEPENDENT_KINASE_4",
    "CYCLIN-DEPENDENT_KINASE_6",
    "CYCLIN-DEPENDENT_KINASE_9",
    "DUAL_SPECIFICITY_MITOGEN-ACTIVATED_PROTEIN_KINASE_KINASE_1",
    "DUAL_SPECIFICITY_MITOGEN-ACTIVATED_PROTEIN_KINASE_KINASE_2",
    "EPIDERMAL_GROWTH_FACTOR_RECEPTOR",
    "LEUCINE-RICH_REPEAT_SERINE_THREONINE-PROTEIN_KINASE_2",
    "PHOSPHATIDYLINOSITOL_4_5-BISPHOSPHATE_3-KINASE_CATALYTIC_SUBUNIT_ALPHA_ISOFORM",
    "PHOSPHATIDYLINOSITOL_4_5-BISPHOSPHATE_3-KINASE_CATALYTIC_SUBUNIT_DELTA_ISOFORM",
    "PHOSPHATIDYLINOSITOL_4_5-BISPHOSPHATE_3-KINASE_CATALYTIC_SUBUNIT_GAMMA_ISOFORM",
    "RHO-ASSOCIATED_PROTEIN_KINASE_1",
    "RHO-ASSOCIATED_PROTEIN_KINASE_2",
    "SERINE_THREONINE-PROTEIN_KINASE_B-RAF",
    "SERINE_THREONINE-PROTEIN_KINASE_MTOR",
    "TYROSINE-PROTEIN_KINASE_JAK1",
    "TYROSINE-PROTEIN_KINASE_JAK2",
    "TYROSINE-PROTEIN_KINASE_JAK3",
    "MITOGEN-ACTIVATED_PROTEIN_KINASE_KINASE_KINASE_KINASE_1",
    "RECEPTOR-INTERACTING_SERINE_THREONINE-PROTEIN_KINASE_3",
    ]
# 
    for kinase in KINASES:
        print(f"======== Finetuning for kinase: {kinase} ========")
        smi_file = f"data/kinase/{kinase}/keepmean-norepeat_{kinase}_filtered.smi"
        finetune(
            restore_from="data/payrus_H_M_L/kinase/2026-01-05-13_35_14/Prior_transformer-epoch45-valid-80.46875.ckpt",
            voc_file="data/Voc-payrus_H_M_L_patent_kinase",
            smiles_file=smi_file,
            patience=3,
            save_dir=f"data/finetune/payrus_H_M_L/kinase/{kinase}",
            ckpt_name=None,
            plot_name="finetune_loss_patent-payrus_H_M_L.png",
            freeze_embeddings=True,
            lr=5e-5,
            batch_size=128,
            epochs=20
            )

    # Multi-kinase finetuning
    # finetune(
    #     restore_from="data/payrus_H_M_L/kinase/2026-01-05-13_35_14/Prior_transformer-epoch45-valid-80.46875.ckpt",
    #     voc_file="data/Voc-payrus_H_M_L_patent_kinase",
    #     smiles_file="data/kinase/17_kinases/keepmean-norepeat_17_kinases_filtered.smi",
    #     patience=3,
    #     save_dir="data/finetune/payrus_H_M_L/kinase/17_kinases",
    #     ckpt_name=None,
    #     plot_name="finetune_loss_patent-payrus_H_M_L.png",
    #     freeze_embeddings=True,
    #     lr=5e-5,
    #     batch_size=128,
    #     epochs=20
    #     )