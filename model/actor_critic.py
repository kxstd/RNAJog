from model.actor import Actor
from model.critic import Critic
from model.embedder_pretrain import Embedder
import torch.nn as nn
import torch
# torch.autograd.set_detect_anomaly(True)
from utils.random_gen import codon_gen, cod2case_mul, cod2case
from utils.constant import get_mask_f, trans_idx2codon
from torch.utils.tensorboard import SummaryWriter  
from os import path
import os
from tqdm import tqdm
from utils.evaluate import calculate_mfe_mul
from torch.distributions.categorical import Categorical
from model.cai_model import Cai_model
import pandas as pd

class Actor_Critic(nn.Module):
    def __init__(self,cfg, device):
        super(Actor_Critic,self).__init__()
        self.seed = cfg.seed
        self.device = torch.device(device)

        # model
        self.embedder = Embedder(cfg,device)
        self.actor = Actor(cfg,device)
        self.critic = Critic(cfg)
        self.get_mask = get_mask_f()
        # train
        self.lr = cfg.lr
        self.train_bs = cfg.train_bs
        self.eval_bs = cfg.eval_bs
        self.batch_num = cfg.batch_num
        self.max_threads = cfg.max_threads
        self.save_path = cfg.save_path
        self.eval_step = cfg.eval_step
        try:
            self.train_alpha = cfg.train_alpha
            self.cai_table = pd.read_csv(cfg.cai_table_path)
        except:
            pass
        self.length = cfg.length 
        self.start_step = 0

        if cfg.pretrain_weights :
            self.load_state_dict(torch.load(cfg.pretrain_weights))
            self.start_step = int(cfg.pretrain_weights.split("/")[-1].split(".")[0].split("model")[-1])

        self.to(self.device) 

    def forward(self, codons, pro, length, merge_prob, alpha):
        embeddings = self.embedder(codons)
        outputs, log_probs = self.actor(embeddings, pro, length, merge_prob, alpha)
        b = self.critic(embeddings)

        return outputs, log_probs, b
    
    def optimize(self, codons, pro, length, merge_prob, alpha, sample_method, sample_temperature, ban_codon_table, ban_pro_seqs, gc_repress):
        # print(ban_pro_seqs[0])
        # print(ban_pro_seqs[0] in pro[0])
        # print(pro)
        # input("Press Enter to continue...")
        if ban_pro_seqs:
            for ban_pro_seq in ban_pro_seqs:
                for pro_seq in pro:
                    if ban_pro_seq in "".join(pro_seq):
                        raise ValueError("Protein sequence {} is banned".format(ban_pro_seq))
        embeddings = self.embedder(codons)
        outputs, log_probs = self.actor.optimize(embeddings, pro, length, merge_prob, alpha, sample_method, sample_temperature, ban_codon_table, gc_repress)

        return outputs, log_probs


    def train_from_dataset(self, data, eval_data):

        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        mse_loss = nn.MSELoss()
        writer = SummaryWriter(path.join(self.save_path,'log'))
        os.system("cp {} {}".format("config/train.yaml",self.save_path))
        eval_cases_codon, eval_cases_pro, eval_length = cod2case(self.seed, next(iter(eval_data)))
        print("Start training")

        batch_idx = self.start_step
        best_mfe = 0

        cai_model = Cai_model(self.cai_table)
        

        for train_batch in tqdm(data):
            self.train()
            input_batch_codon, input_batch_pro, length = cod2case(self.seed, train_batch)
            cai_score = cai_model.calculate_score(input_batch_pro, length).to(self.device)
            outputs, log_probs, b = self(input_batch_codon, input_batch_pro, length, merge_prob = cai_score, alpha = self.train_alpha)
            
            with torch.no_grad():
                mfe = calculate_mfe_mul(outputs, self.max_threads)
                mfe = torch.tensor(mfe).to(self.device)
                disadvantage = mfe/torch.tensor(length).to(self.device) +b
            actor_loss = torch.mean(disadvantage*log_probs)
            critic_loss = mse_loss(b, -mfe/torch.tensor(length).to(self.device))
            loss = 0.9*actor_loss + 0.1*critic_loss
            writer.add_scalar('mfe', torch.mean(mfe), batch_idx)
            writer.add_scalar('actor_loss', actor_loss, batch_idx)
            writer.add_scalar('critic_loss', critic_loss, batch_idx)
            writer.add_scalar('loss', loss, batch_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            optimizer.step()

            batch_idx += 1

            if batch_idx % self.eval_step == 0:
                # torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))
                cai_score = cai_model.calculate_score(eval_cases_pro, eval_length).to(self.device)
                outputs, _ , _ = self(eval_cases_codon,eval_cases_pro, eval_length, merge_prob = cai_score, alpha = self.train_alpha)
                with torch.no_grad():
                    mfe = calculate_mfe_mul(outputs, self.max_threads)
                    mfe = torch.tensor(mfe).to(self.device)
                avg_mfe = torch.mean(mfe)
                print("batch:{}, mfe: {}".format(batch_idx, avg_mfe))
                if avg_mfe < best_mfe:
                    torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))
            
        cai_score = cai_model.calculate_score(eval_cases_pro, eval_length).to(self.device)
        outputs, _ , _ = self(eval_cases_codon,eval_cases_pro, eval_length, merge_prob = cai_score, alpha = self.train_alpha)
        with torch.no_grad():
            mfe = calculate_mfe_mul(outputs, self.max_threads)
            mfe = torch.tensor(mfe).to(self.device)
        avg_mfe = torch.mean(mfe)
        print("batch:{}, mfe: {}".format(batch_idx, avg_mfe))
        if avg_mfe < best_mfe:
            torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))

    def train_from_random_data(self, eval_data):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        mse_loss = nn.MSELoss()
        writer = SummaryWriter(path.join(self.save_path,'log'))
        os.system("cp {} {}".format("config/train.yaml",self.save_path))
        eval_cases_codon, eval_cases_pro, eval_length = cod2case(self.seed, next(iter(eval_data)))
        print("Start training")

        batch_idx = self.start_step
        best_mfe = 0

        cai_model = Cai_model(self.cai_table)
        

        for _ in tqdm(range(self.batch_num)):
            self.train()
            # input_batch_codon, input_batch_pro, length = cod2case(self.seed, train_batch)
            length = 1017* batch_idx//self.batch_num + 5
            input_batch_codon, input_batch_pro= codon_gen(self.seed,self.train_bs,length)
            length = [length for i in range(self.train_bs)]
            cai_score = cai_model.calculate_score(input_batch_pro, length).to(self.device)
            outputs, log_probs, b = self(input_batch_codon, input_batch_pro, length, merge_prob = cai_score, alpha = self.train_alpha)
            
            with torch.no_grad():
                mfe = calculate_mfe_mul(outputs, self.max_threads)
                mfe = torch.tensor(mfe).to(self.device)
                disadvantage = mfe/torch.tensor(length).to(self.device) +b
            actor_loss = torch.mean(disadvantage*log_probs)
            critic_loss = mse_loss(b, -mfe/torch.tensor(length).to(self.device))
            loss = 0.9*actor_loss + 0.1*critic_loss
            writer.add_scalar('mfe', torch.mean(mfe), batch_idx)
            writer.add_scalar('actor_loss', actor_loss, batch_idx)
            writer.add_scalar('critic_loss', critic_loss, batch_idx)
            writer.add_scalar('loss', loss, batch_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            optimizer.step()

            batch_idx += 1

            if batch_idx % self.eval_step == 0:
                # torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))
                cai_score = cai_model.calculate_score(eval_cases_pro, eval_length).to(self.device)
                outputs, _ , _ = self(eval_cases_codon,eval_cases_pro, eval_length, merge_prob = cai_score, alpha = self.train_alpha)
                with torch.no_grad():
                    mfe = calculate_mfe_mul(outputs, self.max_threads)
                    mfe = torch.tensor(mfe).to(self.device)
                avg_mfe = torch.mean(mfe)
                print("batch:{}, mfe: {}".format(batch_idx, avg_mfe))
                if avg_mfe < best_mfe:
                    torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))
            
        cai_score = cai_model.calculate_score(eval_cases_pro, eval_length).to(self.device)
        outputs, _ , _ = self(eval_cases_codon,eval_cases_pro, eval_length, merge_prob = cai_score, alpha = self.train_alpha)
        with torch.no_grad():
            mfe = calculate_mfe_mul(outputs, self.max_threads)
            mfe = torch.tensor(mfe).to(self.device)
        avg_mfe = torch.mean(mfe)
        print("batch:{}, mfe: {}".format(batch_idx, avg_mfe))
        if avg_mfe < best_mfe:
            torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))



    # def train_from_data(self, data, eval_data):
    #     # actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.lr)
    #     # critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.lr)
    #     optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)

    #     mse_loss = nn.MSELoss()
    #     writer = SummaryWriter(path.join(self.save_path,'log'))
    #     os.system("cp {} {}".format("config/train.yaml",self.save_path))
    #     eval_cases_codon, eval_cases_pro, eval_length = cod2case(self.seed, next(iter(eval_data)))
    #     # eval_cases_codon, eval_cases_pro = codon_gen(self.seed,self.eval_bs,self.length)
    #     print("Start training")

    #     batch_idx = 0
    #     best_mfe = 0
    #     for train_batch in tqdm(data):
    #         self.train()
    #         # input_batch_codon, input_batch_pro = codon_gen(self.seed,self.train_bs,self.length) 
    #         # input_batch_codon, input_batch_pro = cod2case_mul(self.seed, train_batch, self.train_bs)
    #         input_batch_codon, input_batch_pro, length = cod2case(self.seed, train_batch)
    #         # for i in range(10):
    #         #     print(input_batch_codon[i][:10])
    #         # for i in range(10):
    #         #     print(input_batch_pro[i][:10])
    #         # return
    #         prob, b = self(input_batch_codon, input_batch_pro)

    #         log_list = []
    #         idxs = []
    #         for i in range(len(prob)):
    #             distr = Categorical(probs = prob[i][:length[i]])
    #             idx = distr.sample() 
    #             idxs.append(idx)
    #             logs = distr.log_prob(idx)
    #             log_list.append(logs.sum(dim = -1))
    #         log_probs = torch.stack(log_list)
    #             # log_probs = torch.stack([logs[i].sum(dim = -1) for i in range(len(logs))])
    #         # codon_output = 
    #         # output.append("".join(codon_output))
    #     # log_probs = torch.stack([sum(log_probs[i]) for i in range(batch_size)])
        
    #         output = trans_idx2codon(idxs, input_batch_pro)
    #         mfe = calculate_mfe_mul(output, self.max_threads)
    #         mfe = torch.tensor(mfe).to(self.device)
    #         with torch.no_grad():
    #             disadvantage = mfe/torch.tensor(length).to(self.device) +b
    #         actor_loss = torch.mean(disadvantage*log_probs)
    #         critic_loss = mse_loss(b, -mfe/torch.tensor(length).to(self.device))
    #         loss = 0.9*actor_loss + 0.1*critic_loss
    #         # print("MFE:{}  Loss:{}".format(torch.mean(mfe),loss))
    #         writer.add_scalar('mfe', torch.mean(mfe), batch_idx)
    #         writer.add_scalar('actor_loss', actor_loss, batch_idx)
    #         writer.add_scalar('critic_loss', critic_loss, batch_idx)
    #         writer.add_scalar('loss', loss, batch_idx)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
    #         optimizer.step()


    #         if batch_idx % self.eval_step == 0:
    #             # print("batch: {}, loss: {}".format(batch_idx,loss.item()))
    #             output, mfe, b = self.evaluate(eval_cases_codon,eval_cases_pro, eval_length)
    #             avg_mfe = torch.mean(torch.tensor(mfe))
    #             print("batch:{}, mfe: {}".format(batch_idx, avg_mfe))
    #             if avg_mfe < best_mfe:
    #                 torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))

    #         batch_idx += 1

    #     output, mfe, b = self.evaluate(eval_cases_codon,eval_cases_pro, eval_length)
    #     avg_mfe = torch.mean(torch.tensor(mfe))
    #     print("batch:{}, mfe: {}".format(batch_idx, avg_mfe))

    #     torch.save(self.state_dict(), self.save_path+"/model{}.pt".format(batch_idx))

    #     writer.close()

            
        

