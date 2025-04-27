import argparse
import datetime
import json
import os
import random
import time
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from pathlib import Path

import utils
from dataset import create_dataset, create_sampler, create_loader
from my_models.model_person_search import ALBEF
from my_models.tokenization_bert import BertTokenizer
from my_models.vit import interpolate_pos_embed
from optim import create_optimizer
from scheduler import create_scheduler
import random
import math
from torch.utils.data import DistributedSampler

import pickle
# from transformers import BertTokenizer
# from huggingface_hub import snapshot_download
# 设置 transformers 缓存目录
# os.environ['TRANSFORMERS_CACHE'] = '/home/root/.cache/huggingface/transformers'

# cluster images before each epoch begins
from sklearn.cluster import DBSCAN
from my_utils.faiss_rerank import compute_jaccard_distance

os.environ["TRANSFORMERS_CACHE"] = "/dssg/work3/xlyang/xlyang/wcj/cxd/.cache/"
def cluster_begin_epoch(train_loader, model, args,epoch = 0,tokenizer = None):
    device = "cuda"
    feature_size =256 #cuhk是577,融合之后是768
    max_size = len(train_loader.dataset)  #这个是所有的图片和描述对的数量共计6800对左右     
    image_bank = torch.zeros((max_size, feature_size)).to(device)
    index = 0

    model.to(device)
    model = model.eval()
    #TODO这玩意我以后一定改
    with torch.no_grad():
        if args.distributed:
            model=model.module
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Train Epoch的聚类: [{}]'.format(epoch)
        for i, (image1, image2, text1, text2, idx, replace,real_idx) in enumerate(metric_logger.log_every(train_loader, 60000, header)):
            image1 = image1.to(device, non_blocking=True)
            # image2 = image2.to(device, non_blocking=True)
            # idx = idx.to(device, non_blocking=True)
            replace = replace.to(device, non_blocking=True)
            # text_input1 = tokenizer(text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
            # text_input2 = tokenizer(text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
            
            image_embeds = model.visual_encoder(image1)#(13,577,768)
            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image1.device)#注意力掩码全一表示所有图像token都应该被关注
            image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)#用于取cls token的特征,shape(13,577)
            # extract text features
            # text_output = model.text_encoder.bert(text_input2.input_ids, attention_mask=text_input2.attention_mask,
                                                # return_dict=True, mode='text')
            # text_embeds = text_output.last_hidden_state
            # text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)#同样是取cls token的特征
            batch_size = image1.shape[0]
            
            # output_pos =model.text_encoder.bert(encoder_embeds=text_embeds,
            #                                 attention_mask=text_input2.attention_mask,
            #                                 encoder_hidden_states=image_embeds,
            #                                 encoder_attention_mask=image_atts,
            #                                 return_dict=True,
            #                                 mode='fusion',
            #                                 )
            # fusion_feat=output_pos.last_hidden_state[:, 0, :]#shape(13,768)
            image_bank[index: index + batch_size] = image_feat
            index = index + batch_size
            

        image_bank = image_bank[:index]  
        print(f"Rank {torch.distributed.get_rank()} | 开始计算不同类之间的距离",index)    
        try:
            
            image_rerank_dist = compute_jaccard_distance(image_bank, k1=30, k2=6, search_option=3)  
        except Exception as e:
            print(f"[Rank {torch.distributed.get_rank()}] 计算距离出错：{e}")     
        # image_rerank_dist = compute_jaccard_distance(image_bank, k1=30, k2=6, search_option=0)  

        # DBSCAN cluster
        print(f"\n[Rank {torch.distributed.get_rank()} | 聚类开始]")
        cluster = DBSCAN(eps= 0.6, min_samples=4, metric='precomputed', n_jobs=-1)

        image_pseudo_labels = cluster.fit_predict(image_rerank_dist)    

        del image_rerank_dist
        # ✅ 打印统计信息
        dataset_len = len(train_loader.dataset)
        num_noise = (image_pseudo_labels == -1).sum()
        num_clusters = len(set(image_pseudo_labels)) - (1 if -1 in image_pseudo_labels else 0)

        print(f"\n[Rank {torch.distributed.get_rank()} | 聚类完成]")
        print(f"Dataset 总长度: {dataset_len}")
        print(f"聚类数（不含 -1）: {num_clusters}")
        print(f"-1 (未归入任何簇) 的数量: {num_noise}\n")
    del image_bank
    # save_path= os.path.join(args.output_dir, f'cluster.pkl')
    # if save_path is not None:
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     with open(save_path, 'wb') as f:
    #         pickle.dump({
    #             'pseudo_labels': image_pseudo_labels,
    #         }, f)
    #     print(f"Pseudo labels (pickle) saved to {save_path}")
    return image_pseudo_labels



def load_pseudo_labels_from_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded pseudo labels from {file_path}")
        return data['pseudo_labels']
    else:
        raise FileNotFoundError(f"No pseudo label pickle found at {file_path}")

class ValidIndexDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=True):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.epoch = 0
        self.valid_indices = self._sync_valid_indices()

    def _sync_valid_indices(self):
        """
        获取并同步 valid_indices，避免多进程不一致。
        """
        # 主进程获取
        if dist.get_rank() == 0:
            if hasattr(self.dataset, 'valid_indices'):
                valid_indices = self.dataset.valid_indices
            else:
                valid_indices = list(range(len(self.dataset)))
        else:
            valid_indices = None

        # 广播同步
        object_list = [valid_indices]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]
    def set_valid_indices(self, valid_indices):
        self.valid_indices = valid_indices
        
        
    def __iter__(self):

        indices = self.valid_indices

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # 统一长度逻辑
        if self.drop_last:
            num_samples_per_replica = len(indices) // self.num_replicas
            total_size = num_samples_per_replica * self.num_replicas
            indices = indices[:total_size]
        else:
            num_samples_per_replica = int(math.ceil(len(indices) / self.num_replicas))
            total_size = num_samples_per_replica * self.num_replicas
            padding_size = total_size - len(indices)
            if padding_size > 0:
                # 重复填充（防止部分 rank 拿不到数据）
                indices += indices[:padding_size]

        assert len(indices) == total_size

        # 分配给当前 rank
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == num_samples_per_replica

        return iter(indices)

    def __len__(self):
        if self.drop_last:
            return len(self.valid_indices) // self.num_replicas
        else:
            return int(math.ceil(len(self.valid_indices) / self.num_replicas))

    def set_epoch(self, epoch):
        self.epoch = epoch

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config,image_pseudo_labels):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_cl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_pitm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_prd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mrtd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    for i, (image1, image2, text1, text2, idx, replace,real_idx) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        batch_size = image1.shape[0]
        #单次的伪标签
        batch_pseudo_id = image_pseudo_labels[real_idx]
        
        image1 = image1.to(device, non_blocking=True)
        image2 = image2.to(device, non_blocking=True)
        replace = replace.to(device, non_blocking=True)
        # print(idx)

        # 转换为 Tensor
        idx = torch.tensor(batch_pseudo_id).to(device, non_blocking=True)
        valid_mask = idx != -1
        valid_mask = valid_mask.to(device, non_blocking=True)
        # 计算有效样本数量
        valid_count = valid_mask.sum().item()
    
        # 初始化损失为0
        loss_cl = loss_pitm = loss_mlm = loss_prd = loss_mrtd = torch.tensor(0.0, device=device)
        
        if valid_count > 0:
            # 有有效样本时正常处理
            # 🧹 Step 2: 根据掩码筛选数据
            valid_image1 = image1[valid_mask]
            valid_image2 = image2[valid_mask]
            valid_replace = replace[valid_mask]
            valid_idx = idx[valid_mask]
            
            # 注意：text 需要是 list[str]，不是 Tensor，因此也按 mask 筛选
            valid_text1 = [t for j, t in enumerate(text1) if valid_mask[j]]
            valid_text2 = [t for j, t in enumerate(text2) if valid_mask[j]]
            
            valid_text_input1 = tokenizer(valid_text1, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
            valid_text_input2 = tokenizer(valid_text2, padding='longest', max_length=config['max_words'], return_tensors="pt").to(device)
            
            if epoch > 0 or not config['warm_up']:
                alpha = config['alpha']
            else:
                alpha = config['alpha'] * min(1.0, i / len(data_loader))
                
            try:
                loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd = model(valid_image1, valid_image2, 
                                                                    valid_text_input1, valid_text_input2,
                                                                    alpha=alpha, idx=valid_idx, replace=valid_replace)
            except Exception as e:
                print("🚨 Model forward error:", str(e))
                print("valid_image1.shape:", valid_image1.shape)
                print("valid_image2.shape:", valid_image2.shape)
                print("valid_text_input1.input_ids.shape:", valid_text_input1['input_ids'].shape)
                print("valid_text_input2.input_ids.shape:", valid_text_input2['input_ids'].shape)
                print("valid_idx.shape:", valid_idx.shape)
                print("valid_replace.shape:", valid_replace.shape)
                raise  # re-raise the exception to not hide the original error
                            
        else:
            # 如果没有有效样本，创建一个很小的损失以保持梯度流动
            # 这个损失足够小，不会影响训练，但能让NCCL同步工作
            dummy_loss = torch.tensor(1e-8, device=device, requires_grad=True)
            loss_cl = loss_pitm = loss_mlm = loss_prd = loss_mrtd = dummy_loss
        
        
        # 计算总损失
        loss = 0.
        for j, los in enumerate((loss_cl, loss_pitm, loss_mlm, loss_prd, loss_mrtd)):
            loss += config['weights'][j] * los
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(loss_cl=loss_cl.item())
        metric_logger.update(loss_pitm=loss_pitm.item())
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_prd=loss_prd.item())
        metric_logger.update(loss_mrtd=loss_mrtd.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # evaluate
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    # extract text features
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_words'], return_tensors="pt").to(device)
        text_output = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    # extract image features
    image_feats = []
    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)
        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)
    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    # compute the feature similarity score for all image-text pairs
    sims_matrix = text_embeds @ image_embeds.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)
    # take the top-k candidates and calculate their ITM score sitm for ranking
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        topk_idx = topk_idx.to(image_feats.device)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder.bert(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                         attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                         encoder_hidden_states=encoder_output.to(device),
                                         encoder_attention_mask=encoder_att,
                                         return_dict=True,
                                         mode='fusion'
                                         )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_t2i.cpu()

@torch.no_grad()
def itm_eval(scores_t2i, img2person, txt2person, eval_mAP):
    img2person = torch.tensor(img2person)
    txt2person = torch.tensor(txt2person)
    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    if eval_mAP:
        real_num = matches.sum(dim=-1)
        tmp_cmc = matches.cumsum(dim=-1).float()
        order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long)
        tmp_cmc /= order
        tmp_cmc *= matches
        AP = tmp_cmc.sum(dim=-1) / real_num
        mAP = AP.mean() * 100.0
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       'mAP': mAP.item()
                       }
    else:
        eval_result = {'r1': ir1,
                       'r5': ir5,
                       'r10': ir10,
                       'r_mean': ir_mean,
                       }
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    print(args)
    print(config)
    print("args.distributed",args.distributed)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    # Dataset
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('ps', config)
    # if args.distributed:
    #     num_tasks = utils.get_world_size()
    #     global_rank = utils.get_rank()
    #     samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    # else:
    #     samplers = [None, None, None]
        
    if args.distributed:
        sampler = ValidIndexDistributedSampler(train_dataset, num_replicas=utils.get_world_size(), rank=utils.get_rank())
    else:
        sampler = None

    samplers = [sampler, None, None]
    #dataloader在这,然后logevery是直接迭代这个
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    cluster_lodaer = create_loader([train_dataset], [None],
                                                          batch_size=[config['batch_size_train']],
                                                          num_workers=[4],
                                                          is_trains=[False],
                                                          collate_fns=[None])[0]
    
    print("args.text_encoder",args.text_encoder)
    # tokenizer = BertTokenizer.from_pretrained(args.text_encoder, 
    #                                     proxies={'http': 'http://127.0.0.1:7890',
    #                                             'https': 'http://127.0.0.1:7890'},
    #                                     trust_remote_code=True, 
    #                                             force_download=True,
    #                                             local_files_only=True,
    #                                             )#

    # tokenizer = BertTokenizer.from_pretrained(
    # "/dssg/work3/xlyang/xlyang/wcj/cxd/.cache/bert-base-uncased/bert-base-uncased",  # 使用绝对路径
    # local_files_only=True,                # 强制仅使用本地文件
    # force_download=False,                 # 必须关闭强制下载
    # proxies=None                          # 明确禁用代理
    # )
    # tokenizer = BertTokenizer.from_pretrained("/dssg/work3/xlyang/xlyang/wcj/cxd/.cache/")
    "/dssg/work3/xlyang/xlyang/wcj/cxd/.cache/bert-local"
    # model_path = snapshot_download("bert-base-uncased", cache_dir="/dssg/work3/xlyang/xlyang/wcj/cxd/.cache/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("/dssg/work3/xlyang/xlyang/wcj/cxd/.cache/bert-local", local_files_only=True)
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    best_log = ''

    # Model
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    model = model.to(device)
    # Optimizer and learning rate scheduler
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']
        else:
            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        
        if not args.evaluate:
            train_dataset.mode = 'cluster'
            
            if args.distributed:
                # 伪标签一个rank生成然后广播
                # pseudo_label_path = os.path.join(args.output_dir, f'cluster.pkl')
                # if os.path.exists(pseudo_label_path):
                #     image_pseudo_labels_np = load_pseudo_labels_from_pickle(pseudo_label_path)
                #     image_pseudo_labels = torch.tensor(image_pseudo_labels_np).long().cuda()
                # else:
                #     if torch.distributed.get_rank() == 0:
                #         image_pseudo_labels_np = cluster_begin_epoch(cluster_lodaer, model, args, epoch, tokenizer)
                #         image_pseudo_labels = torch.tensor(image_pseudo_labels_np).long().cuda()
                #     else:
                #         image_pseudo_labels_np = cluster_begin_epoch(cluster_lodaer, model, args, epoch, tokenizer)

                # 全局同步
                image_pseudo_labels_np = cluster_begin_epoch(cluster_lodaer, model, args, epoch, tokenizer)
                image_pseudo_labels = torch.tensor(image_pseudo_labels_np).long().cuda()
                torch.distributed.broadcast(image_pseudo_labels, src=0)
                
            else:
                image_pseudo_labels_np = cluster_begin_epoch(cluster_lodaer, model, args,epoch,tokenizer)
                # image_pseudo_labels_np = load_pseudo_labels_from_pickle(pseudo_label_path)
                image_pseudo_labels = torch.tensor(image_pseudo_labels_np).long().cuda()
            image_pseudo_labels=image_pseudo_labels.tolist()
            image_num_cluster = len(set(image_pseudo_labels)) - (1 if -1 in image_pseudo_labels else 0)
            # print("==> Statistics for epoch [{}]: {} image clusters".format(epoch, image_num_cluster))
            
            
            # print(f"Rank {torch.distributed.get_rank()} | 伪标签数量: {len(image_pseudo_labels)}")
            # print(f"Rank {torch.distributed.get_rank()} | 样本数: {train_dataset.__len__()}")
            
            train_dataset.mode = 'train'
            train_dataset.set_pseudo_labels(image_pseudo_labels)
            # print(f"Rank {torch.distributed.get_rank()} | 写入伪标签后的样本数: {train_dataset.__len__()}")
            
            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)
            if args.distributed:
                train_loader.sampler.set_valid_indices(train_dataset.valid_indices)
                train_loader.sampler.set_epoch(epoch)
                
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config,torch.tensor(image_pseudo_labels).to(device))
        
        
        if epoch >= config['eval_epoch'] or args.evaluate:
            score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
            if utils.is_main_process():
                test_result = itm_eval(score_test_t2i, test_dataset.img2person, test_dataset.txt2person, args.eval_mAP)
                print('Test:', test_result, '\n')
                if args.evaluate:
                    log_stats = {'epoch': epoch,
                                 **{f'test_{k}': v for k, v in test_result.items()}
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    log_stats = {'epoch': epoch,
                                 **{f'train_{k}': v for k, v in train_stats.items()},
                                 **{f'test_{k}': v for k, v in test_result.items()},
                                 }
                    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                        'best': best,
                        'best_epoch': best_epoch
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_epoch%02d.pth' % epoch))
                    if test_result['r1'] > best:
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = test_result['r1']
                        best_epoch = epoch
                        best_log = log_stats
        if args.evaluate:
            break
        dist.barrier()
        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch: {best_epoch} / {max_epoch}\n")
            f.write(f"{best_log}\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/PS_cuhk_pedes.yaml')
    parser.add_argument('--output_dir', default='output/cuhk-pedes')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval_mAP', action='store_true', help='whether to evaluate mAP')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('--embed_dim', default=577, type=int)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)
