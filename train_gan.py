import argparse
import os
import torch
import json
import random
import numpy as np
import wandb
from datetime import datetime
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from dataload.data_utils import loadH5Full
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd
import pickle as pkl
from config import getConfig
from agents import get_agent

parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--n_cluster", type=int, default=8, help="number of k means clusters")
parser.add_argument("--quadratic", type=bool, default=True, help="quadratic features for discrminator")


random.seed(2021)
DATA_DIR = "data/Chair"
JSON_DIR = "data/parts_json"
RESOLUTION = 64

def readJsonPartCategories(chairID, nParts):
    categories = []
    path = os.path.join(JSON_DIR, chairID, "result.json")
    try:
        f = open(path)
        data = json.load(f)
        for (i, el) in enumerate(data[0]['children']):
            partName = el['text']
            if partName == "Chair Back":
                categories.append(1)
            elif partName == "Chair Seat":
                categories.append(2)
            elif partName == "Chair Arm":
                categories.append(3)
            elif partName == "Chair Base":
                # this will also apply to following parts
                for j in range(i, nParts):
                    categories.append(4)
        f.close()
    except Exception as e:
        categories = [0]*nParts # missing corresponding json labels for h5
    return categories[0:nParts] # sometimes json file has more labels than voxelized parts, in which case results are truncated

def getChairPartInfos(encoder, filename):
    path = os.path.join(DATA_DIR, filename)

    nParts, partVoxel, dataPoints, dataVals, scales, translations, size = loadH5Full(path, resolution=RESOLUTION)
    voxelTensor = torch.tensor(partVoxel.astype(np.float), dtype=torch.float32).unsqueeze(1)  # (nParts, 1, dim, dim, dim)
    with torch.no_grad():
        latentVecs = encoder(voxelTensor.cuda()).cpu().numpy()
    nParts = latentVecs.shape[0]
    strid = filename[0:-3]
    categories = np.array(readJsonPartCategories(strid, nParts))

    # numpy arrays
    # vecs: (nParts, latent dimension)
    # scales: (nParts, 1)
    # translations: (nParts, 3)
    # categories: (nParts)
    return {'vecs':latentVecs, 'scales':scales, 'translations':translations, 'categories':categories,
            'matchedCategories': len(categories)==len(latentVecs), 'filenames':[filename]*len(latentVecs)}


def loadAllChairsInfoIterable():
    # part latent vectors, categories, and affine transforms
    filenames = filter(lambda filename:filename.endswith(".h5"), os.listdir(DATA_DIR))

    config = getConfig()
    agent = get_agent("partae", config)
    agent.loadChkPt(config.ckpt)
    config.resolution = RESOLUTION
    encoder = agent.net.encoder
    
    chairInfos = filter(lambda x:x is not None, map(lambda filename: getChairPartInfos(encoder, filename), filenames))
    
    return chairInfos



chairInfo = list(loadAllChairsInfoIterable())

vecs = np.concatenate(list(map(lambda x:x['vecs'], chairInfo)))
normalized_vecs = normalize(vecs)
pca_obj = PCA(n_components=75)
pca_out = pca_obj.fit_transform(normalized_vecs)
variance_preserved_at_dimension = pca_obj.explained_variance_ratio_.cumsum()

threshold = 0.95
num_dim = np.argmax(variance_preserved_at_dimension > threshold) + 1
pca_out = pca_out[:, 0:num_dim]

centroid = np.mean(np.concatenate(list(map(lambda x:x['translations'], chairInfo))), axis=0)
reference_yz_vector = np.zeros(3, dtype=float)
reference_yz_vector[2] = 1.0
proj_onto_yz = np.array([0.0,1.0,1.0])



def get_part_clusters_for_chairs_iterable(kmfit):
    return map(lambda d:{'clusters':kmfit.predict(pca_obj.transform(d['vecs'])[:, 0:num_dim]), 'scales':d['scales'], 'translations':d['translations']}, chairInfo)

def max_part_count_on_chair(kmfit, chairPartClusteredInfo):
    return np.amax(np.stack(list(map(lambda d:np.bincount(d['clusters'], minlength=kmfit.n_clusters), chairPartClusteredInfo))), axis=0)

def p_percentile_part_count_on_chair(kmfit, chairPartClusteredInfo, p):
    return np.percentile(np.stack(list(map(lambda d:np.bincount(d['clusters'], minlength=kmfit.n_clusters), chairPartClusteredInfo))), q=p, axis=0)

# https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vectorize_chair_info(maxCountPerCluster, chairClusterDict):
    maxCountPerCluster = maxCountPerCluster.astype(int)
    clusterEndIndices = maxCountPerCluster.cumsum()
    clusterStartIndices = clusterEndIndices - maxCountPerCluster
    count_dim = clusterEndIndices[-1]
    affine_dim = clusterEndIndices[-1] * 4
    count_vec = np.zeros(count_dim, dtype=bool)
    affine_vec = np.zeros(affine_dim, dtype=float)
    for (cluster_id, (cluster_max_count, cluster_start_index)) in enumerate(zip(maxCountPerCluster, clusterStartIndices)):
        matched_part_indices = np.where(chairClusterDict['clusters']==cluster_id)[0]
        matched_affines = []
        for (part_num, part_index) in enumerate(matched_part_indices):
            if part_num == cluster_max_count:
                break
            part_start_index = cluster_start_index + part_num
            count_vec[part_start_index] = True
            matched_affines.append([part_num, chairClusterDict['scales'][part_index], chairClusterDict['translations'][part_index]])
        matched_affines.sort(key=lambda x:angle_between((x[2]-centroid)*proj_onto_yz, reference_yz_vector)) # sort by rotational angle w.r.t. centroid on yz plane
        for (part_num, scale, translation) in matched_affines:
            part_start_index = cluster_start_index + part_num
            affine_vec[part_start_index*4] = scale
            affine_vec[part_start_index*4+1:part_start_index*4+4] = translation - centroid
    return {'count':count_vec, 'affine':affine_vec}

class GanDataset(Dataset):
    def __init__(self, kmfit, max_count_percentile):
        super(GanDataset, self).__init__()
        partClustersDicts = list(get_part_clusters_for_chairs_iterable(kmfit))
        maxCountPerCluster = p_percentile_part_count_on_chair(kmfit, partClustersDicts, max_count_percentile)
        self.chairVectorDicts = list(map(lambda d:vectorize_chair_info(maxCountPerCluster, d), get_part_clusters_for_chairs_iterable(kmfit)))
        self.countDim = int(maxCountPerCluster.sum())
        self.affineDim = int(self.countDim * 4)
        
    def __len__(self):
        return len(self.chairVectorDicts)
    
    def __getitem__(self, index):
        d = self.chairVectorDicts[index]
        return {'count':torch.cuda.BoolTensor(d['count']), 'affine':torch.cuda.FloatTensor(d['affine'])}


def engineer_feature_vec(counts, affines, quadratic):
    # counts: [bs, parts]
    # affines: [bs, parts*4]
    # mask: [bs, parts]
    bs = counts.shape[0]
    mask = (counts > 0.5).detach()
    affines = (affines.view(bs, -1, 4) * mask.unsqueeze(-1)).view(bs, -1)
    if quadratic:
        counts_quadric = counts.view(bs, -1, 1) * counts.view(bs, 1, -1) # [bs, parts, parts]
        counts_quadric = counts_quadric.view(bs, -1) # [bs, parts*parts]
        features = torch.cat([counts_quadric, affines], dim=1) # [bs, parts^2 + parts*4]
    else:
        features = torch.cat([counts, affines], dim=1) # [bs, parts*5]
    return features
    
    
# based on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
class Generator(nn.Module):
    def __init__(self, latent_dim, count_dim, affine_dim):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048)
        )
        self.count_layer = nn.Sequential(nn.Linear(2048, count_dim), nn.Sigmoid())
        self.affine_layer = nn.Sequential(nn.Linear(2048, affine_dim))

    def forward(self, z):
        features = self.model(z)
        counts = self.count_layer(features)
        affines = self.affine_layer(features)
        return {'count':counts, 'affine':affines}


class Discriminator(nn.Module):
    def __init__(self, count_dim, affine_dim, quadratic):
        super(Discriminator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        if quadratic:
            feature_dim = int(count_dim**2 + count_dim*4)
        else:
            feature_dim = int(count_dim*5)
        self.model = nn.Sequential(
            *block(feature_dim, 2048, normalize=False),
            *block(2048, 1024),
            *block(1024, 512),
            *block(512, 256),
            nn.Linear(256, 1)
        )
        

    def forward(self, features):
        validity = self.model(features)
        return validity

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_gp(kmfit, quadratic_feature_engineering, n_epoch=50,
                  batch_size=64, learning_rate=0.00005, betas=[0.5, 0.999], latent_dim=128, lambda_gp=10, n_critic=5):
    wandb.init(project="764gan", entity="xda35")
    n_cluster = kmfit.n_clusters
    wandb.config = {
      "epochs": n_epoch,
      "learning_rate": learning_rate,
      "batch_size": batch_size,
      "n_critic": 5,
      "quadratic_features": quadratic_feature_engineering,
      "n_clusters": n_cluster
    }
    train_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    session_str = train_time + "_km" + str(n_cluster)
    if quadratic_feature_engineering:
        session_str = session_str + "_q/"
    wandb.run.name = session_str
    weights_dir = os.path.join("data/","weights/", session_str)
    os.makedirs(weights_dir, exist_ok=True)
    pkl.dump(kmfit, open(os.path.join(weights_dir, 'kmfit.pkl'), 'wb')) # save k-means model
    gweights_dir = os.path.join(weights_dir, "generator/")
    dweights_dir = os.path.join(weights_dir, "discriminator/")
    os.makedirs(gweights_dir, exist_ok=True)
    os.makedirs(dweights_dir, exist_ok=True)
    
    ds = GanDataset(kmfit, 95)
    count_dim = ds.countDim
    affine_dim = ds.affineDim
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    generator = Generator(latent_dim, count_dim, affine_dim).cuda()
    discriminator = Discriminator(count_dim, affine_dim, quadratic_feature_engineering).cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)
    for epoch in range(n_epoch):
        num_d_iter = 0
        num_g_iter = 0
        epoch_d_loss = torch.zeros(1).cuda()
        epoch_g_loss = torch.zeros(1).cuda()
        for i, real_batch in enumerate(dataloader):
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False
            num_d_iter += 1
            real_count = real_batch['count']
            real_affine = real_batch['affine']
            real_features = engineer_feature_vec(real_count, real_affine, quadratic_feature_engineering)
            # ----------------------------------------
            # Train D()
            # ----------------------------------------
            optimizer_D.zero_grad()
            # sample noise for G()
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_features.shape[0], latent_dim)))
            fake_batch = generator(z)
            fake_count = fake_batch['count']
            fake_affine = fake_batch['affine']
            fake_features = engineer_feature_vec(fake_count, fake_affine, quadratic_feature_engineering)
            real_validity = discriminator(real_features)
            fake_validity = discriminator(fake_features)
            gradient_penalty = compute_gradient_penalty(discriminator, real_features, fake_features)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            epoch_d_loss += d_loss.detach()
            if i % n_critic == 0:
                for p in discriminator.parameters():
                    p.requires_grad = False
                for p in generator.parameters():
                    p.requires_grad = True
                num_g_iter += 1
                # ----------------------------------------
                # Train G()
                # ----------------------------------------
                fake_batch = generator(z)
                fake_count = fake_batch['count']
                fake_affine = fake_batch['affine']
                fake_features = engineer_feature_vec(fake_count, fake_affine, quadratic_feature_engineering)
                fake_validity = discriminator(fake_features)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
                epoch_g_loss += g_loss.detach()
        wandb.log({'epoch':epoch+1, 'epoch_d_loss':epoch_d_loss.item()/num_d_iter, 'epoch_g_loss':epoch_g_loss.item()/num_g_iter})
        if epoch % 2500 == 0:
            torch.save(generator.state_dict(),os.path.join(gweights_dir, str(epoch)+'.pth'))
            torch.save(discriminator.state_dict(),os.path.join(dweights_dir, str(epoch)+'.pth'))
        
if __name__ == "__main__":
    opt = parser.parse_args()
    kmfit_ = MiniBatchKMeans(n_clusters=opt.n_cluster,
                            random_state=0,
                            compute_labels=False).fit(pca_out)
    train_wgan_gp(kmfit_, opt.quadratic, n_epoch=opt.n_epoch, n_critic=opt.n_critic)
    
