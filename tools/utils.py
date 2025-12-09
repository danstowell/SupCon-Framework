import torch
import albumentations as A
from albumentations import pytorch as AT
from torch.utils.data import DataLoader
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import FaissKNN
from faiss import IndexFlatIP

from .losses import LOSSES
from .optimizers import OPTIMIZERS
from .schedulers import SCHEDULERS
from .models import SupConModel
from .datasets import create_supcon_dataset


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_to_logs(logging, message):
    logging.info(message)


def add_to_tensorboard_logs(writer, message, tag, index):
    writer.add_scalar(tag, message, index)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, crop_transform):
        self.crop_transform = crop_transform

    def __call__(self, x):
        return [self.crop_transform(image=x), self.crop_transform(image=x)]


def build_transforms(second_stage):
    if second_stage:
        train_transforms = A.Compose([
            #A.Flip(),
            #A.Rotate(),
            A.Resize(224, 224),
            A.Normalize(),
            AT.ToTensorV2()
        ])
        valid_transforms = A.Compose([A.Resize(224, 224), A.Normalize(), AT.ToTensorV2()])

        transforms_dict = {
            "train_transforms": train_transforms,
            "valid_transforms": valid_transforms,
        }
    else:
        train_transforms = A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.15, 1.)),
            A.Rotate(),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.9),
            A.ToGray(p=0.2),
            A.Normalize(),
            AT.ToTensorV2(),
        ])

        valid_transforms = A.Compose([A.Resize(224, 224), A.Normalize(), AT.ToTensorV2()])

        transforms_dict = {
            "train_transforms": train_transforms,
            'valid_transforms': valid_transforms,
        }

    return transforms_dict


def build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=False):
    dataset_name = data_dir.split('/')[-1]

    if second_stage:
        train_features_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=True,
                                               transform=transforms['train_transforms'], second_stage=True)
    else:
        # train_features_dataset is used for evaluation -> hence, we don't need TwoCropTransform
        train_features_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=True,
                                               transform=transforms['valid_transforms'], second_stage=True)

        train_supcon_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=True,
                                               transform=TwoCropTransform(transforms['train_transforms']), second_stage=False)

    valid_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=False,
                                               transform=transforms['valid_transforms'], second_stage=True)

    if not second_stage:
        train_supcon_loader = DataLoader(
            train_supcon_dataset, batch_size=batch_sizes['train_batch_size'], shuffle=True,
            num_workers=num_workers, pin_memory=True)
    train_features_loader = DataLoader(
        train_features_dataset, batch_size=batch_sizes['train_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_sizes['valid_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    if second_stage:
        return {'train_features_loader': train_features_loader, 'valid_loader': valid_loader}
    return {'train_supcon_loader': train_supcon_loader, 'train_features_loader': train_features_loader, 'valid_loader': valid_loader}


def build_model(backbone, second_stage=False, num_classes=None, ckpt_pretrained=None, projection_dim=None, projmode=None):
    model = SupConModel(backbone=backbone, second_stage=second_stage, num_classes=num_classes, projection_dim=projection_dim, projmode=projmode)

    if ckpt_pretrained:
        model.load_state_dict(torch.load(ckpt_pretrained)['model_state_dict'], strict=False)

    return model


def build_optim(model, optimizer_params, scheduler_params, loss_params_list, projmode):
    criteria = []
    for loss_params in loss_params_list:
        if loss_params['name'] in ['SupCon', 'koleo']:
            if 'params' not in loss_params:
                loss_params['params'] = {}
            loss_params['params']['projmode'] = projmode  # pushing the projection mode through to the losses

        if 'params' in loss_params:
            new_criterion = LOSSES[loss_params['name']](**loss_params['params'])
        else:
            new_criterion = LOSSES[loss_params['name']]()
        criteria.append((new_criterion, loss_params.get('weight', 1)))

    optimizer = OPTIMIZERS[optimizer_params["name"]](model.parameters(), **optimizer_params["params"])

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criteria": criteria, "optimizer": optimizer, "scheduler": scheduler}


def compute_embeddings(loader, model, scaler, donormalise=True):
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    total_embeddings = np.zeros((len(loader)*loader.batch_size, model.embed_dim))
    total_labels = np.zeros(len(loader)*loader.batch_size)

    for idx, (images, labels) in enumerate(loader):
        images = images.cuda()
        bsz = labels.shape[0]
        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images, donormalise)
                total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
        else:
            embed = model(images, donormalise)
            total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
            total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

        del images, labels, embed
        torch.cuda.empty_cache()

    return np.float32(total_embeddings), total_labels.astype(int)


def train_epoch_constructive(train_loader, model, criteria, optimizer, scaler, ema):
    model.train()
    train_loss = []

    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0]['image'], images[1]['image']], dim=0)
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        assert not images.isnan().any()

        if scaler:
            with torch.cuda.amp.autocast():
                isbad = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
                assert not isbad
                embed = model(images)
                f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
                embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = 0
                for (criterion, weight) in criteria:
                    if weight != 0:
                        loss += criterion(embed, labels) * weight

        else:
            embed = model(images)
            f1, f2 = torch.split(embed, [bsz, bsz], dim=0)
            embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = 0
            for (criterion, weight) in criteria:
                if weight != 0:
                     loss += criterion(embed, labels) * weight

        del images, labels, embed
        torch.cuda.empty_cache()

        train_loss.append(loss.item())

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema:
            ema.update(model.parameters())

    return {'loss': np.mean(train_loss)}


twopi = 2 * np.pi
def circular_torus_embed(X):
    """Clifford projection. Input data is assumed to have a shape of [npoints, ndims] and a period of 1, e.g. for a flat torus defined over [0,1]^D.
    Note: Output is UN-normalised (the norm won't be 1). To normalise, divide by sqrt(dim), or divide by dim after you take dotprod."""
    dimension = X.shape[1]
    X2pi = X * twopi     # don't use *= here, because we don't want to mangle the input array
    cosX = np.cos(X2pi)
    sinX = np.sin(X2pi)
    #print(sinX.shape)
    return np.concatenate((cosX, sinX), axis=1)

def unwrap_pairwise_torus(X, pandas=True):
    """Convert from the 'torusN' representation back to a flat [0,1] hypertorus.
    IMPORTANT: This assumes the dimensions have been normalised in adjacent pairs.
    This is produced by torusN but NOT by torusC i.e. circular_torus_embed() in which
    the pairs are not adjacent."""
    if pandas:
        X1 = X.iloc[:, 0::2]
        X2 = X.iloc[:, 1::2]
    else:
        X1 = X[:, 0::2]
        X2 = X[:, 1::2]
    Xnew = (np.arctan2(X2, X1) / twopi) % 1.0
    return Xnew

def validation_constructive(valid_loader, train_loader, model, scaler, projmode):
    calculator = AccuracyCalculator(k=1,
        knn_func = FaissKNN(
                     index_init_fn=IndexFlatIP  # inner product
                     )
            )
    model.eval()

    query_embeddings, query_labels = compute_embeddings(valid_loader, model, scaler)
    reference_embeddings, reference_labels = compute_embeddings(train_loader, model, scaler)

    if projmode in ['torus', 'torusC']:
        query_embeddings = circular_torus_embed(query_embeddings)
        reference_embeddings = circular_torus_embed(reference_embeddings)

    acc_dict = calculator.get_accuracy(
        query_embeddings,
        query_labels,
        reference_embeddings,
        reference_labels,
        ref_includes_query=False
    )

    del query_embeddings, query_labels, reference_embeddings, reference_labels
    torch.cuda.empty_cache()

    return acc_dict


def train_epoch_ce(train_loader, model, criteria, optimizer, scaler, ema):
    model.train()
    train_loss = []

    for batch_i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = 0
                for (criterion, weight) in criteria:
                    if weight != 0:
                        loss += criterion(embed, labels) * weight
                train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(data)
            loss = 0
            for (criterion, weight) in criteria:
                if weight != 0:
                    loss += criterion(embed, labels) * weight
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        if ema:
            ema.update(model.parameters())

        del data, target, output
        torch.cuda.empty_cache()

    return {"loss": np.mean(train_loss)}


def validation_ce(model, criteria, valid_loader, scaler):
    model.eval()
    val_loss = []
    valid_bs = valid_loader.batch_size
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    y_pred, y_true = np.zeros(len(valid_loader)*valid_bs), np.zeros(len(valid_loader)*valid_bs)
    correct_samples = 0

    for batch_i, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    if criteria:
                        loss = 0
                        for (criterion, weight) in criteria:
                            loss += criterion(embed, labels) * weight
                        val_loss.append(loss.item())
            else:
                output = model(data)
                if criteria:
                    loss = 0
                    for (criterion, weight) in criteria:
                        loss += criterion(embed, labels) * weight
                    val_loss.append(loss.item())

            correct_samples += (
                target.detach().cpu().numpy() == np.argmax(output.detach().cpu().numpy(), axis=1)
            ).sum()
            y_pred[batch_i * valid_bs : (batch_i + 1) * valid_bs] = np.argmax(output.detach().cpu().numpy(), axis=1)
            y_true[batch_i * valid_bs : (batch_i + 1) * valid_bs] = target.detach().cpu().numpy()

            del data, target, output
            torch.cuda.empty_cache()

    valid_loss = np.mean(val_loss)
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_score_macro = f1_score(y_true, y_pred, average='macro')
    accuracy_score = correct_samples / (len(valid_loader)*valid_bs)

    metrics = {"loss": valid_loss, "accuracy": accuracy_score, "f1_scores": f1_scores, 'f1_score_macro': f1_score_macro}
    return metrics


def copy_parameters_from_model(model):
    copy_of_model_parameters = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    return copy_of_model_parameters


def copy_parameters_to_model(copy_of_model_parameters, model):
    for s_param, param in zip(copy_of_model_parameters, model.parameters()):
        if param.requires_grad:
            param.data.copy_(s_param.data)

"""This stuff is to update some keywords in logfilenames
from the keywords used in development to the clearer publication version"""
projmode_mapper = {'hyprs':'sphere', 'torus': 'torusC', 'torul': 'torusN'}
projmode_mapper_underscored = {f"_{k}_":f"_{v}_" for k, v in projmode_mapper.items()}
def standardise_projmode_name(astr):
	return projmode_mapper.get(astr, astr)
def standardise_projmode_name_in_loggingname(astr):
	for k, v in projmode_mapper_underscored.items():
		astr = astr.replace(k, v)
	return astr
