"""Pretrained ResNet-18 cat/dog inference and grouped-score Grad-CAM."""
from pathlib import Path
import argparse, hashlib, json, platform, random, re
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
NAMES = ['cat', 'dog']
CAT_IDS = list(range(281, 286))  # domestic cat categories only
DOG_IDS = list(range(151, 269))  # domestic dog breeds, excludes wolves/foxes
GROUPS = [CAT_IDS, DOG_IDS]
# Explicit Oxford-IIIT Pet breed mapping, verified against the supplied archive.
CAT_BREEDS = {'Abyssinian','Bengal','Birman','Bombay','British_Shorthair','Egyptian_Mau',
              'Maine_Coon','Persian','Ragdoll','Russian_Blue','Siamese','Sphynx'}
DOG_BREEDS = {'american_bulldog','american_pit_bull_terrier','basset_hound','beagle','boxer',
              'chihuahua','english_cocker_spaniel','english_setter','german_shorthaired',
              'great_pyrenees','havanese','japanese_chin','keeshond','leonberger',
              'miniature_pinscher','newfoundland','pomeranian','pug','saint_bernard',
              'samoyed','scottish_terrier','shiba_inu','staffordshire_bull_terrier',
              'wheaten_terrier','yorkshire_terrier'}


def label_for(path):
    """Accept class directories or cat.123.jpg/dog.123.jpg; never guess."""
    breed = path.stem.rsplit('_', 1)[0]
    if breed in CAT_BREEDS: return 0
    if breed in DOG_BREEDS: return 1
    aliases = {'cat': 0, 'cats': 0, 'dog': 1, 'dogs': 1}
    for parent in path.parents:
        if parent.name.lower() in aliases:
            return aliases[parent.name.lower()]
    match = re.match(r'^(cat|dog)[._\-\s]', path.name.lower())
    return aliases[match.group(1)] if match else None


def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def select_images(root, per_class=200, seed=SEED):
    """Random balanced subset, after excluding unreadable/duplicate files."""
    pools = {0: [], 1: []}
    unknown = 0
    for p in sorted(Path(root).rglob('*')):
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            label = label_for(p)
            if label is None:
                unknown += 1
            else:
                pools[label].append(p)
    rng = random.Random(seed)
    selected, skipped, seen = [], [], set()
    for label in (0, 1):
        rng.shuffle(pools[label])
        count = 0
        for p in pools[label]:
            try:
                im = load_rgb(p)
                digest = hashlib.sha256(im.size.__repr__().encode() + im.tobytes()).hexdigest()
                if digest in seen:
                    skipped.append({'file': str(p.relative_to(root)), 'reason': 'duplicate pixels'})
                    continue
                seen.add(digest)
                selected.append({'path': p, 'label': label, 'sha256_pixels': digest})
                count += 1
                if count == per_class:
                    break
            except Exception as exc:
                skipped.append({'file': str(p.relative_to(root)), 'reason': str(exc)})
        if count < per_class:
            raise ValueError(f'Only {count} valid unique {NAMES[label]} images; requested {per_class}. '
                             'Check DATA_ROOT / class folder names, or reduce PER_CLASS.')
    rng.shuffle(selected)
    return selected, skipped, {'discovered': {NAMES[k]: len(v) for k,v in pools.items()},
                               'unlabeled_files_ignored': unknown}


def group_logits(logits):
    # Softmax over these logsumexp scores is equivalent to summing ImageNet
    # probabilities within each group, then conditioning on cat OR dog.
    return torch.stack([torch.logsumexp(logits[:, ids], dim=1) for ids in GROUPS], dim=1)


def build_model(device):
    weights = ResNet18_Weights.IMAGENET1K_V1
    categories = weights.meta['categories']
    assert categories[151] == 'Chihuahua' and categories[268] == 'Mexican hairless'
    assert categories[281] == 'tabby' and categories[285] == 'Egyptian cat'
    model = resnet18(weights=weights).eval().to(device)
    # Frozen weights; input requires_grad will still permit Grad-CAM.
    model.requires_grad_(False)
    return model, weights.transforms(), categories


def predict(model, preprocess, categories, selected, root, device, batch_size=32):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start+batch_size]
            x = torch.stack([preprocess(load_rgb(item['path'])) for item in batch]).to(device)
            logits = model(x)
            binary = group_logits(logits).softmax(1).cpu()
            all_probs = logits.softmax(1)
            mass = all_probs[:, CAT_IDS + DOG_IDS].sum(1).cpu()
            top = logits.argmax(1).cpu()
            for i, item in enumerate(batch):
                pred = int(binary[i].argmax())
                rows.append({'file': str(item['path'].relative_to(root)),
                             'true_label': NAMES[item['label']], 'predicted_label': NAMES[pred],
                             'correct': pred == item['label'], 'cat_score': float(binary[i,0]),
                             'dog_score': float(binary[i,1]), 'cat_dog_mass': float(mass[i]),
                             'imagenet_top1': categories[int(top[i])],
                             'sha256_pixels': item['sha256_pixels']})
            print(f'Predicted {min(start+batch_size, len(selected))}/{len(selected)}', flush=True)
    return pd.DataFrame(rows)


def input_rgb(x):
    mean = torch.tensor([.485,.456,.406]).view(3,1,1)
    std = torch.tensor([.229,.224,.225]).view(3,1,1)
    return (x.detach().cpu()*std+mean).clamp(0,1).permute(1,2,0).numpy()


def gradcam(model, x, target):
    """Gradient of the chosen grouped logit at layer4[-1] block output."""
    activations = []
    handle = model.layer4[-1].register_forward_hook(lambda m, i, o: activations.append(o))
    try:
        with torch.enable_grad():
            logits = model(x.detach().clone().requires_grad_(True))
            score = group_logits(logits)[0, target]
            a = activations[0]
            g = torch.autograd.grad(score, a)[0]
            cam = (g.mean(dim=(2,3),keepdim=True)*a).sum(1,keepdim=True).relu()
            cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0,0]
            peak = float(cam.detach().max())
            cam = cam / peak if peak > 0 else torch.zeros_like(cam)
            return cam.detach().cpu().numpy(), peak == 0
    finally:
        handle.remove()


def save_gallery(df, root, preprocess, out, correct):
    subset = df[df.correct == correct].head(6)
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    for ax in axes.ravel(): ax.axis('off')
    if subset.empty:
        axes[0,0].text(.05,.5,'No examples in this subset',fontsize=14)
    for ax, (_, row) in zip(axes.ravel(), subset.iterrows()):
        ax.imshow(input_rgb(preprocess(load_rgb(root / row.file))))
        ax.set_title(f'True: {row.true_label} | Pred: {row.predicted_label}\n'
                     f'Cat {row.cat_score:.3f} / Dog {row.dog_score:.3f}', fontsize=11)
    fig.suptitle('Correct predictions' if correct else 'Incorrect predictions',fontsize=20)
    fig.tight_layout()
    fig.savefig(out / ('correct_predictions.png' if correct else 'incorrect_predictions.png'),dpi=150)
    plt.close(fig)


def save_cam_examples(df, root, model, preprocess, device, out):
    # Fixed selection rule: first two correct per true class, then first two errors.
    choices = pd.concat([df[df.correct & (df.true_label == c)].head(2) for c in NAMES]
                        + [df[~df.correct].head(2)]).drop_duplicates('file')
    manifest = []
    for n, (_, row) in enumerate(choices.iterrows(), 1):
        original = load_rgb(root / row.file)
        x = preprocess(original)
        rgb = input_rgb(x)
        target = NAMES.index(row.predicted_label)
        cam, blank = gradcam(model, x.unsqueeze(0).to(device), target)
        overlay = np.clip(.6*rgb + .4*plt.get_cmap('jet')(cam)[...,:3],0,1)
        fig, axes = plt.subplots(1,4,figsize=(14,3.8))
        for ax, im, title in zip(axes, [original,rgb,cam,overlay],
                               ['Original image','Model input crop','Grad-CAM','Overlay on input crop']):
            if title == 'Grad-CAM':
                ax.imshow(im,cmap='jet',vmin=0,vmax=1)
            else: ax.imshow(im)
            ax.set_title(title); ax.axis('off')
        fig.suptitle(f'{row.file} | True: {row.true_label} | Predicted: {row.predicted_label}\n'
                     f'Target: grouped {row.predicted_label} score | '
                     f'Conditional score: {max(row.cat_score,row.dog_score):.3f}'
                     + (' | Zero positive activation' if blank else ''),fontsize=11)
        fig.tight_layout(rect=[0,0,1,.86])
        name = f'gradcam_{n:02d}.png'
        fig.savefig(out/name,dpi=160); plt.close(fig)
        manifest.append({'image':row.file,'figure':name,'true':row.true_label,
                         'predicted':row.predicted_label,'zero_map':blank})
    (out/'gradcam_manifest.json').write_text(json.dumps(manifest,indent=2))


def run(root, out='outputs', per_class=200, seed=SEED, batch_size=32):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if per_class < 1 or batch_size < 1: raise ValueError('Counts must be positive.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selected, skipped, inventory = select_images(root,per_class,seed)
    model, preprocess, categories = build_model(device)
    df = predict(model,preprocess,categories,selected,root,device,batch_size)
    df.to_csv(out/'predictions.csv',index=False)
    (out/'skipped_files.json').write_text(json.dumps(skipped,indent=2))
    cm = confusion_matrix(df.true_label,df.predicted_label,labels=NAMES)
    report = classification_report(df.true_label,df.predicted_label,labels=NAMES,output_dict=True,zero_division=0)
    summary = {'dataset':'yashdogra/cats-and-dogs/versions/1', 'seed':seed,
               'per_class':per_class,'total':len(df),'correct':int(df.correct.sum()),
               'incorrect':int((~df.correct).sum()),'accuracy':float(df.correct.mean()),
               'model':'resnet18','weights':'IMAGENET1K_V1','device':str(device),
               'torch':torch.__version__,'torchvision':torchvision.__version__,
               'python':platform.python_version(),'inventory':inventory,
               'confusion_matrix':cm.tolist(),'classification_report':report}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    (out/'class_mapping.json').write_text(json.dumps({NAMES[i]:{str(k):categories[k] for k in ids}
                                                     for i,ids in enumerate(GROUPS)},indent=2))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm,cmap='Blues');ax.set_xticks([0,1],NAMES);ax.set_yticks([0,1],NAMES)
    ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'Confusion matrix | Accuracy {df.correct.mean():.2%}')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=22,color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    save_gallery(df,root,preprocess,out,True);save_gallery(df,root,preprocess,out,False)
    save_cam_examples(df,root,model,preprocess,device,out)
    print(json.dumps(summary,indent=2))
    return df, summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--out',default='outputs')
    parser.add_argument('--per-class',type=int,default=200);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args()
    run(args.data,args.out,args.per_class,args.seed,args.batch_size)
