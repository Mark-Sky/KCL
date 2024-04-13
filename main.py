import random
import argparse
import yaml
from utils import *
from dataload import CustomDataload, ImagenetDataload
from model.TipAdapter import TipAdapter, KCLTipAdapter
from model.CoOp import CoOp, KCLCoOp
from model.Clip import Clip, KCLClip
from model.TipAdapterF import TipAdapterF, KCLTipAdapterF
from model.ClipAdapter import CLipAdapter, KCLClipAdapter
from model.Maple import Maple, KCLMaple


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--model', dest='model')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    cfg['cache_dir'] = cache_dir

    if args.shots:
        cfg['shots'] = args.shots
        print('******************** shots = %d *************************' % args.shots)
    print('******************** dataset = %s *************************' % cfg['dataset'])

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # Prepare dataset
    random.seed(0)
    torch.manual_seed(0)
    if cfg['dataset'] == 'imagenet':
        data = ImagenetDataload(cfg, clip_model, preprocess)
        # pass
    else:
        data = CustomDataload(cfg, clip_model, preprocess)

    clip_classifier(cfg, data.dataset.classnames, data.dataset.template, clip_model)

    '''
    Training Free: Tip-Adapter, APE
    '''
    if args.model == 'Clip':
        model = Clip(cfg, clip_model)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'KCLClip':
        model = KCLClip(cfg, clip_model)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'TipAdapter':
        model = TipAdapter(cfg, clip_model)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'KCLTipAdapter':
        model = KCLTipAdapter(cfg, clip_model)
        model.evaluate(data.test_features, data.test_labels)
        model.save_pse_cache()

    '''
    Training: Tip-Adapter-F, CoOp, Clip-Adapter
    '''
    if args.model == 'TipAdapterF':
        model = TipAdapterF(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'KCLTipAdapterF':
        model = KCLTipAdapterF(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'CoOp':
        model = CoOp(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'KCLCoOp':
        model = KCLCoOp(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'ClipAdapter':
        model = CLipAdapter(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'KCLClipAdapter':
        model = KCLClipAdapter(cfg, clip_model)
        model.train(data.test_features, data.test_labels, data.train_loader_F)
        model.evaluate(data.test_features, data.test_labels)

    if args.model == 'Maple':
        model = Maple(cfg, clip_model)
        model.evaluate(model.test_features, model.test_labels)

    if args.model == 'KCLMaple':
        model = KCLMaple(cfg, clip_model)
        model.evaluate(model.test_features, model.test_labels)
