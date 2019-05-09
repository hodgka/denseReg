class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'msra':
            return '/u/big/trainingdata/MSRA/cvpr15_MSRAHandGestureDB/'  # folder that contains VOCdevkit/.
        elif dataset == 'nyu':
            return '/u/big/trainingdata/NYUHANDS/dataset/'     # foler that contains leftImg8bit/
        elif dataset == 'icvl':
            return '/u/big/trainingdata/ICVL/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
