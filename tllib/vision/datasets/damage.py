"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Damage(ImageList):
    """Damage Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'E'``: ecuador_eq, \
            ``'M'``: matthew_hurricane, ``'N'``: nepal_eq and ``'R'``: ruby_typhoon.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again. (defualts false) 
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            ecuador_eq/
                images/
                    no_damage/
                        *.jpg
                        ...
            nepal_eq/
            matthew_hurricane/
            ruby_typhoon/
            image_list/
                ecuador_eq.txt
                nepal_eq.txt
                matthew_hurricane.txt
                ruby_typhoon.txt
    """
    # under image_list/ecuador_eq.txt each line will follow ecuador_eq/images/{class}/{image_name}.jpg {label}
    download_list = []
    
    # For training and evaluating model and A-distance larger sample
    # image_list = {
    #     "E": "image_list/ecuador_eq.txt",
    #     "N": "image_list/nepal_eq.txt",
    #     "M": "image_list/matthew_hurricane.txt",
    #     "R": "image_list/ruby_typhoon.txt"
    # }

    # For A-distance calculation
    image_list = {
        "E": "sample_image_list/ecuador_eq.txt",
        "N": "sample_image_list/nepal_eq.txt",
        "M": "sample_image_list/matthew_hurricane.txt",
        "R": "sample_image_list/ruby_typhoon.txt"
    }

    CLASSES = ['no_damage', 'damage']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Damage, self).__init__(root, Damage.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())