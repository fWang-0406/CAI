# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import argparse


class DirectionTree(object):
    """生成目录树
    @ pathname: 目标目录
    @ filename: 要保存成文件的名称
    """

    def __init__(self, pathname='.', filename='tree.txt'):
        super(DirectionTree, self).__init__()
        self.pathname = Path(pathname)
        self.filename = filename
        self.tree = ''

    def set_path(self, pathname):
        self.pathname = Path(pathname)

    def set_filename(self, filename):
        self.filename = filename

    def generate_tree(self, prefix=''):
        if self.pathname.is_file():
            self.tree += prefix + '└── ' + self.pathname.name + '  \n'  # 行末添加两个空格
        elif self.pathname.is_dir():
            # 如果目录名以特定前缀开头，则不展开
            skip_dirs = ['imgs', 'pretrain_models', '.idea', 'original_data', 'data', 'checkpoints', 'models']
            if any(self.pathname.name.startswith(s) for s in skip_dirs):
                self.tree += prefix + '└── ' + self.pathname.name + '\\' + ' (未展开)' + '  \n'  # 行末添加两个空格
            else:
                self.tree += prefix + '└── ' + self.pathname.name + '\\' + '  \n'  # 行末添加两个空格
                children = list(self.pathname.iterdir())
                for i, cp in enumerate(children):
                    if i == len(children) - 1:
                        new_prefix = prefix + '    '
                    else:
                        new_prefix = prefix + '│   '
                    self.pathname = Path(cp)
                    self.generate_tree(new_prefix)

    def save_file(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.tree)


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="生成指定文件夹的目录树并保存到文件")
    # 添加参数
    parser.add_argument('-p', '--path', type=str, default='.', help='目标目录路径，默认为当前目录')
    parser.add_argument('-o', '--output', type=str, default='file_structure.txt', help='输出文件路径，默认为 file_structure.txt')
    # 解析参数
    args = parser.parse_args()

    # 检查路径是否存在
    if not Path(args.path).exists():
        print(f"路径 '{args.path}' 不存在！")
        sys.exit(1)

    # 生成目录树并保存到文件
    dirtree = DirectionTree(args.path, args.output)
    dirtree.generate_tree()
    print(dirtree.tree)
    dirtree.save_file()
    print(f"目录树已保存到文件: {args.output}")


if __name__ == '__main__':
    main()