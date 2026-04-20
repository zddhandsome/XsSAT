#!/usr/bin/env python3
"""
检查 test.pt 文件中 SAT/UNSAT 的比例
"""
import sys
import torch

class OrderedData:
    """Placeholder for utils.dataset_utils.OrderedData"""
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

def check_sat_ratio(filepath):
    print(f"Loading {filepath}...\n")

    try:
        # 创建虚拟的 utils 模块
        class FakeModule:
            def __getattr__(self, name):
                if name == 'OrderedData':
                    return OrderedData
                return type('FakeClass', (), {})

        sys.modules['utils'] = FakeModule()
        sys.modules['utils.dataset_utils'] = FakeModule()

        # 现在可以加载文件
        data = torch.load(filepath, weights_only=False)

        if isinstance(data, tuple):
            ordered_data = data[0]
            data_dict = data[1]

            # 获取 SAT/UNSAT 标签
            sat_labels = None

            # 优先从 OrderedData._store 中获取 'y'
            if hasattr(ordered_data, '_store') and 'y' in ordered_data._store:
                sat_labels = ordered_data._store['y']
                print("Found SAT/UNSAT label in OrderedData._store['y']")
            elif 'y' in data_dict:
                sat_labels = data_dict['y']
                print("Found SAT/UNSAT label in data_dict['y']")

            if sat_labels is None:
                print("Error: Could not find SAT/UNSAT label")
                return

            print(f"Label shape: {sat_labels.shape}")
            print(f"Label dtype: {sat_labels.dtype}\n")

            # 检查是否是二分类
            unique_vals = torch.unique(sat_labels).tolist()
            if set(unique_vals) != {0, 1}:
                print(f"Warning: Expected binary labels (0/1), but found: {sorted(unique_vals)}")
                return

            # 计算比例
            total = len(sat_labels)
            sat_count = (sat_labels == 1).sum().item()
            unsat_count = (sat_labels == 0).sum().item()

            sat_ratio = sat_count / total * 100
            unsat_ratio = unsat_count / total * 100

            print(f"{'='*50}")
            print(f"数据集: {filepath}")
            print(f"{'='*50}")
            print(f"总样本数:      {total}")
            print(f"SAT 样本:      {sat_count:5d} ({sat_ratio:6.2f}%)")
            print(f"UNSAT 样本:    {unsat_count:5d} ({unsat_ratio:6.2f}%)")
            print(f"{'='*50}\n")

            # 显示标签分布
            unique, counts = torch.unique(sat_labels, return_counts=True)
            print(f"标签分布:")
            for label, count in zip(unique.tolist(), counts.tolist()):
                label_name = "SAT" if label == 1 else "UNSAT"
                ratio = count / total * 100
                print(f"  {label_name} (label={int(label)}): {count:5d} samples ({ratio:6.2f}%)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    filepath = 'data_test_cv3/test.pt'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    check_sat_ratio(filepath)
