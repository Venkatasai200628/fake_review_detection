# Leave-one-category-out: performance on a product type never seen in training

Each row: the whole category is removed from training -- every photo, review and campaign -- the model is trained on the other 14, and tested only on the held-out one.

| held-out category | rows | accuracy | F1 | MCC | ROC-AUC | image-only ROC | text-only ROC | edit-CNN ROC |
|---|---|---|---|---|---|---|---|---|
| Bluetooth speaker | 225 | 0.898 | 0.893 | 0.796 | 0.958 | 0.826 | 0.841 | 0.880 |
| Handbags | 224 | 0.924 | 0.917 | 0.856 | 0.967 | 0.855 | 0.815 | 0.908 |
| Knifes | 222 | 0.910 | 0.900 | 0.833 | 0.971 | 0.840 | 0.836 | 0.872 |
| Office cair | 222 | 0.829 | 0.826 | 0.658 | 0.908 | 0.805 | 0.798 | 0.785 |
| Phone case | 222 | 0.869 | 0.859 | 0.745 | 0.925 | 0.806 | 0.816 | 0.873 |
| Shoes | 222 | 0.874 | 0.856 | 0.769 | 0.927 | 0.828 | 0.815 | 0.833 |
| Sunglasses | 223 | 0.883 | 0.867 | 0.785 | 0.952 | 0.799 | 0.800 | 0.944 |
| Watches | 224 | 0.902 | 0.890 | 0.816 | 0.958 | 0.849 | 0.805 | 0.923 |
| Water bottle | 224 | 0.915 | 0.905 | 0.842 | 0.969 | 0.835 | 0.836 | 0.936 |
| Wireless_Earbuds | 224 | 0.920 | 0.912 | 0.848 | 0.988 | 0.867 | 0.808 | 0.963 |
| Yoga mat | 226 | 0.867 | 0.850 | 0.745 | 0.925 | 0.813 | 0.801 | 0.848 |
| backpack | 222 | 0.901 | 0.892 | 0.810 | 0.957 | 0.850 | 0.780 | 0.908 |
| mouse | 223 | 0.892 | 0.878 | 0.803 | 0.979 | 0.859 | 0.807 | 0.971 |
| skincare set | 222 | 0.878 | 0.872 | 0.759 | 0.934 | 0.817 | 0.832 | 0.820 |
| suitcases | 223 | 0.879 | 0.864 | 0.771 | 0.933 | 0.803 | 0.816 | 0.844 |
| **mean** | | **0.889** | **0.879** | **0.789** | **0.950** | **0.830** | **0.814** | **0.887** |

Worst category: Office cair (MCC 0.658); best: Handbags (MCC 0.856).