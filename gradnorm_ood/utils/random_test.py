
from test_utils import get_measures
import numpy as np

# in_res = -np.random.random((50000,1))*0.5
# out_res = -np.random.random((5000,1))
in_res = np.random.random((50000,1))
out_res = np.random.random((5000,1))
auroc, aupr_in, aupr_out, fpr95 = get_measures(in_res, out_res)

print('AUROC: {}'.format(auroc))
print('AUPR (In): {}'.format(aupr_in))
print('AUPR (Out): {}'.format(aupr_out))
print('FPR95: {}'.format(fpr95))
