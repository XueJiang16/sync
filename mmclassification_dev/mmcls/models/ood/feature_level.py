from mmcv.runner import BaseModule    # noqa
import torch
import os
import numpy as np  # noqa
from collections import Counter  # noqa

from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model    # noqa


def no_ood_detector(**kwargs):
    raise RuntimeError("No Feature-level OOD Detector Configured!")


target_channel = [652, 629, 1761, 745, 1502, 886, 240, 675, 1778, 659, 1996, 1066, 1279, 1114, 312, 1080, 1457, 1747, 359, 1444, 1664, 214, 2030, 36,
1984, 956, 1745, 268, 1323, 714, 433, 1384, 461, 1429, 735, 160, 221, 439, 943, 1766, 1475, 371, 1574, 1851, 1608, 1365, 1339, 1206, 679,
55, 574, 1863, 513, 660, 1756, 1536, 1317, 1086, 323, 138, 504, 1019, 1689, 349, 288, 1034, 1577, 1897, 1110, 1767, 1805, 423, 1045, 908,
1807, 731, 938, 1576, 1002, 1914, 1654, 1803, 550, 1173, 258, 1488, 1783, 470, 1790, 534, 124, 1516, 1227, 730, 1958, 377, 1099, 26, 673,
743, 1340, 1199, 502, 958, 1635, 590, 1831, 1262, 1148, 1721, 1195, 295, 509, 1188, 66, 426, 840, 529, 847, 303, 685, 226, 1828, 1249,
1489, 1482, 1252, 760, 864, 997, 197, 1147, 1447, 1763, 1987, 2033, 692, 1496, 1655, 1441, 1369, 432, 320, 114, 1253, 1186, 460, 145, 1337,
1583, 727, 272, 863, 610, 92, 940, 1892, 1498, 1975, 1394, 1362, 762, 1272, 799, 1479, 1078, 746, 1328, 916, 28, 1028, 317, 511, 506,
1570, 137, 58, 670, 291, 552, 1499, 1844, 1822, 858, 884, 1691, 821, 468, 854, 548, 2046, 1157, 1347, 1591, 1380, 1177, 912, 1872, 186,
1549, 1529, 163, 1667, 1311, 1124, 1990, 207, 1486, 1003, 1835, 1508, 613, 1594, 852, 2010, 978, 1069, 1674, 868, 930, 256, 472, 2042, 1011,
1994, 1216, 1057, 1354, 525, 1167, 171, 2025, 1566, 846, 269, 921, 1816, 101, 51, 41, 723, 1277, 892, 1531, 1911, 1000, 1281, 1981, 575,
1506, 1372, 499, 1334, 1393, 1235, 708, 628, 1201, 620, 656, 1100, 335, 752, 246, 1276, 1383, 859, 998, 35, 1933, 2009, 1724, 1645, 215,
881, 865, 99, 1715, 1133, 1172, 166, 500, 910, 1168, 1454, 290, 1200, 1442, 1614, 15, 856, 733, 73, 1543, 1853, 953, 1557, 252, 257,
1995, 47, 1883, 584, 1309, 1647, 754, 1503, 1403, 1046, 1084, 1682, 591, 639, 817, 585, 261, 993, 1748, 414, 1151, 1795, 725, 520, 702,
1718, 1884, 1985, 1907, 586, 1010, 972, 1562, 1686, 1514, 819, 1757, 1287, 1130, 187, 592, 512, 1246, 1043, 1251, 1538, 1209, 1266, 1278, 636,
407, 1379, 1120, 93, 1997, 922, 770, 1725, 1320, 1283, 1452, 2, 1722, 703, 1298, 1058, 1633, 25, 1762, 1467, 1414, 2028, 1860, 597, 984,
193, 22, 1861, 1238, 1314, 1799, 1877, 4, 724, 404, 963, 1044, 1601, 519, 1634, 1770, 1356, 1620, 1313, 232, 905, 394, 1673, 877, 538,
1966, 1611, 1786, 2007, 1225, 773, 195, 1552, 947, 1268, 1504, 340, 970, 478, 530, 932, 570, 543, 937, 1575, 84, 1900, 209, 1440, 1032,
247, 766, 1652, 1381, 909, 1743, 45, 1007, 1426, 328, 1345, 732, 420, 1228, 1463, 497, 994, 72, 365, 598, 466, 737, 1649, 1597, 1613,
1392, 227, 1407, 402, 782, 1988, 977, 1774, 514, 558, 1751, 1595, 1450, 593, 2001, 321, 336, 1434, 412, 885, 1604, 1528, 1013, 1826, 604,
167, 130, 198, 276, 547, 1458, 1232, 1348, 1202, 1929, 515, 1841, 1338, 242, 1777, 524, 21, 1375, 435, 809, 143, 791, 6, 1915, 1729,
369, 1089, 96, 1160, 1327, 964, 559, 274, 1460, 139, 158, 633, 1607, 196, 928, 1873, 1242, 1109, 1612, 961, 1190, 353, 1473, 162, 1523,
1637, 495, 1972, 176, 1336, 413, 694, 49, 996, 726, 1956, 1970, 1358, 634, 70, 1257, 1127, 918, 1798, 1622, 200, 967, 528, 635, 631,
1049, 1485, 781, 250, 1605, 1223, 1408, 1210, 1073, 792, 1946, 907, 565, 1692, 1126, 2000, 1385, 1424, 612, 736, 1259, 980, 1158, 1145, 1824,
1585, 311, 1497, 1773, 1843, 919, 851, 2035, 1671, 1248, 302, 1192, 222, 1154, 1150, 1416, 1097, 699, 285, 1830, 305, 379, 1437, 1755, 1474,
668, 1284, 1712, 382, 322, 179, 1396, 765, 1660, 653, 111, 1331, 77, 95, 966, 695, 1854, 1639, 1796, 110, 1675, 1749, 926, 1197, 1267,
1180, 1349, 14, 697, 987, 79, 1213, 1074, 1858, 1346, 1734, 201, 1095, 1952, 516, 140, 430, 327, 1237, 1801, 360, 804, 1449, 1122, 39,
1571, 1776, 825, 1121, 1368, 1738, 1108, 1849, 1868, 1364, 1215, 1035, 1153, 753, 1954, 1491, 411, 16, 189, 1132, 823, 1070, 767, 1355, 1219,
1606, 12, 873, 383, 1642, 1937, 1451, 786, 1087, 934, 1986, 1144, 985, 319, 356, 1908, 948, 52, 463, 1695, 123, 1889, 1819, 571, 183,
476, 1781, 1226, 248, 346, 600, 761, 157, 54, 828, 1020, 1701, 683, 657, 1610, 1581, 772, 1377, 474, 62, 1794, 1391, 451, 409, 1742,
1885, 1814, 1750, 1388, 839, 1717, 1993, 53, 1663, 810, 882, 1402, 632, 674, 1650, 1624, 1016, 1668, 1052, 1091, 768, 893, 588, 182, 1832,
1465, 120, 546, 1304, 783, 7, 301, 744, 1406, 86, 1524, 1470, 1116, 1191, 1546, 150, 1544, 589, 1752, 1760, 355, 396, 465, 988, 1657,
490, 2017, 13, 824, 380, 774, 531, 944, 806, 1753, 1229, 1033, 1676, 293, 849, 449, 1823, 1421, 1301, 74, 929, 1481, 522, 878, 712,
1075, 245, 1302, 1077, 1351, 2032, 368, 816, 88, 1603, 203, 1134, 5, 1703, 949, 1117, 637, 1428, 690, 1422, 1, 1708, 2037, 24, 1852,
1282, 1913, 1561, 1510, 1243, 485, 280, 583, 790, 1143, 1118, 507, 1706, 475, 479, 1079, 503, 1207, 1006, 764, 1411, 134, 1836, 243, 2034,
155, 1245, 313, 1165, 1890, 606, 576, 706, 1771, 2004, 989, 501, 1492, 212, 1842, 370, 484, 711, 228, 1741, 89, 1174, 969, 389, 278,
860, 1893, 867, 128, 1179, 1404, 1212, 1254, 1425, 119, 831, 85, 1093, 1333, 1469, 1864, 496, 1234, 1628, 1500, 1297, 434, 1905, 936, 1941,
343, 1730, 1629, 1839, 952, 898, 462, 1430, 572, 1588, 942, 473, 1417, 393, 526, 902, 1061, 924, 133, 1918, 1678, 1462, 331, 1420, 97,
951, 1903, 1082, 1062, 223, 458, 875, 1846, 556, 1930, 1643, 149, 1541, 1540, 561, 879, 152, 135, 569, 1307, 623, 1198, 1602, 1312, 655,
1600, 170, 1455, 1273, 931, 129, 1866, 945, 1681, 843, 1532, 289, 174, 1959, 1662, 1711, 427, 205, 1439, 237, 1236, 345, 1847, 959, 580,
1131, 279, 895, 1185, 1023, 1050, 1935, 1322, 173, 1732, 1784, 300, 599, 1797, 624, 1048, 1324, 1051, 1587, 874, 1231, 34, 1698, 267, 1196,
82, 1189, 1800, 1096, 1415, 388, 65, 805, 2018, 564, 925, 329, 1111, 1556, 431, 758, 667, 687, 1501, 1886, 663, 666, 617, 1106, 1064, ]


@OOD.register_module()
class PatchSim(BaseModule):
    def __init__(self, num_crop, img_size, threshold, order=1, ood_detector=None, mode='cosine',**kwargs):
        super(PatchSim, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.num_crop = num_crop
        self.img_size = img_size
        self.threshold = threshold
        self.order = order
        self.mode = mode

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            img = input['img']
            img_size = self.img_size
            crop_size = int(img_size / self.num_crop)
            corner_list = []
            crops = []
            for h in range(self.num_crop):
                for w in range(self.num_crop):
                    corner_list.append([h * crop_size, w * crop_size])
            for h, w in corner_list:
                crop = img[:, :, h: h + crop_size, w: w + crop_size]
                input['img'] = crop
                _, crop_feature = self.ood_detector.classifier(return_loss=False, softmax=False,
                                                               post_process=False, require_features=True, **input)
                crops.append(crop_feature)
            input['img'] = img
            input['type'] = type
            patch_sim = 0
            count = 0
            for i in range(len(crops)-1):
                for j in range(i+1, len(crops)):
                    if self.mode == 'cosine':
                        tmp = - (crops[i] * crops[j]).sum(dim=1)
                        tmp = tmp / (torch.norm(crops[i], dim=1) * torch.norm(crops[j], dim=1))
                        tmp = (tmp + 1) / 2
                    elif self.mode == 'euclidean':
                        tmp = torch.norm(crops[i]-crops[j], dim=1)
                    patch_sim += tmp
                    count += 1
            patch_sim /= count
            # ood_scores = patch_sim
            if self.has_ood_detector:
                ood_scores, _ = self.ood_detector(**input)
                patch_sim = ((1 / self.threshold) ** (self.order)) * torch.pow(patch_sim, self.order)
                patch_sim[patch_sim > 1] = 1
                ood_scores *= patch_sim
            else:
                ood_scores = patch_sim
        return ood_scores, type


@OOD.register_module()
class FeatureMapSim(BaseModule):
    def __init__(self, num_crop, img_size, threshold, order=1, ood_detector=None, mode='cosine',**kwargs):
        super(FeatureMapSim, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.num_crop = num_crop
        self.img_size = img_size
        self.threshold = threshold
        self.order = order
        self.mode = mode

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            _, feature_c5 = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
                                                         require_backbone_features=True, **input)
            input['type'] = type
            if self.mode in ['cosine', 'euclidean']:
                feature_crops = torch.nn.functional.interpolate(feature_c5, size=self.num_crop, mode='bilinear')
                feature_crops = feature_crops.flatten(2)
                patch_sim = 0
                count = 0
                for i in range(self.num_crop**2-1):
                    for j in range(i+1, self.num_crop**2):
                        if self.mode == 'cosine':
                            tmp = - (feature_crops[:, :, i] * feature_crops[:, :, j]).sum(dim=1)
                            tmp = tmp / (torch.norm(feature_crops[:, :, i], dim=1) *
                                         torch.norm(feature_crops[:, :, j], dim=1))
                            tmp = (tmp + 1) / 2
                        elif self.mode == 'euclidean':
                            tmp = torch.norm(feature_crops[:, :, i]-feature_crops[:, :, j], dim=1)
                        patch_sim += tmp
                        count += 1
                patch_sim /= count
                # ood_scores = patch_sim
            elif self.mode == 'std':
                # feature_c5 = feature_c5[:,:,1:6,1:6]
                feature_crops = feature_c5.flatten(2)  # (B, C, H*W)
                patch_sim = feature_crops.std(-1).mean(-1)
            elif self.mode == 'mean':
                # feature_c5 = feature_c5[:,:,1:6,1:6]
                feature_crops = feature_c5.flatten(2)
                feature_crops = feature_crops[:, target_channel]
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'extract_feature_sim':
                # (N, C, H*W) -> (N, C) -> (C,) -> argsort -> topK_idx -> id_ood_inference -> feature_crops[:, topK_idx]
                feature_crops = feature_c5.flatten(2)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C, 1)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(0, 2))  # for ID: .mean(dim=(0, 2))

            ood_scores = patch_sim
        # if self.has_ood_detector:
        #     ood_scores, _ = self.ood_detector(**input)
        #     patch_sim = ((1 / self.threshold) ** (self.order)) * torch.pow(patch_sim, self.order)
        #     patch_sim[patch_sim > 1] = 1
        #     ood_scores *= patch_sim
        # else:
        #     ood_scores = patch_sim
        return ood_scores, type
