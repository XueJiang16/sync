# pip install scikit-learn
# ./scripts/test.sh GradNorm SUN
# ./scripts/test.sh GradNorm Places
# ./scripts/test.sh GradNorm Textures
# ./scripts/test.sh GradNorm iNaturalist

#!/bin/bash  

 ./custom_test_v4.sh new SUN ood_ckpt/ckpt/inat/epoch_100.pth inat_test/ ./meta/train_LT_a8.txt 0
# ./custom_test_v4.sh new Places /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet_LT_100e.pth new0408_v4/imagenet_LT /mapai/haowenguo/data/ood_data/ImageNet-LT/ImageNet_LT_train.txt 0
# ./custom_test_v4.sh new Textures /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet_LT_100e.pth new0408_v4/imagenet_LT /mapai/haowenguo/data/ood_data/ImageNet-LT/ImageNet_LT_train.txt 0
# ./custom_test_v4.sh new iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet_LT_100e.pth new0408_v4/imagenet_LT /mapai/haowenguo/data/ood_data/ImageNet-LT/ImageNet_LT_train.txt 0

# for i in {3,6,9}  
# do  
# ./custom_test_v4.sh new SUN /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a7/epoch_100.pth checkpoint0408_v4/new_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0 
# ./custom_test_v4.sh new Places /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a7/epoch_100.pth checkpoint0408_v4/new_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0
# ./custom_test_v4.sh new Textures /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a7/epoch_100.pth checkpoint0408_v4/new_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0
# ./custom_test_v4.sh new iNaturalist /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a7/epoch_100.pth checkpoint0408_v4/new_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0
# done
#for i in {2..8}
#do
#./scripts/tune_mahalanobis.sh /mapai/haowenguo/ckpt/ood_ckpt/LT_a$i/epoch_100.pth ckpt_mahalanobis/LT_a$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt
#done
# for i in {101..110}  
# do  
# ./custom_test.sh GradNorm SUN /mapai/haowenguo/ckpt/ood_ckpt/epoch_$i.pth checkpoint/epoch_$i
# ./custom_test.sh GradNorm Places /mapai/haowenguo/ckpt/ood_ckpt/epoch_$i.pth checkpoint/epoch_$i
# ./custom_test.sh GradNorm Textures /mapai/haowenguo/ckpt/ood_ckpt/epoch_$i.pth checkpoint/epoch_$i
# ./custom_test.sh GradNorm iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/epoch_$i.pth checkpoint/epoch_$i
# done

# for i in {2,3,4,5,6,7,8}  
# do  
# # ./custom_test_v2.sh GradNorm SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a$i/epoch_100.pth checkpoint/test /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt
# # ./custom_test_v2.sh GradNorm Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a$i/epoch_100.pth checkpoint/LT_a$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt
# # ./custom_test_v2.sh GradNorm Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a$i/epoch_100.pth checkpoint/LT_a$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt
# ./custom_test_v2.sh GradNorm iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a$i/epoch_100.pth checkpoint0324/LT_a$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/val_labeled.txt /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt
# done
# ./custom_test.sh GradNorm iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a8/epoch_100.pth checkpoint/test
# ./custom_test_v2.sh GradNorm iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a4/epoch_100.pth checkpoint/test /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/val_LT_a4_counter.txt
