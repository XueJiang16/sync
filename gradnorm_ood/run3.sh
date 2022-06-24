# pip install scikit-learn

# CUDA_VISIBLE_DEVICES=0,1,2,3 


# list_method=( 'MSP' 'ODIN' 'Energy' 'new' 'GradNorm' 'Mahalanobis')
list_method=('new')

list_dataset=('SUN' 'Places' 'Textures' 'iNaturalist')
# # list_ckpt=('mobile_LT_a8/epoch_600' 'resnet152_LT_a8/epoch_100' 'resnet50_LT_a8/epoch_100')
for method in ${list_method[*]}
do
for dataset in ${list_dataset[*]}
do
# for i in {3,6,9}
# do
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth \
#     checkpoint0501/imagnet10percent/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
#     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/inat/epoch_100.pth \
#     checkpoint0512/inat_res101_plus2/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 0 \
#bash ./custom_test_v3_.sh $method $dataset ./ood_ckpt/ckpt/mobile_LT_a8/epoch_600.pth \
#    test/mobile/$method ./meta/train_LT_a8.txt 8 \
bash ./custom_test_v3_.sh $method $dataset ./ood_ckpt/ood_ckpt_other/LT_a8/epoch_100.pth \
    checkpoint0624/LT_a8_kl/$method ./meta/train_LT_a8.txt 8
# done
done
done

# for i in 3
# do  
# ./custom_test_v3_.sh GradNorm SUN /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoints/test /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0 
# # ./custom_test_v3_.sh GradNorm Places /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/gradnorm_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0
# # ./custom_test_v3_.sh GradNorm Textures /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/gradnorm_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0
# # ./custom_test_v3_.sh GradNorm iNaturalist /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/gradnorm_a7_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a7.txt 0
# done



# ./custom_test_v3_.sh Mahalanobis SUN /mapai/haowenguo/ckpt/ood_ckpt/ckpt/ \
#     checkpoint_baseline/Mahalanobis/test /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
#     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/baseline/tune_mahalanobis
# ./custom_test_v3_.sh Mahalanobis Places /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth \
#  checkpoint_baseline/Mahalanobis/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
#   --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/baseline/tune_mahalanobis
# ./custom_test_v3_.sh Mahalanobis Textures /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth \
#  checkpoint_baseline/Mahalanobis/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
#   --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/baseline/tune_mahalanobis
# ./custom_test_v3_.sh Mahalanobis iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth \
#  checkpoint_baseline/Mahalanobis/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
#   --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/baseline/tune_mahalanobis



# for i in 3
# do  
# ./custom_test_v3.sh new SUN /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/new_a8_repeat${i}_test /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a8.txt 0 
# ./custom_test_v3.sh new Places /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/new_a8_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a8.txt 0
# ./custom_test_v3.sh new Textures /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/new_a8_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a8.txt 0
# ./custom_test_v3.sh new iNaturalist /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth checkpoint0408/new_a8_repeat${i} /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a8.txt 0
# done
# ./custom_test_v3.sh MSP SUN /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth checkpoint0415/test/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0
# ./custom_test_v3_.sh MSP Places /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth checkpoint_baseline/MSP/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0
# ./custom_test_v3_.sh MSP Textures /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth checkpoint_baseline/MSP/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0
# ./custom_test_v3_.sh MSP iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth checkpoint_baseline/MSP/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0

# for i in {2..8}
# do  
# ./custom_test_v3.sh MSP SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0417/MSP_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3.sh MSP Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0417/MSP_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3.sh MSP Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0417/MSP_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

#  ./custom_test_v3.sh MSP iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0417/MSP_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 
# done

# for i in {2..8}
# do  
# ./custom_test_v3_.sh ODIN SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3_.sh ODIN Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3_.sh ODIN Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

#  ./custom_test_v3_.sh ODIN iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_pluscos_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 
# done

# for i in {2..8}
# do  
# ./custom_test_v3.sh ODIN SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3.sh ODIN Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3.sh ODIN Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

#  ./custom_test_v3.sh ODIN iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0418/ODIN_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 
# done



# for i in 3
# do  
# # ./custom_test_v3.sh MSP SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
# #  checkpoint0415/MSP_plus_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# ./custom_test_v3_.sh MSP Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoint0414/MSP_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# # ./custom_test_v3.sh MSP Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
# #  checkpoint0415/MSP_plus_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 

# #  ./custom_test_v3.sh MSP iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
# #  checkpoint0415/MSP_plus_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 
# done

# for i in {2..8} 
# do  
# ./custom_test_v3.sh Energy SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoints0417/Energy_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 
# ./custom_test_v3.sh Energy Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoints0417/Energy_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i
# ./custom_test_v3.sh Energy Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoints0417/Energy_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i
# ./custom_test_v3.sh Energy iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#  checkpoints0417/Energy_plusabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i
# done

# for i in 3
# do  
# ./custom_test_v3_.sh Energy SUN /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#   checkpoints0417/Energy_plus_noabs_a${i}_$i_test /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i 
# ./custom_test_v3_.sh Energy Places /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#   checkpoints0417/Energy_plus_noabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i
# ./custom_test_v3_.sh Energy Textures /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#   checkpoints0417/Energy_plus_noabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i
# ./custom_test_v3_.sh Energy iNaturalist /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#   checkpoints0417/Energy_plus_noabs_a${i}_$i /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a$i.txt $i
# done

