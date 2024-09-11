# Leveraging Point Cloud Completion for Point Cloud-based Place Recognition

## Motivation
In this experiment, we aim to evaluate the influence of point cloud completion on place recognition qualitatively and quantitatively.

## Methods
* Semantic Scene Completion: [JS3C-Net](https://arxiv.org/abs/2012.03762)
* Place Recognition: [PointNetVLAD](https://arxiv.org/abs/1804.03492)
## Datasets
Download the SemanticKITTI datasets: http://www.semantic-kitti.org/dataset.html 

Sequences 00, 02, 05, 06, 07, 08 are used as test sequences for place recognition.

The location of the datasets on the students' GPU server: `/mnt/data4/Students/Xiaoang/Completed_PointCloud/full_completed_only_set/dataset` \
There're 4 types of datasets under this directory:
1) `/dataset`: raw point clouds $\textbf{without}$ point-wise labels
2) `/sequences_original_labeled`: raw point clouds $\textbf{with}$ point-wise labels. The coordinates are not aligned with those in 1.
3) `/sequences_completed_labeled`: ground truth completed point clouds $\textbf{with}$ point-wise labels
4) `/sequences_with_labels`: predicted completed point clouds by JS3C-Net $\textbf{with}$ point-wise labels 

Those 4 datasets are ready to use for PointNetVLAD

## How to set up enviroment

We use docker files to build our environments.
### JS3C-Net 
Run the docker file `Dockerfile_js3c`. Rename the file as `Dockerfile` in order to be able to use autobuild. \
After the docker image is created, build a docker container for JS3C-Net.\
Make sure that the GPUs as well as enough shared memories(>=64GB) are available to the container. \
Mount the SemanticKITTI dataset to the docker container. 

Inside the docker container, the following things must be further installed:
+ CMAKE of the most recent version:
```bash
    sudo apt remove --purge --auto-remove cmake \
    && sudo apt update && \
        sudo apt install -y software-properties-common lsb-release && \
        sudo apt clean all \
    &&  wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && sudo apt update \
    && sudo apt install cmake 
```
+ [spconv1.0](https://github.com/traveller59/spconv)  in `/lib/spconv`. We use the same version as [PointGroup](https://github.com/Jia-Research-Lab/PointGroup), you can install it according to their instruction. Higher version spconv will cause issues.

Clone our repository to your local computer.
```bash
git clone https://gitlab.lrz.de/student-projects-lmt-misik/improving-point-cloud-based-place-recognition-and-re-localization-using-point-cloud-completion-approaches.git
```
Change these files to the JS3C-Net's working directory in the following way:
+ Replace the file `test_kitti_ssc.py` with `gen_complet_submap.py`
+ Replace the file `log/JS3C-Net-kitti/args.txt` with `args.txt` from our repository
+ Replace the file `opt/JS3C_default_kitti.yaml` with `JS3C_default_kitti.yaml` from our repository
+ Change the validation sequences in `kitti_dataset.py` to `["00", "02", "05", "06", "07", "08"]`

### PointNetVLAD
Run the docker file `Dockerfile_pointnetvlad`. Rename the file as `Dockerfile` in order to be able to use autobuild.

Mount the datasets in  `/mnt/data4/Students/Xiaoang/Completed_PointCloud/full_completed_only_set/dataset` to the docker container.

Then add the log_ folders from [here](https://drive.google.com/file/d/1PpcWRjI-FY0SC3fdppYtHTkzjF-20uxw/view) to the working directory of PointNetVLAD, which contain the models trained with 1-Fold strategy:
+ `log_fold00/`
+ `log_fold02/`
+ `log_fold05/`
+ `log_fold06/`
+ `log_fold07/`
+ `log_fold08/`

At the end, add/replace the following files from our repository to the working directory of PointNetVLAD:
+ `loading_pointclouds_kitti.py`
+ `gen_KITTI_descriptors.py`
+ `KITTI_evaluation.py`
+ `evaluate_with_semantics.py`

## Generate Completed Point Clouds
Inside the docker container for JS3C-Net:
### predicted completed point clouds 
```bash
/home/user/miniconda/bin/python /home/user/JS3C-Net/gen_complet_submap.py --submap pred
```
### ground truth completed point clouds
```bash
/home/user/miniconda/bin/python /home/user/JS3C-Net/gen_complet_submap.py --submap gt
```
## Generate Ground Truth for PointNetVLAD
Inside the docker container for PointNetVLAD:
```bash
/home/user/miniconda/bin/python /home/user/poir]tnetvlad/submap-generation/KITTI/gen_gt.py
```
Under default setting, the distance threshold is 3m while the time threshold is 0s. By increasing the time threshold, more adjacent LiDAR frames of an anchor are neglected.\
After this step, you should see the new folder `KITTI_all/` in the working directory containing the file `positive_sequence_D-3_T-0.json`.

## Generate Global Descriptors
Inside the docker container for PointNetVLAD: \
In line 20 of the file `loading_pointclouds_kitti.py` you can define the number of points for downsampling, e.g. 4096.

In the method `get_global_descriptors()` you can define, for which sequences you want to generate their global descriptors. \
In line 69 you can define how to call the folder, where you will store the global descriptors. \
As an example, the database of ground truth completed point clouds downsampled to 4096 points is named as `feature_database/database_gt_4096_completed`

Then run:
```bash
/home/user/miniconda/bin/python /home/user/pointnetvlad/gen_KITTI_descriptors.py
```
to generate the database for the ground truth completed scenes.

## Evaluation

### Precision and Recall
Run the $\textbf{section 1}$ in the main function of `KITTI_evaluation.py`, where it uses the function `evaluate_precision_recall1()` to evaluate:
```bash
/home/user/miniconda/bin/python /home/user/pointnetvlad/KITTI_evaluation.py
```
to let the model try to retrieve all positives defined by .json file in `KITTI_all/`. \
If you want to evaluate @5 precision, change the `num_pred_neighbors=num_true_neighbors` in line 335 to `num_pred_neighbors=5`

If you want to evaluate the place recognition performance when the distribution of semantic labels is considered, run the following script: 
```bash
/home/user/miniconda/bin/python /home/user/pointnetvlad/evaluate_with_semantics.py
```

### PR-Curves & F1-Scores
Run the $\textbf{section 2}$ in the main function of `KITTI_evaluation.py`, where it uses the function `evaluate_precision_recall2()` to evaluate:
```bash
/home/user/miniconda/bin/python /home/user/pointnetvlad/KITTI_evaluation.py
```
## Visualization of Point Clouds
You can run the visualization scripts on your local computer, if the server has no monitor connected. \
All necesssary scripts for visualization are already in the directory `Visualization/` of our repository.

There are many parameters that can be customized for the visualization function in `visualize_semantic_scene.py`:
```bash
    parser.add_argument('--pointcloud_dir', type=str, help='Path to the point cloud file')
    parser.add_argument('--pc_perspective_dir', type=str, help='Path to the point cloud perspective file')
    parser.add_argument('--plot_semantic', type=bool, default=False, help='Whether to plot semantic masks (only for completed scenes)')
    parser.add_argument('--overview', type=bool, default=False, help='Whether to show the scene from an overview perspective')
    parser.add_argument('--downsample', type=bool, default=False, help='Whether to downsample the point cloud')
    parser.add_argument('--ground_thresh', type=float, default=False, help='The threshold for cropping the ground points')
```
An example:
```bash
/bin/python3 /home/xiaoang/Forschungspraxis/visualize_semantic_scene.py --pointcloud_dir /home/xiaoang/Forschungspraxis/sequence_completed_labeled/00/completion/000000.bin --pc_perspective_dir /home/xiaoang/Forschungspraxis/sequence_completed_labeled/00/completion/000000.bin --ground_thresh -1 --downsample True --plot_semantic True
```
As a result:
![](Visualization/Readme_example.png)

