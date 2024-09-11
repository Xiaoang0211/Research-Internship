import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize a point cloud scene')
    parser.add_argument('--pointcloud_dir', type=str, help='Path to the point cloud file')
    parser.add_argument('--pc_perspective_dir', type=str, help='Path to the point cloud perspective file')
    parser.add_argument('--plot_semantic', type=bool, default=False, help='Whether to plot semantic masks (only for completed scenes)')
    parser.add_argument('--overview', type=bool, default=False, help='Whether to show the scene from an overview perspective')
    parser.add_argument('--downsample', type=bool, default=False, help='Whether to downsample the point cloud')
    parser.add_argument('--ground_thresh', type=float, default=False, help='The threshold for cropping the ground points')
    args = parser.parse_args()
    return args

def get_mask_colors(pointcloud_mask, color_config):
    num_points = pointcloud_mask.shape[0]
    color_idx = pointcloud_mask[0,3]
    color = tuple(np.array(color_config[color_idx])/255)
    colors = [color]*num_points
    return colors
        

def pc_random_sampling(pointcloud, NUM_POINTS=16384):
    if pointcloud.shape[0] >= NUM_POINTS:
        ind = np.random.choice(pointcloud.shape[0], NUM_POINTS, replace=False)
        pointcloud = pointcloud[ind, :]
    else:
        ind = np.random.choice(pointcloud.shape[0], NUM_POINTS, replace=True)
        pointcloud = pointcloud[ind, :]

    return pointcloud

def extract_roi(pc, ground_thresh):
    l = 25.6
    ind = np.argwhere(pc[:, 0] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 0] >= -l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 1] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 1] >= -l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 2] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 2] >= ground_thresh).reshape(-1) # cropping ground. If you set it to -2, the gorund will be kept.
    pc = pc[ind]
    
    return pc
    
def visualize_incomplete_scene(pointcloud_dir, pc_perspective_dir, plot_semantic, overview, downsample, ground_thresh):
    pointcloud_perspect = np.fromfile(pc_perspective_dir, dtype=np.float32)
    pointcloud_perspect = pointcloud_perspect.reshape(-1,4)
    xs_perspect = pointcloud_perspect[:,0]
    ys_perspect = pointcloud_perspect[:,1]
    zs_perspect = pointcloud_perspect[:,2]
    
    pointcloud = np.fromfile(pointcloud_dir, dtype=np.float32)
    pointcloud = pointcloud.reshape(-1,4)
    
    if downsample == True:
       pointcloud = pc_random_sampling(pointcloud)

    # if plot_semantic == False:
    #     pointcloud = extract_roi(pointcloud, ground_thresh)
    pointcloud = extract_roi(pointcloud, ground_thresh)
    
    xs_whole = pointcloud[:,0]
    ys_whole = pointcloud[:,1]
    zs_whole = pointcloud[:,2]
    
    if plot_semantic==True: # plot with semantic masks (only for completed scenes)
        try:       
            pointcloud_masks = []
            # generate the semantic masks
            for i in range(1,20):
                idx = (pointcloud[:,3] == i)
                if idx.any() == True:
                    pointcloud_mask = pointcloud[idx,:]
                    pointcloud_masks.append(pointcloud_mask)
                else: 
                    continue
            # plot the point cloud masks
            pointcloud1 = pointcloud_masks[0]
            xs = pointcloud1[:,0]
            ys = pointcloud1[:,1]
            zs = pointcloud1[:,2]
            
            fig = plt.figure(figsize=(12,12))
            ax = fig.add_subplot(111, projection='3d')        
            ax.set_box_aspect((np.ptp(xs_perspect), np.ptp(ys_perspect), np.ptp(zs_perspect)))
            if overview == True:
                ax.view_init(azim=-90, elev=80)
            color_config = yaml.safe_load(open('/home/xiaoang/Forschungspraxis/mapping_colors.yaml', 'r'))
            colors = get_mask_colors(pointcloud1, color_config)
            ax.scatter(xs,ys,zs, c=colors, s=0.05, alpha=0.8)
            for i in range(1, len(pointcloud_masks)):
                pointcloud_mask = pointcloud_masks[i]
                colors = get_mask_colors(pointcloud_mask, color_config)
                xs = pointcloud_mask[:,0]
                ys = pointcloud_mask[:,1]
                zs = pointcloud_mask[:,2]
                ax.scatter(xs,ys,zs, c=colors, s=0.05, alpha=0.8)
                ax.axis(False)
            plt.show()
            print('done!')
        except:
            print("The raw scan does not contain any semantic labels!")
    else: # plot without semantic masks
        
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((np.ptp(xs_perspect), np.ptp(ys_perspect), np.ptp(zs_perspect)))
        if overview == True:
            ax.view_init(azim=-90, elev=80)
        ax.scatter(xs_whole,ys_whole,zs_whole, s=0.05, alpha=0.8)
        ax.axis(False)
        plt.show()
        
        
if __name__ == "__main__":
    args = parse_args()
    # pointcloud_dir = 'sequence_completed_labeled/00/completion/000000.bin'
    # pc_perspective_dir = pc_complet_gt_dir = 'sequence_completed_labeled/00/completion/000000.bin'
    visualize_incomplete_scene(args.pointcloud_dir, args.pc_perspective_dir, args.plot_semantic, args.overview, args.downsample, args.ground_thresh)
