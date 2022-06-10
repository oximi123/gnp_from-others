import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh
import imageio

import torch
from torch.utils.data import DataLoader
from .utils import get_rays
from scipy.ndimage import gaussian_filter

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()



def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    forward_vector = - normalize(centers) # camera direction (OpenGL convention!)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class DyNeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=30):
        super().__init__()
        self.opt = opt         
        self.video_frame_num = opt.video_frame_num                                                                      
        self.video_frame_start = opt.video_frame_start                                                                     
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode # colmap, blender, llff 
        self.preload = opt.preload # preload data into GPU 
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses. 
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.rand_pose_interval = opt.rand_pose_interval
        self.error_type = opt.error_type

        self.training = self.type in ['train', 'all']
        self.num_rays = self.opt.num_rays if self.training else -1

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda d: d['file_path'])    
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.times = []
            self.images = None
            for i in range(self.video_frame_num + 1):
                ratio = np.sin(((i / self.video_frame_num) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
                self.times.append(i*1.0/self.video_frame_num)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all': use all frames
            
            self.poses = []
            self.times = []       # frame times
            self.images = []
            self.cam_idx = []
            self.isg_map = []
            self.ist_map = []
            for cam_idx, f in tqdm.tqdm(enumerate(frames), desc=f'Loading {type} data:'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender':
                    f_path += '.mp4' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                # scale camera position into bounding box ? How do you make it?
                # Only suitable for fox dataset, how to set other dataset ?
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)        

                # read mp4 here
                vid = imageio.get_reader(f_path, 'ffmpeg')
                cam_images = []
                for frame_idx, raw_image in enumerate(vid):
                    # limit video time
                    if frame_idx < self.video_frame_start:
                        continue
                    if frame_idx >= (self.video_frame_num + self.video_frame_start):
                        break
                    
                    if self.H is None or self.W is None:
                        self.H = raw_image.shape[0] // downscale
                        self.W = raw_image.shape[1] // downscale

                    # add support for the alpha channel as a mask.
                    if raw_image.shape[-1] == 3: 
                        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pass
                    else:
                        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                        pass

                    image = cv2.resize(raw_image, (self.W, self.H))
                    image = image.astype(np.float32) / 255 # [H, W, 3/4]    # downsample images and scale the rgb value into [0,1]

                    self.poses.append(pose)
                    self.images.append(image)
                    self.times.append((frame_idx*1.0/self.video_frame_num))    # 时间归一化到[0,1)
                    self.cam_idx.append(cam_idx)

                    # cam_image = cv2.GaussianBlur(cam_image, (11, 11), 0)
                    cam_images.append(cv2.resize(raw_image, (128, 128)))

                cam_images = np.stack(cam_images, axis=0).astype(np.float) # [N,H,W]
                m_cam_images = np.mean(cam_images, axis = 0)
                std_cam_images = np.sum(np.std(cam_images, axis = 0), axis = -1)
                std_max, std_min = np.max(std_cam_images), np.min(std_cam_images)
                # calculate ISG weights
                # calculate IST weights
                for im_idx in range(len(cam_images)):
                    
                    isg_map = std_cam_images
                    isg_map = gaussian_filter(isg_map + 1, sigma=11)
                    isg_map = isg_map/np.sum(isg_map) # [128, 128]
                    self.isg_map.append(isg_map.reshape(-1))

                    # calculate ist map
                    if im_idx == 0:
                        ist_map = np.ones((128,128))
                    else:
                        ist_map = np.sum(np.abs(cam_images[im_idx] - cam_images[im_idx - 1]), axis = -1)
                        ist_map = gaussian_filter(ist_map, sigma=11)
                    ist_map = ist_map/np.sum(ist_map) # [128, 128]
                    self.ist_map.append(ist_map.reshape(-1))
                
            self.cam_idx = torch.from_numpy(np.array(self.cam_idx))
            self.isg_map = torch.from_numpy(np.stack(self.isg_map, axis=0))
            self.ist_map = torch.from_numpy(np.stack(self.ist_map, axis=0))
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        self.times = torch.from_numpy(np.array(self.times))

        # initialize error_map
        # error_map is sample method!
        # the ray sample probability is proportional to the loss value between the prediction and ground truth. 
        if type == 'train' and self.error_type == "error":
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # uncomment to view all training poses.
        #visualize_poses(self.poses.numpy())
        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(torch.half if self.fp16 else torch.float).to(self.device)
            if self.type == "train":
                self.cam_idx = self.cam_idx.to(self.device)
                if self.error_type == "error":
                    self.error_map = self.error_map.to(self.device)
                if self.error_type == "isg":
                    self.isg_map = self.isg_map.to(self.device)
                if self.error_type == "ist":
                    self.ist_map = self.ist_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # always 1

        # random pose without gt images.
        if index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.bound)

            # may need different stragegy, e.g., downscaled whole image (CLIP), or random patch (piecewise smoothness).
            rays = get_rays(poses, self.intrinsics, self.H, self.W, -1)

            return {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'rays_t': (torch.ones(rays['rays_o'].shape[:2]) * self.times[index]).type(torch.float32).to(self.device)
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]
        
        if self.type == "train" and self.error_type == "isg":
            isg_map = self.isg_map[index]
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, isg_map)
        elif self.type == "train" and self.error_type == "ist":
            ist_map = self.ist_map[index]
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, ist_map)
        elif self.type == "train" and self.error_type == "error":
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.error_map[index])
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'rays_t' : (torch.ones(rays['rays_o'].shape[:2]) * self.times[index]).type(torch.float32).to(self.device)
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if self.type == "train" and self.error_type == "error":
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose_interval > 0:
            size += size // self.rand_pose_interval # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader
