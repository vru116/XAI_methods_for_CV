import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from perlin_noise import PerlinNoise
from scipy.ndimage import gaussian_filter


class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        # for i in tqdm(range(N), desc='Generating filters'):
        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

        self.method = 'orig'


    def generate_gaussian_masks(self, N, s, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.normal(loc=0.5, scale=0.25, size=(N, s, s))
        grid = np.clip(grid, 0, 1)

        self.masks = np.empty((N, *self.input_size))

        # for i in tqdm(range(N), desc='Generating filters'):
        if s != self.input_size:
            for i in range(N):
                # Random shifts
                x = np.random.randint(0, cell_size[0])
                y = np.random.randint(0, cell_size[1])
                # Linear upsampling and cropping
                self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                            anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        else:
            self.masks = grid
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        
        self.method = 'gaus'            

    def generate_perlin_masks(self, N, s, scale=5, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.empty((N, s, s), dtype=np.float32)
        for i in range(N):
            noise = PerlinNoise(octaves=scale, seed=np.random.randint(10000))
            for xi in range(s):
                for yi in range(s):
                    grid[i, xi, yi] = noise([xi / s, yi / s])

            grid[i] -= grid[i].min()
            grid[i] /= (grid[i].max() + 1e-8)

        self.masks = np.empty((N, *self.input_size))

        # for i in tqdm(range(N), desc='Generating filters'):
        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = N

        self.method = 'perlin'

    # def load_masks(self, filepath):
    #     self.masks = np.load(filepath)
    #     self.masks = torch.from_numpy(self.masks).float().cuda()
    #     self.N = self.masks.shape[0]

    def load_masks(self, filepath, p1=0.1):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
        self.p1 = p1

    # def forward(self, x):
    #     N = self.N
    #     _, _, H, W = x.size()
    #     # Apply array of filters to the image
    #     stack = torch.mul(self.masks, x.data)

    #     # p = nn.Softmax(dim=1)(model(stack)) processed in batches
    #     p = []
    #     for i in range(0, N, self.gpu_batch):
    #         p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
    #     p = torch.cat(p)
    #     # Number of classes
    #     CL = p.size(1)
    #     sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
    #     sal = sal.view((CL, H, W))
    #     sal = sal / N / self.p1
    #     return sal
    
    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()

        # print(self.masks.shape) # (N, 1, H, W)
        # print(x.data.shape) # (B, C, H, W)
        stack = torch.mul(self.masks, x.data)

        p = []
        for i in range(0, N, self.gpu_batch):
            batch_masks = stack[i:min(i + self.gpu_batch, N)]
            p.append(self.model(batch_masks))
        
        p = torch.cat(p)
        # print("Min/max logits:", p.min().item(), p.max().item()) 

        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))

        if self.method == 'orig':
            sal = sal / N / self.p1
        else:
            sal = sal / N / self.masks.mean()
        return sal
