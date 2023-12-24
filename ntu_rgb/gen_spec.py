import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import math
import gc

class GenSpec:
    def __init__(self,path:str,save_to:str,drop_col=None): #ex  'path=/Users/kunkerdthaisong/ipu/SampleSkeleton/', 'path=/Users/kunkerdthaisong/ipu/'
        self.hop_1_dict = {}
        self.hop_1_pairs = [(24, 25), (25, 12), (12, 11), (11, 10), (10, 9),
                (9, 21), (3, 4),(4,3),(21, 5),
                (5, 6), (6, 7), (7, 8), (8, 22), (22, 23),(23,8),
                (2, 1), (1, 17),
                (13, 14), (14, 15), (15, 16),(16,15),
                (17, 18), (18, 19), (19, 20),(20,19)]
        self.zones={1:[25,24,12,11,10,9],2:[20,19,18,17],3:[16,15,14,13],4:[23,22,8,7,6,5],5:[1,2,21,3,4]}

        for start, end in self.hop_1_pairs:
            self.hop_1_dict.setdefault(start, []).append(end)

        self.all_files = glob.glob(path+"*.npy", recursive=True)
        self.save_to=save_to
        self.drop_col=drop_col


    def get_dis(self, x, y, z) -> float:
        return math.sqrt(x**2 + y**2 + z**2)

    def get_two_dis(self, x1, y1, z1, x2, y2, z2) -> float:
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    def get_two_angle(self, x1, y1, z1, x2, y2, z2) -> float:
        denominator = (math.sqrt(x1**2 + y1**2 + z1**2) * math.sqrt(x2**2 + y2**2 + z2**2))
        if denominator == 0:
            return 0
        else:
            numerator = (x1 * x2) + (y1 * y2) + (z1 * z2)
            res = numerator / denominator
            res2 = math.acos(res) * (180 / math.pi)
            return res2
        
    def get_zone(self,joint)->int:
        for j in range(1,6):
            if(joint in self.zones.get(j)):
                return j

    def get_dis_angle_eachjoint(self, movement, hop_type: dict,hop:str):
        dis_from_hop = []
        angle_from_hop = []
        max_frame = movement.iloc[len(movement) - 1, 4]
        for frame in range(1, max_frame + 1):
            movement_f = movement.loc[movement['frame'] == frame]
            for i in range(1, 26):
                hop_indices = hop_type.get(i)
                if hop_indices is not None and len(hop_indices) > 0:
                    x1, y1, z1 = movement.loc[i, ['x', 'y', 'z']]
                    x2, y2, z2 = movement.loc[hop_indices[0], ['x', 'y', 'z']]
                    res = self.get_two_dis(x1, y1, z1, x2, y2, z2)
                    res1 = self.get_two_angle(x1, y1, z1, x2, y2, z2)
                    dis_from_hop.append(res)
                    angle_from_hop.append(res1)

        movement[f'dis_from_{hop}'] = dis_from_hop
        movement[f'angle_from_{hop}'] = angle_from_hop


        del dis_from_hop,angle_from_hop,res,res1,x1,y1,z1,x2,y2,z2
        gc.collect()

        return movement
    

    def gen_table(self,path):#gen_feature
        VIDEO_ = np.load(path, mmap_mode=None, allow_pickle=True)
        shape0, shape1, shape2 = VIDEO_.tolist()['skel_body0'].shape   #shape0==numbe_of_frame ,shape1==3(x,y,z) ,shape2==1
        movement1 = VIDEO_.tolist()['skel_body0'].reshape(1, shape0, shape1, shape2, 1)
        pos_x,pos_y,pos_z=[],[],[]
        joint=[]
        frames=[]
        f=0
        for frame in movement1[0]:#1 frame (25,3,1) , each movement specific frames
            j=0
            f=f+1
            for x,y,z in frame: #position
                pos_x.append(x[0])
                pos_y.append(y[0])
                pos_z.append(z[0])
                j=j+1
                joint.append(j)
                frames.append(f)

        df_movement1=pd.DataFrame({'x':pos_x,'y':pos_y,'z':pos_z})
        df_movement1['joint']=joint
        df_movement1['frame']=frames
        df_movement1['zone']=df_movement1['joint'].apply(lambda joint:self.get_zone(joint))
        df_movement1['dis_from_00']=df_movement1.apply(lambda x:self.get_dis(x.x,x.y,x.z),axis=1)

        del joint,frames,VIDEO_
        gc.collect()

        last_df=self.get_dis_angle_eachjoint(df_movement1,hop_type=self.hop_1_dict,hop="hop1")
        del joint,frames,VIDEO_,df_movement1
        gc.collect()

        return last_df

    def gen_spectogram(self,df,name_file_save_to,drop_col=None,dpi=56): #ex drop_col =["x","y","z"]
        data_matrix = df.select_dtypes(include=[np.number]).values.T   #transpose to gen spectogram (time domain)
        if drop_col is not None:
            data_matrix = data_matrix.drop(axis=1,columns=drop_col)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
        ax.axis('off')
        fig.set_size_inches(448 / dpi, 448 / dpi)  
        fig.savefig(name_file_save_to, bbox_inches='tight', pad_inches=0, dpi=dpi)
        
        del data_matrix
        gc.collect()


    def run_all(self,exist_dataframe=None,gen_spec=None):
        #gen spectogram and collect dataframes
        count=0
        for i in tqdm(self.all_files): # ex i='/Users/kunkerdthaisong/ipu/SampleSkeleton/S003C001P019R002A005.skeleton.npy'
            if count==0:
                res=self.gen_table(path=i)
                if(gen_spec==True):
                    self.gen_spectogram(res,name_file_save_to=self.save_to+i.split("/")[5]+".png",drop_col=None) #/Users/kunkerdthaisong/ipu/SampleSkeleton/ + "S003C001P019R002A005.skeleton.npy" +".png"
                res["file_path"]=[i]*len(res)
                count+=1
            else:
                res1=self.gen_table(path=i)
                if(gen_spec==True):
                    self.gen_spectogram(res,name_file_save_to=self.save_to+i.split("/")[5]+".png",drop_col=None)
                res1["file_path"]=[i]*len(res1)
                res=pd.concat([res,res1])
                
                del res1
                gc.collect()
        res.to_csv(self.save_to+"dataframe.csv")
        return res