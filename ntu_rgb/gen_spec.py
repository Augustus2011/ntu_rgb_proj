import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import math
import polars as pl
import torch

class GenSpec:
    def __init__(self, path:str=None, save_to:str=None, drop_col=None, gen_type:int=0, path_parquet:str=None,feature_imp:bool=None): #ex  'path=/Users/kunkerdthaisong/ipu/SampleSkeleton/', 'path=/Users/kunkerdthaisong/ipu/'
        self.hop_1_dict = {}
        self.hop_1_pairs = [(24, 25), (25, 12), (12, 11), (11, 10), (10, 9),
                (9, 21), (3, 4),(4,3),(21, 5),
                (5, 6), (6, 7), (7, 8), (8, 22), (22, 23),(23,8),
                (2, 1), (1, 17),
                (13, 14), (14, 15), (15, 16),(16,15),
                (17, 18), (18, 19), (19, 20),(20,19)]
        
        self.zones={1:[25,24,12,11,10,9],2:[20,19,18,17],3:[16,15,14,13],4:[23,22,8,7,6,5],5:[1,2,21,3,4]}
        self.gen_type=gen_type
        self.df=None
        self.path=path
        self.path_l=glob.glob(self.path+"*.npy")
        self.path_l=sorted(self.path_l)
        self.save_to=save_to
        self.drop_col=drop_col

        #filter self.path #in this case only use class 50-60
        #this
        #self.class_l=[]
        #for i in self.path_l:
            
        #    self.class_l.append(i.split("A")[1]) #Skeleton_Coordinate/raw_npy1TO60/S015C002P008R001A018.skeleton.npy'# to get 018.skeleton.npy
        #self.unique_class_l=set(self.class_l)
        #self.unique_class_l=sorted(self.unique_class_l)
        #self.two_p_class=list(self.unique_class_l)[49:]
        #two_p_class_paths=[]
        #for i in self.path_l:
        #    for c in self.two_p_class:
        #        if c in i:
        #            two_p_class_paths.append(i)
        #self.path_l=two_p_class_paths
        #to this
        
        if path_parquet is not None :
            assert gen_type==3, "gen_type should =3 if you want to genspec from exist parquet"
            self.df=pl.read_parquet(path_parquet)  #parquet
            if feature_imp ==True: #to find feature importance
                self.l=self.df["file_path"].unique().sort()[:30].to_list() #select only first 30 actions
                self.df=self.df.filter(pl.col("file_path").is_in(self.l))
            
        for start, end in self.hop_1_pairs:
            self.hop_1_dict.setdefault(start, []).append(end)

        
    def lin_filter(self,df:pl.DataFrame,start:int)->int: #df should be sorted by filename_n_class
        for i,j in enumerate(df["filename_n_class"][start:]):
            if(df["filename_n_class"][start]!=j):
                start=start+i
                break
        return start
    
    def get_33_action(self,df:pl.DataFrame,start:int): #df should be sorted by filename_n_class #ex a50-a60 *3
        n=0
        prv_ns=0
        while(n<33):
            ns=self.lin_filter(df,prv_ns)
            n+=1
            prv_ns=ns
        return df[:ns]


    def get_dis(self, x, y, z) -> float:
        #threshold
        self.x2,self.y2,self.z2=0,-0.42,3.2 #where to set sensor
        return math.sqrt((x-self.x2)**2 +(y-self.y2)**2 + (z-self.z2)**2)

    def get_two_dis(self, x1, y1, z1, x2, y2, z2) -> float:
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

    def get_two_angle(self, x1, y1, z1, x2, y2, z2) -> float:
        denominator = (math.sqrt(x1**2 + y1**2 + z1**2) * math.sqrt(x2**2 + y2**2 + z2**2))
        if denominator == 0:
            return 0.00
        else:
            try:
                numerator = (x1 * x2) + (y1 * y2) + (z1 * z2)
                res = numerator / denominator
                res2 = torch.tanh(torch.tensor(res))
                angle = math.acos(res2.item())
                angle_degrees = math.degrees(angle)
                return angle_degrees
            
            except Exception as e:
                print(e)
                print(numerator)
                print(denominator)
                return 0.00
        
    def get_zone(self,joint)->int:
        for j in range(1,6):
            if(joint in self.zones.get(j)):
                return j

    def get_dis_angle_eachjoint(self, movement, hop_type: dict,hop:str):
        dis_from_hop = []
        angle_from_hop = []
        max_frame = movement['frame'].max()
        for frame in range(1, max_frame + 1):
            movement_f = movement.filter(movement['frame'] == frame)
            for i in range(1, 26):
                hop_indices = hop_type.get(i)
                if hop_indices is not None and len(hop_indices) > 0:
                    joint_data = movement.filter(movement['joint'] == i)
                    x1, y1, z1 = joint_data.select(['x', 'y', 'z']).to_numpy()[0]
                    x2, y2, z2 = movement.filter(movement['joint'] == hop_indices[0]).select(['x', 'y', 'z']).to_numpy()[0]
                    res = self.get_two_dis(x1, y1, z1, x2, y2, z2)
                    res1 = self.get_two_angle(x1, y1, z1, x2, y2, z2)
                    dis_from_hop.append(res)
                    angle_from_hop.append(res1)

        #new_col={f'dis_from_{hop}':dis_from_hop}
        movement = movement.with_columns(pl.Series(f"dis_from_{hop}",dis_from_hop))
        #new_col={f'angle_from_{hop}':angle_from_hop}
        movement = movement.with_columns(pl.Series(f"angle_from_{hop}",angle_from_hop))
        del dis_from_hop,angle_from_hop,max_frame,movement_f,res,res1

        return movement
    

    def gen_table(self,path):#gen_feature
        #gentable for skel_body2
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

        df_movement1 = pl.DataFrame({
                'frame': frames,
                'joint': joint,
                'skel_body':[0]*len(frames),
                'x': pos_x,
                'y': pos_y,
                'z': pos_z
                })
        
        del shape0,shape1,shape2,movement1,pos_x,pos_y,pos_z,joint,frames,f,j

        if VIDEO_.tolist()['nbodys'][0]==2: #gentable for skel_body1
            shape0, shape1, shape2 = VIDEO_.tolist()['skel_body1'].shape   #shape0==numbe_of_frame ,shape1==3(x,y,z) ,shape2==1
            movement1 = VIDEO_.tolist()['skel_body1'].reshape(1, shape0, shape1, shape2, 1)
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

            df_movement1_skel1 = pl.DataFrame({
                    'frame': frames,
                    'joint': joint,
                    'skel_body':[1]*len(frames),
                    'x': pos_x,
                    'y': pos_y,
                    'z': pos_z
                    })
            self.l=[df_movement1,df_movement1_skel1]
            df_movement1=pl.concat(self.l)
            del self.l
        
        #df_movement1 = df_movement1.with_columns(pl.Series(name='zone',values=df_movement1['joint'].apply(lambda joint: self.get_zone(joint)))) # Add 'zone' column 
        df_movement1 = df_movement1.with_columns(df_movement1.map_rows(lambda row:self.get_dis(row[3], row[4], row[5]))) #get distance
        df_movement1= df_movement1.rename({"map": "dis_from_sensor"})
        #last_df = self.get_dis_angle_eachjoint(df_movement1, hop_type=self.hop_1_dict, hop="hop1") #get angle
        return df_movement1

    def gen_spectogram(self, df:pl.DataFrame, name_file_save_to:str, drop_col=None, dpi=56):
        if drop_col is not None:
            df = df.drop(columns=drop_col)

        data_matrix=df.drop("file_path")
        data_matrix = data_matrix.to_numpy().T
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
        del data_matrix
        ax.axis('off')

        fig.set_size_inches(448 / dpi, 448 / dpi)
        fig.savefig(name_file_save_to, bbox_inches='tight', pad_inches=0, dpi=dpi)

    def gen_spectogram2(self,df:pl.DataFrame,drop_col=None): #df only have [4,16,20,22,24]
        prev=0
        n=0
        while prev!=self.lin_filter(df,prev):
            ns=self.lin_filter(df,prev)
            #print(prev,ns,ns-prev)
            action=df[prev:ns]
            filename=os.path.basename(action["filename_n_class"][(prev+ns)//2])
            action_4=action.filter(pl.col("joint")==4)# head
            action_16=action.filter(pl.col("joint")==16)#left foot
            action_20=action.filter(pl.col("joint")==20)#right foot
            action_22=action.filter(pl.col("joint")==22)#left hand
            action_24=action.filter(pl.col("joint")==24)#right hand
            

            os.path.join(self.save_to, f"{filename}.png")
            #get velocity from dis_from_sensor
            
            
            prev=ns

    def run_all(self):
        gen_both=None
        gen_spec=None
        gen_table=None

        if self.gen_type==0:
            gen_both=True

        elif self.gen_type==1:
            gen_table=True

        elif self.gen_type==2:
            gen_spec=True

        elif self.gen_type==3:
            self.df=self.df.filter(pl.col("joint").is_in([4,16,20,22,24])) #use only head,left hand,left foot, right hand, right foot
            l_df=len(self.df) #int
            if len(self.df["skel_body"].unique())>1:
                self.gen_spectogram2()
                
            else:   
                for i in tqdm(self.df["file_path"].unique(maintain_order=True)):
                    filename = os.path.basename(i)
                    self.gen_spectogram2(self.df.filter(pl.col("file_path") == i), drop_col=self.drop_col)
                pass

        if gen_both==True:
            gen_table=True
            gen_spec=True

        dfs =[]
        while (self.gen_type!=3):
            for i in tqdm(self.path_l):
                df_i = self.gen_table(path=i)
                df_i=df_i.with_columns(pl.Series("file_path",[i] * len(df_i)))
                
                if gen_spec:
                    filename = os.path.basename(i)
                    self.gen_spectogram(df_i, name_file_save_to=os.path.join(self.save_to, f"{filename}.png"), drop_col=self.drop_col)
                if gen_table:
                    dfs.append(df_i)
            if gen_table:
                res = pl.concat(dfs)
                res.write_parquet(os.path.join(self.save_to, "one_person_dataframe.parquet"))

            del df_i,dfs
            break
