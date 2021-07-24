from os import stat
import pandas as pd
from copy import deepcopy

class Candidate_Eliminate():
    def __init__(self,data_path):
        '''
        初始化Candidate_Eliminate算法
        '''
        self.parse_data(data_path) #解析数据
        self.S = [['#' for i in range(self.attri_num)]] #初始S
        self.G = [['?' for i in range(self.attri_num)]] #初始G
        self.H = [] #最终获得的假设集合
        self.running_time = 0  #已处理的样本数


    def parse_data(self,data_path):
        '''
        读取并解析数据
        '''
        data = pd.read_csv(data_path)  
        self.attri_list = [attri for attri in data.columns.values.tolist()[:-1]]  #获得属性类型列表
        self.attri_num = len(self.attri_list)   #属性类型数量 
        self.attri_dict =   {attri : [] for attri in self.attri_list}   #每种属性类型类型对于的属性种类
        for attri in self.attri_dict.keys():  
            self.attri_dict[attri] = list(set(data[attri]))
        self.samples = data.values.tolist()   #获得数据
    
    def print_process_start(self):
        '''
        打印初始信息
        '''
        print("初始化 G : ", self.G)
        print("初始化 S : ", self.S)
        print("开始处理...")
        print("\n")
        
    def print_process_running(self,sample):
        '''
        打印运行中信息
        '''
        self.running_time += 1
        print("add the {}th sample : ".format(self.running_time), sample)
        print("G : " , self.G)
        print("S : " , self.S)
        print("\n")
    
    def print_process_end(self):
        '''
        打印结果信息
        '''
        print("trained all samples")
        print("results of G : ", self.G)
        print("results of S : ", self.S)
        print("results of H : ", self.H)
        
        
    def process(self):
        '''
        主要处理函数
        '''
        self.print_process_start()
        
        for sample in self.samples:
            if sample[-1] == "Yes":  #正例样本
                new_G = []
                for hypo_g in self.G:  #首先去除G中不覆盖改正例的假设
                    if self.is_consistent(hypo_g,sample,self.attri_num): 
                        new_G.append(hypo_g)
                self.G  = new_G
                
                new_S = deepcopy(self.S)  #先复制S
                for hypo_s in self.S:
                    if not self.is_consistent(hypo_s,sample,self.attri_num):
                        new_S.remove(hypo_s)   #去除S中与该样本不一致的假设hypo_s
                        mini_general = self.minimal_general(hypo_s,sample) #生成hypo_s的极小泛化式
                        if mini_general is not None: 
                            new_S.append(mini_general)
                self.S = new_S  #更新S
                self.check_more_general_in_S()  #移除S中更general的假设
                
            else: #反例样本
                new_S = []
                for hypo_s in self.S: #首先去除S中覆盖了反例的假设
                    if not self.is_consistent(hypo_s,sample,self.attri_num):
                        new_S.append(hypo_s)
                self.S = new_S
                
                new_G = deepcopy(self.G) #先复制G
                for hypo_g in self.G:
                    if self.is_consistent(hypo_g,sample,self.attri_num):
                        new_G.remove(hypo_g) #去除G中覆盖了该反例的假设
                        mini_special = self.minimal_special(hypo_g,sample) #生成极小特殊式（多个）
                        for mini_special_hypo in mini_special:
                            if not mini_special_hypo in new_G:  #去除重复的极小特殊式
                                new_G.append(mini_special_hypo)
                self.G = new_G #更新G
                self.check_more_special_in_G() #移除G中更special的假设
            self.print_process_running(sample)
        
        self.generate_hypos()  #生成G和S的中间结果H
        self.print_process_end()
                        
    
    def generate_hypos(self):
        '''
        生成最终结果，即G和S中间的H
        '''
        for hypo_s in self.S:
            for hypo_g in self.G:
                for i in range(self.attri_num):
                    if hypo_s[i] == hypo_g[i]:   #不存在中间假设
                        continue
                    hypo = deepcopy(hypo_g)
                    if hypo_g[i] == "?":      #将hypo特殊化
                        hypo[i] = hypo_s[i]     
                        if not hypo in self.H:
                            self.H.append(hypo)
                
            
    def minimal_general(self,hypo_s,sample):
        '''
        生成hypo_s相对于样本sample的极小泛化式
        '''
        new_hypo_s =deepcopy(hypo_s)
        for i in range(self.attri_num):
            if hypo_s[i] == '#':
                new_hypo_s[i] = sample[i]
            elif not self.attri_match(hypo_s[i],sample[i]):
                new_hypo_s[i] = "?"
        for hypo_g in self.G: #需要保证生成的new_hypo_s比G中至少一个假设更加special
            if self.more_general(hypo_g,new_hypo_s):
                return new_hypo_s
        return None   
    
    def minimal_special(self,hypo_g,sample):
        '''
        生成hypo_g相对于样本sample的极小特殊式（多个）
        '''
        new_hypo_g_list = []
        for i, hypo_g_attri in enumerate(hypo_g):
            if hypo_g_attri == "?":   #遇到"?"时才会特殊化
                attir_values = self.attri_dict[self.attri_list[i]]
                for value in attir_values:  #遍历该属性类型的值
                    if value != sample[i]:  #只处理非原本属性的属性值
                        new_hypo_g = deepcopy(hypo_g)
                        new_hypo_g[i] = value
                        for k , new_hypo_g_attri in enumerate(new_hypo_g):
                            if new_hypo_g_attri == "?":
                                continue
                            elif new_hypo_g_attri == sample[k]:
                                new_hypo_g[k] = "?"
                        for hypo_s in self.S:
                            if self.more_general(new_hypo_g,hypo_s):  #需要保证生成的new_hpyo_g比S中至少一个假设更加general
                                new_hypo_g_list.append(new_hypo_g)
                                break
        return new_hypo_g_list
        
        
    
    def check_more_general_in_S(self):
        '''
        对于S中任意两个假设，如果存在more general的偏序关系，则删除更general的那个元素
        '''
        for hypo_s_i in self.S:
            '''
            注意：
            这里必须要从头开始遍历，保证对于self.S中的任意两个元素，分别进行了more_general(a,b)和more_general(b,a)两种判断
            如果只进行了more_general(a,b)的判断，那么就遗漏了b比a更加general的情况，而没有将b进行remove
            '''
            for hypo_s_j in self.S: 
                if hypo_s_i != hypo_s_j and self.more_general(hypo_s_i,hypo_s_j):
                    self.S.remove(hypo_s_i)
        
    def check_more_special_in_G(self):
        '''
        对于G中的任意两个假设，如果存在more special的偏序关系，则删除更加special的那个元素
        '''
        for hypo_g_i in self.G:
            '''
            注意：
            与上面同理，以n^2的复杂度从头遍历，防止遗漏
            '''
            for hypo_g_j in self.G:
                if hypo_g_i != hypo_g_j and self.more_general(hypo_g_i,hypo_g_j):
                    self.G.remove(hypo_g_j)
                
                
    #定义静态方法，以便在外部调用接口时可以不进行实例化
    @staticmethod
    def is_consistent(hypo,sample,attri_num):
        '''
        判断假设hypo是否与样本sample一致
        '''
        for i in range(attri_num):
            if not Candidate_Eliminate.attri_match(hypo[i],sample[i]):
                return False
        return True
    
    #定义静态方法，以便在外部调用接口时可以不进行实例化
    @staticmethod
    def attri_match(attri_a, attri_b):
        '''
        比较属性a和属性b是否一致
        '''
        return attri_a == "?" or attri_b == "?" or attri_a == attri_b
    
    
    #定义静态方法，以便在外部调用接口时可以不进行实例化
    @staticmethod
    def more_general(hypo_a,hypo_b):
        '''
        比较假设hypo_a是否比假设hypo_b更general
        '''
        
        '''
        注意：
        more_general是定义在假设集合H的偏序关系。如果返回True，表示hype_a比hype_b 更 general， 或者说 hypo_b 比 hypo_a 更special 
        但是返回False，则hype_b不一定比hype_a更general，反例可以选择训练集中的任意两个实例。即返回False无法判断是否有more general 或则 more special 的关系
        '''
        for attri_a,attri_b in zip(hypo_a,hypo_b):
            if (not attri_a == "?") and (not attri_b == "#") and attri_a != attri_b:
                return False
        return True 
    



if __name__ == "__main__":
    ce_instance = Candidate_Eliminate("data.csv")
    ce_instance.process()
        