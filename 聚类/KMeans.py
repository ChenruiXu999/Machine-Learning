##### import ###
import os
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

#配置一些环境变量，用于kmeans模型预测之后的规则模型筛选
envX = os.environ
envX['partyId_cnt']='1000'
envX['DC_RATE1']='0.85'
envX['DC_RATE2']='1.15'
envX['t_avg']='200000'
envX['AVG_CNT_MONTH']='5'
envX['CASH_AVG_Divide_T_AVG']='0.3'
envX['CASH_CNT_Divide_T_CNT']='0.3'
envX['LC_AVG_Divide_T_AVG']='0.5'
envX['LC_CNT_Divide_T_CNT']='0.6'

###主函数###
def process(source_file: str, result_file: str, dt: str): ## 环境变量自动获取
    data=pd.read_csv(source_file)
    size = data.shape[0]
    if size == 0:
        print("模型输入文本内容为空，请检查数据! ## Input text content is empty, check data!")
        sys.exit(0)
    ### 临时的数据预处理工作 ### 针对上一版本特征，上线时需要针对最新版特征修改    
    df = data.copy()
    df.drop_duplicates(inplace=True)
    party_id=df['party_id']
    df = df.drop(['party_id'],axis=1)
    df_use = df.replace('(null)',0)
    #### 预期生产线上的数据 #### 在以上字段基础上增加是否上报标签等
    ## feature_use ##### 用来筛选模型计算需要的特征列
    ### 按分行来聚类 ####### 
    ######## 定义模型主体 ##### model 
    def kmeans_model(gp_df):
    ### 计算距离前 先标准化
        standard_data = StandardScaler().fit_transform(gp_df)
        ## 定义搜索函数,自动寻参
        def silhouette_All(n,bt,r):
            data_cluster = MiniBatchKMeans(init='k-means++',n_clusters=n,batch_size = bt,reassignment_ratio=r)
            data_cluster.fit(standard_data)
            label = data_cluster.labels_
            silhouette_coefficient = silhouette_score(standard_data,label)
            return silhouette_coefficient
        #### 网格搜索,最佳参数越接近于1 越好，范围[-1,1]
        y = []
        score = -1
        nb = 0
        rb = 0
        bat = 1000
        print('开始搜寻最佳参数：') 
        for n in range(2,5): # 最小两类,
            bt=50
            r=0.01
            data_silhouette_mean = silhouette_All(n,bt,r)
            y.append(data_silhouette_mean)
            if data_silhouette_mean > score:
                score = data_silhouette_mean
                nb = n
                bat = bt
                rb = r
        max_score = max(y)
        best_K = nb
        print('轮廓系数最大值：',max_score)
        print('最佳分类数：',best_K)
        ## 也可以人工设置 轮廓系数阈值，达到阈值才停止
        #### 选择以上参数作为最佳模型参数，得到分类结果 ####
        data_cluster = MiniBatchKMeans(init='k-means++',n_clusters=nb,batch_size = bat,reassignment_ratio=rb,random_state=9)
        data_cluster.fit(standard_data) 
        label = data_cluster.labels_
        ### 标签结果拼接 
        gp_df['labels'] = list(label)
        gp_df['party_id']=party_id
        df = gp_df.copy()
        #### 指标筛选 ### 簇内客户过多，筛选展示
        ## 簇触发条件 + 基础业务触发条件 + 任意一个业务标签
        ########## 以下进行列筛选 ##################### 
        ##规则模型
        ####### 簇触发条件：簇内客户数量 <= 1000 + 存在已上报案例的客户 ###
        ### 环境变量参数获取：partyId_cnt ## 
        ########################################################################################################################################
        ########################################################################################################################################
        thres0 = int(envX['partyId_cnt'])
        size_label=df.groupby('labels').size().reset_index()
        size_label.columns = ['labels','group_size']
        size_df= df.set_index('labels').join(size_label,how='left')
        thres_df = size_df[(size_df['group_size']<=thres0)]
        #### 上报字段 shangbao = 1/0
        thres_list = []
        for idx,gp in thres_df.groupby('labels'):
            if sum(gp['labels']>=1)>=1:
                thres_list.append(gp)
        ## 所有满足条件的簇df拼接起来
		if len(thres_list)==0:
			thres_df1=pd.DataFrame(col_use)
		else:
			thres_df1 = pd.concat(thres_list,axis=0)

        ####### 基础业务触发条件： 总流入金额/总流出金额：85%～115%，且月均交易金额 >= 20w,月均交易笔数 >= 5 笔 ###
        ### 环境变量参数获取：DC_RATE & t_avg & t_cnt/12  ## 
        ## 阈值获取 
        # thres1 = float(envX['DC_RATE1'])
        # thres11 = float(envX['DC_RATE2'])
        # thres2 = float(envX['t_avg'])
        # thres3 = float(envX['AVG_CNT_MONTH'])
        # #(thres1<=thres_df1['dc_rate']<=thres11)&
        # thres_df2 = thres_df1[(thres_df1['t_avg']>=thres2)&(thres_df1['t_cnt']/12>=thres3)]
        # #### 以下 用 or 条件 连接 
        # ####### 业务标签1： 现金交易 ###
        # ####### 月均现金金额/月均交易金额 >= 30% & 月均现金交易笔数/月均交易笔数 >= 30% ###
        # ##### CASH_AVG/t_avg & CASH_CNT/t_cnt
        # ### 环境变量参数获取：CASH_AVG_Divide_T_AVG,CASH_CNT_Divide_T_CNT ## 
        # thres4 = float(envX['CASH_AVG_Divide_T_AVG'])
        # thres5 = float(envX['CASH_CNT_Divide_T_CNT'])
        # thres_df2['CASH_AVG_Divide_T_AVG'] = round((thres_df2['CASH_AVG']/thres_df2['t_avg']),2)
        # thres_df2['CASH_CNT_Divide_T_CNT'] = round((thres_df2['CASH_CNT']/thres_df2['t_cnt']),2)
        # ####### 业务标签2： 凌晨交易 ###
        # ####### 月均凌晨交易金额/月均交易金额 >= 60% & 月均凌晨交易笔数/月均交易笔数 >= 50% ###
        # ##### LC_AVG/t_avg & LC_CNT/t_cnt
        # ### 环境变量参数获取：LC_AVG_Divide_T_AVG, LC_CNT_Divide_T_CNT## 
        # thres6 = float(envX['LC_AVG_Divide_T_AVG'])
        # thres7 = float(envX['LC_CNT_Divide_T_CNT'])
        # thres_df2['LC_AVG_Divide_T_AVG'] = round((thres_df2['LC_AVG']/thres_df2['t_avg']),2)
        # thres_df2['LC_CNT_Divide_T_CNT'] = round((thres_df2['LC_CNT']/thres_df2['t_cnt']),2)
        # ####### 业务标签3： 私转公 ###
        # ####### 月均私转公交易金额/月均交易金额 >= 50% & 月均私转公交易笔数/月均交易笔数 >= 50% ###
        # ##### IC_AVG/t_avg & IC_CNT/t_cnt
        # ### 环境变量参数获取：IC_AVG, t_avg, IC_CNT, t_cnt ## 
        # ####### 业务标签4： POS ###
        # ####### 月均POS消费金额/月均交易金额 >= 30% & 月均POS交易笔数/月均交易笔数 >= 50% ###
        # ##### POS_AVG/t_avg & POS_CNT/t_cnt
        # ### 环境变量参数获取：POS_AVG, t_avg, POS_CNT, t_cnt ## 

        # ####### 业务标签5： 交易对手 ###
        # ####### 月均交易对手数量 >= 70 & 月均交易金额 >= 100w ###
        # ##### OPP_AVG & t_avg
        # ### 环境变量参数获取：OPP_AVG, t_avg


        # ####### 业务标签6： 第三方交易 ###
        # ####### 月均第三方交易金额/月均交易金额 >= 30% & 月均第三方交易笔数/月均交易笔数 >= 30% ###
        # ##### D3_AVG/t_avg  & D3_CNT/t_cnt
        # ### 环境变量参数获取：D3_AVG, t_avg, D3_CNT, t_cnt
        # ### 跨境交易
        # ### 对手法人数量
        # ### 对手黑名单
        # thres_df3 = thres_df2[((thres_df2['CASH_AVG_Divide_T_AVG']>=thres4)&(thres_df2['CASH_CNT_Divide_T_CNT']>=thres5))
                             
        #                      |((thres_df2['LC_AVG_Divide_T_AVG']>=thres6)&(thres_df2['LC_CNT_Divide_T_CNT']>=thres7))           
        #                      ]
        return thres_df1
    ###### 对每个分行进行聚类分析 #####
    print('start minibatch kmeans !!!!!')
    start_time = time.time()
    org_df = list()
    for idx,gp in tqdm(df_use.groupby('suborgankey')):
        thres_org = kmeans_model(gp)
        print(thres_org)
        org_df.append(thres_org)
	org_all = pd.DataFrame(columns=['party_id','labels','suborgankey','group_size','dc_rate','t_avg','t_amt','t_cnt'])		
    if len(org_df)==0:
        org_all = org_all
    else:
        org_all = pd.concat(org_df,axis=0)
    org_all['dt']=dt
    col_use = ['party_id','labels','suborgankey','group_size','dc_rate','t_avg','t_amt','t_cnt','dt'] ##各项指标
    end_time =time.time()
    print('模型分析共计耗时：',end_time-start_time)
    ## 写出 csv 
    result = org_all[col_use]
    result.to_csv(result_file,index=False)
    print('finished minibatch kmeans !!!!!')

def main():
    parser = argparse.ArgumentParser(
        prog="Run model",
        description="This Python script runs your machine learning models.")
    parser.add_argument(
        '--source-file',
        dest="source_file",
        help="The source file used to do machine learning batch inference on.")
    parser.add_argument(
        '--result-file',
        dest="result_file",
        help="The result file name to store your batch inferred results.")
    parser.add_argument(
        '--dt',
        dest="dt",
        help="The result file name to store dt.")
    def run_model(args):
        print("Running your batch inference code...")
        source_file = args.source_file
        result_file = args.result_file
        dt=args.dt
        process(source_file=source_file, result_file=result_file,dt=dt)
    parser.set_defaults(func=run_model)
    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)

if __name__ == "__main__":
    main()