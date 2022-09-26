from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth(data, weight=0.9):  
    '''用于平滑曲线 类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重 处于0-1之间 数值越高说明越平滑 一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,path=None,tag='train',save_fig='True',show_fig=True):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    if save_fig == True:
        plt.savefig(f"{path}/{tag}ing_curve.png")
    if show_fig == True:
        plt.show()
    
def save_results(res_dic, tag='train', path = None):
    Path(path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res_dic)
    df.to_csv(f"{path}/{tag}ing_results.csv",index=None)
    print('Results saved!')
