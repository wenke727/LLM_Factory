#%%
import json
import pandas as pd

class SubtitlesParser:
    def __init__(self, subtitle_data):
        """
        Initialize the class with subtitle data.
        :param subtitle_data: List of subtitle data.
        """
        self.subtitles_df = pd.DataFrame(subtitle_data)

    def get_dataframe(self):
        """
        Returns the subtitles data as a pandas DataFrame.
        """
        return self.subtitles_df

    def get_subtitles_by_time(self, start_time=None, end_time=None):
        """
        Extracts subtitles within a given time range.
        :param start_time: Starting time (inclusive).
        :param end_time: Ending time (inclusive).
        :return: DataFrame of subtitles within the specified time range.
        """
        df = self.subtitles_df
        if start_time is not None:
            df = df[df['from'] >= start_time]
        if end_time is not None:
            df = df[df['to'] <= end_time]
        return df

    def concatenate_subtitles(self, start_time=None, end_time=None, sep=' ', prompt_style=False):
        """
        Concatenates the text of subtitles in a specified time range.
        :param start_time: Starting time (inclusive).
        :param end_time: Ending time (inclusive).
        :param sep: Separator for concatenating subtitles.
        :param prompt_style: If True, formats text into formal written language.
        :return: String of concatenated subtitles.
        """
        subtitles = self.get_subtitles_by_time(start_time, end_time)
        text = sep.join(subtitles['content'].tolist())

        if prompt_style:
            # Additional processing can be added here for formatting into formal language
            pass

        return text

# Create an instance of the class with the provided data
file_path = '../data/CLIP 改进工作串讲（下）.json'

with open(file_path, 'r', encoding='utf-8') as file:
    subtitles_data = json.load(file)

subtitles_data

#%%
subtitles_parser = SubtitlesParser(subtitles_data['body'])

df_example = subtitles_parser.get_dataframe()
df_example.query("tag == tag")

#%%
# 2. Getting subtitles for a specific time range
subtitles_example = subtitles_parser.get_subtitles_by_time(2262.77, 2440.97	)

# 3. Concatenating subtitles
concatenated_subtitles = subtitles_parser.concatenate_subtitles(2935, 3700, sep='|')

print(concatenated_subtitles)


# %%
"""
将下面的讲话整理成文字稿，要求：清理和预处理；语法和标点修正；修改错别字；适当简化和重组，变成书面用语；格式化和排版，根据内容分段；审查和复习；英文不翻译，直接输出文本即可
```
那说完了最新的这个 video transformer啊 | 最后呢 | 我们就来把之前讲过的所有东西呢 | 大概代串一下 | 首先呢就是从14年开始啊 | 最早的在 zlexnet 之后啊 | 把这个卷积 | 神经网络用到视频领域里来 | 就是 deep video 这篇论文 | 但是呢 | deep video 没有很好的利用这种运动信息 | 所以说他刚开始的效果呢不是很好啊 | 他还没有这种手工特征 idt 的效果好 | 所以说呢 | 这个双流网络的作者就想了一下 | 那为什么不把 idt | 里用的这种运动特征加进来呢 | 所以说这两个方法结合起来 | 就造就了这个双流网络 | 所以双流网络的结果呢就变得很高 | 那从此就开始了用这个深度学习 | 做这个视频理解的时代 | 那 | 因为双流网络证明了他的这个有效性 | 啊所以说接下来大家就在他之上呢 | 去做各种各样的改进 | 那首先最常见的一个改进呢 | 就是说对视频来说啊 | 我们想要更长的这种时序理解的能力 | 所以说我们加上这个 lstm | 就有了这个 cvpr15 | 的 beyond short snippts 这篇论文 | 然后呢 | 因为这个双流网络又是做的这种 | late fusion 啊 | 就在最后把这个结果 | 加做了个加权平均 | 那所以肯定就会有人想啊 | 我们能不能做一下 early fusion 呢 | 那于是就有了这个cvpr16的这个 | early fusion 这篇论文 | 然后呢因为双流方法里 | 只是简单的利用了这个光流啊 | 并没有真的根据光流的这个轨迹啊 | 去做这种视频特征的叠加 | 所以说在双流网络的基础上啊 | cvpr15的 tdd 这篇论文 | 就按照轨迹去直接加特征啊 | 得到了非常好的效果 | 那最后一点呢就是对于视频啊 | 我们想要处理这种长时间的视频信息 | 而且要让他变得更加的这个高效 | 这就有了 eccv 16的这个 tsn 这篇论文啊 | temporal segment networks | 就是把一个视频呢打成几个段 | 然后每一段你去做这个分析 | 最后把这个分析呢取一个合并 | 那因为 tsn 这个做的呢比较简单 | 所以很快在 tsn 的基础上 | 大家又纷纷的去做各种改进和尝试 | 就是把之前 | 传统学习里的这种全局建模啊 | 已经加了进来 | 那比如说全局建模 | 有这个 face reacting coding 啊 | vlad encoding 这些方法 | 那像这个dovf 呢 | 就是把这个 face reacting coding 用了进来 | 这个 tle 呢 | 就是把这个bi-linearing encoding 给用了进来 | 那这个 action vlad 呢 | 其实顾名思义就是把 vlad 给用了进来 | 那基本到这个阶段呢 | 其实这个2 d 的双流网络 | 就已经把 ucf 101和 hmdb | 51呢基本给刷的非常高了 | 也没有什么太多可以做的了 | 然后同时呢 | 也就是在2017年 | 这个 i 3 d 这个网络就出来了 | 那所以我们现在再来看3 d 网络这条线 | 在这个 deepvideo 提出这个 sports one million | 之后呢这个很多人就想啊 | 那既然这个视频是一个3 d 的输入 | 我们为什么不用一个3 d | 的网络来处理它呢 | 那用一个3 d 的卷积神经网络来学习 | 肯定听起来是一个更合理的事情 | 很快呢就有人做了这个 c 3 d 啊 | 就是纯粹的用一个3 d 神经网络来学习 | 视频而且因为有了 sports one million | 这么大一个数据集啊 | 他觉得他还是能训练出来一个 | 非常好的网络的 | 那事实上呢 | 这个网络抽取特征效果还可以啊 | 但是如果在一些刷分比较厉害的这个 | 数据集上呢 | 他跟别的方法效果还是差的比较远 | 那接下来大家就想啊 | 那为什么3 d 就是训练不好呢 | 可能还是因为初始化做的不好 | 那如果我能把这个初始化做的再 | 好一点这个训练啊 | 这个优化就会变得更简单啊 | 于是就有了 i 3 d 这篇论文啊 | 就是 inflated 3 d network | 它呢可以借助这个2 d 网络的初始化 | 和2 d 网络的这个结构 | 很快呢就能得到一个很好的结果 | 所以说呢 | i 3 d 呢 | 把 ucf 101和 hmdb 51这两个数据集呢 | 直接就给刷爆了啊 | 在那之后呢 | 大家就开始用这个 k 400啊去汇报结果 | 所以说在 3 d 网络这边呢 | 基本上所有的这些工作呢 | 都是在 k 400或者说 somethingsomething |  这几个数据集上去汇报结果 | 那在3 d 网络这边呢 | 一旦证明了这个 i3d 网络的有效性 | 那接下来呢就有几个方向 | 那首先第一个方向就是说 | 那你这个i 3 d 呢 | 是从一个2 d 的inceptionnet 搬到3 d 的 | 那其实我也可以把别的2 d 网络搬过来 | 那所以说把一个2 d 的 resnet 搬过来 | 就成了 r 3 d 啊 | 把一个2 d 的 resnext 的搬过来就是 mfnet | 那如果把一个2 d 的 senet 搬过来啊 | 那就是 stc | 那第二个思路呢 | 就是说这个2 d 网络呢表现也不错 | 而且呢2 d 比3 d 呢要便宜很多 | 那我们能不能把这个2 d 和3 d | 结合起来使用呢 | 所以这也就有了接下来几个工作啊 | 就是 s3d 、r（2+1）d、eco 和这个 p3d | 他们的想法呢都很像啊 | 基本都是把这个3 d 呢给 p 成2 d 和1 d 了 | 准确度上呢没什么降低 | 但是这个计算效率和训练速度上呢 | 都有了大幅度的提高 | 那第三个方向呢 | 当然跟2 d 或者双流网络也一样啊 | 我呢想去尝试这种 lstm | 或者尝试这种 tsn | 去处理更长的这种时间序列 | 那所以说呢也有对应的一系列工作 | 比如说这个 ltc | 就已经使用了120帧所输入 | 而接下来这个 t3d 啊non local、 v4d 啊 | 这些工作 | 通通都是为了能处理 | 更长的这个时序信息 | 而设计的 | 那最后一个方向呢就是做这 | 种高效率的 | 比如说这种 csn 啊就是 channel separate network | 还有我们今天刚才说过的这个 slow fast | 啊还有通过这个 auto ml | 搜出来的是 x3d 这个网络啊 | 非常小但是效果也很好 | 然后呢因为 x 3 d 啊 | 使用了这种auto ml的方式 | 去搜索这个网络 | 所以说搜出来的这个网络呢 | 不仅效果好 | 而且参数量呢特别的小 | 那基本上就很难再有人能打过他了 | 那正当大家发愁 | 这个视频理解该怎么往下做的时候呢 | 大救星来了 | vision transformer出现了 | 所以大家迅速就抛弃了3 d 网络啊 | 全去搞vision transformer | 因为反正3 d 网络我也刷不动 | 那在短短几个月的时间里呢 | 在2月份就有了times former 啊 | 3月份有了 vivit | 然后4月份呢 | 我们的 vidtr |  和这个 facebook mvit 也就都放出来了 | 基本上方法呢都差不多 | 就说这个 joint spacetime | attention啊这种合并起来的 | 这个时空自注意力呢 | 太贵了啊 | 这个视频里呢 | 直接做这种transformer做不起 | 所以我们就要拆分它 | 那要么我先做一下时间 | 再做一下空间啊 | 就跟 r （2+1） d 一样在这个维度上去拆分 | 要么呢我就是先做一下局部 | 再做一下全局 | 就是在用小窗口大窗口之上啊 | 去拆分一下 | 总之呢 | vision transformer在视频理解里的应用呢 | 还是比较初级的 | 我能想象到的方向呢 | 就是利用transformer | 这种长时间序列建模的能力啊 | 去做这种长时间 | 的视频理解 | 而且呢还有就是这种多模态的学习 | 还有这种自监督的学习 | 总之呢 | 虽然我关注了视频学习这么多年 | 但我觉得呢 | 视频领域 | 还是处在一个非常初级的阶段 | 接下来呢有太多太多可以做的 | 而且有太多太多需要做的任务了 | 我虽然不知道啊 | 真实的这个视觉应用里 | 到底有多少需要这种视频理解 | 但是我非常赞同Andrej Karpathy | 之前在twitter上说过的一句话 | 如果想 | 训练一个非常强大的这个视觉模型 | 拿视频数据做输入 

"""

#%%%
""" 总之呢 3 d 网络从17年 i 3 d 火了之后呢 | 就一直霸占这个视频理解领域 | 一直到2020年 | 然后自从有了vision transformer啊 | 大家这个注意力呢 | 又全都转移到这上面了 | 所以说 | 就到了我们今天要讲的第四个部分 | 啊就是 video transformer | 咱们把在图像上用的比较好的 | 这个vision transformer啊 | 能移植到视频领域里来 | 做一个视频的vision transformer | 那video transformer这篇呢 | 我们就准备讲一篇啊这个 times former 啊 | 也是最早的一篇把vision transformer | 用到视频理解领域的一篇论文 | 他这个题目呢也是让人一看就懂啊 | 时空注意力啊 | 是不是 all you need 啊 | 这个呢 | 就跟2017年transformer的原文是一样的 | 自注意力啊 | 是不是 all you need | 那这篇论文呢 | 我们之前在讲 r （2+1） d 的时候也说过 | 是跟 r （2+1） d 的作者团队呢差不多的 | 那这篇论文呢 | 也是一间比较实验性的论文 | 就是探索了一下 | 怎么把这个vision transformer啊 | 从这个图像领域迁移到视频领域里来 | 那具体来说呢 | 作者就尝试了这五种结构 | 那第一种结构呢 | space attention啊 | 就是只在这个空间特征图上 | 去做这种自注意力 | 那这个呢 | 其实就是图像里啊 | 这个vision transformer 用的这个自注意力 | 意思就是说你一个这个特征进来 | 做一次这个空间自注意力 | 然后再接一个 mlp 啊 | 然后通过残差连接最后得到这个输出 | 那这个就属于是图像上的baseline了 | 那接下来呢 | 就是说怎么用到这个视频上 | 那首先视频上第一个呢 | 就是我暴力地去 使用对吧 | 那我就是给一个视频 | 那我就在这个视频 | 三个维度上通通去做这个自注意力啊 | 我一起做啊 |也就是他这里说的这个joint space-time |  attention  | 那这样呢 | 一个输入进来过这个时空自注意力 | 然后再过一个 mlp | 然后通过残差连接 | 就得到最后的输出了 | 但是这种方式呢 | 其实我们肯定就能想到这个 | gpu 内存肯定塞不下 | 呀 | 那图像那边vision transformer都快塞不下了 | 那更别提你现在把这个视频啊 | 30帧、60帧扔给一个vision transformer | 你在三个维度上去算这个自注意力 | 他肯定是塞不下的 | 所以你一旦这个 gpu 内存塞不下 | 那该怎么办呢 | 那其实就像R(2+1)D 做的一样| 你3D塞不下我就把你拆分了吗| 我就把你拆成2D 和1D就行了 | 那这里呢其实还是同样的套路 | 那你既然一个3D的 | 这个时空的自注意力你做不动 | 那我就在时间上先做自注意力 | 然后再在空间上再做自注意力 | 不就行了吗 | 所以说啊就按照这个 R(2+1)D的方式啊| 作者就提出了他们这篇论文里的这个 | divided space-time attention | 就是把这个 时空啊分开来做啊 | 这里呢他没有用 factorize 个词 | 他换了个词换成 divided | 但其实呢都是一个意思 | 具体做法呢就是你一个特征进来 | 那我先在这个时间维度上 | 给你做一下自注意力 | 然后残差连接 | 然后我再在空间上呢 | 做一下这个自注意力 | 然后最后过一个 mlp 啊最后得到输出 | 这样这个计算复杂度呢就大大降低了 | 因为你每一个这个序列 | 长度都变得很小 """
 
"""" 当然了除了这种 | 时间和空间上的这种拆分之外呢 | 你还可以有很多别的拆分方式 | 那比如之前大家常用的呢 | 就是这种 local global 的形式 | 就是既然你全局的这个序列长度太长 | 你没法算啊 | 那我就先在局部算 | 那算好以后呢 | 我再在这个全局上去算啊 | 这个呢其实就有点这个swin | transformer的意思啊 | 我在一个局部的一个小窗口里 | 去算这个自注意力 | 他这个计算复杂度呢也会变得很低 | 所以说这里就是说我先一个特征进来 | 我先局部算特征 | 然后我再去全局算特征 | 那这个复杂度也就会降低 | 最后一种呢也就是这个 axial attention | 就是说我只沿着你特定的这个轴 | 去做attention | 这个 | 之前我们在讲vision transformer的时候呢 | 也提到过 | 就是说呢我一个特征进来 | 我先沿着这个时间轴去做自注意力 | 然后我再沿着这个横轴 | 啊就是这个图像的宽去做自注意力 | 然后再沿着这个图像的高度啊 | 这个竖轴去做自注意力 | 那意思就是说把一个三维的问题呢 | 拆成了三个一维的问题 | 那这个复杂度呢 | 一下也会降的很低很低 | 但总之呢你一看这个图 | 你就会立马联想起来 | 跟 r 2+1 d 这篇论文的联想 | r 2+1 d 呢也画了5个图 | 分别讲了讲这个2 d 和3 d 该怎么融合啊 | 到底哪种结构好啊 | 那这篇timesformer 呢 | 他们也是画了5个图啊 | 讲了讲这个vision transformer | 怎么能用到视频里来 | 套路呢是非常相似的 | 然后呢反正是实验性论文 | 可以写的这个方法部分呢并不多 | 这作者呢为了让这个读者更好理解 | 所以把刚才的五个图呢 | 又用可视化的方式展现了出来啊 | 形象的告诉你 | 这五种自注意力的方式 | 到底是怎么做的 | 这个图啊确实画的也很好 | 首先我们来看啊 | 就是单纯用这种图像上的这种 | 空间自注意力 | 那意思就是说啊如果我们有啊 | 前一帧那 frame t 减1啊 | 中间这一帧 frame t | 和下一帧 frame t 加1的话呢 | 如果我们拿这个蓝色的 patch 当基准点 | 那其实他会跟谁去算这个自注意力呢 | 他只会跟他当前的这一帧 | 去算自注意力 | 因为他就是一个图像的自注意力 | 他不牵扯这个时间轴上的 | 所以说呢他看不到前后这两帧 | 他只是跟本帧上所有的别的 patch | 去做这个自注意力操作 | 那接下来的这四种方法呢 | 这全是跟视频有一些关系了 | 那比如说接下来的这个啊 | 在时空上一起做自注意力的方式 | 那这个当然是最好了 | 因为我们可以看到 | 如果拿这个蓝色的patch当基准点 | 前一帧 | 这一帧和后面一帧所有的这个patch呢 | 他都要去做自注意力 | 效果肯定应该是最好的 | 但是这个复杂度呢也是你承受不起的 | 那我们就来看看后面三种 | 这个简化方式是怎么做 | 那第三种呢 | 也就是他们这篇论文中 | 主要提出的方法啊 | 这个 divided 啊 | space time 自注意力 | 那同样呢 | 我们还是以这个蓝色的 patch 作为基准 | 那先做这个时间上的自注意力呢 | 也就是说他先做这个patch对应的 | 前一帧的这个patch上啊 | 他去跟这个patch去做自注意力 | 然后同样呢 | 在下一帧同样的位置上啊 | 这个patch上去做自注意力 | 也就是这个这个这个他们在 | 做这个自注意力操作 | 他跟前一帧剩下所有的这些 patch | 和下一帧所有这些 patch 呢 | 都是不做主自注意力的啊 | 没有关系 | 那做完了这一步 | 这个时间上的自注意力 | 接下就该做空间自注意力了 | 那这个空间自注意力呢 | 就跟第一个 | 这个2 d 的空间自注意力是一样的啊 | 他跟这个图像里的所有的别的 patch | 都会去算一下自注意力 | 那接下来第四种方式呢 | 这种先局部再全局的方式 | 就说首先 | 我先去算这个局部 | 小窗口里的自注意力 | 那如果我们拿这个蓝色 patch | 当这个基准点 | 那对于当前帧来说呢 | 我们就算这个小窗口里 | 那前一帧呢也是这个小窗口 | 后一帧呢也是这个小窗口 | 那当算完了所有这个小窗口之后呢 | 接下来就该算这个全局的了 | 全局呢为了减少这个计算量 | 他就得变得非常的稀疏啊 | 所以也就是这些点 | 是要去做这个自注意力操作的 | 那最后呢就到了这个轴自注意力 | 那也就是说 | 把一个三维问题变成了三个一维问题 | 啊这里呢也非常形象啊 | 就说如果这个是基准 patch的话呢 | 他的时间上 | 是跟前后这两层绿色去做自注意力 | 那他在横轴上呢 | 是跟这个黄色的去做自注意力 | 在竖轴上呢 | 是跟这个紫色去做自注意力 | 那其实呢这个 axial attention | 也就是这个 divided | spacetime attention 的一个极端情况 | 就说为了进一步的减少这个消耗呢 | 那在当前帧上呢 | 也不做所有patch的这个自注意力 | 而是专门去选这个轴上去做 | 但这个效果呢肯定是会有trade off | 总之呢 | 作者就是详尽的把这些设计方案呢 | 全部都试了一遍 | 看看到底哪个设计方案 | 最后的效果最好 | 最后呢就依据中间的这个divided space time | attention提出了他们这个 times former 的架构 """
 
 
"""| 那我们简单来看一下这个消融实验 | 那消融实验这个表1里呢 | 就是把刚才说的这五种设计方案呢 | 全对比了一下 | 那从结果上来看呢 | 这个 divided space time 呢 | 确实取得的效果是最好的 | 那在 k-400上呢 | 单纯用这种2d 的这种自注意力 | 效果也不错啊 | 或者说这种 joint 的这种时空自注意力 | 效果也不错 | 但是呢 | 单纯用2d 的这自注意力之所以效果好 | 是因为 k-400这个数据集呢 | 是一个比较偏重这个静态图像的 | 那如果我们换到这个 SSv2 | 数据集上了 | 我们就可以看到 | 光用这个2 d 的这个自注意力呢 | 效果一下就下降了很多啊 | 我们还是要把这个时间维度考虑上 | 所以说呢 | 唯一表现比较好的 | 就剩下这个 joint space time 和 divided | space time 这两种方式 | 但是我们之前也说过啊 | 这个joint的这种时空的自注意力啊 | 他的这个内存消耗是非常巨大的啊 | 作者也就在这个图三里画了一下 | 就是如果我们把这个视频 | 帧的大小变大啊 | 从224变成336,448或者更大呢 | 他们这个 timesformer | 这个计算量的增长呢 | 是基本成线性的 | 但是这个 joint 的自注意力呢 | 就会增长得非常快啊 | 尤其是后面这个灰色部分呢 | 其实就已经是 gpu | out of memory 就已经爆显存了 | 其实是没法进行训练的 | 那同样的道理 | 如果我们把这个输入加长 | 从8帧变成32帧一直到最后的96帧 | 我们可以看到这个 joint space time attention | 其实从32帧开始就已经爆显存了 | 你也是没法训练的 | 所以最后总结下来就是说啊 | 即使这个joint space time | 他的这个结果呢也不错啊 | 而且他这个参数量呢稍微少一点 | 但是呢因为他这个占用的 gpu | 显存实在是太大了 | 我们根本训练不动 | 那最后最好的方案呢 | 就只剩下这个divided space time attention了 | 那最后呢我们来看一下结果 | 首先呢我们来看一下这个表2 | 作者这里呢就是对比一下times former啊 | 和之前的这两种3d 网络 |I3D呢是最经典的啊 | slow fast呢是最新的啊 | 所以说他就跟这两种就做了一个对比 | 那其实我们这里可以发现啊 | 如果单纯从这个 | 准确度的角度上来说呢 | 啊这个 times former 呢不是太行 | 所以他这里啊 | 用这个78黑体 说他们是最高的 | 但我们也知道这个 slow fast | 如果你换成 res 101啊 | 或者换成 non local 之后呢 | 他这个准确度已经到80了 | 那作者这里呢 | 为了突出使用这个vision transformer的好处 | 作者这里就发现啊 | 他的这个训练的时间啊 | 和这个做推理的 | 这个 flops 呢都变低了 | 那比如说训练时间的话呢 | slow fast一个 res 50就要训练6,000多个小时 | 但是他们这个 times former 呢 | 只需要训练400个小时 | 这个vision transformer 拿来做微调啊 | 这个确实效果非常好 | 那推理上呢 | 因为transformer需要的这个输入更少 | 所以说他推理时候用的这个 t flops 呢 | 也更低也就说啊从效率角度而言 | times former是优于之前的这个3 d 网络的 | 但是呢大家也知道啊 | cv 领域的人呢一般只认这个效果啊 | 不太认这些效率这些东西啊 | 所以说作者肯定还是得再刷一刷分的 | 那作者呢 | 就是把这个 timesformer 变得更大啊 | 用了 timesformer large | 然后又用了image net21 k | 去做预训练 | 最后呢还是把 k 400刷到这个上80了 | 那这样子的话呢就别人无话可说了 | 这个效果确实是最好的 | 在ssv2上呢 | 这个结果也是不错 | 总之呢作为第一篇 | 把这个vision | transformer 用到视频理解领域来 | 本文的结果呢其实已经算不错了 """

""" 接下来很快啊 | 我们组这边呢也有一篇啊 | vidtr | 也是用类似的思想去做video transformer 的 | 然后 facebook 那边呢还有另外一片 mvit 啊 | multi scale vision transformer | 也是做video transformer的 | 效果会更好 | 然后 google 那边呢还有一个 vivit | 想法呢都差不多 | 都是把这种时空自注意力呢拆分 | 只不过用了拆分的方式呢不太一样 | 所以其他的这些工作呢 | 我这里就不讲了 | 那最后呢我们来总结一下这篇论文 | 那其实作者在这里也总结说 | 他们的这个times former呢有四个好处啊 | 第一个呢就是想法非常的简单 | 第二个呢就是效果非常好啊 | 在好几个 | 这个动作识别的这个数据集上呢 | 都取得了最好的效果 | 第三个呢就是不论是训练啊 | 还是在做推理的时候 | 他这个开销呢都是非常小的 | 这点对于视频理解来说 | 还是非常具有吸引力的 | 然后最后呢其实我并没有讲 | 就是说这个times former呢 | 可以用于处理超过一分钟的视频 | 也就意味着说呢 | 我们之后可以做这种长时间的这 | 个视频理解了 | 这个呢 | 其实视频理解里非常重要的一个部分 | 啊但是之前呢很少有工作做 | 因为确实这个训练成本啊 | 各方面都非常的高啊 | 这个任务也比较困难 | 而且也没有合适的一个数据集啊 | 去做这种测试 | 所以说啊综合以上这些优点 | 而且vision transformer | 们在视觉这方面大杀四方 | 这个扩展性稳健性都这么的好 | 我觉得接下来呢 | video transformer 确实是大有可为 | 而且更重要的是呢 | 视频本身就是一个多模态的信号 | 他可能会有字幕啊这种文字信息 | 然后他可能会有这种音频信息 | 然后你如果从中抽取这种光流 | 或者抽取这种深度图啊 | 他就是属于别的这种多模态信息 | 他自己呢 | 就是一个非常丰富的这个输入来源 | 从中啊可以设计成各种各样的这个自 | 监督信号 | 如果我们再和这个transformer结合 | 起来很有可能就能达到 cv 里面啊"""

