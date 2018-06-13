# 6D-PoseEst


识别出目标的位置跟姿态。
之前看到过好多bin picking的展示，就是一个箱子里杂乱放着某一类零件，能够识别并把零件抓出来摆好。但是，这方面的开源代码非常少，
我就找到个linemod的，只能眼巴巴看着，完全不知道怎么自己弄。
好在这方面的论文还是不少的，实在不行可以慢慢把论文复现出来。我发现这方面主要分为4种方法。

第一种是基于霍夫变换的。这个很好理解，随便选点猜个姿态，最后投票多的就是。
这个方向经典的有2010年的论文 ppf (point pair feature)：Model  Globally,  Match  Locally:  Efficient and Robust 3D Object Recognition

第二种是基于模板匹配的。基本思路很简单，拿到物体的模型，从各个方向提取到RGBD图像，经过九九81道工序腌制成模板，然后拿去在实际图像每一个位置匹配，
配上了就知道位置跟姿态了。
这方面最经典的是2011-12年的linemod算法(论文有2篇，一篇基础的，一篇稍微改进的)。linemod的效果要好于之前的ppf，最可贵的是linemod在opencv里有实现的源码
在基于ros的object recognition kitchen中有配套的模型生成图像的程序、ICP后处理的教程跟代码。

第三种是基于patch匹配+random forest的。linemod还是有些缺点的，比如有遮挡的时候识别率就会下降。那么一个很自然的想法就是，
我把原来的训练图像拆成好几个patch，每个patch做成模板，然后拿去配测试图像。Latent-Class Hough Forests(论文也有两篇)就大概这么个思路，
用random forest做了点改进：如果单纯按照原来的样子匹配，模板量得好几倍才行。
通过比较模板之间的相似性训练random forest，就像把一堆数据用二叉树存起来，大大提高匹配效率。当然，在这中间还有一些小技巧，
在匹配结束后也有一些后处理的方法，这里就不谈了。

基于这个思路的论文就很多了，举几个例子：Learning 6D Object Pose Estimation using 3D Object Coordinates不要linemod了，
用pixel difference作为feature度量相似性，然后用random forest。Recovering 6D Object Pose and Predicting Next-Best-View in the Crowd
为啥要手撸feature呢？用auto encoder搞出个embedding来度量相似性，然后forest。
当然，我觉得更好的方法是用人脸识别、行人重识别那一套，什么Siamese Network啊，triplet loss啊，
因为这个相似性度量就是要搞成类内距离小，类间距离大的一个效果，所以完全是这些领域的拿手好戏嘛。反正不管怎样，
把embedding搞出来，然后random forest快速匹配一下就行了(或者也有别的快速匹配embedding的方法？
比如搞人脸识别的大公司这么多数据是怎么匹配的呢)

第四种就是现在的end-to-end大法了：反正我搞一个超大CNN，管他怎么回事，一锅全给炖了。基于这种思路有论文起了个很皮的标题：
SSD-6D: Making RGB-based 3D detection and 6D pose estimation great again虽然论文有这么多，但是开源的特别少，除了Linemod之外几乎没有。
