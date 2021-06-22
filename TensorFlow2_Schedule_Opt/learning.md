逻辑：
encoder:
仓库embedding、与【顾客，需求量】的embedding进行拼接

#访问过的就标记为True,depot会多次访问，因此mask会变动
mask=[depot_mask,customer_mask]
step_context:[node_embedding,D_embedding]#D denotes remaining vehicle capacity

decoder:

#######
场景几乎和船队排入多个闸室的场景一样：
闸室作为第一个节点（类似仓库），船队中的每一个船只都是一个其它节点（类似顾客）
初始时，闸室面积未被利用，船队船也未被选中，然后选则一只船，按照规则放入闸室；更新闸室上下文（剩余空间位置），放不下的船
会把mask掉（与get_mask_D类似）