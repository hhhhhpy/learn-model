import torch
#anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_sizes = ((32,))
aspect_ratios = ((0.5,1.0,2.0))
anchor_sizes = torch.as_tensor(anchor_sizes,dtype=torch.float32,device="cpu")
aspect_ratios = torch.as_tensor(aspect_ratios,dtype=torch.float32,device="cpu")
#print(aspect_ratios)
h_ratios = torch.sqrt(aspect_ratios)
w_ratios = 1 / h_ratios
#print(anchor_sizes[:,None].shape)
ws = (w_ratios[:, None] * anchor_sizes[None, :]).view(-1)

hs = (h_ratios[:, None] * anchor_sizes[None, :]).view(-1)
#print(hs)
base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

print(base_anchors.round().shape)
"""
grid_anchors
"""
shifts_x = torch.arange(
                0, 7, dtype=torch.float32, device="cpu" ) * 2 #2:stride *stride对应的是原图
shifts_y = torch.arange(
                0, 8, dtype=torch.float32, device="cpu") * 2
shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
print(shift_y)
shift_x = shift_x.reshape(-1)
shift_y = shift_y.reshape(-1)
shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
print((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).flatten(0,-2).shape[0])

print(shifts.shape)
anchors = [[1,2,3,4],[2,3,4,5]]
gt=[[1,2,3,4]]

x = torch.rand(2,3,4)
y=x.split([1,1,1],dim=1)
print(y[1].shape)

a=torch.full((4,),2)
print(a)

b=torch.rand(2,3,4,5,6)
_,idx = b.topk(2,dim=1)
print(idx)
print(b.flatten(0,-2).shape)

boxes1=torch.rand(4,1,1)
print(boxes1)
print((boxes1>=0.5).to(dtype=torch.float32))
boxes2 = torch.rand(3,1,1)
#lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
gt_box=torch.rand(6,4)
match_idx=torch.tensor([0,1,1,2])

print(gt_box[match_idx])

x=torch.rand(4,1)
print(torch.flatten(x).shape)