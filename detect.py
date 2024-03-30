import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def numpy_euclidian_distance(point_1, point_2):
    array_1, array_2 = np.array(point_1), np.array(point_2)
    squared_distance = np.sum(np.square(array_1 - array_2))
    distance = np.sqrt(squared_distance)
    return distance


def num_sim(n1, n2):
  """ calculates a similarity score between 2 numbers """
  return 1 - abs(n1 - n2) / (n1 + n2)

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'model.pt') #loading the model
                                                    

# Images
img = 'image.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()
results.save()

print(results.pandas())
list=(results.pandas().xyxy[0])
# meters=list.filter(like='meter-boxes')
# print(list)
list2=list[list["name"]=='meter-boxes']
print(list2)
a=(list['name']).tolist()
x1=(list2['xmin']).tolist()
y1=(list2['ymin']).tolist()
x2=(list2['xmax']).tolist()
y2=(list2['ymax']).tolist()


center=[]


for i in range(len(x1)):
    temp=[0,0]
    xcor=(x1[i]+x2[i])/2
    ycor=(y1[i]+y2[i])/2
    temp[0]=xcor
    temp[1]=ycor
    center.append(temp) 

print(center)
res1=[]
res2=[]
im = plt.imread(img)
implot = plt.imshow(im)
for i in range(len(center)):
    x_cord = center[i][0]  #x cord
    y_cord = center[i][1]  #y cord
    res1.append(x_cord)
    res2.append(y_cord)
    plt.scatter([x_cord], [y_cord])
print("*******************************************************************************************")
# res1, res2 = map(list, zip(*center))
res1=np.array(res1)
res2=np.array(res2)
print(res1)
print(res2)
print("*******************************************************************************************")

# plt.plot(res1,c="hotpink")
# plt.plot(res2,c="red")
# plt.savefig('output.jpg',dpi=1200)


# plt.show()
meter=0
for i in a:
    if(i=='meter-boxes'):
        meter+=1
print(f"meter count is {meter}")
col=0
tem_center=center
col_cache=[0]*len(tem_center)
col_list=[]
counter=0
col_count=0
temp=[]
# idx={}
for i in range(len(tem_center)):
    for j in range(len(tem_center)):
                if(col_cache[i]==0):
                    if(num_sim(tem_center[i][0],tem_center[j][0])>0.90):
                        temp.append(center[j])
                        # print(tem_center[j])
                        if(i!=j):
                             col_cache[j]=1
    col_cache[i]=1
    if(len(temp)>0):
        col_list.append(temp)
    temp=[]
# print(col_cache)
print("Cols are:")
for i in col_list:
    print(i)
print("Col Count is ",len(col_list))



row_cache=[0]*len(tem_center)
row_list=[]
counter=0
col_count=0
temp=[]
# idx={}
for i in range(len(tem_center)):
    for j in range(len(tem_center)):
                if(row_cache[i]==0):
                    if(num_sim(tem_center[i][1],tem_center[j][1])>0.90):
                        temp.append(center[j])
                        # print(tem_center[j])
                        if(i!=j):
                             row_cache[j]=1
    row_cache[i]=1
    if(len(temp)>0):
        row_list.append(temp)
    temp=[]
# print(row_cache)
print("Rows are:")
for i in row_list:
    print(i)
print("Row Count is ",len(row_list))
# print(row_list.shape())





for i in range(len(row_list)):
     for j in range(len(row_list[i])-1):
        x_values = [row_list[i][j][0], row_list[i][j+1][0]]
        y_values = [row_list[i][j][1], row_list[i][j+1][1]]
        point1=[x_values[0],y_values[0]]
        point2=[x_values[1],y_values[1]]
        dist=round(math.dist(point1,point2))
        plt.plot(x_values, y_values, 'bo', linestyle="-")
        plt.text((x_values[0]+x_values[1])/2,y_values[0],str(dist))
        

for i in ((col_list)):
     for j in range(len(i)-1):
        x_values = [i[j][0], i[j+1][0]]
        y_values = [i[j][1], i[j+1][1]]
        point1=[i[j][0],i[j][1]]
        point2=[i[j+1][0],i[j+1][1]]
        dist=round(math.dist(point1,point2))
        plt.plot(x_values, y_values, 'bo', linestyle="-")
        plt.text(x_values[0],(y_values[0]+y_values[1])/2,str(dist))
plt.show()   
plt.savefig("output.jpg")
     
# hello(center)