import time
import ray

import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from hyper_parameters import *
from helper_func import *

import tensorflow as tf
import glob
images_folder     = './VOC2007/JPEGImages/'
ray.shutdown()
ray.init()
def get_train_val_test_fnames(path_to_filenames):

    all_files = {}    
    
    
    with open(path_to_filenames+ 'train.txt') as file:
        lines = file.read().splitlines() 
        for line in lines:
            all_files[line]= 'train'
            
    with open(path_to_filenames+ 'test.txt') as file:
        lines = file.read().splitlines()
        for line in lines:
            all_files[line] = 'test'
            
    with open(path_to_filenames+ 'val.txt') as file:
        lines = file.read().splitlines()
        for line in lines:
            all_files[line] = 'validate'
            
    return all_files



def parse_for_annotation(xml_file,cat2idx):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
#        print(images_folder)
        filepath= images_folder +filename
        
        width =int(root.find('size').find('width').text)
        height =int( root.find('size').find('height').text)
        depth =int(root.find('size').find('depth').text)
 
        ymin_list = []
        xmin_list = []
        ymax_list = []
        xmax_list = []
        cat_id = []
        cat_name = []
       
        for boxes in root.iter('object'):
            
            cat_name.append(boxes.find('name').text)
            cat_id.append(cat2idx[boxes.find('name').text])
            
            box = boxes.find("bndbox")
            
            ymin_list.append( int(box.find("ymin").text) )
            xmin_list.append( int(box.find("xmin").text) )
            ymax_list.append( int(box.find("ymax").text) )
            xmax_list.append( int(box.find("xmax").text) )
        
        
        annotation = {'filename': filename,
                      'filepath': filepath,
                      'height':height,
                      'width':width,
                      'depth':depth,
                      'cat_name': cat_name,
                      'cat_id' : cat_id,
                      'xmin':xmin_list,
                      'ymin':ymin_list,
                      'xmax':xmax_list,
                      'ymax':ymax_list
                }
        
        
        return annotation

def _bytes_feature(value):
  """Python string,bytes to Byteslist converter"""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _StrList2Bytelist(value):
    lst =[]
    for val in value:
        lst.append(val.encode())
#        tf.train.BytesList(value=annotation['cat_name'])
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=lst))

def generate_example(annotation):    
    with open (annotation['filepath'],'rb') as imgfile: #image file path
        imgdata = imgfile.read()
        
    feature = {
        'image/height':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[annotation['height']])),
        'image/width':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[annotation['width']])),
        'image/depth':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[annotation['depth']])),
        'image/object/bbox/xmin':
        tf.train.Feature(float_list=tf.train.FloatList(value=annotation['xmin'])),
        'image/object/bbox/ymin':
        tf.train.Feature(float_list=tf.train.FloatList(value=annotation['ymin'])),
        'image/object/bbox/xmax':
        tf.train.Feature(float_list=tf.train.FloatList(value=annotation['xmax'])),
        'image/object/bbox/ymax':
        tf.train.Feature(float_list=tf.train.FloatList(value=annotation['ymax'])),
        'image/object/class/label':
        tf.train.Feature(int64_list=tf.train.Int64List(value=annotation['cat_id'])),
        'image/object/class/text':
        _StrList2Bytelist(annotation['cat_name']),
        'image/encoded':
        _bytes_feature(imgdata),
        'image/filename':
        _bytes_feature(annotation['filename'].encode())
        }
        
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example






##Can try to write 2 shards parallely by ray process in future
#@ray.remote
#def write_single_file(sample):
#    pass
@ray.remote(num_cpus=4)
def write_single_file(record,path,cat2idx):
    
    with tf.io.TFRecordWriter(path) as writer:
        for sample in record:
            annotation = parse_for_annotation(sample,cat2idx)
            example = generate_example(annotation)                                  
            writer.write(example.SerializeToString())
    
    return path
    

def write_records(anno_list, num_shards, record_name,cat2idx):

    # we split annotation list into multiple records and write each to a file
    num_samples = len(anno_list)
    split_len = num_samples//num_shards
    records = []
    
    for i in range(num_shards):
        start =i*split_len
        end = start+split_len
        records.append(anno_list[start:end])

    output = []
    for i,record in enumerate(records):
        path = record_name +str(i+1)+"_of_"+str(num_shards)+".tfrecords"
        path = write_single_file.remote(record,path,cat2idx)
#       status = ray.get([write_single_file.remote(sample) for sample in records])      
        output.append(path)
    output = ray.get(output)
    for path in output:
        print("Successfully written File: "+path )

    return record_name




def main():
    #get train test and validate filenames
    all_files = get_train_val_test_fnames(path_to_filenames)
    
    #get categories and resp id
    with open('./categories.txt') as file:
        lines = file.read().splitlines()
        cat2idx = {cat:i for i,cat in enumerate(lines)}
#    print (cat2idx)
    training_annotation = []
    testing_annotation = []
    validation_annotation = []
#    print(len(training_files),training_files.keys())
#    annotation data is small so lets read all of them and store in ram
    for file in os.listdir('./VOC2007/Annotations'):
#        print(file)
        fname = file[:-4]
        if(all_files[fname] == 'train'):
            training_annotation.append('./VOC2007/Annotations/'+file) 
        elif(all_files[fname] == 'test'):
            testing_annotation.append('./VOC2007/Annotations/'+file)
        elif(all_files[fname] == 'validate'):
            validation_annotation.append('./VOC2007/Annotations/'+file)
        else:
            print("ERROR_:  FILE JUST HAS ANNOTATION ! NO IMAGE")
#    
#    #lets write datasets and annotations to tf records
#    #config TFrecords
    no_of_train_shards = 2 
    no_of_test_shards = 1
    no_of_val_shards = 1

#    #pics up image file from disk as per annotation write it in 
#    #tfrecord along with annotation data
    #code is commented cause one should not use this main again and again
    # it burdens my machine 
    print("uncomment me to generate tfrecords but it will kill your machine")
#    write_records(training_annotation,no_of_train_shards,"train_record",cat2idx)
#    write_records(testing_annotation,no_of_test_shards,"test_record",cat2idx)
#    write_records(validation_annotation,no_of_val_shards,"val_record",cat2idx)

if __name__ == "__main__":
    main()




#dataset = tf.data.Dataset.list_files(glob.glob(images_folder+'/*.jpg'))
#
#def parse_image(filename):
#  image = tf.io.read_file(filename)
#  image = tf.image.decode_jpeg(image)
#  image = tf.image.convert_image_dtype(image, tf.float32)
#  image = tf.image.resize(image, [IMAGE_W, IMAGE_H])
#  return image,filename
#
#img_dataset = dataset.map(parse_image)
#for elem in img_dataset:
#    print( elem.shape)
#    break
#it = iter(dataset)
#print(next(it).numpy())

#img_it = iter(img_dataset)
#print(type(next(img_it).numpy()))
#man,fname = next(img_it)
#img = (man.numpy()*255).astype('uint8')
#img = Image.fromarray(img)
#img.show()
#print(fname)
#file_path = next(iter(list_ds))
#image, label = parse_image(file_path)














































def drawBox(boxes, image):
    for i in range(0, len(boxes)):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[i].x_min, boxes[i].y_min), (boxes[i].x_max, boxes[i].y_max), (255, 0, 0), 2)
    cv2.imshow("img", image[...,::-1])#back convert to bgr from rgb to display
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def box_scale(box,width,height):

    targetSize =IMAGE_H
    x_scale = targetSize / width
    y_scale = targetSize / height

    xmin = int(np.round(box.x_min * x_scale))
    ymin = int(np.round(box.y_min * y_scale))
    xmax = int(np.round(box.x_max * x_scale))
    ymax = int(np.round(box.y_max * y_scale))
    b = Box([xmin,ymin,xmax,ymax])
    return b







class Batch_Generator():
    def __init__(self):   
         self.X = np.zeros((BATCH_SIZE,IMAGE_W,IMAGE_H,NUM_CHANNAL))
         self.Y = np.random.rand(BATCH_SIZE,GRID_DIM*GRID_DIM*NUM_ANCHOR*(5+NUM_CLASSES))
         self.counter= 0    
         self.image_files = os.listdir(images_folder)
         self.num_train = len(self.image_files)

    def next_batch(self):              
       temp = self.image_files[self.counter:self.counter+BATCH_SIZE]  
       
       for i,image in enumerate(temp):
         label = os.path.join(annotation_folder, image[:-4]+'.xml')
         name, boxes, cat = read_content(label)
         y_vector = self.generate_y_vec(boxes,cat,name) 
         data =cv2.imread(os.path.join(images_folder,image),cv2.IMREAD_COLOR)
         data = data[...,::-1]  #convert from rgb to bgr
         
#         print(cat)
         if data is not None:
             image_data =  (np.ndarray.astype(data,dtype = 'float32')-(255.0 / 2)) / 255.0
             image_data = cv2.resize(image_data, (IMAGE_H, IMAGE_W))
#             print(image_data.shape)
#             out = drawBox(boxes,image_data)
         self.X[i] = image_data
         self.Y[i] = y_vector.flatten()
       
       self.update_counter()
       return self.X,self.Y 
         
    def update_counter(self):
        self.counter = (self.counter +1)%(self.num_train)      
    
    
    def generate_true_boxes(self,boxes,cat,name):     
       total_grids = GRID_DIM*GRID_DIM
       
       y_box = []
       for i in range(total_grids):
           small_box=[]
           for j in range(NUM_ANCHOR):
               a =YOLO_Box(np.random.rand(4),i,ANCHORS[j],cat[0],0)
               small_box.append(a)
           y_box.append(small_box)
       
       
       
       index_list = []
       for box in boxes:
           index_list.append(box.calc_index())
       

       c=zip(index_list,boxes)
       boxes = sorted(c, key = lambda x:x[0])
       boxes = [b[1] for b in boxes]
       
       c=zip(index_list,cat)
       cat = sorted(c, key = lambda x:x[0])
       cat = [b[1] for b in cat]
       index_list.sort()
       print(boxes)
              
       for i in range(total_grids):
            count = index_list.count(i)
            
            if(count == 1):
                box =boxes[index_list.index(i)]
                c = cat[index_list.index(i)]
                for n in range(NUM_ANCHOR):
                       box.to_yolo_box(ANCHORS[n])
                       box.objectness = 1
                       box.category=np.zeros(NUM_CLASSES)
                       box.category[categories2idx[c]] = 1
                       box.classname = c
                       y_box[i][n].copy_this(box)
            if(count > 1):
                first = index_list.index(i)

                for n in range(NUM_ANCHOR):
                       box =boxes[(first+n)%count]
                       c = cat[(first+n)%count]
                       box.to_yolo_box(ANCHORS[n])
                       box.objectness = 1
                       box.category=np.zeros(NUM_CLASSES)
                       box.category[categories2idx[c]] = 1
                       box.classname = c
                       y_box[i][n].copy_this(box)
                       
       return y_box

        
    def generate_y_vec(self,boxes,cat,name):
        y_box = self.generate_true_boxes(boxes,cat,name)
        y_vec = np.random.rand(GRID_DIM*GRID_DIM*NUM_ANCHOR,(5+NUM_CLASSES))
        
        for i,each_grid in enumerate(y_box):
            for box in each_grid:
                
                a = np.array([box.objectness,\
                              box.sigma_tx,\
                              box.sigma_ty,\
                              box.tw,\
                              box.th,\
                              box.category])
                
              
                y_vec[i,:]=np.hstack(a)
                
        return y_vec
           
#Gen = Batch_Generator()
#x,y = Gen.next_batch()

#def get obj_vec_indices(y):
#    num = NUM_ANCHOR*(1+4+NUM_CLASSES)
#    var = y.shape[1]
#    for i in range(0,var,num):    
#           if()
#get no_obj_vec_indices()           
#           
           
           
           
           
           
           
           
           
           
           
           
           
           