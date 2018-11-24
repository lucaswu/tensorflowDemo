import tensorflow as tf
import numpy as np
import scipy

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph

def wirteData2Txt(data,h,w,channel,fileName):
    file = open(fileName,'w')
    for i in range(0, h):
        for j in range(0, w):
            file.write('[')
            for c in range(0, channel):
                if data.ndim ==3:
                    file.write(" %f "%data[i,j,c])
                elif data.ndim ==4:
                    file.write(" %f " % data[0,i, j, c])
            file.write(']')
        file.write('\n')
    file.close()
    print('file save....!\n')

def load_txt(filename, shape=[8,8,3]):
    fp=open(filename, 'r')
    a = fp.read()
    fp.close()
    b = a.replace('[', '').replace(']', '').split()
    a = [float(x) for x in b]
    a=np.reshape(np.array(a), [shape[0],shape[1],shape[2]])
    #print(a)
    return a

if __name__ == '__main__':
    graph = load_graph('model.pb')

    for op in graph.get_operations():
        # print(op.name,op.type,op.values())
        # if op.type == 'Const':
        #     # print(op)
        #     print(op.outputs[0].name)
        #     with tf.Session(graph=graph) as sess:
        #         tf_tensor = op.outputs[0].eval(session=sess)
        #         print(tf_tensor,tf_tensor.shape)
        
        if op.type == 'Conv2D':
            print(op.outputs[0].name)
            print("input:")
            for tf_input in op.inputs:
                print(tf_input.name)
            print("---------")


    # x = graph.get_tensor_by_name('input/lr_holder:0')
    # y = graph.get_tensor_by_name('output:0')

    # imgData = scipy.misc.imread("64_64.jpg")
    # #print(imgData.shape)

    # #for i in range(0,10):
    #  #   print(i,imgData[0,i,0],imgData[0,i,1],imgData[0,i,2])

    # imgData = imgData-127.5;
    # h = imgData.shape[0]
    # w = imgData.shape[1]
    # #print(h,w)




    # a = load_txt('input.dat', [64,64,3])
    # #for i in range(0,10):
    #  #   print(i,a[0,i,0],a[0,i,1],a[0,i,2])

    # input = np.reshape(a,[1,h,w,3])

    # print(input[0,0,0,0],input[0,0,0,1],input[0,0,0,2])
    # wirteData2Txt(input,input.shape[1], input.shape[2], input.shape[3], "input_pc.dat")

    # with tf.Session(graph=graph) as sess:
    #     output = sess.run(y,feed_dict={x:input})
    #     print(x)
    #     wirteData2Txt(output,output.shape[1], output.shape[2], output.shape[3], "out_pc.dat")

    #     outpu2 = np.reshape(output,[h*4,w*4,3]);


    #     outpu2 = outpu2+127.5
    #     for i in range(0,10):                                  
    #         print(i,outpu2[0,i,0],outpu2[0,i,1],outpu2[0,i,2])
    #     saveimg = scipy.misc.toimage(outpu2,cmin=0,cmax=255)
    #     saveimg.save('SR.png')
