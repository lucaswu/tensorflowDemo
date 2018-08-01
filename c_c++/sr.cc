#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"
#include "tf_utils.h"
#include<string>
using namespace cv;
using namespace std;

#define INPUT_LAYER_NAME	"input/lr_holder"
#define OUTPUT_LAYER_NAME	"output"

#define asize(a) (sizeof(a) / sizeof(a[0]))

void check_status_ok(TF_Status* status, const char* step) {
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Error at step \"%s\", status is: %u\n", step, TF_GetCode(status));
        fprintf(stderr, "Error message: %s\n", TF_Message(status));
        exit(EXIT_FAILURE);
    } else {
        printf("%s\n", step);
    }
}

void deallocate(void* data, size_t a, void* b)
{

	free(data);
}


void printTFTensorInfo(TF_Tensor *tensor)
{
    int dimNum = TF_NumDims(tensor);
    
    int *dims = (int*)malloc(sizeof(int)*dimNum);

    for(int i=0;i<dimNum;i++){
        dims[i] = TF_Dim(tensor,i);
    }
    switch(dimNum){
        case 1:
            printf("tensor:(%d)\n",dims[0]);break;
        case 2:
            printf("tensor:(%d,%d)\n",dims[0],dims[1]);break;
        case 3:
            printf("tensor:(%d,%d,%d)\n",dims[0],dims[1],dims[2]);break;
        case 4:
            printf("tensor:(%d,%d,%d,%d)\n",dims[0],dims[1],dims[2],dims[3]);break;
    }
    free(dims);
}

int main(int argc,char *argv[]){

    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Status* status = TF_NewStatus();
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    check_status_ok(status, "Initialization of TensorFlow session");

    TF_Buffer* graph_def = NULL;
    if ((graph_def = buffer_read_from_file (argv[1])) == NULL)
  	{
  		printf("read model file error!");
    	return 1;
  	}

  	TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();

  	TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
  	check_status_ok(status, "Loading of .pb graph");


  	cv::Mat img = imread(argv[2]);
    cv::Mat img1 = imread(argv[3]);
  	// cvtColor(img,img,CV_BGR2RGB);


    int height = img.rows;
    int width = img.cols;

    printf("input size:(%d,%d)\n",height,width);
    
    size_t size = height*width*3*sizeof(float)*2;
    float *inData = (float*)malloc(size);
    // float *outData = (float*)malloc(size*16);
  
    memset(inData,0,size);
    // memset(outData,0,size*16);

    unsigned char*p = img.data;
    unsigned char*p1 = img1.data;
    int patchSize = width*height*3;
    for(int i=0;i<width*height*3;i++){
        inData[i] = p[i]-127.5f;//-1.0f;
        inData[i+patchSize] = p1[i]-127.5f;//-1.0f;

        if(i<6){
        	printf("%d=%f\n",i,inData[i] );
        }
    }


    cv::Mat inMat(height, width, CV_8UC3);
    p = inMat.data;
    for(int i=0;i<width*height*3;i++){
        p[i] = (unsigned char)(inData[i]+127.5f);
        
    }
    imwrite("in-1.jpg",inMat);


    int64_t dims[] = {2, height, width, 3 };
   
    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT,dims,asize(dims),inData,size,deallocate,(void*)"input");

    TF_Tensor *output_tensor;


    TF_Operation* in_op = TF_GraphOperationByName(graph, INPUT_LAYER_NAME);
	TF_Operation* out_op = TF_GraphOperationByName(graph, OUTPUT_LAYER_NAME);

	TF_Output inputs[] = {
		{ in_op, 0 }
	};
	TF_Output outputs[] = {
		{ out_op, 0 }
	};

	const TF_Operation* const_out_op = out_op;

	TF_SessionRun(session,
		NULL,
		inputs,&input_tensor,1,
		outputs,&output_tensor,1,
		NULL, 0, /* target operations, number of targets */
    	NULL, /* run metadata */
    	status);

	if (TF_GetCode (status) != TF_OK){
    	fprintf (stderr, "ERR: Could not run session: %s\n", TF_Message (status));
  	}
	check_status_ok(status, "sessionRun");


	const float* data = (float*)TF_TensorData (output_tensor);

    printTFTensorInfo(output_tensor);

	for(int i=0;i<10;i++){
		printf("%d=%f\n",i,data[i]);
	}

	cv::Mat outputMat(height*4, width*4, CV_8UC3);
   
    
 
 
    p = outputMat.data;
 
    for(int i=0;i<width*height*16*3;i++){
        float temp = (data[i] +127.5f);
        temp = (temp<0.0f)?0.0f:temp;
        temp = (temp>255.f)?255.f:temp;
        p[i] = temp;

    }
 
    imwrite(argv[4],outputMat);


    cv::Mat outputMat1(height*4, width*4, CV_8UC3);
    p = outputMat1.data;
    for(int i=0;i<width*height*16*3;i++){
        float temp = (data[i+width*height*16*3] +127.5f);
        temp = (temp<0.0f)?0.0f:temp;
        temp = (temp>255.f)?255.f:temp;
        p[i] = temp;

    }
    imwrite(argv[5],outputMat1);
    return 0;
}