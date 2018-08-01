export  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/lucas/github/tensorflow/build/lib

g++ -g -o sr sr.cc -I /home/lucas/github/tensorflow/build/include  -L /home/lucas/github/tensorflow/build/lib -ltensorflow `pkg-config --cflags --libs opencv` 
if [ $? -ne 0 ];then
	echo "build  error !"
	exit 1
fi
./sr model.pb 22.jpg 22_meitu_1.jpg out.jpg out-2.jpg