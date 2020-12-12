//g++ reduxOpenmp.cpp -o mpout -std=c++11 `pkg-config --cflags --libs opencv4` -fopenmp
// ./mpout CP77.jpg cp480.jpg 12

#include <opencv2/highgui.hpp>
#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;

cv::Mat flatResample;
cv::Mat flat;

double coeficientWIDTH;

double coeficientHEIGHT;

int WIDTH;
int HEIGHT;

const int LENGTH = 408960;

int THREADS;

void *resample(int ID)
{
    int initIteration, endIteration;
    initIteration = (LENGTH / THREADS) * ID;

    if (ID == THREADS - 1)
        endIteration = LENGTH;
    else
        endIteration = initIteration + ((LENGTH / THREADS) - 1);

    int index = 0;

    for (int aux = initIteration; aux < endIteration; aux++)
    {
        int j = aux % 852;
        int i = (aux - j) / 852;
        index = (j + i * 852) * 3;
        int x = j * coeficientWIDTH;
        int y = i * coeficientHEIGHT;

        int indexAux = (x + y * WIDTH) * 3;

        flatResample.data[index] = flat.data[indexAux];
        flatResample.data[index + 1] = flat.data[indexAux + 1];
        flatResample.data[index + 2] = flat.data[indexAux + 2];
    }
    return 0;
}



int main(int argc, char *argv[])
{

    THREADS = atoi(argv[3]);

    cv::Mat resampleImage(480, 852, CV_8UC3, Scalar(0, 0, 0));

    struct timeval tval_before, tval_after, tval_result;

    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    WIDTH = image.size().width;
    HEIGHT = image.size().height;

    uint totalElements = image.total() * image.channels();

    flat = image.reshape(1, totalElements);

    if (!image.isContinuous())
    {
        flat = flat.clone(); // O(N),
    }

    std::vector<uchar> vec(flat.data, flat.data + flat.total());

    uint totalElementsResampleImage = resampleImage.total() * resampleImage.channels();
    flatResample = resampleImage.reshape(1, totalElementsResampleImage);

    if (!resampleImage.isContinuous())
    {
        flatResample = flatResample.clone(); // O(N),
    }

    auto *ptrResample = flatResample.data;

    std::vector<uchar> vecResample(flatResample.data, flatResample.data + flatResample.total());

    coeficientWIDTH = WIDTH / 852.0;
    coeficientHEIGHT = HEIGHT / 480.0;

    gettimeofday(&tval_before, NULL);

#pragma omp parallel num_threads(THREADS)
    {
        int ID = omp_get_thread_num();
        resample(ID);
    }

    gettimeofday(&tval_after, NULL);

    resampleImage = cv::Mat(480, 852, resampleImage.type(), ptrResample);

    cv::namedWindow(argv[2], cv::WINDOW_AUTOSIZE);
    cv::imshow(argv[2], resampleImage);

    timersub(&tval_after, &tval_before, &tval_result);
    /*
    ofstream myfile;
    myfile.open("openmp.data");
    myfile << THREADS << " " << ("%ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    myfile.close(); 
*/
    printf("Time elapsed: %ld.%06ld\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    imwrite(argv[2], resampleImage);
    cv::waitKey(0);
    return 0;
}
