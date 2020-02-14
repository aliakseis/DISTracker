// OPTICAL FLOW USING FARNEBACK ALGORITHM

#include "pch.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//#include <opencv2/optflow.hpp>

#include <iostream>



using namespace cv;
using namespace std;

//#if 0
#define M_PI 3.14159265358979323846

// https://gist.github.com/voidqk/fc5a58b7d9fc020ecf7f2f5fc907dfa5
inline float fastAtan2_(float y, float x)
{
    static const float c1 = M_PI / 4.0;
    static const float c2 = M_PI * 3.0 / 4.0;
    //if (y == 0 && x == 0)
    //    return 0;

    if (y == 0)
        return 0;
    if (x == 0)
        return (y > 0) ? (M_PI / 2.) : (M_PI / 2.);

    float abs_y = fabsf(y);
    float angle;
    if (x >= 0)
        angle = c1 - c1 * ((x - abs_y) / (x + abs_y));
    else
        angle = c2 - c1 * ((x + abs_y) / (abs_y - x));
    if (y < 0)
        return -angle;
    return angle;
}
//#endif

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    //const float rad = hypot(fx, fy); //sqrt(fx * fx + fy * fy);
    const float rad = std::max(std::abs(fx), std::abs(fy)); //sqrt(fx * fx + fy * fy);
    const float a = fastAtan2_(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const auto col0 = colorWheel[k0][b];// / 255.f;
        const auto col1 = colorWheel[k1][b];// / 255.f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 255 - rad * (255 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(/*255.f * */col);
    }

    return pix;
}


// Function to compute the optical flow map
#if 0
void drawOpticalFlow(const Mat& flowImage, Mat& flowImageGray)
{
    //const int stepSize = 16;
    const int stepSize = 10;
    //Scalar color = Scalar(0, 255, 0);
    
    // Draw the uniform grid of points on the input image along with the motion vectors
    for(int y = 0; y < flowImageGray.rows; y += stepSize)
    {
        for(int x = 0; x < flowImageGray.cols; x += stepSize)
        {

            
            // Lines to indicate the motion vectors
            Point2f pt = flowImage.at<Point2f>(y, x);

            const auto length = std::hypot(pt.y, pt.x);
            if (length > FLT_EPSILON)
            {
                const auto coeff = log(length + 1) * 4 / length;
                pt *= coeff;
            }

            const auto color = computeColor(pt.x, pt.y);

            // Circles to indicate the uniform grid of points
            int radius = 1;
            int thickness = -1;
            circle(flowImageGray, Point(x, y), radius, color, thickness);

            line(flowImageGray, Point(x,y), Point(cvRound(x+pt.x), cvRound(y+pt.y)), color);
        }
    }
}
#endif

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1)
{
    //dst.create(flow.size(), CV_8UC3);
    //dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                Point2f u = flow(y, x);

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f u = flow(y, x);

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}


int main(int argc, char** argv)
{
    // set default values for tracking algorithm and video
    string videoPath = (argc == 2) ? argv[1] : "videos/run.mp4";


    // create a video capture object to read videos
    cv::VideoCapture cap(videoPath);

    if(!cap.isOpened())
    {
        cerr << "Unable to open the file. Exiting!" << endl;
        return -1;
    }
    
    Mat curGray, prevGray, flowImage, flowImageGray, frame;
    string windowName = "Optical Flow";
    namedWindow(windowName, 1);
    float scalingFactor = 0.75;
    
    auto optFlow = cv::DISOpticalFlow::create(DISOpticalFlow::PRESET_FAST);

    // Iterate until the user presses the Esc key
    while(true)
    {
        // Capture the current frame
        cap >> frame;
        
        if(frame.empty())
            break;
        
        // Resize the frame
        //resize(frame, frame, Size(), scalingFactor, scalingFactor, INTER_AREA);
        
        // Convert to grayscale
        cvtColor(frame, curGray, COLOR_BGR2GRAY);
        
        // Check if the image is valid
        if (prevGray.data)
#if 1
        {
            //cv::optflow::createOptFlow_DIS()
            optFlow->calc(prevGray, curGray, flowImage);

            // Convert to 3-channel RGB
            cvtColor(prevGray, flowImageGray, COLOR_GRAY2BGR);

            // Draw the optical flow map
            drawOpticalFlow(flowImage, flowImageGray);

            // Display the output image
            imshow(windowName, flowImageGray);
        }
#endif

#if 0
        {
            // Initialize parameters for the optical flow algorithm
            float pyrScale = 0.5;
            int numLevels = 3;
            int windowSize = 15;
            int numIterations = 3;
            int neighborhoodSize = 5;
            float stdDeviation = 1.2;
            
            // Calculate optical flow map using Farneback algorithm
            calcOpticalFlowFarneback(prevGray, curGray, flowImage, pyrScale, numLevels, windowSize, numIterations, neighborhoodSize, stdDeviation, 0);// OPTFLOW_USE_INITIAL_FLOW);
            
            // Convert to 3-channel RGB
            cvtColor(prevGray, flowImageGray, COLOR_GRAY2BGR);
            
            // Draw the optical flow map
            drawOpticalFlow(flowImage, flowImageGray);
            
            // Display the output image
            imshow(windowName, flowImageGray);
        }
#endif
        
        // Break out of the loop if the user presses the Esc key
        char ch = waitKey(1);
        if(ch == 27)
            break;
        
        // Swap previous image with the current image
        std::swap(prevGray, curGray);
    }
    
    return 0;
}


