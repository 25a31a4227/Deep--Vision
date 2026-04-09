#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    // Load class names
    vector<string> classes;
    ifstream ifs("coco.names");
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load YOLO model
    Net net = readNet("yolov3.weights", "yolov3.cfg");

    // Use CPU
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening camera\n";
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Create blob
        Mat blob;
        blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(), true, false);
        net.setInput(blob);

        // Forward pass
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Process detections
        for (auto &output : outputs) {
            for (int i = 0; i < output.rows; i++) {
                float confidence = output.at<float>(i, 4);
                if (confidence > 0.5) {
                    int centerX = (int)(output.at<float>(i, 0) * frame.cols);
                    int centerY = (int)(output.at<float>(i, 1) * frame.rows);
                    int width = (int)(output.at<float>(i, 2) * frame.cols);
                    int height = (int)(output.at<float>(i, 3) * frame.rows);

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    rectangle(frame, Point(left, top), Point(left + width, top + height), Scalar(0,255,0), 2);
                    putText(frame, "Object", Point(left, top - 10),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
                }
            }
        }

        imshow("Deep Vision", frame);

        if (waitKey(1) == 27) break; // ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
