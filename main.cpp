#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;

const string res_folder = "D:\\CheckFace";
const string face_model = res_folder + "\\" + "face_model.xml";
const string face_search_xml = "D:\\CheckFace\\haarcascade_frontalface_alt.xml";

const int frame_num = 96; //每个视频提取96帧


//人脸采集
//读196帧数据
void face_collection() {
    CascadeClassifier cascade_face;
    if(!cascade_face.load(face_search_xml)) {
        cerr << "Failed to load face detection model" << endl;
        return;
    }
    for(const auto & person_dir : filesystem::directory_iterator(res_folder)) {

        if (!filesystem::is_directory(person_dir))
            continue;

        for(const auto& mp4_path : filesystem::directory_iterator(person_dir)) {

            if(mp4_path.path().extension() == ".mp4") {
                VideoCapture capture(mp4_path.path().string());
                if (!capture.isOpened()) {
                    cerr << "Failed to open mp4: " << mp4_path.path().string() << endl;
                    continue;
                }

                Mat frame;
                vector<Rect> face_rect;

                for (int i = 0; i < frame_num; i++) {
                    capture >> frame;
                    if (frame.empty())
                        continue;

                    cvtColor(frame, frame, COLOR_BGR2GRAY);
                    cascade_face.detectMultiScale(frame, face_rect, 1.1, 3, 0, Size(80, 80));

                    for (const Rect &face: face_rect) {
                        Mat face_img = frame(face);
                        string output_filename =
                                mp4_path.path().parent_path().string() + "\\" + mp4_path.path().stem().string() + "_" +
                                to_string(i) + ".jpg";
                        imwrite(output_filename, face_img);
                        cout << output_filename << endl;
                    }
                }
            }
        }
    }
}


//训练模型
void model_train(){
    vector<Mat> images;
    vector<int> levels;

    //分类读取图片
    for(const auto& person_dir : filesystem::directory_iterator(res_folder)){
        if(!filesystem::is_directory(person_dir))
            continue;
        int level = stoi(person_dir.path().filename().string());

        for(const auto& img_path : filesystem::directory_iterator(person_dir)){
            if(img_path.path().extension() == ".jpg") {
                Mat img = imread(img_path.path().string(), IMREAD_GRAYSCALE);

                images.push_back(img);
                levels.push_back(level);
            }
        }
    }
    //训练模型
    Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
    model->train(images, levels);
    model->save(face_model);

    cout << "train end" << endl;

}

//人脸识别
void check_face(){
    //加载模型
    Ptr<face::LBPHFaceRecognizer> model = face::LBPHFaceRecognizer::create();
    model->read(face_model);

    //打开摄像头
    VideoCapture capture(0);
    if(!capture.isOpened())
        cout << "failed to open capture" << endl;

    //创建人脸检测器
    CascadeClassifier faceClassifier;
    faceClassifier.load(face_search_xml);

    //创建循环检测
    while(true){
        Mat frame;
        capture >> frame;
        if(frame.empty())
            break;

        //转化为灰度图
        Mat gray_frame;
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

        //人脸检测
        vector<Rect> faces;
        faceClassifier.detectMultiScale(gray_frame, faces, 1.1, 3, 0,Size(80, 80));

        for(const Rect& face : faces){
            Mat face_img = gray_frame(face);
            int level = -1;
            double confident_dis = 0;
            model->predict(face_img, level, confident_dis);

            rectangle(frame, face, Scalar(255, 0, 0), 2);

            Point labelPosition(face.x, face.y - 10);

            if(confident_dis < 80)
                switch(level){
                    case 0 :
                        putText(frame, "DingZhen" + to_string(confident_dis), labelPosition,
                                FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 0), 2);
                        break;
                    case 1 :
                        putText(frame, "ShuaiGe" + to_string(confident_dis), labelPosition,
                                FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 0), 2);
                        break;
                    case 2 :
                        putText(frame, "LuBenWei" + to_string(confident_dis), labelPosition,
                                FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 0), 2);
                        break;
                    case 3 :
                        putText(frame, "LiLaoBa" + to_string(confident_dis), labelPosition,
                                FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 0), 2);
                        break;
                    case 4 :
                        putText(frame, "SunXiaoChuan" + to_string(confident_dis), labelPosition,
                                FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 255, 0), 2);
                        break;
                    default:
                        putText(frame, "UnKnown" + to_string(confident_dis), labelPosition,
                                FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 0, 255), 2);
                        break;
                }
            else
                putText(frame, "UnKnown" + to_string(confident_dis), labelPosition,
                        FONT_HERSHEY_PLAIN, 0.9, Scalar(0, 0, 255), 2);
        }

        // 显示视频流
        imshow("Face Recognition", frame);

        // 按下ESC键退出
        if (waitKey(1) == 27) break;
    }
}


int main() {
    //face_collection();
    //model_train();
    check_face();

    return 0;
}