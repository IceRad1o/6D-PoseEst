#include "cxx_3d_seg.h"
#include <chrono>

#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"
using namespace std;
using namespace cv;
// for test
namespace test_helper {
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "  elasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
template<typename T>
std::vector<T> unique(const cv::Mat& input, bool sort = true)
{
    std::vector<T> out;
    for (int y = 0; y < input.rows; ++y)
    {
        auto row_ptr = input.ptr<T>(y);
        for (int x = 0; x < input.cols; ++x)
        {
            T value = row_ptr[x];

            if ( std::find(out.begin(), out.end(), value) == out.end() )
                out.push_back(value);
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());

    return out;
}

cv::Mat draw_axis(cv::Mat& rgb, cv::Mat& T, cv::Mat& K){
    // unit is mm
    cv::Mat result = rgb;
    float data[] = {
                    100, 0,   0,   0,
                    0,   100, 0,   0,
                    0,   0,   100, 0,
                    1,   1,   1,   1
                    };
    cv::Mat points = cv::Mat(4,4,CV_32FC1, data);

    float tran[] = {
                    1,   0,   0,   0,
                    0,   1,   0,   0,
                    0,   0,   1,   0,
                    };
    cv::Mat tran_mat = cv::Mat(3,4,CV_32FC1, tran);

    cv::Mat p_3d = K*tran_mat*T*points;
    std::vector<cv::Point2f> p_2d(4);
    for(int i=0; i<4; i++){
        p_2d[i].x = p_3d.col(i).at<float>(0,0)/p_3d.col(i).at<float>(2,0);
        p_2d[i].y = p_3d.col(i).at<float>(1,0)/p_3d.col(i).at<float>(2,0);
    }
    cv::line(result, p_2d[0], p_2d[3], cv::Scalar(255, 0, 0), 3);
    cv::line(result, p_2d[1], p_2d[3], cv::Scalar(0, 255, 0), 3);
    cv::line(result, p_2d[2], p_2d[3], cv::Scalar(0, 0, 255), 3);
    return result;
}

}

void dataset_test(){
    int train_size = 1000;
//    string prefix = "/home/meiqua/6DPose/public/datasets/doumanoglou/test/01/";
    string prefix = "/home/meiqua/6DPose/public/datasets/tejani/test/06/";
    for(int i=0;i<train_size;i++){
        auto i_str = to_string(i);
        for(int pad=4-i_str.size();pad>0;pad--){
            i_str = '0'+i_str;
        }
        Mat rgb = cv::imread(prefix+"rgb/"+i_str+".png");
        Mat depth = cv::imread(prefix+"depth/"+i_str+".png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        test_helper::Timer timer;

        auto rgb_slimage = slimage::ConvertToSlimage(rgb);
        auto dep_slimage = slimage::ConvertToSlimage(depth);
        slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
        slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

        auto test_group = asp::DsapGrouping(img_color, img_depth);
        Mat idxs = slimage::ConvertToOpenCv(test_group);

        timer.out("grouping");

        std::vector<int> unik = test_helper::unique<int>(idxs, true);
        std::map<int, Vec3b> color_map;
        for(auto idx: unik){
            auto color = Vec3b(rand()%255, rand()%255, rand()%255);
            color_map[idx] = color;
        }

        Mat show = Mat(idxs.size(), CV_8UC3, Scalar(0));
        auto show_iter = show.begin<Vec3b>();
        for(auto idx_iter = idxs.begin<int>(); idx_iter<idxs.end<int>();idx_iter++, show_iter++){
            if(*idx_iter>0){
                auto color = color_map.find(*idx_iter)->second;
                *show_iter = color;
            }
        }

        imshow("show", show);
        imshow("rgb", rgb);
        waitKey(3000);
    }
}

void simple_test(){
    string prefix = "/home/meiqua/6DPose/cxx_3d_seg/test/2/";
    Mat rgb = cv::imread(prefix+"rgb/0001.png");
    Mat depth = cv::imread(prefix+"depth/0001.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

//    pyrDown(rgb, rgb);
//    pyrDown(depth, depth);

    test_helper::Timer timer;

    auto rgb_slimage = slimage::ConvertToSlimage(rgb);
    auto dep_slimage = slimage::ConvertToSlimage(depth);
    slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
    slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

    auto test_group = asp::DsapGrouping(img_color, img_depth);
    Mat idxs = slimage::ConvertToOpenCv(test_group);

    timer.out("grouping");

    std::vector<int> unik = test_helper::unique<int>(idxs, true);
    std::map<int, Vec3b> color_map;
    for(auto idx: unik){
        auto color = Vec3b(rand()%255, rand()%255, rand()%255);
        color_map[idx] = color;
    }

    Mat show = Mat(idxs.size(), CV_8UC3, Scalar(0));
    auto show_iter = show.begin<Vec3b>();
    for(auto idx_iter = idxs.begin<int>(); idx_iter<idxs.end<int>();idx_iter++, show_iter++){
        if(*idx_iter>0){
            auto color = color_map.find(*idx_iter)->second;
            *show_iter = color;
        }
    }
    imshow("show", show);
    imshow("rgb", rgb);
    waitKey(0);
//    Mat rgb_ = slimage::ConvertToOpenCv(rgb_slimage);
//    Mat depth_ = slimage::ConvertToOpenCv(dep_slimage);

}

void super4pcs_test(){
    string prefix = "/home/meiqua/6DPose/cxx_3d_seg/test/2/";
    Mat rgb = cv::imread(prefix+"rgb/0000.png");
    Mat depth = cv::imread(prefix+"depth/0000.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

    test_helper::Timer timer;

    auto rgb_slimage = slimage::ConvertToSlimage(rgb);
    auto dep_slimage = slimage::ConvertToSlimage(depth);
    slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
    slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

    auto test_group = asp::DsapGrouping(img_color, img_depth);
    Mat idxs = slimage::ConvertToOpenCv(test_group);

    timer.out("grouping");

    int test_which = 3;
    int test_count = 0;
//    Mat show = Mat(idxs.size(), CV_8UC3, Scalar(0));
    Mat  show = rgb.clone();
    auto show_iter = show.begin<Vec3b>();
    for(auto idx_iter = idxs.begin<int>(); idx_iter<idxs.end<int>();idx_iter++, show_iter++){
        if(*idx_iter==test_which){
            *show_iter = {0, 0, 255};
            test_count ++;
        }
    }
    std::cout << "test_count: " << test_count << std::endl;

//    imshow("rgb", rgb);
    imshow("show", show);

    waitKey(0);

    timer.reset();
    Mat test_seg = idxs == test_which;

//    Mat test_seg = imread(prefix+"test_seg.png");
//    cvtColor(test_seg, test_seg, CV_BGR2GRAY);

    Mat test_dep;
    depth.copyTo(test_dep, test_seg);

    Mat sceneK = (Mat_<float>(3,3)
                  << 550.0, 0.0, 316.0, 0.0, 540.0, 244.0, 0.0, 0.0, 1.0);
    cv::Mat sceneCloud;
    cv::rgbd::depthTo3d(test_dep, sceneK, sceneCloud);

    std::vector<GlobalRegistration::Point3D> test_cloud;

    for(auto cloud_iter = sceneCloud.begin<cv::Vec3f>();
        cloud_iter!=sceneCloud.end<cv::Vec3f>(); cloud_iter++){
        if(cv::checkRange(*cloud_iter)){
            GlobalRegistration::Point3D p;
            p.x() = (*cloud_iter)[0]*1000;
            p.y() = (*cloud_iter)[1]*1000;
            p.z() = (*cloud_iter)[2]*1000;
            test_cloud.push_back(p);
        }
    }

    std::vector<GlobalRegistration::Point3D> model_v;
    std::vector<typename GlobalRegistration::Point3D::VectorType> model_n;
    {
        IOManager iom;
        std::vector<Eigen::Matrix2f> tex_coords;
        std::vector<tripple> tris;
        std::vector<std::string> mtls;
        iom.ReadObject((prefix+"model.ply").c_str(), model_v, tex_coords, model_n, tris, mtls);
    }

    Eigen::Matrix4f	transformation = Eigen::Matrix4f::Identity();
    float score = 0;
    {
        GlobalRegistration::Match4PCSOptions options;
        options.sample_size = 30;
        options.max_time_seconds = 1;
        constexpr GlobalRegistration::Utils::LogLevel loglvl = GlobalRegistration::Utils::Verbose;
        GlobalRegistration::Utils::Logger logger(loglvl);
        GlobalRegistration::MatchSuper4PCS matcher(options, logger);
        score = matcher.ComputeTransformation(model_v, &test_cloud, transformation);
    }
    std::cout << "final LCP: " << score << std::endl;
    cout << transformation.inverse() << endl;
    timer.out("super4pcs");

    waitKey(0);
}

void ppf_test(){
    using namespace ppf_match_3d;
    string prefix = "/home/meiqua/6DPose/cxx_3d_seg/test/2/";
    Mat rgb = cv::imread(prefix+"rgb/0000.png");
    Mat depth = cv::imread(prefix+"depth/0000.png", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

    test_helper::Timer timer;

    auto rgb_slimage = slimage::ConvertToSlimage(rgb);
    auto dep_slimage = slimage::ConvertToSlimage(depth);
    slimage::Image3ub img_color = slimage::anonymous_cast<unsigned char,3>(rgb_slimage);
    slimage::Image1ui16 img_depth = slimage::anonymous_cast<uint16_t,1>(dep_slimage);

    auto test_group = asp::DsapGrouping(img_color, img_depth);
    Mat idxs = slimage::ConvertToOpenCv(test_group);

    timer.out("grouping");

    int test_which = 3;
    int test_count = 0;
//    Mat show = Mat(idxs.size(), CV_8UC3, Scalar(0));
    Mat  show = rgb.clone();
    auto show_iter = show.begin<Vec3b>();
    for(auto idx_iter = idxs.begin<int>(); idx_iter<idxs.end<int>();idx_iter++, show_iter++){
        if(*idx_iter==test_which){
            *show_iter = {0, 0, 255};
            test_count ++;
        }
    }
    std::cout << "test_count: " << test_count << std::endl;

//    imshow("rgb", rgb);
    imshow("show", show);

    waitKey(0);


    Mat test_seg = idxs == test_which;

//    Mat test_seg = imread(prefix+"test_seg.png");
//    cvtColor(test_seg, test_seg, CV_BGR2GRAY);

    Mat test_dep;
    depth.copyTo(test_dep, test_seg);

    Mat sceneK = (Mat_<float>(3,3)
                  << 550.0, 0.0, 316.0, 0.0, 540.0, 244.0, 0.0, 0.0, 1.0);
    cv::Mat sceneCloud;
    cv::rgbd::depthTo3d(test_dep, sceneK, sceneCloud);

    int valid_cloud_count = 0;
    for(auto iter = sceneCloud.begin<cv::Vec3f>();
        iter!=sceneCloud.end<cv::Vec3f>(); iter++){
        if(cv::checkRange(*iter)){
            valid_cloud_count++;
        }
    }
    cv::Mat sceneCloud_ = cv::Mat(valid_cloud_count, 3, CV_32FC1);
    valid_cloud_count = 0;
    for(auto iter = sceneCloud.begin<cv::Vec3f>();
        iter!=sceneCloud.end<cv::Vec3f>(); iter++){
        if(cv::checkRange(*iter)){
            sceneCloud_.row(valid_cloud_count) = (*iter)/1000;
            valid_cloud_count++;
        }
    }

    timer.reset();
    cv::Mat cloud_with_normal;
    cv::ppf_match_3d::computeNormalsPC3d(sceneCloud_, cloud_with_normal, 20, false, {0,0,1});

    cv::Mat model_cloud = loadPLYSimple((prefix+"model.ply").c_str(), 1);

    timer.out("ppf train start");
    ppf_match_3d::PPF3DDetector detector(0.025, 0.05);
    detector.trainModel(model_cloud);
    timer.out("ppf train");

    vector<Pose3DPtr> results;
    detector.match(cloud_with_normal, results, 1.0/40.0, 0.05);
    timer.out("ppf match");

    // Get only first N results
    int N = 2;
    vector<Pose3DPtr> resultsSub(results.begin(),results.begin()+N);

    // Create an instance of ICP
    ICP icp(100, 0.005f, 2.5f, 8);
    // Register for all selected poses
    cout << endl << "Performing ICP on " << N << " poses..." << endl;
    icp.registerModelToScene(model_cloud, cloud_with_normal, resultsSub);
    timer.out("icp");

    for (size_t i=0; i<resultsSub.size(); i++)
    {
        Pose3DPtr result = resultsSub[i];
        cout << "Pose Result " << i << endl;
        result->printPose();
    }

//    Mat pc = loadPLYSimple(modelFileName.c_str(), 1);
}

int main(){
//    simple_test();
//    dataset_test();
    super4pcs_test();
//    ppf_test();
    cout << "end" << endl;
    return 0;
}
