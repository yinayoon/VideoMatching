#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::ml;

int main()
{
	Mat SourceImg = imread("source 004.jpg", IMREAD_COLOR); // 템플릿 이미지
	if (SourceImg.empty()) { cerr << "File Open Error" << endl; exit(1); }

	VideoCapture video("test_video_002.mp4"); // 비디오 이미지
	if (!video.isOpened()) { cerr << "File Open Error" << endl; exit(1); }

	Mat Capture; // 비디오로 부터 켑쳐된 이미지
	Mat result(video.get(CAP_PROP_FRAME_HEIGHT), video.get(CAP_PROP_FRAME_WIDTH) + SourceImg.cols, CV_8UC3, Scalar::all(0)); //결과 영상

	vector<KeyPoint> KeyPtSource, KeyPtVideo; // 특징점을 위한 배열
	Mat DescriptSource, DescriptVideo; // 기술자
	Ptr<BRISK> surf = BRISK::create(); // 특징점 추출 및 기술자 계산에 BRISK와 SURF를 사용
	surf->detectAndCompute(SourceImg, noArray(), KeyPtSource, DescriptSource); // 특징점 및 기술자를 추출


	int Count = 0; //영상 프레임의 카운트를 위한 변수
	VideoWriter Encoder("FinalVideo.avi", CV_FOURCC('P', 'I', 'M', '1'), 29.f, Size(SourceImg.cols + video.get(CAP_PROP_FRAME_WIDTH), video.get(CAP_PROP_FRAME_HEIGHT))); // 결과 영상 저장을 위한 부분
	if (!Encoder.isOpened()) { cerr << endl << "File Write Error" << endl; }

	while (1) {
		video >> Capture;
		if (!video.read(Capture)) break;
		 
		surf->detectAndCompute(Capture, noArray(), KeyPtVideo, DescriptVideo); // 프레임당 특징점 및 기술자를 추출

		BFMatcher Matcher(NORM_L1);
		vector<DMatch> matches; //매칭점 넣을 배열 선언
		Matcher.match(DescriptSource, DescriptVideo, matches); //템플릿 이미지와 비디오 이미지 사이에서 기술자 간 차이가 가장 적은 것을 선택하기 위한 과정

		vector<KeyPoint> matchedSource, matchedCapture; //선별된 매칭점의 각 이미지별 배열
		vector<DMatch> matched_matches; 
		for (int i = 0; i < matches.size(); i++) { 
			if (matches[i].distance < 1500.f) {
				matchedSource.push_back(KeyPtSource[matches[i].queryIdx]);
				matchedCapture.push_back(KeyPtVideo[matches[i].trainIdx]);
				matched_matches.push_back(DMatch(matchedSource.size() - 1, matchedCapture.size() - 1, 0));
			}
		}

		if (matchedSource.size() >= 4) { // 호모그래피 라인을 그리기 위한 과정
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;

			for (size_t i = 0; i < matched_matches.size(); i++) 
			{
				obj.push_back(matchedSource[matched_matches[i].queryIdx].pt);
				scene.push_back(matchedCapture[matched_matches[i].trainIdx].pt);
			}

			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = Point(0, 0);
			obj_corners[1] = Point(SourceImg.cols, 0);
			obj_corners[2] = Point(SourceImg.cols, SourceImg.rows);
			obj_corners[3] = Point(0, SourceImg.rows);
			std::vector<Point2f> scene_corners(4);

			Mat H = findHomography(obj, scene, RANSAC);
			perspectiveTransform(obj_corners, scene_corners, H);
			
			line(result, scene_corners[0] + Point2f((float)SourceImg.cols, 0), scene_corners[1] + Point2f((float)SourceImg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
			line(result, scene_corners[1] + Point2f((float)SourceImg.cols, 0), scene_corners[2] + Point2f((float)SourceImg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
			line(result, scene_corners[2] + Point2f((float)SourceImg.cols, 0), scene_corners[3] + Point2f((float)SourceImg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);
			line(result, scene_corners[3] + Point2f((float)SourceImg.cols, 0), scene_corners[0] + Point2f((float)SourceImg.cols, 0), Scalar(0, 255, 0), 2, LINE_AA);

			imshow("aaaa", result);
		}

		drawMatches(SourceImg, matchedSource, Capture, matchedCapture, matched_matches, result);
		
		if (Encoder.isOpened()) { Encoder << result; }

		Count++;
		
		if (waitKey(10) == 27) { // 종료 : ESC
			destroyWindow("Result Matching");
			break;
		}
	}
	return 0;
}
