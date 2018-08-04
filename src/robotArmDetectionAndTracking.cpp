// Detection of circular bounding box
// Hasnat ASE Master Thesis
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sstream>      // std::ostringstream

//#include <string.h>

#include <cv.h>

#include <iostream>
#include <vector>



using namespace cv;
using namespace std;

Mat element, imageOut;
struct recircle{
        float x;
        float y;
        float r;
        float redetects;
        int typ;

        recircle(float x, float y, float r){
                this->x = x; this->y = y; this->r = r; this->redetects = 0; typ=0;
        }
};

Mat image, imagegray, canny_output;
int rmin = 11; //35;
int rmax = 12; //35;
int cannythreshmax = 110;
int centerthresh = 11;
int cthreshl = 44;//10;
int cthreshh = 138;
int maxapproxdist = 7; //20;
int minbbheight = 41; //123;
int minbbwidth = 7; //21;
int minbbheight2 = 15; //45;
int minbbwidth2 = 3; //8;
vector<recircle> circlespm;
int maxhredetects = 0;
int maxhredetects3 = 0;
size_t maxhidx = 0;

float seg12 = 0;
float seg23 = 0;
float seg13 = 0;

int dist(Point p1, Point p2){

        int dx = p1.x-p2.x;
        int dy = p1.y-p2.y;

        return sqrt(dx*dx+dy*dy);


}

void trackbarchange(int, void*) // void pointer can point any object, does not know which type
{
	//image.copyTo(imageOut);
	//imshow("Output Image", image);


    //1. circle detect
    vector<Vec3f> circles;
    HoughCircles( imagegray, circles, CV_HOUGH_GRADIENT, 1, imagegray.rows/8, cannythreshmax, centerthresh, rmin, rmax );
    for( size_t i = 0; i < circles.size(); i++ )
    {
     Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
     int radius = cvRound(circles[i][2]);
     circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
     circle( image, center, radius, Scalar(0,0,0), 2, 8, 0 ); //black -- all circles detected by HCircle
     int isredetect = 0;
     for( size_t j = 0; j < circlespm.size(); j++ ){
    	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 //80
             if(dist(center,Point(circlespm[j].x, circlespm[j].y)) < 27){

                     circlespm[j].redetects++;
                     circlespm[j].x = center.x;
                     circlespm[j].y = center.y;
                     isredetect = 1;
             }

     }
     if(!isredetect){

             recircle r (circles[i][0],circles[i][1],circles[i][2]);
             circlespm.push_back( r );

     }

    }

    if(circlespm.empty()) return;

    maxhredetects = 0;
    for( size_t j = 0; j < circlespm.size(); j++ ){

            circlespm[j].typ = 0;

            if(circlespm[j].redetects > maxhredetects){
                    maxhidx = j;
                    maxhredetects = circlespm[j].redetects;
            }

            circlespm[j].redetects=circlespm[j].redetects-maxhredetects3+33;

            if(circlespm[j].redetects<-3){

                    circlespm.erase(circlespm.begin()+j);
            }

    }
    circle( image, Point(cvRound(circlespm[maxhidx].x), cvRound(circlespm[maxhidx].y)), cvRound(circlespm[maxhidx].r), Scalar(0,255,0), 6, 8, 0 ); //green

    int maxhredetects2 = 0;
    size_t maxhidx2 = 0;
    for( size_t j = 0; j < circlespm.size(); j++ ){

                    if(circlespm[j].redetects > maxhredetects2
                                    && dist(Point(circlespm[j].x, circlespm[j].y),Point(circlespm[maxhidx].x, circlespm[maxhidx].y))>2){
                            maxhidx2 = j;
                            maxhredetects2 = circlespm[j].redetects;
                    }

            }
    circle( image, Point(cvRound(circlespm[maxhidx2].x), cvRound(circlespm[maxhidx2].y)), cvRound(circlespm[maxhidx2].r), Scalar(255,0,0), 6, 8, 0 ); //blue

    maxhredetects3 = 0;
    size_t maxhidx3 = 0;
            for( size_t j = 0; j < circlespm.size(); j++ ){

                            if( (circlespm[j].redetects > maxhredetects3)
                                            && (dist(Point(circlespm[j].x, circlespm[j].y),Point(circlespm[maxhidx].x, circlespm[maxhidx].y))>2)
                                            && (dist(Point(circlespm[j].x, circlespm[j].y),Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y))>2) ){
                                    maxhidx3 = j;
                                    maxhredetects3 = circlespm[j].redetects;
                            }

                    }
    circle( image, Point(cvRound(circlespm[maxhidx3].x), cvRound(circlespm[maxhidx3].y)), cvRound(circlespm[maxhidx3].r), Scalar(0,0,255), 6, 8, 0 ); // red

    //3. Bounding Box
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;
     Canny( imagegray, canny_output, cthreshl, cthreshh, 3 );
     findContours( canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
     vector<vector<Point> > contoursApprox( contours.size() );
     vector<bool > allowedToDraw( contours.size() );
     vector<RotatedRect> boundRect( contours.size() );
     vector<Point2f>center( contours.size() );
     vector<float>radius( contours.size() );
     Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
     for(size_t i = 0; i<contours.size(); i++){//ruckelt langsam approx poly notwendig
                     //approxPolyDP(contours[i],contoursApprox[i], (float)maxapproxdist/100, 1); //always CV_POLY_APPROX_DP
                     boundRect[i] = minAreaRect( Mat(contours[i]) );//boundingRect
                     //minEnclosingCircle( (Mat)contoursApprox[i], center[i], radius[i] );
             }
     for(size_t i=0; i<boundRect.size(); i++){

             RotatedRect curRectangle = boundRect[i]; //50
             if( (abs(curRectangle.size.height - minbbheight)<17 && abs(curRectangle.size.width - minbbwidth)<17 ) ||
                             ( abs(curRectangle.size.width - minbbheight)<17 && abs(curRectangle.size.height - minbbwidth)<17) ){

                     Point2f rect_points[4]; boundRect[i].points( rect_points );
                                               for( int j = 0; j < 4; j++ )
                                                      line( image, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0), 1, 8 );
             }
     }


     //for each pair of circles check bounding box distance smaller 20
     for( size_t i = 0; i < circlespm.size(); i++ ){

             for( size_t j = 0; j < circlespm.size(); j++ ){

                     Point p1(cvRound(circlespm[i].x), cvRound(circlespm[i].y));
                     Point p2(cvRound(circlespm[j].x), cvRound(circlespm[j].y));
                     Point segcenter( (p1.x+p2.x)/2, (p1.y+p2.y)/2 );

                     for(size_t k=0; k<boundRect.size(); k++){

                             RotatedRect curRectangle = boundRect[k]; //30
                             if( (abs(curRectangle.size.height - minbbheight)<10 && abs(curRectangle.size.width - minbbwidth)<10 ) ||
                                     ( abs(curRectangle.size.width - minbbheight)<10 && abs(curRectangle.size.height - minbbwidth)<10) ){

                                     if( dist(curRectangle.center, segcenter) < 40 && dist(p1,p2) < 90 && dist(p1,p2) > 70){ //270,210,werte mit scalar factor parametrisieren um zoombar zu machen wie bei Antennendetektion

                                             line( image, p1, p2, Scalar(0,200,0), 3, 8 ); //200
                                             circlespm[i].redetects++;
                                             circlespm[j].redetects++;

                                     }

                             }

                     }

             }
     }


     //check distance parameters 2 ,
      for( size_t i = 0; i < circlespm.size(); i++ ){

                      for( size_t j = 0; j < circlespm.size(); j++ ){

                              Point p1(cvRound(circlespm[i].x), cvRound(circlespm[i].y));
                              Point p2(cvRound(circlespm[j].x), cvRound(circlespm[j].y));
                              Point segcenter( (p1.x+p2.x)/2, (p1.y+p2.y)/2 );

                              for(size_t k=0; k<boundRect.size(); k++){

                                      RotatedRect curRectangle = boundRect[k]; // 30
                                      if( (abs(curRectangle.size.height - minbbheight2)<10 && abs(curRectangle.size.width - minbbwidth2)<10 ) ||
                                              ( abs(curRectangle.size.width - minbbheight2)<10 && abs(curRectangle.size.height - minbbwidth2)<10) ){
                                    	  	  // finding segment circles
                                              if( dist(curRectangle.center, segcenter) < 13 && dist(p1,p2) < 80 && dist(p1,p2) > 73 ){ //40, 270 , 210

                                                      line( image, p1, p2, Scalar(0,0,0), 3, 8 ); // 200
                                                      //bonus for both endpoints
                                                      circlespm[i].redetects++;
                                                      circlespm[j].redetects++;
                                                      	  //10
                                                      if(dist(p1, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 4 && dist(p2, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 4){

                                                              if(seg12 < 165) seg12+=1.5; // 500

                                                      }

                                                      if(dist(p2, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 4 && dist(p1, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 4){

                                                              if(seg12 < 165) seg12+=1.5; // 500

                                                      }

                                                      if(dist(p1, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 4 && dist(p2, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 4){

                                                              if(seg23 < 165) seg23+=1.5; // 500

                                                      }

                                                      if(dist(p1, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 4 && dist(p2, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y ) ) < 4){

                                                              if(seg23 < 165) seg23+=1.5; // 500

                                                      }

                                                      if(dist(p1, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 4 && dist(p2, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 4){

                                                              if(seg13 < 165) seg13+=1.5; // 500

                                                      }

                                                      if(dist(p1, Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y ) ) < 4 && dist(p2, Point(circlespm[maxhidx].x,circlespm[maxhidx].y ) ) < 4){

                                                              if(seg13 < 165) seg13+=1.5; // 500

                                                      }
                                              }

                                      }

                              }

                      }
              }


      if(seg12 > 0) seg12 --; //getmin(seg12, seg13, seg23);
      if(seg23 > 0) seg23--; //getmin(seg12, seg13, seg23);
      if(seg13 > 0) seg13--;///getmin(seg12, seg13, seg23);



      /*line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                      Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                      Scalar(30,255,30), 15, 8 );
      line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                              Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                              Scalar(30,255,30), 15, 7 );*/
      // collision detection 	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  // 165
      if( abs(circlespm[maxhidx].redetects-circlespm[maxhidx2].redetects)>500 && abs(circlespm[maxhidx].redetects-circlespm[maxhidx3].redetects)>500 && abs(circlespm[maxhidx2].redetects-circlespm[maxhidx3].redetects)>500   )
      {

              if(seg12 < seg23 && seg12 < seg13){
            	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  //237
                              if(dist(Point(circlespm[maxhidx].x, circlespm[maxhidx].y),Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y)) > 40 )
                                                              line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                      Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                      Scalar(30,255,30), 1, 8 );
                              else
                                      line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                              Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                              Scalar(0,0,255), 15, 8 );

                      }

                      if(seg23 < seg12 && seg23 < seg13){

                                      if(dist(Point(circlespm[maxhidx2].x, circlespm[maxhidx2].y),Point(circlespm[maxhidx3].x, circlespm[maxhidx3].y))> 40 )
                                                                      line( image, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                              Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                              Scalar(30,255,30), 1, 8 );
                                      else
                                              line( image, Point(circlespm[maxhidx2].x,circlespm[maxhidx2].y),
                                                                                      Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                      Scalar(0,0,255), 15, 8 );

                              }

                      if(seg13 < seg12 && seg13 < seg23){

                                              if(dist(Point(circlespm[maxhidx].x, circlespm[maxhidx].y),Point(circlespm[maxhidx3].x, circlespm[maxhidx3].y))>40 )
                                                                              line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                                      Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                                      Scalar(30,255,30), 1, 8 );
                                              else
                                                      line( image, Point(circlespm[maxhidx].x,circlespm[maxhidx].y),
                                                                                              Point(circlespm[maxhidx3].x,circlespm[maxhidx3].y),
                                                                                              Scalar(0,0,255), 15, 8 );

                                      }

      }

      //imshow("Output Image", imageOut);


}

int main(int argc, char** argv ){

	namedWindow("Original Video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED );
    namedWindow("img",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    namedWindow("img2",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    namedWindow("trackbars",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    createTrackbar("rmin","trackbars", &rmin, 40, trackbarchange, 0);
    createTrackbar("rmax","trackbars", &rmax, 40, trackbarchange, 0);
    createTrackbar("cannythreshmax","trackbars", &cannythreshmax, 200, trackbarchange, 0);
    createTrackbar("centerthresh","trackbars", &centerthresh, 200, trackbarchange, 0);
    createTrackbar("cthreshl","trackbars", &cthreshl, 255, trackbarchange, 0);
    createTrackbar("cthreshh","trackbars", &cthreshh, 255, trackbarchange, 0);
    createTrackbar("maxapproxdist","trackbars", &maxapproxdist, 200, trackbarchange, 0);
    createTrackbar("minbbheight","trackbars", &minbbheight, 200, trackbarchange, 0);
    createTrackbar("minbbwidth","trackbars", &minbbwidth, 2000, trackbarchange, 0);
    createTrackbar("minbbheight2","trackbars", &minbbheight2, 200, trackbarchange, 0);
    createTrackbar("minbbwidth2","trackbars", &minbbwidth2, 200, trackbarchange, 0);


	//segmentsV = new vector<segmentel>();
	//subcirclesV = new vector<segmentCircle>();

	VideoCapture video = VideoCapture("videos/robotArmTUCreal_CLIPCHAMP_keep_CLIPCHAMP_480p.ogv");

	if(!video.isOpened()){					// if video file is missing
		cout << "Video not found" << endl;
		return 1;
	}
	while(video.read(image)){

		imshow("Original Video", image);

		Mat grad, grad_x, grad_y, abs_grad_x, abs_grad_y, sobelImage;

		int ddepth = CV_16S;
		int scale = 1;
		int delta = 0;

		//erode(image,image,0,2);
		int erosion_type;
		int erosion_elem = 0;
		int erosion_size = 10;
		if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
		else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
		else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

		element = getStructuringElement( erosion_type,
											   Size( 2*erosion_size + 1, 2*erosion_size+1 ),
											   Point( erosion_size, erosion_size ) );

		//1. dilate all small things
		//2. erode remaining

		Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_x, abs_grad_x );
		Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		convertScaleAbs( grad_y, abs_grad_y );
		addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobelImage );

		cvtColor( image, imagegray, CV_RGB2GRAY );
		//imagegray = Scalar::all(128) - imagegray;
		//equalizeHist( imagegray, imagegray );
		Mat mask;
		inRange(image, Scalar(130,100,100), Scalar(180,200,200), mask);
		Mat black_image(imagegray.size(), CV_8U, Scalar(0));
		black_image.copyTo(imagegray, mask);

		imshow( "Sobel Edge Video", sobelImage);

		trackbarchange(0,0);

        if(!canny_output.empty()) imshow( "img", canny_output);
        imshow( "img2", image);

		int k = waitKey(1);
        if(k == 27)
        {
            break;
        }

	}


}

