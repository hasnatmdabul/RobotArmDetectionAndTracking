// Detection of circular bounding box
// Hasnat ASE Master Thesis
#include "opencv2/imgcodecs.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <cv.h>

#include <iostream>
#include <vector>



using namespace cv;
using namespace std;


struct segmentel{
	float x1;
	float y1;
	float r1;
	float x2;
	float y2;
	float r2;
	int isPart;
};

struct segmentCircle{ // segcircle
	float x, y, r;
	int redetects;
};

vector<segmentel>* segmentsV;
segmentel ls;  // long segments

segmentCircle centerCircle, bottomCircle, topCircle, centerSubCircle;
vector <segmentCircle>* subcirclesV; //-- later

int frame;
Mat image, element, imageGray; //imagegray
Mat imageOut; //imageout
RNG rng(12345);

int cannyThreshMax = 102; // min  100 -  cannythreshmax, centerthresh, rmin, rmax
int centerThresh = 8; // max 15
int radiusMin = 9; // min 25 for real, 12 for simulation
int radiusMax = 14; //max 52 for real, 14 for simulation
int circleClose = 490; //min 49 for real, 490 for simulation;
int segmentCircleClose = 500; // 49 for real

int cannyThreshMaxSmall = 102; // min  100 -  cannythreshmax, centerthresh, rmin, rmax
int centerThreshSmall = 15; // max 15
int radiusMinSmall = 19; // min 25 for real, 12 for simulation
int radiusMaxSmall = 21; //max 52 for real, 14 for simulation
int circleCloseSmall = 490; //min 49 for real, 490 for simulation;

int cthreshl = 53;//10; canny thresh low
int cthreshh = 138;//100; canny thresh high

int minDiagonal = 11; // 110
int maxDiagonal = 100; // 277
int maxApproximateDistance = 20;

int recsegdist = 45;

void detectcccb(segmentel s){


	//if current segment has the ct then set other one to cc
	if( s.x1 == topCircle.x && s.y1 == topCircle.y){

		centerCircle.x = s.x2;
		centerCircle.y = s.y2;
		centerCircle.r = s.r2;
	}
	else if(s.x2 == topCircle.x && s.y2 == topCircle.y){

		centerCircle.x = s.x1;
		centerCircle.y = s.y1;
		centerCircle.r = s.r1;
	}


	//if current segment has the cc and not the ct then set cb
	if( s.x1 == centerCircle.x && s.y1 == centerCircle.y && !(s.x2 == topCircle.x && s.y2 == topCircle.y) ){

			bottomCircle.x = s.x2;
			bottomCircle.y = s.y2;
			bottomCircle.r = s.r2;
	}
	else if( s.x2 == centerCircle.x && s.y2 == centerCircle.y && !(s.x1 == topCircle.x && s.y1 == topCircle.y)  ){

		bottomCircle.x = s.x1;
		bottomCircle.y = s.y1;
		bottomCircle.r = s.r1;
	}



}

void trackbarChange(int, void*) // void pointer can point any object, does not know which type
{
	image.copyTo(imageOut);

	//  hough circle for big circles --- Color WHITE
	vector<Vec3f> circles; // circle center co-ordinate x,y and radius r -- (x,y,r)
	HoughCircles( imageGray, circles, CV_HOUGH_GRADIENT, 1, imageGray.rows/8, cannyThreshMax, centerThresh, radiusMin, radiusMax );

	for( size_t i = 0; i < circles.size(); i++ )
	{
	 Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	 int radius = cvRound(circles[i][2]);
	 // Draw circle center
	 //circle( imageOut, center, 3, Scalar(0,255,0), -1, 8, 0 );
	 // Draw circle outline
	 circle( imageOut, center, radius, Scalar(255,255,255), 1, 8, 0 ); // white
	}

	for( size_t j = 0; j < circles.size(); j++ ){

				 //if close to  then assign to this
				 float dx = circles[j][0] - topCircle.x;
				 float dy = circles[j][1] - topCircle.y;
				 float distance = sqrt( dx*dx + dy*dy ); // dist
				 cout << "distance" << topCircle.x << endl;
				 if( distance < circleClose ){

					 topCircle.x = circles[j][0];
					 topCircle.y = circles[j][1];
					 topCircle.r = circles[j][2];

					 Point centerBigCircle(cvRound(topCircle.x), cvRound(topCircle.y));
					 int radiusBigCircle = cvRound(topCircle.r);
					 circle( imageOut, centerBigCircle, radiusBigCircle, Scalar(0,0,255), 3, 8, 0 ); // red


				 }
	 }

	//  hough circle for small circles --- color BLACK
	vector<Vec3f> circlesSmall;
	HoughCircles( imageGray, circlesSmall, CV_HOUGH_GRADIENT, 1, imageGray.rows/8, cannyThreshMaxSmall, centerThreshSmall, radiusMinSmall, radiusMaxSmall );

	for( size_t i = 0; i < circlesSmall.size(); i++ )
	{
		 cout << "Number of small circles : " << circlesSmall.size() << endl;

		 // vector pointer refference direfference ....
		if(subcirclesV->empty()){

			segmentCircle csub;
			csub.x = circlesSmall[i][0];
			csub.y = circlesSmall[i][1];
			csub.r = circlesSmall[i][2];
			csub.redetects = 0;
			 Point centercsub(cvRound(csub.x), cvRound(csub.y));
			 int radiuscsub = cvRound(csub.r);
			 circle( imageOut, centercsub, radiuscsub, Scalar(0,255,255), 3, 8, 0 );
			 cout << "Number of small Sub circles : " << subcirclesV->size() << endl;

			subcirclesV->push_back(csub);
		}
		else{

			for(size_t m=0; m<subcirclesV->size(); m++){

				float dx = (*subcirclesV)[m].x - circlesSmall[i][0];
				float dy = (*subcirclesV)[m].y - circlesSmall[i][1];
				float dist = sqrt( dx*dx + dy*dy );

				//update model with closest one, by circle dist first used by segments segcircleclose
				if(dist < segmentCircleClose){

					(*subcirclesV)[m].x = circlesSmall[i][0];
					(*subcirclesV)[m].y = circlesSmall[i][1];
					(*subcirclesV)[m].r = circlesSmall[i][2];
					(*subcirclesV)[m].redetects++;
				}




				//closest big circle is ct
				 for( size_t j = 0; j < circles.size(); j++ ){

					 float curx = circles[j][0];
					 float cury = circles[j][1];

					 //if(ct.x != 0){
					//	 curx = ct.x;
					//	 cury = ct.y;
					 //}
					 //detectionen von blue, nur oft detektierte blue setzen

					 //if close to subcircle then assign to this
					 float dx = curx - (*subcirclesV)[m].x;
					 float dy = cury - (*subcirclesV)[m].y;
					 float distance = sqrt( dx*dx + dy*dy );
					 if( distance < circleClose && (*subcirclesV)[m].redetects>2 ){

						centerSubCircle.x = (*subcirclesV)[m].x;
						centerSubCircle.y = (*subcirclesV)[m].y;
						centerSubCircle.r = (*subcirclesV)[m].r;
						topCircle.x = circles[j][0];
						topCircle.y = circles[j][1];
						topCircle.r = circles[j][2];

					 }
				 }
			}

		}

	 Point centerSmallCircles(cvRound(circlesSmall[i][0]), cvRound(circlesSmall[i][1]));
	 int radiusSmallCircles = cvRound(circlesSmall[i][2]);
	 // circle center
	 //circle( imageOut, centerSmallCircles, 3, Scalar(0,0,0), -1, 8, 0 );
	 // circle outline
	 circle( imageOut, centerSmallCircles, radiusSmallCircles, Scalar(0,0,0), 3, 8, 0 );

	}
	//############### end small circles
	//draw ctop
	//circle( imageOut, Point(topCircle.x, topCircle.y), 3, Scalar(0,255,0), -1, 8, 0 );
	//circle( imageOut, Point(topCircle.x, topCircle.y), topCircle.r, Scalar(0,255,0), 3, 8, 0 );

	//------------ bounding box start -------------

	Mat canny_output;
	vector<vector<Point> > contours;

	vector<Vec4i> hierarchy;
	Canny( imageGray, canny_output, cthreshl, cthreshh, 3 );//relation 2:1

	//dilate( canny_output, canny_output, element );

	findContours( canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	//andere approxmethoden bringen scheinbar nur minimal was
	//rectangle is in 4 vertice mode

	vector<vector<Point> > contoursApprox( contours.size() );
	vector<bool > allowedToDraw( contours.size() );
	vector<RotatedRect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );
	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

	//1. iterate contours, approx contour movement from last movement to track also good contours
	for(size_t i = 0; i<contours.size(); i++){

		//ruckelt langsam approx poly notwendig
		approxPolyDP(contours[i],contoursApprox[i], (float)maxApproximateDistance/100, 1); //always CV_POLY_APPROX_DP
		boundRect[i] = minAreaRect( Mat(contoursApprox[i]) );//boundingRect
		minEnclosingCircle( (Mat)contoursApprox[i], center[i], radius[i] );

	}

	//2.
	for(size_t i=0; i<boundRect.size(); i++){

		RotatedRect curRectangle = boundRect[i];
		//float ratio1 = (float) curRectangle.height / (float) curRectangle.width;
		//float ratio2 = (float) curRectangle.height / (float) curRectangle.width;

		//if( (abs(ratio1-cratio) < 1.0f) //|| //threash has only effect of shift of param
			//(abs(ratio2-cratio) < ratiothresh)
		//){

			//if (curRectangle.size.area() < detarea && curRectangle.size.area() > detareamin){
			float curdiag = sqrt(curRectangle.size.height*curRectangle.size.height+curRectangle.size.width*curRectangle.size.width);
			if ( curdiag > minDiagonal && curdiag < maxDiagonal ){

				allowedToDraw[i] = 1;
				//detheight = (detheight + boundRect[i].height)/2;
				//detratio = (detratio + (float)boundRect[i].height/ (float)boundRect[i].width)/2;
				//update current rectangle with this information for drawing
				//curRectangle.tl = Point(curRectangle.x,curRectangle.y);
				//float curwidth = boundRect[i].height / detratio;
				//curRectangle.br = Point(curRectangle.x+curwidth, curRectangle.y+curRectangle.height);
				//boundRect[i].width = curwidth;

				//update current model dimensions, rescaling ratio
				//ratio detect markant object on middle dist
					//can try an error through parameter space until detectin

				//location by barcode on nearest dist

				// //allow to track wrong dimensions after detection of condition object in cur
				//situation
			//}


		}

	}


		//draw bounding box
		for( size_t i = 0; i< contours.size(); i++ )
		 {
		   //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		   if(allowedToDraw[i]){

			   //drawContours( drawing, contoursApprox, i, color, 2, 8, hierarchy, 0, Point() );
			   //rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
			   //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );

			   Point2f rect_points[4]; boundRect[i].points( rect_points );
			  for( int j = 0; j < 4; j++ )
				 line( imageOut, rect_points[j], rect_points[(j+1)%4], Scalar(255,0,0), 1, 8 );

		   }

		 }
		//------------ bounding box end -------------

		//Segments
		if(segmentsV->empty() && !circles.empty() ){

			segmentel s;
			s.x1 = circles[0][0];
			s.y1 = circles[0][1];
			s.r1 = circles[0][2];
			s.x2 = circles[1][0];
			s.y2 = circles[1][1];
			s.r2 = circles[1][2];
			segmentsV->push_back(s);


			segmentel s2;
			s2.x1 = circles[1][0];
			s2.y1 = circles[1][1];
			s2.r1 = circles[1][2];
			s2.x2 = circles[2][0];
			s2.y2 = circles[2][1];
			s2.r2 = circles[2][2];
			segmentsV->push_back(s2);


			segmentel s3;
			s3.x1 = circles[2][0];
			s3.y1 = circles[2][1];
			s3.r1 = circles[2][2];
			s3.x2 = circles[0][0];
			s3.y2 = circles[0][1];
			s3.r2 = circles[0][2];
			segmentsV->push_back(s3);



		}
		//on update assign current segment to closest segment, allows for redetection... like before
		//foreach circle
		for( size_t i = 0; i < segmentsV->size(); i++ ){

			//	update to closest segment position foreach segment
			//int x = (*segmentsV)[i].x1;
			//int

			//foreach circle
			for( size_t j = 0; j < circles.size(); j++ ){

				//for each endpoint of segment checking
				float dx = (*segmentsV)[i].x1 - circles[j][0];
				float dy = (*segmentsV)[i].y1 - circles[j][1];
				float curdist = sqrt( dx*dx + dy*dy );

				if( curdist < segmentCircleClose ){

					(*segmentsV)[i].x1 = circles[j][0];
					(*segmentsV)[i].y1 = circles[j][1];
					(*segmentsV)[i].r1 = circles[j][2];
				}


				dx = (*segmentsV)[i].x2 - circles[j][0];
				dy = (*segmentsV)[i].y2 - circles[j][1];
				curdist = sqrt( dx*dx + dy*dy );

				if( curdist < segmentCircleClose ){

					(*segmentsV)[i].x2 = circles[j][0];
					(*segmentsV)[i].y2 = circles[j][1];
					(*segmentsV)[i].r2 = circles[j][2];
				}


			}

			//    if close update position


		}

		//get biggest segment for special case
		float dist = 0;
		for( size_t i = 0; i < segmentsV->size(); i++ ){

			float dx = (*segmentsV)[i].x1 - (*segmentsV)[i].x2;
		    float dy = (*segmentsV)[i].y1 - (*segmentsV)[i].y2;
		    float curdist = sqrt( dx*dx + dy*dy );


			if(curdist > dist){

				dist = curdist;
				ls = (*segmentsV)[i];
			}
		}

		int segcount =0;
		//select segments by middle bounding box detection
		for( size_t i = 0; i < segmentsV->size(); i++ ){


			//draw segments with closest center to rectangles center
			float x1 = (*segmentsV)[i].x1;
			float x2 = (*segmentsV)[i].x2;
			float y1 = (*segmentsV)[i].y1;
			float y2 = (*segmentsV)[i].y2;
			float cx = (x2+x1)/2;
			float cy = (y2+y1)/2;

			for( size_t l = 0; l< contours.size(); l++ ){

				if(allowedToDraw[l]){

					//Point bc = (boundRect[l].tl()+boundRect[l].br());
					//bc.x = bc.x/2; bc.y = bc.y /2;
					Point bc = boundRect[l].center;

					int dist = sqrt((bc.x-cx)*(bc.x-cx) + (bc.y-cy)*(bc.y-cy));

					if( dist < recsegdist  ){


						(*segmentsV)[i].isPart = 1;
						segcount++;

					}
				}
			}

		}

		//Segments end

		//redetect circleBottom, circleCenter and circleTop
		for( size_t i = 0; i < segmentsV->size(); i++ ){

					float x1 = (*segmentsV)[i].x1;
					float x2 = (*segmentsV)[i].x2;
					float y1 = (*segmentsV)[i].y1;
					float y2 = (*segmentsV)[i].y2;

					//if there are three allowed segments then skipp longes
					if(segcount == 3 && ls.x1==x1 && ls.x2==x2 && ls.y1==y1 && ls.y2==y2){

						continue;
					}

					if((*segmentsV)[i].isPart)
						detectcccb((*segmentsV)[i]);
		}
		//output segments
		//size_t segVlength = segmentsV->size();
		for( size_t i = 0; i < segmentsV->size(); i++ ){

			float x1 = (*segmentsV)[i].x1;
			float x2 = (*segmentsV)[i].x2;
			float y1 = (*segmentsV)[i].y1;
			float y2 = (*segmentsV)[i].y2;

			//if there are three allowed segments then skipp longes
			if(segcount == 3 && ls.x1==x1 && ls.x2==x2 && ls.y1==y1 && ls.y2==y2){

				continue;
			}

			//segement cb - ct is not part
			if( ((*segmentsV)[i].x1 == bottomCircle.x && (*segmentsV)[i].y1 == bottomCircle.y && (*segmentsV)[i].x2 == topCircle.x && (*segmentsV)[i].y2 == topCircle.y)||((*segmentsV)[i].x1 == topCircle.x && (*segmentsV)[i].y1 == topCircle.y && (*segmentsV)[i].x2 == bottomCircle.x && (*segmentsV)[i].y2 == bottomCircle.y))
			{
				(*segmentsV)[i].isPart = 0;

			}

			if((*segmentsV)[i].isPart)
				line( imageOut,
					  Point( (*segmentsV)[i].x1, (*segmentsV)[i].y1 ),
					  Point( (*segmentsV)[i].x2, (*segmentsV)[i].y2 ),
					  Scalar(255,255,255),
					  3,
					  CV_AA);
		}




			//draw cc cb
			//circle( imageOut, Point(centerCircle.x, centerCircle.y), 3, Scalar(0,213,0), -1, 8, 0 );
			circle( imageOut, Point(centerCircle.x, centerCircle.y), centerCircle.r, Scalar(255,0,0), 3, 8, 0 );
			//circle( imageOut, Point(bottomCircle.x, bottomCircle.y), 3, Scalar(0,150,0), -1, 8, 0 );
			circle( imageOut, Point(bottomCircle.x, bottomCircle.y), bottomCircle.r, Scalar(120,123,120), 3, 8, 0 );


	imshow("Output Image", imageOut);
}

int main(int argc, char** argv ){

	namedWindow("Original Video", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED );
	namedWindow("Trackbar", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED );
	namedWindow("Trackbar2", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED );
	createTrackbar("cannyThreshMax","Trackbar", &cannyThreshMax, 200, trackbarChange, 0);
	createTrackbar("centerThresh","Trackbar", &centerThresh, 200, trackbarChange, 0);
	createTrackbar("radiusMin","Trackbar", &radiusMin, 200, trackbarChange, 0);
	createTrackbar("radiusMax","Trackbar", &radiusMax, 200, trackbarChange, 0);

	createTrackbar("cannyThreshMaxSmall","Trackbar", &cannyThreshMaxSmall, 200, trackbarChange, 0);
	createTrackbar("centerThreshSmall","Trackbar", &centerThreshSmall, 200, trackbarChange, 0);
	createTrackbar("radiusMinSmall","Trackbar", &radiusMinSmall, 200, trackbarChange, 0);
	createTrackbar("radiusMaxSmall","Trackbar", &radiusMaxSmall, 200, trackbarChange, 0);

	createTrackbar("minDiagonal","Trackbar", &minDiagonal, 2000, trackbarChange, 0);
	createTrackbar("maxDiagonal","Trackbar", &maxDiagonal, 2000, trackbarChange, 0);

	createTrackbar("maxApproximateDistance","Trackbar", &maxApproximateDistance, 200, trackbarChange, 0);
	createTrackbar("recsegdist","Trackbar", &recsegdist, 600, trackbarChange, 0);

	createTrackbar("Circle Close","Trackbar2", &circleClose, 600, trackbarChange, 0);
	createTrackbar("Circle Close Small","Trackbar2", &circleCloseSmall, 600, trackbarChange, 0);
	createTrackbar("Segment Close","Trackbar2", &segmentCircleClose, 600, trackbarChange, 0);

	segmentsV = new vector<segmentel>();
	subcirclesV = new vector<segmentCircle>();
	   //cout << "size of v1 = " << v1.size() << endl;

	VideoCapture video = VideoCapture("videos/robotArmTUCreal_CLIPCHAMP_keep_CLIPCHAMP_480p.ogv");

	frame = 0;

	if(!video.isOpened()){					// if video file is missing
		cout << "Video not found" << endl;
		return 1;
	}
	while(video.read(image)){

		//Sobel(image, sobelImage, int 1, int 2, int 2, 3, 1, 0, BORDER_DEFAULT )
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

		cvtColor( image, imageGray, CV_RGB2GRAY );
		//imageGray = Scalar::all(128) - imageGray;
		//equalizeHist( imageGray, imageGray );
		Mat mask;
		inRange(image, Scalar(130,100,100), Scalar(180,200,200), mask);
		Mat black_image(imageGray.size(), CV_8U, Scalar(0));
		black_image.copyTo(imageGray, mask);

		imshow( "Sobel Edge Video", sobelImage);
		imshow( "Mask Video", mask);

		trackbarChange(0,0);


		//Sobel edge detection
		//Sobel edge detection end
		int k = waitKey(1);
        if(k == 27)
        {
            break;
        }

	}


}

