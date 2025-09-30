//
//  frame_separation.cpp
//  main
//
//  Created by 田中海斗 on 2022/06/01.
//

#include "frame_separation.hpp"
std::vector<cv::Mat> Framedetect::frame_detect(cv::Mat &src_page){//コマ抽出関数
    std::vector<cv::Mat> panel_images;//コマ選択のための一時的格納配列
    
    cv::Size img_size(src_page.cols, src_page.rows);//画像サイズ
    cv::Mat zeros = cv::Mat::zeros(img_size, CV_8UC1);
    cv::Mat color_page = src_page;
    cv::cvtColor(color_page, color_page, cv::COLOR_GRAY2BGR);
    cv::Mat gray_img;
    if (src_page.type() != CV_8UC1) {//入力画像がグレースケール(チャネル１)でない時
        cv::cvtColor(src_page, gray_img, cv::COLOR_BGR2GRAY);
    }else{
        gray_img = src_page;
    }
    
    cv::Mat binForSpeechBalloon_img;
    
    threshold(gray_img.clone(), binForSpeechBalloon_img, 230, 255, cv::THRESH_BINARY);//二値化
    cv::erode(binForSpeechBalloon_img, binForSpeechBalloon_img, cv::Mat(),cv::Point(-1,-1), 1);//収縮
    cv::dilate(binForSpeechBalloon_img, binForSpeechBalloon_img, cv::Mat(),cv::Point(-1,-1), 1);//膨張
    
    std::vector<cv::Vec4i> hierarchy2;
    std::vector<std::vector<cv::Point> > hukidashi_contours;//輪郭情報
    
    //吹き出し塗りつぶしのための輪郭抽出
    cv::findContours(binForSpeechBalloon_img, hukidashi_contours, hierarchy2, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    
    cv::Mat gaussian_img;
    
    cv::GaussianBlur(gray_img, gaussian_img, cv::Size(3,3), 0);//平滑化
    
    extractSpeechBalloon(hukidashi_contours,hierarchy2, gaussian_img);//吹き出しによるコマ未検出を防ぐため吹き出し検出で塗りつぶし
    cv::Mat inverse_bin_img;  //階調反転二値画像
    threshold(gaussian_img, inverse_bin_img, 210, 255, cv::THRESH_BINARY_INV);//二値化
    
    findFrameExistenceArea(inverse_bin_img);//コマ存在領域推定
    
    cv::Mat canny_img;  //キャニーフィルタ後の画像
    cv::Canny(gray_img, canny_img, 120, 130, 3);
    
    std::vector<cv::Point2f> lines;
    std::vector<cv::Point2f> lines2;
    HoughLines(canny_img, lines, 1, M_PI/180.0, 50);  //直線検出
    HoughLines(canny_img, lines2, 1, M_PI/360.0, 50);
    
    cv::Mat lines_img = cv::Mat::zeros(img_size, CV_8UC1);
    
    drawHoughLines(lines, lines_img);
    drawHoughLines2(lines2, lines_img);
    
    cv::Mat and_img = cv::Mat::zeros(img_size, CV_8UC1);  //論理積画像
    bitwise_and(inverse_bin_img, lines_img, and_img);  //二値画像と直線検出画像の論理積
    
    std::vector<std::vector<cv::Point>> contours; //輪郭点群
    cv::Mat tmp_img; //一次的に利用する画像
    and_img.copyTo(tmp_img);
    cv::findContours(tmp_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); //輪郭検出 (ラベリング)
    
    cv::Mat boundingbox_from_and_img = and_img.clone();
    
    cv::Mat complement_and_img;
    createAndImgWithBoundingBox(boundingbox_from_and_img, contours, inverse_bin_img, complement_and_img);
    
    std::vector <std::vector<cv::Point>>contours3;
    std::vector<cv::Rect> bounding_boxes; //バウンディングボックス群

    findContours(complement_and_img, contours3, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//輪郭抽出
    
    //バウンディングボックスの登録
    for (int i=0; i<contours3.size(); i++) {
        cv::Rect tmp_bounding_box = cv::boundingRect (contours3[i]); //バウンディングボックスの取得
        //int Area = tmp_bounding_box.area();
        
        if(judgeAreaOfBoundingBox(tmp_bounding_box, complement_and_img.cols*complement_and_img.rows) == true){
            bounding_boxes.push_back(tmp_bounding_box); //バウンディングボックスの登録
        }
    }
    //ページごとでコマの輪郭描画
    for(int i=0; i<contours3.size(); i++){
        
        //近似の座標リストの宣言
        std::vector<cv::Point> approx;
        
        //輪郭を直線近似する
        approxPolyDP(contours3[i], approx, 6, true);
        
        //外接矩形生成
        cv::Rect brect = boundingRect(contours3[i]);
        
        //左上,右下の座標(xmin,ymin),(xmax,ymax)
        int xmin = brect.x;
        int ymin = brect.y;
        int xmax = brect.x+brect.width;
        int ymax = brect.y+brect.height;

        //端に寄せる
        if(xmin<6)xmin=0;
        if(xmax>inverse_bin_img.cols-6)xmax=inverse_bin_img.cols;
        if(ymin<6)ymin=0;
        if(ymax>inverse_bin_img.rows-6)ymax=inverse_bin_img.rows;
        
        //バウンディングボックス（外接矩形）の4点
        Points bbPoint(cv::Point(xmin,ymin),cv::Point(xmin,ymax),cv::Point(xmax,ymin),cv::Point(xmax,ymax));
        
        //最終的な代表点
        Points definitePanelPoint;
        
        //点が確定か否か
        bool flag_LT = false;
        bool flag_LB = false;
        bool flag_RT = false;
        bool flag_RB = false;
        
        //最小距離初期値（ページ高さにしとくことで必ず更新される）
        double bb_min_LT=src_page.rows;
        double bb_min_RT=src_page.rows;
        double bb_min_LB=src_page.rows;
        double bb_min_RB=src_page.rows;
        
        bool isOverlap=true;
        
        if(judgeAreaOfBoundingBox(brect,src_page.cols * src_page.rows) == true){
            //バウンディングボックスのオーバーラップ判定
            judgeBoundingBoxOverlap(isOverlap, bounding_boxes, brect);
        }else{
            isOverlap=false;
        }
        
        if(isOverlap==false)continue;
        
        for (int j=0; j<approx.size(); ++j ) {
            
            cv::Point p = approx[j];
            
            definePanelCorners(flag_LT, p, bb_min_LT,  pageCorners.lt,  definitePanelPoint.lt, bbPoint.lt);
            definePanelCorners(flag_LB, p, bb_min_LB,  pageCorners.lb,  definitePanelPoint.lb, bbPoint.lb);
            definePanelCorners(flag_RT, p, bb_min_RT,  pageCorners.rt,  definitePanelPoint.rt, bbPoint.rt);
            definePanelCorners(flag_RB, p, bb_min_RB,  pageCorners.rb,  definitePanelPoint.rb, bbPoint.rb);
            align2edge(definitePanelPoint, inverse_bin_img);
        }
        
        definitePanelPoint.renew_line();
        
        //透過画像（4チャンネル）
        cv::Mat alphaImage;
        
        //RGB-→RGBA
        cvtColor(src_page, alphaImage, cv::COLOR_RGB2RGBA);
        //コマ以外を透過
        createAlphaImage(alphaImage, definitePanelPoint);
        
        //サイズを合わせる
        cv::Mat cut_img(alphaImage, brect);
        
        //コマ画像を保存
        panel_images.push_back(cut_img);
    //    cv::waitKey(); //キー入力待ち (止める)
//        std::cout << "xmin :" << xmin <<std::endl;;
//        std::cout << "ymin :" << ymin <<std::endl;
//        std::cout << "xmax :" << xmax <<std::endl;
//        std::cout << "ymax :" << ymax <<std::endl;
        //コマ線描画
        cv::line(color_page, definitePanelPoint.lt, definitePanelPoint.rt, CV_RGB(255,0,0), 2, 8, 0);
        cv::line(color_page, definitePanelPoint.rt, definitePanelPoint.rb, CV_RGB(255,0,0), 2, 8, 0);
        cv::line(color_page, definitePanelPoint.rb, definitePanelPoint.lb, CV_RGB(255,0,0), 2, 8, 0);
        cv::line(color_page, definitePanelPoint.lb, definitePanelPoint.lt, CV_RGB(255,0,0), 2, 8, 0);
    }
//    cv::imshow("a", color_page); //画像の表示
//    cv::waitKey(); //キー入力待ち (止める)
   
    return panel_images;
}
//吹き出し抽出
void Framedetect::extractSpeechBalloon(std::vector<std::vector<cv::Point> > fukidashi_contours, std::vector<cv::Vec4i> hierarchy2, cv::Mat &gaussian_img){
    // 輪郭の描画
    
    for(int i=0;i<fukidashi_contours.size();i++){
        
        //(1) 極端に小さい，または大きい輪郭の除外
        //凸包の取得面積が一定以上なら取得
        double area = cv::contourArea(fukidashi_contours[i]);
        double length = cv::arcLength(fukidashi_contours[i], true);
        double en=0;
        
        if(gaussian_img.rows * gaussian_img.cols * 0.008 <= area &&
           area < gaussian_img.rows * gaussian_img.cols * 0.03){
            
            en= 4.0 * M_PI * area / (length * length);
            if( en >0.4 ){//en >0.4
                // 画像，輪郭，描画輪郭指定インデックス，色，太さ，種類，階層構造，描画輪郭の最大レベル
                cv::drawContours(gaussian_img, fukidashi_contours, i, 0, -1, cv::LINE_AA, hierarchy2, 1);
                //cv::drawContours(mask_Mat, contours, i, 255, -1, CV_AA, hierarchy, max_level);
                //                baloon_contours.push_back(contours[i]);
                //                tmp_contours.push_back(contours[i]);
            }
        }
    }
}
void Framedetect::findFrameExistenceArea(const cv::Mat &inverse_bin_img){
    
    int *histogram;
    histogram = (int *)calloc(inverse_bin_img.cols, sizeof(int));
    
    for(int y=0; y<inverse_bin_img.rows; y++){
        for(int x=0; x<inverse_bin_img.cols; x++){
            if(x<=2 || x>=inverse_bin_img.cols-2 || y<=2 || y>=inverse_bin_img.cols-2){
                continue;
            }
            if(0<inverse_bin_img.at<uchar>(y,x)){
                histogram[x]++;
                
            }
        }
    }
    
    int min_x=0, max_x=inverse_bin_img.cols-1;
    
    for(int x=0; x<inverse_bin_img.cols; x++){
        if(0<histogram[x]){
            min_x = x;
            break;
        }
    }
    
    for(int x=inverse_bin_img.cols-1; x>=0; x--){
        if(0<histogram[x]){
            max_x = x;
            break;
        }
    }
    
    //誤差は両端に寄せる
    if(min_x<6)min_x=0;
    if(max_x>inverse_bin_img.cols-6)max_x=inverse_bin_img.cols;

    
    pageCorners.lt = cv::Point(min_x,0);
    pageCorners.rt = cv::Point(max_x,0);
    pageCorners.lb = cv::Point(min_x,inverse_bin_img.rows);
    pageCorners.rb = cv::Point(max_x,inverse_bin_img.rows);
    
    cv::Mat rec_img(cv::Size(inverse_bin_img.cols,inverse_bin_img.rows),CV_8UC3);
}

//直線描画(lines1)
void Framedetect::drawHoughLines(std::vector<cv::Point2f> lines, cv::Mat &drawLinesImage){
    
    for(int i=0; i<MIN(lines.size(), 100); i++){
        
        cv::Point2f line = lines[i];
        
        float rho = line.x;  //ρ
        float theta =line.y; //θ
        
        //直線: a(x-xθ) = b(y=yθ)
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a*rho;
        double y0 = b*rho;
        
        cv::Point pt1, pt2;
        
        pt1.x = x0-2000*b;
        pt1.y = y0+2000*a;
        pt2.x = x0+2000*b;
        pt2.y = y0-2000*a;
        
        cv::line(drawLinesImage, pt1, pt2, cv::Scalar(255), 1, cv::LINE_AA);
    }
}

//直線描画(lines2)
void Framedetect::drawHoughLines2(std::vector<cv::Point2f> lines, cv::Mat &drawLinesImage){
    
    for(int i=0; i<MIN(lines.size(), 100); i++){
        
        cv::Point2f line = lines[i];
        
        float rho = line.x;  //ρ
        float theta =line.y; //θ
        
        //直線: a(x-xθ) = b(y=yθ)
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a*rho;
        double y0 = b*rho;
        
        cv::Point pt1, pt2;
        
        pt1.x = x0-2000*b;
        pt1.y = y0+2000*a;
        pt2.x = x0+2000*b;
        pt2.y = y0-2000*a;
        
        cv::line(drawLinesImage, pt1, pt2, cv::Scalar(255), 1, cv::LINE_AA);
    }
}
//バウンディングボックスを利用して矩形の欠損を埋める処理（階調反転二値画像と輪郭の論理積）
void Framedetect::createAndImgWithBoundingBox(cv::Mat &src_img, const std::vector<std::vector<cv::Point>> contours, cv::Mat inverse_bin_img, cv::Mat &dst_img){
    //バウンディングボックスが描画された画像
   // Mat tmp_bounding_box_img = Mat::zeros(src_img.size(), CV_8UC1);
    
    for(int i=0; i<contours.size(); i++){
        cv::Rect bounding_box = boundingRect(contours[i]);
        if(judgeAreaOfBoundingBox(bounding_box, src_img.cols*src_img.rows) == false){
            continue;
        }
        //矩形描画
        rectangle(src_img, bounding_box,  cv::Scalar(255), 3);
    }
    bitwise_and(src_img, inverse_bin_img, dst_img);
}
//バウンディングボックス面積判定
bool Framedetect::judgeAreaOfBoundingBox(const cv::Rect bounding_box, int page_area){
    if(bounding_box.area() < 0.048 * page_area){
        return(false);
    }
    return(true);
}
//バウンディングボックスが内包されてるか
void Framedetect::judgeBoundingBoxOverlap(bool &isOverlap,  std::vector<cv::Rect> bounding_boxes, cv::Rect brect){
    for (int j=0; j<bounding_boxes.size(); j++) {
        //同一ならスキップ
        if (bounding_boxes[j].x == brect.x &&
            bounding_boxes[j].y == brect.y &&
            bounding_boxes[j].width == brect.width &&
            bounding_boxes[j].height == brect.height ) {
            continue;
        }
        
        cv::Rect overlap_rect = brect & bounding_boxes[j];
        if (overlap_rect.x==0 && overlap_rect.y==0 && overlap_rect.width==0 && overlap_rect.height==0) continue;
        
        if (overlap_rect.x == brect.x &&
            overlap_rect.y == brect.y &&
            overlap_rect.width == brect.width &&
            overlap_rect.height == brect.height ) {
            isOverlap=false;
        }
    }
}
//コマ角の決定
void Framedetect::definePanelCorners(bool &definite, cv::Point currentPoint, double &boundingBoxMinDist, cv::Point PageCornerPoint, cv::Point &definitePanelPoint, cv::Point boundingBoxPoint){
    if(definite==false){
        double pageCornerDist = norm(boundingBoxPoint-PageCornerPoint);
        if(pageCornerDist<8){
            definitePanelPoint = PageCornerPoint;
            definite = true;
        }else{
            double boundingBoxDist = norm(boundingBoxPoint- currentPoint);
            if(boundingBoxDist < boundingBoxMinDist){
                boundingBoxMinDist = boundingBoxDist;
                definitePanelPoint = currentPoint;
            }
        }
    }
}
//端の点を揃える
void Framedetect::align2edge(Points &definitePanelPoint,  cv::Mat inverse_bin_img){
    int th_edge=6;  //6px以内なら
    if(definitePanelPoint.lt.x<th_edge)definitePanelPoint.lt.x=0;
    if(definitePanelPoint.lt.y<th_edge)definitePanelPoint.lt.y=0;
    if(definitePanelPoint.rt.x>inverse_bin_img.cols-th_edge)definitePanelPoint.rt.x=inverse_bin_img.cols;
    if(definitePanelPoint.rt.y<th_edge)definitePanelPoint.rt.y=0;
    if(definitePanelPoint.lb.x<th_edge)definitePanelPoint.lb.x=0;
    if(definitePanelPoint.lb.y>inverse_bin_img.rows-th_edge)definitePanelPoint.lb.y=inverse_bin_img.rows;
    if(definitePanelPoint.rb.x>inverse_bin_img.cols-th_edge)definitePanelPoint.rb.x=inverse_bin_img.cols;
    if(definitePanelPoint.rb.y>inverse_bin_img.rows-th_edge)definitePanelPoint.rb.y=inverse_bin_img.rows;
}
//コマ以外を透過する
void Framedetect::createAlphaImage(cv::Mat &alphaImage, Points definitePanelPoint){
    for(int y=0; y<alphaImage.rows; y++){
        for(int x=0; x<alphaImage.cols; x++){
            cv::Vec4b px = alphaImage.at<cv::Vec4b>(y,x);
            //ここで領域外を指定
            if(definitePanelPoint.outside(cv::Point(x,y))){
                px[3]=0;  //アルファチャンネルを0に
                alphaImage.at<cv::Vec4b>(y,x)=px;
            }
        }
    }
}
// コンストラクタ
Points::Points(cv::Point lt, cv::Point lb ,cv::Point rt, cv::Point rb){
    this->lt = lt;
    this->lb = lb;
    this->rt = rt;
    this->rb = rb;
    renew_line();
}
void Points::renew_line(){
    top_line = {lt,rt,false};
    bottom_line = {lb,rb,false};
    left_line = {lb,lt,true};
    right_line = {rb,rt,true};
}
// 四角形の内側(直線上は含まない)にあるかの判定
bool Points::outside(cv::Point p){
    return top_line.judgeArea(p) == 1 ||
    right_line.judgeArea(p) == 0 ||
    bottom_line.judgeArea(p) == 0 ||
    left_line.judgeArea(p) == 1;
}
//コンストラクタ　メンバ変数の初期化
Line::Line(cv::Point p1, cv::Point p2, bool y2x){
    this->p1 = p1;  //this->メンバ変数 = 引数
    this->p2 = p2;
    this->y2x = y2x;
    calc();
}
Line::Line() {}
//傾きと切片を求める（左右の辺か縦の辺で分ける）
void Line::calc(){
    if(y2x){
        a = float(p2.x-p1.x)/(p2.y-p1.y);
        b = p1.x-p1.y*a;
    }else{
        a = float(p2.y-p1.y)/(p2.x-p1.x);
        b = p1.y-p1.x*a;
    }
}
//領域の判定
int Line::judgeArea(cv::Point p){
    if(y2x) p={p.y, p.x};
    //直線よりも上の領域
    if(p.y>a*p.x+b) return 0;
    //直線よりも下の領域
    else if(p.y<a*p.x+b) return 1;
    //直線上
    else return 2;
}
