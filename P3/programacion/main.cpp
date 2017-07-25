 /*
 * main.cpp
 *
 *  Created on: 02/01/2017
 *      Author: Ignacio Martín Requena
 */

#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}


using namespace std;
using namespace cv;


Mat leeimagen(string filename, int flagColor) {
	Mat res = imread(filename, flagColor);
	return res;
}

void pintaI(Mat im, string name = "Ventana") {
	namedWindow(name, im.channels());
	imshow(name, im);
	cvWaitKey();
	destroyWindow(name);
}

void pintaImagenes(vector<Mat> vim, string windowname, int col = CV_8UC3) {
	Mat fin = vim[0];
	for (uint i = 1; i < (vim.size()); i++) {
		Size sz1 = fin.size();
		Size sz2 = vim[i].size();
		Mat auxfin(sz1.height, sz1.width + sz2.width, col);
		Mat izq(auxfin, Rect(0, 0, sz1.width, sz1.height));
		fin.copyTo(izq);
		Mat der(auxfin, Rect(sz1.width, 0, sz2.width, sz2.height));
		vim[i].copyTo(der);
		fin = auxfin;
	}

	imshow(windowname, fin);
	cvWaitKey();
	destroyAllWindows();
}

/*****************************************************************************************/
/*****************************************************************************************/
/*********************************** TRABAJO 2 *******************************************/
/*****************************************************************************************/
/*****************************************************************************************/

///////////////////////////////////////////////////////
///////////////// EJERCICIO 2 /////////////////////////
///////////////////////////////////////////////////////

void kazeFuncion(Mat& img, vector<KeyPoint>& keypoints, Mat& kaze, Mat& descriptors){
	//Creamos el detector con los parámetros deseados
	Ptr<KAZE> KAZE = KAZE::create();
	//Hacemos la detección de los puntos clave en la imágen
	KAZE->detect(img, keypoints);
	//Obtenemos los descriptores
	KAZE->compute(img, keypoints, descriptors);
	//Pintamos los keypoints en la imagen
	drawKeypoints(img, keypoints, kaze);
}

void akazeFuncion(Mat& img, vector<KeyPoint>& keypoints, Mat& akaze, Mat& descriptors) {
	//Creamos el detector con los parámetros deseados
	Ptr<AKAZE> AKAZE = AKAZE::create();
	//Hacemos la detección de los puntos clave en la imágen
	AKAZE->detect(img, keypoints);
	//Obtenemos los descriptores
	AKAZE->compute(img, keypoints, descriptors);
	//Pintamos los keypoints en la imagen
	drawKeypoints(img, keypoints, akaze);
}

void PuntosEnComun(Mat im1, Mat im2, Mat descriptors1, Mat descriptors2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, bool crossCheck, Mat& salida, vector<DMatch>& matches){

	//Realizamos la búsqueda de las correspondencias
	BFMatcher matcher(NORM_L2, crossCheck);
	matcher.match(descriptors1, descriptors2, matches);

	//Ordenamos las correspondencias para buscar las mejores de forma más eficiente
	sort(matches.begin(), matches.end());

	//Pintamos los puntos de correspondencia
	drawMatches(im1, keypoints1, im2, keypoints2, matches, salida);

	cout << "He han conseguido " << matches.size() << " correspondencias" << endl;

}

/*****************************************************************************************/
/*****************************************************************************************/
/*********************************** TRABAJO 3 *******************************************/
/*****************************************************************************************/
/*****************************************************************************************/


/*****************************************************************************************/
/*********************************** Ejercicio 1 *****************************************/
/*****************************************************************************************/

/* Función que genera una matriz de cámara finita P a partir de valores aleatorios */
void GenerarMatrizP(Mat &P){
	//Matriz M que será M 3x3
	Mat M(3, 3, CV_32F);
	bool buena = false;

	//Mientras P no sea válida:
	while (!buena){
		//Genero P con valores entre -1 y 1
		P = Mat(3, 4, CV_32F);
		randu(P, -1, 1);
		//Extraigo y compruebo el determinante de M;
		M = P.colRange(0, 3);
		if (determinant(M) > 0){
			//Si es válida me la quedo
			buena = true;
		}
	}
}

/* Función para generar el patrón 3D */
vector<Point3d> GenerarPatron3D(){
	vector<Point3d> resultado;
	//Vamos a usar x1 y x2 con valores entre 0.1 hasta 1, aumentando de 0.1 en 0.1
	for (double x1 = 0.1; x1 <= 1; x1 += 0.1){
		for (double x2 = 0.1; x2 <= 1; x2 += 0.1){
			//Generamos los puntos p1 y p2 como p1 = (0,x1,x2) y p2 = (x2,x1,0)
			Point3d p1 = Point3d(0.0, x1, x2);
			Point3d p2 = Point3d(x2, x1, 0.0);
			resultado.push_back(p1);
			resultado.push_back(p2);
		}
	}
	return resultado;
}

/* Función que a partir de puntos 3D se transforman a matrices 4x1 */
vector<Mat> generarMatricesDePuntos(vector<Point3d> punt){
	vector<Mat> resultado;
	for (uint i = 0; i < punt.size(); i++){
		Mat tmp = Mat(4, 1, CV_32F);
		tmp.at<float>(0, 0) = punt[i].x;
		tmp.at<float>(1, 0) = punt[i].y;
		tmp.at<float>(2, 0) = punt[i].z;
		tmp.at<float>(3, 0) = 1;
		resultado.push_back(tmp);
	}
	return resultado;
}

/* Función que genera puntos proyectados 3x1 a partir de la matriz P y de una matriz punto */
vector<Mat> generarPuntosProyectados(Mat P, vector<Mat> mat){
	vector<Mat> res;
	for (uint i = 0; i<mat.size(); i++){
		Mat aux = P*mat[i];
		res.push_back(aux);
	}
	return res;
}

/* Función que a partir de los puntos proyectados, genera las coordenadas pixel */
vector<Mat> generarCoordenadasPixel(vector<Mat> pp){
	vector<Mat> res;
	for (uint i = 0; i < pp.size(); i++){
		Mat aux = Mat(2, 1, CV_32F);
		aux.at<float>(0, 0) = pp[i].at<float>(0, 0) / pp[i].at<float>(2, 0);
		aux.at<float>(1, 0) = pp[i].at<float>(1, 0) / pp[i].at<float>(2, 0);
		res.push_back(aux);
	}
	return res;
}


/* Función para calcular las rotaciones, así como la matriz K y R */
void calcularRotaciones(Mat M, Mat &K, Mat &R){
	Mat Qx = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	Mat Qy = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	Mat Qz = (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

	float c = M.at<float>(2, 2) / sqrt(pow(M.at<float>(2, 2), 2) + pow(M.at<float>(2, 1), 2));
	float s = M.at<float>(2, 1) / sqrt(pow(M.at<float>(2, 2), 2) + pow(M.at<float>(2, 1), 2));

	Qx.at<float>(1, 1) = c;
	Qx.at<float>(1, 2) = s;
	Qx.at<float>(2, 1) = (-1)*s;
	Qx.at<float>(2, 2) = c;

	K = M*Qx;

	c = K.at<float>(2, 2) / sqrt(pow(K.at<float>(2, 2), 2) + pow(K.at<float>(2, 0), 2));
	s = K.at<float>(2, 0) / sqrt(pow(K.at<float>(2, 2), 2) + pow(K.at<float>(2, 0), 2));

	Qy.at<float>(0, 0) = c;
	Qy.at<float>(0, 2) = (-1)*s;
	Qy.at<float>(2, 0) = s;
	Qy.at<float>(2, 2) = c;

	K = K*Qy;

	c = K.at<float>(1, 1) / sqrt(pow(K.at<float>(1, 1), 2) + pow(K.at<float>(1, 0), 2));
	s = K.at<float>(1, 0) / sqrt(pow(K.at<float>(1, 1), 2) + pow(K.at<float>(1, 0), 2));

	Qz.at<float>(0, 0) = c;
	Qz.at<float>(0, 1) = s;
	Qz.at<float>(1, 0) = (-1)*s;
	Qz.at<float>(1, 1) = c;

	K = K*Qz;

	R = Qx*Qy*Qz;
	R = R.t();
}

/* Función para obtener P a partir de K, R y T */
Mat obtenerP(Mat K, Mat R, Mat T){
	Mat resultado = (Mat_<float>(3, 4));
	for (int i = 0; i < resultado.rows; i++){
		for (int j = 0; j < resultado.cols; j++){
			(j == 3) ? resultado.at<float>(i, j) = T.at<float>(i, 0) : resultado.at<float>(i, j) = R.at<float>(i, j);
		}
	}
	return K*resultado;
}

/* Función para aplicar el algoritmo DLT a la matriz P */
Mat aplicarDLT(Mat P){
	Mat M(3, 3, CV_32F);
	Mat m(3, 1, CV_32F);

	M = P.colRange(0, 3);
	m = P.colRange(3, 4);

	Mat K, R;
	//Vamos a obtener a partir de M que es 3x3, las matrices K y R
	calcularRotaciones(M,K,R);

	//Vamos a obtener la matriz T
	Mat T = K.inv()*m;

	//Por último obtendremos la P solución de SVD
	Mat res = obtenerP(K, R, T);
	return res;
}

/* Función para calcular el error a partir de la P generada de forma aleatoria, y la P obtenida por DLT */
float calcularError(Mat P, Mat PP){
	float error = 0.0;

	for (int i = 0; i < P.rows; i++){
		for (int j = 0; j < P.cols; j++){
			error += pow(P.at<float>(i, j) - PP.at<float>(i, j), 2);
		}
	}
	return error;
}

void Ejercicio1Trabajo3(){
	//1.a
	Mat P;
	GenerarMatrizP(P);
	//1.b
	vector<Point3d> puntos_mundo_3D;
	puntos_mundo_3D = GenerarPatron3D();
	//1.c
	vector<Mat> MatricesDePuntos = generarMatricesDePuntos(puntos_mundo_3D);
	vector<Mat> PuntosProyectados = generarPuntosProyectados(P , MatricesDePuntos);
	vector<Mat> CoordenadasPixel = generarCoordenadasPixel(PuntosProyectados);
	//1.d
	Mat PP;
	PP = aplicarDLT(P);
	//1.e
	float error = calcularError(P, PP);
	cout << " P (aleatoria)= " << endl << P << endl << endl;
	cout << " P (algoritmo DLT)= " << endl << PP << endl << endl;
	cout << "El error cuadratico de Frobenius: " << error << endl;


}


/*****************************************************************************************/
/*********************************** Ejercicio 2 *****************************************/
/*****************************************************************************************/

/* Función para determinar si una imagen es válida para calibrar una cámara */
void Ejer2ApartA(Mat image, bool &devol, vector<Mat> &sal, vector<Point2f> &corners, int numCornersHor, int numCornersVer){
	Size board_sz(numCornersHor, numCornersVer);
	Mat gray_image;
	//Transformamos la imagen
	cvtColor(image, gray_image, CV_BGR2GRAY);
	//Buscamos con ayuda de OpenCV
	bool found = findChessboardCorners(gray_image, board_sz, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	if (found) {
		devol = true;
		//Obtenemos los corners
		cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		//Dibujamos sobre la imagen
		drawChessboardCorners(gray_image, board_sz, corners, found);
		sal.push_back(gray_image);
	}
}

/* Función para calcular los valores de los parámetros de la cámara */
void Ejer2ApartB(Mat image, vector<Mat> &sal, vector<Point2f> corners, int numCornersHor, int numCornersVer){
	vector <vector<Point3f> > object_points;
	vector <vector<Point2f> > image_points;
	vector<Point3f> obj;
	int numSquares = numCornersHor * numCornersVer;
	for (int j = 0; j<numSquares; j++)
		obj.push_back(Point3f(j / 13, j % 13, 0.0f));
	image_points.push_back(corners);
	object_points.push_back(obj);

	Mat intrinsic = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;

	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;
	calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
	cout << endl << "distCoeffs con distorsion = " << distCoeffs << endl << endl;

	calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K1 or CV_CALIB_FIX_K2 or CV_CALIB_FIX_K3 or CV_CALIB_FIX_K4 or CV_CALIB_FIX_K5 or CV_CALIB_FIX_K6 or CV_CALIB_ZERO_TANGENT_DIST);
	cout << endl << "distCoeffs sin distorsion= " << distCoeffs << endl << endl;

	sal.push_back(image);
}

void Ejercicio2Trabajo3(){
	//El primer paso va a ser leer las imágenes del chessboard
	vector<Mat> chessboard;
	for (int i = 1; i <= 25; i++){
		string Result;
		stringstream convert;
		convert << i;
		Result = convert.str();
		string im = "imagenes/Image" + Result + ".tif";
		chessboard.push_back(leeimagen(im, 1));
	}

	//2.a
	for (uint i = 0; i < chessboard.size(); i++){
		bool devol = false;
		vector<Mat> sal;
		vector<Point2f> corners;
		int numCornersHor = 13;
		int numCornersVer = 12;
		Ejer2ApartA(chessboard[i], devol, sal, corners, numCornersHor, numCornersVer);
		if (devol){
			cout << "La imagen numero " << i + 1 << " es valida para calibrar una camara." << endl;
			pintaI(sal[0]);
		}
		//2.b
		if (devol){
			Ejer2ApartB(chessboard[i], sal, corners, numCornersHor, numCornersVer);
			pintaI(sal[1]);
		}
	}
}

/*****************************************************************************************/
/*********************************** Ejercicio 3 *****************************************/
/*****************************************************************************************/

/* Función para sacar las correspondencias sobre las imágenes usando BRISK/ORB */
Mat Ejer3ApartA(Mat img1, Mat img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches){
	Mat descriptor1, descriptor2;

	//Para este apartado, me voy a basar en el descriptor BRISK
	Mat kaze1, kaze2;
	akazeFuncion(img1, keypoints1, kaze1, descriptor1);
	akazeFuncion(img2, keypoints2, kaze2, descriptor2);

	//Como en la práctica anterior, sacamos los puntos en correspondencias
	Mat img_salida;
	bool crossCheck = 1;
	PuntosEnComun(img1, img2, descriptor1, descriptor2, keypoints1, keypoints2, crossCheck, img_salida, matches);
	return img_salida;
}

/* Función para convertir los keypoints en puntos 2f */
void convertirKeypoints(vector<KeyPoint> k1, vector<KeyPoint> k2, vector<DMatch> mb, vector<Point2f> &p1, vector<Point2f> &p2){
	for (uint i = 0; i < mb.size(); i++){
		p1.push_back(k1[mb[i].queryIdx].pt);
		p2.push_back(k2[mb[i].queryIdx].pt);
	}
}

/* Función para dibujar los puntos epipolares en las imágenes */
void dibujarPuntosEpipolares(Mat &img, vector<Point2f> p){
	RNG rng(12345);
	Scalar color;
	vector<Point2f>::const_iterator it = p.begin();
	while (it != p.end()){
		color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		circle(img, *it, 3, color, 2);
		++it;
	}
	pintaI(img, "Con puntos epipolares");
}

/* Función para calcular F */
Mat Ejer3ApartB(Mat img1, Mat img2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> matches, vector<Point2f> &points1, vector<Point2f> &points2){
	vector<Mat> imags;
	Mat F = Mat(3, 3, CV_32F);
	//Primero convertimos los keypoints obtenidos en el apartado anterior en puntos 2f
	convertirKeypoints(keypoints1, keypoints2, matches, points1, points2);

	//Dibujamos los puntos epipolares en las imágenes
	dibujarPuntosEpipolares(img1, points1);
	dibujarPuntosEpipolares(img2, points2);

	//Con la función que proporciona OpenCV buscamos la matriz fundamental aplicando RANSAC
	F = findFundamentalMat(Mat(points1), Mat(points2), CV_FM_RANSAC, 3, 0.999);
	cout << " F = " << F << endl << endl;
	return F;
}

/* Función para dibujar las lineas epipolares sobre las imágenes */
Mat dibujarLineaEpipolar(Mat img, vector<Vec3f> lines){
	RNG rng(200);
	Scalar color;
	Mat sal = img;
	uint condfin = 200;
	for (uint i=0; i<condfin; i++){
		color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		line(sal, Point(0, -lines[i][2] / lines[i][1]), Point(img.cols, -(lines[i][2] + lines[i][0] * img.cols) / lines[i][1]), color);
	}
	return sal;
}

/* Función para pintar las líneas epipolares en las imágenes */
void Ejer3ApartC(Mat F, Mat img1, vector<Point2f> points1, vector<Vec3f> &lines1, Mat img2, vector<Point2f> points2, vector<Vec3f> &lines2, vector<Mat> &sal){
	Mat aux;
	computeCorrespondEpilines(Mat(points1), 1, F, lines1);
	aux = dibujarLineaEpipolar(img2, lines1);
	sal.push_back(aux);
	computeCorrespondEpilines(Mat(points2), 2, F, lines2);
	aux = dibujarLineaEpipolar(img1, lines2);
	sal.push_back(aux);
}

/* Función para calcular la media de la distancia ortogonal */
float mediaDistanciaLineasYPuntos(vector<Point2f> p, vector<Vec3f> l){
	float media = 0.0;
	for (uint i = 0; i < p.size(); i++){
		media += abs((l[i](0)*p[i].x + l[i](1)*p[i].y + l[i](2)) / sqrt(l[i](0)*l[i](0) + l[i](1)*l[i](1)));
	}
	media /= p.size();
	return media;
}

/* Función para calcular el error medio */
float Ejer3ApartD(vector<Point2f> points1, vector<Vec3f> lines1, vector<Point2f> points2, vector<Vec3f> lines2){
	float media=0.0;
	media += mediaDistanciaLineasYPuntos(points1, lines2);
	media += mediaDistanciaLineasYPuntos(points2, lines1);
	return media;
}

void Ejercicio3Trabajo3(){
	//Leemos las imágenes
	Mat img1 = leeimagen("imagenes/Vmort1.pgm", 1);
	Mat img2 = leeimagen("imagenes/Vmort2.pgm", 1);

	//3.a
	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	Mat salida;
	salida = Ejer3ApartA(img1,img2,keypoints1,keypoints2, matches);
	pintaI(salida, "Apartado 3A");

	//3.b
	vector<Point2f> points1, points2;
	Mat F = Mat(3, 3, CV_32F);
	F = Ejer3ApartB(img1, img2, keypoints1, keypoints2, matches, points1, points2);

	//3.c
	vector<Vec3f> lines1, lines2;
	vector<Mat> sal;
	Ejer3ApartC(F, img1, points1, lines1, img2, points2, lines2, sal);
	pintaImagenes(sal,"Apartado 3C");

	//3.d
	float mediadistancia;
	mediadistancia = Ejer3ApartD(points1, lines1, points2, lines2);
	cout << "La media distancia ortogonal es: " << mediadistancia << endl;
}

/*****************************************************************************************/
/*********************************** Ejercicio 4 *****************************************/
/*****************************************************************************************/

/* Función para leer los archivos .ppm.camera */
void LeerCamera(string archivo, Mat &K, Mat &R, Mat &t){
	K = Mat(3, 3, CV_32F);
	R = Mat(3, 3, CV_32F);
	t = Mat(3, 1, CV_32F);
	char real[256];
	int cont = 0, i = 0, j = 0;
	ifstream in(archivo.c_str());
	if (!in)
		cout << "\nError: Fallo al abrir el fichero " << archivo << endl;
	else {
		do {
			in >> real;
			float num = atof(real);
			if (cont < 9){
				K.at<float>(i, j) = num;
				j++;
				if (j % 3 == 0){
					j = 0;
					i++;
				}
			}
			else if (cont >= 12 && cont < 21){
				R.at<float>(i, j) = num;
				j++;
				if (j % 3 == 0){
					j = 0;
					i++;
				}
			}
			else if (cont >= 12 && cont < 24){
				t.at<float>(i, j) = num;
				i++;
			}
			cont++;
			if (cont == 12 || cont == 21)
				i = 0;
		} while (!in.eof() || cont == 24);
		in.close();
	}
}



/* Función para calcular las parejas de puntos en correspondencias */
void Ejer4ApartB(vector<Mat> &imgs, string archivo1, string archivo2, Mat &K1, Mat &K2, Mat &R1, Mat &R2, Mat &t1, Mat &t2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches, vector<Point2f> &points1, vector<Point2f> &points2){
	LeerCamera(archivo1, K1, R1, t1);
	LeerCamera(archivo2, K2, R2, t2);
	Mat salida = Ejer3ApartA(imgs[0], imgs[1], keypoints1, keypoints2, matches);
	convertirKeypoints(keypoints1, keypoints2, matches, points1, points2);
}


void Ejercicio4Trabajo3(){
	Mat K1, P1, P2;
	vector<Point2f> points1, points2;

	//4.a
	Mat rdimg0 = leeimagen("imagenes/rdimage.000.ppm", 1);
	Mat rdimg1 = leeimagen("imagenes/rdimage.001.ppm", 1);
	Mat rdimg4 = leeimagen("imagenes/rdimage.004.ppm", 1);
	string archivo1 = "imagenes/rdimage.000.ppm.camera";
	string archivo2	 = "imagenes/rdimage.001.ppm.camera";
	vector<Mat> imgs;
	imgs.push_back(rdimg0);
	imgs.push_back(rdimg1);

	//4.b
	Mat K2, R1, R2, t1, t2;
	vector<KeyPoint> keypoints1, keypoints2;
	vector<DMatch> matches;
	Ejer4ApartB(imgs,archivo1, archivo2, K1, K2, R1, R2, t1, t2, keypoints1, keypoints2, matches, points1, points2);

	//4.c
	Mat F = Mat(3, 3, CV_32F);
	F = findFundamentalMat(Mat(points1), Mat(points2), CV_FM_RANSAC, 1, 0.99);
	F.convertTo(F, CV_32F);
	Mat E = K2.t()*F*K1;
	Mat R_E, T_E;
	cout << "Matriz Esencial " << endl;
	cout << "E = " << endl << " " << E << endl << endl;
	cout << "R_E = " << endl << " " << R_E << endl << endl;
	cout << "T_E = " << endl << " " << T_E << endl << endl;
}


int main(int argc, char* argv[]) {
	cout << endl;

	cout << "EJERCICIO 1" << endl;
	Ejercicio1Trabajo3();

	cout << endl << "EJERCICIO 2" << endl;
	Ejercicio2Trabajo3();

	cout << endl << "EJERCICIO 3" << endl;
	Ejercicio3Trabajo3();

	cout << endl << "EJERCICIO 4" << endl;
	Ejercicio4Trabajo3();


}






