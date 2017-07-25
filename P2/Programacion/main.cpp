 /*
 * main.cpp
 *
 *  Created on: 7/11/2016
 *      Author: Ignacio Martín Requena
 */


#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


Mat leeimagen(string filename, int flagColor) {
	Mat res = imread(filename, flagColor);
	return res;
}

void pintaI(Mat im, string name = "Ventana") {
	namedWindow(name);
	imshow(name, im);
	cvWaitKey();
	destroyWindow(name);
}

void pintaIVEC(vector<Mat> vim, string windowname, int col = CV_8UC3) {
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


//Estructura para almacenar los valores de un pixel de una matriz
struct MatrixToPoint{
	float value;
	Point p;
};

struct Ordenacion{
    bool operator() (MatrixToPoint pt1, MatrixToPoint pt2) { return (pt1.value > pt2.value);}
} sortfunc;


///////////////////////////////////////////////////////
///////////////// EJERCICIO 1 /////////////////////////
///////////////////////////////////////////////////////

void Ejercicio1Trabajo2(Mat &im){
	cout << "\n\nEjecutando Ejercicio 1 Trabajo 2" << endl;

	//Variables para la gestion de las imágenes
	Mat im_aux= im.clone();

	Mat im_gray, dst, dst_norm, dst_nomax, imlvl2, imlvl3;
	cvtColor(im_aux, im_gray, cv::COLOR_BGR2GRAY);
	/// Parametros para la función cornerHarris
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	//variables para la gestion de los puntos Harris
	MatrixToPoint punto;
	vector<MatrixToPoint> vectorsort, vectorsort_aux;

	//Variables para la supresion de no-maximos

	//-------------------------------------------------------------------------
	//            Primer nivel de la Pirámide
	/// Detectar esquinas
	cornerHarris( im_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizar
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32F);

	//Ordenar los valores de la matriz normalizada de mayor a menor
	for( int j = 0; j < dst_norm.rows ; j++ ){
		for( int i = 0; i < dst_norm.cols; i++ ){
			punto.value = dst_norm.at<float>(j,i);
			punto.p.x = i;
			punto.p.y = j;

			vectorsort_aux.push_back(punto);
		  }
	}
	std::sort(vectorsort_aux.begin(), vectorsort_aux.end(), sortfunc);

	//Seleccionar los mejores valores
	for( int j = 0; j < 1500*0.7 ; j++ ){
		vectorsort.push_back(vectorsort_aux.at(j));
	}

	vectorsort_aux.clear();


	//-------------------------------------------------------------------------
	//                 Segundo nivel de la Pirámide
	pyrDown(im_gray, imlvl2, Size( im_aux.cols/2, im_aux.rows/2 ) );

	/// Detectar esquinas
	cornerHarris( imlvl2, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizar
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32F);

	//Ordenar los valores de la matriz normalizada de mayor a menor
	for( int j = 0; j < dst_norm.rows ; j++ ){
		for( int i = 0; i < dst_norm.cols; i++ ){
			punto.value = dst_norm.at<float>(j,i);
			punto.p.x = i*2;
			punto.p.y = j*2;

			vectorsort_aux.push_back(punto);
		  }
	}
	std::sort(vectorsort_aux.begin(), vectorsort_aux.end(), sortfunc);

	//Seleccionar los mejores valores
	for( int j = 0; j < 1500*0.2 ; j++ ){
		vectorsort.push_back(vectorsort_aux.at(j));
	}

	vectorsort_aux.clear();


	//-------------------------------------------------------------------------
	//             Tercer nivel de la Pirámide

	pyrDown( imlvl2, imlvl3, Size( imlvl2.cols/2, imlvl2.rows/2 ) );

	/// Detectar esquinas
	cornerHarris( imlvl3, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizar
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32F);

	//Ordenar los valores de la matriz normalizada de mayor a menor
	for( int j = 0; j < dst_norm.rows ; j++ ){
		for( int i = 0; i < dst_norm.cols; i++ ){
			punto.value = dst_norm.at<float>(j,i);
			punto.p.x = i*4;
			punto.p.y = j*4;

			vectorsort_aux.push_back(punto);
		  }
	}
	std::sort(vectorsort_aux.begin(), vectorsort_aux.end(), sortfunc);

	//Seleccionar los mejores valores
	for( int j = 0; j < 1500*0.1 ; j++ ){
		vectorsort.push_back(vectorsort_aux.at(j));
	}

	//Ordenación del vector final
	std::sort(vectorsort.begin(), vectorsort.end(), sortfunc);

	/// Dibujar circulos alrededor de las esquinas
	for(uint j = 0; j < vectorsort.size()-1; j++ ){
		 circle( im_aux, vectorsort.at(j).p, 5,  Scalar(0), 1, 4, 0 );
	}

	///Mostrar resultados
	pintaI(im_aux, "1Harris");

	cout << "Fin Ejecucion Ejercicio 1 Trabajo 2" << endl;
}

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

void PuntosEnComun(Mat im1, Mat im2, Mat descriptors1, Mat descriptors2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, bool crossCheck, Mat& salida, vector<DMatch>& matchesbuenos, int limite_correspondencias=15){
	vector<DMatch> matches;

	//Realizamos la búsqueda de las correspondencias
	BFMatcher matcher(NORM_L2, crossCheck);
	matcher.match(descriptors1, descriptors2, matches);

	//Ordenamos las correspondencias para buscar las mejores de forma más eficiente
	sort(matches.begin(), matches.end());
	float min_dist = matches[0].distance;

	//Buscamos las mejores correspondencias
	for (uint i = 0; i < matches.size(); i++){
		if (matches[i].distance <= limite_correspondencias * min_dist){
			matchesbuenos.push_back(matches[i]);
		}
		else if (matchesbuenos.size() < 4){ //en el caso de que no consiga un mínimo de 4 correspondencias
				//aumento el factor de correspondencia
				limite_correspondencias++;
				i--;
		}
	}
	cout << "He pasado de " << matches.size() << " correspondencias a " << matchesbuenos.size() << ". Valor limite_correspondencias: " << limite_correspondencias << endl;

	//Pintamos los puntos de correspondencia
	drawMatches(im1, keypoints1, im2, keypoints2, matchesbuenos, salida);
}

void Ejercicio2Trabajo2(Mat im1, Mat im2, int detector){
	cout << "\n\nEjecutando Ejercicio 2 Trabajo 2" << endl;

	//Creamos las variables tanto para los detectores, como para la lista de keypoints como para los descriptores
	Mat kaze1, kaze2, akaze1, akaze2, out;
	vector<KeyPoint> keypoints1, keypoints2, keypoints3, keypoints4;
	Mat descriptor1, descriptor2, descriptor3, descriptor4;


	//Factor de correspondencia, para hacer la selección de las correspondencias mejores
	int limite_correspondencias = 50;
	bool crossCheck = 1;

	if (detector == 0 || detector == 1){
		//Paso el detector KAZE
		kazeFuncion(im1, keypoints1, kaze1, descriptor1);
		kazeFuncion(im2, keypoints2, kaze2, descriptor2);
		//Muestro los resultados del detector kaze
		cout << "KAZE:Se han encontrado " << keypoints1.size() << " keypoints y " << keypoints2.size() << " keypoints" << endl;

		Mat img_salida;
		vector<DMatch> matchesbuenos;
		//Ahora llamaremos a nuestra función que calcula los puntos en comun con las imagenes, descriptores, keypoints y el flag si queremos crossCheck
		PuntosEnComun(im1, im2, descriptor1, descriptor2, keypoints1, keypoints2, crossCheck, img_salida, matchesbuenos, limite_correspondencias);
		pintaI(img_salida, "KAZE");


	}

	if (detector == 0 || detector == 2){
		//Paso el detector akaze a ambas imágenes
		akazeFuncion(im1, keypoints3, akaze1, descriptor3);
		akazeFuncion(im2, keypoints4, akaze2, descriptor4);
		//Pinto los resultados del detector akaze
		cout << "AKAZE:Se han encontrado " << keypoints3.size() << " keypoints y " << keypoints4.size() << " keypoints" << endl;
		Mat img_salida;
		vector<DMatch> matchesbuenos;
		//Ahora llamaremos a nuestra función que calcula los puntos en comun con las imagenes, descriptores, keypoints y el flag si queremos crossCheck
		PuntosEnComun(im1, im2, descriptor3, descriptor4, keypoints3, keypoints4, crossCheck, img_salida, matchesbuenos, limite_correspondencias);
		pintaI(img_salida, "AKAZE");
	}

	cout << "Fin Ejecucion Ejercicio 2 Trabajo 2" << endl;
}



///////////////////////////////////////////////////////
///////////////// EJERCICIO 3 /////////////////////////
///////////////////////////////////////////////////////

void crearMosaico(Mat& mosaico, Mat im1, int limite_correspondencias=15){
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2, empty;

	//Primero tenemos que sacar los keypoints, para ello uso kaze
	kazeFuncion(im1, keypoints1, empty, descriptors1);
	kazeFuncion(mosaico, keypoints2, empty, descriptors2);

	Mat img_sal;
	vector<DMatch> matchesbuenos;
	//Sacamos los puntos en correspondencias
	PuntosEnComun(im1, mosaico, descriptors1, descriptors2, keypoints1, keypoints2, 1, img_sal, matchesbuenos, limite_correspondencias);

	vector<Point2f> key1, key2;
	//Mostramos el numero de correspondencias encotradas y guardamos sus coordenadas
	cout << "Total de correspondencias para las dos imagenes: " << matchesbuenos.size() << endl;
	for (uint i = 0; i < matchesbuenos.size(); i++){
		key1.push_back(keypoints1[matchesbuenos[i].queryIdx].pt);
		key2.push_back(keypoints2[matchesbuenos[i].trainIdx].pt);
	}

	//Buscamos la homografía de las imágenes
	/*[R11,R12,T1]
	[R21,R22,T2]
	[ P , P , 1]*/
	Mat H = findHomography(key1, key2, CV_RANSAC);
	//Montamos las imagenes
	warpPerspective(im1, mosaico, H, mosaico.size(), CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS, BORDER_TRANSPARENT);
}

void InicializarMosaico(Mat img, Mat& mosaico, int mirows, int micols){
	//Creo el almacén del mosaico
	mosaico = Mat(mirows, micols, img.type());
	resize(mosaico, mosaico, Size(mirows, micols));

	//Coloco la imagen en el centro
	/*[R11,R12,T1]
	[R21,R22,T2]
	[ P , P , 1]*/
	Mat H0 = (Mat_<double>(3, 3) << 1, 0, (mosaico.cols / 2) - (img.cols), 0, 1, (mosaico.rows/2) - (img.rows ), 0, 0, 1);
	warpPerspective(img, mosaico, H0, Size(mirows, micols));
}

void crearSuperMosaico(vector<Mat> imagenes, Mat& mosaico_res, int mirows, int micols, int limite_correspondencias){
	cout << "En total hay " << imagenes.size() << " imagenes." << endl;

	Mat mosaico;
	int indice_comienzo = (imagenes.size()-1)/2;
	InicializarMosaico(imagenes[indice_comienzo],mosaico,mirows,micols);
	cout << "Imagen " << indice_comienzo+1 << " introducida." << endl;

	//Comienzo el mosaico del centro a la izquierda
	for (int i = indice_comienzo -1; i >= 0; i--){
		Mat im1 = imagenes[i];
		crearMosaico(mosaico, im1, limite_correspondencias);
		cout << "Imagen " << i+1 << " introducida." << endl;
	}
	//Sigo el mosaico del centro a la derecha
	for (uint i = indice_comienzo +1 ; i < imagenes.size(); i++){
		Mat im1 = imagenes[i];
		crearMosaico(mosaico, im1, limite_correspondencias);
		cout << "Imagen " << i+1 << " introducida." << endl;
	}
	//Guardo el resultado en lo que devolveré
	mosaico_res = mosaico;
}

void Ejercicio3Trabajo2(vector<Mat> imagenes, int rows_pantalla, int cols_pantalla, int limite_correspondencias){
	cout << "\nEjecutando Ejercicio 3 Trabajo 2" << endl;

	//Muestro las imágenes que van a participar en el mosaico
	if (imagenes.size() > 4){
		vector<Mat> parte1,parte2;
		for (uint i=0;i<imagenes.size()/2;i++)
			parte1.push_back(imagenes[i]);
		for (uint i=imagenes.size()/2;i<imagenes.size();i++)
			parte2.push_back(imagenes[i]);
		pintaIVEC(parte1, "3Primera parte imagenes del mosaico");
		pintaIVEC(parte2, "3Segunda parte imagenes del mosaico");
	}
	else
		pintaIVEC(imagenes, "3Imagenes del mosaico");

	//Comienzo la creación del mosaico
	Mat mosaico;
	crearSuperMosaico(imagenes, mosaico, rows_pantalla, cols_pantalla, limite_correspondencias);
	pintaI(mosaico,"3Mosaico");

	cout << "Fin Ejecucion Ejercicio 3 Trabajo 2" << endl;
}

int main(int argc, char** argv)
{

	//Lectura de todas las imágenes usadas en el trabajo 2
	Mat img_mosaico1 = leeimagen("imagenes/mosaico002.jpg", 1);
	Mat img_mosaico2 = leeimagen("imagenes/mosaico003.jpg", 1);
	Mat img_mosaico3 = leeimagen("imagenes/mosaico004.jpg", 1);
	Mat img_mosaico4 = leeimagen("imagenes/mosaico005.jpg", 1);
	Mat img_mosaico5 = leeimagen("imagenes/mosaico006.jpg", 1);
	Mat img_mosaico6 = leeimagen("imagenes/mosaico007.jpg", 1);
	Mat img_mosaico7 = leeimagen("imagenes/mosaico008.jpg", 1);
	Mat img_mosaico8 = leeimagen("imagenes/mosaico009.jpg", 1);
	Mat img_mosaico9 = leeimagen("imagenes/mosaico010.jpg", 1);
	Mat img_mosaico10 = leeimagen("imagenes/mosaico011.jpg", 1);
	vector<Mat> imagenes_mosaico;
	imagenes_mosaico.push_back(img_mosaico1);
	imagenes_mosaico.push_back(img_mosaico2);
	imagenes_mosaico.push_back(img_mosaico3);
	imagenes_mosaico.push_back(img_mosaico4);
	imagenes_mosaico.push_back(img_mosaico5);
	imagenes_mosaico.push_back(img_mosaico6);
	imagenes_mosaico.push_back(img_mosaico7);
	imagenes_mosaico.push_back(img_mosaico8);
	imagenes_mosaico.push_back(img_mosaico9);
	imagenes_mosaico.push_back(img_mosaico10);

	Mat img_yosemite1 = leeimagen("imagenes/yosemite1.jpg", 1);
	Mat img_yosemite2 = leeimagen("imagenes/yosemite2.jpg", 1);
	Mat img_yosemite3 = leeimagen("imagenes/yosemite3.jpg", 1);
	Mat img_yosemite4 = leeimagen("imagenes/yosemite4.jpg", 1);
	vector<Mat> imagenes_yosemite;
	imagenes_yosemite.push_back(img_yosemite1);
	imagenes_yosemite.push_back(img_yosemite2);
	imagenes_yosemite.push_back(img_yosemite3);
	imagenes_yosemite.push_back(img_yosemite4);


	/*************************************************/
	/********************* EJERCICIO 1 ***************/
	/*************************************************/


	Ejercicio1Trabajo2(imagenes_yosemite[0]);


	//-------------------------------------------------------------------------

	/*************************************************/
	/********************* EJERCICIO 2 ***************/
	/*************************************************/

	//0 ambos, 1 kaze, 2 akaze
	int detector = 0;
	Ejercicio2Trabajo2(imagenes_yosemite[0], imagenes_yosemite[1], detector);


	//-------------------------------------------------------------------------

	/*************************************************/
	/********************* EJERCICIO 3 ***************/
	/*************************************************/

	int rows_pantalla = 1800, cols_pantalla = 950;
	int limite_correspondencias = 1;
	Ejercicio3Trabajo2(imagenes_mosaico,rows_pantalla,cols_pantalla,limite_correspondencias);
	Ejercicio3Trabajo2(imagenes_yosemite,rows_pantalla,cols_pantalla,limite_correspondencias);


	//-------------------------------------------------------------------------


	waitKey(0);


	return 1;
}





