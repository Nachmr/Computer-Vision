 /*
 * main.cpp
 *
 *  Created on: 23/09/2016
 *      Author: Ignacio Martín Requena
 */

#include <iostream>
#include <opencv2/opencv.hpp>

#define SIGMAGATOBAJAS 50
#define SIGMAGATOALTAS 20

#define SIGMAPERROBAJAS 40
#define SIGMAPERROALTAS 12

#define SIGMAEINSTEINBAJAS 25
#define SIGMAEINSTEINALTAS 10

#define SIGMAMARILYNBAJAS 30
#define SIGMAMARILYNALTAS 10

#define SIGMAPEZBAJAS 45
#define SIGMAPEZALTAS 14

#define SIGMASUBMARINOBAJAS 40
#define SIGMASUBMARINOALTAS 25

#define SIGMAPAJAROBAJAS 22
#define SIGMAPAJAROALTAS 15

#define SIGMAAVIONBAJAS 22
#define SIGMAAVIONALTAS 10

#define SIGMABICICLETABAJAS 50
#define SIGMABICICLETAALTAS 10

#define SIGMAMOTOBAJAS 50
#define SIGMAMOTOALTAS 19



using namespace std;
using namespace cv;

/* Función que calcula la  convolucion de una imagen a partir de una mascara */
void my_imGaussConvol(Mat im, Mat& maskConvol, Mat& out){
	flip(maskConvol, maskConvol, -1); //No seria necesario ya que la mascara al ser gaussiana es simétrica

	Mat imaux(im);
	filter2D(im, imaux, im.depth(), maskConvol, Point(-1,-1), 0 ,BORDER_DEFAULT);
	transpose(imaux,imaux);
	filter2D(imaux, out, im.depth(), maskConvol, Point(-1,-1), 0 ,BORDER_DEFAULT);
	transpose(out,out);

}

/* Función para crear una pirámide Gaussiana */
void PiramideGaussiana(Mat &imagen, Mat &pyr) {
	Mat aux1 = imagen.clone();
	Mat aux2 = imagen.clone();
	vector<Mat> img_reducidas; //vector para almacenar la reducción de cada imagen
	img_reducidas.push_back(aux1);
	int niveles = 5;


	for(int i=0; i < niveles; i++){
		pyrDown(img_reducidas[i], aux2, Size(aux2.cols / 2, aux2.rows / 2));
		img_reducidas.push_back(aux2);
	}

	Mat img_negra, nivel_completo;
	vector<Mat> reducidas_con_negro;
	for(uint i=2; i<img_reducidas.size(); i++){ //añado los trozos en negro para que los escalones tengan la misma anchura que la imagen de primer nivel
		Mat img_negra(img_reducidas[i].rows, img_reducidas[1].cols - img_reducidas[i].cols, img_reducidas[i-1].type());
		img_negra = Scalar(0);
		hconcat(img_reducidas[i], img_negra, nivel_completo);
		reducidas_con_negro.push_back(nivel_completo);
	}


	Mat piramide = reducidas_con_negro.back();
	reducidas_con_negro.pop_back();

	//concateno todos los trozos de imagen con negro
	while(!reducidas_con_negro.empty()){
		Mat ultimo = reducidas_con_negro.back();
		reducidas_con_negro.pop_back();
		vconcat(ultimo, piramide, piramide);
	}

	vconcat(img_reducidas[1], piramide, piramide); //añado la del nivel uno ya que no hace falta ajustar sus dimensiones

	//Reajuste de las dimensiones de la piramide para concatenarla con la original
	Mat nivel_entero_negr(img_reducidas[0].rows-piramide.rows, img_reducidas[1].cols, img_reducidas[1].type());
	nivel_entero_negr = Scalar(0);
	vconcat(piramide, nivel_entero_negr,piramide);
	hconcat(img_reducidas[0], piramide, piramide);

	pyr = piramide.clone();
}

/*Calculo de las frecuencias altas*/
void calculo_freq_altas(Mat &im, Mat &freq_bajas, Mat &out){
	out = (im/2 - freq_bajas/2)+127;

}

/*Ajuste de las frecuencias de la imagen hibrida*/
void calculo_img_hibrida(Mat &altas, Mat &bajas, Mat & hibrida){
	hibrida = (altas + bajas)/2;
}


void EjercicioA(Mat & im, Mat &mask_bajas){
	Mat freq_bajas = im.clone();
	for(int i=1; i<100; i+=20){
			mask_bajas = getGaussianKernel(i,6*i + 1,CV_32F); //kernel para imagen de frecuencias bajas
			my_imGaussConvol(im, mask_bajas, freq_bajas); //calculo frecuencias bajas para imagen de frecuencias altas
			cout << "Sigma: " << i << endl << "Mascara usada: " << mask_bajas << endl << endl;
			imshow("Imagen Ejercicio 1", freq_bajas);
			waitKey(0);
		}
		destroyWindow("Imagen Ejercicio 1");
}

void EjercicioB(int sigma_bajas,int sigma_altas, Mat &im ,Mat &im2, Mat &hibrida){
	Mat mask_bajas = im.clone();
	Mat freq_bajas = im.clone();
	Mat mask_altas = im2.clone();
	Mat freq_altas = im2.clone();


	//Calculo frecuencias baja
	mask_bajas = getGaussianKernel(sigma_bajas,6*sigma_bajas + 1,CV_32F); //kernel para imagen de frecuencias bajas
	my_imGaussConvol(im, mask_bajas, freq_bajas); //calculo frecuencias bajas para imagen de frecuencias altas

	//Calculo frecuencias altas
	mask_altas = getGaussianKernel(sigma_altas,6*sigma_altas + 1,CV_32F); //kernel para imagen de frecuencias altas
	my_imGaussConvol(im2, mask_altas, freq_altas); //calculo de frecuencias bajas para calcular imagen de frecuencias altas
	calculo_freq_altas(im2, freq_altas, freq_altas); //primero se divide entre dos para que el rango de todas este entre 0 y 256. Se le suma el 127 para que el valor final sea positivo

	imshow("Imagen alta", freq_altas);
	waitKey(0);

	//Calculo imagen hibrida
	calculo_img_hibrida(freq_bajas, freq_altas, hibrida);
	imshow("Imagen Hibrida",hibrida);
	waitKey(0);
	destroyWindow("Imagen Hibrida");


	//Alta, baja e híbrida en una misma ventana
	Mat H1, H2;
	hconcat(freq_bajas, freq_altas, H1);//para mostrar varias imagenes a la vez
	hconcat(H1, hibrida , H2);
	imshow("Imagen Concatenada", H2);
	waitKey(0);
	destroyWindow("Imagen Concatenada");
}

int main(int argc, char** argv)
{
	string imageName_freq_bajas, imageName_freq_altas;


	int sigma_bajas, sigma_altas;

	imageName_freq_bajas = "imagenes/bicycle.bmp";
	imageName_freq_altas = "imagenes/motorcycle.bmp";


	/************************ Asignacion Sigma a imagen de frecuencias bajas *************************/
	if(imageName_freq_bajas.find("bird") != string::npos)
		sigma_bajas = SIGMAPAJAROBAJAS;
	else if(imageName_freq_bajas.find("plane") != string::npos)
		sigma_bajas = SIGMAAVIONBAJAS;
	else if(imageName_freq_bajas.find("cat") != string::npos)
		sigma_bajas = SIGMAGATOBAJAS;
	else if(imageName_freq_bajas.find("dog") != string::npos)
		sigma_bajas = SIGMAPERROBAJAS;
	else if(imageName_freq_bajas.find("fish") != string::npos)
		sigma_bajas = SIGMAPEZBAJAS;
	else if(imageName_freq_bajas.find("submarine") != string::npos)
		sigma_bajas = SIGMASUBMARINOBAJAS;
	else if(imageName_freq_bajas.find("bicycle") != string::npos)
		sigma_bajas = SIGMABICICLETABAJAS;
	else if(imageName_freq_bajas.find("motorcycle") != string::npos)
		sigma_bajas = SIGMAMOTOBAJAS;
	else if(imageName_freq_bajas.find("einstein") != string::npos)
		sigma_bajas = SIGMAEINSTEINBAJAS;
	else if(imageName_freq_bajas.find("marilyn") != string::npos)
		sigma_bajas = SIGMAMARILYNBAJAS;

	/************************ Asignacion Sigma a imagen de frecuencias altas *************************/
	if(imageName_freq_altas.find("bird") != string::npos)
		sigma_altas = SIGMAPAJAROALTAS;
	else if(imageName_freq_altas.find("plane") != string::npos)
		sigma_altas = SIGMAAVIONALTAS;
	else if(imageName_freq_altas.find("cat") != string::npos)
		sigma_altas = SIGMAGATOALTAS;
	else if(imageName_freq_altas.find("dog") != string::npos)
		sigma_altas = SIGMAPERROALTAS;
	else if(imageName_freq_altas.find("fish") != string::npos)
		sigma_altas = SIGMAPEZALTAS;
	else if(imageName_freq_altas.find("submarine") != string::npos)
		sigma_altas = SIGMASUBMARINOALTAS;
	else if(imageName_freq_altas.find("bicycle") != string::npos)
		sigma_altas = SIGMABICICLETAALTAS;
	else if(imageName_freq_altas.find("motorcycle") != string::npos)
		sigma_altas = SIGMAMOTOALTAS;
	else if(imageName_freq_altas.find("einstein") != string::npos)
		sigma_altas = SIGMAEINSTEINALTAS;
	else if(imageName_freq_altas.find("marilyn") != string::npos)
		sigma_altas = SIGMAMARILYNALTAS;



	/************************ Lectura y declaracion de las imagenes usadas*************************/
	Mat im = imread(imageName_freq_bajas.c_str(),IMREAD_COLOR), freq_bajas_b = im, freq_altas_b = im, mask_bajas= im, mask_altas= im, freq_altas = im, hibrida;
	Mat im2 = imread(imageName_freq_altas.c_str(),IMREAD_COLOR);//usada para las frecuencias bajas
	if( im.empty() || im2.empty()){
		cout << "error de lectura"<< endl;
	}


	/*************************** Ejercicio A *******************************/
	EjercicioA(im, mask_bajas);

	//-------------------------------------------------------------------------

	/*************************** Ejercicio B *******************************/

	Mat im_aux = imread(imageName_freq_bajas.c_str(),IMREAD_COLOR);
	Mat im2_aux = imread(imageName_freq_altas.c_str(),IMREAD_COLOR);
	EjercicioB(sigma_bajas, sigma_altas, im_aux, im2_aux, hibrida);

	//-------------------------------------------------------------------------

	/*************************** Ejercicio C *******************************/
	Mat pyr;
	PiramideGaussiana(hibrida, pyr);
	imshow("Piramide", pyr);

	//-------------------------------------------------------------------------


	waitKey(0);


	return 1;
}





