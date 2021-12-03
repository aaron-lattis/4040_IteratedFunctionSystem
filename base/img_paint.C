//------------------------------------------------
//
//  img_paint
//
//
//-------------------------------------------------



#include <cmath>
#include <omp.h>
#include "imgproc.h"
#include "CmdLineFind.h"
#include <vector>



#include <GL/gl.h>   // OpenGL itself.
#include <GL/glu.h>  // GLU support library.
#include <GL/glut.h> // GLUT support library.


#include <iostream>
#include <stack>


using namespace std;
using namespace img;


ImgProc image;


void setNbCores( int nb )
{
   omp_set_num_threads( nb );
}

void cbMotion( int x, int y )
{
}

void cbMouse( int button, int state, int x, int y )
{
}

void cbDisplay( void )
{
   glClear(GL_COLOR_BUFFER_BIT );
   glDrawPixels( image.nx(), image.ny(), GL_RGB, GL_FLOAT, image.raw() );
   glutSwapBuffers();
}

void cbIdle()
{
   glutPostRedisplay();	
}

void cbOnKeyboard( unsigned char key, int x, int y )
{
   switch (key) 
   {
      case 'c':
	 image.compliment();
	 cout << "Compliment\n";
	 break;
   }
}

void PrintUsage()
{
   cout << "\n\nIFS is currently running.\n"; 
   cout << "The generated image will be shown when 10,000,000 iterations are complete.\n\n";
}


int main(int argc, char** argv)
{


   lux::CmdLineFind clf( argc, argv );

   setNbCores(8);

   //keeping these two commented out lines (86 and 92) so it is easy to switch to reading in an image
   //string imagename = clf.find("-image", "", "Image to drive color");
   
   clf.usage("-h");
   clf.printFinds();
   PrintUsage();

   //image.load(imagename);

   image.clear(1920, 1080, 3);

   image.iteratedFunctionSystem();

   // GLUT routines
   glutInit(&argc, argv);

   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
   glutInitWindowSize( image.nx(), image.ny() );

   // Open a window 
   char title[] = "img_paint";
   glutCreateWindow( title );
   
   glClearColor( 1,1,1,1 );

   glutDisplayFunc(&cbDisplay);
   glutIdleFunc(&cbIdle);
   glutKeyboardFunc(&cbOnKeyboard);
   glutMouseFunc( &cbMouse );
   glutMotionFunc( &cbMotion );

   glutMainLoop();
   return 1;
};
