
#include <cmath>
#include "imgproc.h"

#include <iostream>

#include <OpenImageIO/imageio.h>
OIIO_NAMESPACE_USING

using namespace img;



ImgProc::ImgProc() :
  Nx (0),
  Ny (0),
  Nc (0),
  Nsize (0),
  img_data (nullptr)
{}

ImgProc::~ImgProc()
{
   clear();
}

   
void ImgProc::clear()
{
   if( img_data != nullptr ){ delete[] img_data; img_data = nullptr;}
   Nx = 0;
   Ny = 0;
   Nc = 0;
   Nsize = 0;
}


void ImgProc::clear(int nX, int nY, int nC)
{
   clear();
   Nx = nX;
   Ny = nY;
   Nc = nC;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
#pragma omp parallel for
   for(long i=0;i<Nsize;i++){ img_data[i] = 0.0; }
}


bool ImgProc::load( const std::string& filename )
{
   auto in = ImageInput::create (filename);
   if (!in) {return false;}
   ImageSpec spec;
   in->open (filename, spec);
   clear();
   Nx = spec.width;
   Ny = spec.height;
   Nc = spec.nchannels;
   Nsize = (long)Nx * (long)Ny * (long)Nc;
   img_data = new float[Nsize];
   in->read_image(TypeDesc::FLOAT, img_data);
   in->close ();
   return true;
}


void ImgProc::value( int i, int j, std::vector<float>& pixel) const
{
   pixel.clear();
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   pixel.resize(Nc);
   for( int c=0;c<Nc;c++ )
   {
      pixel[c] = img_data[index(i,j,c)];
   }
   return;
}


void ImgProc::set_value( int i, int j, const std::vector<float>& pixel)
{
   if( img_data == nullptr ){ return; }
   if( i<0 || i>=Nx ){ return; }
   if( j<0 || j>=Ny ){ return; }
   if( Nc > (int)pixel.size() ){ return; }
#pragma omp parallel for
   for( int c=0;c<Nc;c++ )
   {
      img_data[index(i,j,c)] = pixel[c];
   }
   return;
}


ImgProc::ImgProc(const ImgProc& v) :
  Nx (v.Nx),
  Ny (v.Ny),
  Nc (v.Nc),
  Nsize (v.Nsize)
{
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
}


ImgProc& ImgProc::operator=(const ImgProc& v)
{
   if( this == &v ){ return *this; }
   if( Nx != v.Nx || Ny != v.Ny || Nc != v.Nc )
   {
      clear();
      Nx = v.Nx;
      Ny = v.Ny;
      Nc = v.Nc;
      Nsize = v.Nsize;
   }
   img_data = new float[Nsize];
#pragma omp parallel for
   for( long i=0;i<Nsize;i++){ img_data[i] = v.img_data[i]; }
   return *this;
}


void ImgProc::operator*=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] *= v; }
}


void ImgProc::operator/=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] /= v; }
}


void ImgProc::operator+=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] += v; }
}


void ImgProc::operator-=(float v)
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] -= v; }
}


void ImgProc::compliment()
{
   if( img_data == nullptr ){ return; }
#pragma omp parallel for
   for( long i=0;i<Nsize;i++ ){ img_data[i] = 1.0 - img_data[i]; }
}


void ImgProc::iteratedFunctionSystem()
{
   if( img_data == nullptr ){ return; }

   Point point; //This is a struct with double variables X and Y. Point is defined at the top of imgproc.h
   
   point.X = 2 * drand48() - 1; 
   point.Y = 2 * drand48() - 1;

   int alpha [Nx][Ny] = {0}; //This array will be used to track the number of times each point has been iterated to

   float functionWeight [4] = {0.258754, 0.32167, 0.823765, 0.538494};
   float weight = drand48(); 

   int i;
   int j;

   int chooseFunction;

   double lookUpTableColor [3];

   std::vector <float> colorAtPoint;

   for (int cntr = 0; cntr < 10000000; cntr++) //the IFS will iterate 10,000,000 times
   {
      chooseFunction = (int)(drand48() * 4);
	
      functions(point, chooseFunction); //This will apply the selected function to point 

      weight = (weight + functionWeight [chooseFunction]) / 2; 
		
      if (cntr > 20)
      {
         lookUpTable(weight, lookUpTableColor); //chooses a color based on the weight, puts it in lookUpTableColor

         i = (int)((point.X + 1) / 2.0 * Nx); 
         j = (int)((point.Y + 1) / 2.0 * Ny); 

         if (i >= 0 && i < Nx && j >= 0 && j < Ny)
         {
            value(i, j, colorAtPoint); //gets the current color at the point that was iterated to, puts it in colorAtPoint

            colorAtPoint [0] *= alpha [i][j];
            colorAtPoint [1] *= alpha [i][j]; 
            colorAtPoint [2] *= alpha [i][j];

            colorAtPoint [0] += lookUpTableColor [0];
            colorAtPoint [1] += lookUpTableColor [1];
            colorAtPoint [2] += lookUpTableColor [2];

            alpha [i][j] += 1;

            colorAtPoint [0] /= alpha [i][j]; 
            colorAtPoint [1] /= alpha [i][j];
            colorAtPoint [2] /= alpha [i][j];

            set_value(i, j, colorAtPoint);  
         }
      }
   }
}


void ImgProc::functions(Point &p, int function)
{
   if (function == 0) //Variation 8 - Disc
   {
      double theta = atan(p.X/p.Y);
      double r = sqrt(pow(p.X, 2) + pow(p.Y, 2));

      p.X = (theta / M_PI) * sin(M_PI * r);
      p.Y = (theta / M_PI) * cos(M_PI * r);
   }
   else if (function == 1) //Variation 27 - Eyefish
   {
      double r = sqrt(pow(p.X, 2) + pow(p.Y, 2));

      p.X = (2 / (r + 1)) * p.X;
      p.Y = (2 / (r + 1)) * p.Y;
   }
   else if (function == 2) //Variation 28 - Bubble
   {
      double r = sqrt(pow(p.X, 2) + pow(p.Y, 2));

      p.X = (4 / (pow(r, 2) + 4)) * p.X;
      p.Y = (4 / (pow(r, 2) + 4)) * p.Y;
   }
   else //Variation 20 - Cosine
   {
      double sinPiX = sin(M_PI * p.X); //calculate this value now because p.X will be changed

      p.X = cos(M_PI * p.X) * cosh(p.Y);
      p.Y = sinPiX * sinh(p.Y);
   }
}


void ImgProc::lookUpTable(float W, double color[3])
{
   if (W < 0 || W > 1) { color[0] = 0; color[1] = 0; color[2] = 0; }

   else if (W <= .1) { color[0] = 45.0/255.0; color[1] = 223.0/255.0; color[2] = 255.0/255.0; }

   else if (W <= .2) { color[0] = 245.0/255.0; color[1] = 244.0/255.0; color[2] = 116.0/255.0; }

   else if (W <= .3) { color[0] = 227.0/255.0; color[1] = 60.0/255.0; color[2] = 199.0/255.0; }

   else if (W <= .4) { color[0] = 255.0/255.0; color[1] = 170.0/255.0; color[2] = 71.0/255.0; }

   else if (W <= .5) { color[0] = 245.0/255.0; color[1] = 77.0/255.0; color[2] = 40.0/255.0; }

   else if (W <= .6) { color[0] = 255.0/255.0; color[1] = 200.0/255.0; color[2] = 207.0/255.0; }

   else if (W <= .7) { color[0] = 245.0/255.0; color[1] = 223.0/255.0; color[2] = 255.0/255.0; }

   else if (W <= .8) { color[0] = 246.0/255.0; color[1] = 243.0/255.0; color[2] = 175.0/255.0; }

   else if (W <= .9) { color[0] = 201.0/255.0; color[1] = 233.0/255.0; color[2] = 139.0/255.0; }

   else { color[0] = 174.0/255.0; color[1] = 224.0/255.0; color[2] = 243.0/255.0; }
}


long ImgProc::index(int i, int j, int c) const
{
   return (long) c + (long) Nc * index(i,j); // interleaved channels

   // return index(i,j) + (long)Nx * (long)Ny * (long)c; // sequential channels
}

long ImgProc::index(int i, int j) const
{
   return (long) i + (long)Nx * (long)j;
}

void img::swap(ImgProc& u, ImgProc& v)
{
   float* temp = v.img_data;
   int Nx = v.Nx;
   int Ny = v.Ny;
   int Nc = v.Nc;
   long Nsize = v.Nsize;

   v.Nx = u.Nx;
   v.Ny = u.Ny;
   v.Nc = u.Nc;
   v.Nsize = u.Nsize;
   v.img_data = u.img_data;

   u.Nx = Nx;
   u.Ny = Ny;
   u.Nc = Nc;
   u.Nsize = Nsize;
   u.img_data = temp;
}


