/*

  FFT sets up apprx 1Hz 
  Display is PSD on logarthmic scale
  
  FFT copied from Arduino FFT_03

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
  Written by Limor Fried/Ladyada for Adafruit Industries.
  MIT license, all text above must be included in any redistribution
 ***************************************************

*/
#include <SPI.h>
//#include "Adafruit_GFX.h"
//#include "Adafruit_HX8357.h"
#include "arduinoFFT.h"

#if defined __SAMD51__
   #define STMPE_CS 6
   #define TFT_CS   9
   #define TFT_DC   10
   #define SD_CS    5
#endif

#define TFT_RST -1

// Use hardware SPI and the above for CS/DC
//Adafruit_HX8357 tft = Adafruit_HX8357(TFT_CS, TFT_DC, TFT_RST);
															  
arduinoFFT FFT = arduinoFFT(); /* Create FFT object */

#define CHANNEL A0
const uint16_t samples = 128; //This value MUST ALWAYS be a power of 2
const double samplingFrequency = 100; //Hz, must be less than 10000 due to ADC

unsigned int sampling_period_us;
unsigned long microseconds;

/*
These are the input and output vectors
Input vectors receive computed results from FFT
*/
double vReal[samples];
double vImag[samples];
double vPSDs[samples];

#define SCL_INDEX 0x00
#define SCL_TIME 0x01
#define SCL_FREQUENCY 0x02
#define SCL_PLOT 0x03
int MaxV = 100; //Maximum PSD value * 10

/*
  Calculating y value
  y= 0 - 49 is Title area
  y = 290 – 320 is Bottom label area
  so y range is 50 to 290
  
  VALy is the input from the array
  
  so VALy * z + b = y0
  
  If VALy = MaxV(100) then y0 = 50. (The rectangle will be maximum)
    MaxV * z + b = 50 (First equation)
  
  If VALy = 0 the y0 = 290
    0 * z + b = 290 (Second equation)
  
  Solving Second equation for b: b = 290
  Reducing First equation
    MaxV * z + 290 = 50
    MaxV * z = -240
    z = -240/MaxV
    uint16_t yz = VALy * (-240.0/Max) + 290
    h = 290 – yz

*/
uint16_t yz;  // Uses to calculate yval on TFT
//uint16_t Color; // Color for TFT

//placeholders
int alpha = 0;
int beta = 0;
int gamma = 0;
int theta = 0;
int delta = 0;

// Prototypes
void drawGraph();
void PrintVector(double , uint16_t , uint8_t );

									  
void setup(){
  sampling_period_us = round(1000000*(1.0/samplingFrequency));
  Serial.begin(115200);
//  while(!Serial); // Comment out if battery powered
//  tft.begin();
//  tft.setRotation(1);
//  tft.fillScreen(HX8357_BLACK); // Clear screen set to black backgroung
//
//  drawGraph(); // draws the graph

}

void loop(){
  /*SAMPLING*/
  microseconds = micros();
  for(int i=0; i<samples; i++){
//      vReal[i] = analogRead(CHANNEL);
      vReal[i] = analogRead(CHANNEL)-512.0;  //modified for zero balance
//      Serial.println(vReal[i]);
      vImag[i] = 0;
      while(micros() - microseconds < sampling_period_us){
        //empty loop
      }
      microseconds += sampling_period_us;
  }
// while (1);
  /* Print the results of the sampling according to time */
  FFT.Windowing(vReal, samples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);	/* Weigh data */
  FFT.Compute(vReal, vImag, samples, FFT_FORWARD); /* Compute FFT */
  FFT.ComplexToMagnitude(vReal, vImag, samples); /* Compute magnitudes */
  PrintVector(vReal, (samples >> 1), SCL_FREQUENCY);
  double x = FFT.MajorPeak(vReal, samples, samplingFrequency);
  Serial.println(x, 6); //Print out what frequency is the most dominant.

  //tft.fillRect(0,50,480,250,HX8357_BLACK);  //Clear previous data

  for (int i = 0; i<(samples >> 1); i++){
    Serial.println(vPSDs[i],2);

    /* 
     *  
     * MTS estimated PSD values are from -2 to 8, so I added to to make 0-10
     * Value multiplied by 10 so value is 0 to 100 (MaxV)
     * The calculations are explained above
      */
    yz = 10 * (vPSDs[i] + 2) * (-240.0/MaxV) + 290;
    if (yz < 50) yz = 50; // bin 0 is always big

    if (i <= 5)                {
      //Color =  HX8357_RED; //Delta (0 - 4Hz 0 - 3.91)
      delta++;
    }
    if ((i > 5) && (i < 12))   {
      //Color = HX8357_GREEN; //Theta (4 - 8Hz 4.69 - 7.81)
      theta++;
    }
    if ((i >= 12) && (i < 16)) {
      //Color = HX8357_YELLOW; //Alpha (8 - 12Hz 9.38 - 11.72)
      alpha++;
    }
    if ((i >= 16) && (i < 39)) {
      //Color = HX8357_CYAN; //Beta (12 - 30Hz 12.5 - 29.69)
      beta++;
    }
    if (i >= 39){               
      //Color = HX8357_MAGENTA; //Gamma (30+ Hz 31.25+)
      gamma++;
    }
    //tft.fillRect(i * 7 , yz, 7, 290 - yz, Color);

  }
    Serial.print("##Delta: ");
    Serial.print(delta);
    Serial.print("Theta: ");
    Serial.print(theta);
    Serial.print("Alpha: ");
    Serial.print(alpha);
    Serial.print("Beta: ");
    Serial.print(beta);
    Serial.print("Gamma: ");
    Serial.print(gamma);
    Serial.println("$$");

    delta = 0;
    theta = 0;
    alpha = 0;
    beta = 0;
    gamma = 0;
  //while(1); /* Run Once */
   delay(1000); /* Repeat after delay */
}

//void drawGraph(){
//
//  // draw title
//  tft.setCursor(10, 10);
//  tft.setTextColor(HX8357_RED);
//  tft.setTextSize(4);
//  tft.println("EEG");
//
//  // draw labels
//  tft.setCursor(0, 300);
//  tft.setTextColor(HX8357_RED);
//  tft.setTextSize(3);
//  tft.println("De");
//  tft.setCursor(41, 300);
//  tft.setTextColor(HX8357_GREEN);
//  tft.setTextSize(3);
//  tft.println("Th");
//  tft.setCursor(81, 300);
//  tft.setTextColor(HX8357_YELLOW);
//  tft.setTextSize(3);
//  tft.println("Al");
//  tft.setCursor(150, 300);
//  tft.setTextColor(HX8357_CYAN);
//  tft.setTextSize(3);
//  tft.println("Beta");
//  tft.setCursor(320, 300);
//  tft.setTextColor(HX8357_MAGENTA);
//  tft.setTextSize(3);
//  tft.println("Gamma");
//}


void PrintVector(double *vData, uint16_t bufferSize, uint8_t scaleType)
{
  for (uint16_t i = 0; i < bufferSize; i++){
    double abscissa;
    abscissa = ((i * 1.0 * samplingFrequency) / samples);
    Serial.print(abscissa, 6);
    Serial.print(",");
    Serial.println(vData[i], 4);
    vPSDs[i] = log10(vData[i]*vData[i]/0.78125);  //save PSD values
  }
  Serial.println();
}
