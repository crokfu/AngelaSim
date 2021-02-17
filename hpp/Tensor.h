#pragma once
#include "pch.h"
#include <cstdint>
#include <stdint.h>
#include "Storage.h"
#include <string>
#include <ctime>
#include <stdio.h>
#include <stdint.h>


//! Structure tDim defines tensor dimension.
struct typeDim {
	int width;
	int height;
	int depth;
	int batch;
};

struct paddingDim {
	int width;
	int height;
	int depth;
};

struct convGroupData {
	int in_col_num;
	int in_row_num;
	int in_chan_num_per_group;
	int group_num;
	int in_chan_num;
	int out_chan_num;
	int out_chan_num_per_group;
	int stride_x;
	int stride_y;
	int out_col_num;
	int out_row_num;
};

struct attrib
{
	bool C_CONTIGUOUS;
	bool F_CONTIGUOUS;
	bool OWNDATA;
	bool WRITEABLE;
	bool ALIGNED;
	bool UPDATEIFCOPY;
};

enum typeMatrixGen {
	Zeros	= 0,
	Ones	= 1,
	Random	= 2,
	Eye = 3,
	Identity = 4,
	CenterOne = 5,
	RandomBias = 6
};

//! Tensor is a basic class that implements all convolution computation, including 2D convolutoin, 3D convolution, Depthwise convolution, element convolution. 

class Tensor {
	string tensor_name;
public:
	/// array memory pointer. MUST be reserved in constructor when dimension is known
	tensor_elem_size *M;
	long mult_count;
	long sum_count;
	long mem_read_count;
	long mem_write_count;
	long shift_count;

	bool saturate_bit; // sticky bit
	bool overflow_bit;
	// Numpy-Style Array Attribute

	/// tuple consisting of array dimensions -- return tuple (1,2,3) or (2,3,4)
	typeDim shape;		
	
	/// number of dimension -- return integer, 1,2,3,4 etc.
	int num_dim;			

	/// the length of each element of array in bytes. -- return 1,2,3 byte number
	int item_size;	

	/// Tensor array mem attribute

	/*! C_CONTIGUOUS(C) ---	The data is in a single, C - style contiguous segment */

	/*! F_CONTIGUOUS(F) ---	The data is in a single, Fortran - style contiguous segment */

	/*! OWNDATA(O) ---		The array owns the memory it uses or borrows it from another object */

	/*! WRITEABLE(W)  ---		The data area can be written to.Setting this to False locks the data, making it read - only */

	/*! ALIGNED(A)  ---		The data and all elements are aligned appropriately for the hardware */

	/*! UPDATEIFCOPY(U) ---	This array is a copy of some other array.When this array is deallocated, the base array will be updated with the contents of this array */

	attrib flag; 

	
	int getItemsize();


	/// setTensorname(): set tensot name. It will be usedful in debugging print out. 
	/// @param name -- tensor name

	void setTensorname(string name) {
		tensor_name = name;
	}
	void printShape();

	string getTensorName() {
		return tensor_name;
	}
	/// reshape() to be implemented.
	/// @param depth
	/// @param height
	/// @param width
	void reshape();

	// Array Creation Routines
	Tensor();
	Tensor(int x, int y, int z);
	~Tensor();

	/// get Data in accordance with (x, y, z)
	/// @return values at [x,y,z] location
	/// @param x -- column #
	/// @param y -- row #
	/// @param z -- depth #

	int getMData(int x, int y, int z); 
	
	/// get Data in accordance with (x, y, z)
	/// @return values at [x,y,z] location
	/// @param x -- column #
	/// @param y -- row #
	/// @param z -- depth #

	void setMData(int x, int y, int z, int data);

	// matrix library
	void ones(int verbose=1);
	void zeros(int verbose = 1);
	void identity(int verbose = 1);

	/// generate tensor matrix using random number 
	/// @param min : min value of random range, could be negative or positive
	/// @param max : max value of random range, more than min, could be negative or positive
	inline bool checkOverflow(int x, int bitwidth) {
		if (x >= (1 << bitwidth)) {
			cout << "Tensor ERROR: Tensor  Name: <" << tensor_name << "> : Value = " << x << " >= Threshold :" << (1 << bitwidth) << endl;
			return false;
		}
		else { return true; }
	}


	/// apply bit mask to tensor element 
	/// the saturation threshold is 2^bitwidth
	/// if saturated, give (2^bitwidth - 1) 
	/// if not saturated, give out x.
	/// @param x		: input value
	/// @param bitwidth : bitwidth
	inline int maskTensorEle(int x, int bitwidth) {
		//cout << "saturation threshold = " << (x&(~((1 << bitwidth) - 1))) << endl;
		//cout << "saturation value = " << ((1 << bitwidth) - 1) << endl;		
		//cout << "value if not saturated = " << (x &(~((~0) << bitwidth))) << endl;
		//return (x&(~((1 << bitwidth) - 1)))?((1<<bitwidth)-1):(x &(~((~0) << bitwidth)));
		if (x >= ((1 << bitwidth) - 1))
			return 127;
		else if (x <= (-1 * (1 << bitwidth)))
			return -128;
		else
			return x;
	}


	/// randGen() generate random number matrix. Dimension is defined in tensor constructor
	/// @param verbose : print log debugging level 
	/// @param min : min value of random value generated
	/// @param max : max value of random value generated
	
	void randGen(bool saturation_on = false, int mask_bit = 8, int min = 0, int max = RAND_MAX, int verbose = 1 );
	void randGen(int seed, bool saturation_on = false, int mask_bit = 8, int min = 0, int max = RAND_MAX, int verbose = 1);

	/// centerOne() generate centerOne Matrix used in  3x3 convolution bypass mode. 
	/// 0 0 0
	/// 0 1 0
	/// 0 0 0
	/// @param verbose : print log debugging level 
	void centerOne(int verbose = 1);

	/// get shape information :  column nuber
	/// @return values at [x,y,z] location

	int getWidth();


	/// get shape information : row number
	/// @return row values

	int getHeight();


	/// get  shape information : channel number
	/// @return values at [x,y,z] location

	int getDepth();

	int getBatch();

	/// set shape information :  column nuber
	/// @param values at [x,y,z] location

	void setWidth(int width);


	/// set shape information :  column nuber
	/// @param values at [x,y,z] location

	void setHeight(int height);


	/// set shape information :  column nuber
	/// @param values at [x,y,z] location

	void setDepth(int depth);

	void setBatch(int depth);

	// i/o w

	/// print Vect in std IO 
	/// @param vecName the vecName to display when print in std IO

	void printVec(const std::string vecName);
	void shiftChannelWise(Tensor &InShift);

	/// write tensor to a ASCII file -- print hex format with bitwidth defined
	/// output_file << std::hex <<getMData(j, i, l) <<' ';
	/// @param vecName the vecName to display when print in std IO

	void writeOutputActivation(std::string vecName);

	/// write tensor to a ASCII file. -- print hex format with 2-bit width defined.
	/// fprintf(fptr, "%02x ", getMData(j, i, l));
	/// @param vecName the vecName to display when print in std IO

	void writeHexFormatOutputActivation(std::string vecName, string caseName, bool readIn= false);
	void writeHexFormatGlobalPoolOutputActivation(std::string vecName, string caseName);
	void writeHexFormatIntermediateGlobalPoolOutputActivation(std::string vecName, string caseName);
	
	void writeDecimalFormatOutputActivation(string vecName, string caseName, bool readIn = false);
	void writeDecimalFormatGlobalPoolOutputActivation(string vecName, string caseName);
	void writeMemoryBin(string vecName, string caseName, bool readIn = false);
	void writeMemoryDepthBin(string vecName, string caseName, bool readIn = false);
	void writeMemoryBin32b(string vecName, string caseName, bool readIn = false);


	/// write tensor to an ASCII file -- print hex format with 2-bit width defined.
	/// fprintf(fptr, "%02x ", getMData(j, i, l));
	/// @param vecName the vecName to display when print in std IO
	void writeFormatOutputActivation32b(string vecName, string caseName, bool readIn = false);

	void writeFormatInputActivation(std::string vecName, string caseName, bool readIn = false);
	void writeDecimalFormatInputActivation(std::string vecName, string caseName, bool readIn = false);

	///	printOutputActivation() print Vector in std IO
	/// printf("%#4x \t", getMData(j, i, l));
	/// @param vecName the vecName to display when print in std IO

	void printOutputActivation(std::string vecName);

	// math function
	///	ReLUFunc() do element-level ReLU float version. 
	/// @return ReLU Out Value
	/// @param inTensor1 -- activation input
	
	float	ReLUFunc(float x);


	///	ReLUFunc() do element-level ReLU integer version. 
	/// @param in -- input value
	
	int		ReLUFunc(int   in);


	///	ReLUFunc() do element-level ReLU integer version. 
	/// @param in -- input value
	/// @param x  -- ReLUX cap value

	int		ReLUXFunc(int   in, int x);
	void	ReLUXTensor(int x);
	void	ReLUTensor();

	///	MaxPool() do pooling for 2D kernel
	/// @param inTensor -- input tensor 
	/// @param size  -- pooling range size 2x2 or 3x3
	/// @param stride -- stride between each pooling iteration.

	int		MaxPool(Tensor & inTensor, int size =3, int stride = 1,int padding = 1, int verbose = 1);
	int		MaxPoolNoPad(Tensor & inTensor, int size = 3, int stride = 1, int padding = 1, int verbose = 1);
	void	copyTensor(Tensor &in);
	void	copyMultTensor(Tensor &in);
	void	globalSum(Tensor &in);
	void	globalSum(Tensor *in);
	
	void	globalIntermediateSum(Tensor &in);
	void	globalIntermediateSum(Tensor *in);


	///	AvgPool() do Average pooling for 2D kernel
	/// @param inTensor -- input tensor 
	/// @param size  -- pooling range size 2x2 or 3x3
	/// @param stride -- stride between each pooling iteration.

	int		AvgPool(Tensor & inTensor,int size=3, int stride = 1, int verbose = 1);
	int		AvgPoolNoPad(Tensor & inTensor, int size = 3, int stride = 1, int verbose = 1);
	int		AvgPool2x2NoPad(Tensor & inTensor, int size = 3, int stride = 1, int verbose = 1);

	int		AvgPoolPad(Tensor & inTensor, int size = 3, int stride = 1, int verbose = 1);

	/// max() output the max value of one-dimension vector 
	/// @param in -- 1-D input vector

	int		max(vector<int> &in);
	
	/// max() output the max value of 2-dimension maxtrix. It find max value of all 2D array. used in MaxPooling.
	/// it firstly find the max value each row. and push averages to a 1-D vector.
	/// it  find max value in 1-D average vector just generated before
	/// @param in -- 2-D input vector array.

	int		max(vector<vector<int>> &in);

	/// max() output the average value of one-dimension vector 
	/// @param in -- 1-D input vector

	int		accum(vector<int> &in);

	/// max() output the average value of 2-dimension maxtrix. It find max value of all 2D array. used in MaxPooling.
	/// it firstly find the average value each row. and push averages to a 1-D vector.
	/// it  find average value in 1-D average vector just generated before
	/// @param in -- 2-D input vector array.


	int		accum(vector<vector<int>> &in);
	void signTensor(Tensor &in);
	
	int  sign(int x);

	bool checkPrintDim4Alloc(int verbose =1);


	///	to_csv() write tensor to csv file. 
	/// @param CSVFileName -- csv output file Name
	/// @param verbose  -- tensor print verbose #

	void to_csv( std::string CSVFileName, std::string caseName="case00", bool readIn = false , int verbose = 1);
	std::vector<std::pair<std::string, std::vector<int>>> read_csv(std::string CSVFileName);

	inline int shiftAndRound(const float x, int bit) {
		float temp = x;
		for (int i = 0; i < bit + 1; i++) {
			temp = temp / 2.0;
			//cout << "temp[" << i << "]=" << temp << endl;
		}
		return (int) round(temp);
	}

	//inline int shiftAndRoundPosNeg(const float x, int bit) {
	//	float temp = x;
	//	if (bit >= 0) {
	//		for (int i = 0; i < bit + 1; i++) {
	//			temp = temp / 2.0;
	//			//cout << "temp[" << i << "]=" << temp << endl;
	//		}
	//	}
	//	else {
	//		for (int i = bit; i < 0; i++) {
	//			temp = temp * 2.0;
	//			//cout << "temp[" << i << "]=" << temp << endl;
	//		}
	//	}
	//	return round(temp);
	//}

	inline int shiftAndRoundInVec(const float x, int bit) {
		float temp = x;
		for (int i = 0; i < bit; i++) {
			temp = temp / 2.0;
			//cout << "temp[" << i << "]=" << temp << endl;
		}
		return (int) round(temp);
	}
	int biasShiftAndRound(int wgtSum, int bias, int RShift_bit);
	void printNegHex(FILE *fptr, int x);
	
	void printNegHex32b(FILE *fptr, int x);
	void printNegHex32bWoSpace(FILE *fptr, int x);

	void print24bDataHexFormat(FILE *fptr, int x);
	void printNegHex(int x);
	int str2Hex(string str);
	int str2Dec(string str);
	bool operator += (Tensor &inTensor1);
	int shiftPosNeg(int shift_in, int shift_pos);
	void printNegHexWoSpace(FILE *fptr, int x);

	friend Tensor operator + (Tensor &inTensor1, Tensor &inTensor2);
};
