#include "pch.h"
#include "Utilities.h"
#include "Tensor.h"

Tensor::Tensor() {

#if STATIC_DATA
	tensor_elem_size M[MAX_T_DEPTH][MAX_T_HEIGHT][MAX_T_WIDTH];
#endif // STATIC_DATA
#ifndef STATIC_DATA
	M = (tensor_elem_size *)malloc(sizeof(tensor_elem_size)*shape.width * shape.height * shape.depth);
#endif //

	
	item_size = sizeof(tensor_elem_size);

	flag.C_CONTIGUOUS = true;
	flag.F_CONTIGUOUS = false;
	flag.OWNDATA = false;
	flag.WRITEABLE = true;
	flag.ALIGNED = false;
	flag.UPDATEIFCOPY = false;
}
Tensor::Tensor(int x, int y, int z) {
	//cout << "tensor constructor" << endl;
	shape.width = x;
	shape.height = y;
	shape.depth = z;

#if STATIC_DATA
	tensor_elem_size M[MAX_T_DEPTH][MAX_T_HEIGHT][MAX_T_WIDTH];
#endif // STATIC_DATA

#ifndef STATIC_DATA
	M = (tensor_elem_size *)malloc(sizeof(tensor_elem_size)*shape.width * shape.height * shape.depth);
#endif // STATIC_DATA

	item_size = sizeof(tensor_elem_size);

	flag.C_CONTIGUOUS = true;
	flag.F_CONTIGUOUS = false;
	flag.OWNDATA = false;
	flag.WRITEABLE = true;
	flag.ALIGNED = false;
	flag.UPDATEIFCOPY = false;
}

void Tensor::printShape() {
	cout << "Size([" << getDepth() << "," << getHeight() << "," << getWidth() << "])" << endl;
}
Tensor::~Tensor() {
	
}

int Tensor::getMData(int x, int y, int z) {
	return M[z*getWidth()*getHeight() + y * getWidth() + x];
}


void Tensor::setMData(int x, int y, int z, int data) {
	M[z*getWidth()*getHeight() + y * getWidth() + x] = data;
}


void Tensor::printVec(const string vecName) {

	accumulator_set<double, stats<tag::mean, tag::count, tag::variance > > acc;

	cout << "Tensor::printVec(): Now we print Vector (" << vecName << ") " << endl;
	cout <<	"we have <x,y,z> = " <<getWidth() <<"," << getHeight() << "," << getDepth()<< endl;
	for (int l = 0; l < getDepth(); l++) {
		cout << "Page[" << l << "]" << endl << "<"<<endl;
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				printf("[%d,%d,%d] %d \t", l, i, j, getMData(j, i, l));
				acc(getMData(j, i, l));
			}
			cout << endl;
		}
		cout << ">" << endl;
	}
	cout << endl << endl;
	//std::cout << "Tensor Statistical : " << std::endl<<endl;
	//std::cout << "\t Count:		\t" << boost::accumulators::count(acc) << std::endl;
	//std::cout << "\t Mean:		\t " << mean(acc) << std::endl;
	//std::cout << "\t Variance:	\t " << variance(acc) << std::endl;



}

void Tensor::shiftChannelWise(Tensor &in_shift) {
	cout << "Skip Shift Bit Vector " << in_shift.getHeight() << ","<<in_shift.getWidth() <<endl;
	/*for (int i = 0; i < in_shift.getHeight(); i++) {
		printf("in_shift[0,%d,0] %d \n", i, in_shift.getMData(0, i, 0));
	}*/

	cout << "Tensor::printVec(): Now we print Input Volume (SkipShift) we have <x,y,z> = " << getWidth() << "," << getHeight() << "," << getDepth() << endl;
	//for (int l = 0; l < getDepth(); l++) {
	//	cout << "Page[" << l << "]" << endl << "<" << endl;
	//	for (int i = 0; i < getHeight(); i++) {
	//		for (int j = 0; j < getWidth(); j++) {
	//			printf("[%d,%d,%d] %d \t", l, i, j, getMData(j, i, l));
	//		}
	//		cout << endl;
	//	}
	//	cout << ">" << endl;
	//}
	//cout << endl << endl;

	cout << endl << endl;
	
	for (int l = 0; l < getDepth(); l++) {
		//cout << "Page[" << l << "]" << endl << "<" << endl;
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				//printf("[%d,%d,%d] %d<<%d  =  %d\t", l, i, j, getMData(j, i, l), in_shift.getMData(0, l, 0), shiftPosNeg(getMData(j, i, l),in_shift.getMData(0, l, 0)));
				this->M[l*getHeight()*getWidth() + i * getWidth() + j] = shiftPosNeg(getMData(j, i, l), in_shift.getMData(0, l, 0));
			}
			//cout << endl;
		}
		//cout << ">" << endl;
	}

}

void Tensor::printOutputActivation(string vecName) {

	cout << "Tensor::printOutputActivation(): Now we print Output Activation (" << vecName << ") " << endl;
	cout << "#Number of output channels = <" << shape.depth<< ">" << endl;
	cout << "#Number of columns = <" << shape.width << ">" << endl;
	cout << "#Number of rows = <" << shape.height << ">" << endl;
	
	for (int l = 0; l < getDepth(); l++) {
		cout << "Channel [" << l << "]" << endl;
		for (int i = 0; i < getWidth(); ++i) {
			cout << "#ROW " << i << endl;
			for (int j = 0; j < getHeight(); j++) {
				//printf("[%d,%d,%d] <%d> \t", l, i, j, getMData(j, i, l));
				printf("%#4x \t", getMData(j, i, l));
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl << endl;
}

void Tensor::writeOutputActivation(string vecName) {

	ofstream output_file;
	output_file.open("out\\"+vecName+".txt");
	output_file << "#Number of output channels = " << shape.depth  << endl;
	output_file << "#Number of columns = " << shape.width << endl;
	output_file << "#Number of rows = " << shape.height  << endl;

	for (int l = 0; l < getDepth(); l++) {
		output_file << "#CHANNEL " << std::dec << l << endl;
		for (int i = 0; i < getWidth(); ++i) {
			output_file << "#ROW " << std::dec << i << endl;
			for (int j = 0; j < getHeight(); j++) {
				output_file << std::hex <<getMData(j, i, l) <<' ';
				//printf("%#4x \t", getMData(j, i, l));
			}
			output_file << endl;
		}
		output_file << endl;
	}
	output_file << endl << endl;
	output_file.close();
}
void Tensor::writeFormatOutputActivation32b(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false) 
		s= caseName + "\\out\\" + vecName + ".txt";
	else 
		s = "in\\" + caseName + "\\out\\" + vecName + ".txt";
	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);
	fprintf(fptr, "#Number of columns = %d\n", shape.width);
	fprintf(fptr, "#Number of rows = %d\n\n", shape.height);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			fprintf(fptr, "#ROW %d\n", i);
			for (int j = 0; j < getWidth(); j++) {
				//output_file << std::hex << getMData(j, i, l) << ' ';
				//fprintf(fptr,"%08x ", getMData(j, i, l));
				this->printNegHex32b(fptr, getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}

void Tensor::writeHexFormatOutputActivation(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false)
		s = caseName + "\\out\\" + vecName + ".txt";
	else
		s = "in\\"+ caseName + "\\out\\" + vecName + ".txt";

	char* cstr = const_cast<char*>(s.c_str());
	errno_t err = fopen_s(&fptr, cstr, "w");

	if (err == 0 and SHOW_VECTOR == true)
	{
		//printf("Tensor::writeHexFormatOutputActivation.c: The file 'OutputActivation.txt' was opened\n");
		if (debug_level > 1) cout << "Tensor::writeHexFormatOutputActivation.c: The file " << vecName << " was opened" << endl;
	}
	else
	{
		//printf("Tensor::writeHexFormatOutputActivation.c: The file 'OutputActivation.txt' was NOT opened\n");
		if (debug_level > 1) cout << "Tensor::writeHexFormatOutputActivation.c: The file " << vecName << " was NOT opened" << endl;
	}

	fprintf(fptr,"#Number of output channels = %d\n", shape.depth );
	fprintf(fptr,"#Number of columns = %d\n", shape.width);
	fprintf(fptr,"#Number of rows = %d\n\n", shape.height);
	
	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr,"#CHANNEL %d \n", l );
		for (int i = 0; i < getHeight(); ++i) {
			fprintf(fptr, "#ROW %d\n", i );
			for (int j = 0; j < getWidth(); j++) {
				//output_file << std::hex << getMData(j, i, l) << ' ';
				//fprintf(fptr,"%02x ", getMData(j, i, l));
				this->printNegHex(fptr, getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}

void Tensor::writeHexFormatGlobalPoolOutputActivation(string vecName, string caseName) {

	FILE *fptr;
	string s = caseName + "\\out\\" + vecName + ".txt";
	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			
			for (int j = 0; j < getWidth(); j++) {
				this->printNegHex(fptr, getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}
void Tensor::writeHexFormatIntermediateGlobalPoolOutputActivation(string vecName, string caseName) {

	FILE *fptr;
	string s = caseName + "\\out\\" + vecName + ".txt";
	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);
	fprintf(fptr, "#Number of output rows = %d\n", shape.height);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			fprintf(fptr, "#ROW %d\n", i);
			for (int j = 0; j < getWidth(); j++) {
				this->printNegHex32b(fptr, getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}
void Tensor::writeDecimalFormatOutputActivation(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false) 
		s = caseName + "\\out\\decimal\\" + vecName + ".txt";
	else
		s = "in\\" + caseName + "\\out\\decimal\\" + vecName + ".txt";
	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);
	fprintf(fptr, "#Number of columns = %d\n", shape.width);
	fprintf(fptr, "#Number of rows = %d\n\n", shape.height);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			fprintf(fptr, "#ROW %d\n", i);
			for (int j = 0; j < getWidth(); j++) {
				//output_file << std::hex << getMData(j, i, l) << ' ';
				fprintf(fptr,"%02d ", getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}

void Tensor::writeDecimalFormatGlobalPoolOutputActivation(string vecName, string caseName) {

	FILE *fptr;
	string s = caseName + "\\out\\decimal\\" + vecName + ".txt";
	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			for (int j = 0; j < getWidth(); j++) {
				//output_file << std::hex << getMData(j, i, l) << ' ';
				fprintf(fptr, "%02d ", getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}

void Tensor::writeFormatInputActivation(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false)
		s= caseName + "\\out\\" + vecName + ".txt";
	else
		s="in\\"+ caseName + "\\out\\" + vecName + ".txt";

	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);
	fprintf(fptr, "#Number of columns = %d\n", shape.width);
	fprintf(fptr, "#Number of rows = %d\n\n", shape.height);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			fprintf(fptr, "#ROW %d\n", i);
			for (int j = 0; j < getWidth(); j++) {
				//output_file << std::hex << getMData(j, i, l) << ' ';
				//fprintf(fptr, "%02x ", getMData(j, i, l));
				this->printNegHex(fptr, getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}

void Tensor::writeDecimalFormatInputActivation(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn== false) 
		s= caseName + "\\out\\decimal\\" + vecName + ".txt";
	else 
		s = "in\\" + caseName + "\\out\\decimal\\" + vecName + ".txt";

	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");


	fprintf(fptr, "#Number of output channels = %d\n", shape.depth);
	fprintf(fptr, "#Number of columns = %d\n", shape.width);
	fprintf(fptr, "#Number of rows = %d\n\n", shape.height);

	for (int l = 0; l < getDepth(); l++) {
		fprintf(fptr, "#CHANNEL %d \n", l);
		for (int i = 0; i < getHeight(); ++i) {
			fprintf(fptr, "#ROW %d\n", i);
			for (int j = 0; j < getWidth(); j++) {
				//output_file << std::hex << getMData(j, i, l) << ' ';
				fprintf(fptr, "%02d ", getMData(j, i, l));
			}
			fprintf(fptr, "\n");
		}
		fprintf(fptr, "\n");
	}
	fprintf(fptr, "\n\n");
	fclose(fptr);
}

float Tensor::ReLUFunc(float x) {
	if (x > 0)
		return x;
	else
		return 0;
}
int Tensor::ReLUFunc(int x) { 
	if (x > 0)
		return x;
	else
		return 0;
	; 
}

void Tensor::ReLUTensor() {
	int k, i, j;

	for (k = 0; k < this->getDepth(); k++) {
		for (i = 0; i < this->getHeight(); i++) {
			for (j = 0; j < this->getWidth(); j++) {
				M[j + i * this->getWidth() + k * this->getWidth() * this->getHeight()] = this->ReLUFunc(M[j + i * this->getWidth() + k * this->getWidth() * this->getHeight()]);
			}
		}
	}
}

void Tensor::ReLUXTensor(int x) {
	int k, i, j;

	for (k = 0; k < this->getDepth(); k++) {
		for (i = 0; i < this->getHeight(); i++) {
			for (j = 0; j < this->getWidth(); j++) {
				M[j + i * this->getWidth() + k * this->getWidth() * this->getHeight()] = this->ReLUXFunc(M[j + i * this->getWidth() + k * this->getWidth() * this->getHeight()],x);
			}
		}
	}
}

int Tensor::ReLUXFunc(int in, int x) {
	if (x < 0) {
		cout << "Tensor:ReLUX threshold can NOT be less than 0" << endl;
		exit(-1);
	}
	else if (x == 0){
		if (in > 0)
			return in;
		else
			return 0;
	}
	else {
		if (in > int(pow(2.0, x) - 1))
			return int(pow(2.0, x) - 1);
		else if (in > 0)
			return in;
		else
			return 0;
	}
}

void Tensor::copyTensor(Tensor &in) {
	for (int i = 0; i < this->shape.depth; i++) {
		for (int j = 0; j < this->shape.height; j ++) {
			for (int k = 0; k < this->shape.width; k++) {
				M[i*getWidth()*getHeight() + j * getWidth() + k] = in.getMData(k, j, i);
			}
		}
	}
}

void Tensor::copyMultTensor(Tensor &in) {
	for (int i = 0; i < this->shape.depth; i++) {
		for (int j = 0; j < this->shape.height; j++) {
			for (int k = 0; k < this->shape.width; k++) {
				if (SHOW_VECTOR==true && debug_level>1) cout << in.getMData(k, 0, 0) << "\t";
				M[i*getWidth()*getHeight() + j * getWidth() + k] = in.getMData(k, 0, 0);
			}
			if (SHOW_VECTOR == true && debug_level > 1) cout << endl;
		}
	}
}

int Tensor::sign(int x) {
	return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

void Tensor::signTensor(Tensor &in) {
	for (int i = 0; i < this->shape.depth; i++) {
		for (int j = 0; j < this->shape.height; j++) {
			for (int k = 0; k < this->shape.width; k++) {
				M[i*getWidth()*getHeight() + j * getWidth() + k] = sign(in.getMData(k, j, i));
			}
		}
	}
}

void Tensor::globalSum(Tensor &in) {
	int sum_per_channel;
	for (int i = 0; i < in.shape.depth; i++) {
		sum_per_channel = 0;
		for (int j = 0; j < in.shape.height; j++) {
			for (int k = 0; k < in.shape.width; k++) {
				sum_per_channel = sum_per_channel + in.getMData(k, j, i);
			}
		}
		M[i] = sum_per_channel; 
	}
}

void Tensor::globalSum(Tensor *in) {
	int sum_per_channel;
	for (int i = 0; i < in->shape.depth; i++) {
		sum_per_channel = 0;
		for (int j = 0; j < in->shape.height; j++) {
			for (int k = 0; k < in->shape.width; k++) {
				sum_per_channel = sum_per_channel + in->getMData(k, j, i);
			}
		}
		M[i] = sum_per_channel;
	}
}

void Tensor::globalIntermediateSum(Tensor &in) {
	int sum_per_channel;
	for (int i = 0; i < in.shape.depth; i++) {
		sum_per_channel = 0;
		for (int j = 0; j < in.shape.height; j++) {
			for (int k = 0; k < in.shape.width; k++) {
				sum_per_channel = sum_per_channel + in.getMData(k, j, i);
			}
			M[i*getHeight() + j] = sum_per_channel;
		}
	}
}
void Tensor::globalIntermediateSum(Tensor *in) {
	int sum_per_channel;
	for (int i = 0; i < in->shape.depth; i++) {
		sum_per_channel = 0;
		for (int j = 0; j < in->shape.height; j++) {
			for (int k = 0; k < in->shape.width; k++) {
				sum_per_channel = sum_per_channel + in->getMData(k, j, i);
			}
			M[i*getHeight() + j] = sum_per_channel;
		}
	}
}
int Tensor::MaxPool(Tensor &in, int size, int stride, int padding, int verbose) {
	cout << "This MAX Pool Function" << endl;

	vector<int> pad;
	int max_pool;


	cout << "This Max Pool : kernel size = " << size << endl;
	cout << "This Max Pool : pool stride = " << stride << endl;
	for (int i = 0; i < shape.depth; i++) {
		if (verbose >= 2) cout << "page [" << i << "]" << endl;
		for (int j = 0; j <= in.shape.height - size; j=j+stride) {
			for (int k = 0; k <= in.shape.width - size; k=k+stride) {
				pad.clear();
				max_pool = 0;
				if (verbose >= 2) cout << "get data pad"<<endl;
				for (int l = 0; l < size; l++) {
					for (int m = 0; m < size; m++) {
						pad.push_back(in.getMData(k + m, j + l, i));
						if (verbose >= 2) cout << in.getMData(k + m, j + l, i) << "\t";
					}
					if (verbose >= 2) cout << endl;
				}
				//max_pool = *std::max_element(std::begin(pad), std::end(pad));
				max_pool = max(pad);
				if (verbose >= 2) cout << "plMax[" << i << "][" << j << "][" << k << "] = " << max_pool << "\t";
				M[i*getWidth()*getHeight() + (j / stride) * getWidth() + (k / stride)] = max_pool ;
				if (verbose >= 2) cout << endl;
			}
			if (verbose >= 2) cout << endl;
		}
	}

	return 0;
}

int Tensor::MaxPoolNoPad(Tensor &in, int size, int stride, int padding, int verbose) {
	cout << "This MAX Pool Function" << endl;

	vector<int> pad;
	int max_pool;


	cout << "This Max Pool : kernel size = " << size << endl;
	cout << "This Max Pool : pool stride = " << stride << endl;
	for (int i = 0; i < in.shape.depth; i++) {
		if (verbose >= 2) cout << "page [" << i << "]" << endl;
		for (int j = 0; j <= in.shape.height - size; j = j + stride) {
			for (int k = 0; k <= in.shape.width - size; k = k + stride) {
				pad.clear();
				max_pool = 0;
				if (verbose >= 2) cout << "get data pad" << endl;
				for (int l = 0; l < size; l++) {
					for (int m = 0; m < size; m++) {
						pad.push_back(in.getMData(k + m, j + l, i));
						if (verbose >= 2) cout << in.getMData(k + m, j + l, i) << "\t";
					}
					if (verbose >= 2) cout << endl;
				}
				//max_pool = *std::max_element(std::begin(pad), std::end(pad));
				max_pool = max(pad);
				if (verbose >= 2) cout << "plMax[" << i << "][" << j << "][" << k << "] = " << max_pool << "\t";
				M[i*getWidth()*getHeight() + (j / stride) * getWidth() + (k / stride)] = max_pool;
				if (verbose >= 2) cout << endl;
			}
			if (verbose >= 2) cout << endl;
		}
	}
	//this->setWidth(4);
	//this->setHeight(4);
	return 0;
}

int Tensor::AvgPool(Tensor & in, int size, int stride, int verbose) {
	vector<int> pad;
	int sum;
	
	cout << "This Avg Pool : size = " <<size << endl;
	cout << "This Avg Pool : stride = " << stride << endl;
	for (int i = 0; i < in.shape.depth; i++) {
		if (SHOW_VECTOR == true)cout << "page [" << i << "]" << endl;
		for (int j = 0; j <= in.shape.height - size; j=j+stride) {
			for (int k = 0; k <= in.shape.width - size; k=k+stride) {
				pad.clear();
				sum = 0;
				// get data pad
				for (int l = 0; l < size; l++) {
					for (int m = 0; m < size; m++) {
						pad.push_back(in.getMData(k + m, j + l, i));
						if (verbose == 2) cout << in.getMData(k + m, j + l, i) << "\t";
						//pad.push_back(in.getMData(k + m, j + l, i));
						//if (verbose == 2) cout << in.getMData(k*stride + m, j*stride + l, i) << "\t";
					}
					if (verbose == 2) cout << endl;
				}
				sum = accum(pad);
				if (SHOW_VECTOR == true) cout << "plAvg[" << i << "][" << j << "] = " << round((float(sum)/float(size*size))) <<"\t";
				//M[i*getWidth()*getHeight() + j * getWidth() + k] = round((float(sum) / float(size*size)));
				M[i*getWidth()*getHeight() + (j / stride) * getWidth() + (k / stride)] = round((float(sum) / float(size*size)));;
				if (verbose == 2) cout << endl;
			}
			if (SHOW_VECTOR == true) cout << endl;
		}
		if (SHOW_VECTOR == true) cout << endl;
	}


	return 0;
}

int Tensor::AvgPoolNoPad(Tensor & in, int size, int stride, int verbose) {
	vector<int> pad;
	int sum;
	float avg_result;
	float avg_result_int;
	int div_count;

	//assert(in.getWidth() == this->getWidth());
	cout << "This Avg Pool : size = " << size << endl;
	cout << "This Avg Pool : stride = " << stride << endl;
	for (int i = 0; i < in.getDepth(); i++) {
		if (SHOW_VECTOR == true||verbose >=2)cout << "page [" << i << "]" << endl;
		for (int j = 0; j < in.getHeight(); j = j + stride) {
			for (int k = 0; k < in.getWidth(); k = k + stride) {
				pad.clear();
				sum = 0;
				div_count = 0;

				if ((j == 0 && k ==0)) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(                  k + m, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout <<"LT"<< in.getMData(k + m, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size-1)*(size-1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				//else if (k == shape.width-1 && j == 0) {
				else if (k == in.shape.width - 1 && j == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(                    k + m-1, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "RT" << in.getMData(k + m-1, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size -1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				//else if (j == shape.height - 1 && k ==0) {
				else if (j == in.shape.height - 1 && k == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size-1; m++) {
							pad.push_back(in.getMData(                    k + m, j + l-1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BL" << in.getMData(k + m, j + l-1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size -1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == in.shape.height - 1 && k == in.shape.width - 1) {
				//else if (j == shape.height - 1 && k == shape.width - 1) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(                    k + m - 1, j + l-1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BR" << in.getMData(k + m - 1, j + l-1, i) << "\t";
						}
						if (verbose == 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size -1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size; m++) {
							//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							if ((k + m - 1) < in.getWidth()) {
								pad.push_back(in.getMData(k + m - 1, j + l, i));
								if (SHOW_VECTOR == true || verbose >= 2) cout << "TE" << in.getMData(k + m - 1, j + l, i) << "\t";
								div_count++;
							}
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					//avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result = (float(sum) / float(div_count));

					//avg_result_int = round((float(sum) / float((size - 1)*(size))));
					avg_result_int = round((float(sum) / float(div_count)));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum <<";Count:"<< div_count <<";Avg:" << avg_result_int << ") \t";

				}
				//else if (j == shape.height - 1) {
				else if (j == in.shape.height - 1) {

					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(                    k + m - 1, j + l-1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BE" << in.getMData(k + m - 1, j + l-1, i) << "\t";
						}
						if (verbose == 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == 0 ) {
					for (int l = 0; l < size ; l++) {
						for (int m = 0; m < size -1; m++) {
							//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							if ((j + l - 1) < in.getHeight()) {
								pad.push_back(in.getMData(k + m, j + l - 1, i));
								if (SHOW_VECTOR == true || verbose >= 2) cout << "LE" << in.getMData(k + m, j + l - 1, i) << "\t";
								div_count++;
							}
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					//avg_result = (float(sum) / float((size - 1)*(size)));
					//avg_result_int = round((float(sum) / float((size - 1)*(size))));
					avg_result = (float(sum) / float(div_count));

					avg_result_int = round((float(sum) / float(div_count)));


					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == in.shape.width - 1) {

				//else if ( k == shape.width - 1) {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(                    k + m -1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "RE" << in.getMData(k + m -1, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size)*(size - 1)));
					avg_result_int = round((float(sum) / float((size)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				
				else{
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size; m++) {
							//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							if ((k + m - 1) < in.getWidth() && (j + l - 1) < in.getHeight()) {
								pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
								if (SHOW_VECTOR == true || verbose >= 2) cout << in.getMData(k + m - 1, j + l - 1, i) << "\t";
								div_count++;
							}
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					//avg_result = (float(sum) / float((size)*(size)));
					//avg_result_int = round((float(sum) / float((size)*(size))));
					avg_result = (float(sum) / float(div_count));
					avg_result_int = round((float(sum) / float(div_count)));

					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				M[i*getWidth()*getHeight() + (j / stride) * getWidth() + (k / stride)] = avg_result_int;
				
				if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
			}
			if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
		}
		if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
	}


	return 0;
}

int Tensor::AvgPool2x2NoPad(Tensor & in, int size, int stride, int verbose) {
	vector<int> pad;
	int sum;
	float avg_result;
	float avg_result_int;

	//assert(in.getWidth() == this->getWidth());
	cout << "This Avg Pool : size = " << size << endl;
	cout << "This Avg Pool : stride = " << stride << endl;
	for (int i = 0; i < in.getDepth(); i++) {
		if (SHOW_VECTOR == true || verbose >= 2)cout << "page [" << i << "]" << endl;
		for (int j = 0; j < in.getHeight(); j = j + stride) {
			for (int k = 0; k < in.getWidth(); k = k + stride) {
				pad.clear();
				sum = 0;

				if ((j == 0 && k == 0)) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "LT" << in.getMData(k + m, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == shape.width - 1 && j == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "RT" << in.getMData(k + m - 1, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == shape.height - 1 && k == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BL" << in.getMData(k + m, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == shape.height - 1 && k == shape.width - 1) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BR" << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (verbose == 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "TE" << in.getMData(k + m - 1, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == shape.height - 1) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BE" << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (verbose == 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == 0) {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "LE" << in.getMData(k + m, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == shape.width - 1) {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "RE" << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size)*(size - 1)));
					avg_result_int = round((float(sum) / float((size)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}

				else {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size)*(size)));
					avg_result_int = round((float(sum) / float((size)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				M[i*getWidth()*getHeight() + (j / stride) * getWidth() + (k / stride)] = avg_result_int;

				if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
			}
			if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
		}
		if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
	}


	return 0;
}

int Tensor::AvgPoolPad(Tensor & in, int size, int stride, int verbose) {
	vector<int> pad;
	int sum;
	float avg_result;
	float avg_result_int;

	//assert(in.getWidth() == this->getWidth());
	cout << "This Avg Pool : size = " << size << endl;
	cout << "This Avg Pool : stride = " << stride << endl;
	for (int i = 0; i < in.getDepth(); i++) {
		if (SHOW_VECTOR == true || verbose >= 2)cout << "page [" << i << "]" << endl;
		for (int j = 0; j < in.getHeight(); j = j + stride) {
			for (int k = 0; k < in.getWidth(); k = k + stride) {
				pad.clear();
				sum = 0;

				if ((j == 0 && k == 0)) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "LT" << in.getMData(k + m, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == shape.width - 1 && j == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "RT" << in.getMData(k + m - 1, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == shape.height - 1 && k == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BL" << in.getMData(k + m, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == shape.height - 1 && k == shape.width - 1) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BR" << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (verbose == 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size - 1)));
					avg_result_int = round((float(sum) / float((size - 1)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == 0) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "TE" << in.getMData(k + m - 1, j + l, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (j == shape.height - 1) {
					for (int l = 0; l < size - 1; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "BE" << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (verbose == 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == 0) {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "LE" << in.getMData(k + m, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size - 1)*(size)));
					avg_result_int = round((float(sum) / float((size - 1)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				else if (k == shape.width - 1) {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size - 1; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << "RE" << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size)*(size - 1)));
					avg_result_int = round((float(sum) / float((size)*(size - 1))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}

				else {
					for (int l = 0; l < size; l++) {
						for (int m = 0; m < size; m++) {
							pad.push_back(in.getMData(k + m - 1, j + l - 1, i));
							if (SHOW_VECTOR == true || verbose >= 2) cout << in.getMData(k + m - 1, j + l - 1, i) << "\t";
						}
						if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
					}
					sum = accum(pad);
					avg_result = (float(sum) / float((size)*(size)));
					avg_result_int = round((float(sum) / float((size)*(size))));
					if (SHOW_VECTOR == true || verbose >= 2) cout << "\t\t\t\tplAvg[" << j << "][" << k << "] = " << avg_result << " (S:" << sum << ";Avg:" << avg_result_int << ") \t";

				}
				M[i*getWidth()*getHeight() + (j / stride) * getWidth() + (k / stride)] = avg_result_int;

				if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
			}
			if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
		}
		if (SHOW_VECTOR == true || verbose >= 2) cout << endl;
	}


	return 0;
}

int Tensor::getWidth() {
	return shape.width;
}
int Tensor::getHeight() {
	return shape.height;
}
int Tensor::getDepth() {
	return shape.depth;
}
int Tensor::getBatch() {
	return shape.batch;
}

void Tensor::setWidth(int width) {

	shape.width = width;
}

void Tensor::setHeight(int height) {

	shape.height = height;
}

void Tensor::setDepth(int depth) {
	shape.depth = depth;
}

void Tensor::setBatch(int batch) {
	shape.batch = batch;
}


bool Tensor::checkPrintDim4Alloc(int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::checkPrintDim4Alloc check & print Dim Before memory alloc(): input vector dimension <b, d, h x w> = [" << getBatch() << " , " << getDepth() << " , " << getHeight() << " , " << getWidth() << "]" << endl;
		cout << "dimension must be larget than 0, Alloc() function call is valid" << endl;
	}

	if (getWidth() <= 0) { cout << "Tensor::checkPrintDim4Alloc(): width is wrong = " << getWidth() << endl; return false; }
	if (getHeight()<= 0) { cout << "Tensor::checkPrintDim4Alloc(): height is wrong = " << getHeight() << endl; return false; }
	if (getDepth() <= 0) { cout << "Tensor::checkPrintDim4Alloc(): depth is wrong = " << getDepth() << endl; return false; }
	if (getBatch() < 0) { cout << "Tensor::checkPrintDim4Alloc(): group/batch is wrong = " << getBatch() << endl; return false; }

	return true;
}

void Tensor::ones(int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::ones() : height =" << getHeight() << endl;
		cout << "Tensor::ones() : width =" << getWidth() << endl;
		cout << "Tensor::ones() : depth =" << getDepth() << endl;
	}
	for (int m = 0; m < getDepth(); m++) {
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++)
				M[m*getWidth()*getHeight() + i * getWidth() + j] = 1;
		}
	}
	
	if (verbose >= 2) {
		for (int m = 0; m < getDepth(); m++) {
			cout << "channel[" << m << "]" << endl;
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++)
					cout << "M["<<i<<","<<j<<"]="<< M[m*getWidth()*getHeight() + i * getWidth() + j]<< "\t"; cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
}

void Tensor::zeros(int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::ones() : height =" << getHeight() << endl;
		cout << "Tensor::ones() : width =" << getWidth() << endl;
		cout << "Tensor::ones() : depth =" << getDepth() << endl;
	}
	for (int m = 0; m < getDepth(); m++) {
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++)
				M[m*getWidth()*getHeight() + i * getWidth() + j] = 0;
		}
	}

	if (verbose >= 2) {
		for (int m = 0; m < getDepth(); m++) {
			cout << "channel[" << m << "]" << endl;
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++)
					cout << "M[" << i << "," << j << "]=" << M[m*getWidth()*getHeight() + i * getWidth() + j] << "\t";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
}
void Tensor::centerOne(int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::ones() : height =" << getHeight() << endl;
		cout << "Tensor::ones() : width =" << getWidth() << endl;
		cout << "Tensor::ones() : depth =" << getDepth() << endl;
	}
	if ((getHeight() % 2 != 0) && (getWidth() % 2 != 0) && (getDepth() == 1))
		cout << "OK: The Dimension is x, y is odd, depth is 1";
	else
		cout << "ERROR: Tensor::centerOnes() The dimension is wrong. x, y is odd, depth is 1";
	
	cout << "Center position" << int(getWidth() / 2) << "," << int(getHeight() / 2) << endl;
	int setOneCount = 0;
	for (int m = 0; m < getDepth(); m++) {
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				if ((j == int(getWidth()/2)) && (i == int(getHeight() / 2))) {
					M[m*getWidth()*getHeight() + i * getWidth() + j] = 1;
					setOneCount++;
				}
				else
				{
					M[m*getWidth()*getHeight() + i * getWidth() + j] = 0;
				}
			}
		}
	}
	if (setOneCount != 1) cout << "ERROR: Tensor::centerOnes() The one was set not just ONCE." << endl;
	if (verbose >= 2) {
		for (int m = 0; m < getDepth(); m++) {
			cout << "channel[" << m << "]" << endl;
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++)
					cout << "M[" << i << "," << j << "]=" << M[m*getWidth()*getHeight() + i * getWidth() + j] << "\t";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;

	}
}

void Tensor::identity(int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::ones() : height =" << getHeight() << endl;
		cout << "Tensor::ones() : width =" << getWidth() << endl;
		cout << "Tensor::ones() : depth =" << getDepth() << endl;
	}
	for (int m = 0; m < getDepth(); m++) {
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				if (i == j) {
					M[m*getWidth()*getHeight() + i * getWidth() + j] = 1;
				}
				else
				{
					M[m*getWidth()*getHeight() + i * getWidth() + j] = 0;
				}
			}
		}
	}

	if (verbose >= 2) {
		for (int m = 0; m < getDepth(); m++) {
			cout << "channel[" << m << "]" << endl;
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++)
					cout << "M[" << i << "," << j << "]=" << M[m*getWidth()*getHeight() + i * getWidth() + j] << "\t";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;

	}
}

void Tensor::randGen(bool saturation_on, int mask_bit, int min, int max, int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::random() : height =" << getHeight() << endl;
		cout << "Tensor::random() : width =" << getWidth() << endl;
		cout << "Tensor::random() : depth =" << getDepth() << endl;
	}
	assert(max >= min);
	int delta = max - min;
	srand(unsigned int(time(0)));

	for (int m = 0; m < getDepth(); m++) {
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				int rand_temp = rand() % delta;
				//cout << "rand temp = " << rand_temp << "\t";
				M[m*getWidth()*getHeight() + i * getWidth() + j] = int(rand_temp) + min;
				//cout << "M [" << m << "][" << i << "][" << j << "]="<<M[m*getWidth()*getHeight() + i * getWidth() + j] <<"\t";
			}
			//cout << endl;
		}
	}

	if (verbose >= 2) {
		cout << endl<< "Tensor::random() : generated random matrix [d, h x w] = <" << getDepth()<<","<<getHeight()<<","<<getWidth()<<">" << endl;
		for (int m = 0; m < getDepth(); m++) {
			cout << "channel[" << m << "]" << endl;
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++)
					cout << "M[" << i << "," << j << "]=" << M[m*getWidth()*getHeight() + i * getWidth() + j] << "\t";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;

	}
}

void Tensor::randGen(int seed, bool saturation_on, int mask_bit, int min, int max, int verbose) {
	if (verbose >= 2) {
		cout << "Tensor::random() : height =" << getHeight() << endl;
		cout << "Tensor::random() : width =" << getWidth() << endl;
		cout << "Tensor::random() : depth =" << getDepth() << endl;
	}
	assert(max >= min);
	int delta = max - min;
	srand(seed);

	for (int m = 0; m < getDepth(); m++) {
		for (int i = 0; i < getHeight(); i++) {
			for (int j = 0; j < getWidth(); j++) {
				int rand_temp = rand() % delta;
				//cout << "rand temp = " << rand_temp << "\t";
				M[m*getWidth()*getHeight() + i * getWidth() + j] = int(rand_temp) + min;
				//cout << "M [" << m << "][" << i << "][" << j << "]="<<M[m*getWidth()*getHeight() + i * getWidth() + j] <<"\t";
			}
			//cout << endl;
		}
	}

	if (verbose >= 2) {
		cout << endl << "Tensor::random() : generated random matrix [d, h x w] = <" << getDepth() << "," << getHeight() << "," << getWidth() << ">" << endl;
		for (int m = 0; m < getDepth(); m++) {
			cout << "channel[" << m << "]" << endl;
			for (int i = 0; i < getHeight(); i++) {
				for (int j = 0; j < getWidth(); j++)
					cout << "M[" << i << "," << j << "]=" << M[m*getWidth()*getHeight() + i * getWidth() + j] << "\t";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;

	}
}


int Tensor::getItemsize() {

	return item_size; // defined in Storage.h
}

void Tensor::reshape() {
	cout << "reshape funciton"<<endl; 
}


int Tensor::accum(vector<int> &in) {
	return std::accumulate(in.begin(), in.end(), 0);
}
int Tensor::max(vector<int> & in) {
	return *std::max_element(in.begin(), in.end());
}
int Tensor::accum(vector<vector<int>> & in) {

	int temp = 0;
	for (int i = 0; i < in.size(); i++) {
		temp += accum(in[i]);
	}
	cout << "tensor pool sum " << temp << endl;

	return temp;
}

int Tensor::max(vector<vector<int>> &in) {
	vector<int> temp_pool;

	for (int i = 0; i < in.size(); i++) {
		temp_pool.push_back(this->max(in[i]));
	}

	//cout << "pool avg. " << accumulate(pool.begin(), pool.end(), 0) /float(pool.size())<< endl;
	cout << "tensor pool max." << *max_element(temp_pool.begin(), temp_pool.end()) << endl;
	return *max_element(temp_pool.begin(), temp_pool.end());
}


void Tensor::to_csv( std::string CSVFileName, std::string caseName, bool readIn, int verbose) {
	if (verbose >= 2) cout << "Tensor::toCSV()-- File Name: " << CSVFileName + ".csv" << endl;
	if (verbose >= 2) cout << "<x,y,z> = " << getWidth() << "," << getHeight() << "," << getDepth() << endl;
	string fileName;
	if (readIn == false)
		fileName = caseName + "\\out\\csv\\" + CSVFileName + ".csv";
	else
		fileName = "in\\" + caseName + "\\out\\csv\\" + CSVFileName + ".csv";

	std::ofstream file(fileName);
	for (int i = 0; i < getWidth(); ++i) {
		file << "T<" << i << ">";
		if (i != getWidth() - 1) file << ",";
	}
	file << endl;
	for (int z = 0; z < getDepth(); z++) {
		for (int y = 0; y < getHeight(); y++) {
			for (int x = 0; x < getWidth(); x++) {
				if (verbose >=2) cout << getMData(x, y, z) << ", ";
				file << getMData(x, y, z);
				if (x != getWidth() - 1) file << ",";
			}
			file << endl;
			if (verbose >= 2) cout << endl;
		}
	}
	file.close();
}

std::vector<std::pair<std::string, std::vector<int>>> Tensor::read_csv(std::string CSVFileName) {
	cout << "Tensor::read_csv()" << CSVFileName << endl;

	// Reads a CSV file into a vector of <string, vector<int>> pairs where
	// each pair represents <column name, column values>

	// Create a vector of <string, int vector> pairs to store the result
	std::vector<std::pair<std::string, std::vector<int>>> result;

	// Create an input filestream
	std::ifstream myFile(CSVFileName);

	// Make sure the file is open
	if (!myFile.is_open()) throw std::runtime_error("Could not open file");

	// Helper vars
	std::string line, colname;
	int val;

	// Read the column names
	if (myFile.good())
	{
		// Extract the first line in the file
		std::getline(myFile, line);

		// Create a stringstream from line
		std::stringstream ss(line);

		// Extract each column name
		while (std::getline(ss, colname, ',')) {

			// Initialize and add <colname, int vector> pairs to result
			result.push_back({ colname, std::vector<int> {} });
		}
	}

	// Read data, line by line
	while (std::getline(myFile, line))
	{
		// Create a stringstream of the current line
		std::stringstream ss(line);

		// Keep track of the current column index
		int colIdx = 0;

		// Extract each integer
		while (ss >> val) {

			// Add the current integer to the 'colIdx' column's values vector
			result.at(colIdx).second.push_back(val);

			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',') ss.ignore();

			// Increment the column index
			colIdx++;
		}
	}

	// Close file
	myFile.close();

	return result;
}


int Tensor::biasShiftAndRound(int wgtSum, int bias, int RShift_bit) {
	
	return shiftAndRound((wgtSum + bias), RShift_bit);

	//return shiftAndRoundPosNeg((wgtSum + bias), RShift_bit);

}

void Tensor::printNegHex(FILE *fptr, int x) {
	char pTmp[9];
	
	assert(fptr != NULL);
	
	if (x < 0) {
		sprintf_s(pTmp, "%02X", x);

		for (int i = 6; i < 8; i++) {
			//cout << pTmp[i];//Looping 5 times to print out [0],[1],[2],[3],[4]
			fprintf(fptr,"%c", pTmp[i]);
		}
		fprintf(fptr, " ");
	}
	else {
		fprintf(fptr, "%02X ", x&0xff);
	}
}


void Tensor::printNegHexWoSpace(FILE *fptr, int x) {
	char pTmp[9];

	assert(fptr != NULL);

	if (x < 0) {
		sprintf_s(pTmp, "%02x", x);

		for (int i = 6; i < 8; i++) {
			fprintf(fptr, "%c", pTmp[i]);
		}
	}
	else {
		fprintf(fptr, "%02x", x & 0xff);
	}
}

void Tensor::printNegHex32b(FILE *fptr, int x) {
	char pTmp[9];

	assert(fptr != NULL);

	if (x < 0) {
		sprintf_s(pTmp, "%08x", x);
		//printf(" Converted to %s\n\n", pTmp);

		for (int i = 0; i < 8; i++) {
			//cout << pTmp[i];//Looping 5 times to print out [0],[1],[2],[3],[4]

			//printf("%c", pTmp[i]);
			fprintf(fptr, "%c", pTmp[i]);
		}
		fprintf(fptr, " ");
	}
	else {
		//fprintf(fptr, "%02x ", x);
		fprintf(fptr, "%08x ", x);
	}
}

void Tensor::printNegHex32bWoSpace(FILE *fptr, int x) {
	char pTmp[9];

	assert(fptr != NULL);

	if (x < 0) {
		sprintf_s(pTmp, "%08x", x);
		//printf(" Converted to %s\n\n", pTmp);

		for (int i = 0; i < 8; i++) {
			//cout << pTmp[i];//Looping 5 times to print out [0],[1],[2],[3],[4]

			//printf("%c", pTmp[i]);
			fprintf(fptr, "%c", pTmp[i]);
		}
		//fprintf(fptr, " ");
	}
	else {
		//fprintf(fptr, "%02x ", x);
		fprintf(fptr, "%08x", x);
	}
}
void Tensor::print24bDataHexFormat(FILE *fptr, int x) {
	char pTmp[9];

	assert(fptr != NULL);

	if (x < 0) {
		sprintf_s(pTmp, "%06X", x);
		//printf(" Converted to %s\n\n", pTmp);

		for (int i = 2; i < 8; i++) {
			//cout << pTmp[i];//Looping 5 times to print out [0],[1],[2],[3],[4]
			//printf("%c", pTmp[i]);
			fprintf(fptr, "%c", pTmp[i]);
		}
		fprintf(fptr, " ");
	}
	else {
		fprintf(fptr, "%06X ", x);
	}
}

void Tensor::printNegHex(int x) {
	char pTmp[9];


	if (x < 0) {
		sprintf_s(pTmp, "%02X", x);
		printf("x");
		for (int i = 6; i < 8; i++) {
			printf("%c", pTmp[i]);
		}
		printf("");
	}
	else {
		printf("x%02x", x);
	}
}

int Tensor::str2Hex(string str) {
	int value;
	std::stringstream sstream;
	sstream << str;
	sstream >> std::hex >> value;
	//if (value == 128) // fixed a bug
	//	return -128;
	if (value >= 128)
		return value-256;
	else 
		return value;
}

int Tensor::str2Dec(string str) {
	int value;
	std::stringstream sstream;
	sstream << str;
	sstream >> value;
	if (value > 128)
		return value - 256;
	else
		return value;
}


//int Tensor::str2Hex(string str) {
//	int value;
//	std::stringstream sstream;
//	sstream << str;
//	sstream >> std::hex >> value;
//
//	return value;
//}

bool Tensor::operator += (Tensor &inTensor1) {
	Tensor res(inTensor1.getWidth(), inTensor1.getHeight(), inTensor1.getDepth());
	for (int k = 0; k < inTensor1.getDepth(); k++) {
		for (int i = 0; i < inTensor1.getHeight(); i++) {
			for (int j = 0; j < inTensor1.getWidth(); j++)
				M[j + i * inTensor1.getWidth() + k * inTensor1.getWidth() * inTensor1.getHeight()] =
				M[j + i * inTensor1.getWidth() + k * inTensor1.getWidth() * inTensor1.getHeight()] +
				inTensor1.getMData(j, i, k);
		}
	}
	return true;
}

Tensor operator + (Tensor &inTensor1, Tensor &inTensor2) {
	Tensor res(inTensor1.getWidth(), inTensor1.getHeight(), inTensor1.getDepth());
	for (int k = 0; k < inTensor1.getDepth(); k++) {
		for (int i = 0; i < inTensor1.getHeight(); i++) {
			for (int j = 0; j < inTensor1.getWidth(); j++)
				res.setMData(j,i,k,	inTensor1.getMData(j, i, k) + inTensor2.getMData(j, i, k));
		}
	}
	return res;
}

int Tensor::shiftPosNeg(int shift_in, int shift_pos) {
	if (shift_pos > 0) {
		if (shift_in > 0)
			return (shift_in << shift_pos);
		else if (shift_in < 0)
			return (-1 * ((-1 * shift_in) << shift_pos));
		else if (shift_in == 0)
			return  0;
		else {
			cout << "Tensor::shiftPosNeg():unexpected shift_in " << endl;
			exit(0);
		}

	}
	else if (shift_pos < 0){
		if (shift_in > 0)
			return shiftAndRoundInVec(shift_in, (-1*shift_pos));
		else if (shift_in < 0)
			return -1 * shiftAndRoundInVec((-1 * shift_in), (-1*shift_pos));
		else if (shift_in == 0)
			return  0;
		else {
			cout << "Tensor::shiftPosNeg():unexpected shift_in " << endl;
			exit(0);
		}
	}
	else if (shift_pos == 0)
		return shift_in;
	else {
		cout << "Tensor::shiftPosNeg():unexpected shift_pos " << endl;
		exit(0);
	}
		
}

void Tensor::writeMemoryBin(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false)
		s = caseName + "\\out\\bin\\" + vecName + ".bin";
	else
		s = "in\\" + caseName + "\\out\\bin\\" + vecName + ".bin";

	char* cstr = const_cast<char*>(s.c_str());
	errno_t err = fopen_s(&fptr, cstr, "w");

	if (err == 0 and SHOW_VECTOR == true)
	{
		if (debug_level > 1) cout << "Tensor::writeHexFormatOutputActivation.c: The file " << vecName << " was opened" << endl;
	}
	else
	{
		if (debug_level > 1) cout << "Tensor::writeHexFormatOutputActivation.c: The file " << vecName << " was NOT opened" << endl;
	}

	for (int l = 0; l < getDepth(); l++) {
		for (int i = 0; i < getHeight(); ++i) {
			for (int j = 0; j < getWidth(); j++) {
				this->printNegHexWoSpace(fptr, getMData(j, i, l));
				fprintf(fptr, "\n");
			}
		}
	}
	fclose(fptr);
}

void Tensor::writeMemoryDepthBin(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false)
		s = caseName + "\\out\\bin\\" + vecName + ".bin";
	else
		s = "in\\" + caseName + "\\out\\bin\\" + vecName + ".bin";

	char* cstr = const_cast<char*>(s.c_str());
	errno_t err = fopen_s(&fptr, cstr, "w");

	if (err == 0 and SHOW_VECTOR == true)
	{
		if (debug_level > 1) cout << "Tensor::writeHexFormatOutputActivation.c: The file " << vecName << " was opened" << endl;
	}
	else
	{
		if (debug_level > 1) cout << "Tensor::writeHexFormatOutputActivation.c: The file " << vecName << " was NOT opened" << endl;
	}
	for (int i = 0; i < getHeight(); ++i) {
		for (int j = 0; j < getWidth(); j++) {
			for (int l = 0; l < getDepth(); l++) {				
				this->printNegHexWoSpace(fptr, getMData(j, i, l));
				fprintf(fptr, "\n");
			}
		}
	}
	fclose(fptr);
}

void Tensor::writeMemoryBin32b(string vecName, string caseName, bool readIn) {

	FILE *fptr;
	string s;
	if (readIn == false)
		s = caseName + "\\out\\csv\\" + vecName + ".bin";
	else
		s = "in\\" + caseName + "\\out\\csv\\" + vecName + ".bin";
	char* cstr = const_cast<char*>(s.c_str());
	fopen_s(&fptr, cstr, "w");

	for (int l = 0; l < getDepth(); l++) {
		for (int i = 0; i < getHeight(); ++i) {
			for (int j = 0; j < getWidth(); j++) {
				this->printNegHex32bWoSpace(fptr, getMData(j, i, l));
			}
		}
	}
	fclose(fptr);
}
