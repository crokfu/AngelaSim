
#include "pch.h"
#include "NEngine.h"
#include "InputVolume.h"
#include "WeightVector.h"
#include "GWeight.h"
#include "BiasShift.h"
#include "NeedleDataStruct.h"
#include "ComputeGraph.h"
#include <windows.h>
#include <stdlib.h>
#include <direct.h>
#include <stdio.h>

static string sw_ver = "1.0";

static RegBlock register_file("CSR");

using namespace std;

int debug_level;
Logger logFile;

void needleEngine(ConvMethod vectorType, GenTensorMethod gen_tensor_method, ComputeGraph *computeGraph, std::vector<std::vector<std::string>> param_vector_setting, bool regresionEnable,
	string RegressionDir, bool Algorithm_Test = false, string out_dir = "Output", string case_dir = "case00", int verbose = 1);

int main(int argc, char* argv[])
{

	std::ifstream configFile;
	bool python_enable = false;
	string inputFileName;
	string configFileName = "config.txt";
	const char * logFileName = "log.txt";
	string outputFileName = "output.txt";

	std::vector<std::vector<string>> args;
	debug_level = 2;
	bool readIn = true;
	int prog_debug_level = 1;

	// ****************************
	//
	//     1. Prepare Files & Logs
	//
	// ****************************

	ofstream outFile(outputFileName);
	args = ParseConfigFile(configFileName);
	vector<string> inFileNames;
	inFileNames = readParameters(args);
	string output_file_name;
	string home_dir;
	string regress_cfg_dir;
	string regress_cfg_file;
	string conv_type_step2;
	string play_ground_en;
	string gen_vector_method;
	string cfg_dir = "cfg";
	string in_dir = "in";
	string out_dir = "out";
	string case_dir = "case00";
	string algo_test_en_readin;
	int rand_seed;
	logFile.openLog(logFileName);

	print_version(sw_ver);
	bool recursive_test = false;
	bool mobileface = true;
	int needle_mode = 0;

	bool Algorithm_Test;
	string regress_config_dir;
	// ---------------------------------------------
	// 
	//  1. read sim configuration from file
	// 
	// ---------------------------------------------

	std::cout << "_______________ config.txt Parameters _________________" << endl << endl;

	for (int i = 0; i < args.size(); i++) {

		if (my_string_insensitive_compare(args[i][0].c_str(), "log") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "log file : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			logFileName = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "HomeDir") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Home Directory : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			home_dir = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "Verbose") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Verbose debugging level : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			debug_level = atoi(args[i][1].c_str());
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "RegConfigDir") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Regression Directory : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			regress_cfg_dir = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "RegConfigFile") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Regression File : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			regress_cfg_file = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "output") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "result output file : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			output_file_name = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "PlayGround") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "playground Enable : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			play_ground_en = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "ConvType") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Step 2 Conv Type : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			conv_type_step2 = args[i][1].c_str();
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Step 2 Conv Type :" << setw(30) << std::left << conv_type_step2 << std::endl;
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "GenVectorMethod") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Generate Vector Method : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			gen_vector_method = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "CaseDir") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Case Directory : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			case_dir = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "InDir") == 0) {
			if (debug_level >= 1) std::cout << setw(30) << std::left << "Input Directory : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			in_dir = args[i][1].c_str();

		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "CfgDir") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Configuration Directory : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			cfg_dir = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "OutDir") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Output Directroy : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			out_dir = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "AlgoTest") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Algorithm Test : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			algo_test_en_readin = args[i][1].c_str();
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "seed") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Random Seed : " << setw(30) << std::left << args[i][1].c_str() << std::endl;
			string str_int_temp;
			str_int_temp = args[i][1].c_str();
			std::string::size_type sz;   // alias of size_t

			rand_seed = std::stoi(args[i][1].c_str(), &sz);

		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "RecursiveTest") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Regression Mode : " << setw(30) << std::left << args[i][1].c_str() << std::endl;

			if (my_string_insensitive_compare(args[i][1].c_str(), "Yes") == 0)
				recursive_test = true;
			else
				recursive_test = false;
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "MobileFace") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "MobileFace Mode : " << setw(30) << std::left << args[i][1].c_str() << std::endl;

			if (my_string_insensitive_compare(args[i][1].c_str(), "Yes") == 0)
				mobileface = true;
			else
				mobileface = false;
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Mobileface = " << setw(30) << std::left << mobileface << std::endl;
		}
		else if (my_string_insensitive_compare(args[i][0].c_str(), "CfgDir") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Regression Config Dir : " << setw(30) << std::left << args[i][1].c_str() << std::endl;

			regress_config_dir = args[i][1].c_str();
		}

		else if (my_string_insensitive_compare(args[i][0].c_str(), "NeedleMode") == 0) {
			if (debug_level >= 2) std::cout << setw(30) << std::left << "Needle Mode : " << setw(30) << std::left << args[i][1].c_str() << endl << setw(50) << std::right << "1-single block; 2-regression; 3-mobileface" << std::endl;
			string str_int_temp;
			str_int_temp = args[i][1].c_str();
			std::string::size_type sz;   // alias of size_t
			int mode = std::stoi(args[i][1].c_str(), &sz);

			if (mode == 0)
				needle_mode = SingleBlock;
			else if (mode == 1)
				needle_mode = RegressionFile;
			else if (mode == 2)
				needle_mode = RegressionConfig;
			else if (mode == 3)
				needle_mode = MobileFace;
			else {
				std::cout << "Mode setting is incorrect." << endl;
				exit(0);
			}
		}
	}

	if (strcmp(play_ground_en.c_str(), "true") == 0)
		//play_ground(register_file, param_vector_setting);
		play_ground();

	if (argc == 1) {
		printf("Usage: Heimdall.exe "
			"[-in <activatation input>] "
			"[-config <configuration file>]"
			"[-log <log file>]"
			"[-v or -verbose <verbose number>] \n\n\n");

		return 0;
	}

	// command interpret

	for (int i = 1; i < argc; ++i) {
		if (debug_level >= 2) cout << "argv[" << i << "] = " << argv[i] << endl;

		if (!strcmp(argv[i], "-in")) {
			inputFileName = argv[++i];
		}
		else if (!strcmp(argv[i], "-config")) {
			configFileName = argv[++i];
		}
		else if (!strcmp(argv[i], "-log")) {
			logFileName = argv[++i];
		}
		else if (!strcmp(argv[i], "-verbose")) {
			debug_level = std::stoi(argv[++i]);
		}
		else if (!strcmp(argv[i], "-v")) {
			debug_level = std::stoi(argv[++i]);
		}

	}

	std::cout << endl << "Needle Command argument inputs: \n" << endl << endl;
	std::cout << "configFile = " << configFileName << endl;
	std::cout << "inputFile = " << inputFileName << endl;
	std::cout << "logFile = " << logFileName << endl << endl;

	if (prog_debug_level > debug_level)
		debug_level = prog_debug_level;

	Featuremap bufferFMap(128, 128, 64, "OutFeatureMap");
	Featuremap * intermediateFeatureMap = new Featuremap(128, 128, 128, "InterMediateMap");
	std::string block_dir = "block_0";
	GenTensorMethod gen_tensor_method;

	ConvMethod convType_at_Step1 = ECONV;
	ConvMethod convType_at_Step2;
	ConvMethod convType_at_Step3 = ECONV;

	if (my_string_insensitive_compare(conv_type_step2.c_str(), "StandardConv") == 0)
		convType_at_Step2 = (ConvMethod)0x1;
	else if (my_string_insensitive_compare(conv_type_step2.c_str(), "GroupConv") == 0)
		convType_at_Step2 = (ConvMethod)0x0;
	else if (my_string_insensitive_compare(conv_type_step2.c_str(), "DepthwiseConv") == 0)
		convType_at_Step2 = (ConvMethod)0x2;
	else if (my_string_insensitive_compare(conv_type_step2.c_str(), "VECTGEN") == 0)
		convType_at_Step2 = (ConvMethod)0x6;
	else {
		std::cout << "Error: ConvMethod not defined" << endl << endl;
		exit(0);
	}


	if (my_string_insensitive_compare(algo_test_en_readin.c_str(), "true") == 0)
		Algorithm_Test = true;
	else if (my_string_insensitive_compare(algo_test_en_readin.c_str(), "false") == 0)
		Algorithm_Test = false;


	if (my_string_insensitive_compare(gen_vector_method.c_str(), "Random") == 0)
		gen_tensor_method = (GenTensorMethod)0x3;
	else if (my_string_insensitive_compare(gen_vector_method.c_str(), "ReadFile") == 0)
		gen_tensor_method = (GenTensorMethod)0x0;
	else if (my_string_insensitive_compare(gen_vector_method.c_str(), "RegressConfig") == 0)
		gen_tensor_method = (GenTensorMethod)0x4;
	else {
		std::cout << "Error: Vector Generation Method not defined" << endl << endl;
		exit(0);
	}

	string needle_cfg;
	string needle_emulate_cfg_src;
	string needle_emulate_cfg_dest;
	string needle_cfg_src;
	string param_map_src;
	string param_map_dest;
	string cfgFile;
	string cfgFile_dest;

	string csr_map_src;
	string csr_map_dest;
	string py_needle_step1_src;
	string py_needle_step2_src;
	string py_needle_step3_src;
	string py_needle_src;
	string compare_pyc_src;
	string compare_in_out_src;
	string cp2in_src;
	string needle_random_src;
	string needle_readin_src;

	string py_needle_step1_dest;
	string py_needle_step2_dest;
	string py_needle_step3_dest;
	string py_needle_dest;
	string compare_pyc_dest;
	string compare_in_out_dest;
	string cp2in_dest;
	string needle_random_dest;
	string needle_readin_dest;
	string gen_mem_src;
	string gen_mem_dest;
	string clear_in_src;
	string clear_in_dest;


	if (needle_mode == SingleBlock) {
		// ****************************
		//
		//     2. CSR Setup
		//
		// ****************************
		out_dir = "out";
		string output_dir = home_dir + "\\" + case_dir + "\\" + out_dir;
		string input_dir = home_dir + "\\" + case_dir + "\\" + in_dir;
		string config_dir = home_dir + "\\" + case_dir + "\\" + cfg_dir;

		std::cout << "______ Start Build File Structure _______" << endl;
		std::cout << "output directory " << output_dir << endl;
		std::cout << "input directory " << input_dir << endl;
		std::cout << "config directory " << config_dir << endl;

		clearCaseDirDecOut(output_dir);
		clearCaseDirHexOut(output_dir);
		clearCaseDirHMapOut(output_dir);
		clearCaseDirCSVOut(output_dir);

		//removeDirectory(home_dir);
		string case_single_block_dir = joinPath(home_dir, case_dir);

		if (createCaseDir(case_single_block_dir, false) != 0) {
			std::cout << case_single_block_dir << " Case directory created successfully" << endl << endl;
		}
		else {
			std::cout << case_single_block_dir << " Case directory creation failed" << endl << endl;
			exit(0);
		}


		// setup register memory map, register to field map and register physical storage space
		std::vector<std::vector<std::string>> csr_mm_readin = register_file.readNeuroEngineCSRMap("NeuroEngineCSRMap.txt");
		register_file.buildNeuroEngineCSRMap(csr_mm_readin);

		// setup parameter to register map
		std::vector<std::vector<std::string>> param_reg_readin = register_file.readNeuroEngineParamFromConfigFile("needle_engine_param_map.txt");
		register_file.buildNeuroEngineParamMap(param_reg_readin);

		// read in neuron engine parameter data and set CSR register space
		// param_vector_setting is the parameters read-in from needle_engine.cfg.txt 
		// and can also be used as program variables settings directly.
		std::vector<std::vector<std::string>> param_vector_setting;
		if (gen_tensor_method == READFILE)
			param_vector_setting = register_file.readParamDataFromConfigFile("in\\needle_engine_cfg.txt"); 		// regression single block test needle_engine_cfg
		else if (gen_tensor_method == GENRAND)
			//param_vector_setting = register_file.readParamDataFromConfigFile("C:\\Local_Documents\\Needle_Emulator\\C_arg\\needle_engine_cfg.txt"); // nelson single block test needle_engine_cfg
			param_vector_setting = register_file.readParamDataFromConfigFile(home_dir + "\\needle_engine_cfg.txt"); // nelson single block test needle_engine_cfg
		else if (gen_tensor_method == READCONFIG)
			param_vector_setting = register_file.readParamDataFromConfigFile(home_dir + "\\needle_engine_cfg.txt"); // nelson single block test needle_engine_cfg

		register_file.configNeuroEngineParamsWithFile(param_vector_setting);

		// ****************************
		//
		//     3. Compute Graph Setup and Compute Resource Calculate
		//
		// ****************************
		int step2_pad_enable = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_PAD_ENABLE", 1);
		if (debug_level >= 2) std::cout << "Step 2 pad enable " << step2_pad_enable << endl;
		ComputeGraph * computeGraph = new ComputeGraph(register_file, param_vector_setting);
		computeGraph->printComputationGraph(register_file, param_vector_setting);
		computeGraph->writeComputationGraph("computational_graph", block_dir, readIn);
		if (register_file.getRegValueByParamName(param_vector_setting, "STEP_1_FC_LAYER_EN") == 1)
			convType_at_Step1 = FC;
		if (register_file.getRegValueByParamName(param_vector_setting, "STEP_3_FC_LAYER_EN") == 1)
			convType_at_Step3 = FC;
		computeGraph->genInWeightCfg(convType_at_Step1, "weights-1", 1, debug_level);
		computeGraph->genInWeightCfg(convType_at_Step2, "weights-2", 2, debug_level);
		computeGraph->genInWeightCfg(convType_at_Step3, "weights-3", 3, debug_level);
		string RegressionDir = "";
		string output_full_path = output_dir;
		string needle_top_py_src, needle_top_py_dest;
		if (register_file.getRegValueByParamName(param_vector_setting, "VMU_ENABLE", 1) == 1) {
			convType_at_Step2 = (ConvMethod)0x5;
		}



		//// ****************************
		////
		////     4. Computation Core Parts
		////
		//// ****************************
		needleEngine(convType_at_Step2, gen_tensor_method, computeGraph, param_vector_setting, false /*regressionEnable*/, RegressionDir, false, out_dir, case_dir, debug_level); // needle compute

		if (readIn == false) {
			needle_cfg = case_single_block_dir + "/out/needle_engine_cfg.txt";
			needle_emulate_cfg_src = case_single_block_dir + "/cfg/needle_engine_cfg.txt";
			if (gen_tensor_method == READFILE)
				needle_cfg_src = home_dir + "in/needle_engine_cfg.txt";
			else if (gen_tensor_method == GENRAND)
				needle_cfg_src = case_single_block_dir + "/needle_engine_cfg.txt";
			param_map_src = case_single_block_dir + "/cfg/needle_engine_param_map.txt";
			cfgFile = case_single_block_dir + "/cfg/NeuroEngineConfigFile.txt";
			csr_map_src = case_single_block_dir + "/cfg/NeuroEngineCSRMap.txt";
			py_needle_step1_src = case_single_block_dir + "/out/csv/needle_step1.py";
			py_needle_step2_src = case_single_block_dir + "/out/csv/needle_step2.py";
			py_needle_step3_src = case_single_block_dir + "/out/csv/needle_step3.py";
			compare_pyc_src = home_dir + "\\torch\\compare_pyc.py";
			compare_in_out_src = home_dir + "\\torch\\compare_in_out.py";
			needle_top_py_src = home_dir + "\\torch\\needle.py";

		}
		else {
			needle_cfg = case_single_block_dir + "\\out\\needle_engine_cfg.txt";
			needle_emulate_cfg_src = home_dir + "\\config.txt";
			needle_emulate_cfg_dest = case_single_block_dir + "\\cfg\\config.txt";
			if (gen_tensor_method == READFILE)
				needle_cfg_src = home_dir + "\\in\\needle_engine_cfg.txt";
			else if (gen_tensor_method == GENRAND)
				needle_cfg_src = home_dir + "\\needle_engine_cfg.txt";

			//needle_cfg_src = home_dir + "\\in\\needle_engine_cfg.txt";

			param_map_src = home_dir + "\\needle_engine_param_map.txt";
			param_map_dest = case_single_block_dir + "\\cfg\\needle_engine_param_map.txt";

			cfgFile = home_dir + "\\needle_engine_cfg.txt";
			cfgFile_dest = home_dir + "\\cfg\\needle_engine_cfg.txt";

			csr_map_src = home_dir + "\\NeuroEngineCSRMap.txt";
			csr_map_dest = case_single_block_dir + "\\cfg\\NeuroEngineCSRMap.txt";

			py_needle_step1_src = home_dir + "\\torch\\needle_step1.py";
			py_needle_step2_src = home_dir + "\\torch\\needle_step2.py";
			py_needle_step3_src = home_dir + "\\torch\\needle_step3.py";
			compare_pyc_src = home_dir + "\\torch\\compare_pyc.py";
			compare_in_out_src = home_dir + "\\torch\\compare_in_out.py";
			cp2in_src = home_dir + "\\torch\\compare_in_out.py";
			needle_random_src = home_dir + "\\torch\\cnn.py";
			needle_readin_src = home_dir + "\\torch\\cnn_readin.py";
			gen_mem_src = home_dir + "\\torch\\gen_mem.py";
			clear_in_src = home_dir + "\\torch\\clear_in.py";
			needle_top_py_src = home_dir + "\\torch\\needle.py";

			py_needle_step1_dest = case_single_block_dir + "\\out\\csv\\needle_step1.py";
			py_needle_step2_dest = case_single_block_dir + "\\out\\csv\\needle_step2.py";
			py_needle_step3_dest = case_single_block_dir + "\\out\\csv\\needle_step3.py";
			compare_pyc_dest = case_single_block_dir + "\\out\\csv\\compare_pyc.py";
			compare_in_out_dest = case_single_block_dir + "\\out\\csv\\compare_in_out.py";
			cp2in_dest = case_single_block_dir + "\\out\\csv\\cp2in.py";
			needle_random_dest = case_single_block_dir + "\\out\\csv\\cnn.py";
			needle_readin_dest = case_single_block_dir + "\\out\\csv\\cnn_readin.py";
			gen_mem_dest = case_single_block_dir + "\\out\\csv\\gen_mem.py";
			clear_in_dest = case_single_block_dir + "\\out\\csv\\clear_in.py";
			needle_top_py_dest = case_single_block_dir + "\\out\\csv\\needle.py";
		}

		if (copyFile(const_cast<char*>(needle_cfg_src.c_str()), const_cast<char*> (needle_cfg.c_str())) &&
			copyFile(const_cast<char*>(needle_emulate_cfg_src.c_str()), const_cast<char*> (needle_emulate_cfg_dest.c_str())) &&
			copyFile(const_cast<char*> (param_map_src.c_str()), const_cast<char*> (param_map_dest.c_str())) &&
			copyFile(const_cast<char*> (cfgFile.c_str()), const_cast<char*> (cfgFile_dest.c_str())) && copyFile(const_cast<char*> (csr_map_src.c_str()), const_cast<char*> (csr_map_dest.c_str())))
			if (debug_level >= 2) std::cout << "SUCCESS : copy Needle config files succeed" << endl;
			else
				if (debug_level >= 2) std::cout << "ERROR : copy Needle config files failed" << endl;

		if (copyFile(const_cast<char*>(py_needle_step1_src.c_str()), const_cast<char*> (py_needle_step1_dest.c_str())) &&
			copyFile(const_cast<char*>(py_needle_step2_src.c_str()), const_cast<char*> (py_needle_step2_dest.c_str())) &&
			copyFile(const_cast<char*> (py_needle_step3_src.c_str()), const_cast<char*> (py_needle_step3_dest.c_str())) &&
			copyFile(const_cast<char*> (compare_pyc_src.c_str()), const_cast<char*> (compare_pyc_dest.c_str())) &&
			copyFile(const_cast<char*> (compare_in_out_src.c_str()), const_cast<char*> (compare_in_out_dest.c_str())) &&
			copyFile(const_cast<char*> (needle_random_src.c_str()), const_cast<char*> (needle_random_dest.c_str())) &&
			copyFile(const_cast<char*> (needle_readin_src.c_str()), const_cast<char*> (needle_readin_dest.c_str())) &&
			copyFile(const_cast<char*> (clear_in_src.c_str()), const_cast<char*> (clear_in_dest.c_str())) &&
			copyFile(const_cast<char*> (gen_mem_src.c_str()), const_cast<char*> (gen_mem_dest.c_str()))&&
			copyFile(const_cast<char*> (needle_top_py_src.c_str()), const_cast<char*> (needle_top_py_dest.c_str()))
			)
			std::cout << "SUCCESS : copy Needle pytorch script succeed" << endl;
		else {
			if (debug_level >= 2)
				std::cout << "ERROR : copy Needle pytorch script failed" << endl;
		}
		if (python_enable == true) {
			cout << "_________________________________ Python Envir Setting ____________________________________" << endl << endl;
			cout << "Current Directory : \t";
			string command = "cd";
			system(command.c_str());
			cout << "Home Directory : \t" << home_dir << endl;
			string py_dir = home_dir + "\\case00\\out\\csv";
			command += " " + py_dir;
			cout << command << endl;
			system(command.c_str());
			//cout << "----------------- " << endl;
			auto path = fs::current_path(); //getting path
			fs::current_path(path); //setting path
			_chdir("case00\\out\\csv");
			cout << "Current Directory : \t";
			//command = "cd";
			//system(command.c_str());
			command = "ipython check.py";
			system(command.c_str());
		}
	}
	else
	{
		cout << "Error: Neither recursive Test nor Mobile Face Mode is enabled!";
	}
	logFile.closeLog();
	return 1;
}

void needleEngine(ConvMethod convType_at_Step2, GenTensorMethod gen_tensor_method, ComputeGraph *computeGraph, std::vector<std::vector<std::string>> param_vector_setting, bool regressionEnable,
	string RegressionDir, bool Algorithm_Test, string out_dir, string case_dir, int verbose) {

	cout << " Algorithm_Test = " << Algorithm_Test << endl;
	if (verbose >= 2 && debug_level >= 2) cout << "Needle Engine Dir (needleEngine) " << RegressionDir << endl;

	if (verbose >= 2) cout << "Step 2 Conv Type =" << convType_at_Step2 << "(GCONV = 0x0, STDCONV = 0x1, DWSCONV = 0x2, ECONV = 0x3)" << endl;

	//--------------------- GROUP CONVOLUTION ----------------------------
	if (gen_tensor_method == GENRAND && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 1) {
		NE_GConv(computeGraph, register_file, param_vector_setting, "Data", case_dir, /*verbose*/ verbose);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 1 && regressionEnable == true) {
		NE_GConv_RG(computeGraph, register_file, param_vector_setting, RegressionDir, case_dir, /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 1 && regressionEnable == false) {
		NE_GConv_RF(computeGraph, register_file, param_vector_setting, case_dir, /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READCONFIG && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 1 && regressionEnable == false) {
		NE_GConv_RFG(computeGraph, register_file, param_vector_setting, out_dir, case_dir, /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == GENRAND && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 0) {
		NE_GConv_NoPad(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 0 && regressionEnable == false) {
		NE_GConv_NoPad_RF(computeGraph, register_file, param_vector_setting, case_dir, /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 0 && regressionEnable == true) {
		NE_GConv_NoPad_RG(computeGraph, register_file, param_vector_setting, RegressionDir, case_dir, /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READCONFIG && convType_at_Step2 == GCONV && Algorithm_Test == false && computeGraph->Step_2.padding == 0 && regressionEnable == false) {
		cout << "create GConv NoPad RFG function" << endl;
		NE_GConv_NoPad_RFG(computeGraph, register_file, param_vector_setting, out_dir, case_dir, /*verbose*/ debug_level);
	}

	//--------------------- STANDARD CONVOLUTION ----------------------------
	else if (gen_tensor_method == GENRAND && convType_at_Step2 == STDCONV && Algorithm_Test == false) {
		NE_Std_Conv(computeGraph, register_file, param_vector_setting, "case00",  /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == STDCONV && Algorithm_Test == true) {
		NE_Std_Conv_RF_Algo(computeGraph, register_file, param_vector_setting, "case00",/*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == STDCONV && Algorithm_Test == false) {

		NE_Std_Conv_RF(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == GENRAND && convType_at_Step2 == STDCONV_VEC && Algorithm_Test == false) {
		NE_Vec_Conv_Step1(computeGraph, register_file, param_vector_setting, "case00",  /*verbose*/ debug_level);
	}

	//--------------------- DEPTHWISE SEPARABLE CONVOLUTION ----------------------------
	else if (gen_tensor_method == READFILE && convType_at_Step2 == DWSCONV && Algorithm_Test == true) {
		NE_Dephwise_Conv_RF_Algo(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == READFILE && convType_at_Step2 == DWSCONV && Algorithm_Test == false) {
		NE_Dephwise_Conv_ReadFile(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}
	else if (gen_tensor_method == GENRAND && convType_at_Step2 == DWSCONV && Algorithm_Test == false) {
		NE_Dephwise_Conv_RandomGen(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}
	//--------------------- VMU ----------------------------

	else if (gen_tensor_method == GENRAND && convType_at_Step2 == VMU) {
		VecMultUnit(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}

	else if (gen_tensor_method == READFILE && convType_at_Step2 == VMU) {
		cout << "TO BE Developed" << endl;
		exit(EXIT_FAILURE);
		//VecMultUnit(computeGraph, register_file, param_vector_setting, "case00", /*verbose*/ debug_level);
	}
	else {
		cout << "ERROR: Needle Engine Vector Generation Mode is undefined. Check GenMethod and ConvType in config File." << endl;
		exit(EXIT_FAILURE);
	}



}


