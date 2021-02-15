#include "pch.h"

#include "NEngine.h"

void NE_Vec_Conv_Step1(ComputeGraph *computeGraph, RegBlock register_file, std::vector<std::vector<std::string>> param_vector_setting, string case_dir, int verbose) {
	// Readin SWI configuration
	int pooling_enable = register_file.getRegValueByParamName(param_vector_setting, "STEP_3_POOLING_ENABLE");
	int pooling_type = register_file.getRegValueByParamName(param_vector_setting, "STEP_3_POOLING_TYPE");// 0 -- max pooling; 1--average pooling
	int pooling_stride = register_file.getRegValueByParamName(param_vector_setting, "STEP_3_POOLING_STRIDE");
	int step1_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_BYPASS");
	int step2_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_BYPASS");
	int step3_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_3_BYPASS");
	int step1_relu_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_NON_LINEARITY_BYPASS");
	int step2_relu_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_NON_LINEARITY_BYPASS");
	int step3_relu_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_3_NON_LINEARITY_BYPASS");
	int residual_sum = register_file.getRegValueByParamName(param_vector_setting, "RESIDUAL_SUM_ENABLE");

	int step2_padding_value = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_PAD_VALUE");
	int step2_padding_enable = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_PAD_ENABLE");
	int step1_relu_threshold = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_RELU_X_THRESHOLD");


	std::map <string, StepInputTensor> NEInData;
	std::map <string, StepDataFeatureMap <WeightVector>> NEFeatureMapData;
	std::map <string, std::vector<Featuremap *>> NEFeatureMap;

	cout << "NEEDLE Engine Architecture ------- STARDARD CONVOLUTION ------ " << endl;
	
	assert(computeGraph->Step_2.in_channel_num == computeGraph->Step_2.in_channel_num_per_group);
	int step_1_w, step_1_h, step_1_c;
	int step_2_w, step_2_h, step_2_c;
	int step_3_w, step_3_h, step_3_c;
	int step_1_wgt_w, step_1_wgt_h, step_1_wgt_c;
	int step_3_wgt_w, step_3_wgt_h, step_3_wgt_c;

	step_1_w = computeGraph->Step_1.in_x;
	step_1_h = computeGraph->Step_1.in_y;
	step_1_c = computeGraph->Step_1.out_channel_num;
	step_1_wgt_w = computeGraph->Step_1.in_channel_num;
	step_1_wgt_h = computeGraph->Step_1.out_channel_num;
	step_1_wgt_c = 1;

	step_2_w = computeGraph->Step_2.in_x;
	step_2_h = computeGraph->Step_2.in_y;
	step_2_c = computeGraph->Step_2.out_channel_num;

	step_3_w = computeGraph->Step_3.in_x;
	step_3_h = computeGraph->Step_3.in_y;
	step_3_c = computeGraph->Step_3.out_channel_num;

	step_3_wgt_w = computeGraph->Step_3.in_channel_num;
	step_3_wgt_h = computeGraph->Step_3.out_channel_num;
	step_3_wgt_c = 1;


	cout << endl << "Step 1 Starting " << endl;
	vector<Featuremap *> Step1_CalcResult;

	// ___________________________________________
	//
	//	1. Input Vector
	//	2. Weight Vector
	//	3. Bias Vector
	//	4. Output Vector
	// ___________________________________________

	// 1. Input Vector
		InputVolume inVec(computeGraph->Step_1.in_x, computeGraph->Step_1.in_y, computeGraph->Step_1.in_channel_num, 0, 5, Random, 1);
		if (verbose >= 2 && SHOW_VECTOR == true)
			inVec.printVec("NE_Std_Conv(): Input Tensor");
		inVec.writeFormatInputActivation("InputActivation", case_dir, false);
		inVec.writeDecimalFormatInputActivation("InputActivation", case_dir);
		inVec.to_csv("invec_c", case_dir);
		inVec.to_csv("input_c", case_dir);
		inVec.writeMemoryDepthBin("invec", case_dir); // Verilog Vector Bin

	// 2. Weight Vector
	int step1_weights_rand_min = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_WEIGHTS_MIN");
	int step1_weights_rand_max = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_WEIGHTS_MAX"); // 8-bit weights, read from cfg. 

	WeightVector weights_Step1(step_1_wgt_w, step_1_wgt_h, step_1_wgt_c, Random, step1_weights_rand_min, step1_weights_rand_max, 1);

	if (verbose >= 2 && SHOW_VECTOR == true)
		weights_Step1.printVec("NE_Std_Conv(): Weight Tensor");

	weights_Step1.writeFormatWeights("Conv_1x1_Weights_Step1", case_dir, /*readIn*/ false);
	weights_Step1.writeDecimalFormatWeights("Conv_1x1_LUTWeights_Step1", case_dir, /*readIn*/ false);
	weights_Step1.to_csv("weight_1_c", case_dir);
	weights_Step1.writeMemoryBin("weight_1_c", case_dir); // Verilog Vector Bin

	// 3. Bias Vector
	int step1_bias_rand_min = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_BIAS_MIN");
	int step1_bias_rand_max = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_BIAS_MAX");
	if (verbose > 1 && SHOW_VECTOR == true) {
		cout << "Step 1 Bias Min = " << step1_bias_rand_min << endl;
		cout << "Step 1 Bias Max = " << step1_bias_rand_max << endl;
	}

	BiasShift bias_Step1(/*height*/ computeGraph->Step_1.out_channel_num, RandomBias,
		/*min*/step1_bias_rand_min, /*max*/step1_bias_rand_max, /*min shift*/4, /*max shift*/6, /*verbose*/verbose);

	if (verbose > 1 && SHOW_VECTOR == true) {
		bias_Step1.printVec("main(): Step 1 Bias Tensor Gen'ed Random");
	}

	bias_Step1.writeHexFormatBiasShift("Conv_1x1_Bias_Shift_Step1", case_dir, false);
	bias_Step1.writeDecimalFormatBiasShift("Conv_1x1_Bias_Shift_Step1", case_dir, false);
	bias_Step1.to_csv("bias_1_c", case_dir);
	bias_Step1.writeMemoryBin("bias_1_c", case_dir); // Verilog Vector Bin
	

	// 4 Output Vector -- Definition
		StepInputTensor Step1_Data = { inVec,weights_Step1, bias_Step1 };

		Featuremap hiddenLayer1(step_1_w, step_1_h, step_1_c, "hiddenLayer1");		// generate 1x1 conv at step 1
		Featuremap hiddenLayer1_preBias(step_1_w, step_1_h, step_1_c, "hiddenLayer1_preBias");		// generate 1x1 conv at step 1
		Featuremap hiddenLayer1_preShift(step_1_w, step_1_h, step_1_c, "hiddenLayer1_preShift");		// generate 1x1 conv at step 1
		Featuremap hiddenLayer1_preReLU(step_1_w, step_1_h, step_1_c, "hiddenLayer1_preReLU");		// generate 1x1 conv at step 1

		hiddenLayer1.copyTensor(inVec);

		cout << endl << "Step 1 Completed" << endl;

		hiddenLayer1.writeHexFormatOutputActivation("OutputActivation_Step1", case_dir, false);
		hiddenLayer1.writeDecimalFormatOutputActivation("OutputActivation_Step1", case_dir, false);
		hiddenLayer1.to_csv("stage_1_out_c", case_dir);
		hiddenLayer1.to_csv("stage_1_out_c", case_dir);
		//hiddenLayer1.writeMemoryBin("stage_1_out_c", case_dir); // Verilog Vector Bin


	// 4 Output Vector -- Computation
		// Conv 1x1
		hiddenLayer1_preBias.convPointWiseBias_Step1(inVec, weights_Step1, bias_Step1, register_file, param_vector_setting, true, 7, CONV_ONLY, step1_relu_threshold, verbose);
		if (SHOW_VECTOR == true && verbose > 1) hiddenLayer1_preBias.printVec("hiddenLayer1_preBias");
		hiddenLayer1_preBias.writeHexFormatOutputActivation("AccumOutput_Step1", case_dir);
		hiddenLayer1_preBias.to_csv("AccumOutput_Step1", case_dir);
		//Step1_CalcResult.push_back(&hiddenLayer1_preBias);
		hiddenLayer1_preBias.writeMemoryBin("stage_1_out_c", case_dir); // Verilog Vector Bin


		hiddenLayer1_preShift.convPointWiseBias_Step1(inVec, weights_Step1, bias_Step1, register_file, param_vector_setting, true, 7, CONV_BIAS, step1_relu_threshold, verbose);
		if (SHOW_VECTOR == true && verbose > 1) hiddenLayer1_preShift.printVec("hiddenLayer1_preShift");
		hiddenLayer1_preShift.writeFormatOutputActivation32b("AccumOutput+Bias_Step1", case_dir);
		hiddenLayer1_preShift.to_csv("AccumOutput+Bias_Step1", case_dir);
		//Step1_CalcResult.push_back(&hiddenLayer1_preShift);


		hiddenLayer1_preReLU.convPointWiseBias_Step1(inVec, weights_Step1, bias_Step1, register_file, param_vector_setting, true, 7, CONV_BIAS_SHIFT, step1_relu_threshold, verbose);
		if (SHOW_VECTOR == true && verbose > 1) hiddenLayer1_preReLU.printVec("hiddenLayer1_preReLU");
		hiddenLayer1_preReLU.writeHexFormatOutputActivation("AccumOutput+Bias+Shift_Step1", case_dir);
		hiddenLayer1_preReLU.to_csv("AccumOutput+Bias+Shift_Step1");
		//Step1_CalcResult.push_back(&hiddenLayer1_preReLU);
		//int step1_relu_bypass = register_file.getRegValueByParamName(param_vector_setting, "STEP_1_NON_LINEARITY_BYPASS");;
		assert(step1_relu_bypass == 0 || step1_relu_bypass == 1);
		if (step1_relu_bypass == 0)
			hiddenLayer1.convPointWiseBias_Step1(inVec, weights_Step1, bias_Step1, register_file, param_vector_setting, true, 7, CONV_BIAS_SHIFT_RELU, step1_relu_threshold, verbose);
		else if (step1_relu_bypass == 1)
			hiddenLayer1.copyTensor(hiddenLayer1_preReLU);

		Step1_CalcResult.push_back(&hiddenLayer1);
		hiddenLayer1.to_csv("stage_1_out_c", case_dir);

	
	
	//convGroupData convG_data_struct;
	//convG_data_struct = register_file.calculateConvDimension(param_vector_setting, 0);
	//int step2_weights_rand_min = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_WEIGHTS_MIN");
	//int step2_weights_rand_max = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_WEIGHTS_MAX");
	//GWeight weights_Step2(3, 3, convG_data_struct.in_chan_num_per_group * convG_data_struct.out_chan_num_per_group*convG_data_struct.group_num,
	//	/*group*/convG_data_struct.group_num, Random, step2_weights_rand_min, step2_weights_rand_max, 1);
	//weights_Step2.writeFormatWeights("Conv_3x3_Weights_Step2", case_dir, false);
	//weights_Step2.writeDecimalFormatWeights("Conv_3x3_Weights_Step2", case_dir, false);
	//weights_Step2.to_csv("weight_1_c", case_dir);
	//weights_Step2.writeMemoryBin("wgtvec", case_dir);

	//// Step 2 Pad Setting
	//int step2_bias_rand_min = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_BIAS_MIN");
	//int step2_bias_rand_max = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_BIAS_MAX");

	//BiasShift bias_Step2(/*height*/ computeGraph->Step_2.out_channel_num, RandomBias, /*min*/step2_bias_rand_min, /*max*/step2_bias_rand_max, /*min_shift*/9, /*max_shift*/10, /*verbose*/debug_level);
	//if (debug_level > 1) bias_Step2.printVec("_____________Stage 2 Bias Vector_____________");
	//bias_Step2.to_csv("bias_1_c", case_dir);
	//bias_Step2.writeHexFormatBiasShift("Conv_3x3_Bias_Shift_Step2", case_dir, false);
	//bias_Step2.writeDecimalFormatBiasShift("Conv_3x3_Bias_Shift_Step2", case_dir, false);
	//bias_Step2.writeMemoryBin("biasvec", case_dir);

	//Featuremap hiddenLayer2_GConv(computeGraph->Step_2.in_x, computeGraph->Step_2.in_y, computeGraph->Step_2.out_channel_num, "hiddenLayer2_stdConv");	// generate depthwise conv at step 2
	//Featuremap hiddenLayer2_preBias(computeGraph->Step_2.in_x, computeGraph->Step_2.in_y, computeGraph->Step_2.out_channel_num, "hiddenLayer2_preBias");		// generate 1x1 conv at step 1
	//Featuremap hiddenLayer2_preShift(computeGraph->Step_2.in_x, computeGraph->Step_2.in_y, computeGraph->Step_2.out_channel_num, "hiddenLayer2_preShift");		// generate 1x1 conv at step 1
	//Featuremap hiddenLayer2_preReLU(computeGraph->Step_2.in_x, computeGraph->Step_2.in_y, computeGraph->Step_2.out_channel_num, "hiddenLayer2_preReLU");		// generate 1x1 conv at step 1
	//Featuremap hiddenLayer2(computeGraph->Step_2.in_x, computeGraph->Step_2.in_y, computeGraph->Step_2.out_channel_num, "hiddenLayer2");		// generate 1x1 conv at step 1
	//hiddenLayer2.writeHexFormatOutputActivation("OutputActivation_Step2", case_dir, false);
	//hiddenLayer2.writeDecimalFormatOutputActivation("OutputActivation_Step2", case_dir, false);

	//int step2_pad_value = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_PAD_VALUE");
	//int step2_pad_enable = register_file.getRegValueByParamName(param_vector_setting, "STEP_2_PAD_ENABLE");


	//cout << "layer 2 dimension out <z,y,x> = " << computeGraph->Step_2.out_channel_num << "," << computeGraph->Step_2.out_y << "," << computeGraph->Step_2.out_x << endl;
	////Featuremap hiddenLayer2_GConv(computeGraph->Step_2.in_x, computeGraph->Step_2.in_y, computeGraph->Step_2.out_channel_num, "hiddenLayer2_stdConv");	// generate depthwise conv at step 2
	//////**** hiddenLayer2_stdConv.convStandard(hiddenLayer1, weights_Step2, /*Pad*/ 1, /*strideX*/ computeGraph->Step_2.stride_x_size,/*strideY*/computeGraph->Step_2.stride_y_size, 1);
	//cout << "hiddenLayer1 shape" << hiddenLayer1.getWidth() << "," << hiddenLayer1.getHeight() << "," << hiddenLayer1.getDepth() << endl;
	//hiddenLayer2_GConv.convGroupPad(hiddenLayer1, weights_Step2, bias_Step2, /*strideX*/ convG_data_struct.stride_x,/*strideY*/ convG_data_struct.stride_y,  /*Pad*/ 1,/*Pad Value*/ step2_padding_value,
	//	/*group*/ convG_data_struct.group_num, CONV_ONLY, "case00", /*verbose*/2);
	//if (SHOW_VECTOR == true && debug_level > 1)
	//	hiddenLayer2_GConv.printVec("layer 2 Group Convolution Only");
	//if (SHOW_VECTOR == true && debug_level > 1) weights_Step2.printVec("Step 2 weight");

	//hiddenLayer2_preBias.convGroupPad(hiddenLayer1, weights_Step2, bias_Step2, /*strideX*/ convG_data_struct.stride_x,/*strideY*/ convG_data_struct.stride_y,  /*Pad*/ 1,/*Pad Value*/ step2_padding_value,
	//	/*group*/ convG_data_struct.group_num, CONV_ONLY, "case00", /*verbose*/0);
	//hiddenLayer2_preBias.writeHexFormatOutputActivation("AccumOutput_Step2", case_dir);
	//hiddenLayer2_preBias.to_csv("AccumOutput_Step2", case_dir);

	//if (SHOW_VECTOR == true && debug_level > 1)
	//	hiddenLayer2_preBias.printVec("layer 2 Group Convolution Only");
	//hiddenLayer2_preBias.to_csv("stage_2_out_c_prebias", case_dir);

	//hiddenLayer2_preShift.convGroupPad(hiddenLayer1, weights_Step2, bias_Step2, /*strideX*/ convG_data_struct.stride_x,/*strideY*/ convG_data_struct.stride_y,  /*Pad*/ 1,/*Pad Value*/ step2_padding_value,
	//	/*group*/ convG_data_struct.group_num, CONV_BIAS, "case00", /*verbose*/0);
	//hiddenLayer2_preShift.writeFormatOutputActivation32b("AccumOutput+Bias_Step2", case_dir);
	//hiddenLayer2_preShift.to_csv("AccumOutput+Bias_Step2");

	//if (verbose > 1 && SHOW_VECTOR == true)
	//	hiddenLayer2_preShift.printVec("layer 2 Group Convolution + Bias");

	//hiddenLayer2_preReLU.convGroupPad(hiddenLayer1, weights_Step2, bias_Step2, /*strideX*/ convG_data_struct.stride_x,/*strideY*/ convG_data_struct.stride_y,  /*Pad*/ 1,/*Pad Value*/ step2_padding_value,
	//	/*group*/ convG_data_struct.group_num, CONV_BIAS_SHIFT, "case00", /*verbose*/0);
	//hiddenLayer2_preReLU.writeHexFormatOutputActivation("AccumOutput+Bias+Shift_Step2", case_dir);
	//hiddenLayer2_preReLU.to_csv("AccumOutput+Bias+Shift_Step2");

	//if (verbose > 1 && SHOW_VECTOR == true)
	//	hiddenLayer2_preReLU.printVec("layer 2 Group Convolution + Bias + Shift");


	//// FIXME: 8-bit saturation
	//assert(step2_relu_bypass == 0 || step2_relu_bypass == 1);
	//if (step2_relu_bypass == 0) {
	//	hiddenLayer2.convGroupPad(hiddenLayer1, weights_Step2, bias_Step2, /*strideX*/ convG_data_struct.stride_x,/*strideY*/ convG_data_struct.stride_y,  /*Pad*/ 1,/*Pad Value*/ step2_padding_value,
	//		/*group*/ convG_data_struct.group_num, CONV_BIAS_SHIFT_RELU, "case00", /*verbose*/0);
	//	hiddenLayer2.ReLUXTensor(6);
	//}
	//else if (step2_relu_bypass == 1)
	//	hiddenLayer2.convGroupPad(hiddenLayer1, weights_Step2, bias_Step2, /*strideX*/ convG_data_struct.stride_x,/*strideY*/ convG_data_struct.stride_y,  /*Pad*/ 1,/*Pad Value*/ step2_padding_value,
	//		/*group*/ convG_data_struct.group_num, CONV_BIAS_SHIFT, "case00",/*verbose*/0);

	//hiddenLayer2.to_csv("stage_2_out_c", case_dir);
}

