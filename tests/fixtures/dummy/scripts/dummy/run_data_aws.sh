
if [ -f "${MAD_DATAHOME}/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12.onnx" ]; then
    echo "${MAD_DATAHOME}/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12.onnx is present"
    echo "performance: $RANDOM samples_per_second"
else
    echo "${MAD_DATAHOME}/bert-large-uncased_L_24_H_1024_A_16_V_30528_S_512_Dp_0.1_optimized_layer_norm_opset12.onnx is NOT present"
    exit 1
fi


