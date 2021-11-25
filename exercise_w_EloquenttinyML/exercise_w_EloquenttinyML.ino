#include <EloquentTinyML.h>
#include "exercisemodel.h"


#define DEBUG 0


namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input0 = nullptr;
  TfLiteTensor* model_input1 = nullptr;
  TfLiteTensor* model_input2 = nullptr;
  TfLiteTensor* model_output = nullptr;
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize]; 
}  



void setup() {
  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;


  model = tflite::GetModel(exercisemodel);


#if DEBUG
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
       model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }else{
    error_reporter->Report( "El modelo soportado es : %d" , TFLITE_SCHEMA_VERSION);
  }
#endif 

  static tflite::ops::micro::AllOpsResolver resolver;


  Serial.print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
#if DEBUG
  error_reporter->Report("\n\n\n\n\n\n\n\n\n\n\n\n\n\nAún no hay problemas");
#endif 


  static tflite::MicroInterpreter static_interpreter(
      model, 
      resolver, 
      tensor_arena, 
      kTensorArenaSize,
      error_reporter);
      
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  
#if DEBUG
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1); 
  }
#endif


  model_input0 = interpreter->input(0);
  model_input1 = interpreter->input(1);
  model_input2 = interpreter->input(2);
  model_output = interpreter->output(0);


#if DEBUG
  error_reporter->Report("Input information:");
  error_reporter->Report("Name of input: %s ", model_input0->name); 
  error_reporter->Report("Number of dimensions: %d ", model_input0->dims->size);
  error_reporter->Report("Dim 1 size: %d ", model_input0->dims->data[0]);
  error_reporter->Report("Dim 2 size: %d ", model_input0->dims->data[1]);
  error_reporter->Report("Input type: %d \n", model_input0->type);

  error_reporter->Report("Input information:");
  error_reporter->Report("Name of input: %s ", model_input1->name); 
  error_reporter->Report("Number of dimensions: %d ", model_input1->dims->size);
  error_reporter->Report("Dim 1 size: %d ", model_input1->dims->data[0]);
  error_reporter->Report("Dim 2 size: %d ", model_input1->dims->data[1]);
  error_reporter->Report("Input type: %d \n", model_input1->type);

  error_reporter->Report("Input information :");
  error_reporter->Report("Name of input: %s ", model_input2->name); 
  error_reporter->Report("Number of dimensions: %d ", model_input2->dims->size);
  error_reporter->Report("Dim 1 size: %d ", model_input2->dims->data[0]);
  error_reporter->Report("Dim 2 size: %d ", model_input2->dims->data[1]);
  error_reporter->Report("Input type: %d \n", model_input2->type);


  error_reporter->Report("Output information :");
  error_reporter->Report("Name of Output: %s ", model_output->name); 
  error_reporter->Report("Number of dimensions: %d ", model_output->dims->size);
  error_reporter->Report("Dim 1 size: %d ", model_output->dims->data[0]);
  error_reporter->Report("Dim 2 size: %d ", model_output->dims->data[1]);
  error_reporter->Report("Input type: %d ", model_output->type);

#endif 


  // He aquí la magía:

  Serial.println("|-----------------------TensorFlowLite-ESP32-----------------------|");
  Serial.println("|-------------------------EXERCISE DATASET-------------------------|");
  Serial.println("|-----------Prediction-ESP32----------|---------Real label---------|");
  
  int features = 5; 
  int samples = 15;
  for (int i = 0; i < samples; i++) {
    for (int u = 0; u < features; u++) {
      model_input0->data.f[u] = x_test[i][0][u];
      model_input1->data.f[u] = x_test[i][1][u];
      model_input2->data.f[u] = x_test[i][2][u];
    }
    
    // Estas líneas de código son para comparar con las etiquetas reales
    
    float real1 = labels[i][0];
    float real2 = labels[i][1];
    float real3 = labels[i][2];

    
    TfLiteStatus invoke_status = interpreter->Invoke();
  
#if DEBUG
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_test);
    return;
  }
#endif
 
      // Estos son los valores de salida:
      
      float out_1 = model_output->data.f[0];
      float out_2 = model_output->data.f[1];
      float out_3 = model_output->data.f[2];
  
  
      Serial.printf("|     %.5f   %.5f   %.5f     |       ", out_1, out_2, out_3); 
      Serial.printf("%.1f   %.1f   %.1f      |\n", real1, real2, real3);

      
#if DEBUG
    error_reporter->Report("out_1: %f", out_1);
    error_reporter->Report("out_2: %f", out_2);
    error_reporter->Report("out_3: %f", out_3);
#endif

  }
  Serial.println("|-------------------------------------|----------------------------|");

}

void loop() {}

//by Sandro Ormeño
