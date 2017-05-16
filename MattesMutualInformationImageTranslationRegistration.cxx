// https://itk.org/Doxygen/html/Examples_2RegistrationITKv4_2ImageRegistration4_8cxx-example.html

#include <itkImageRegistrationMethodv4.h>
#include <itkTranslationTransform.h>
#include <itkEuler3DTransform.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkCenteredTransformInitializer.h>
#include <itkCommand.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkCommandLineArgumentParser.h"
#include "agtkTransformImageFilter.h"

class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro(Self);
protected:
  CommandIterationUpdate() {};
public:
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  typedef   const OptimizerType *                            OptimizerPointer;
  void Execute(itk::Object *caller, const itk::EventObject & event) ITK_OVERRIDE
  {
    Execute((const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event) ITK_OVERRIDE
  {
    OptimizerPointer optimizer = static_cast<OptimizerPointer>(object);
    if (!itk::IterationEvent().CheckEvent(&event)) {
      return;
    }
    std::cout << optimizer->GetCurrentIteration() << "   ";
    std::cout << optimizer->GetValue() << "   ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
};

const    unsigned int    Dimension = 3;
typedef  float           PixelType;
typedef itk::Image< PixelType, Dimension >  FixedImageType;
typedef itk::Image< PixelType, Dimension >  MovingImageType;
typedef itk::TranslationTransform< double, Dimension >         TransformType;
typedef itk::RegularStepGradientDescentOptimizerv4<double>     OptimizerType;
typedef itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, TransformType> RegistrationType;
typedef itk::MattesMutualInformationImageToImageMetricv4<FixedImageType, MovingImageType > MetricType;

using namespace agtk;

int main(int argc, char *argv[])
{
  agtk::CommandLineArgumentParser::Pointer parser = agtk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  std::string fixedFile;
  parser->GetValue("-fixed", fixedFile);

  std::string movingFile;
  parser->GetValue("-moving", movingFile);

  std::string outputFile;
  parser->GetValue("-output", outputFile);

  FixedImageType::Pointer fixedImage = FixedImageType::New();
  if (!readImage(fixedImage, fixedFile))
    return EXIT_FAILURE;

  std::cout << " fixed image " << fixedImage->GetLargestPossibleRegion().GetSize() << std::endl;
  std::cout << fixedImage->GetOrigin() << std::endl;
  std::cout << fixedImage->GetSpacing() << std::endl;
  std::cout << fixedImage->GetDirection() << std::endl;

  MovingImageType::Pointer movingImage = MovingImageType::New();
  if (!readImage(movingImage, movingFile))
    return EXIT_FAILURE;

  std::cout << "moving image " << movingImage->GetLargestPossibleRegion().GetSize() << std::endl;
  std::cout << movingImage->GetOrigin() << std::endl;
  std::cout << movingImage->GetSpacing() << std::endl;
  std::cout << movingImage->GetDirection() << std::endl;

  //-------------------------------------------------------------------------------------------------------
  // initialize initial transform
  typedef itk::Euler3DTransform<double> Euler3DTransformType;
  Euler3DTransformType::Pointer movingInitialTransform = Euler3DTransformType::New();

  typedef itk::CenteredTransformInitializer <Euler3DTransformType, FixedImageType, MovingImageType> TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  initializer->SetTransform(movingInitialTransform);
  initializer->SetFixedImage(fixedImage);
  initializer->SetMovingImage(movingImage);
  initializer->GeometryOn();
  initializer->InitializeTransform();

  std::cout << "moving initial transform" << std::endl;
  std::cout << movingInitialTransform->GetParameters() << std::endl;

  if (parser->ArgumentExists("-initial")) {
    typedef TransformImageFilter<MovingImageType> TransformImageFilterType;
    TransformImageFilterType::Pointer transformImage = TransformImageFilterType::New();
    transformImage->SetInput(movingImage);
    transformImage->SetTransform(movingInitialTransform->GetInverseTransform());
    try {
      transformImage->Update();
    }
    catch (itk::ExceptionObject & excep) {
      std::cerr << excep << std::endl;
      return EXIT_FAILURE;
    }

    std::string fileName;
    parser->GetValue("-initial", fileName);

    std::cout << "initial file " << fileName << std::endl;
    if (!writeImage(transformImage->GetOutput(), fileName)) {
      return EXIT_FAILURE;
    }
  }

  //-------------------------------------------------------------------------------------------------------
  OptimizerType::Pointer      optimizer = OptimizerType::New();
  RegistrationType::Pointer   registration = RegistrationType::New();

  typedef itk::IdentityTransform<double, Dimension> IdentityTransformType;
  registration->SetFixedInitialTransform(IdentityTransformType::New());

  registration->SetMovingInitialTransform(movingInitialTransform);
  registration->SetOptimizer(optimizer);
  MetricType::Pointer metric = MetricType::New();
  registration->SetMetric(metric);

  unsigned int numberOfBins = 24;
  metric->SetNumberOfHistogramBins(numberOfBins);
  metric->SetUseMovingImageGradientFilter(false);
  metric->SetUseFixedImageGradientFilter(false);

  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);
  optimizer->SetLearningRate(8.00);
  optimizer->SetMinimumStepLength(0.001);
  optimizer->SetNumberOfIterations(200);
  optimizer->ReturnBestParametersAndValueOn();
  optimizer->SetRelaxationFactor(0.8);

  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);
  const unsigned int numberOfLevels = 1;
  
  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize(1);
  shrinkFactorsPerLevel[0] = 1;

  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize(1);
  smoothingSigmasPerLevel[0] = 0;

  registration->SetNumberOfLevels(numberOfLevels);
  registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
  registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);
  RegistrationType::MetricSamplingStrategyType  samplingStrategy = RegistrationType::RANDOM;
  double samplingPercentage = 0.20;
  
  registration->SetMetricSamplingStrategy(samplingStrategy);
  registration->SetMetricSamplingPercentage(samplingPercentage);
  try {
    registration->Update();
    std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
  }
  catch (itk::ExceptionObject & err) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }
  TransformType::ParametersType finalParameters = registration->GetOutput()->Get()->GetParameters();

  std::cout << std::endl;
  std::cout << "Result = " << std::endl;
  std::cout << " final parameters = " << finalParameters << std::endl;
  std::cout << " Iterations    = " << optimizer->GetCurrentIteration() << std::endl;
  std::cout << " Metric value  = " << optimizer->GetValue() << std::endl;
  std::cout << " Stop Condition  = " << optimizer->GetStopConditionDescription() << std::endl;

  // output
  if (parser->ArgumentExists("-output")) {
    typedef itk::CompositeTransform<double, Dimension> CompositeTransformType;
    CompositeTransformType::Pointer transform = CompositeTransformType::New();
    transform->AddTransform(movingInitialTransform);
    transform->AddTransform(registration->GetModifiableTransform());

    typedef TransformImageFilter<MovingImageType> TransformImageFilterType;
    TransformImageFilterType::Pointer transformImage = TransformImageFilterType::New();
    transformImage->SetInput(movingImage);
    transformImage->SetTransform(transform->GetInverseTransform());
    try {
      transformImage->Update();
    }
    catch (itk::ExceptionObject & excep) {
      std::cerr << excep << std::endl;
      return EXIT_FAILURE;
    }

    std::string fileName;
    parser->GetValue("-output", fileName);

    std::cout << "output file " << fileName << std::endl;
    if (!writeImage(transformImage->GetOutput(), fileName)) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}