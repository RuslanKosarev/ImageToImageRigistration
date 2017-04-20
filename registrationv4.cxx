#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>
#include <itkEuler3DTransform.h>
#include <itkTranslationTransform.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkImageMomentsCalculator.h>
#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkLBFGSBOptimizerv4.h>

#include <itkImageRegistrationMethodv4.h>
#include <itkTranslationTransform.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSubtractImageFilter.h>

#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkPath.h"
#include "agtkCommandLineArgumentParser.h"
#include "agtkInitializeOptimizer.h"
#include "agtkInitializeTransform.h"

/*
https://itk.org/Doxygen/html/Examples_2RegistrationITKv4_2ImageRegistration1_8cxx-example.html
*/

using namespace agtk;

const unsigned int    Dimension = 3;
typedef FloatImage3D  FixedImageType;
typedef FloatImage3D  MovingImageType;
typedef  float        PixelType;

typedef itk::IdentityTransform<double, Dimension> IdentityTransformType;
typedef itk::Transform< double, Dimension > TransformType;

FloatImage3D::Pointer imageTransform(FloatImage3D::Pointer image, TransformType::Pointer transform);

class CommandIterationUpdate : public itk::Command
{
public:
  typedef CommandIterationUpdate   Self;
  typedef itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro(Self);
protected:
  CommandIterationUpdate() {};
public:
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  typedef const OptimizerType*                               OptimizerPointer;
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
    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
};

int main(int argc, char *argv[])
{
  /** Create a command line argument parser. */
  agtk::CommandLineArgumentParser::Pointer parser = agtk::CommandLineArgumentParser::New();
  parser->SetCommandLineArguments(argc, argv);

  std::string fixedFile;
  parser->GetValue("-fixed", fixedFile);

  std::string movingFile;
  parser->GetValue("-moving", movingFile);

  std::string outputFile;
  parser->GetValue("-output", outputFile);

  size_t numberOfIterations = 100;
  parser->GetValue("-iter", numberOfIterations);

  bool useEstimator = false;
  parser->GetValue("-estimator", useEstimator);

  std::cout << "number of iterations " << numberOfIterations << std::endl;
  std::cout << "       use estimator " << useEstimator << std::endl;

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
  //  
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  typedef itk::MeanSquaresImageToImageMetricv4 <FixedImageType, MovingImageType >    MetricType;
//typedef itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, itk::TranslationTransform< double, Dimension >>    RegistrationType;
  typedef itk::ImageRegistrationMethodv4<FixedImageType, MovingImageType, itk::Euler3DTransform<double>> RegistrationType;

  MetricType::Pointer         metric = MetricType::New();

  RegistrationType::Pointer   registration = RegistrationType::New();
  registration->SetMetric(metric);
  typedef itk::LinearInterpolateImageFunction<FixedImageType, double > FixedLinearInterpolatorType;
  typedef itk::LinearInterpolateImageFunction<MovingImageType, double > MovingLinearInterpolatorType;
  FixedLinearInterpolatorType::Pointer fixedInterpolator = FixedLinearInterpolatorType::New();
  MovingLinearInterpolatorType::Pointer movingInterpolator = MovingLinearInterpolatorType::New();
  metric->SetFixedInterpolator(fixedInterpolator);
  metric->SetMovingInterpolator(movingInterpolator);

  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);

  //-------------------------------------------------------------------------------------------------------
  // initialize transform
  typedef itk::BinaryThresholdImageFilter<FloatImage3D, BinaryImage3D> ThresholdImageFilterType;
  ThresholdImageFilterType::Pointer fixedThreshold = ThresholdImageFilterType::New();
  fixedThreshold->SetInput(fixedImage);
  fixedThreshold->SetLowerThreshold(50);
  fixedThreshold->SetUpperThreshold(FloatLimits::max());
  fixedThreshold->Update();

  typedef itk::BinaryThresholdImageFilter<FloatImage3D, BinaryImage3D> ThresholdImageFilterType;
  ThresholdImageFilterType::Pointer movingThreshold = ThresholdImageFilterType::New();
  movingThreshold->SetInput(movingImage);
  movingThreshold->SetLowerThreshold(50);
  movingThreshold->SetUpperThreshold(FloatLimits::max());
  movingThreshold->Update();

  // moment calculators
  typedef itk::ImageMomentsCalculator<BinaryImage3D>  ImageCalculatorType;
  ImageCalculatorType::Pointer fixedImageCalculator = ImageCalculatorType::New();
  fixedImageCalculator->SetImage(fixedThreshold->GetOutput());
  fixedImageCalculator->Compute();

  // moment calculators
  typedef itk::ImageMomentsCalculator<BinaryImage3D>  ImageCalculatorType;
  ImageCalculatorType::Pointer movingImageCalculator = ImageCalculatorType::New();
  movingImageCalculator->SetImage(movingThreshold->GetOutput());
  movingImageCalculator->Compute();

  FixedImageType::PointType fixedLandmark;
  if ( parser->ArgumentExists("-fixedlandmark") ) {
    parser->GetITKValue("-fixedlandmark", fixedLandmark);
  }
  else {
    fixedLandmark = fixedImageCalculator->GetCenterOfGravity();
  }

  MovingImageType::PointType movingLandmark;

  if ( parser->ArgumentExists("-movinglandmark") ) {
    parser->GetITKValue("-movinglandmark", movingLandmark);
  }
  else {
    movingLandmark = movingImageCalculator->GetCenterOfGravity();
  }
 
  std::cout << " fixed landmark " << fixedLandmark << std::endl;
  std::cout << "moving landmark " << movingLandmark << std::endl;

  //-------------------------------------------------------------------------------------------------------
  //
  typedef agtk::InitializeTransform<double> InitializeTransformType;
  InitializeTransformType::Pointer initializeTransform = InitializeTransformType::New();
//initializeTransform->SetTypeOfTransform(Transform::Translation);
  initializeTransform->SetTypeOfTransform(Transform::Euler3D);
  initializeTransform->SetCenter(fixedImageCalculator->GetCenterOfGravity());
  initializeTransform->SetTranslation(movingLandmark - fixedLandmark);
  initializeTransform->Update();
  initializeTransform->PrintReport();

  itk::Transform<double>::Pointer movingInitialTransform = initializeTransform->GetTransform();

  if (parser->ArgumentExists("-initial")) {
    MovingImageType::Pointer output = imageTransform(movingImage, movingInitialTransform->GetInverseTransform());

    std::string fileName;
    parser->GetValue("-initial", fileName);

    std::cout << "initial file " << fileName << std::endl;
    if (!writeImage(output, fileName)) {
      return EXIT_FAILURE;
    }
  }

  registration->SetFixedInitialTransform(IdentityTransformType::New());
  registration->SetMovingInitialTransform(movingInitialTransform);

  typedef agtk::InitializeOptimizer InitializeOptimizerType;
  InitializeOptimizerType::Pointer initializeOptimizer = InitializeOptimizerType::New();
  initializeOptimizer->Update();
  initializeOptimizer->RegularStepGradientDescentOptimizer->AddObserver(itk::IterationEvent(), CommandIterationUpdate::New());
  initializeOptimizer->RegularStepGradientDescentOptimizer->SetNumberOfIterations(numberOfIterations);
  if ( useEstimator ) {
    typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
    ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
    scalesEstimator->SetMetric(metric);
    scalesEstimator->SetTransformForward(true);
    initializeOptimizer->RegularStepGradientDescentOptimizer->SetScalesEstimator(scalesEstimator);
    initializeOptimizer->RegularStepGradientDescentOptimizer->SetDoEstimateLearningRateOnce(true);
  }

  InitializeOptimizerType::OptimizerPointer optimizer = initializeOptimizer->GetOptimizer();
  registration->SetOptimizer(optimizer);

  const unsigned int numberOfLevels = 1;
  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize(1);
  shrinkFactorsPerLevel[0] = 1;

  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize(1);
  smoothingSigmasPerLevel[0] = 1;

  registration->SetNumberOfLevels(numberOfLevels);
  registration->SetSmoothingSigmasPerLevel(smoothingSigmasPerLevel);
  registration->SetShrinkFactorsPerLevel(shrinkFactorsPerLevel);
  try {
    registration->Update();
    std::cout << "Optimizer stop condition: " << registration->GetOptimizer()->GetStopConditionDescription() << std::endl;
  }
  catch (itk::ExceptionObject & err) {
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  TransformType::ConstPointer transform = registration->GetTransform();

  std::cout << "results" << std::endl;
  std::cout << "parameters    = " << transform->GetParameters() << std::endl;
  std::cout << "iterations    = " << optimizer->GetCurrentIteration() << std::endl;
  std::cout << "metric value  = " << optimizer->GetValue() << std::endl;

  if (parser->ArgumentExists("-output")) {
    typedef itk::CompositeTransform<double, Dimension> CompositeTransformType;
    CompositeTransformType::Pointer transform = CompositeTransformType::New();
    transform->AddTransform(movingInitialTransform);
    transform->AddTransform(registration->GetModifiableTransform());

    typedef itk::ResampleImageFilter<MovingImageType, FixedImageType >  ResampleFilterType;
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetInput(movingImage);
    resampler->SetTransform(transform);
    resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(fixedImage->GetOrigin());
    resampler->SetOutputSpacing(fixedImage->GetSpacing());
    resampler->SetOutputDirection(fixedImage->GetDirection());
    resampler->SetDefaultPixelValue(0);
    try {
      resampler->Update();
    }
    catch (itk::ExceptionObject & err) {
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

    std::string fileName;
    parser->GetValue("-output", fileName);
    std::cout << "output file " << fileName << std::endl;
    if (!writeImage(resampler->GetOutput(), fileName)) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

FloatImage3D::Pointer imageTransform(FloatImage3D::Pointer image, itk::Transform<double,3,3>::Pointer transform)
{
  typedef itk::ImageDuplicator<FloatImage3D> DuplicatorType;
  DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(image);
  duplicator->Update();
  FloatImage3D::Pointer output = duplicator->GetOutput();

  FloatImage3D::PointType origin = transform->TransformPoint(output->GetOrigin());
  output->SetOrigin(origin);

  return output;
}
