#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryMask3DMeshSource.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>
#include <itkEuler3DTransform.h>

#include "itkImageRegistrationMethod.h"
#include "itkTranslationTransform.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "agtkTypes.h"
#include "agtkIO.h"
#include "agtkPath.h"
#include "agtkCommandLineArgumentParser.h"

using namespace agtk;

typedef FloatImage3D  FixedImageType;
typedef FloatImage3D  MovingImageType;

const    unsigned int    Dimension = 3;
typedef  float           PixelType;

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
  typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
  typedef const OptimizerType*                     OptimizerPointer;
  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute((const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event)
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

  std::string fixedImageFile;
  parser->GetValue("-fixed", fixedImageFile);

  std::string movingImageFile;
  parser->GetValue("-moving", movingImageFile);

  std::string outputFile;
  parser->GetValue("-output", outputFile);

  FixedImageType::Pointer fixedImage = FixedImageType::New();
  if (!readImage(fixedImage, fixedImageFile))
    return EXIT_FAILURE;

  MovingImageType::Pointer movingImage = MovingImageType::New();
  if (!readImage(movingImage, movingImageFile))
    return EXIT_FAILURE;

  std::cout << " fixed image " << fixedImage->GetLargestPossibleRegion().GetSize() << std::endl;
  std::cout << fixedImage->GetSpacing() << std::endl;
  std::cout << fixedImage->GetDirection() << std::endl;

  std::cout << "moving image " << movingImage->GetLargestPossibleRegion().GetSize() << std::endl;
  std::cout << movingImage->GetSpacing() << std::endl;
  std::cout << movingImage->GetDirection() << std::endl;

  typedef itk::TranslationTransform< double, Dimension> TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  std::cout << transform->GetParameters() << std::endl;

  typedef itk::RegularStepGradientDescentOptimizer      OptimizerType;
  typedef itk::MeanSquaresImageToImageMetric <FixedImageType, MovingImageType > MetricType;
  typedef itk::LinearInterpolateImageFunction <MovingImageType, double> InterpolatorType;
  typedef itk::ImageRegistrationMethod <FixedImageType, MovingImageType> RegistrationType;

  MetricType::Pointer         metric = MetricType::New();

  OptimizerType::Pointer      optimizer = OptimizerType::New();
  OptimizerType::ScalesType scales(transform->GetNumberOfParameters());
  scales[0] = 1;
  scales[1] = 1;
  scales[2] = 1;
  std::cout << scales << std::endl;

  optimizer->SetScales(scales);

  InterpolatorType::Pointer   interpolator = InterpolatorType::New();
  RegistrationType::Pointer   registration = RegistrationType::New();

  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetInterpolator(interpolator);
  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);
  registration->SetFixedImageRegion(fixedImage->GetBufferedRegion());

  typedef RegistrationType::ParametersType ParametersType;
  ParametersType initialParameters(transform->GetNumberOfParameters());

  for (int n = 0; n < transform->GetNumberOfParameters(); ++n) {
    initialParameters[n] = transform->GetParameters()[n];
  }
  std::cout << initialParameters << std::endl;

  registration->SetInitialTransformParameters(initialParameters);
  optimizer->SetMaximumStepLength(1.000);
  optimizer->SetMinimumStepLength(0.001);
  optimizer->SetNumberOfIterations(100);

  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), observer);

  try {
    registration->Update();
  }
  catch (itk::ExceptionObject & err) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  ParametersType finalParameters = registration->GetLastTransformParameters();

  const double TranslationAlongX = finalParameters[0];
  const double TranslationAlongY = finalParameters[1];
  const double TranslationAlongZ = finalParameters[2];
  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double bestValue = optimizer->GetValue();

  std::cout << "Result = " << std::endl;
  std::cout << optimizer->GetStopConditionDescription() << std::endl;
  std::cout << " Translation X = " << TranslationAlongX << std::endl;
  std::cout << " Translation Y = " << TranslationAlongY << std::endl;
  std::cout << " Translation Z = " << TranslationAlongZ << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue << std::endl;

  const unsigned int WindowRadius = 2;
  typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
  typedef itk::ConstantBoundaryCondition<FloatImage3D> BoundaryConditionType;
  typedef itk::WindowedSincInterpolateImageFunction<FloatImage3D, WindowRadius, WindowFunctionType, BoundaryConditionType, double> SincInterpolatorType;
  SincInterpolatorType::Pointer sincinterpolator = SincInterpolatorType::New();

  typedef itk::ResampleImageFilter <MovingImageType, FixedImageType> ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetInterpolator(sincinterpolator);
  resampler->SetTransform(registration->GetOutput()->Get());
  resampler->SetOutputOrigin(movingImage->GetOrigin());
  resampler->SetOutputDirection(movingImage->GetDirection());
  resampler->SetSize(movingImage->GetLargestPossibleRegion().GetSize());
  resampler->SetOutputSpacing(movingImage->GetSpacing());
  resampler->SetInput(movingImage);

  writeImage(resampler->GetOutput(), outputFile);


  if (parser->ArgumentExists("-diff")) {
    std::string outputFile;
    parser->GetValue("-diff", outputFile);

    //difference
    typedef itk::ResampleImageFilter <MovingImageType, FixedImageType> ResampleFilterType;
    ResampleFilterType::Pointer resampler1 = ResampleFilterType::New();
    resampler1->SetInterpolator(sincinterpolator);
    resampler1->SetOutputOrigin(fixedImage->GetOrigin());
    resampler1->SetOutputDirection(fixedImage->GetDirection());
    resampler1->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampler1->SetOutputSpacing(fixedImage->GetSpacing());
    resampler1->SetInput(movingImage);

    typedef itk::SubtractImageFilter <FixedImageType, FixedImageType, FixedImageType > DifferenceFilterType;
    DifferenceFilterType::Pointer difference1 = DifferenceFilterType::New();
    difference1->SetInput1(fixedImage);
    difference1->SetInput2(resampler1->GetOutput());
    difference1->Update();
    writeImage(difference1->GetOutput(), outputFile);

    //difference with transform
    ResampleFilterType::Pointer resampler2 = ResampleFilterType::New();
    resampler2->SetInterpolator(sincinterpolator);
    resampler2->SetTransform(registration->GetOutput()->Get());
    resampler2->SetOutputOrigin(fixedImage->GetOrigin());
    resampler2->SetOutputDirection(fixedImage->GetDirection());
    resampler2->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampler2->SetOutputSpacing(fixedImage->GetSpacing());
    resampler2->SetInput(movingImage);

    DifferenceFilterType::Pointer difference2 = DifferenceFilterType::New();
    difference2->SetInput1(fixedImage);
    difference2->SetInput2(resampler2->GetOutput());
    difference2->Update();

    outputFile = addFileNameSuffix(outputFile, "_transform");
    writeImage(difference2->GetOutput(), outputFile);
  }

  return EXIT_SUCCESS;
}

int extractImage(FloatImage3D::Pointer image, FloatImage3D::PointType leftPoint, FloatImage3D::PointType rightPoint)
{
  FloatImage3D::IndexType leftIndex;
  FloatImage3D::IndexType rightIndex;

  bool leftIsInside = image->TransformPhysicalPointToIndex(leftPoint, leftIndex);
  bool rightIsInside = image->TransformPhysicalPointToIndex(rightPoint, rightIndex);

  for (int n = 0; n < FloatImage3D::ImageDimension; ++n) {
    //    if (leftIndex[n] > rightIndex[n]) {
    //      std::swap(leftIndex[n], rightIndex[n]);
    //    }

    int index;
    index = image->GetLargestPossibleRegion().GetIndex()[n];
    if (leftIndex[n] < index) {
      leftIndex[n] = index;
    }

    index = image->GetLargestPossibleRegion().GetUpperIndex()[n];
    if (rightIndex[n] > index) {
      rightIndex[n] = index;
    }
  }


  FloatImage3D::RegionType fixedRegion;
  fixedRegion.SetIndex(leftIndex);
  fixedRegion.SetUpperIndex(rightIndex);

  typedef itk::RegionOfInterestImageFilter <FloatImage3D, FloatImage3D> RegionOfInterestImageFilterType;
  RegionOfInterestImageFilterType::Pointer filter = RegionOfInterestImageFilterType::New();
  filter->SetRegionOfInterest(fixedRegion);
  filter->SetInput(image);
  filter->Update();

  image->Graft(filter->GetOutput());

  return EXIT_SUCCESS;
}

