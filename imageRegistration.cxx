#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>
#include <itkEuler3DTransform.h>
#include <itkTranslationTransform.h>
#include <itkImageMaskSpatialObject.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkLBFGSBOptimizer.h>
#include <itkImageMomentsCalculator.h>
#include <itkMeanSquaresImageToImageMetric.h>

#include <itkImageRegistrationMethod.h>
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

using namespace agtk;

typedef FloatImage3D  FixedImageType;
typedef FloatImage3D  MovingImageType;

const    unsigned int    Dimension = 3;
typedef  float           PixelType;

typedef itk::Euler3DTransform<double> TransformType;
FloatImage3D::Pointer imageTransform(FloatImage3D::Pointer image, TransformType::Pointer transform);

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
  // initialize transform
  typedef itk::BinaryThresholdImageFilter<FloatImage3D, BinaryImage3D> ThresholdImageFilterType;
  ThresholdImageFilterType::Pointer fixedThreshold = ThresholdImageFilterType::New();
  fixedThreshold->SetInput(fixedImage);
  fixedThreshold->SetLowerThreshold(50);
  fixedThreshold->SetUpperThreshold(FloatLimits::max());
  fixedThreshold->Update();

  // moment calculators
  typedef itk::ImageMomentsCalculator<BinaryImage3D>  ImageCalculatorType;
  ImageCalculatorType::Pointer fixedImageCalculator = ImageCalculatorType::New();
  fixedImageCalculator->SetImage(fixedThreshold->GetOutput());
  fixedImageCalculator->Compute();
  std::cout << "center of fixed image  " << fixedImageCalculator->GetCenterOfGravity() << std::endl;

  ThresholdImageFilterType::Pointer movingThreshold = ThresholdImageFilterType::New();
  movingThreshold->SetInput(movingImage);
  movingThreshold->SetLowerThreshold(50);
  movingThreshold->SetUpperThreshold(FloatLimits::max());
  movingThreshold->Update();

  // moment calculators
  typedef itk::ImageMomentsCalculator<BinaryImage3D>  ImageCalculatorType;
  ImageCalculatorType::Pointer movingImageCalculator = ImageCalculatorType::New();
  movingImageCalculator->SetImage(movingThreshold->GetOutput());
  movingImageCalculator->Compute();
  std::cout << "center of moving image " << movingImageCalculator->GetCenterOfGravity() << std::endl;

  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  transform->SetCenter(fixedImageCalculator->GetCenterOfGravity());
  transform->SetTranslation(movingImageCalculator->GetCenterOfGravity() - fixedImageCalculator->GetCenterOfGravity());
  size_t numberOfParameters = transform->GetNumberOfParameters();

  std::cout << "initial transform " << std::endl;
  std::cout << transform->GetParameters() << std::endl;
  std::cout << "     center " << transform->GetCenter() << std::endl;
  std::cout << "translation " << transform->GetTranslation() << std::endl;

  if (parser->ArgumentExists("-initial")) {
    FloatImage3D::Pointer output = imageTransform(movingImage, transform);

    std::string fileName;
    parser->GetValue("-initial", fileName);
    if (!writeImage(output, fileName)) {
      return EXIT_FAILURE;
    }
  }

  typedef itk::LinearInterpolateImageFunction <MovingImageType, double> InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  typedef itk::MeanSquaresImageToImageMetric <FixedImageType, MovingImageType > MetricType;
  MetricType::Pointer metric = MetricType::New();

  // optimizer
  itk::Array<size_t> modeBounds;
  modeBounds.set_size(numberOfParameters);
  modeBounds.fill(0);

  itk::Array<double> lowerBounds;
  lowerBounds.set_size(numberOfParameters);
  lowerBounds.fill(0);

  itk::Array<double> upperBounds;
  upperBounds.set_size(numberOfParameters);
  upperBounds.fill(0);

  for (int i = 0; i < 3; ++i) {
    modeBounds[i] = 2;
    lowerBounds[i] = -(5./180)*itk::Math::pi;
    upperBounds[i] =  (5./180)*itk::Math::pi;
  }

  typedef itk::LBFGSBOptimizer OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetBoundSelection(modeBounds);
  optimizer->SetLowerBound(lowerBounds);
  optimizer->SetUpperBound(upperBounds);
  optimizer->SetMaximumNumberOfEvaluations(100);

  typedef itk::ImageRegistrationMethod <FixedImageType, MovingImageType> RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetInterpolator(interpolator);
  registration->SetFixedImageRegion(fixedImage->GetLargestPossibleRegion());
  registration->SetFixedImage(fixedImage);
  registration->SetMovingImage(movingImage);
  registration->SetInitialTransformParameters(transform->GetParameters());
  registration->SetTransform(transform);
  try {
    registration->Update();
  }
  catch (itk::ExceptionObject & excep) {
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << optimizer->GetStopConditionDescription() << std::endl; 
  std::cout << "       iterations " << optimizer->GetCurrentIteration() << std::endl;
  std::cout << "     metric value " << optimizer->GetValue() << std::endl;
  std::cout << " initial position " << optimizer->GetInitialPosition() << std::endl;
  std::cout << "current iteration " << optimizer->GetCurrentPosition() << std::endl;

  if (parser->ArgumentExists("-output")) {
    typedef itk::ResampleImageFilter <MovingImageType, FixedImageType> ResampleFilterType;
    ResampleFilterType::Pointer resampler = ResampleFilterType::New();
    resampler->SetOutputDirection(movingImage->GetDirection());
    resampler->SetOutputOrigin(fixedImage->GetOrigin());
    resampler->SetOutputSpacing(movingImage->GetSpacing());
    resampler->SetSize(movingImage->GetLargestPossibleRegion().GetSize());
    resampler->SetInput(movingImage);
    resampler->SetTransform(transform);
    resampler->Update();

    std::string fileName;
    parser->GetValue("-output", fileName);
    if (!writeImage(resampler->GetOutput(), fileName)) {
      return EXIT_FAILURE;
    }
  }

  /*
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
  */

  /*
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
  */
  return EXIT_SUCCESS;
}

FloatImage3D::Pointer imageTransform(FloatImage3D::Pointer image, TransformType::Pointer transform)
{
  typedef itk::ResampleImageFilter <MovingImageType, FixedImageType> ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetOutputDirection(image->GetDirection());
  resampler->SetOutputOrigin(transform->GetInverseTransform()->TransformPoint(image->GetOrigin()));
  resampler->SetOutputSpacing(image->GetSpacing());
  resampler->SetSize(image->GetLargestPossibleRegion().GetSize());
  resampler->SetInput(image);
  resampler->SetTransform(transform);
  resampler->Update();

  return resampler->GetOutput();
}

/*
FloatImage3D::Pointer extractImage(FloatImage3D::Pointer image, FloatImage3D::PointType leftPoint, FloatImage3D::PointType rightPoint)
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

  return filter->GetOutput();
}
*/
