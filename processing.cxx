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
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkImageSpatialObject.h"
#include "itkCenteredTransformInitializer.h"

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

int imageTransform(FloatImage3D::Pointer image);
FloatImage3D::Pointer extractImage(FloatImage3D::Pointer image, FloatImage3D::PointType leftPoint, FloatImage3D::PointType rightPoint);

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

  std::cout << " fixed image " << fixedImage->GetLargestPossibleRegion().GetSize() << std::endl;
  imageTransform(fixedImage);
  std::cout << fixedImage->GetOrigin() << std::endl;
  std::cout << fixedImage->GetSpacing() << std::endl;
  std::cout << fixedImage->GetDirection() << std::endl;


  MovingImageType::Pointer movingImage = MovingImageType::New();
  if (!readImage(movingImage, movingImageFile))
    return EXIT_FAILURE;

  std::cout << "moving image " << movingImage->GetLargestPossibleRegion().GetSize() << std::endl;
  std::cout << movingImage->GetOrigin() << std::endl;
  std::cout << movingImage->GetSpacing() << std::endl;
  std::cout << movingImage->GetDirection() << std::endl;

  typedef itk::ImageSpatialObject<3, float> FixedSpatialObject;
  FixedSpatialObject::Pointer fixedSO = FixedSpatialObject::New();
  fixedSO->SetImage(fixedImage);

  typedef itk::ImageSpatialObject<3, float> MovingSpatialObject;
  MovingSpatialObject::Pointer movingSO = MovingSpatialObject::New();
  movingSO->SetImage(movingImage);

  std::cout << fixedSO->GetBoundingBox()->GetBounds() << std::endl;
  std::cout << movingSO->GetBoundingBox()->GetBounds() << std::endl;

  typedef itk::VectorContainer<int, FloatImage3D::PointType> VectorContainerType;
  typedef  itk::BoundingBox<int, 3, double, VectorContainerType> BoundingBoxType;

  BoundingBoxType::BoundsArrayType bounds;
  FloatImage3D::PointType leftPoint, rightPoint;

  for (int n = 0; n < 3; ++n) {
    int ind1 = 2 * n;
    int ind2 = 2 * n + 1;

    leftPoint[n] = std::max(fixedSO->GetBoundingBox()->GetBounds()[ind1], movingSO->GetBoundingBox()->GetBounds()[ind1]);
    rightPoint[n] = std::min(fixedSO->GetBoundingBox()->GetBounds()[ind2], movingSO->GetBoundingBox()->GetBounds()[ind2]);
  }
  std::cout << bounds << std::endl;

  FloatImage3D::Pointer fixedImage2 = extractImage(fixedImage, leftPoint, rightPoint);
  FloatImage3D::Pointer movingImage2 = extractImage(movingImage, leftPoint, rightPoint);


  writeImage(fixedImage2, "fixed_region.nrrd");
  writeImage(movingImage2, "moving_region.nrrd");


  return EXIT_SUCCESS;
}

int imageTransform(FloatImage3D::Pointer image)
{

  FloatImage3D::DirectionType direction;
  direction.SetIdentity();

  Image3DSize outputSize;
  outputSize[0] = image->GetLargestPossibleRegion().GetSize()[0];
  outputSize[1] = image->GetLargestPossibleRegion().GetSize()[2];
  outputSize[2] = image->GetLargestPossibleRegion().GetSize()[1];

  Image3DSpacing outputSpacing;
  outputSpacing[0] = image->GetSpacing()[0];
  outputSpacing[1] = image->GetSpacing()[2];
  outputSpacing[2] = image->GetSpacing()[1];

  itk::ContinuousIndex<double, FloatImage3D::ImageDimension> indexOrigin;
  indexOrigin[0] = image->GetLargestPossibleRegion().GetIndex()[0];
  indexOrigin[1] = image->GetLargestPossibleRegion().GetIndex()[1] + image->GetLargestPossibleRegion().GetSize()[1];
  indexOrigin[2] = image->GetLargestPossibleRegion().GetIndex()[2];

  Image3DPoint outputOrigin;
  image->TransformContinuousIndexToPhysicalPoint(indexOrigin, outputOrigin);

  typedef itk::ResampleImageFilter <MovingImageType, FixedImageType> ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetOutputDirection(direction);
  resampler->SetOutputOrigin(outputOrigin);
  resampler->SetOutputSpacing(outputSpacing);
  resampler->SetSize(outputSize);
  resampler->SetInput(image);
  resampler->Update();

  image->Graft(resampler->GetOutput());

  /*
  std::cout << image->GetLargestPossibleRegion().GetSize() << std::endl;
  std::cout << image->GetOrigin() << std::endl;
  std::cout << image->GetSpacing() << std::endl;
  std::cout << image->GetDirection() << std::endl;
  */
  return EXIT_SUCCESS;
}

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

  //image->Graft(filter->GetOutput());

  return filter->GetOutput();
}

