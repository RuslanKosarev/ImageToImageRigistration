#pragma once

#include <itkImageToImageFilter.h>
#include <itkTransform.h>
#include <itkCastImageFilter.h>
#include <itkMacro.h>

namespace agtk
{
template< typename TInputImage, typename TOutputImage = TInputImage>
class TransformImageFilter : public itk::ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef TransformImageFilter                                   Self;
  typedef itk::ImageToImageFilter< TInputImage, TOutputImage >   Superclass;
  typedef itk::SmartPointer< Self >                              Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformImageFilter, itk::ImageToImageFilter);

  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;
  typedef double ScalarType;
  typedef itk::Transform<ScalarType, InputImageType::ImageDimension> TransformType;

  itkSetConstObjectMacro(Transform, TransformType);
  itkGetConstObjectMacro(Transform, TransformType);

protected:
  TransformImageFilter()
  {
    m_Transform = nullptr;
  }

  ~TransformImageFilter() {}

  /** Does the real work. */
  virtual void GenerateData()
  {
    typedef itk::CastImageFilter<InputImageType, OutputImageType> CastImageFilterType;
    CastImageFilterType::Pointer cast = CastImageFilterType::New();
    cast->SetInput(this->GetInput());
    try {
      cast->Update();
    }
    catch (itk::ExceptionObject & excep) {
      itkExceptionMacro(<< excep)
    }

    typename OutputImageType::Pointer output = cast->GetOutput();

    // modify direction
    typename OutputImageType::DirectionType direction = output->GetDirection();

    for (size_t col = 0; col < direction.ColumnDimensions; ++col) {
      TransformType::InputVectorType vector;
      for (size_t row = 0; row < direction.RowDimensions; ++row) {
        vector[row] = direction[row][col];
      }

      vector = m_Transform->TransformVector(vector);
      for (size_t row = 0; row < direction.RowDimensions; ++row) {
        direction[row][col] = vector[row];
      }
    }

    output->SetDirection(direction);

    // modify origin
    output->SetOrigin(m_Transform->TransformPoint(output->GetOrigin()));

    this->GraftOutput(output);
  }

private:
  TransformImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  
  typename TransformType::ConstPointer m_Transform;
};
}
