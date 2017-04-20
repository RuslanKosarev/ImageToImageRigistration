#pragma once

#include <itkStatisticalShapeModelTransform.h>
#include <itkCompositeTransform.h>
#include <itkTranslationTransform.h>
#include <itkEuler3DTransform.h>
#include <itkSimilarity3DTransform.h>
#include <itkScaleSkewVersor3DTransform.h>
#include <itkLogger.h>

namespace agtk
{
  enum class Transform
  {
    Translation,
    Euler3D,
    Similarity,
    ScaleSkewVersor3D
  };

  template<typename TParametersValueType, unsigned int NDimensions = 3>
  class InitializeTransform : public itk::ProcessObject
  {
  public:
    /** Standard class typedefs. */
    typedef InitializeTransform                     Self;
    typedef itk::ProcessObject                      Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);
    itkTypeMacro(InitializeTransform, itk::ProcessObject);

    /** typedefs */
    itkStaticConstMacro(PointDimension, unsigned int, NDimensions);
    static_assert(PointDimension == 3U, "Invalid dimension of the input shape model. Dimension 3 is supported.");

    typedef typename itk::Transform<TParametersValueType, PointDimension> TransformType;
    typedef typename TransformType::InputPointType InputPointType;
    typedef typename TransformType::OutputVectorType OutputVectorType;
    typedef itk::Array<double> ParametersType;
    typedef itk::Array<unsigned int> ModeBoundsType;

    // Set logger
    itkSetObjectMacro(Logger, itk::Logger);

    // Get transform
    itkGetObjectMacro(Transform, TransformType);

    // Set/Get type of transform
    itkSetEnumMacro(TypeOfTransform, Transform);
    itkGetEnumMacro(TypeOfTransform, Transform);
    void SetTypeOfTransform(const size_t & type) { this->SetTypeOfTransform(static_cast<Transform>(type)); }

    itkSetMacro(RotationScale, double);
    itkGetMacro(RotationScale, double);

    itkSetMacro(TranslationScale, double);
    itkGetMacro(TranslationScale, double);

    itkSetMacro(ScalingScale, double);
    itkGetMacro(ScalingScale, double);

    itkSetMacro(SkewScale, double);
    itkGetMacro(SkewScale, double);

    // Get scales and bounds
    itkGetMacro(Scales, ParametersType);
    itkGetMacro(ModeBounds, ModeBoundsType);
    itkGetMacro(LowerBounds, ParametersType);
    itkGetMacro(UpperBounds, ParametersType);

    // Set/Get center and translation
    itkGetMacro(Center, InputPointType);
    itkSetMacro(Center, InputPointType);
    itkGetMacro(Translation, OutputVectorType);
    itkSetMacro(Translation, OutputVectorType);

    void Update()
    {
      switch (m_TypeOfTransform) {
      case Transform::Translation: {
        // Translation transform
        typedef typename itk::TranslationTransform<TParametersValueType, PointDimension> TranslationTransformType;
        typename TranslationTransformType::Pointer transform = TranslationTransformType::New();
        transform->Translate(m_Translation);

        m_Transform = transform;
        this->Allocate();

        // define scales
        m_NumberOfTranslationComponents = 3;

        size_t count = 0;

        for (size_t i = 0; i < m_NumberOfTranslationComponents; ++i, ++count) {
          m_Scales[count] = m_TranslationScale;
          m_ModeBounds[count] = 0;
        }

        break;
      }
      case Transform::Euler3D:{
        // Euler3DTransform
        typedef itk::Euler3DTransform<TParametersValueType> Euler3DTransformType;
        typename Euler3DTransformType::Pointer transform = Euler3DTransformType::New();
        transform->SetIdentity();
        transform->SetCenter(m_Center);
        transform->SetTranslation(m_Translation);

        m_Transform = transform;
        this->Allocate();

        // define scales
        m_NumberOfRotationComponents = 3;
        m_NumberOfTranslationComponents = 3;

        size_t count = 0;

        for (size_t i = 0; i < m_NumberOfRotationComponents; ++i, ++count) {
          m_Scales[count] = m_RotationScale;
          m_ModeBounds[count] = 2;
        }

        for (size_t i = 0; i < m_NumberOfTranslationComponents; ++i, ++count) {
          m_Scales[count] = m_TranslationScale;
          m_ModeBounds[count] = 0;
        }

        break;
      }

      case Transform::Similarity:{
        // Similarity3DTransform
        typedef itk::Similarity3DTransform<TParametersValueType> Similarity3DTransformType;
        typename Similarity3DTransformType::Pointer transform = Similarity3DTransformType::New();
        transform->SetIdentity();
        transform->SetCenter(m_Center);
        transform->SetTranslation(m_Translation);

        m_Transform = transform;
        this->Allocate();

        // define scales
        m_NumberOfRotationComponents = 3;
        m_NumberOfTranslationComponents = 3;
        m_NumberOfScalingComponents = 1;

        size_t count = 0;

        for (size_t i = 0; i < m_NumberOfRotationComponents; ++i, ++count) {
          m_Scales[count] = m_RotationScale;
          m_ModeBounds[count] = 2;
        }

        for (size_t i = 0; i < m_NumberOfTranslationComponents; ++i, ++count) {
          m_Scales[count] = m_TranslationScale;
          m_ModeBounds[count] = 0;
        }

        for (size_t i = 0; i < m_NumberOfScalingComponents; ++i, ++count) {
          m_Scales[count] = m_ScalingScale;
          m_ModeBounds[count] = 2;
        }

        break;
      }

      case Transform::ScaleSkewVersor3D:{
        typedef itk::ScaleSkewVersor3DTransform<TParametersValueType> ScaleSkewVersor3DTransformType;
        typename ScaleSkewVersor3DTransformType::Pointer transform = ScaleSkewVersor3DTransformType::New();
        transform->SetIdentity();
        transform->SetCenter(m_Center);
        transform->SetTranslation(m_Translation);

        m_Transform = transform;
        this->Allocate();

        // define scales
        m_NumberOfRotationComponents = 3;
        m_NumberOfTranslationComponents = 3;
        m_NumberOfScalingComponents = 3;
        m_NumberOfSkewComponents = 6;

        size_t count = 0;

        for (size_t i = 0; i < m_NumberOfRotationComponents; ++i, ++count) {
          m_Scales[count] = m_RotationScale;
          m_ModeBounds[count] = 2;
        }

        for (size_t i = 0; i < m_NumberOfTranslationComponents; ++i, ++count) {
          m_Scales[count] = m_TranslationScale;
          m_ModeBounds[count] = 0;
        }

        for (size_t i = 0; i < m_NumberOfScalingComponents; ++i, ++count) {
          m_Scales[count] = m_ScalingScale;
          m_ModeBounds[count] = 2;
        }

        for (size_t i = 0; i < m_NumberOfSkewComponents; ++i, ++count) {
          m_Scales[count] = m_SkewScale;
          m_ModeBounds[count] = 2;
        }

        break;
      }

      default:
        itkExceptionMacro(<< "Invalid type of transform");
      }

      m_NumberOfParameters = m_Transform->GetNumberOfParameters();

      for (size_t n = 0; n < m_NumberOfParameters; ++n) {
        m_LowerBounds[n] = m_Transform->GetParameters()[n] - m_Scales[n];
        m_UpperBounds[n] = m_Transform->GetParameters()[n] + m_Scales[n];
      }
    }

    void PrintReport() const
    {
      m_Message.str("");
      m_Message << this->GetNameOfClass() << std::endl;
      m_Message << "spatial transform    " << m_Transform->GetTransformTypeAsString() << std::endl;
      m_Message << "center               " << m_Center << std::endl;
      m_Message << "translation          " << m_Translation << std::endl;
      m_Message << "fixed parameters     " << m_Transform->GetFixedParameters() << m_Transform->GetNumberOfFixedParameters() << std::endl;
      m_Message << "parameters           " << m_Transform->GetParameters() << ", " << m_Transform->GetNumberOfParameters() << std::endl;
      m_Message << "number of parameters " << m_Transform->GetNumberOfParameters() << std::endl;
      m_Message << "scales               " << std::endl << m_Scales << std::endl;
      m_Message << "mode bounds          " << std::endl << m_ModeBounds << std::endl;
      m_Message << "lower bounds         " << std::endl << m_LowerBounds << std::endl;
      m_Message << "upper bounds         " << std::endl << m_UpperBounds << std::endl;
      m_Message << std::endl;
      std::cout << m_Message.str();
    }

  protected:
    Transform m_TypeOfTransform = Transform::Similarity;
    typename TransformType::Pointer m_Transform = nullptr;

    InputPointType m_Center;
    OutputVectorType m_Translation;

    /** Set the boundary condition for each variable, where
    * select[i] = 0 if x[i] is unbounded,
    *           = 1 if x[i] has only a lower bound,
    *           = 2 if x[i] has both lower and upper bounds, and
    *           = 3 if x[1] has only an upper bound */
    ModeBoundsType m_ModeBounds;
    ParametersType m_LowerBounds;
    ParametersType m_UpperBounds;
    ParametersType m_Scales;

    double m_TranslationScale = 1;
    double m_RotationScale = 0.2;
    double m_ScalingScale = 0.2;
    double m_SkewScale = 0.2;

    size_t m_NumberOfTranslationComponents = 0;
    size_t m_NumberOfRotationComponents = 0;
    size_t m_NumberOfScalingComponents = 0;
    size_t m_NumberOfSkewComponents = 0;
    size_t m_NumberOfParameters = 0;

    itk::Logger::Pointer m_Logger;
    mutable std::ostringstream m_Message;

    InitializeTransform()
    {
      this->SetNumberOfRequiredInputs(0);
      this->SetNumberOfRequiredOutputs(0);
      m_Logger = itk::Logger::New();
    }

    void Allocate()
    {
      size_t size = m_Transform->GetNumberOfParameters();
      m_Scales.set_size(size);
      m_ModeBounds.set_size(size);
      m_LowerBounds.set_size(size);
      m_UpperBounds.set_size(size);
    }

    ~InitializeTransform() {}
  };
}
