#pragma once

#include <itkObjectToObjectOptimizerBase.h>
#include <itkLBFGSBOptimizerv4.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>
#include <itkLogger.h>

namespace agtk
{
  enum class Optimizer
  {
    LBFGSBOptimizerv4,
    RegularStepGradientDescentOptimizerv4
  };

  class InitializeOptimizer : public itk::ProcessObject
  {
  public:
    /** Standard class typedefs. */
    typedef InitializeOptimizer                     Self;
    typedef itk::ProcessObject                      Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);
    itkTypeMacro(InitializeOptimizer, itk::ProcessObject);

    /** typedefs */
    typedef double RealType;
    typedef itk::ObjectToObjectOptimizerBaseTemplate<RealType>          OptimizerType;
    typedef typename OptimizerType::Pointer                             OptimizerPointer;
    typedef itk::Array<double> ParametersType;
    typedef itk::Array<unsigned int> ModeBoundsType;

    // Set logger
    itkSetObjectMacro(Logger, itk::Logger);

    // Set/Get optimizer type
    itkSetEnumMacro(OptimizerType, Optimizer);
    itkGetEnumMacro(OptimizerType, Optimizer);
    void SetOptimizerType(const size_t & type) { this->SetOptimizerType(static_cast<Optimizer>(type)); }

    /** Get optimizer */
    itkGetObjectMacro(Optimizer, OptimizerType);

    // Set/Get initial parameters
    itkSetMacro(InitialParameters, ParametersType);
    itkGetMacro(InitialParameters, ParametersType);

    // Set/Get bounds and scales
    itkSetMacro(Scales, ParametersType);
    itkGetMacro(Scales, ParametersType);

    itkSetMacro(LowerBounds, ParametersType);
    itkGetMacro(LowerBounds, ParametersType);

    itkSetMacro(UpperBounds, ParametersType);
    itkGetMacro(UpperBounds, ParametersType);

    itkSetMacro(ModeBounds, ModeBoundsType);
    itkGetMacro(ModeBounds, ModeBoundsType);

    void Update()
    {
      switch (m_OptimizerType) {
      case Optimizer::RegularStepGradientDescentOptimizerv4: {
        RegularStepGradientDescentOptimizer = itk::RegularStepGradientDescentOptimizerv4<RealType>::New();
        RegularStepGradientDescentOptimizer->SetLearningRate(4);
        RegularStepGradientDescentOptimizer->SetMinimumStepLength(1.0e-05);
        RegularStepGradientDescentOptimizer->SetRelaxationFactor(0.5);
        
        bool useEstimator = true;
        /*
        if ( useEstimator ) {
        typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
        ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
        scalesEstimator->SetMetric(metric);
        scalesEstimator->SetTransformForward(true);
        optimizer->SetScalesEstimator(scalesEstimator);
        optimizer->SetDoEstimateLearningRateOnce(true);
        }*/

        RegularStepGradientDescentOptimizer->SetNumberOfIterations(10);

        m_Optimizer = RegularStepGradientDescentOptimizer;

        m_OptimizerMessage.str("");
        m_OptimizerMessage << "number of iterations       " << RegularStepGradientDescentOptimizer->GetNumberOfIterations() << std::endl;
        m_OptimizerMessage << "relaxation factor          " << RegularStepGradientDescentOptimizer->GetRelaxationFactor() << std::endl;
        m_OptimizerMessage << "minimum step length        " << RegularStepGradientDescentOptimizer->GetMinimumStepLength() << std::endl;
        m_OptimizerMessage << "gradient tolerance         " << RegularStepGradientDescentOptimizer->GetGradientMagnitudeTolerance() << std::endl;
        m_OptimizerMessage << "scales                     " << std::endl << m_Scales << std::endl;

        break;
      }
      case Optimizer::LBFGSBOptimizerv4: {
        m_LBFGSBOptimizer = itk::LBFGSBOptimizerv4::New();
        m_LBFGSBOptimizer->SetBoundSelection(m_ModeBounds);
        m_LBFGSBOptimizer->SetLowerBound(m_LowerBounds);
        m_LBFGSBOptimizer->SetUpperBound(m_UpperBounds);
        m_LBFGSBOptimizer->SetMaximumNumberOfCorrections(50);
        m_LBFGSBOptimizer->SetCostFunctionConvergenceFactor(1);
        m_LBFGSBOptimizer->SetGlobalWarningDisplay(false);
        m_LBFGSBOptimizer->SetTrace(false);
        m_Optimizer = m_LBFGSBOptimizer;

        m_OptimizerMessage.str("");
        m_OptimizerMessage << "mode bounds                " << std::endl << m_ModeBounds << std::endl;
        m_OptimizerMessage << "lower bounds               " << std::endl << m_LowerBounds << std::endl;
        m_OptimizerMessage << "upper bounds               " << std::endl << m_UpperBounds << std::endl;
        break;
      }
      default:
        itkExceptionMacro(<< "Invalid type of optimizer");
      }

      //m_Optimizer->SetInitialPosition(m_InitialParameters);
    }

    void PrintReport() const
    {
      m_Message.str("");
      m_Message << this->GetNameOfClass() << std::endl;
      m_Message << "optimizer type       " << m_Optimizer->GetNameOfClass() << std::endl;
      m_Message << m_OptimizerMessage.str() << std::endl;
      m_Message << std::endl;
      m_Logger->Info(m_Message.str());
    }

    itk::RegularStepGradientDescentOptimizerv4<RealType>::Pointer RegularStepGradientDescentOptimizer;
    itk::LBFGSBOptimizerv4::Pointer m_LBFGSBOptimizer;

  protected:
    OptimizerType::Pointer m_Optimizer;

    Optimizer m_OptimizerType;
    ModeBoundsType m_ModeBounds;
    ParametersType m_LowerBounds;
    ParametersType m_UpperBounds;
    ParametersType m_Scales;
    ParametersType m_InitialParameters;

    std::ostringstream m_OptimizerMessage;
    mutable std::ostringstream m_Message;
    itk::Logger::Pointer m_Logger;

    InitializeOptimizer()
    {
      this->SetNumberOfRequiredInputs(0);
      this->SetNumberOfRequiredOutputs(0);
      m_OptimizerType = Optimizer::RegularStepGradientDescentOptimizerv4;
      m_Logger = itk::Logger::New();
    }

    ~InitializeOptimizer() {}
  };
}
