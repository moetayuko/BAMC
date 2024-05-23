#include "mex.hpp"
#include "mexAdapter.hpp"

#include <Eigen/Core>
#include <vector>

namespace {
// https://ww2.mathworks.cn/matlabcentral/answers/436916-how-to-access-raw-data-via-the-mex-c-api#answer_415645

//! Extracts the pointer to underlying data from the non-const iterator
//! (`TypedIterator<T>`).
/*! This function does not throw any exceptions. */
template <typename T>
inline T *ToPointer(const matlab::data::TypedIterator<T> &it) MW_NOEXCEPT {
  static_assert(
      std::is_arithmetic<T>::value && !std::is_const<T>::value,
      "Template argument T must be a std::is_arithmetic and non-const type.");
  return it.operator->();
}

/*! Extracts pointer to the first element in the array.
 *  Example usage:
 *  \code
 *  ArrayFactory factory;
 *  TypedArray<double> A = factory.createArray<double>({ 2,2 },
 * { 1.0, 3.0, 2.0, 4.0 }); auto ptr = getPointer(A); \endcode \note Do not call
 * `getPointer` with temporary object. e.g., the following code is ill-formed.
 *        auto ptr=getPointer(factory.createArray<double>({ 2,2
 * },{ 1.0, 3.0, 2.0, 4.0 }));
 */
template <typename T>
inline T *GetPointer(matlab::data::TypedArray<T> &arr) MW_NOEXCEPT {
  static_assert(std::is_arithmetic<T>::value,
                "Template argument T must be a std::is_arithmetic type.");
  return ToPointer(arr.begin());
}

template <typename T>
inline const T *GetPointer(const matlab::data::TypedArray<T> &arr) MW_NOEXCEPT {
  return GetPointer(const_cast<matlab::data::TypedArray<T> &>(arr));
}
};  // namespace

namespace {

using MexArr = matlab::data::TypedArray<double>;

template <typename Derived>
struct PosVisitor {
  typedef typename Derived::Scalar Scalar;

  EIGEN_DEVICE_FUNC PosVisitor() : count(0), sum(0) {}
  Scalar sum;
  std::size_t count;
  EIGEN_DEVICE_FUNC inline void init(const Scalar &value, Eigen::Index,
                                     Eigen::Index = 0) {
    if (value > 0) {
      sum = value;
      count = 1;
    } else {
      sum = 0;
      count = 0;
    }
  }
  EIGEN_DEVICE_FUNC inline void operator()(const Scalar &value, Eigen::Index,
                                           Eigen::Index = 0) {
    if (value > 0) {
      sum += value;
      ++count;
    }
  }
};

size_t eproj_simplex_vec(Eigen::Ref<Eigen::VectorXd> v, const double k) {
  size_t ft = 1;

  double n = v.size();
  v = v.array() - v.mean() + k / n;
  double vmin = v.minCoeff();

  if (vmin < 0.0) {
    double f = 1.0;
    double lambda_m = 0.0;
    Eigen::VectorXd v1(v.size());
    PosVisitor<decltype(v)> pos_visitor;
    for (; ft <= 100 && std::fabs(f) > 1e-10; ++ft) {
      v1 = v.array() - lambda_m;
      v1.visit(pos_visitor);
      f = pos_visitor.sum - k;
      lambda_m += f / pos_visitor.count;
    }
    v = v1.cwiseMax(0);
  }
  return ft;
}

void eproj_simplex(Eigen::Ref<Eigen::MatrixXd> v, const double k) {
  const auto num = v.cols();
#pragma omp parallel for
  for (auto i = 0; i < num; ++i) eproj_simplex_vec(v.col(i), k);
}
};  // namespace

namespace Eigen {
namespace internal {
template <typename Derived>
struct functor_traits<PosVisitor<Derived>> {
  using Scalar = typename Derived::Scalar;
  enum {
    Cost = NumTraits<Scalar>::AddCost,
    LinearAccess = true,
    PacketAccess = false
  };
};
};  // namespace internal
};  // namespace Eigen

class MexFunction : public matlab::mex::Function {
 public:
  void operator()(matlab::mex::ArgumentList outputs,
                  matlab::mex::ArgumentList inputs) {
    if (outputs.empty()) return;
    checkArguments(inputs);

    MexArr v = std::move(inputs[0]);
    const double k = inputs.size() == 2 ? inputs[1][0] : 1.0;
    // columns are data samples
    const auto num = v.getDimensions()[1];

    auto VecToEigen = [](MexArr &rhs) {
      return Eigen::Map<Eigen::VectorXd>(GetPointer(rhs),
                                         rhs.getNumberOfElements());
    };
    auto MatToEigen = [](MexArr &rhs) {
      auto dims = rhs.getDimensions();
      return Eigen::Map<Eigen::MatrixXd>(GetPointer(rhs), dims[0], dims[1]);
    };

    if (num == 1) {
      auto v_eigen = VecToEigen(v);
      eproj_simplex_vec(v_eigen, k);
    } else {
      auto v_eigen = MatToEigen(v);
      eproj_simplex(v_eigen, k);
    }

    outputs[0] = std::move(v);
  }

  void checkArguments(matlab::mex::ArgumentList inputs) {
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    matlab::data::ArrayFactory factory;

    if (inputs.size() < 1 || inputs.size() > 2 ||
        inputs[0].getType() != matlab::data::ArrayType::DOUBLE ||
        inputs[0].getMemoryLayout() !=
            matlab::data::MemoryLayout::COLUMN_MAJOR) {
      matlabPtr->feval(u"error", 0,
                       std::vector<matlab::data::Array>(
                           {factory.createScalar("Incorrect input")}));
    }
  }
};
