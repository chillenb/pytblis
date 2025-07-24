#include "tblis/frame/base/basic_types.h"
#include "tblis.h"
#include "tblis/frame/1t/shift.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

using namespace tblis;

static nb::dlpack::dtype get_nb_dtype_from_tblis_type(type_t type) {
  switch (type) {
  case TYPE_FLOAT:
    return nb::dtype<float>();
  case TYPE_DOUBLE:
    return nb::dtype<double>();
  case TYPE_SCOMPLEX:
    return nb::dtype<scomplex>();
  case TYPE_DCOMPLEX:
    return nb::dtype<dcomplex>();
  default:
    throw std::runtime_error("Invalid TBLIS scalar type!");
  }
}

// Convert a tblis_scalar to a Python object
// Necessary because nanobind can't bind templates.
static nb::object pyobj_from_tblis_scalar(const tblis_scalar &scalar) {
  switch (scalar.type) {
  case TYPE_FLOAT:
    return nb::float_((double)scalar.data.s);
  case TYPE_DOUBLE:
    return nb::float_(PyFloat_FromDouble(scalar.data.d));
  case TYPE_SCOMPLEX:
    return nb::steal(PyComplex_FromDoubles((double)scalar.data.c.real(),
                                           (double)scalar.data.c.imag()));
  case TYPE_DCOMPLEX:
    return nb::steal(
        PyComplex_FromDoubles(scalar.data.z.real(), scalar.data.z.imag()));
  default:
    throw std::runtime_error("Invalid TBLIS scalar type!");
  }
}

// make a label_vector from a string
static label_vector string_to_label_vector(const std::string &str) {
  return label_vector(str.begin(), str.end());
}

// convert a nanobind ndarray to a tblis_tensor.
// allocates len and stride arrays. these must be freed using tblis_tensor_free_lenstride.
static tblis_tensor ndarray_to_scaled_tblis_tensor(const nb::ndarray<> &arr,
                                            dcomplex scalar = 1.0,
                                            bool conj = false) {
  tblis_tensor tensor;
  len_type *len = new len_type[arr.ndim()];
  stride_type *stride = new stride_type[arr.ndim()];

  for (int i = 0; i < arr.ndim(); ++i) {
    len[i] = (len_type)arr.shape(i);
    stride[i] = (stride_type)arr.stride(i);
  }

  nb::dlpack::dtype dtype = arr.dtype();
  if (dtype == nb::dtype<float>()) {
    tensor.type = TYPE_FLOAT;
    tblis_init_scalar_s(&tensor.scalar, scalar.real());
    if (scalar.imag() != 0.0f) {
      throw std::runtime_error(
          "Imaginary part of scalar not supported for float type");
    }
  } else if (dtype == nb::dtype<double>()) {
    tensor.type = TYPE_DOUBLE;
    tblis_init_scalar_d(&tensor.scalar, scalar.real());
    if (scalar.imag() != 0.0) {
      throw std::runtime_error(
          "Imaginary part of scalar not supported for double type");
    }
  } else if (dtype == nb::dtype<scomplex>()) {
    tensor.type = TYPE_SCOMPLEX;
    tblis_init_scalar_c(&tensor.scalar, scomplex(scalar.real(), scalar.imag()));
  } else if (dtype == nb::dtype<dcomplex>()) {
    tensor.type = TYPE_DCOMPLEX;
    tblis_init_scalar_z(&tensor.scalar, dcomplex(scalar.real(), scalar.imag()));
  } else {
    throw std::runtime_error("Unsupported dlpack scalar type. Supported types are float, double, scomplex, and dcomplex.");
  }

  tensor.ndim = arr.ndim();
  tensor.data = arr.data();
  tensor.conj = conj ? 1 : 0;

  tensor.len = len;
  tensor.stride = stride;
  return tensor;
}

static void tblis_tensor_free_lenstride(tblis_tensor &tensor) {
  if (tensor.len) {
    delete[] tensor.len;
    tensor.len = nullptr;
  }
  if (tensor.stride) {
    delete[] tensor.stride;
    tensor.stride = nullptr;
  }
}

NB_MODULE(_pytblis_impl, m) {
  m.doc() = "Python bindings for TBLIS";
  m.def(
      "add",
      [](const nb::ndarray<> &A, nb::ndarray<> &B, std::string idx_A,
         std::string idx_B, dcomplex alpha = 1.0, dcomplex beta = 1.0,
         bool conja = false, bool conjb = false) {
        tblis_tensor a = ndarray_to_scaled_tblis_tensor(A, alpha, conja);
        tblis_tensor b = ndarray_to_scaled_tblis_tensor(B, beta, conjb);
        label_vector idx_A_vec = string_to_label_vector(idx_A);
        label_vector idx_B_vec = string_to_label_vector(idx_B);
        tblis_tensor_add(NULL, NULL, &a, idx_A_vec.data(), &b,
                         idx_B_vec.data());
        tblis_tensor_free_lenstride(a);
        tblis_tensor_free_lenstride(b);
      },
      nb::arg("A"), nb::arg("B"), nb::arg("idx_A"), nb::arg("idx_B"),
      nb::arg("alpha") = 1.0, nb::arg("beta") = 1.0, nb::arg("conja") = false,
      nb::arg("conjb") = false,
      "B_[idx_B] = alpha * A_[idx_A] + beta * B_[idx_B]");

  m.def(
      "dot",
      [](const nb::ndarray<> &A, const nb::ndarray<> &B, const std::string &idx_A,
         const std::string &idx_B, dcomplex alpha = 1.0, dcomplex beta = 1.0,
         bool conja = false, bool conjb = false) {
        tblis_tensor a = ndarray_to_scaled_tblis_tensor(A, alpha, conja);
        tblis_tensor b = ndarray_to_scaled_tblis_tensor(B, beta, conjb);
        label_vector idx_A_vec = string_to_label_vector(idx_A);
        label_vector idx_B_vec = string_to_label_vector(idx_B);
        tblis_scalar result(0.0, 0.0);
        result.type = a.type;
        if (a.type != b.type) {
          throw std::runtime_error("Tensor types do not match for dot product");
        }
        tblis_tensor_dot(NULL, NULL, &a, idx_A_vec.data(), &b, idx_B_vec.data(),
                         &result);
        tblis_tensor_free_lenstride(a);
        tblis_tensor_free_lenstride(b);
        return pyobj_from_tblis_scalar(result);
      },
      nb::arg("A"), nb::arg("B"), nb::arg("idx_A"), nb::arg("idx_B"),
      nb::arg("alpha") = 1.0, nb::arg("beta") = 1.0, nb::arg("conja") = false,
      nb::arg("conjb") = false,
      "result = alpha * A_[idx_A] . beta * B_[idx_B]");

  nb::enum_<reduce_t>(m, "reduce_t")
      .value("REDUCE_SUM", REDUCE_SUM)
      .value("REDUCE_SUM_ABS", REDUCE_SUM_ABS)
      .value("REDUCE_MAX", REDUCE_MAX)
      .value("REDUCE_MAX_ABS", REDUCE_MAX_ABS)
      .value("REDUCE_MIN", REDUCE_MIN)
      .value("REDUCE_MIN_ABS", REDUCE_MIN_ABS)
      .value("REDUCE_NORM_1", REDUCE_NORM_1)
      .value("REDUCE_NORM_2", REDUCE_NORM_2)
      .value("REDUCE_NORM_INF", REDUCE_NORM_INF)
      .export_values();

  m.def(
      "mult",
      [](const nb::ndarray<> &A, const nb::ndarray<> &B, nb::ndarray<> &C,
         const std::string &idx_A, const std::string &idx_B,
         const std::string &idx_C, dcomplex alpha = 1.0,
         dcomplex beta = 1.0, bool conja = false, bool conjb = false) {
        // alpha is the product of the scalars of A and B
        // beta is the scalar of C
        tblis_tensor a = ndarray_to_scaled_tblis_tensor(A, alpha, conja);
        tblis_tensor b = ndarray_to_scaled_tblis_tensor(B, 1.0, conjb);
        tblis_tensor c = ndarray_to_scaled_tblis_tensor(C, beta, false);
        label_vector idx_A_vec = string_to_label_vector(idx_A);
        label_vector idx_B_vec = string_to_label_vector(idx_B);
        label_vector idx_C_vec = string_to_label_vector(idx_C);
        tblis_tensor_mult(NULL, NULL, &a, idx_A_vec.data(), &b,
                          idx_B_vec.data(), &c, idx_C_vec.data());
        tblis_tensor_free_lenstride(a);
        tblis_tensor_free_lenstride(b);
        tblis_tensor_free_lenstride(c);
      },
      nb::arg("A"), nb::arg("B"), nb::arg("C"), nb::arg("idx_A"),
      nb::arg("idx_B"), nb::arg("idx_C"), nb::arg("alpha") = 1.0,
      nb::arg("beta") = 1.0, nb::arg("conja") = false, nb::arg("conjb") = false,
      "C_[idx_C] = alpha * A_[idx_A] * B_[idx_B] + beta * C_[idx_C]");

  m.def(
      "shift",
      [](nb::ndarray<> &A, const std::string &idx_A, const dcomplex alpha = 1.0,
         const dcomplex beta = 0.0) {
        // beta is passed to tblis_tensor_shift as the scalar of A
        tblis_tensor a = ndarray_to_scaled_tblis_tensor(A, beta, false);
        label_vector idx_A_vec = string_to_label_vector(idx_A);
        // Convert alpha to tblis_scalar of proper type
        tblis_scalar alpha_scalar(alpha);
        tblis_scalar alpha_with_a_type = alpha_scalar.convert(a.type);
        tblis_tensor_shift(NULL, NULL, &alpha_with_a_type, &a,
                           idx_A_vec.data());
        tblis_tensor_free_lenstride(a);
      },
      nb::arg("A"), nb::arg("idx_A"), nb::arg("alpha") = 0.0,
      nb::arg("beta") = 1.0, "A_[idx_A] = alpha + beta * A_[idx_A]");

  m.def(
      "reduce",
      [](const nb::ndarray<> &A, const std::string &idx_A, reduce_t op,
         bool conja = false) -> nb::object {
        tblis_tensor a = ndarray_to_scaled_tblis_tensor(A, 1.0, false);
        label_vector idx_A_vec = string_to_label_vector(idx_A);
        tblis_scalar result(0.0, 0.0);
        result.type = a.type;
        len_type idx = 0;
        tblis_tensor_reduce(NULL, NULL, op, &a, idx_A_vec.data(), &result,
                            &idx);
        tblis_tensor_free_lenstride(a);
        if (op == REDUCE_MAX_ABS || op == REDUCE_MIN_ABS ||
            op == REDUCE_MAX || op == REDUCE_MIN) {
          return nb::make_tuple(pyobj_from_tblis_scalar(result), idx);
        } else {
          return pyobj_from_tblis_scalar(result);
        }
      },
      nb::arg("A"), nb::arg("idx_A"), nb::arg("op"), nb::arg("conja") = false,
      "Reduce A_[idx_A] using reduction operator op.");
}
