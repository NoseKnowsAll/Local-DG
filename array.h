#ifndef __ARRAY
#define __ARRAY

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <cmath>

#define ARRAY_MAX_NDIM 6
typedef int dgSize;

template <typename T> class array {
 protected:
  T *_data;
  int _ndim;
  dgSize _size;
  dgSize _dims[ARRAY_MAX_NDIM];
  dgSize _strides[ARRAY_MAX_NDIM]; // Omits first stride which is always 1.
  bool _alloc;
  void allocmem()
    { _data=new T[_size]; std::fill(_data, _data+_size, T(0)); _alloc=true; }
  void deallocmem()
    { if (_alloc) delete[] _data; _alloc=false; }
  void setstrides()
  {
    if (_ndim > 0) {
      std::partial_sum(_dims, _dims+_ndim, _strides, std::multiplies<dgSize>());
      _size=_strides[_ndim-1];
    }
    else {
      _size=0;
    }
  }

 public:
  typedef T value_type;

  //////////////////
  // Constructors

  // A NULL array.
  explicit array() : _data(NULL), _ndim(0), _size(0), _alloc(false) {}

  // An array filled with zeros.
  template <typename idxType>
  explicit array(int ndim, idxType const* dims)
    { resize(ndim, dims); allocmem(); }
  explicit array(dgSize s0)
    { resize(s0); allocmem(); }
  explicit array(dgSize s0, dgSize s1)
    { resize(s0, s1); allocmem(); }
  explicit array(dgSize s0, dgSize s1, dgSize s2)
    { resize(s0, s1, s2); allocmem(); }
  explicit array(dgSize s0, dgSize s1, dgSize s2, dgSize s3)
    { resize(s0, s1, s2, s3); allocmem(); }
  explicit array(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4)
    { resize(s0, s1, s2, s3, s4); allocmem(); }
  explicit array(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4, dgSize s5)
    { resize(s0, s1, s2, s3, s4, s5); allocmem(); }
  // template <typename idxType>
  // explicit array(std::initializer_list<idxType> s)
  //   { resize(s); allocmem(); }

  // A view into the data.
  template <typename idxType>
  explicit array(T *data, int ndim, idxType const* dims) : _data(data), _alloc(false)
    { resize(ndim, dims); }
  explicit array(T *data, dgSize s0) : _data(data), _alloc(false)
    { resize(s0); }
  explicit array(T *data, dgSize s0, dgSize s1) : _data(data), _alloc(false)
    { resize(s0, s1); }
  explicit array(T *data, dgSize s0, dgSize s1, dgSize s2) : _data(data), _alloc(false)
    { resize(s0, s1, s2); }
  explicit array(T *data, dgSize s0, dgSize s1, dgSize s2, dgSize s3) : _data(data), _alloc(false)
    { resize(s0, s1, s2, s3); }
  explicit array(T *data, dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4) : _data(data), _alloc(false)
    { resize(s0, s1, s2, s3, s4); }
  explicit array(T *data, dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4, dgSize s5) : _data(data), _alloc(false)
    { resize(s0, s1, s2, s3, s4, s5); }
  // template <typename idxType>
  // explicit array(T* data, std::initializer_list<idxType> s) : _data(data), _alloc(false)
  //   { resize(s); }

  // A view into the array.
  // Used when we pass arrays by value.
  array(array<T> const& a) : _data(a._data), _ndim(a._ndim), _size(a._size), _alloc(false)
  {
    std::copy(a._dims, a._dims+a._ndim, _dims);
    std::copy(a._strides, a._strides+a._ndim, _strides);
  }

  ////////////////
  // Destructor
  ~array() { deallocmem(); }

  /////////////
  // Methods

  // Resize (or reshape) the array. Does not reallocate the underlying data.
  // Use realloc if the array size will change.
  template <typename idxType>
  void resize(int ndim, idxType const* dims)
  {
    if (ndim > ARRAY_MAX_NDIM)
      throw std::invalid_argument("array<T>::resize: ndim > ARRAY_MAX_NDIM");
    _ndim=ndim;
    std::copy(dims, dims+ndim, _dims);
    setstrides();
  }
  void resize(dgSize s0)
    { dgSize dims[1] = {s0}; resize(1, dims); }
  void resize(dgSize s0, dgSize s1)
    { dgSize dims[2] = {s0, s1}; resize(2, dims); }
  void resize(dgSize s0, dgSize s1, dgSize s2)
    { dgSize dims[3] = {s0, s1, s2}; resize(3, dims); }
  void resize(dgSize s0, dgSize s1, dgSize s2, dgSize s3)
    { dgSize dims[4] = {s0, s1, s2, s3}; resize(4, dims); }
  void resize(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4)
    { dgSize dims[5] = {s0, s1, s2, s3, s4}; resize(5, dims); }
  void resize(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4, dgSize s5)
    { dgSize dims[6] = {s0, s1, s2, s3, s4, s5}; resize(6, dims); }
  // template <typename idxType>
  // void resize(std::initializer_list<idxType> s) {
  //   resize(s.size(), s.begin());
  // }

  // Update the data pointer.
  void setdata(T *data)
    { deallocmem(); _data = data; }

  // Refer to an existing array or data pointer.
  void setreference(array const& a)
    { deallocmem(); _data = (T*) a; resize(a.ndim(), a.dims()); }
  template <typename idxType>
  void setreference(T *data, int ndim, idxType const *dims)
    { deallocmem(); _data = data; resize(ndim, dims); }
  void setreference(T *data, dgSize s0)
    { deallocmem(); _data = data; resize(s0); }
  void setreference(T *data, dgSize s0, dgSize s1)
    { deallocmem(); _data = data; resize(s0, s1); }
  void setreference(T *data, dgSize s0, dgSize s1, dgSize s2)
    { deallocmem(); _data = data; resize(s0, s1, s2); }
  void setreference(T *data, dgSize s0, dgSize s1, dgSize s2, dgSize s3)
    { deallocmem(); _data = data; resize(s0, s1, s2, s3); }
  void setreference(T *data, dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4)
    { deallocmem(); _data = data; resize(s0, s1, s2, s3, s4); }
  void setreference(T *data, dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4, dgSize s5)
    { deallocmem(); _data = data; resize(s0, s1, s2, s3, s4, s5); }
  // template <typename idxType>
  // void setreference(T* data, std::initializer_list<idxType> s)
  //   { deallocmem(); _data = data; resize(s); }

  // Reallocate array.
  template <typename S>
  void realloc_like(array<S> const& rhs)
    { deallocmem(); resize(rhs.ndim(), rhs.dims()); allocmem(); }
  template <typename idxType>
  void realloc(int ndim, idxType const* dims)
    { deallocmem(); resize(ndim, dims); allocmem(); }
  void realloc(dgSize s0)
    { deallocmem(); resize(s0); allocmem(); }
  void realloc(dgSize s0, dgSize s1)
    { deallocmem(); resize(s0,s1); allocmem(); }
  void realloc(dgSize s0, dgSize s1, dgSize s2)
    { deallocmem(); resize(s0, s1, s2); allocmem(); }
  void realloc(dgSize s0, dgSize s1, dgSize s2, dgSize s3)
    { deallocmem(); resize(s0, s1, s2, s3); allocmem(); }
  void realloc(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4)
    { deallocmem(); resize(s0, s1, s2, s3, s4); allocmem(); }
  void realloc(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4, dgSize s5)
    { deallocmem(); resize(s0, s1, s2, s3, s4, s5); allocmem(); }
  // void realloc(std::initializer_list<dgSize> s)
  //   { deallocmem(); resize(s); allocmem(); }

  // Reset to the NULL array.
  void reset()
    { deallocmem(); _data=NULL; _ndim=0; _size=0; }

  // Direct access to information about the array.
  dgSize size() const { return _size; }
  int ndim() const { return _ndim; }
  dgSize const* dims() const { return _dims; }
  T* data() const { return _data; }
  bool alloc() const { return _alloc; }

  // WARNING: strides() omits the first stride which is always 1.
  dgSize const* strides() const { return _strides; }

  // Iterator access
  T* begin() const { return _data; }
  T* end() const { return _data+_size; }

  // Size and Stride of i-th dimension.
  // Negative numbers count backwards from last dimension.
  dgSize size(int i) const {
    if (i < 0) i += _ndim;
    return i<_ndim ? _dims[i] : 1;
  }
  dgSize stride(int i) const {
    if (i < 0) i += _ndim;
    return (i == 0) ? 1 : _strides[i-1];
  }

  // Element access.
  T& operator[](dgSize ix) const
    { return _data[ix]; }
  T& operator()(dgSize i0) const
    { return _data[i0]; }
  T& operator()(dgSize i0, dgSize i1) const
    { return _data[i0+_strides[0]*i1]; }
  T& operator()(dgSize i0, dgSize i1, dgSize i2) const
    { return _data[i0+_strides[0]*i1+_strides[1]*i2]; }
  T& operator()(dgSize i0, dgSize i1, dgSize i2, dgSize i3) const
    { return _data[i0+_strides[0]*i1+_strides[1]*i2+_strides[2]*i3]; }
  T& operator()(dgSize i0, dgSize i1, dgSize i2, dgSize i3, dgSize i4) const
    { return _data[i0+_strides[0]*i1+_strides[1]*i2+_strides[2]*i3+_strides[3]*i4]; }
  T& operator()(dgSize i0, dgSize i1, dgSize i2, dgSize i3, dgSize i4, dgSize i5) const
    { return _data[i0+_strides[0]*i1+_strides[1]*i2+_strides[2]*i3+_strides[3]*i4+_strides[4]*i5]; }

  // Implicit cast to the data.
  operator T*() const { return _data; }

  // Swap
  void swap(array<T>& x) {
    std::swap(this->_data, x._data);
    std::swap(this->_ndim, x._ndim);
    std::swap(this->_size, x._size);
    std::swap_ranges(this->_dims, this->_dims+ARRAY_MAX_NDIM, x._dims);
    std::swap_ranges(this->_strides, this->_strides+ARRAY_MAX_NDIM, x._strides);
    std::swap(this->_alloc, x._alloc);
  }

  void fill(T const val) const { std::fill(_data, _data+_size, val); }
  bool owndata() const { return _alloc; }

  // Fill array with a value or data
  array const& operator=(T const x) const
    { std::fill(_data, _data+_size, x); return *this; }
  array const& operator=(array const& x) const
    { std::copy(x._data, x._data+_size, _data); return *this; }
  array const& operator=(T const* x) const
    { std::copy(x, x+_size, _data); return *this; }

#define __ARRAY_OPERATOR(op)						\
  array const& operator op(T const x) const				\
    { for(dgSize i=0; i<_size; i++) _data[i] op x  ; return *this; }	\
  array const& operator op(array const& x) const			\
    { for(dgSize i=0; i<_size; i++) _data[i] op x[i]; return *this; }	\
  array const& operator op(T const* x) const				\
    { for(dgSize i=0; i<_size; i++) _data[i] op x[i]; return *this; }

  __ARRAY_OPERATOR(+=)
  __ARRAY_OPERATOR(-=)
  __ARRAY_OPERATOR(*=)
  __ARRAY_OPERATOR(/=)
#undef __ARRAY_OPERATOR

  void negate() const { for(dgSize i=0; i<_size; i++) _data[i] = -_data[i]; }
  void duplicate(array const& x) {
    realloc(x._ndim, x._dims);
    std::copy(x._data, x._data+_size, _data);
  }

  // Check if array has a specific shape.
  template <typename idxType>
  bool has_shape(int ndim, idxType const* dims) const {
    return ndim == _ndim && std::equal(dims, dims+ndim, _dims);
  }
  bool has_shape() const
    { return _ndim == 0; }
  bool has_shape(dgSize s0) const
    { return _ndim == 1 && _dims[0] == s0; }
  bool has_shape(dgSize s0, dgSize s1) const
    { return _ndim == 2 && _dims[0] == s0 && _dims[1] == s1; }
  bool has_shape(dgSize s0, dgSize s1, dgSize s2) const
    { return _ndim == 3 && _dims[0] == s0 && _dims[1] == s1 && _dims[2] == s2; }
  bool has_shape(dgSize s0, dgSize s1, dgSize s2, dgSize s3) const
    { return _ndim == 4 && _dims[0] == s0 && _dims[1] == s1 && _dims[2] == s2 && _dims[3] == s3; }
  bool has_shape(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4) const
    { return _ndim == 5 && _dims[0] == s0 && _dims[1] == s1 && _dims[2] == s2 && _dims[3] == s3 && _dims[4] == s4; }
  bool has_shape(dgSize s0, dgSize s1, dgSize s2, dgSize s3, dgSize s4, dgSize s5) const
    { return _ndim == 6 && _dims[0] == s0 && _dims[1] == s1 && _dims[2] == s2 && _dims[3] == s3 && _dims[4] == s4 && _dims[5] == s5; }
  template <typename S>
  bool has_shape_like(array<S> const& rhs) const
    { return has_shape(rhs.ndim(), rhs.dims()); }
  // bool has_shape(std::initializer_list<dgSize> dims) const {
  //   return (int) dims.size() == _ndim
  //     && std::equal(dims.begin(), dims.end(), _dims);
  // }
};

template <typename T> inline void swap(array<T>& x, array<T>& y)
{ x.swap(y); }

template <typename T> inline void sort(array<T> const& a)
{ std::sort((T*)a, (T*)a+a.size()); }

template <typename T> inline T max(array<T> const& a)
{ return *std::max_element((T const*)a, (T const*)a+a.size()); }

template <typename T> inline T min(array<T> const& a)
{ return *std::min_element((T const*)a, (T const*)a+a.size()); }

template <typename T> inline T sum(array<T> const& a)
{ return std::accumulate((T const*)a, (T const*)a+a.size(),T(0)); }

template <typename T> inline T prod(array<T> const& a)
{ return std::accumulate((T const*)a, (T const*)a+a.size(), T(1), std::multiplies<T>()); }

template <typename T> inline T norm(array<T> const& a)
{ return std::sqrt(std::inner_product((T const*)a, (T const*)a+a.size(), (T const*)a, T(0))); }

template <typename T> inline T infnorm(array<T> const& a) {
  T nrm=0.0;
  for (dgSize i=0; i<a.size(); i++) {
    T absa=std::abs(a[i]);
    if (absa>nrm)
      nrm=absa;
  }
  return nrm;
}

template <typename T> inline bool anynan(array<T> const& a) {
  for (dgSize i=0; i<a.size(); i++)
    if (std::isnan(a[i]))
      return true;
  return false;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const array<T>& a) {
  for (dgSize i = 0; i < a.size()-1; ++i) {
    out << a[i] << "\n";
  }
  out << a[a.size()-1];
  return out;
}

typedef array<double> darray;
typedef array<float> farray;
typedef array<int> iarray;
typedef array<long int> larray;
typedef array<bool> barray;

#endif
