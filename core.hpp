/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_AFFINE3_HPP
#define OPENCV_CORE_AFFINE3_HPP

#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{

//! @addtogroup core
//! @{

    /** @brief Affine transform
      @todo document
     */
    template<typename T>
    class Affine3
    {
    public:
        typedef T float_type;
        typedef Matx<float_type, 3, 3> Mat3;
        typedef Matx<float_type, 4, 4> Mat4;
        typedef Vec<float_type, 3> Vec3;

        Affine3();

        //! Augmented affine matrix
        Affine3(const Mat4& affine);

        //! Rotation matrix
        Affine3(const Mat3& R, const Vec3& t = Vec3::all(0));

        //! Rodrigues vector
        Affine3(const Vec3& rvec, const Vec3& t = Vec3::all(0));

        //! Combines all contructors above. Supports 4x4, 4x3, 3x3, 1x3, 3x1 sizes of data matrix
        explicit Affine3(const Mat& data, const Vec3& t = Vec3::all(0));

        //! From 16th element array
        explicit Affine3(const float_type* vals);

        //! Create identity transform
        static Affine3 Identity();

        //! Rotation matrix
        void rotation(const Mat3& R);

        //! Rodrigues vector
        void rotation(const Vec3& rvec);

        //! Combines rotation methods above. Suports 3x3, 1x3, 3x1 sizes of data matrix;
        void rotation(const Mat& data);

        void linear(const Mat3& L);
        void translation(const Vec3& t);

        Mat3 rotation() const;
        Mat3 linear() const;
        Vec3 translation() const;

        //! Rodrigues vector
        Vec3 rvec() const;

        Affine3 inv(int method = cv::DECOMP_SVD) const;

        //! a.rotate(R) is equivalent to Affine(R, 0) * a;
        Affine3 rotate(const Mat3& R) const;

        //! a.rotate(rvec) is equivalent to Affine(rvec, 0) * a;
        Affine3 rotate(const Vec3& rvec) const;

        //! a.translate(t) is equivalent to Affine(E, t) * a;
        Affine3 translate(const Vec3& t) const;

        //! a.concatenate(affine) is equivalent to affine * a;
        Affine3 concatenate(const Affine3& affine) const;

        template <typename Y> operator Affine3<Y>() const;

        template <typename Y> Affine3<Y> cast() const;

        Mat4 matrix;

#if defined EIGEN_WORLD_VERSION && defined EIGEN_GEOMETRY_MODULE_H
        Affine3(const Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>& affine);
        Affine3(const Eigen::Transform<T, 3, Eigen::Affine>& affine);
        operator Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>() const;
        operator Eigen::Transform<T, 3, Eigen::Affine>() const;
#endif
    };

    template<typename T> static
    Affine3<T> operator*(const Affine3<T>& affine1, const Affine3<T>& affine2);

    template<typename T, typename V> static
    V operator*(const Affine3<T>& affine, const V& vector);

    typedef Affine3<float> Affine3f;
    typedef Affine3<double> Affine3d;

    static Vec3f operator*(const Affine3f& affine, const Vec3f& vector);
    static Vec3d operator*(const Affine3d& affine, const Vec3d& vector);

    template<typename _Tp> class DataType< Affine3<_Tp> >
    {
    public:
        typedef Affine3<_Tp>                               value_type;
        typedef Affine3<typename DataType<_Tp>::work_type> work_type;
        typedef _Tp                                        channel_type;

        enum { generic_type = 0,
               depth        = DataType<channel_type>::depth,
               channels     = 16,
               fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
               type         = CV_MAKETYPE(depth, channels)
             };

        typedef Vec<channel_type, channels> vec_type;
    };

//! @} core

}

//! @cond IGNORED

///////////////////////////////////////////////////////////////////////////////////
// Implementaiton

template<typename T> inline
cv::Affine3<T>::Affine3()
    : matrix(Mat4::eye())
{}

template<typename T> inline
cv::Affine3<T>::Affine3(const Mat4& affine)
    : matrix(affine)
{}

template<typename T> inline
cv::Affine3<T>::Affine3(const Mat3& R, const Vec3& t)
{
    rotation(R);
    translation(t);
    matrix.val[12] = matrix.val[13] = matrix.val[14] = 0;
    matrix.val[15] = 1;
}

template<typename T> inline
cv::Affine3<T>::Affine3(const Vec3& _rvec, const Vec3& t)
{
    rotation(_rvec);
    translation(t);
    matrix.val[12] = matrix.val[13] = matrix.val[14] = 0;
    matrix.val[15] = 1;
}

template<typename T> inline
cv::Affine3<T>::Affine3(const cv::Mat& data, const Vec3& t)
{
    CV_Assert(data.type() == cv::DataType<T>::type);

    if (data.cols == 4 && data.rows == 4)
    {
        data.copyTo(matrix);
        return;
    }
    else if (data.cols == 4 && data.rows == 3)
    {
        rotation(data(Rect(0, 0, 3, 3)));
        translation(data(Rect(3, 0, 1, 3)));
    }
    else
    {
        rotation(data);
        translation(t);
    }

    matrix.val[12] = matrix.val[13] = matrix.val[14] = 0;
    matrix.val[15] = 1;
}

template<typename T> inline
cv::Affine3<T>::Affine3(const float_type* vals) : matrix(vals)
{}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::Identity()
{
    return Affine3<T>(cv::Affine3<T>::Mat4::eye());
}

template<typename T> inline
void cv::Affine3<T>::rotation(const Mat3& R)
{
    linear(R);
}

template<typename T> inline
void cv::Affine3<T>::rotation(const Vec3& _rvec)
{
    double theta = norm(_rvec);

    if (theta < DBL_EPSILON)
        rotation(Mat3::eye());
    else
    {
        double c = std::cos(theta);
        double s = std::sin(theta);
        double c1 = 1. - c;
        double itheta = (theta != 0) ? 1./theta : 0.;

        Point3_<T> r = _rvec*itheta;

        Mat3 rrt( r.x*r.x, r.x*r.y, r.x*r.z, r.x*r.y, r.y*r.y, r.y*r.z, r.x*r.z, r.y*r.z, r.z*r.z );
        Mat3 r_x( 0, -r.z, r.y, r.z, 0, -r.x, -r.y, r.x, 0 );

        // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
        // where [r_x] is [0 -rz ry; rz 0 -rx; -ry rx 0]
        Mat3 R = c*Mat3::eye() + c1*rrt + s*r_x;

        rotation(R);
    }
}

//Combines rotation methods above. Suports 3x3, 1x3, 3x1 sizes of data matrix;
template<typename T> inline
void cv::Affine3<T>::rotation(const cv::Mat& data)
{
    CV_Assert(data.type() == cv::DataType<T>::type);

    if (data.cols == 3 && data.rows == 3)
    {
        Mat3 R;
        data.copyTo(R);
        rotation(R);
    }
    else if ((data.cols == 3 && data.rows == 1) || (data.cols == 1 && data.rows == 3))
    {
        Vec3 _rvec;
        data.reshape(1, 3).copyTo(_rvec);
        rotation(_rvec);
    }
    else
        CV_Assert(!"Input marix can be 3x3, 1x3 or 3x1");
}

template<typename T> inline
void cv::Affine3<T>::linear(const Mat3& L)
{
    matrix.val[0] = L.val[0]; matrix.val[1] = L.val[1];  matrix.val[ 2] = L.val[2];
    matrix.val[4] = L.val[3]; matrix.val[5] = L.val[4];  matrix.val[ 6] = L.val[5];
    matrix.val[8] = L.val[6]; matrix.val[9] = L.val[7];  matrix.val[10] = L.val[8];
}

template<typename T> inline
void cv::Affine3<T>::translation(const Vec3& t)
{
    matrix.val[3] = t[0]; matrix.val[7] = t[1]; matrix.val[11] = t[2];
}

template<typename T> inline
typename cv::Affine3<T>::Mat3 cv::Affine3<T>::rotation() const
{
    return linear();
}

template<typename T> inline
typename cv::Affine3<T>::Mat3 cv::Affine3<T>::linear() const
{
    typename cv::Affine3<T>::Mat3 R;
    R.val[0] = matrix.val[0];  R.val[1] = matrix.val[1];  R.val[2] = matrix.val[ 2];
    R.val[3] = matrix.val[4];  R.val[4] = matrix.val[5];  R.val[5] = matrix.val[ 6];
    R.val[6] = matrix.val[8];  R.val[7] = matrix.val[9];  R.val[8] = matrix.val[10];
    return R;
}

template<typename T> inline
typename cv::Affine3<T>::Vec3 cv::Affine3<T>::translation() const
{
    return Vec3(matrix.val[3], matrix.val[7], matrix.val[11]);
}

template<typename T> inline
typename cv::Affine3<T>::Vec3 cv::Affine3<T>::rvec() const
{
    cv::Vec3d w;
    cv::Matx33d u, vt, R = rotation();
    cv::SVD::compute(R, w, u, vt, cv::SVD::FULL_UV + cv::SVD::MODIFY_A);
    R = u * vt;

    double rx = R.val[7] - R.val[5];
    double ry = R.val[2] - R.val[6];
    double rz = R.val[3] - R.val[1];

    double s = std::sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.val[0] + R.val[4] + R.val[8] - 1) * 0.5;
    c = c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c;
    double theta = acos(c);

    if( s < 1e-5 )
    {
        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            double t;
            t = (R.val[0] + 1) * 0.5;
            rx = std::sqrt(std::max(t, 0.0));
            t = (R.val[4] + 1) * 0.5;
            ry = std::sqrt(std::max(t, 0.0)) * (R.val[1] < 0 ? -1.0 : 1.0);
            t = (R.val[8] + 1) * 0.5;
            rz = std::sqrt(std::max(t, 0.0)) * (R.val[2] < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R.val[5] > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= std::sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }

    return cv::Vec3d(rx, ry, rz);
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::inv(int method) const
{
    return matrix.inv(method);
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::rotate(const Mat3& R) const
{
    Mat3 Lc = linear();
    Vec3 tc = translation();
    Mat4 result;
    result.val[12] = result.val[13] = result.val[14] = 0;
    result.val[15] = 1;

    for(int j = 0; j < 3; ++j)
    {
        for(int i = 0; i < 3; ++i)
        {
            float_type value = 0;
            for(int k = 0; k < 3; ++k)
                value += R(j, k) * Lc(k, i);
            result(j, i) = value;
        }

        result(j, 3) = R.row(j).dot(tc.t());
    }
    return result;
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::rotate(const Vec3& _rvec) const
{
    return rotate(Affine3f(_rvec).rotation());
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::translate(const Vec3& t) const
{
    Mat4 m = matrix;
    m.val[ 3] += t[0];
    m.val[ 7] += t[1];
    m.val[11] += t[2];
    return m;
}

template<typename T> inline
cv::Affine3<T> cv::Affine3<T>::concatenate(const Affine3<T>& affine) const
{
    return (*this).rotate(affine.rotation()).translate(affine.translation());
}

template<typename T> template <typename Y> inline
cv::Affine3<T>::operator Affine3<Y>() const
{
    return Affine3<Y>(matrix);
}

template<typename T> template <typename Y> inline
cv::Affine3<Y> cv::Affine3<T>::cast() const
{
    return Affine3<Y>(matrix);
}

template<typename T> inline
cv::Affine3<T> cv::operator*(const cv::Affine3<T>& affine1, const cv::Affine3<T>& affine2)
{
    return affine2.concatenate(affine1);
}

template<typename T, typename V> inline
V cv::operator*(const cv::Affine3<T>& affine, const V& v)
{
    const typename Affine3<T>::Mat4& m = affine.matrix;

    V r;
    r.x = m.val[0] * v.x + m.val[1] * v.y + m.val[ 2] * v.z + m.val[ 3];
    r.y = m.val[4] * v.x + m.val[5] * v.y + m.val[ 6] * v.z + m.val[ 7];
    r.z = m.val[8] * v.x + m.val[9] * v.y + m.val[10] * v.z + m.val[11];
    return r;
}

static inline
cv::Vec3f cv::operator*(const cv::Affine3f& affine, const cv::Vec3f& v)
{
    const cv::Matx44f& m = affine.matrix;
    cv::Vec3f r;
    r.val[0] = m.val[0] * v[0] + m.val[1] * v[1] + m.val[ 2] * v[2] + m.val[ 3];
    r.val[1] = m.val[4] * v[0] + m.val[5] * v[1] + m.val[ 6] * v[2] + m.val[ 7];
    r.val[2] = m.val[8] * v[0] + m.val[9] * v[1] + m.val[10] * v[2] + m.val[11];
    return r;
}

static inline
cv::Vec3d cv::operator*(const cv::Affine3d& affine, const cv::Vec3d& v)
{
    const cv::Matx44d& m = affine.matrix;
    cv::Vec3d r;
    r.val[0] = m.val[0] * v[0] + m.val[1] * v[1] + m.val[ 2] * v[2] + m.val[ 3];
    r.val[1] = m.val[4] * v[0] + m.val[5] * v[1] + m.val[ 6] * v[2] + m.val[ 7];
    r.val[2] = m.val[8] * v[0] + m.val[9] * v[1] + m.val[10] * v[2] + m.val[11];
    return r;
}



#if defined EIGEN_WORLD_VERSION && defined EIGEN_GEOMETRY_MODULE_H

template<typename T> inline
cv::Affine3<T>::Affine3(const Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>& affine)
{
    cv::Mat(4, 4, cv::DataType<T>::type, affine.matrix().data()).copyTo(matrix);
}

template<typename T> inline
cv::Affine3<T>::Affine3(const Eigen::Transform<T, 3, Eigen::Affine>& affine)
{
    Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)> a = affine;
    cv::Mat(4, 4, cv::DataType<T>::type, a.matrix().data()).copyTo(matrix);
}

template<typename T> inline
cv::Affine3<T>::operator Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>() const
{
    Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)> r;
    cv::Mat hdr(4, 4, cv::DataType<T>::type, r.matrix().data());
    cv::Mat(matrix, false).copyTo(hdr);
    return r;
}

template<typename T> inline
cv::Affine3<T>::operator Eigen::Transform<T, 3, Eigen::Affine>() const
{
    return this->operator Eigen::Transform<T, 3, Eigen::Affine, (Eigen::RowMajor)>();
}

#endif /* defined EIGEN_WORLD_VERSION && defined EIGEN_GEOMETRY_MODULE_H */

//! @endcond

#endif /* __cplusplus */

#endif /* OPENCV_CORE_AFFINE3_HPP */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_BASE_HPP
#define OPENCV_CORE_BASE_HPP

#ifndef __cplusplus
#  error base.hpp header must be compiled as C++
#endif

#include "opencv2/opencv_modules.hpp"

#include <climits>
#include <algorithm>

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"

namespace cv
{

//! @addtogroup core_utils
//! @{

namespace Error {
//! error codes
enum Code {
    StsOk=                       0,  //!< everything is ok
    StsBackTrace=               -1,  //!< pseudo error for back trace
    StsError=                   -2,  //!< unknown /unspecified error
    StsInternal=                -3,  //!< internal error (bad state)
    StsNoMem=                   -4,  //!< insufficient memory
    StsBadArg=                  -5,  //!< function arg/param is bad
    StsBadFunc=                 -6,  //!< unsupported function
    StsNoConv=                  -7,  //!< iteration didn't converge
    StsAutoTrace=               -8,  //!< tracing
    HeaderIsNull=               -9,  //!< image header is NULL
    BadImageSize=              -10,  //!< image size is invalid
    BadOffset=                 -11,  //!< offset is invalid
    BadDataPtr=                -12,  //!<
    BadStep=                   -13,  //!< image step is wrong, this may happen for a non-continuous matrix.
    BadModelOrChSeq=           -14,  //!<
    BadNumChannels=            -15,  //!< bad number of channels, for example, some functions accept only single channel matrices.
    BadNumChannel1U=           -16,  //!<
    BadDepth=                  -17,  //!< input image depth is not supported by the function
    BadAlphaChannel=           -18,  //!<
    BadOrder=                  -19,  //!< number of dimensions is out of range
    BadOrigin=                 -20,  //!< incorrect input origin
    BadAlign=                  -21,  //!< incorrect input align
    BadCallBack=               -22,  //!<
    BadTileSize=               -23,  //!<
    BadCOI=                    -24,  //!< input COI is not supported
    BadROISize=                -25,  //!< incorrect input roi
    MaskIsTiled=               -26,  //!<
    StsNullPtr=                -27,  //!< null pointer
    StsVecLengthErr=           -28,  //!< incorrect vector length
    StsFilterStructContentErr= -29,  //!< incorrect filter structure content
    StsKernelStructContentErr= -30,  //!< incorrect transform kernel content
    StsFilterOffsetErr=        -31,  //!< incorrect filter offset value
    StsBadSize=                -201, //!< the input/output structure size is incorrect
    StsDivByZero=              -202, //!< division by zero
    StsInplaceNotSupported=    -203, //!< in-place operation is not supported
    StsObjectNotFound=         -204, //!< request can't be completed
    StsUnmatchedFormats=       -205, //!< formats of input/output arrays differ
    StsBadFlag=                -206, //!< flag is wrong or not supported
    StsBadPoint=               -207, //!< bad CvPoint
    StsBadMask=                -208, //!< bad format of mask (neither 8uC1 nor 8sC1)
    StsUnmatchedSizes=         -209, //!< sizes of input/output structures do not match
    StsUnsupportedFormat=      -210, //!< the data format/type is not supported by the function
    StsOutOfRange=             -211, //!< some of parameters are out of range
    StsParseError=             -212, //!< invalid syntax/structure of the parsed file
    StsNotImplemented=         -213, //!< the requested function/feature is not implemented
    StsBadMemBlock=            -214, //!< an allocated block has been corrupted
    StsAssert=                 -215, //!< assertion failed
    GpuNotSupported=           -216, //!< no CUDA support
    GpuApiCallError=           -217, //!< GPU API call error
    OpenGlNotSupported=        -218, //!< no OpenGL support
    OpenGlApiCallError=        -219, //!< OpenGL API call error
    OpenCLApiCallError=        -220, //!< OpenCL API call error
    OpenCLDoubleNotSupported=  -221,
    OpenCLInitError=           -222, //!< OpenCL initialization error
    OpenCLNoAMDBlasFft=        -223
};
} //Error

//! @} core_utils

//! @addtogroup core_array
//! @{

//! matrix decomposition types
enum DecompTypes {
    /** Gaussian elimination with the optimal pivot element chosen. */
    DECOMP_LU       = 0,
    /** singular value decomposition (SVD) method; the system can be over-defined and/or the matrix
    src1 can be singular */
    DECOMP_SVD      = 1,
    /** eigenvalue decomposition; the matrix src1 must be symmetrical */
    DECOMP_EIG      = 2,
    /** Cholesky \f$LL^T\f$ factorization; the matrix src1 must be symmetrical and positively
    defined */
    DECOMP_CHOLESKY = 3,
    /** QR factorization; the system can be over-defined and/or the matrix src1 can be singular */
    DECOMP_QR       = 4,
    /** while all the previous flags are mutually exclusive, this flag can be used together with
    any of the previous; it means that the normal equations
    \f$\texttt{src1}^T\cdot\texttt{src1}\cdot\texttt{dst}=\texttt{src1}^T\texttt{src2}\f$ are
    solved instead of the original system
    \f$\texttt{src1}\cdot\texttt{dst}=\texttt{src2}\f$ */
    DECOMP_NORMAL   = 16
};

/** norm types
- For one array:
\f[norm =  \forkthree{\|\texttt{src1}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
{ \| \texttt{src1} \| _{L_1} =  \sum _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
{ \| \texttt{src1} \| _{L_2} =  \sqrt{\sum_I \texttt{src1}(I)^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }\f]

- Absolute norm for two arrays
\f[norm =  \forkthree{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
{ \| \texttt{src1} - \texttt{src2} \| _{L_1} =  \sum _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
{ \| \texttt{src1} - \texttt{src2} \| _{L_2} =  \sqrt{\sum_I (\texttt{src1}(I) - \texttt{src2}(I))^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }\f]

- Relative norm for two arrays
\f[norm =  \forkthree{\frac{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}}    }{\|\texttt{src2}\|_{L_{\infty}} }}{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_INF}\) }
{ \frac{\|\texttt{src1}-\texttt{src2}\|_{L_1} }{\|\texttt{src2}\|_{L_1}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_L1}\) }
{ \frac{\|\texttt{src1}-\texttt{src2}\|_{L_2} }{\|\texttt{src2}\|_{L_2}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_L2}\) }\f]

As example for one array consider the function \f$r(x)= \begin{pmatrix} x \\ 1-x \end{pmatrix}, x \in [-1;1]\f$.
The \f$ L_{1}, L_{2} \f$ and \f$ L_{\infty} \f$ norm for the sample value \f$r(-1) = \begin{pmatrix} -1 \\ 2 \end{pmatrix}\f$
is calculated as follows
\f{align*}
    \| r(-1) \|_{L_1} &= |-1| + |2| = 3 \\
    \| r(-1) \|_{L_2} &= \sqrt{(-1)^{2} + (2)^{2}} = \sqrt{5} \\
    \| r(-1) \|_{L_\infty} &= \max(|-1|,|2|) = 2
\f}
and for \f$r(0.5) = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}\f$ the calculation is
\f{align*}
    \| r(0.5) \|_{L_1} &= |0.5| + |0.5| = 1 \\
    \| r(0.5) \|_{L_2} &= \sqrt{(0.5)^{2} + (0.5)^{2}} = \sqrt{0.5} \\
    \| r(0.5) \|_{L_\infty} &= \max(|0.5|,|0.5|) = 0.5.
\f}
The following graphic shows all values for the three norm functions \f$\| r(x) \|_{L_1}, \| r(x) \|_{L_2}\f$ and \f$\| r(x) \|_{L_\infty}\f$.
It is notable that the \f$ L_{1} \f$ norm forms the upper and the \f$ L_{\infty} \f$ norm forms the lower border for the example function \f$ r(x) \f$.
![Graphs for the different norm functions from the above example](pics/NormTypes_OneArray_1-2-INF.png)
 */
enum NormTypes { NORM_INF       = 1,
                 NORM_L1        = 2,
                 NORM_L2        = 4,
                 NORM_L2SQR     = 5,
                 NORM_HAMMING   = 6,
                 NORM_HAMMING2  = 7,
                 NORM_TYPE_MASK = 7,
                 NORM_RELATIVE  = 8, //!< flag
                 NORM_MINMAX    = 32 //!< flag
               };

//! comparison types
enum CmpTypes { CMP_EQ = 0, //!< src1 is equal to src2.
                CMP_GT = 1, //!< src1 is greater than src2.
                CMP_GE = 2, //!< src1 is greater than or equal to src2.
                CMP_LT = 3, //!< src1 is less than src2.
                CMP_LE = 4, //!< src1 is less than or equal to src2.
                CMP_NE = 5  //!< src1 is unequal to src2.
              };

//! generalized matrix multiplication flags
enum GemmFlags { GEMM_1_T = 1, //!< transposes src1
                 GEMM_2_T = 2, //!< transposes src2
                 GEMM_3_T = 4 //!< transposes src3
               };

enum DftFlags {
    /** performs an inverse 1D or 2D transform instead of the default forward
        transform. */
    DFT_INVERSE        = 1,
    /** scales the result: divide it by the number of array elements. Normally, it is
        combined with DFT_INVERSE. */
    DFT_SCALE          = 2,
    /** performs a forward or inverse transform of every individual row of the input
        matrix; this flag enables you to transform multiple vectors simultaneously and can be used to
        decrease the overhead (which is sometimes several times larger than the processing itself) to
        perform 3D and higher-dimensional transformations and so forth.*/
    DFT_ROWS           = 4,
    /** performs a forward transformation of 1D or 2D real array; the result,
        though being a complex array, has complex-conjugate symmetry (*CCS*, see the function
        description below for details), and such an array can be packed into a real array of the same
        size as input, which is the fastest option and which is what the function does by default;
        however, you may wish to get a full complex array (for simpler spectrum analysis, and so on) -
        pass the flag to enable the function to produce a full-size complex output array. */
    DFT_COMPLEX_OUTPUT = 16,
    /** performs an inverse transformation of a 1D or 2D complex array; the
        result is normally a complex array of the same size, however, if the input array has
        conjugate-complex symmetry (for example, it is a result of forward transformation with
        DFT_COMPLEX_OUTPUT flag), the output is a real array; while the function itself does not
        check whether the input is symmetrical or not, you can pass the flag and then the function
        will assume the symmetry and produce the real output array (note that when the input is packed
        into a real array and inverse transformation is executed, the function treats the input as a
        packed complex-conjugate symmetrical array, and the output will also be a real array). */
    DFT_REAL_OUTPUT    = 32,
    /** specifies that input is complex input. If this flag is set, the input must have 2 channels.
        On the other hand, for backwards compatibility reason, if input has 2 channels, input is
        already considered complex. */
    DFT_COMPLEX_INPUT  = 64,
    /** performs an inverse 1D or 2D transform instead of the default forward transform. */
    DCT_INVERSE        = DFT_INVERSE,
    /** performs a forward or inverse transform of every individual row of the input
        matrix. This flag enables you to transform multiple vectors simultaneously and can be used to
        decrease the overhead (which is sometimes several times larger than the processing itself) to
        perform 3D and higher-dimensional transforms and so forth.*/
    DCT_ROWS           = DFT_ROWS
};

//! Various border types, image boundaries are denoted with `|`
//! @see borderInterpolate, copyMakeBorder
enum BorderTypes {
    BORDER_CONSTANT    = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
    BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
    BORDER_WRAP        = 3, //!< `cdefgh|abcdefgh|abcdefg`
    BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
    BORDER_TRANSPARENT = 5, //!< `uvwxyz|absdefgh|ijklmno`

    BORDER_REFLECT101  = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
    BORDER_DEFAULT     = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
    BORDER_ISOLATED    = 16 //!< do not look outside of ROI
};

//! @} core_array

//! @addtogroup core_utils
//! @{

//! @cond IGNORED

//////////////// static assert /////////////////
#define CVAUX_CONCAT_EXP(a, b) a##b
#define CVAUX_CONCAT(a, b) CVAUX_CONCAT_EXP(a,b)

#if defined(__clang__)
#  ifndef __has_extension
#    define __has_extension __has_feature /* compatibility, for older versions of clang */
#  endif
#  if __has_extension(cxx_static_assert)
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  elif __has_extension(c_static_assert)
#    define CV_StaticAssert(condition, reason)    _Static_assert((condition), reason " " #condition)
#  endif
#elif defined(__GNUC__)
#  if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L)
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#elif defined(_MSC_VER)
#  if _MSC_VER >= 1600 /* MSVC 10 */
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#endif
#ifndef CV_StaticAssert
#  if !defined(__clang__) && defined(__GNUC__) && (__GNUC__*100 + __GNUC_MINOR__ > 302)
#    define CV_StaticAssert(condition, reason) ({ extern int __attribute__((error("CV_StaticAssert: " reason " " #condition))) CV_StaticAssert(); ((condition) ? 0 : CV_StaticAssert()); })
#  else
     template <bool x> struct CV_StaticAssert_failed;
     template <> struct CV_StaticAssert_failed<true> { enum { val = 1 }; };
     template<int x> struct CV_StaticAssert_test {};
#    define CV_StaticAssert(condition, reason)\
       typedef cv::CV_StaticAssert_test< sizeof(cv::CV_StaticAssert_failed< static_cast<bool>(condition) >) > CVAUX_CONCAT(CV_StaticAssert_failed_at_, __LINE__)
#  endif
#endif

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(_MSC_VER)
    #define CV_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
    #define CV_DO_PRAGMA(x) _Pragma (#x)
#else
    #define CV_DO_PRAGMA(x)
#endif

#ifdef _MSC_VER
#define CV_SUPPRESS_DEPRECATED_START \
    CV_DO_PRAGMA(warning(push)) \
    CV_DO_PRAGMA(warning(disable: 4996))
#define CV_SUPPRESS_DEPRECATED_END CV_DO_PRAGMA(warning(pop))
#elif defined (__clang__) || ((__GNUC__)  && (__GNUC__*100 + __GNUC_MINOR__ > 405))
#define CV_SUPPRESS_DEPRECATED_START \
    CV_DO_PRAGMA(GCC diagnostic push) \
    CV_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define CV_SUPPRESS_DEPRECATED_END CV_DO_PRAGMA(GCC diagnostic pop)
#else
#define CV_SUPPRESS_DEPRECATED_START
#define CV_SUPPRESS_DEPRECATED_END
#endif
#define CV_UNUSED(name) (void)name
//! @endcond

/*! @brief Signals an error and raises the exception.

By default the function prints information about the error to stderr,
then it either stops if setBreakOnError() had been called before or raises the exception.
It is possible to alternate error processing by using redirectError().
@param _code - error code (Error::Code)
@param _err - error description
@param _func - function name. Available only when the compiler supports getting it
@param _file - source file name where the error has occurred
@param _line - line number in the source file where the error has occurred
@see CV_Error, CV_Error_, CV_ErrorNoReturn, CV_ErrorNoReturn_, CV_Assert, CV_DbgAssert
 */
CV_EXPORTS void error(int _code, const String& _err, const char* _func, const char* _file, int _line);

#ifdef __GNUC__
# if defined __clang__ || defined __APPLE__
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Winvalid-noreturn"
# endif
#endif

/** same as cv::error, but does not return */
CV_INLINE CV_NORETURN void errorNoReturn(int _code, const String& _err, const char* _func, const char* _file, int _line)
{
    error(_code, _err, _func, _file, _line);
#ifdef __GNUC__
# if !defined __clang__ && !defined __APPLE__
    // this suppresses this warning: "noreturn" function does return [enabled by default]
    __builtin_trap();
    // or use infinite loop: for (;;) {}
# endif
#endif
}
#ifdef __GNUC__
# if defined __clang__ || defined __APPLE__
#   pragma GCC diagnostic pop
# endif
#endif

#if defined __GNUC__
#define CV_Func __func__
#elif defined _MSC_VER
#define CV_Func __FUNCTION__
#else
#define CV_Func ""
#endif

#ifdef CV_STATIC_ANALYSIS
// In practice, some macro are not processed correctly (noreturn is not detected).
// We need to use simplified definition for them.
#define CV_Error(...) do { abort(); } while (0)
#define CV_Error_(...) do { abort(); } while (0)
#define CV_Assert(cond) do { if (!(cond)) abort(); } while (0)
#define CV_ErrorNoReturn(...) do { abort(); } while (0)
#define CV_ErrorNoReturn_(...) do { abort(); } while (0)

#else // CV_STATIC_ANALYSIS

/** @brief Call the error handler.

Currently, the error handler prints the error code and the error message to the standard
error stream `stderr`. In the Debug configuration, it then provokes memory access violation, so that
the execution stack and all the parameters can be analyzed by the debugger. In the Release
configuration, the exception is thrown.

@param code one of Error::Code
@param msg error message
*/
#define CV_Error( code, msg ) cv::error( code, msg, CV_Func, __FILE__, __LINE__ )

/**  @brief Call the error handler.

This macro can be used to construct an error message on-fly to include some dynamic information,
for example:
@code
    // note the extra parentheses around the formatted text message
    CV_Error_( CV_StsOutOfRange,
    ("the value at (%d, %d)=%g is out of range", badPt.x, badPt.y, badValue));
@endcode
@param code one of Error::Code
@param args printf-like formatted error message in parentheses
*/
#define CV_Error_( code, args ) cv::error( code, cv::format args, CV_Func, __FILE__, __LINE__ )

/** @brief Checks a condition at runtime and throws exception if it fails

The macros CV_Assert (and CV_DbgAssert(expr)) evaluate the specified expression. If it is 0, the macros
raise an error (see cv::error). The macro CV_Assert checks the condition in both Debug and Release
configurations while CV_DbgAssert is only retained in the Debug configuration.
*/
#define CV_Assert( expr ) if(!!(expr)) ; else cv::error( cv::Error::StsAssert, #expr, CV_Func, __FILE__, __LINE__ )

/** same as CV_Error(code,msg), but does not return */
#define CV_ErrorNoReturn( code, msg ) cv::errorNoReturn( code, msg, CV_Func, __FILE__, __LINE__ )

/** same as CV_Error_(code,args), but does not return */
#define CV_ErrorNoReturn_( code, args ) cv::errorNoReturn( code, cv::format args, CV_Func, __FILE__, __LINE__ )

#endif // CV_STATIC_ANALYSIS

/** replaced with CV_Assert(expr) in Debug configuration */
#ifdef _DEBUG
#  define CV_DbgAssert(expr) CV_Assert(expr)
#else
#  define CV_DbgAssert(expr)
#endif

/*
 * Hamming distance functor - counts the bit differences between two strings - useful for the Brief descriptor
 * bit count of A exclusive XOR'ed with B
 */
struct CV_EXPORTS Hamming
{
    enum { normType = NORM_HAMMING };
    typedef unsigned char ValueType;
    typedef int ResultType;

    /** this will count the bits in a ^ b
     */
    ResultType operator()( const unsigned char* a, const unsigned char* b, int size ) const;
};

typedef Hamming HammingLUT;

/////////////////////////////////// inline norms ////////////////////////////////////

template<typename _Tp> inline _Tp cv_abs(_Tp x) { return std::abs(x); }
inline int cv_abs(uchar x) { return x; }
inline int cv_abs(schar x) { return std::abs(x); }
inline int cv_abs(ushort x) { return x; }
inline int cv_abs(short x) { return std::abs(x); }

template<typename _Tp, typename _AccTp> static inline
_AccTp normL2Sqr(const _Tp* a, int n)
{
    _AccTp s = 0;
    int i=0;
#if CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
    {
        _AccTp v0 = a[i], v1 = a[i+1], v2 = a[i+2], v3 = a[i+3];
        s += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
#endif
    for( ; i < n; i++ )
    {
        _AccTp v = a[i];
        s += v*v;
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL1(const _Tp* a, int n)
{
    _AccTp s = 0;
    int i = 0;
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4 )
    {
        s += (_AccTp)cv_abs(a[i]) + (_AccTp)cv_abs(a[i+1]) +
            (_AccTp)cv_abs(a[i+2]) + (_AccTp)cv_abs(a[i+3]);
    }
#endif
    for( ; i < n; i++ )
        s += cv_abs(a[i]);
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normInf(const _Tp* a, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
        s = std::max(s, (_AccTp)cv_abs(a[i]));
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL2Sqr(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    int i= 0;
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4 )
    {
        _AccTp v0 = _AccTp(a[i] - b[i]), v1 = _AccTp(a[i+1] - b[i+1]), v2 = _AccTp(a[i+2] - b[i+2]), v3 = _AccTp(a[i+3] - b[i+3]);
        s += v0*v0 + v1*v1 + v2*v2 + v3*v3;
    }
#endif
    for( ; i < n; i++ )
    {
        _AccTp v = _AccTp(a[i] - b[i]);
        s += v*v;
    }
    return s;
}

static inline float normL2Sqr(const float* a, const float* b, int n)
{
    float s = 0.f;
    for( int i = 0; i < n; i++ )
    {
        float v = a[i] - b[i];
        s += v*v;
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normL1(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    int i= 0;
#if CV_ENABLE_UNROLLED
    for(; i <= n - 4; i += 4 )
    {
        _AccTp v0 = _AccTp(a[i] - b[i]), v1 = _AccTp(a[i+1] - b[i+1]), v2 = _AccTp(a[i+2] - b[i+2]), v3 = _AccTp(a[i+3] - b[i+3]);
        s += std::abs(v0) + std::abs(v1) + std::abs(v2) + std::abs(v3);
    }
#endif
    for( ; i < n; i++ )
    {
        _AccTp v = _AccTp(a[i] - b[i]);
        s += std::abs(v);
    }
    return s;
}

inline float normL1(const float* a, const float* b, int n)
{
    float s = 0.f;
    for( int i = 0; i < n; i++ )
    {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}

inline int normL1(const uchar* a, const uchar* b, int n)
{
    int s = 0;
    for( int i = 0; i < n; i++ )
    {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}

template<typename _Tp, typename _AccTp> static inline
_AccTp normInf(const _Tp* a, const _Tp* b, int n)
{
    _AccTp s = 0;
    for( int i = 0; i < n; i++ )
    {
        _AccTp v0 = a[i] - b[i];
        s = std::max(s, std::abs(v0));
    }
    return s;
}

/** @brief Computes the cube root of an argument.

 The function cubeRoot computes \f$\sqrt[3]{\texttt{val}}\f$. Negative arguments are handled correctly.
 NaN and Inf are not handled. The accuracy approaches the maximum possible accuracy for
 single-precision data.
 @param val A function argument.
 */
CV_EXPORTS_W float cubeRoot(float val);

/** @brief Calculates the angle of a 2D vector in degrees.

 The function fastAtan2 calculates the full-range angle of an input 2D vector. The angle is measured
 in degrees and varies from 0 to 360 degrees. The accuracy is about 0.3 degrees.
 @param x x-coordinate of the vector.
 @param y y-coordinate of the vector.
 */
CV_EXPORTS_W float fastAtan2(float y, float x);

/** proxy for hal::LU */
CV_EXPORTS int LU(float* A, size_t astep, int m, float* b, size_t bstep, int n);
/** proxy for hal::LU */
CV_EXPORTS int LU(double* A, size_t astep, int m, double* b, size_t bstep, int n);
/** proxy for hal::Cholesky */
CV_EXPORTS bool Cholesky(float* A, size_t astep, int m, float* b, size_t bstep, int n);
/** proxy for hal::Cholesky */
CV_EXPORTS bool Cholesky(double* A, size_t astep, int m, double* b, size_t bstep, int n);

////////////////// forward declarations for important OpenCV types //////////////////

//! @cond IGNORED

template<typename _Tp, int cn> class Vec;
template<typename _Tp, int m, int n> class Matx;

template<typename _Tp> class Complex;
template<typename _Tp> class Point_;
template<typename _Tp> class Point3_;
template<typename _Tp> class Size_;
template<typename _Tp> class Rect_;
template<typename _Tp> class Scalar_;

class CV_EXPORTS RotatedRect;
class CV_EXPORTS Range;
class CV_EXPORTS TermCriteria;
class CV_EXPORTS KeyPoint;
class CV_EXPORTS DMatch;
class CV_EXPORTS RNG;

class CV_EXPORTS Mat;
class CV_EXPORTS MatExpr;

class CV_EXPORTS UMat;

class CV_EXPORTS SparseMat;
typedef Mat MatND;

template<typename _Tp> class Mat_;
template<typename _Tp> class SparseMat_;

class CV_EXPORTS MatConstIterator;
class CV_EXPORTS SparseMatIterator;
class CV_EXPORTS SparseMatConstIterator;
template<typename _Tp> class MatIterator_;
template<typename _Tp> class MatConstIterator_;
template<typename _Tp> class SparseMatIterator_;
template<typename _Tp> class SparseMatConstIterator_;

namespace ogl
{
    class CV_EXPORTS Buffer;
    class CV_EXPORTS Texture2D;
    class CV_EXPORTS Arrays;
}

namespace cuda
{
    class CV_EXPORTS GpuMat;
    class CV_EXPORTS HostMem;
    class CV_EXPORTS Stream;
    class CV_EXPORTS Event;
}

namespace cudev
{
    template <typename _Tp> class GpuMat_;
}

namespace ipp
{
#if OPENCV_ABI_COMPATIBILITY > 300
CV_EXPORTS   unsigned long long getIppFeatures();
#else
CV_EXPORTS   int getIppFeatures();
#endif
CV_EXPORTS   void setIppStatus(int status, const char * const funcname = NULL, const char * const filename = NULL,
                             int line = 0);
CV_EXPORTS   int getIppStatus();
CV_EXPORTS   String getIppErrorLocation();
CV_EXPORTS_W bool useIPP();
CV_EXPORTS_W void setUseIPP(bool flag);

} // ipp

//! @endcond

//! @} core_utils




} // cv

#include "opencv2/core/neon_utils.hpp"

#endif //OPENCV_CORE_BASE_HPP
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.

#ifndef OPENCV_CORE_BUFFER_POOL_HPP
#define OPENCV_CORE_BUFFER_POOL_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4265)
#endif

namespace cv
{

//! @addtogroup core
//! @{

class BufferPoolController
{
protected:
    ~BufferPoolController() { }
public:
    virtual size_t getReservedSize() const = 0;
    virtual size_t getMaxReservedSize() const = 0;
    virtual void setMaxReservedSize(size_t size) = 0;
    virtual void freeAllReservedBuffers() = 0;
};

//! @}

}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif // OPENCV_CORE_BUFFER_POOL_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef __OPENCV_BUILD
#error this is a compatibility header which should not be used inside the OpenCV library
#endif

#include "opencv2/core.hpp"
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CUDA_HPP
#define OPENCV_CORE_CUDA_HPP

#ifndef __cplusplus
#  error cuda.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/cuda_types.hpp"

/**
  @defgroup cuda CUDA-accelerated Computer Vision
  @{
    @defgroup cudacore Core part
    @{
      @defgroup cudacore_init Initalization and Information
      @defgroup cudacore_struct Data Structures
    @}
  @}
 */

namespace cv { namespace cuda {

//! @addtogroup cudacore_struct
//! @{

//===================================================================================
// GpuMat
//===================================================================================

/** @brief Base storage class for GPU memory with reference counting.

Its interface matches the Mat interface with the following limitations:

-   no arbitrary dimensions support (only 2D)
-   no functions that return references to their data (because references on GPU are not valid for
    CPU)
-   no expression templates technique support

Beware that the latter limitation may lead to overloaded matrix operators that cause memory
allocations. The GpuMat class is convertible to cuda::PtrStepSz and cuda::PtrStep so it can be
passed directly to the kernel.

@note In contrast with Mat, in most cases GpuMat::isContinuous() == false . This means that rows are
aligned to a size depending on the hardware. Single-row GpuMat is always a continuous matrix.

@note You are not recommended to leave static or global GpuMat variables allocated, that is, to rely
on its destructor. The destruction order of such variables and CUDA context is undefined. GPU memory
release function returns error if the CUDA context has been destroyed before.

@sa Mat
 */
class CV_EXPORTS GpuMat
{
public:
    class CV_EXPORTS Allocator
    {
    public:
        virtual ~Allocator() {}

        // allocator must fill data, step and refcount fields
        virtual bool allocate(GpuMat* mat, int rows, int cols, size_t elemSize) = 0;
        virtual void free(GpuMat* mat) = 0;
    };

    //! default allocator
    static Allocator* defaultAllocator();
    static void setDefaultAllocator(Allocator* allocator);

    //! default constructor
    explicit GpuMat(Allocator* allocator = defaultAllocator());

    //! constructs GpuMat of the specified size and type
    GpuMat(int rows, int cols, int type, Allocator* allocator = defaultAllocator());
    GpuMat(Size size, int type, Allocator* allocator = defaultAllocator());

    //! constucts GpuMat and fills it with the specified value _s
    GpuMat(int rows, int cols, int type, Scalar s, Allocator* allocator = defaultAllocator());
    GpuMat(Size size, int type, Scalar s, Allocator* allocator = defaultAllocator());

    //! copy constructor
    GpuMat(const GpuMat& m);

    //! constructor for GpuMat headers pointing to user-allocated data
    GpuMat(int rows, int cols, int type, void* data, size_t step = Mat::AUTO_STEP);
    GpuMat(Size size, int type, void* data, size_t step = Mat::AUTO_STEP);

    //! creates a GpuMat header for a part of the bigger matrix
    GpuMat(const GpuMat& m, Range rowRange, Range colRange);
    GpuMat(const GpuMat& m, Rect roi);

    //! builds GpuMat from host memory (Blocking call)
    explicit GpuMat(InputArray arr, Allocator* allocator = defaultAllocator());

    //! destructor - calls release()
    ~GpuMat();

    //! assignment operators
    GpuMat& operator =(const GpuMat& m);

    //! allocates new GpuMat data unless the GpuMat already has specified size and type
    void create(int rows, int cols, int type);
    void create(Size size, int type);

    //! decreases reference counter, deallocate the data when reference counter reaches 0
    void release();

    //! swaps with other smart pointer
    void swap(GpuMat& mat);

    //! pefroms upload data to GpuMat (Blocking call)
    void upload(InputArray arr);

    //! pefroms upload data to GpuMat (Non-Blocking call)
    void upload(InputArray arr, Stream& stream);

    //! pefroms download data from device to host memory (Blocking call)
    void download(OutputArray dst) const;

    //! pefroms download data from device to host memory (Non-Blocking call)
    void download(OutputArray dst, Stream& stream) const;

    //! returns deep copy of the GpuMat, i.e. the data is copied
    GpuMat clone() const;

    //! copies the GpuMat content to device memory (Blocking call)
    void copyTo(OutputArray dst) const;

    //! copies the GpuMat content to device memory (Non-Blocking call)
    void copyTo(OutputArray dst, Stream& stream) const;

    //! copies those GpuMat elements to "m" that are marked with non-zero mask elements (Blocking call)
    void copyTo(OutputArray dst, InputArray mask) const;

    //! copies those GpuMat elements to "m" that are marked with non-zero mask elements (Non-Blocking call)
    void copyTo(OutputArray dst, InputArray mask, Stream& stream) const;

    //! sets some of the GpuMat elements to s (Blocking call)
    GpuMat& setTo(Scalar s);

    //! sets some of the GpuMat elements to s (Non-Blocking call)
    GpuMat& setTo(Scalar s, Stream& stream);

    //! sets some of the GpuMat elements to s, according to the mask (Blocking call)
    GpuMat& setTo(Scalar s, InputArray mask);

    //! sets some of the GpuMat elements to s, according to the mask (Non-Blocking call)
    GpuMat& setTo(Scalar s, InputArray mask, Stream& stream);

    //! converts GpuMat to another datatype (Blocking call)
    void convertTo(OutputArray dst, int rtype) const;

    //! converts GpuMat to another datatype (Non-Blocking call)
    void convertTo(OutputArray dst, int rtype, Stream& stream) const;

    //! converts GpuMat to another datatype with scaling (Blocking call)
    void convertTo(OutputArray dst, int rtype, double alpha, double beta = 0.0) const;

    //! converts GpuMat to another datatype with scaling (Non-Blocking call)
    void convertTo(OutputArray dst, int rtype, double alpha, Stream& stream) const;

    //! converts GpuMat to another datatype with scaling (Non-Blocking call)
    void convertTo(OutputArray dst, int rtype, double alpha, double beta, Stream& stream) const;

    void assignTo(GpuMat& m, int type=-1) const;

    //! returns pointer to y-th row
    uchar* ptr(int y = 0);
    const uchar* ptr(int y = 0) const;

    //! template version of the above method
    template<typename _Tp> _Tp* ptr(int y = 0);
    template<typename _Tp> const _Tp* ptr(int y = 0) const;

    template <typename _Tp> operator PtrStepSz<_Tp>() const;
    template <typename _Tp> operator PtrStep<_Tp>() const;

    //! returns a new GpuMat header for the specified row
    GpuMat row(int y) const;

    //! returns a new GpuMat header for the specified column
    GpuMat col(int x) const;

    //! ... for the specified row span
    GpuMat rowRange(int startrow, int endrow) const;
    GpuMat rowRange(Range r) const;

    //! ... for the specified column span
    GpuMat colRange(int startcol, int endcol) const;
    GpuMat colRange(Range r) const;

    //! extracts a rectangular sub-GpuMat (this is a generalized form of row, rowRange etc.)
    GpuMat operator ()(Range rowRange, Range colRange) const;
    GpuMat operator ()(Rect roi) const;

    //! creates alternative GpuMat header for the same data, with different
    //! number of channels and/or different number of rows
    GpuMat reshape(int cn, int rows = 0) const;

    //! locates GpuMat header within a parent GpuMat
    void locateROI(Size& wholeSize, Point& ofs) const;

    //! moves/resizes the current GpuMat ROI inside the parent GpuMat
    GpuMat& adjustROI(int dtop, int dbottom, int dleft, int dright);

    //! returns true iff the GpuMat data is continuous
    //! (i.e. when there are no gaps between successive rows)
    bool isContinuous() const;

    //! returns element size in bytes
    size_t elemSize() const;

    //! returns the size of element channel in bytes
    size_t elemSize1() const;

    //! returns element type
    int type() const;

    //! returns element type
    int depth() const;

    //! returns number of channels
    int channels() const;

    //! returns step/elemSize1()
    size_t step1() const;

    //! returns GpuMat size : width == number of columns, height == number of rows
    Size size() const;

    //! returns true if GpuMat data is NULL
    bool empty() const;

    /*! includes several bit-fields:
    - the magic signature
    - continuity flag
    - depth
    - number of channels
    */
    int flags;

    //! the number of rows and columns
    int rows, cols;

    //! a distance between successive rows in bytes; includes the gap if any
    size_t step;

    //! pointer to the data
    uchar* data;

    //! pointer to the reference counter;
    //! when GpuMat points to user-allocated data, the pointer is NULL
    int* refcount;

    //! helper fields used in locateROI and adjustROI
    uchar* datastart;
    const uchar* dataend;

    //! allocator
    Allocator* allocator;
};

/** @brief Creates a continuous matrix.

@param rows Row count.
@param cols Column count.
@param type Type of the matrix.
@param arr Destination matrix. This parameter changes only if it has a proper type and area (
\f$\texttt{rows} \times \texttt{cols}\f$ ).

Matrix is called continuous if its elements are stored continuously, that is, without gaps at the
end of each row.
 */
CV_EXPORTS void createContinuous(int rows, int cols, int type, OutputArray arr);

/** @brief Ensures that the size of a matrix is big enough and the matrix has a proper type.

@param rows Minimum desired number of rows.
@param cols Minimum desired number of columns.
@param type Desired matrix type.
@param arr Destination matrix.

The function does not reallocate memory if the matrix has proper attributes already.
 */
CV_EXPORTS void ensureSizeIsEnough(int rows, int cols, int type, OutputArray arr);

/** @brief BufferPool for use with CUDA streams

 * BufferPool utilizes cuda::Stream's allocator to create new buffers. It is
 * particularly useful when BufferPoolUsage is set to true, or a custom
 * allocator is specified for the cuda::Stream, and you want to implement your
 * own stream based functions utilizing the same underlying GPU memory
 * management.
 */
class CV_EXPORTS BufferPool
{
public:

    //! Gets the BufferPool for the given stream.
    explicit BufferPool(Stream& stream);

    //! Allocates a new GpuMat of given size and type.
    GpuMat getBuffer(int rows, int cols, int type);

    //! Allocates a new GpuMat of given size and type.
    GpuMat getBuffer(Size size, int type) { return getBuffer(size.height, size.width, type); }

    //! Returns the allocator associated with the stream.
    Ptr<GpuMat::Allocator> getAllocator() const { return allocator_; }

private:
    Ptr<GpuMat::Allocator> allocator_;
};

//! BufferPool management (must be called before Stream creation)
CV_EXPORTS void setBufferPoolUsage(bool on);
CV_EXPORTS void setBufferPoolConfig(int deviceId, size_t stackSize, int stackCount);

//===================================================================================
// HostMem
//===================================================================================

/** @brief Class with reference counting wrapping special memory type allocation functions from CUDA.

Its interface is also Mat-like but with additional memory type parameters.

-   **PAGE_LOCKED** sets a page locked memory type used commonly for fast and asynchronous
    uploading/downloading data from/to GPU.
-   **SHARED** specifies a zero copy memory allocation that enables mapping the host memory to GPU
    address space, if supported.
-   **WRITE_COMBINED** sets the write combined buffer that is not cached by CPU. Such buffers are
    used to supply GPU with data when GPU only reads it. The advantage is a better CPU cache
    utilization.

@note Allocation size of such memory types is usually limited. For more details, see *CUDA 2.2
Pinned Memory APIs* document or *CUDA C Programming Guide*.
 */
class CV_EXPORTS HostMem
{
public:
    enum AllocType { PAGE_LOCKED = 1, SHARED = 2, WRITE_COMBINED = 4 };

    static MatAllocator* getAllocator(AllocType alloc_type = PAGE_LOCKED);

    explicit HostMem(AllocType alloc_type = PAGE_LOCKED);

    HostMem(const HostMem& m);

    HostMem(int rows, int cols, int type, AllocType alloc_type = PAGE_LOCKED);
    HostMem(Size size, int type, AllocType alloc_type = PAGE_LOCKED);

    //! creates from host memory with coping data
    explicit HostMem(InputArray arr, AllocType alloc_type = PAGE_LOCKED);

    ~HostMem();

    HostMem& operator =(const HostMem& m);

    //! swaps with other smart pointer
    void swap(HostMem& b);

    //! returns deep copy of the matrix, i.e. the data is copied
    HostMem clone() const;

    //! allocates new matrix data unless the matrix already has specified size and type.
    void create(int rows, int cols, int type);
    void create(Size size, int type);

    //! creates alternative HostMem header for the same data, with different
    //! number of channels and/or different number of rows
    HostMem reshape(int cn, int rows = 0) const;

    //! decrements reference counter and released memory if needed.
    void release();

    //! returns matrix header with disabled reference counting for HostMem data.
    Mat createMatHeader() const;

    /** @brief Maps CPU memory to GPU address space and creates the cuda::GpuMat header without reference counting
    for it.

    This can be done only if memory was allocated with the SHARED flag and if it is supported by the
    hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which
    eliminates an extra copy.
     */
    GpuMat createGpuMatHeader() const;

    // Please see cv::Mat for descriptions
    bool isContinuous() const;
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1() const;
    Size size() const;
    bool empty() const;

    // Please see cv::Mat for descriptions
    int flags;
    int rows, cols;
    size_t step;

    uchar* data;
    int* refcount;

    uchar* datastart;
    const uchar* dataend;

    AllocType alloc_type;
};

/** @brief Page-locks the memory of matrix and maps it for the device(s).

@param m Input matrix.
 */
CV_EXPORTS void registerPageLocked(Mat& m);

/** @brief Unmaps the memory of matrix and makes it pageable again.

@param m Input matrix.
 */
CV_EXPORTS void unregisterPageLocked(Mat& m);

//===================================================================================
// Stream
//===================================================================================

/** @brief This class encapsulates a queue of asynchronous calls.

@note Currently, you may face problems if an operation is enqueued twice with different data. Some
functions use the constant GPU memory, and next call may update the memory before the previous one
has been finished. But calling different operations asynchronously is safe because each operation
has its own constant buffer. Memory copy/upload/download/set operations to the buffers you hold are
also safe.

@note The Stream class is not thread-safe. Please use different Stream objects for different CPU threads.

@code
void thread1()
{
    cv::cuda::Stream stream1;
    cv::cuda::func1(..., stream1);
}

void thread2()
{
    cv::cuda::Stream stream2;
    cv::cuda::func2(..., stream2);
}
@endcode

@note By default all CUDA routines are launched in Stream::Null() object, if the stream is not specified by user.
In multi-threading environment the stream objects must be passed explicitly (see previous note).
 */
class CV_EXPORTS Stream
{
    typedef void (Stream::*bool_type)() const;
    void this_type_does_not_support_comparisons() const {}

public:
    typedef void (*StreamCallback)(int status, void* userData);

    //! creates a new asynchronous stream
    Stream();

    //! creates a new asynchronous stream with custom allocator
    Stream(const Ptr<GpuMat::Allocator>& allocator);

    /** @brief Returns true if the current stream queue is finished. Otherwise, it returns false.
    */
    bool queryIfComplete() const;

    /** @brief Blocks the current CPU thread until all operations in the stream are complete.
    */
    void waitForCompletion();

    /** @brief Makes a compute stream wait on an event.
    */
    void waitEvent(const Event& event);

    /** @brief Adds a callback to be called on the host after all currently enqueued items in the stream have
    completed.

    @note Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronization
    that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
    Callbacks without a mandated order (in independent streams) execute in undefined order and may be
    serialized.
     */
    void enqueueHostCallback(StreamCallback callback, void* userData);

    //! return Stream object for default CUDA stream
    static Stream& Null();

    //! returns true if stream object is not default (!= 0)
    operator bool_type() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    Stream(const Ptr<Impl>& impl);

    friend struct StreamAccessor;
    friend class BufferPool;
    friend class DefaultDeviceInitializer;
};

class CV_EXPORTS Event
{
public:
    enum CreateFlags
    {
        DEFAULT        = 0x00,  /**< Default event flag */
        BLOCKING_SYNC  = 0x01,  /**< Event uses blocking synchronization */
        DISABLE_TIMING = 0x02,  /**< Event will not record timing data */
        INTERPROCESS   = 0x04   /**< Event is suitable for interprocess use. DisableTiming must be set */
    };

    explicit Event(CreateFlags flags = DEFAULT);

    //! records an event
    void record(Stream& stream = Stream::Null());

    //! queries an event's status
    bool queryIfComplete() const;

    //! waits for an event to complete
    void waitForCompletion();

    //! computes the elapsed time between events
    static float elapsedTime(const Event& start, const Event& end);

    class Impl;

private:
    Ptr<Impl> impl_;
    Event(const Ptr<Impl>& impl);

    friend struct EventAccessor;
};

//! @} cudacore_struct

//===================================================================================
// Initialization & Info
//===================================================================================

//! @addtogroup cudacore_init
//! @{

/** @brief Returns the number of installed CUDA-enabled devices.

Use this function before any other CUDA functions calls. If OpenCV is compiled without CUDA support,
this function returns 0. If the CUDA driver is not installed, or is incompatible, this function
returns -1.
 */
CV_EXPORTS int getCudaEnabledDeviceCount();

/** @brief Sets a device and initializes it for the current thread.

@param device System index of a CUDA device starting with 0.

If the call of this function is omitted, a default device is initialized at the fist CUDA usage.
 */
CV_EXPORTS void setDevice(int device);

/** @brief Returns the current device index set by cuda::setDevice or initialized by default.
 */
CV_EXPORTS int getDevice();

/** @brief Explicitly destroys and cleans up all resources associated with the current device in the current
process.

Any subsequent API call to this device will reinitialize the device.
 */
CV_EXPORTS void resetDevice();

/** @brief Enumeration providing CUDA computing features.
 */
enum FeatureSet
{
    FEATURE_SET_COMPUTE_10 = 10,
    FEATURE_SET_COMPUTE_11 = 11,
    FEATURE_SET_COMPUTE_12 = 12,
    FEATURE_SET_COMPUTE_13 = 13,
    FEATURE_SET_COMPUTE_20 = 20,
    FEATURE_SET_COMPUTE_21 = 21,
    FEATURE_SET_COMPUTE_30 = 30,
    FEATURE_SET_COMPUTE_32 = 32,
    FEATURE_SET_COMPUTE_35 = 35,
    FEATURE_SET_COMPUTE_50 = 50,

    GLOBAL_ATOMICS = FEATURE_SET_COMPUTE_11,
    SHARED_ATOMICS = FEATURE_SET_COMPUTE_12,
    NATIVE_DOUBLE = FEATURE_SET_COMPUTE_13,
    WARP_SHUFFLE_FUNCTIONS = FEATURE_SET_COMPUTE_30,
    DYNAMIC_PARALLELISM = FEATURE_SET_COMPUTE_35
};

//! checks whether current device supports the given feature
CV_EXPORTS bool deviceSupports(FeatureSet feature_set);

/** @brief Class providing a set of static methods to check what NVIDIA\* card architecture the CUDA module was
built for.

According to the CUDA C Programming Guide Version 3.2: "PTX code produced for some specific compute
capability can always be compiled to binary code of greater or equal compute capability".
 */
class CV_EXPORTS TargetArchs
{
public:
    /** @brief The following method checks whether the module was built with the support of the given feature:

    @param feature_set Features to be checked. See :ocvcuda::FeatureSet.
     */
    static bool builtWith(FeatureSet feature_set);

    /** @brief There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDA
    code for the given architecture(s):

    @param major Major compute capability version.
    @param minor Minor compute capability version.
     */
    static bool has(int major, int minor);
    static bool hasPtx(int major, int minor);
    static bool hasBin(int major, int minor);

    static bool hasEqualOrLessPtx(int major, int minor);
    static bool hasEqualOrGreater(int major, int minor);
    static bool hasEqualOrGreaterPtx(int major, int minor);
    static bool hasEqualOrGreaterBin(int major, int minor);
};

/** @brief Class providing functionality for querying the specified GPU properties.
 */
class CV_EXPORTS DeviceInfo
{
public:
    //! creates DeviceInfo object for the current GPU
    DeviceInfo();

    /** @brief The constructors.

    @param device_id System index of the CUDA device starting with 0.

    Constructs the DeviceInfo object for the specified device. If device_id parameter is missed, it
    constructs an object for the current device.
     */
    DeviceInfo(int device_id);

    /** @brief Returns system index of the CUDA device starting with 0.
    */
    int deviceID() const;

    //! ASCII string identifying device
    const char* name() const;

    //! global memory available on device in bytes
    size_t totalGlobalMem() const;

    //! shared memory available per block in bytes
    size_t sharedMemPerBlock() const;

    //! 32-bit registers available per block
    int regsPerBlock() const;

    //! warp size in threads
    int warpSize() const;

    //! maximum pitch in bytes allowed by memory copies
    size_t memPitch() const;

    //! maximum number of threads per block
    int maxThreadsPerBlock() const;

    //! maximum size of each dimension of a block
    Vec3i maxThreadsDim() const;

    //! maximum size of each dimension of a grid
    Vec3i maxGridSize() const;

    //! clock frequency in kilohertz
    int clockRate() const;

    //! constant memory available on device in bytes
    size_t totalConstMem() const;

    //! major compute capability
    int majorVersion() const;

    //! minor compute capability
    int minorVersion() const;

    //! alignment requirement for textures
    size_t textureAlignment() const;

    //! pitch alignment requirement for texture references bound to pitched memory
    size_t texturePitchAlignment() const;

    //! number of multiprocessors on device
    int multiProcessorCount() const;

    //! specified whether there is a run time limit on kernels
    bool kernelExecTimeoutEnabled() const;

    //! device is integrated as opposed to discrete
    bool integrated() const;

    //! device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    bool canMapHostMemory() const;

    enum ComputeMode
    {
        ComputeModeDefault,         /**< default compute mode (Multiple threads can use cudaSetDevice with this device) */
        ComputeModeExclusive,       /**< compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice with this device) */
        ComputeModeProhibited,      /**< compute-prohibited mode (No threads can use cudaSetDevice with this device) */
        ComputeModeExclusiveProcess /**< compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice with this device) */
    };

    //! compute mode
    ComputeMode computeMode() const;

    //! maximum 1D texture size
    int maxTexture1D() const;

    //! maximum 1D mipmapped texture size
    int maxTexture1DMipmap() const;

    //! maximum size for 1D textures bound to linear memory
    int maxTexture1DLinear() const;

    //! maximum 2D texture dimensions
    Vec2i maxTexture2D() const;

    //! maximum 2D mipmapped texture dimensions
    Vec2i maxTexture2DMipmap() const;

    //! maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
    Vec3i maxTexture2DLinear() const;

    //! maximum 2D texture dimensions if texture gather operations have to be performed
    Vec2i maxTexture2DGather() const;

    //! maximum 3D texture dimensions
    Vec3i maxTexture3D() const;

    //! maximum Cubemap texture dimensions
    int maxTextureCubemap() const;

    //! maximum 1D layered texture dimensions
    Vec2i maxTexture1DLayered() const;

    //! maximum 2D layered texture dimensions
    Vec3i maxTexture2DLayered() const;

    //! maximum Cubemap layered texture dimensions
    Vec2i maxTextureCubemapLayered() const;

    //! maximum 1D surface size
    int maxSurface1D() const;

    //! maximum 2D surface dimensions
    Vec2i maxSurface2D() const;

    //! maximum 3D surface dimensions
    Vec3i maxSurface3D() const;

    //! maximum 1D layered surface dimensions
    Vec2i maxSurface1DLayered() const;

    //! maximum 2D layered surface dimensions
    Vec3i maxSurface2DLayered() const;

    //! maximum Cubemap surface dimensions
    int maxSurfaceCubemap() const;

    //! maximum Cubemap layered surface dimensions
    Vec2i maxSurfaceCubemapLayered() const;

    //! alignment requirements for surfaces
    size_t surfaceAlignment() const;

    //! device can possibly execute multiple kernels concurrently
    bool concurrentKernels() const;

    //! device has ECC support enabled
    bool ECCEnabled() const;

    //! PCI bus ID of the device
    int pciBusID() const;

    //! PCI device ID of the device
    int pciDeviceID() const;

    //! PCI domain ID of the device
    int pciDomainID() const;

    //! true if device is a Tesla device using TCC driver, false otherwise
    bool tccDriver() const;

    //! number of asynchronous engines
    int asyncEngineCount() const;

    //! device shares a unified address space with the host
    bool unifiedAddressing() const;

    //! peak memory clock frequency in kilohertz
    int memoryClockRate() const;

    //! global memory bus width in bits
    int memoryBusWidth() const;

    //! size of L2 cache in bytes
    int l2CacheSize() const;

    //! maximum resident threads per multiprocessor
    int maxThreadsPerMultiProcessor() const;

    //! gets free and total device memory
    void queryMemory(size_t& totalMemory, size_t& freeMemory) const;
    size_t freeMemory() const;
    size_t totalMemory() const;

    /** @brief Provides information on CUDA feature support.

    @param feature_set Features to be checked. See cuda::FeatureSet.

    This function returns true if the device has the specified CUDA feature. Otherwise, it returns false
     */
    bool supports(FeatureSet feature_set) const;

    /** @brief Checks the CUDA module and device compatibility.

    This function returns true if the CUDA module can be run on the specified device. Otherwise, it
    returns false .
     */
    bool isCompatible() const;

private:
    int device_id_;
};

CV_EXPORTS void printCudaDeviceInfo(int device);
CV_EXPORTS void printShortCudaDeviceInfo(int device);

/** @brief Converts an array to half precision floating number.

@param _src input array.
@param _dst output array.
@param stream Stream for the asynchronous version.
@sa convertFp16
*/
CV_EXPORTS void convertFp16(InputArray _src, OutputArray _dst, Stream& stream = Stream::Null());

//! @} cudacore_init

}} // namespace cv { namespace cuda {


#include "opencv2/core/cuda.inl.hpp"

#endif /* OPENCV_CORE_CUDA_HPP */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CUDAINL_HPP
#define OPENCV_CORE_CUDAINL_HPP

#include "opencv2/core/cuda.hpp"

//! @cond IGNORED

namespace cv { namespace cuda {

//===================================================================================
// GpuMat
//===================================================================================

inline
GpuMat::GpuMat(Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{}

inline
GpuMat::GpuMat(int rows_, int cols_, int type_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline
GpuMat::GpuMat(Size size_, int type_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline
GpuMat::GpuMat(int rows_, int cols_, int type_, Scalar s_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (rows_ > 0 && cols_ > 0)
    {
        create(rows_, cols_, type_);
        setTo(s_);
    }
}

inline
GpuMat::GpuMat(Size size_, int type_, Scalar s_, Allocator* allocator_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    if (size_.height > 0 && size_.width > 0)
    {
        create(size_.height, size_.width, type_);
        setTo(s_);
    }
}

inline
GpuMat::GpuMat(const GpuMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), allocator(m.allocator)
{
    if (refcount)
        CV_XADD(refcount, 1);
}

inline
GpuMat::GpuMat(InputArray arr, Allocator* allocator_) :
    flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), allocator(allocator_)
{
    upload(arr);
}

inline
GpuMat::~GpuMat()
{
    release();
}

inline
GpuMat& GpuMat::operator =(const GpuMat& m)
{
    if (this != &m)
    {
        GpuMat temp(m);
        swap(temp);
    }

    return *this;
}

inline
void GpuMat::create(Size size_, int type_)
{
    create(size_.height, size_.width, type_);
}

inline
void GpuMat::swap(GpuMat& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
    std::swap(allocator, b.allocator);
}

inline
GpuMat GpuMat::clone() const
{
    GpuMat m;
    copyTo(m);
    return m;
}

inline
void GpuMat::copyTo(OutputArray dst, InputArray mask) const
{
    copyTo(dst, mask, Stream::Null());
}

inline
GpuMat& GpuMat::setTo(Scalar s)
{
    return setTo(s, Stream::Null());
}

inline
GpuMat& GpuMat::setTo(Scalar s, InputArray mask)
{
    return setTo(s, mask, Stream::Null());
}

inline
void GpuMat::convertTo(OutputArray dst, int rtype) const
{
    convertTo(dst, rtype, Stream::Null());
}

inline
void GpuMat::convertTo(OutputArray dst, int rtype, double alpha, double beta) const
{
    convertTo(dst, rtype, alpha, beta, Stream::Null());
}

inline
void GpuMat::convertTo(OutputArray dst, int rtype, double alpha, Stream& stream) const
{
    convertTo(dst, rtype, alpha, 0.0, stream);
}

inline
void GpuMat::assignTo(GpuMat& m, int _type) const
{
    if (_type < 0)
        m = *this;
    else
        convertTo(m, _type);
}

inline
uchar* GpuMat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step * y;
}

inline
const uchar* GpuMat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step * y;
}

template<typename _Tp> inline
_Tp* GpuMat::ptr(int y)
{
    return (_Tp*)ptr(y);
}

template<typename _Tp> inline
const _Tp* GpuMat::ptr(int y) const
{
    return (const _Tp*)ptr(y);
}

template <class T> inline
GpuMat::operator PtrStepSz<T>() const
{
    return PtrStepSz<T>(rows, cols, (T*)data, step);
}

template <class T> inline
GpuMat::operator PtrStep<T>() const
{
    return PtrStep<T>((T*)data, step);
}

inline
GpuMat GpuMat::row(int y) const
{
    return GpuMat(*this, Range(y, y+1), Range::all());
}

inline
GpuMat GpuMat::col(int x) const
{
    return GpuMat(*this, Range::all(), Range(x, x+1));
}

inline
GpuMat GpuMat::rowRange(int startrow, int endrow) const
{
    return GpuMat(*this, Range(startrow, endrow), Range::all());
}

inline
GpuMat GpuMat::rowRange(Range r) const
{
    return GpuMat(*this, r, Range::all());
}

inline
GpuMat GpuMat::colRange(int startcol, int endcol) const
{
    return GpuMat(*this, Range::all(), Range(startcol, endcol));
}

inline
GpuMat GpuMat::colRange(Range r) const
{
    return GpuMat(*this, Range::all(), r);
}

inline
GpuMat GpuMat::operator ()(Range rowRange_, Range colRange_) const
{
    return GpuMat(*this, rowRange_, colRange_);
}

inline
GpuMat GpuMat::operator ()(Rect roi) const
{
    return GpuMat(*this, roi);
}

inline
bool GpuMat::isContinuous() const
{
    return (flags & Mat::CONTINUOUS_FLAG) != 0;
}

inline
size_t GpuMat::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t GpuMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int GpuMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int GpuMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int GpuMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t GpuMat::step1() const
{
    return step / elemSize1();
}

inline
Size GpuMat::size() const
{
    return Size(cols, rows);
}

inline
bool GpuMat::empty() const
{
    return data == 0;
}

static inline
GpuMat createContinuous(int rows, int cols, int type)
{
    GpuMat m;
    createContinuous(rows, cols, type, m);
    return m;
}

static inline
void createContinuous(Size size, int type, OutputArray arr)
{
    createContinuous(size.height, size.width, type, arr);
}

static inline
GpuMat createContinuous(Size size, int type)
{
    GpuMat m;
    createContinuous(size, type, m);
    return m;
}

static inline
void ensureSizeIsEnough(Size size, int type, OutputArray arr)
{
    ensureSizeIsEnough(size.height, size.width, type, arr);
}

static inline
void swap(GpuMat& a, GpuMat& b)
{
    a.swap(b);
}

//===================================================================================
// HostMem
//===================================================================================

inline
HostMem::HostMem(AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
}

inline
HostMem::HostMem(const HostMem& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), alloc_type(m.alloc_type)
{
    if( refcount )
        CV_XADD(refcount, 1);
}

inline
HostMem::HostMem(int rows_, int cols_, int type_, AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
    if (rows_ > 0 && cols_ > 0)
        create(rows_, cols_, type_);
}

inline
HostMem::HostMem(Size size_, int type_, AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
    if (size_.height > 0 && size_.width > 0)
        create(size_.height, size_.width, type_);
}

inline
HostMem::HostMem(InputArray arr, AllocType alloc_type_)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(alloc_type_)
{
    arr.getMat().copyTo(*this);
}

inline
HostMem::~HostMem()
{
    release();
}

inline
HostMem& HostMem::operator =(const HostMem& m)
{
    if (this != &m)
    {
        HostMem temp(m);
        swap(temp);
    }

    return *this;
}

inline
void HostMem::swap(HostMem& b)
{
    std::swap(flags, b.flags);
    std::swap(rows, b.rows);
    std::swap(cols, b.cols);
    std::swap(step, b.step);
    std::swap(data, b.data);
    std::swap(datastart, b.datastart);
    std::swap(dataend, b.dataend);
    std::swap(refcount, b.refcount);
    std::swap(alloc_type, b.alloc_type);
}

inline
HostMem HostMem::clone() const
{
    HostMem m(size(), type(), alloc_type);
    createMatHeader().copyTo(m);
    return m;
}

inline
void HostMem::create(Size size_, int type_)
{
    create(size_.height, size_.width, type_);
}

inline
Mat HostMem::createMatHeader() const
{
    return Mat(size(), type(), data, step);
}

inline
bool HostMem::isContinuous() const
{
    return (flags & Mat::CONTINUOUS_FLAG) != 0;
}

inline
size_t HostMem::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t HostMem::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int HostMem::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int HostMem::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int HostMem::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t HostMem::step1() const
{
    return step / elemSize1();
}

inline
Size HostMem::size() const
{
    return Size(cols, rows);
}

inline
bool HostMem::empty() const
{
    return data == 0;
}

static inline
void swap(HostMem& a, HostMem& b)
{
    a.swap(b);
}

//===================================================================================
// Stream
//===================================================================================

inline
Stream::Stream(const Ptr<Impl>& impl)
    : impl_(impl)
{
}

//===================================================================================
// Event
//===================================================================================

inline
Event::Event(const Ptr<Impl>& impl)
    : impl_(impl)
{
}

//===================================================================================
// Initialization & Info
//===================================================================================

inline
bool TargetArchs::has(int major, int minor)
{
    return hasPtx(major, minor) || hasBin(major, minor);
}

inline
bool TargetArchs::hasEqualOrGreater(int major, int minor)
{
    return hasEqualOrGreaterPtx(major, minor) || hasEqualOrGreaterBin(major, minor);
}

inline
DeviceInfo::DeviceInfo()
{
    device_id_ = getDevice();
}

inline
DeviceInfo::DeviceInfo(int device_id)
{
    CV_Assert( device_id >= 0 && device_id < getCudaEnabledDeviceCount() );
    device_id_ = device_id;
}

inline
int DeviceInfo::deviceID() const
{
    return device_id_;
}

inline
size_t DeviceInfo::freeMemory() const
{
    size_t _totalMemory = 0, _freeMemory = 0;
    queryMemory(_totalMemory, _freeMemory);
    return _freeMemory;
}

inline
size_t DeviceInfo::totalMemory() const
{
    size_t _totalMemory = 0, _freeMemory = 0;
    queryMemory(_totalMemory, _freeMemory);
    return _totalMemory;
}

inline
bool DeviceInfo::supports(FeatureSet feature_set) const
{
    int version = majorVersion() * 10 + minorVersion();
    return version >= feature_set;
}


}} // namespace cv { namespace cuda {

//===================================================================================
// Mat
//===================================================================================

namespace cv {

inline
Mat::Mat(const cuda::GpuMat& m)
    : flags(0), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows)
{
    m.download(*this);
}

}

//! @endcond

#endif // OPENCV_CORE_CUDAINL_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CUDA_STREAM_ACCESSOR_HPP
#define OPENCV_CORE_CUDA_STREAM_ACCESSOR_HPP

#ifndef __cplusplus
#  error cuda_stream_accessor.hpp header must be compiled as C++
#endif

/** @file cuda_stream_accessor.hpp
 * This is only header file that depends on CUDA Runtime API. All other headers are independent.
 */

#include <cuda_runtime.h>
#include "opencv2/core/cuda.hpp"

namespace cv
{
    namespace cuda
    {

//! @addtogroup cudacore_struct
//! @{

        /** @brief Class that enables getting cudaStream_t from cuda::Stream
         */
        struct StreamAccessor
        {
            CV_EXPORTS static cudaStream_t getStream(const Stream& stream);
            CV_EXPORTS static Stream wrapStream(cudaStream_t stream);
        };

        /** @brief Class that enables getting cudaEvent_t from cuda::Event
         */
        struct EventAccessor
        {
            CV_EXPORTS static cudaEvent_t getEvent(const Event& event);
            CV_EXPORTS static Event wrapEvent(cudaEvent_t event);
        };

//! @}

    }
}

#endif /* OPENCV_CORE_CUDA_STREAM_ACCESSOR_HPP */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CUDA_TYPES_HPP
#define OPENCV_CORE_CUDA_TYPES_HPP

#ifndef __cplusplus
#  error cuda_types.hpp header must be compiled as C++
#endif

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

#ifdef __CUDACC__
    #define __CV_CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
    #define __CV_CUDA_HOST_DEVICE__
#endif

namespace cv
{
    namespace cuda
    {

        // Simple lightweight structures that encapsulates information about an image on device.
        // It is intended to pass to nvcc-compiled code. GpuMat depends on headers that nvcc can't compile

        template <typename T> struct DevPtr
        {
            typedef T elem_type;
            typedef int index_type;

            enum { elem_size = sizeof(elem_type) };

            T* data;

            __CV_CUDA_HOST_DEVICE__ DevPtr() : data(0) {}
            __CV_CUDA_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}

            __CV_CUDA_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
            __CV_CUDA_HOST_DEVICE__ operator       T*()       { return data; }
            __CV_CUDA_HOST_DEVICE__ operator const T*() const { return data; }
        };

        template <typename T> struct PtrSz : public DevPtr<T>
        {
            __CV_CUDA_HOST_DEVICE__ PtrSz() : size(0) {}
            __CV_CUDA_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

            size_t size;
        };

        template <typename T> struct PtrStep : public DevPtr<T>
        {
            __CV_CUDA_HOST_DEVICE__ PtrStep() : step(0) {}
            __CV_CUDA_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

            size_t step;

            __CV_CUDA_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
            __CV_CUDA_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }

            __CV_CUDA_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
            __CV_CUDA_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
        };

        template <typename T> struct PtrStepSz : public PtrStep<T>
        {
            __CV_CUDA_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
            __CV_CUDA_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
                : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

            template <typename U>
            explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows){}

            int cols;
            int rows;
        };

        typedef PtrStepSz<unsigned char> PtrStepSzb;
        typedef PtrStepSz<float> PtrStepSzf;
        typedef PtrStepSz<int> PtrStepSzi;

        typedef PtrStep<unsigned char> PtrStepb;
        typedef PtrStep<float> PtrStepf;
        typedef PtrStep<int> PtrStepi;

    }
}

//! @endcond

#endif /* OPENCV_CORE_CUDA_TYPES_HPP */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CVSTD_HPP
#define OPENCV_CORE_CVSTD_HPP

#ifndef __cplusplus
#  error cvstd.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include <cstddef>
#include <cstring>
#include <cctype>

#include <string>

// import useful primitives from stl
#  include <algorithm>
#  include <utility>
#  include <cstdlib> //for abs(int)
#  include <cmath>

namespace cv
{
    static inline uchar abs(uchar a) { return a; }
    static inline ushort abs(ushort a) { return a; }
    static inline unsigned abs(unsigned a) { return a; }
    static inline uint64 abs(uint64 a) { return a; }

    using std::min;
    using std::max;
    using std::abs;
    using std::swap;
    using std::sqrt;
    using std::exp;
    using std::pow;
    using std::log;
}

namespace cv {

//! @addtogroup core_utils
//! @{

//////////////////////////// memory management functions ////////////////////////////

/** @brief Allocates an aligned memory buffer.

The function allocates the buffer of the specified size and returns it. When the buffer size is 16
bytes or more, the returned buffer is aligned to 16 bytes.
@param bufSize Allocated buffer size.
 */
CV_EXPORTS void* fastMalloc(size_t bufSize);

/** @brief Deallocates a memory buffer.

The function deallocates the buffer allocated with fastMalloc . If NULL pointer is passed, the
function does nothing. C version of the function clears the pointer *pptr* to avoid problems with
double memory deallocation.
@param ptr Pointer to the allocated buffer.
 */
CV_EXPORTS void fastFree(void* ptr);

/*!
  The STL-compilant memory Allocator based on cv::fastMalloc() and cv::fastFree()
*/
template<typename _Tp> class Allocator
{
public:
    typedef _Tp value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> class rebind { typedef Allocator<U> other; };

    explicit Allocator() {}
    ~Allocator() {}
    explicit Allocator(Allocator const&) {}
    template<typename U>
    explicit Allocator(Allocator<U> const&) {}

    // address
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    pointer allocate(size_type count, const void* =0) { return reinterpret_cast<pointer>(fastMalloc(count * sizeof (_Tp))); }
    void deallocate(pointer p, size_type) { fastFree(p); }

    void construct(pointer p, const _Tp& v) { new(static_cast<void*>(p)) _Tp(v); }
    void destroy(pointer p) { p->~_Tp(); }

    size_type max_size() const { return cv::max(static_cast<_Tp>(-1)/sizeof(_Tp), 1); }
};

//! @} core_utils

//! @cond IGNORED

namespace detail
{

// Metafunction to avoid taking a reference to void.
template<typename T>
struct RefOrVoid { typedef T& type; };

template<>
struct RefOrVoid<void>{ typedef void type; };

template<>
struct RefOrVoid<const void>{ typedef const void type; };

template<>
struct RefOrVoid<volatile void>{ typedef volatile void type; };

template<>
struct RefOrVoid<const volatile void>{ typedef const volatile void type; };

// This class would be private to Ptr, if it didn't have to be a non-template.
struct PtrOwner;

}

template<typename Y>
struct DefaultDeleter
{
    void operator () (Y* p) const;
};

//! @endcond

//! @addtogroup core_basic
//! @{

/** @brief Template class for smart pointers with shared ownership

A Ptr\<T\> pretends to be a pointer to an object of type T. Unlike an ordinary pointer, however, the
object will be automatically cleaned up once all Ptr instances pointing to it are destroyed.

Ptr is similar to boost::shared_ptr that is part of the Boost library
(<http://www.boost.org/doc/libs/release/libs/smart_ptr/shared_ptr.htm>) and std::shared_ptr from
the [C++11](http://en.wikipedia.org/wiki/C++11) standard.

This class provides the following advantages:
-   Default constructor, copy constructor, and assignment operator for an arbitrary C++ class or C
    structure. For some objects, like files, windows, mutexes, sockets, and others, a copy
    constructor or an assignment operator are difficult to define. For some other objects, like
    complex classifiers in OpenCV, copy constructors are absent and not easy to implement. Finally,
    some of complex OpenCV and your own data structures may be written in C. However, copy
    constructors and default constructors can simplify programming a lot. Besides, they are often
    required (for example, by STL containers). By using a Ptr to such an object instead of the
    object itself, you automatically get all of the necessary constructors and the assignment
    operator.
-   *O(1)* complexity of the above-mentioned operations. While some structures, like std::vector,
    provide a copy constructor and an assignment operator, the operations may take a considerable
    amount of time if the data structures are large. But if the structures are put into a Ptr, the
    overhead is small and independent of the data size.
-   Automatic and customizable cleanup, even for C structures. See the example below with FILE\*.
-   Heterogeneous collections of objects. The standard STL and most other C++ and OpenCV containers
    can store only objects of the same type and the same size. The classical solution to store
    objects of different types in the same container is to store pointers to the base class (Base\*)
    instead but then you lose the automatic memory management. Again, by using Ptr\<Base\> instead
    of raw pointers, you can solve the problem.

A Ptr is said to *own* a pointer - that is, for each Ptr there is a pointer that will be deleted
once all Ptr instances that own it are destroyed. The owned pointer may be null, in which case
nothing is deleted. Each Ptr also *stores* a pointer. The stored pointer is the pointer the Ptr
pretends to be; that is, the one you get when you use Ptr::get or the conversion to T\*. It's
usually the same as the owned pointer, but if you use casts or the general shared-ownership
constructor, the two may diverge: the Ptr will still own the original pointer, but will itself point
to something else.

The owned pointer is treated as a black box. The only thing Ptr needs to know about it is how to
delete it. This knowledge is encapsulated in the *deleter* - an auxiliary object that is associated
with the owned pointer and shared between all Ptr instances that own it. The default deleter is an
instance of DefaultDeleter, which uses the standard C++ delete operator; as such it will work with
any pointer allocated with the standard new operator.

However, if the pointer must be deleted in a different way, you must specify a custom deleter upon
Ptr construction. A deleter is simply a callable object that accepts the pointer as its sole
argument. For example, if you want to wrap FILE, you may do so as follows:
@code
    Ptr<FILE> f(fopen("myfile.txt", "w"), fclose);
    if(!f) throw ...;
    fprintf(f, ....);
    ...
    // the file will be closed automatically by f's destructor.
@endcode
Alternatively, if you want all pointers of a particular type to be deleted the same way, you can
specialize DefaultDeleter<T>::operator() for that type, like this:
@code
    namespace cv {
    template<> void DefaultDeleter<FILE>::operator ()(FILE * obj) const
    {
        fclose(obj);
    }
    }
@endcode
For convenience, the following types from the OpenCV C API already have such a specialization that
calls the appropriate release function:
-   CvCapture
-   CvFileStorage
-   CvHaarClassifierCascade
-   CvMat
-   CvMatND
-   CvMemStorage
-   CvSparseMat
-   CvVideoWriter
-   IplImage
@note The shared ownership mechanism is implemented with reference counting. As such, cyclic
ownership (e.g. when object a contains a Ptr to object b, which contains a Ptr to object a) will
lead to all involved objects never being cleaned up. Avoid such situations.
@note It is safe to concurrently read (but not write) a Ptr instance from multiple threads and
therefore it is normally safe to use it in multi-threaded applications. The same is true for Mat and
other C++ OpenCV classes that use internal reference counts.
*/
template<typename T>
struct Ptr
{
    /** Generic programming support. */
    typedef T element_type;

    /** The default constructor creates a null Ptr - one that owns and stores a null pointer.
    */
    Ptr();

    /**
    If p is null, these are equivalent to the default constructor.
    Otherwise, these constructors assume ownership of p - that is, the created Ptr owns and stores p
    and assumes it is the sole owner of it. Don't use them if p is already owned by another Ptr, or
    else p will get deleted twice.
    With the first constructor, DefaultDeleter\<Y\>() becomes the associated deleter (so p will
    eventually be deleted with the standard delete operator). Y must be a complete type at the point
    of invocation.
    With the second constructor, d becomes the associated deleter.
    Y\* must be convertible to T\*.
    @param p Pointer to own.
    @note It is often easier to use makePtr instead.
     */
    template<typename Y>
#ifdef DISABLE_OPENCV_24_COMPATIBILITY
    explicit
#endif
    Ptr(Y* p);

    /** @overload
    @param d Deleter to use for the owned pointer.
    @param p Pointer to own.
    */
    template<typename Y, typename D>
    Ptr(Y* p, D d);

    /**
    These constructors create a Ptr that shares ownership with another Ptr - that is, own the same
    pointer as o.
    With the first two, the same pointer is stored, as well; for the second, Y\* must be convertible
    to T\*.
    With the third, p is stored, and Y may be any type. This constructor allows to have completely
    unrelated owned and stored pointers, and should be used with care to avoid confusion. A relatively
    benign use is to create a non-owning Ptr, like this:
    @code
        ptr = Ptr<T>(Ptr<T>(), dont_delete_me); // owns nothing; will not delete the pointer.
    @endcode
    @param o Ptr to share ownership with.
    */
    Ptr(const Ptr& o);

    /** @overload
    @param o Ptr to share ownership with.
    */
    template<typename Y>
    Ptr(const Ptr<Y>& o);

    /** @overload
    @param o Ptr to share ownership with.
    @param p Pointer to store.
    */
    template<typename Y>
    Ptr(const Ptr<Y>& o, T* p);

    /** The destructor is equivalent to calling Ptr::release. */
    ~Ptr();

    /**
    Assignment replaces the current Ptr instance with one that owns and stores same pointers as o and
    then destroys the old instance.
    @param o Ptr to share ownership with.
     */
    Ptr& operator = (const Ptr& o);

    /** @overload */
    template<typename Y>
    Ptr& operator = (const Ptr<Y>& o);

    /** If no other Ptr instance owns the owned pointer, deletes it with the associated deleter. Then sets
    both the owned and the stored pointers to NULL.
    */
    void release();

    /**
    `ptr.reset(...)` is equivalent to `ptr = Ptr<T>(...)`.
    @param p Pointer to own.
    */
    template<typename Y>
    void reset(Y* p);

    /** @overload
    @param d Deleter to use for the owned pointer.
    @param p Pointer to own.
    */
    template<typename Y, typename D>
    void reset(Y* p, D d);

    /**
    Swaps the owned and stored pointers (and deleters, if any) of this and o.
    @param o Ptr to swap with.
    */
    void swap(Ptr& o);

    /** Returns the stored pointer. */
    T* get() const;

    /** Ordinary pointer emulation. */
    typename detail::RefOrVoid<T>::type operator * () const;

    /** Ordinary pointer emulation. */
    T* operator -> () const;

    /** Equivalent to get(). */
    operator T* () const;

    /** ptr.empty() is equivalent to `!ptr.get()`. */
    bool empty() const;

    /** Returns a Ptr that owns the same pointer as this, and stores the same
       pointer as this, except converted via static_cast to Y*.
    */
    template<typename Y>
    Ptr<Y> staticCast() const;

    /** Ditto for const_cast. */
    template<typename Y>
    Ptr<Y> constCast() const;

    /** Ditto for dynamic_cast. */
    template<typename Y>
    Ptr<Y> dynamicCast() const;

#ifdef CV_CXX_MOVE_SEMANTICS
    Ptr(Ptr&& o);
    Ptr& operator = (Ptr&& o);
#endif

private:
    detail::PtrOwner* owner;
    T* stored;

    template<typename Y>
    friend struct Ptr; // have to do this for the cross-type copy constructor
};

/** Equivalent to ptr1.swap(ptr2). Provided to help write generic algorithms. */
template<typename T>
void swap(Ptr<T>& ptr1, Ptr<T>& ptr2);

/** Return whether ptr1.get() and ptr2.get() are equal and not equal, respectively. */
template<typename T>
bool operator == (const Ptr<T>& ptr1, const Ptr<T>& ptr2);
template<typename T>
bool operator != (const Ptr<T>& ptr1, const Ptr<T>& ptr2);

/** `makePtr<T>(...)` is equivalent to `Ptr<T>(new T(...))`. It is shorter than the latter, and it's
marginally safer than using a constructor or Ptr::reset, since it ensures that the owned pointer
is new and thus not owned by any other Ptr instance.
Unfortunately, perfect forwarding is impossible to implement in C++03, and so makePtr is limited
to constructors of T that have up to 10 arguments, none of which are non-const references.
 */
template<typename T>
Ptr<T> makePtr();
/** @overload */
template<typename T, typename A1>
Ptr<T> makePtr(const A1& a1);
/** @overload */
template<typename T, typename A1, typename A2>
Ptr<T> makePtr(const A1& a1, const A2& a2);
/** @overload */
template<typename T, typename A1, typename A2, typename A3>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9);
/** @overload */
template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10);

//////////////////////////////// string class ////////////////////////////////

class CV_EXPORTS FileNode; //for string constructor from FileNode

class CV_EXPORTS String
{
public:
    typedef char value_type;
    typedef char& reference;
    typedef const char& const_reference;
    typedef char* pointer;
    typedef const char* const_pointer;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;
    typedef char* iterator;
    typedef const char* const_iterator;

    static const size_t npos = size_t(-1);

    String();
    String(const String& str);
    String(const String& str, size_t pos, size_t len = npos);
    String(const char* s);
    String(const char* s, size_t n);
    String(size_t n, char c);
    String(const char* first, const char* last);
    template<typename Iterator> String(Iterator first, Iterator last);
    explicit String(const FileNode& fn);
    ~String();

    String& operator=(const String& str);
    String& operator=(const char* s);
    String& operator=(char c);

    String& operator+=(const String& str);
    String& operator+=(const char* s);
    String& operator+=(char c);

    size_t size() const;
    size_t length() const;

    char operator[](size_t idx) const;
    char operator[](int idx) const;

    const char* begin() const;
    const char* end() const;

    const char* c_str() const;

    bool empty() const;
    void clear();

    int compare(const char* s) const;
    int compare(const String& str) const;

    void swap(String& str);
    String substr(size_t pos = 0, size_t len = npos) const;

    size_t find(const char* s, size_t pos, size_t n) const;
    size_t find(char c, size_t pos = 0) const;
    size_t find(const String& str, size_t pos = 0) const;
    size_t find(const char* s, size_t pos = 0) const;

    size_t rfind(const char* s, size_t pos, size_t n) const;
    size_t rfind(char c, size_t pos = npos) const;
    size_t rfind(const String& str, size_t pos = npos) const;
    size_t rfind(const char* s, size_t pos = npos) const;

    size_t find_first_of(const char* s, size_t pos, size_t n) const;
    size_t find_first_of(char c, size_t pos = 0) const;
    size_t find_first_of(const String& str, size_t pos = 0) const;
    size_t find_first_of(const char* s, size_t pos = 0) const;

    size_t find_last_of(const char* s, size_t pos, size_t n) const;
    size_t find_last_of(char c, size_t pos = npos) const;
    size_t find_last_of(const String& str, size_t pos = npos) const;
    size_t find_last_of(const char* s, size_t pos = npos) const;

    friend String operator+ (const String& lhs, const String& rhs);
    friend String operator+ (const String& lhs, const char*   rhs);
    friend String operator+ (const char*   lhs, const String& rhs);
    friend String operator+ (const String& lhs, char          rhs);
    friend String operator+ (char          lhs, const String& rhs);

    String toLowerCase() const;

    String(const std::string& str);
    String(const std::string& str, size_t pos, size_t len = npos);
    String& operator=(const std::string& str);
    String& operator+=(const std::string& str);
    operator std::string() const;

    friend String operator+ (const String& lhs, const std::string& rhs);
    friend String operator+ (const std::string& lhs, const String& rhs);

private:
    char*  cstr_;
    size_t len_;

    char* allocate(size_t len); // len without trailing 0
    void deallocate();

    String(int); // disabled and invalid. Catch invalid usages like, commandLineParser.has(0) problem
};

//! @} core_basic

////////////////////////// cv::String implementation /////////////////////////

//! @cond IGNORED

inline
String::String()
    : cstr_(0), len_(0)
{}

inline
String::String(const String& str)
    : cstr_(str.cstr_), len_(str.len_)
{
    if (cstr_)
        CV_XADD(((int*)cstr_)-1, 1);
}

inline
String::String(const String& str, size_t pos, size_t len)
    : cstr_(0), len_(0)
{
    pos = min(pos, str.len_);
    len = min(str.len_ - pos, len);
    if (!len) return;
    if (len == str.len_)
    {
        CV_XADD(((int*)str.cstr_)-1, 1);
        cstr_ = str.cstr_;
        len_ = str.len_;
        return;
    }
    memcpy(allocate(len), str.cstr_ + pos, len);
}

inline
String::String(const char* s)
    : cstr_(0), len_(0)
{
    if (!s) return;
    size_t len = strlen(s);
    memcpy(allocate(len), s, len);
}

inline
String::String(const char* s, size_t n)
    : cstr_(0), len_(0)
{
    if (!n) return;
    if (!s) return;
    memcpy(allocate(n), s, n);
}

inline
String::String(size_t n, char c)
    : cstr_(0), len_(0)
{
    if (!n) return;
    memset(allocate(n), c, n);
}

inline
String::String(const char* first, const char* last)
    : cstr_(0), len_(0)
{
    size_t len = (size_t)(last - first);
    if (!len) return;
    memcpy(allocate(len), first, len);
}

template<typename Iterator> inline
String::String(Iterator first, Iterator last)
    : cstr_(0), len_(0)
{
    size_t len = (size_t)(last - first);
    if (!len) return;
    char* str = allocate(len);
    while (first != last)
    {
        *str++ = *first;
        ++first;
    }
}

inline
String::~String()
{
    deallocate();
}

inline
String& String::operator=(const String& str)
{
    if (&str == this) return *this;

    deallocate();
    if (str.cstr_) CV_XADD(((int*)str.cstr_)-1, 1);
    cstr_ = str.cstr_;
    len_ = str.len_;
    return *this;
}

inline
String& String::operator=(const char* s)
{
    deallocate();
    if (!s) return *this;
    size_t len = strlen(s);
    memcpy(allocate(len), s, len);
    return *this;
}

inline
String& String::operator=(char c)
{
    deallocate();
    allocate(1)[0] = c;
    return *this;
}

inline
String& String::operator+=(const String& str)
{
    *this = *this + str;
    return *this;
}

inline
String& String::operator+=(const char* s)
{
    *this = *this + s;
    return *this;
}

inline
String& String::operator+=(char c)
{
    *this = *this + c;
    return *this;
}

inline
size_t String::size() const
{
    return len_;
}

inline
size_t String::length() const
{
    return len_;
}

inline
char String::operator[](size_t idx) const
{
    return cstr_[idx];
}

inline
char String::operator[](int idx) const
{
    return cstr_[idx];
}

inline
const char* String::begin() const
{
    return cstr_;
}

inline
const char* String::end() const
{
    return len_ ? cstr_ + len_ : NULL;
}

inline
bool String::empty() const
{
    return len_ == 0;
}

inline
const char* String::c_str() const
{
    return cstr_ ? cstr_ : "";
}

inline
void String::swap(String& str)
{
    cv::swap(cstr_, str.cstr_);
    cv::swap(len_, str.len_);
}

inline
void String::clear()
{
    deallocate();
}

inline
int String::compare(const char* s) const
{
    if (cstr_ == s) return 0;
    return strcmp(c_str(), s);
}

inline
int String::compare(const String& str) const
{
    if (cstr_ == str.cstr_) return 0;
    return strcmp(c_str(), str.c_str());
}

inline
String String::substr(size_t pos, size_t len) const
{
    return String(*this, pos, len);
}

inline
size_t String::find(const char* s, size_t pos, size_t n) const
{
    if (n == 0 || pos + n > len_) return npos;
    const char* lmax = cstr_ + len_ - n;
    for (const char* i = cstr_ + pos; i <= lmax; ++i)
    {
        size_t j = 0;
        while (j < n && s[j] == i[j]) ++j;
        if (j == n) return (size_t)(i - cstr_);
    }
    return npos;
}

inline
size_t String::find(char c, size_t pos) const
{
    return find(&c, pos, 1);
}

inline
size_t String::find(const String& str, size_t pos) const
{
    return find(str.c_str(), pos, str.len_);
}

inline
size_t String::find(const char* s, size_t pos) const
{
    if (pos >= len_ || !s[0]) return npos;
    const char* lmax = cstr_ + len_;
    for (const char* i = cstr_ + pos; i < lmax; ++i)
    {
        size_t j = 0;
        while (s[j] && s[j] == i[j])
        {   if(i + j >= lmax) return npos;
            ++j;
        }
        if (!s[j]) return (size_t)(i - cstr_);
    }
    return npos;
}

inline
size_t String::rfind(const char* s, size_t pos, size_t n) const
{
    if (n > len_) return npos;
    if (pos > len_ - n) pos = len_ - n;
    for (const char* i = cstr_ + pos; i >= cstr_; --i)
    {
        size_t j = 0;
        while (j < n && s[j] == i[j]) ++j;
        if (j == n) return (size_t)(i - cstr_);
    }
    return npos;
}

inline
size_t String::rfind(char c, size_t pos) const
{
    return rfind(&c, pos, 1);
}

inline
size_t String::rfind(const String& str, size_t pos) const
{
    return rfind(str.c_str(), pos, str.len_);
}

inline
size_t String::rfind(const char* s, size_t pos) const
{
    return rfind(s, pos, strlen(s));
}

inline
size_t String::find_first_of(const char* s, size_t pos, size_t n) const
{
    if (n == 0 || pos + n > len_) return npos;
    const char* lmax = cstr_ + len_;
    for (const char* i = cstr_ + pos; i < lmax; ++i)
    {
        for (size_t j = 0; j < n; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline
size_t String::find_first_of(char c, size_t pos) const
{
    return find_first_of(&c, pos, 1);
}

inline
size_t String::find_first_of(const String& str, size_t pos) const
{
    return find_first_of(str.c_str(), pos, str.len_);
}

inline
size_t String::find_first_of(const char* s, size_t pos) const
{
    if (len_ == 0) return npos;
    if (pos >= len_ || !s[0]) return npos;
    const char* lmax = cstr_ + len_;
    for (const char* i = cstr_ + pos; i < lmax; ++i)
    {
        for (size_t j = 0; s[j]; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline
size_t String::find_last_of(const char* s, size_t pos, size_t n) const
{
    if (len_ == 0) return npos;
    if (pos >= len_) pos = len_ - 1;
    for (const char* i = cstr_ + pos; i >= cstr_; --i)
    {
        for (size_t j = 0; j < n; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline
size_t String::find_last_of(char c, size_t pos) const
{
    return find_last_of(&c, pos, 1);
}

inline
size_t String::find_last_of(const String& str, size_t pos) const
{
    return find_last_of(str.c_str(), pos, str.len_);
}

inline
size_t String::find_last_of(const char* s, size_t pos) const
{
    if (len_ == 0) return npos;
    if (pos >= len_) pos = len_ - 1;
    for (const char* i = cstr_ + pos; i >= cstr_; --i)
    {
        for (size_t j = 0; s[j]; ++j)
            if (s[j] == *i)
                return (size_t)(i - cstr_);
    }
    return npos;
}

inline
String String::toLowerCase() const
{
    if (!cstr_)
        return String();
    String res(cstr_, len_);
    for (size_t i = 0; i < len_; ++i)
        res.cstr_[i] = (char) ::tolower(cstr_[i]);

    return res;
}

//! @endcond

// ************************* cv::String non-member functions *************************

//! @relates cv::String
//! @{

inline
String operator + (const String& lhs, const String& rhs)
{
    String s;
    s.allocate(lhs.len_ + rhs.len_);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.cstr_, rhs.len_);
    return s;
}

inline
String operator + (const String& lhs, const char* rhs)
{
    String s;
    size_t rhslen = strlen(rhs);
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs, rhslen);
    return s;
}

inline
String operator + (const char* lhs, const String& rhs)
{
    String s;
    size_t lhslen = strlen(lhs);
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs, lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline
String operator + (const String& lhs, char rhs)
{
    String s;
    s.allocate(lhs.len_ + 1);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    s.cstr_[lhs.len_] = rhs;
    return s;
}

inline
String operator + (char lhs, const String& rhs)
{
    String s;
    s.allocate(rhs.len_ + 1);
    s.cstr_[0] = lhs;
    memcpy(s.cstr_ + 1, rhs.cstr_, rhs.len_);
    return s;
}

static inline bool operator== (const String& lhs, const String& rhs) { return 0 == lhs.compare(rhs); }
static inline bool operator== (const char*   lhs, const String& rhs) { return 0 == rhs.compare(lhs); }
static inline bool operator== (const String& lhs, const char*   rhs) { return 0 == lhs.compare(rhs); }
static inline bool operator!= (const String& lhs, const String& rhs) { return 0 != lhs.compare(rhs); }
static inline bool operator!= (const char*   lhs, const String& rhs) { return 0 != rhs.compare(lhs); }
static inline bool operator!= (const String& lhs, const char*   rhs) { return 0 != lhs.compare(rhs); }
static inline bool operator<  (const String& lhs, const String& rhs) { return lhs.compare(rhs) <  0; }
static inline bool operator<  (const char*   lhs, const String& rhs) { return rhs.compare(lhs) >  0; }
static inline bool operator<  (const String& lhs, const char*   rhs) { return lhs.compare(rhs) <  0; }
static inline bool operator<= (const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }
static inline bool operator<= (const char*   lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }
static inline bool operator<= (const String& lhs, const char*   rhs) { return lhs.compare(rhs) <= 0; }
static inline bool operator>  (const String& lhs, const String& rhs) { return lhs.compare(rhs) >  0; }
static inline bool operator>  (const char*   lhs, const String& rhs) { return rhs.compare(lhs) <  0; }
static inline bool operator>  (const String& lhs, const char*   rhs) { return lhs.compare(rhs) >  0; }
static inline bool operator>= (const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }
static inline bool operator>= (const char*   lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }
static inline bool operator>= (const String& lhs, const char*   rhs) { return lhs.compare(rhs) >= 0; }

//! @} relates cv::String

} // cv

namespace std
{
    static inline void swap(cv::String& a, cv::String& b) { a.swap(b); }
}

#include "opencv2/core/ptr.inl.hpp"

#endif //OPENCV_CORE_CVSTD_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_CVSTDINL_HPP
#define OPENCV_CORE_CVSTDINL_HPP

#include <complex>
#include <ostream>

//! @cond IGNORED

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif

namespace cv
{

template<typename _Tp> class DataType< std::complex<_Tp> >
{
public:
    typedef std::complex<_Tp>  value_type;
    typedef value_type         work_type;
    typedef _Tp                channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels) };

    typedef Vec<channel_type, channels> vec_type;
};

inline
String::String(const std::string& str)
    : cstr_(0), len_(0)
{
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
}

inline
String::String(const std::string& str, size_t pos, size_t len)
    : cstr_(0), len_(0)
{
    size_t strlen = str.size();
    pos = min(pos, strlen);
    len = min(strlen - pos, len);
    if (!len) return;
    memcpy(allocate(len), str.c_str() + pos, len);
}

inline
String& String::operator = (const std::string& str)
{
    deallocate();
    if (!str.empty())
    {
        size_t len = str.size();
        memcpy(allocate(len), str.c_str(), len);
    }
    return *this;
}

inline
String& String::operator += (const std::string& str)
{
    *this = *this + str;
    return *this;
}

inline
String::operator std::string() const
{
    return std::string(cstr_, len_);
}

inline
String operator + (const String& lhs, const std::string& rhs)
{
    String s;
    size_t rhslen = rhs.size();
    s.allocate(lhs.len_ + rhslen);
    memcpy(s.cstr_, lhs.cstr_, lhs.len_);
    memcpy(s.cstr_ + lhs.len_, rhs.c_str(), rhslen);
    return s;
}

inline
String operator + (const std::string& lhs, const String& rhs)
{
    String s;
    size_t lhslen = lhs.size();
    s.allocate(lhslen + rhs.len_);
    memcpy(s.cstr_, lhs.c_str(), lhslen);
    memcpy(s.cstr_ + lhslen, rhs.cstr_, rhs.len_);
    return s;
}

inline
FileNode::operator std::string() const
{
    String value;
    read(*this, value, value);
    return value;
}

template<> inline
void operator >> (const FileNode& n, std::string& value)
{
    read(n, value, std::string());
}

template<> inline
FileStorage& operator << (FileStorage& fs, const std::string& value)
{
    return fs << cv::String(value);
}

static inline
std::ostream& operator << (std::ostream& os, const String& str)
{
    return os << str.c_str();
}

static inline
std::ostream& operator << (std::ostream& out, Ptr<Formatted> fmtd)
{
    fmtd->reset();
    for(const char* str = fmtd->next(); str; str = fmtd->next())
        out << str;
    return out;
}

static inline
std::ostream& operator << (std::ostream& out, const Mat& mtx)
{
    return out << Formatter::get()->format(mtx);
}

static inline
std::ostream& operator << (std::ostream& out, const UMat& m)
{
    return out << m.getMat(ACCESS_READ);
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Complex<_Tp>& c)
{
    return out << "(" << c.re << "," << c.im << ")";
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const std::vector<Point_<_Tp> >& vec)
{
    return out << Formatter::get()->format(Mat(vec));
}


template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const std::vector<Point3_<_Tp> >& vec)
{
    return out << Formatter::get()->format(Mat(vec));
}


template<typename _Tp, int m, int n> static inline
std::ostream& operator << (std::ostream& out, const Matx<_Tp, m, n>& matx)
{
    return out << Formatter::get()->format(Mat(matx));
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Point_<_Tp>& p)
{
    out << "[" << p.x << ", " << p.y << "]";
    return out;
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Point3_<_Tp>& p)
{
    out << "[" << p.x << ", " << p.y << ", " << p.z << "]";
    return out;
}

template<typename _Tp, int n> static inline
std::ostream& operator << (std::ostream& out, const Vec<_Tp, n>& vec)
{
    out << "[";
    if(Vec<_Tp, n>::depth < CV_32F)
    {
        for (int i = 0; i < n - 1; ++i) {
            out << (int)vec[i] << ", ";
        }
        out << (int)vec[n-1] << "]";
    }
    else
    {
        for (int i = 0; i < n - 1; ++i) {
            out << vec[i] << ", ";
        }
        out << vec[n-1] << "]";
    }

    return out;
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Size_<_Tp>& size)
{
    return out << "[" << size.width << " x " << size.height << "]";
}

template<typename _Tp> static inline
std::ostream& operator << (std::ostream& out, const Rect_<_Tp>& rect)
{
    return out << "[" << rect.width << " x " << rect.height << " from (" << rect.x << ", " << rect.y << ")]";
}

static inline std::ostream& operator << (std::ostream& out, const MatSize& msize)
{
    int i, dims = msize.p[-1];
    for( i = 0; i < dims; i++ )
    {
        out << msize.p[i];
        if( i < dims-1 )
            out << " x ";
    }
    return out;
}

} // cv

#ifdef _MSC_VER
#pragma warning( pop )
#endif

//! @endcond

#endif // OPENCV_CORE_CVSTDINL_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_DIRECTX_HPP
#define OPENCV_CORE_DIRECTX_HPP

#include "mat.hpp"
#include "ocl.hpp"

#if !defined(__d3d11_h__)
struct ID3D11Device;
struct ID3D11Texture2D;
#endif

#if !defined(__d3d10_h__)
struct ID3D10Device;
struct ID3D10Texture2D;
#endif

#if !defined(_D3D9_H_)
struct IDirect3DDevice9;
struct IDirect3DDevice9Ex;
struct IDirect3DSurface9;
#endif


namespace cv { namespace directx {

namespace ocl {
using namespace cv::ocl;

//! @addtogroup core_directx
// This section describes OpenCL and DirectX interoperability.
//
// To enable DirectX support, configure OpenCV using CMake with WITH_DIRECTX=ON . Note, DirectX is
// supported only on Windows.
//
// To use OpenCL functionality you should first initialize OpenCL context from DirectX resource.
//
//! @{

// TODO static functions in the Context class
//! @brief Creates OpenCL context from D3D11 device
//
//! @param pD3D11Device - pointer to D3D11 device
//! @return Returns reference to OpenCL Context
CV_EXPORTS Context& initializeContextFromD3D11Device(ID3D11Device* pD3D11Device);

//! @brief Creates OpenCL context from D3D10 device
//
//! @param pD3D10Device - pointer to D3D10 device
//! @return Returns reference to OpenCL Context
CV_EXPORTS Context& initializeContextFromD3D10Device(ID3D10Device* pD3D10Device);

//! @brief Creates OpenCL context from Direct3DDevice9Ex device
//
//! @param pDirect3DDevice9Ex - pointer to Direct3DDevice9Ex device
//! @return Returns reference to OpenCL Context
CV_EXPORTS Context& initializeContextFromDirect3DDevice9Ex(IDirect3DDevice9Ex* pDirect3DDevice9Ex);

//! @brief Creates OpenCL context from Direct3DDevice9 device
//
//! @param pDirect3DDevice9 - pointer to Direct3Device9 device
//! @return Returns reference to OpenCL Context
CV_EXPORTS Context& initializeContextFromDirect3DDevice9(IDirect3DDevice9* pDirect3DDevice9);

//! @}

} // namespace cv::directx::ocl

//! @addtogroup core_directx
//! @{

//! @brief Converts InputArray to ID3D11Texture2D. If destination texture format is DXGI_FORMAT_NV12 then
//!        input UMat expected to be in BGR format and data will be downsampled and color-converted to NV12.
//
//! @note Note: Destination texture must be allocated by application. Function does memory copy from src to
//!             pD3D11Texture2D
//
//! @param src - source InputArray
//! @param pD3D11Texture2D - destination D3D11 texture
CV_EXPORTS void convertToD3D11Texture2D(InputArray src, ID3D11Texture2D* pD3D11Texture2D);

//! @brief Converts ID3D11Texture2D to OutputArray. If input texture format is DXGI_FORMAT_NV12 then
//!        data will be upsampled and color-converted to BGR format.
//
//! @note Note: Destination matrix will be re-allocated if it has not enough memory to match texture size.
//!             function does memory copy from pD3D11Texture2D to dst
//
//! @param pD3D11Texture2D - source D3D11 texture
//! @param dst             - destination OutputArray
CV_EXPORTS void convertFromD3D11Texture2D(ID3D11Texture2D* pD3D11Texture2D, OutputArray dst);

//! @brief Converts InputArray to ID3D10Texture2D
//
//! @note Note: function does memory copy from src to
//!             pD3D10Texture2D
//
//! @param src             - source InputArray
//! @param pD3D10Texture2D - destination D3D10 texture
CV_EXPORTS void convertToD3D10Texture2D(InputArray src, ID3D10Texture2D* pD3D10Texture2D);

//! @brief Converts ID3D10Texture2D to OutputArray
//
//! @note Note: function does memory copy from pD3D10Texture2D
//!             to dst
//
//! @param pD3D10Texture2D - source D3D10 texture
//! @param dst             - destination OutputArray
CV_EXPORTS void convertFromD3D10Texture2D(ID3D10Texture2D* pD3D10Texture2D, OutputArray dst);

//! @brief Converts InputArray to IDirect3DSurface9
//
//! @note Note: function does memory copy from src to
//!             pDirect3DSurface9
//
//! @param src                 - source InputArray
//! @param pDirect3DSurface9   - destination D3D10 texture
//! @param surfaceSharedHandle - shared handle
CV_EXPORTS void convertToDirect3DSurface9(InputArray src, IDirect3DSurface9* pDirect3DSurface9, void* surfaceSharedHandle = NULL);

//! @brief Converts IDirect3DSurface9 to OutputArray
//
//! @note Note: function does memory copy from pDirect3DSurface9
//!             to dst
//
//! @param pDirect3DSurface9   - source D3D10 texture
//! @param dst                 - destination OutputArray
//! @param surfaceSharedHandle - shared handle
CV_EXPORTS void convertFromDirect3DSurface9(IDirect3DSurface9* pDirect3DSurface9, OutputArray dst, void* surfaceSharedHandle = NULL);

//! @brief Get OpenCV type from DirectX type
//! @param iDXGI_FORMAT - enum DXGI_FORMAT for D3D10/D3D11
//! @return OpenCV type or -1 if there is no equivalent
CV_EXPORTS int getTypeFromDXGI_FORMAT(const int iDXGI_FORMAT); // enum DXGI_FORMAT for D3D10/D3D11

//! @brief Get OpenCV type from DirectX type
//! @param iD3DFORMAT - enum D3DTYPE for D3D9
//! @return OpenCV type or -1 if there is no equivalent
CV_EXPORTS int getTypeFromD3DFORMAT(const int iD3DFORMAT); // enum D3DTYPE for D3D9

//! @}

} } // namespace cv::directx

#endif // OPENCV_CORE_DIRECTX_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


#ifndef OPENCV_CORE_EIGEN_HPP
#define OPENCV_CORE_EIGEN_HPP

#include "opencv2/core.hpp"

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4714 ) //__forceinline is not inlined
#pragma warning( disable: 4127 ) //conditional expression is constant
#pragma warning( disable: 4244 ) //conversion from '__int64' to 'int', possible loss of data
#endif

namespace cv
{

//! @addtogroup core_eigen
//! @{

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
              (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src,
               Matx<_Tp, _rows, _cols>& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        dst = Matx<_Tp, _cols, _rows>(static_cast<const _Tp*>(src.data())).t();
    }
    else
    {
        dst = Matx<_Tp, _rows, _cols>(static_cast<const _Tp*>(src.data()));
    }
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    CV_DbgAssert(src.rows == _rows && src.cols == _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols> static inline
void cv2eigen( const Matx<_Tp, _rows, _cols>& src,
               Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, _rows, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, _cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

template<typename _Tp>  static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
             dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows, int _cols> static inline
void cv2eigen( const Matx<_Tp, _rows, _cols>& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(_rows, _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, _rows, DataType<_Tp>::type,
             dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, _cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

template<typename _Tp> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    CV_Assert(src.cols == 1);
    dst.resize(src.rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

// Matx case
template<typename _Tp, int _rows> static inline
void cv2eigen( const Matx<_Tp, _rows, 1>& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    dst.resize(_rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(1, _rows, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(_rows, 1, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.copyTo(_dst);
    }
}


template<typename _Tp> static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    CV_Assert(src.rows == 1);
    dst.resize(src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}

//Matx
template<typename _Tp, int _cols> static inline
void cv2eigen( const Matx<_Tp, 1, _cols>& src,
               Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    dst.resize(_cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(_cols, 1, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        transpose(src, _dst);
    }
    else
    {
        const Mat _dst(1, _cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        Mat(src).copyTo(_dst);
    }
}

//! @}

} // cv

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_FAST_MATH_HPP
#define OPENCV_CORE_FAST_MATH_HPP

#include "opencv2/core/cvdef.h"

#if ((defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ \
    && defined __SSE2__ && !defined __APPLE__)) && !defined(__CUDACC__)
#include <emmintrin.h>
#endif


//! @addtogroup core_utils
//! @{

/****************************************************************************************\
*                                      fast math                                         *
\****************************************************************************************/

#ifdef __cplusplus
#  include <cmath>
#else
#  ifdef __BORLANDC__
#    include <fastmath.h>
#  else
#    include <math.h>
#  endif
#endif

#ifdef HAVE_TEGRA_OPTIMIZATION
#  include "tegra_round.hpp"
#endif

#if defined __GNUC__ && defined __arm__ && (defined __ARM_PCS_VFP || defined __ARM_VFPV3__ || defined __ARM_NEON__) && !defined __SOFTFP__ && !defined(__CUDACC__)
    // 1. general scheme
    #define ARM_ROUND(_value, _asm_string) \
        int res; \
        float temp; \
        (void)temp; \
        __asm__(_asm_string : [res] "=r" (res), [temp] "=w" (temp) : [value] "w" (_value)); \
        return res
    // 2. version for double
    #ifdef __clang__
        #define ARM_ROUND_DBL(value) ARM_ROUND(value, "vcvtr.s32.f64 %[temp], %[value] \n vmov %[res], %[temp]")
    #else
        #define ARM_ROUND_DBL(value) ARM_ROUND(value, "vcvtr.s32.f64 %[temp], %P[value] \n vmov %[res], %[temp]")
    #endif
    // 3. version for float
    #define ARM_ROUND_FLT(value) ARM_ROUND(value, "vcvtr.s32.f32 %[temp], %[value]\n vmov %[res], %[temp]")
#endif

/** @brief Rounds floating-point number to the nearest integer

 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
CV_INLINE int
cvRound( double value )
{
#if ((defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ \
    && defined __SSE2__ && !defined __APPLE__) || CV_SSE2) && !defined(__CUDACC__)
    __m128d t = _mm_set_sd( value );
    return _mm_cvtsd_si32(t);
#elif defined _MSC_VER && defined _M_IX86
    int t;
    __asm
    {
        fld value;
        fistp t;
    }
    return t;
#elif ((defined _MSC_VER && defined _M_ARM) || defined CV_ICC || \
        defined __GNUC__) && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND_DBL(value);
#elif defined CV_ICC || defined __GNUC__
# if defined ARM_ROUND_DBL
    ARM_ROUND_DBL(value);
# else
    return (int)lrint(value);
# endif
#else
    /* it's ok if round does not comply with IEEE754 standard;
       the tests should allow +/-1 difference when the tested functions use round */
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
#endif
}


/** @brief Rounds floating-point number to the nearest integer not larger than the original.

 The function computes an integer i such that:
 \f[i \le \texttt{value} < i+1\f]
 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
CV_INLINE int cvFloor( double value )
{
    int i = (int)value;
    return i - (i > value);
}

/** @brief Rounds floating-point number to the nearest integer not smaller than the original.

 The function computes an integer i such that:
 \f[i \le \texttt{value} < i+1\f]
 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
CV_INLINE int cvCeil( double value )
{
    int i = (int)value;
    return i + (i < value);
}

/** @brief Determines if the argument is Not A Number.

 @param value The input floating-point value

 The function returns 1 if the argument is Not A Number (as defined by IEEE754 standard), 0
 otherwise. */
CV_INLINE int cvIsNaN( double value )
{
    Cv64suf ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
           ((unsigned)ieee754.u != 0) > 0x7ff00000;
}

/** @brief Determines if the argument is Infinity.

 @param value The input floating-point value

 The function returns 1 if the argument is a plus or minus infinity (as defined by IEEE754 standard)
 and 0 otherwise. */
CV_INLINE int cvIsInf( double value )
{
    Cv64suf ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) == 0x7ff00000 &&
            (unsigned)ieee754.u == 0;
}

#ifdef __cplusplus

/** @overload */
CV_INLINE int cvRound(float value)
{
#if ((defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__ \
    && defined __SSE2__ && !defined __APPLE__) || CV_SSE2) && !defined(__CUDACC__)
    __m128 t = _mm_set_ss( value );
    return _mm_cvtss_si32(t);
#elif defined _MSC_VER && defined _M_IX86
    int t;
    __asm
    {
        fld value;
        fistp t;
    }
    return t;
#elif ((defined _MSC_VER && defined _M_ARM) || defined CV_ICC || \
        defined __GNUC__) && defined HAVE_TEGRA_OPTIMIZATION
    TEGRA_ROUND_FLT(value);
#elif defined CV_ICC || defined __GNUC__
# if defined ARM_ROUND_FLT
    ARM_ROUND_FLT(value);
# else
    return (int)lrintf(value);
# endif
#else
    /* it's ok if round does not comply with IEEE754 standard;
     the tests should allow +/-1 difference when the tested functions use round */
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

/** @overload */
CV_INLINE int cvRound( int value )
{
    return value;
}

/** @overload */
CV_INLINE int cvFloor( float value )
{
    int i = (int)value;
    return i - (i > value);
}

/** @overload */
CV_INLINE int cvFloor( int value )
{
    return value;
}

/** @overload */
CV_INLINE int cvCeil( float value )
{
    int i = (int)value;
    return i + (i < value);
}

/** @overload */
CV_INLINE int cvCeil( int value )
{
    return value;
}

/** @overload */
CV_INLINE int cvIsNaN( float value )
{
    Cv32suf ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) > 0x7f800000;
}

/** @overload */
CV_INLINE int cvIsInf( float value )
{
    Cv32suf ieee754;
    ieee754.f = value;
    return (ieee754.u & 0x7fffffff) == 0x7f800000;
}

#endif // __cplusplus

//! @} core_utils

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_IPPASYNC_HPP
#define OPENCV_CORE_IPPASYNC_HPP

#ifdef HAVE_IPP_A

#include "opencv2/core.hpp"
#include <ipp_async_op.h>
#include <ipp_async_accel.h>

namespace cv
{

namespace hpp
{

/** @addtogroup core_ipp
This section describes conversion between OpenCV and [Intel&reg; IPP Asynchronous
C/C++](http://software.intel.com/en-us/intel-ipp-preview) library. [Getting Started
Guide](http://registrationcenter.intel.com/irc_nas/3727/ipp_async_get_started.htm) help you to
install the library, configure header and library build paths.
 */
//! @{

    //! convert OpenCV data type to hppDataType
    inline int toHppType(const int cvType)
    {
        int depth = CV_MAT_DEPTH(cvType);
        int hppType = depth == CV_8U ? HPP_DATA_TYPE_8U :
                     depth == CV_16U ? HPP_DATA_TYPE_16U :
                     depth == CV_16S ? HPP_DATA_TYPE_16S :
                     depth == CV_32S ? HPP_DATA_TYPE_32S :
                     depth == CV_32F ? HPP_DATA_TYPE_32F :
                     depth == CV_64F ? HPP_DATA_TYPE_64F : -1;
        CV_Assert( hppType >= 0 );
        return hppType;
    }

    //! convert hppDataType to OpenCV data type
    inline int toCvType(const int hppType)
    {
        int cvType = hppType == HPP_DATA_TYPE_8U ? CV_8U :
                    hppType == HPP_DATA_TYPE_16U ? CV_16U :
                    hppType == HPP_DATA_TYPE_16S ? CV_16S :
                    hppType == HPP_DATA_TYPE_32S ? CV_32S :
                    hppType == HPP_DATA_TYPE_32F ? CV_32F :
                    hppType == HPP_DATA_TYPE_64F ? CV_64F : -1;
        CV_Assert( cvType >= 0 );
        return cvType;
    }

    /** @brief Convert hppiMatrix to Mat.

    This function allocates and initializes new matrix (if needed) that has the same size and type as
    input matrix. Supports CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.
    @param src input hppiMatrix.
    @param dst output matrix.
    @param accel accelerator instance (see hpp::getHpp for the list of acceleration framework types).
    @param cn number of channels.
     */
    inline void copyHppToMat(hppiMatrix* src, Mat& dst, hppAccel accel, int cn)
    {
        hppDataType type;
        hpp32u width, height;
        hppStatus sts;

        if (src == NULL)
            return dst.release();

        sts = hppiInquireMatrix(src, &type, &width, &height);

        CV_Assert( sts == HPP_STATUS_NO_ERROR);

        int matType = CV_MAKETYPE(toCvType(type), cn);

        CV_Assert(width%cn == 0);

        width /= cn;

        dst.create((int)height, (int)width, (int)matType);

        size_t newSize = (size_t)(height*(hpp32u)(dst.step));

        sts = hppiGetMatrixData(accel,src,(hpp32u)(dst.step),dst.data,&newSize);

        CV_Assert( sts == HPP_STATUS_NO_ERROR);
    }

    /** @brief Create Mat from hppiMatrix.

    This function allocates and initializes the Mat that has the same size and type as input matrix.
    Supports CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.
    @param src input hppiMatrix.
    @param accel accelerator instance (see hpp::getHpp for the list of acceleration framework types).
    @param cn number of channels.
    @sa howToUseIPPAconversion, hpp::copyHppToMat, hpp::getHpp.
     */
    inline Mat getMat(hppiMatrix* src, hppAccel accel, int cn)
    {
        Mat dst;
        copyHppToMat(src, dst, accel, cn);
        return dst;
    }

    /** @brief Create hppiMatrix from Mat.

    This function allocates and initializes the hppiMatrix that has the same size and type as input
    matrix, returns the hppiMatrix*.

    If you want to use zero-copy for GPU you should to have 4KB aligned matrix data. See details
    [hppiCreateSharedMatrix](http://software.intel.com/ru-ru/node/501697).

    Supports CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F.

    @note The hppiMatrix pointer to the image buffer in system memory refers to the src.data. Control
    the lifetime of the matrix and don't change its data, if there is no special need.
    @param src input matrix.
    @param accel accelerator instance. Supports type:
    -   **HPP_ACCEL_TYPE_CPU** - accelerated by optimized CPU instructions.
    -   **HPP_ACCEL_TYPE_GPU** - accelerated by GPU programmable units or fixed-function
        accelerators.
    -   **HPP_ACCEL_TYPE_ANY** - any acceleration or no acceleration available.
    @sa howToUseIPPAconversion, hpp::getMat
     */
    inline hppiMatrix* getHpp(const Mat& src, hppAccel accel)
    {
        int htype = toHppType(src.type());
        int cn = src.channels();

        CV_Assert(src.data);
        hppAccelType accelType = hppQueryAccelType(accel);

        if (accelType!=HPP_ACCEL_TYPE_CPU)
        {
            hpp32u pitch, size;
            hppQueryMatrixAllocParams(accel, src.cols*cn, src.rows, htype, &pitch, &size);
            if (pitch!=0 && size!=0)
                if ((int)(src.data)%4096==0 && pitch==(hpp32u)(src.step))
                {
                    return hppiCreateSharedMatrix(htype, src.cols*cn, src.rows, src.data, pitch, size);
                }
        }

        return hppiCreateMatrix(htype, src.cols*cn, src.rows, src.data, (hpp32s)(src.step));;
    }

//! @}
}}

#endif

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_MAT_HPP
#define OPENCV_CORE_MAT_HPP

#ifndef __cplusplus
#  error mat.hpp header must be compiled as C++
#endif

#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"

#include "opencv2/core/bufferpool.hpp"

namespace cv
{

//! @addtogroup core_basic
//! @{

enum { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
    ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };

CV__DEBUG_NS_BEGIN

class CV_EXPORTS _OutputArray;

//////////////////////// Input/Output Array Arguments /////////////////////////////////

/** @brief This is the proxy class for passing read-only input arrays into OpenCV functions.

It is defined as:
@code
    typedef const _InputArray& InputArray;
@endcode
where _InputArray is a class that can be constructed from `Mat`, `Mat_<T>`, `Matx<T, m, n>`,
`std::vector<T>`, `std::vector<std::vector<T> >`, `std::vector<Mat>`, `std::vector<Mat_<T> >`,
`UMat`, `std::vector<UMat>` or `double`. It can also be constructed from a matrix expression.

Since this is mostly implementation-level class, and its interface may change in future versions, we
do not describe it in details. There are a few key things, though, that should be kept in mind:

-   When you see in the reference manual or in OpenCV source code a function that takes
    InputArray, it means that you can actually pass `Mat`, `Matx`, `vector<T>` etc. (see above the
    complete list).
-   Optional input arguments: If some of the input arrays may be empty, pass cv::noArray() (or
    simply cv::Mat() as you probably did before).
-   The class is designed solely for passing parameters. That is, normally you *should not*
    declare class members, local and global variables of this type.
-   If you want to design your own function or a class method that can operate of arrays of
    multiple types, you can use InputArray (or OutputArray) for the respective parameters. Inside
    a function you should use _InputArray::getMat() method to construct a matrix header for the
    array (without copying data). _InputArray::kind() can be used to distinguish Mat from
    `vector<>` etc., but normally it is not needed.

Here is how you can use a function that takes InputArray :
@code
    std::vector<Point2f> vec;
    // points or a circle
    for( int i = 0; i < 30; i++ )
        vec.push_back(Point2f((float)(100 + 30*cos(i*CV_PI*2/5)),
                              (float)(100 - 30*sin(i*CV_PI*2/5))));
    cv::transform(vec, vec, cv::Matx23f(0.707, -0.707, 10, 0.707, 0.707, 20));
@endcode
That is, we form an STL vector containing points, and apply in-place affine transformation to the
vector using the 2x3 matrix created inline as `Matx<float, 2, 3>` instance.

Here is how such a function can be implemented (for simplicity, we implement a very specific case of
it, according to the assertion statement inside) :
@code
    void myAffineTransform(InputArray _src, OutputArray _dst, InputArray _m)
    {
        // get Mat headers for input arrays. This is O(1) operation,
        // unless _src and/or _m are matrix expressions.
        Mat src = _src.getMat(), m = _m.getMat();
        CV_Assert( src.type() == CV_32FC2 && m.type() == CV_32F && m.size() == Size(3, 2) );

        // [re]create the output array so that it has the proper size and type.
        // In case of Mat it calls Mat::create, in case of STL vector it calls vector::resize.
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        for( int i = 0; i < src.rows; i++ )
            for( int j = 0; j < src.cols; j++ )
            {
                Point2f pt = src.at<Point2f>(i, j);
                dst.at<Point2f>(i, j) = Point2f(m.at<float>(0, 0)*pt.x +
                                                m.at<float>(0, 1)*pt.y +
                                                m.at<float>(0, 2),
                                                m.at<float>(1, 0)*pt.x +
                                                m.at<float>(1, 1)*pt.y +
                                                m.at<float>(1, 2));
            }
    }
@endcode
There is another related type, InputArrayOfArrays, which is currently defined as a synonym for
InputArray:
@code
    typedef InputArray InputArrayOfArrays;
@endcode
It denotes function arguments that are either vectors of vectors or vectors of matrices. A separate
synonym is needed to generate Python/Java etc. wrappers properly. At the function implementation
level their use is similar, but _InputArray::getMat(idx) should be used to get header for the
idx-th component of the outer vector and _InputArray::size().area() should be used to find the
number of components (vectors/matrices) of the outer vector.
 */
class CV_EXPORTS _InputArray
{
public:
    enum {
        KIND_SHIFT = 16,
        FIXED_TYPE = 0x8000 << KIND_SHIFT,
        FIXED_SIZE = 0x4000 << KIND_SHIFT,
        KIND_MASK = 31 << KIND_SHIFT,

        NONE              = 0 << KIND_SHIFT,
        MAT               = 1 << KIND_SHIFT,
        MATX              = 2 << KIND_SHIFT,
        STD_VECTOR        = 3 << KIND_SHIFT,
        STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
        STD_VECTOR_MAT    = 5 << KIND_SHIFT,
        EXPR              = 6 << KIND_SHIFT,
        OPENGL_BUFFER     = 7 << KIND_SHIFT,
        CUDA_HOST_MEM     = 8 << KIND_SHIFT,
        CUDA_GPU_MAT      = 9 << KIND_SHIFT,
        UMAT              =10 << KIND_SHIFT,
        STD_VECTOR_UMAT   =11 << KIND_SHIFT,
        STD_BOOL_VECTOR   =12 << KIND_SHIFT,
        STD_VECTOR_CUDA_GPU_MAT = 13 << KIND_SHIFT,
        STD_ARRAY         =14 << KIND_SHIFT,
        STD_ARRAY_MAT     =15 << KIND_SHIFT
    };

    _InputArray();
    _InputArray(int _flags, void* _obj);
    _InputArray(const Mat& m);
    _InputArray(const MatExpr& expr);
    _InputArray(const std::vector<Mat>& vec);
    template<typename _Tp> _InputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputArray(const std::vector<_Tp>& vec);
    _InputArray(const std::vector<bool>& vec);
    template<typename _Tp> _InputArray(const std::vector<std::vector<_Tp> >& vec);
    _InputArray(const std::vector<std::vector<bool> >&);
    template<typename _Tp> _InputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputArray(const Matx<_Tp, m, n>& matx);
    _InputArray(const double& val);
    _InputArray(const cuda::GpuMat& d_mat);
    _InputArray(const std::vector<cuda::GpuMat>& d_mat_array);
    _InputArray(const ogl::Buffer& buf);
    _InputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputArray(const cudev::GpuMat_<_Tp>& m);
    _InputArray(const UMat& um);
    _InputArray(const std::vector<UMat>& umv);

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> _InputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _InputArray(const std::array<Mat, _Nm>& arr);
#endif

    Mat getMat(int idx=-1) const;
    Mat getMat_(int idx=-1) const;
    UMat getUMat(int idx=-1) const;
    void getMatVector(std::vector<Mat>& mv) const;
    void getUMatVector(std::vector<UMat>& umv) const;
    void getGpuMatVector(std::vector<cuda::GpuMat>& gpumv) const;
    cuda::GpuMat getGpuMat() const;
    ogl::Buffer getOGlBuffer() const;

    int getFlags() const;
    void* getObj() const;
    Size getSz() const;

    int kind() const;
    int dims(int i=-1) const;
    int cols(int i=-1) const;
    int rows(int i=-1) const;
    Size size(int i=-1) const;
    int sizend(int* sz, int i=-1) const;
    bool sameSize(const _InputArray& arr) const;
    size_t total(int i=-1) const;
    int type(int i=-1) const;
    int depth(int i=-1) const;
    int channels(int i=-1) const;
    bool isContinuous(int i=-1) const;
    bool isSubmatrix(int i=-1) const;
    bool empty() const;
    void copyTo(const _OutputArray& arr) const;
    void copyTo(const _OutputArray& arr, const _InputArray & mask) const;
    size_t offset(int i=-1) const;
    size_t step(int i=-1) const;
    bool isMat() const;
    bool isUMat() const;
    bool isMatVector() const;
    bool isUMatVector() const;
    bool isMatx() const;
    bool isVector() const;
    bool isGpuMatVector() const;
    ~_InputArray();

protected:
    int flags;
    void* obj;
    Size sz;

    void init(int _flags, const void* _obj);
    void init(int _flags, const void* _obj, Size _sz);
};


/** @brief This type is very similar to InputArray except that it is used for input/output and output function
parameters.

Just like with InputArray, OpenCV users should not care about OutputArray, they just pass `Mat`,
`vector<T>` etc. to the functions. The same limitation as for `InputArray`: *Do not explicitly
create OutputArray instances* applies here too.

If you want to make your function polymorphic (i.e. accept different arrays as output parameters),
it is also not very difficult. Take the sample above as the reference. Note that
_OutputArray::create() needs to be called before _OutputArray::getMat(). This way you guarantee
that the output array is properly allocated.

Optional output parameters. If you do not need certain output array to be computed and returned to
you, pass cv::noArray(), just like you would in the case of optional input array. At the
implementation level, use _OutputArray::needed() to check if certain output array needs to be
computed or not.

There are several synonyms for OutputArray that are used to assist automatic Python/Java/... wrapper
generators:
@code
    typedef OutputArray OutputArrayOfArrays;
    typedef OutputArray InputOutputArray;
    typedef OutputArray InputOutputArrayOfArrays;
@endcode
 */
class CV_EXPORTS _OutputArray : public _InputArray
{
public:
    enum
    {
        DEPTH_MASK_8U = 1 << CV_8U,
        DEPTH_MASK_8S = 1 << CV_8S,
        DEPTH_MASK_16U = 1 << CV_16U,
        DEPTH_MASK_16S = 1 << CV_16S,
        DEPTH_MASK_32S = 1 << CV_32S,
        DEPTH_MASK_32F = 1 << CV_32F,
        DEPTH_MASK_64F = 1 << CV_64F,
        DEPTH_MASK_ALL = (DEPTH_MASK_64F<<1)-1,
        DEPTH_MASK_ALL_BUT_8S = DEPTH_MASK_ALL & ~DEPTH_MASK_8S,
        DEPTH_MASK_FLT = DEPTH_MASK_32F + DEPTH_MASK_64F
    };

    _OutputArray();
    _OutputArray(int _flags, void* _obj);
    _OutputArray(Mat& m);
    _OutputArray(std::vector<Mat>& vec);
    _OutputArray(cuda::GpuMat& d_mat);
    _OutputArray(std::vector<cuda::GpuMat>& d_mat);
    _OutputArray(ogl::Buffer& buf);
    _OutputArray(cuda::HostMem& cuda_mem);
    template<typename _Tp> _OutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(std::vector<_Tp>& vec);
    _OutputArray(std::vector<bool>& vec);
    template<typename _Tp> _OutputArray(std::vector<std::vector<_Tp> >& vec);
    _OutputArray(std::vector<std::vector<bool> >&);
    template<typename _Tp> _OutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(Matx<_Tp, m, n>& matx);
    _OutputArray(UMat& m);
    _OutputArray(std::vector<UMat>& vec);

    _OutputArray(const Mat& m);
    _OutputArray(const std::vector<Mat>& vec);
    _OutputArray(const cuda::GpuMat& d_mat);
    _OutputArray(const std::vector<cuda::GpuMat>& d_mat);
    _OutputArray(const ogl::Buffer& buf);
    _OutputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _OutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _OutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _OutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _OutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _OutputArray(const Matx<_Tp, m, n>& matx);
    _OutputArray(const UMat& m);
    _OutputArray(const std::vector<UMat>& vec);

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> _OutputArray(std::array<_Tp, _Nm>& arr);
    template<typename _Tp, std::size_t _Nm> _OutputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _OutputArray(std::array<Mat, _Nm>& arr);
    template<std::size_t _Nm> _OutputArray(const std::array<Mat, _Nm>& arr);
#endif

    bool fixedSize() const;
    bool fixedType() const;
    bool needed() const;
    Mat& getMatRef(int i=-1) const;
    UMat& getUMatRef(int i=-1) const;
    cuda::GpuMat& getGpuMatRef() const;
    std::vector<cuda::GpuMat>& getGpuMatVecRef() const;
    ogl::Buffer& getOGlBufferRef() const;
    cuda::HostMem& getHostMemRef() const;
    void create(Size sz, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    void create(int rows, int cols, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    void create(int dims, const int* size, int type, int i=-1, bool allowTransposed=false, int fixedDepthMask=0) const;
    void createSameSize(const _InputArray& arr, int mtype) const;
    void release() const;
    void clear() const;
    void setTo(const _InputArray& value, const _InputArray & mask = _InputArray()) const;

    void assign(const UMat& u) const;
    void assign(const Mat& m) const;
};


class CV_EXPORTS _InputOutputArray : public _OutputArray
{
public:
    _InputOutputArray();
    _InputOutputArray(int _flags, void* _obj);
    _InputOutputArray(Mat& m);
    _InputOutputArray(std::vector<Mat>& vec);
    _InputOutputArray(cuda::GpuMat& d_mat);
    _InputOutputArray(ogl::Buffer& buf);
    _InputOutputArray(cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(std::vector<_Tp>& vec);
    _InputOutputArray(std::vector<bool>& vec);
    template<typename _Tp> _InputOutputArray(std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(_Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(Matx<_Tp, m, n>& matx);
    _InputOutputArray(UMat& m);
    _InputOutputArray(std::vector<UMat>& vec);

    _InputOutputArray(const Mat& m);
    _InputOutputArray(const std::vector<Mat>& vec);
    _InputOutputArray(const cuda::GpuMat& d_mat);
    _InputOutputArray(const std::vector<cuda::GpuMat>& d_mat);
    _InputOutputArray(const ogl::Buffer& buf);
    _InputOutputArray(const cuda::HostMem& cuda_mem);
    template<typename _Tp> _InputOutputArray(const cudev::GpuMat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const std::vector<_Tp>& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<std::vector<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const std::vector<Mat_<_Tp> >& vec);
    template<typename _Tp> _InputOutputArray(const Mat_<_Tp>& m);
    template<typename _Tp> _InputOutputArray(const _Tp* vec, int n);
    template<typename _Tp, int m, int n> _InputOutputArray(const Matx<_Tp, m, n>& matx);
    _InputOutputArray(const UMat& m);
    _InputOutputArray(const std::vector<UMat>& vec);

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> _InputOutputArray(std::array<_Tp, _Nm>& arr);
    template<typename _Tp, std::size_t _Nm> _InputOutputArray(const std::array<_Tp, _Nm>& arr);
    template<std::size_t _Nm> _InputOutputArray(std::array<Mat, _Nm>& arr);
    template<std::size_t _Nm> _InputOutputArray(const std::array<Mat, _Nm>& arr);
#endif

};

CV__DEBUG_NS_END

typedef const _InputArray& InputArray;
typedef InputArray InputArrayOfArrays;
typedef const _OutputArray& OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray& InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

CV_EXPORTS InputOutputArray noArray();

/////////////////////////////////// MatAllocator //////////////////////////////////////

//! Usage flags for allocator
enum UMatUsageFlags
{
    USAGE_DEFAULT = 0,

    // buffer allocation policy is platform and usage specific
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
    USAGE_ALLOCATE_SHARED_MEMORY = 1 << 2, // It is not equal to: USAGE_ALLOCATE_HOST_MEMORY | USAGE_ALLOCATE_DEVICE_MEMORY

    __UMAT_USAGE_FLAGS_32BIT = 0x7fffffff // Binary compatibility hint
};

struct CV_EXPORTS UMatData;

/** @brief  Custom array allocator
*/
class CV_EXPORTS MatAllocator
{
public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    // let's comment it off for now to detect and fix all the uses of allocator
    //virtual void allocate(int dims, const int* sizes, int type, int*& refcount,
    //                      uchar*& datastart, uchar*& data, size_t* step) = 0;
    //virtual void deallocate(int* refcount, uchar* datastart, uchar* data) = 0;
    virtual UMatData* allocate(int dims, const int* sizes, int type,
                               void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const = 0;
    virtual bool allocate(UMatData* data, int accessflags, UMatUsageFlags usageFlags) const = 0;
    virtual void deallocate(UMatData* data) const = 0;
    virtual void map(UMatData* data, int accessflags) const;
    virtual void unmap(UMatData* data) const;
    virtual void download(UMatData* data, void* dst, int dims, const size_t sz[],
                          const size_t srcofs[], const size_t srcstep[],
                          const size_t dststep[]) const;
    virtual void upload(UMatData* data, const void* src, int dims, const size_t sz[],
                        const size_t dstofs[], const size_t dststep[],
                        const size_t srcstep[]) const;
    virtual void copy(UMatData* srcdata, UMatData* dstdata, int dims, const size_t sz[],
                      const size_t srcofs[], const size_t srcstep[],
                      const size_t dstofs[], const size_t dststep[], bool sync) const;

    // default implementation returns DummyBufferPoolController
    virtual BufferPoolController* getBufferPoolController(const char* id = NULL) const;
};


//////////////////////////////// MatCommaInitializer //////////////////////////////////

/** @brief  Comma-separated Matrix Initializer

 The class instances are usually not created explicitly.
 Instead, they are created on "matrix << firstValue" operator.

 The sample below initializes 2x2 rotation matrix:

 \code
 double angle = 30, a = cos(angle*CV_PI/180), b = sin(angle*CV_PI/180);
 Mat R = (Mat_<double>(2,2) << a, -b, b, a);
 \endcode
*/
template<typename _Tp> class MatCommaInitializer_
{
public:
    //! the constructor, created by "matrix << firstValue" operator, where matrix is cv::Mat
    MatCommaInitializer_(Mat_<_Tp>* _m);
    //! the operator that takes the next value and put it to the matrix
    template<typename T2> MatCommaInitializer_<_Tp>& operator , (T2 v);
    //! another form of conversion operator
    operator Mat_<_Tp>() const;
protected:
    MatIterator_<_Tp> it;
};


/////////////////////////////////////// Mat ///////////////////////////////////////////

// note that umatdata might be allocated together
// with the matrix data, not as a separate object.
// therefore, it does not have constructor or destructor;
// it should be explicitly initialized using init().
struct CV_EXPORTS UMatData
{
    enum { COPY_ON_MAP=1, HOST_COPY_OBSOLETE=2,
        DEVICE_COPY_OBSOLETE=4, TEMP_UMAT=8, TEMP_COPIED_UMAT=24,
        USER_ALLOCATED=32, DEVICE_MEM_MAPPED=64,
        ASYNC_CLEANUP=128
    };
    UMatData(const MatAllocator* allocator);
    ~UMatData();

    // provide atomic access to the structure
    void lock();
    void unlock();

    bool hostCopyObsolete() const;
    bool deviceCopyObsolete() const;
    bool deviceMemMapped() const;
    bool copyOnMap() const;
    bool tempUMat() const;
    bool tempCopiedUMat() const;
    void markHostCopyObsolete(bool flag);
    void markDeviceCopyObsolete(bool flag);
    void markDeviceMemMapped(bool flag);

    const MatAllocator* prevAllocator;
    const MatAllocator* currAllocator;
    int urefcount;
    int refcount;
    uchar* data;
    uchar* origdata;
    size_t size;

    int flags;
    void* handle;
    void* userdata;
    int allocatorFlags_;
    int mapcount;
    UMatData* originalUMatData;
};


struct CV_EXPORTS UMatDataAutoLock
{
    explicit UMatDataAutoLock(UMatData* u);
    ~UMatDataAutoLock();
    UMatData* u;
};


struct CV_EXPORTS MatSize
{
    explicit MatSize(int* _p);
    Size operator()() const;
    const int& operator[](int i) const;
    int& operator[](int i);
    operator const int*() const;
    bool operator == (const MatSize& sz) const;
    bool operator != (const MatSize& sz) const;

    int* p;
};

struct CV_EXPORTS MatStep
{
    MatStep();
    explicit MatStep(size_t s);
    const size_t& operator[](int i) const;
    size_t& operator[](int i);
    operator size_t() const;
    MatStep& operator = (size_t s);

    size_t* p;
    size_t buf[2];
protected:
    MatStep& operator = (const MatStep&);
};

/** @example cout_mat.cpp
An example demonstrating the serial out capabilities of cv::Mat
*/

 /** @brief n-dimensional dense array class

The class Mat represents an n-dimensional dense numerical single-channel or multi-channel array. It
can be used to store real or complex-valued vectors and matrices, grayscale or color images, voxel
volumes, vector fields, point clouds, tensors, histograms (though, very high-dimensional histograms
may be better stored in a SparseMat ). The data layout of the array `M` is defined by the array
`M.step[]`, so that the address of element \f$(i_0,...,i_{M.dims-1})\f$, where \f$0\leq i_k<M.size[k]\f$, is
computed as:
\f[addr(M_{i_0,...,i_{M.dims-1}}) = M.data + M.step[0]*i_0 + M.step[1]*i_1 + ... + M.step[M.dims-1]*i_{M.dims-1}\f]
In case of a 2-dimensional array, the above formula is reduced to:
\f[addr(M_{i,j}) = M.data + M.step[0]*i + M.step[1]*j\f]
Note that `M.step[i] >= M.step[i+1]` (in fact, `M.step[i] >= M.step[i+1]*M.size[i+1]` ). This means
that 2-dimensional matrices are stored row-by-row, 3-dimensional matrices are stored plane-by-plane,
and so on. M.step[M.dims-1] is minimal and always equal to the element size M.elemSize() .

So, the data layout in Mat is fully compatible with CvMat, IplImage, and CvMatND types from OpenCV
1.x. It is also compatible with the majority of dense array types from the standard toolkits and
SDKs, such as Numpy (ndarray), Win32 (independent device bitmaps), and others, that is, with any
array that uses *steps* (or *strides*) to compute the position of a pixel. Due to this
compatibility, it is possible to make a Mat header for user-allocated data and process it in-place
using OpenCV functions.

There are many different ways to create a Mat object. The most popular options are listed below:

- Use the create(nrows, ncols, type) method or the similar Mat(nrows, ncols, type[, fillValue])
constructor. A new array of the specified size and type is allocated. type has the same meaning as
in the cvCreateMat method. For example, CV_8UC1 means a 8-bit single-channel array, CV_32FC2
means a 2-channel (complex) floating-point array, and so on.
@code
    // make a 7x7 complex matrix filled with 1+3j.
    Mat M(7,7,CV_32FC2,Scalar(1,3));
    // and now turn M to a 100x60 15-channel 8-bit matrix.
    // The old content will be deallocated
    M.create(100,60,CV_8UC(15));
@endcode
As noted in the introduction to this chapter, create() allocates only a new array when the shape
or type of the current array are different from the specified ones.

- Create a multi-dimensional array:
@code
    // create a 100x100x100 8-bit array
    int sz[] = {100, 100, 100};
    Mat bigCube(3, sz, CV_8U, Scalar::all(0));
@endcode
It passes the number of dimensions =1 to the Mat constructor but the created array will be
2-dimensional with the number of columns set to 1. So, Mat::dims is always \>= 2 (can also be 0
when the array is empty).

- Use a copy constructor or assignment operator where there can be an array or expression on the
right side (see below). As noted in the introduction, the array assignment is an O(1) operation
because it only copies the header and increases the reference counter. The Mat::clone() method can
be used to get a full (deep) copy of the array when you need it.

- Construct a header for a part of another array. It can be a single row, single column, several
rows, several columns, rectangular region in the array (called a *minor* in algebra) or a
diagonal. Such operations are also O(1) because the new header references the same data. You can
actually modify a part of the array using this feature, for example:
@code
    // add the 5-th row, multiplied by 3 to the 3rd row
    M.row(3) = M.row(3) + M.row(5)*3;
    // now copy the 7-th column to the 1-st column
    // M.col(1) = M.col(7); // this will not work
    Mat M1 = M.col(1);
    M.col(7).copyTo(M1);
    // create a new 320x240 image
    Mat img(Size(320,240),CV_8UC3);
    // select a ROI
    Mat roi(img, Rect(10,10,100,100));
    // fill the ROI with (0,255,0) (which is green in RGB space);
    // the original 320x240 image will be modified
    roi = Scalar(0,255,0);
@endcode
Due to the additional datastart and dataend members, it is possible to compute a relative
sub-array position in the main *container* array using locateROI():
@code
    Mat A = Mat::eye(10, 10, CV_32S);
    // extracts A columns, 1 (inclusive) to 3 (exclusive).
    Mat B = A(Range::all(), Range(1, 3));
    // extracts B rows, 5 (inclusive) to 9 (exclusive).
    // that is, C \~ A(Range(5, 9), Range(1, 3))
    Mat C = B(Range(5, 9), Range::all());
    Size size; Point ofs;
    C.locateROI(size, ofs);
    // size will be (width=10,height=10) and the ofs will be (x=1, y=5)
@endcode
As in case of whole matrices, if you need a deep copy, use the `clone()` method of the extracted
sub-matrices.

- Make a header for user-allocated data. It can be useful to do the following:
    -# Process "foreign" data using OpenCV (for example, when you implement a DirectShow\* filter or
    a processing module for gstreamer, and so on). For example:
    @code
        void process_video_frame(const unsigned char* pixels,
                                 int width, int height, int step)
        {
            Mat img(height, width, CV_8UC3, pixels, step);
            GaussianBlur(img, img, Size(7,7), 1.5, 1.5);
        }
    @endcode
    -# Quickly initialize small matrices and/or get a super-fast element access.
    @code
        double m[3][3] = {{a, b, c}, {d, e, f}, {g, h, i}};
        Mat M = Mat(3, 3, CV_64F, m).inv();
    @endcode
    .
    Partial yet very common cases of this *user-allocated data* case are conversions from CvMat and
    IplImage to Mat. For this purpose, there is function cv::cvarrToMat taking pointers to CvMat or
    IplImage and the optional flag indicating whether to copy the data or not.
    @snippet samples/cpp/image.cpp iplimage

- Use MATLAB-style array initializers, zeros(), ones(), eye(), for example:
@code
    // create a double-precision identity matrix and add it to M.
    M += Mat::eye(M.rows, M.cols, CV_64F);
@endcode

- Use a comma-separated initializer:
@code
    // create a 3x3 double-precision identity matrix
    Mat M = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
@endcode
With this approach, you first call a constructor of the Mat class with the proper parameters, and
then you just put `<< operator` followed by comma-separated values that can be constants,
variables, expressions, and so on. Also, note the extra parentheses required to avoid compilation
errors.

Once the array is created, it is automatically managed via a reference-counting mechanism. If the
array header is built on top of user-allocated data, you should handle the data by yourself. The
array data is deallocated when no one points to it. If you want to release the data pointed by a
array header before the array destructor is called, use Mat::release().

The next important thing to learn about the array class is element access. This manual already
described how to compute an address of each array element. Normally, you are not required to use the
formula directly in the code. If you know the array element type (which can be retrieved using the
method Mat::type() ), you can access the element \f$M_{ij}\f$ of a 2-dimensional array as:
@code
    M.at<double>(i,j) += 1.f;
@endcode
assuming that `M` is a double-precision floating-point array. There are several variants of the method
at for a different number of dimensions.

If you need to process a whole row of a 2D array, the most efficient way is to get the pointer to
the row first, and then just use the plain C operator [] :
@code
    // compute sum of positive matrix elements
    // (assuming that M is a double-precision matrix)
    double sum=0;
    for(int i = 0; i < M.rows; i++)
    {
        const double* Mi = M.ptr<double>(i);
        for(int j = 0; j < M.cols; j++)
            sum += std::max(Mi[j], 0.);
    }
@endcode
Some operations, like the one above, do not actually depend on the array shape. They just process
elements of an array one by one (or elements from multiple arrays that have the same coordinates,
for example, array addition). Such operations are called *element-wise*. It makes sense to check
whether all the input/output arrays are continuous, namely, have no gaps at the end of each row. If
yes, process them as a long single row:
@code
    // compute the sum of positive matrix elements, optimized variant
    double sum=0;
    int cols = M.cols, rows = M.rows;
    if(M.isContinuous())
    {
        cols *= rows;
        rows = 1;
    }
    for(int i = 0; i < rows; i++)
    {
        const double* Mi = M.ptr<double>(i);
        for(int j = 0; j < cols; j++)
            sum += std::max(Mi[j], 0.);
    }
@endcode
In case of the continuous matrix, the outer loop body is executed just once. So, the overhead is
smaller, which is especially noticeable in case of small matrices.

Finally, there are STL-style iterators that are smart enough to skip gaps between successive rows:
@code
    // compute sum of positive matrix elements, iterator-based variant
    double sum=0;
    MatConstIterator_<double> it = M.begin<double>(), it_end = M.end<double>();
    for(; it != it_end; ++it)
        sum += std::max(*it, 0.);
@endcode
The matrix iterators are random-access iterators, so they can be passed to any STL algorithm,
including std::sort().

@note Matrix Expressions and arithmetic see MatExpr
*/
class CV_EXPORTS Mat
{
public:
    /**
    These are various constructors that form a matrix. As noted in the AutomaticAllocation, often
    the default constructor is enough, and the proper matrix will be allocated by an OpenCV function.
    The constructed matrix can further be assigned to another matrix or matrix expression or can be
    allocated with Mat::create . In the former case, the old content is de-referenced.
     */
    Mat();

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(int rows, int cols, int type);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
      */
    Mat(Size size, int type);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int rows, int cols, int type, const Scalar& s);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
      */
    Mat(Size size, int type, const Scalar& s);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(int ndims, const int* sizes, int type);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    */
    Mat(const std::vector<int>& sizes, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int ndims, const int* sizes, int type, const Scalar& s);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(const std::vector<int>& sizes, int type, const Scalar& s);


    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    */
    Mat(const Mat& m);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param step Number of bytes each matrix row occupies. The value should include the padding bytes at
    the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    */
    Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param step Number of bytes each matrix row occupies. The value should include the padding bytes at
    the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    */
    Mat(Size size, int type, void* data, size_t step=AUTO_STEP);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
    set to the element size). If not specified, the matrix is assumed to be continuous.
    */
    Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use CV_8UC1, ..., CV_64FC4 to create 1-4 channel matrices, or
    CV_8UC(n), ..., CV_64FC(n) to create multi-channel (up to CV_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
    set to the element size). If not specified, the matrix is assumed to be continuous.
    */
    Mat(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param rowRange Range of the m rows to take. As usual, the range start is inclusive and the range
    end is exclusive. Use Range::all() to take all the rows.
    @param colRange Range of the m columns to take. Use Range::all() to take all the columns.
    */
    Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param roi Region of interest.
    */
    Mat(const Mat& m, const Rect& roi);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param ranges Array of selected ranges of m along each dimensionality.
    */
    Mat(const Mat& m, const Range* ranges);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param ranges Array of selected ranges of m along each dimensionality.
    */
    Mat(const Mat& m, const std::vector<Range>& ranges);

    /** @overload
    @param vec STL vector whose elements form the matrix. The matrix has a single column and the number
    of rows equal to the number of vector elements. Type of the matrix matches the type of vector
    elements. The constructor can handle arbitrary types, for which there is a properly declared
    DataType . This means that the vector elements must be primitive numbers or uni-type numerical
    tuples of numbers. Mixed-type structures are not supported. The corresponding constructor is
    explicit. Since STL vectors are not automatically converted to Mat instances, you should write
    Mat(vec) explicitly. Unless you copy the data into the matrix ( copyData=true ), no new elements
    will be added to the vector because it can potentially yield vector data reallocation, and, thus,
    the matrix data pointer will be invalid.
    @param copyData Flag to specify whether the underlying data of the STL vector should be copied
    to (true) or shared with (false) the newly constructed matrix. When the data is copied, the
    allocated buffer is managed using Mat reference counting mechanism. While the data is shared,
    the reference counter is NULL, and you should not deallocate the data until the matrix is not
    destructed.
    */
    template<typename _Tp> explicit Mat(const std::vector<_Tp>& vec, bool copyData=false);

#ifdef CV_CXX11
    /** @overload
    */
    template<typename _Tp> explicit Mat(const std::initializer_list<_Tp> list);
#endif

#ifdef CV_CXX_STD_ARRAY
    /** @overload
    */
    template<typename _Tp, size_t _Nm> explicit Mat(const std::array<_Tp, _Nm>& arr, bool copyData=false);
#endif

    /** @overload
    */
    template<typename _Tp, int n> explicit Mat(const Vec<_Tp, n>& vec, bool copyData=true);

    /** @overload
    */
    template<typename _Tp, int m, int n> explicit Mat(const Matx<_Tp, m, n>& mtx, bool copyData=true);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const Point_<_Tp>& pt, bool copyData=true);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const Point3_<_Tp>& pt, bool copyData=true);

    /** @overload
    */
    template<typename _Tp> explicit Mat(const MatCommaInitializer_<_Tp>& commaInitializer);

    //! download data from GpuMat
    explicit Mat(const cuda::GpuMat& m);

    //! destructor - calls release()
    ~Mat();

    /** @brief assignment operators

    These are available assignment operators. Since they all are very different, make sure to read the
    operator parameters description.
    @param m Assigned, right-hand-side matrix. Matrix assignment is an O(1) operation. This means that
    no data is copied but the data is shared and the reference counter, if any, is incremented. Before
    assigning new data, the old data is de-referenced via Mat::release .
     */
    Mat& operator = (const Mat& m);

    /** @overload
    @param expr Assigned matrix expression object. As opposite to the first form of the assignment
    operation, the second form can reuse already allocated matrix if it has the right size and type to
    fit the matrix expression result. It is automatically handled by the real function that the matrix
    expressions is expanded to. For example, C=A+B is expanded to add(A, B, C), and add takes care of
    automatic C reallocation.
    */
    Mat& operator = (const MatExpr& expr);

    //! retrieve UMat from Mat
    UMat getUMat(int accessFlags, UMatUsageFlags usageFlags = USAGE_DEFAULT) const;

    /** @brief Creates a matrix header for the specified matrix row.

    The method makes a new header for the specified matrix row and returns it. This is an O(1)
    operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
    original matrix. Here is the example of one of the classical basic matrix processing operations,
    axpy, used by LU and many other algorithms:
    @code
        inline void matrix_axpy(Mat& A, int i, int j, double alpha)
        {
            A.row(i) += A.row(j)*alpha;
        }
    @endcode
    @note In the current implementation, the following code does not work as expected:
    @code
        Mat A;
        ...
        A.row(i) = A.row(j); // will not work
    @endcode
    This happens because A.row(i) forms a temporary header that is further assigned to another header.
    Remember that each of these operations is O(1), that is, no data is copied. Thus, the above
    assignment is not true if you may have expected the j-th row to be copied to the i-th row. To
    achieve that, you should either turn this simple assignment into an expression or use the
    Mat::copyTo method:
    @code
        Mat A;
        ...
        // works, but looks a bit obscure.
        A.row(i) = A.row(j) + 0;
        // this is a bit longer, but the recommended method.
        A.row(j).copyTo(A.row(i));
    @endcode
    @param y A 0-based row index.
     */
    Mat row(int y) const;

    /** @brief Creates a matrix header for the specified matrix column.

    The method makes a new header for the specified matrix column and returns it. This is an O(1)
    operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
    original matrix. See also the Mat::row description.
    @param x A 0-based column index.
     */
    Mat col(int x) const;

    /** @brief Creates a matrix header for the specified row span.

    The method makes a new header for the specified row span of the matrix. Similarly to Mat::row and
    Mat::col , this is an O(1) operation.
    @param startrow An inclusive 0-based start index of the row span.
    @param endrow An exclusive 0-based ending index of the row span.
     */
    Mat rowRange(int startrow, int endrow) const;

    /** @overload
    @param r Range structure containing both the start and the end indices.
    */
    Mat rowRange(const Range& r) const;

    /** @brief Creates a matrix header for the specified column span.

    The method makes a new header for the specified column span of the matrix. Similarly to Mat::row and
    Mat::col , this is an O(1) operation.
    @param startcol An inclusive 0-based start index of the column span.
    @param endcol An exclusive 0-based ending index of the column span.
     */
    Mat colRange(int startcol, int endcol) const;

    /** @overload
    @param r Range structure containing both the start and the end indices.
    */
    Mat colRange(const Range& r) const;

    /** @brief Extracts a diagonal from a matrix

    The method makes a new header for the specified matrix diagonal. The new matrix is represented as a
    single-column matrix. Similarly to Mat::row and Mat::col, this is an O(1) operation.
    @param d index of the diagonal, with the following values:
    - `d=0` is the main diagonal.
    - `d<0` is a diagonal from the lower half. For example, d=-1 means the diagonal is set
      immediately below the main one.
    - `d>0` is a diagonal from the upper half. For example, d=1 means the diagonal is set
      immediately above the main one.
    For example:
    @code
        Mat m = (Mat_<int>(3,3) <<
                    1,2,3,
                    4,5,6,
                    7,8,9);
        Mat d0 = m.diag(0);
        Mat d1 = m.diag(1);
        Mat d_1 = m.diag(-1);
    @endcode
    The resulting matrices are
    @code
     d0 =
       [1;
        5;
        9]
     d1 =
       [2;
        6]
     d_1 =
       [4;
        8]
    @endcode
     */
    Mat diag(int d=0) const;

    /** @brief creates a diagonal matrix

    The method creates a square diagonal matrix from specified main diagonal.
    @param d One-dimensional matrix that represents the main diagonal.
     */
    static Mat diag(const Mat& d);

    /** @brief Creates a full copy of the array and the underlying data.

    The method creates a full copy of the array. The original step[] is not taken into account. So, the
    array copy is a continuous array occupying total()*elemSize() bytes.
     */
    Mat clone() const;

    /** @brief Copies the matrix to another one.

    The method copies the matrix data to another matrix. Before copying the data, the method invokes :
    @code
        m.create(this->size(), this->type());
    @endcode
    so that the destination matrix is reallocated if needed. While m.copyTo(m); works flawlessly, the
    function does not handle the case of a partial overlap between the source and the destination
    matrices.

    When the operation mask is specified, if the Mat::create call shown above reallocates the matrix,
    the newly allocated matrix is initialized with all zeros before copying the data.
    @param m Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
     */
    void copyTo( OutputArray m ) const;

    /** @overload
    @param m Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
    @param mask Operation mask. Its non-zero elements indicate which matrix elements need to be copied.
    The mask has to be of type CV_8U and can have 1 or multiple channels.
    */
    void copyTo( OutputArray m, InputArray mask ) const;

    /** @brief Converts an array to another data type with optional scaling.

    The method converts source pixel values to the target data type. saturate_cast\<\> is applied at
    the end to avoid possible overflows:

    \f[m(x,y) = saturate \_ cast<rType>( \alpha (*this)(x,y) +  \beta )\f]
    @param m output matrix; if it does not have a proper size or type before the operation, it is
    reallocated.
    @param rtype desired output matrix type or, rather, the depth since the number of channels are the
    same as the input has; if rtype is negative, the output matrix will have the same type as the input.
    @param alpha optional scale factor.
    @param beta optional delta added to the scaled values.
     */
    void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;

    /** @brief Provides a functional form of convertTo.

    This is an internally used method called by the @ref MatrixExpressions engine.
    @param m Destination array.
    @param type Desired destination array depth (or -1 if it should be the same as the source type).
     */
    void assignTo( Mat& m, int type=-1 ) const;

    /** @brief Sets all or some of the array elements to the specified value.
    @param s Assigned scalar converted to the actual array type.
    */
    Mat& operator = (const Scalar& s);

    /** @brief Sets all or some of the array elements to the specified value.

    This is an advanced variant of the Mat::operator=(const Scalar& s) operator.
    @param value Assigned scalar converted to the actual array type.
    @param mask Operation mask of the same size as \*this.
     */
    Mat& setTo(InputArray value, InputArray mask=noArray());

    /** @brief Changes the shape and/or the number of channels of a 2D matrix without copying the data.

    The method makes a new matrix header for \*this elements. The new matrix may have a different size
    and/or different number of channels. Any combination is possible if:
    -   No extra elements are included into the new matrix and no elements are excluded. Consequently,
        the product rows\*cols\*channels() must stay the same after the transformation.
    -   No data is copied. That is, this is an O(1) operation. Consequently, if you change the number of
        rows, or the operation changes the indices of elements row in some other way, the matrix must be
        continuous. See Mat::isContinuous .

    For example, if there is a set of 3D points stored as an STL vector, and you want to represent the
    points as a 3xN matrix, do the following:
    @code
        std::vector<Point3f> vec;
        ...
        Mat pointMat = Mat(vec). // convert vector to Mat, O(1) operation
                          reshape(1). // make Nx3 1-channel matrix out of Nx1 3-channel.
                                      // Also, an O(1) operation
                             t(); // finally, transpose the Nx3 matrix.
                                  // This involves copying all the elements
    @endcode
    @param cn New number of channels. If the parameter is 0, the number of channels remains the same.
    @param rows New number of rows. If the parameter is 0, the number of rows remains the same.
     */
    Mat reshape(int cn, int rows=0) const;

    /** @overload */
    Mat reshape(int cn, int newndims, const int* newsz) const;

    /** @overload */
    Mat reshape(int cn, const std::vector<int>& newshape) const;

    /** @brief Transposes a matrix.

    The method performs matrix transposition by means of matrix expressions. It does not perform the
    actual transposition but returns a temporary matrix transposition object that can be further used as
    a part of more complex matrix expressions or can be assigned to a matrix:
    @code
        Mat A1 = A + Mat::eye(A.size(), A.type())*lambda;
        Mat C = A1.t()*A1; // compute (A + lambda*I)^t * (A + lamda*I)
    @endcode
     */
    MatExpr t() const;

    /** @brief Inverses a matrix.

    The method performs a matrix inversion by means of matrix expressions. This means that a temporary
    matrix inversion object is returned by the method and can be used further as a part of more complex
    matrix expressions or can be assigned to a matrix.
    @param method Matrix inversion method. One of cv::DecompTypes
     */
    MatExpr inv(int method=DECOMP_LU) const;

    /** @brief Performs an element-wise multiplication or division of the two matrices.

    The method returns a temporary object encoding per-element array multiplication, with optional
    scale. Note that this is not a matrix multiplication that corresponds to a simpler "\*" operator.

    Example:
    @code
        Mat C = A.mul(5/B); // equivalent to divide(A, B, C, 5)
    @endcode
    @param m Another array of the same type and the same size as \*this, or a matrix expression.
    @param scale Optional scale factor.
     */
    MatExpr mul(InputArray m, double scale=1) const;

    /** @brief Computes a cross-product of two 3-element vectors.

    The method computes a cross-product of two 3-element vectors. The vectors must be 3-element
    floating-point vectors of the same shape and size. The result is another 3-element vector of the
    same shape and type as operands.
    @param m Another cross-product operand.
     */
    Mat cross(InputArray m) const;

    /** @brief Computes a dot-product of two vectors.

    The method computes a dot-product of two matrices. If the matrices are not single-column or
    single-row vectors, the top-to-bottom left-to-right scan ordering is used to treat them as 1D
    vectors. The vectors must have the same size and type. If the matrices have more than one channel,
    the dot products from all the channels are summed together.
    @param m another dot-product operand.
     */
    double dot(InputArray m) const;

    /** @brief Returns a zero array of the specified size and type.

    The method returns a Matlab-style zero array initializer. It can be used to quickly form a constant
    array as a function parameter, part of a matrix expression, or as a matrix initializer. :
    @code
        Mat A;
        A = Mat::zeros(3, 3, CV_32F);
    @endcode
    In the example above, a new matrix is allocated only if A is not a 3x3 floating-point matrix.
    Otherwise, the existing matrix A is filled with zeros.
    @param rows Number of rows.
    @param cols Number of columns.
    @param type Created matrix type.
     */
    static MatExpr zeros(int rows, int cols, int type);

    /** @overload
    @param size Alternative to the matrix size specification Size(cols, rows) .
    @param type Created matrix type.
    */
    static MatExpr zeros(Size size, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sz Array of integers specifying the array shape.
    @param type Created matrix type.
    */
    static MatExpr zeros(int ndims, const int* sz, int type);

    /** @brief Returns an array of all 1's of the specified size and type.

    The method returns a Matlab-style 1's array initializer, similarly to Mat::zeros. Note that using
    this method you can initialize an array with an arbitrary value, using the following Matlab idiom:
    @code
        Mat A = Mat::ones(100, 100, CV_8U)*3; // make 100x100 matrix filled with 3.
    @endcode
    The above operation does not form a 100x100 matrix of 1's and then multiply it by 3. Instead, it
    just remembers the scale factor (3 in this case) and use it when actually invoking the matrix
    initializer.
    @param rows Number of rows.
    @param cols Number of columns.
    @param type Created matrix type.
     */
    static MatExpr ones(int rows, int cols, int type);

    /** @overload
    @param size Alternative to the matrix size specification Size(cols, rows) .
    @param type Created matrix type.
    */
    static MatExpr ones(Size size, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sz Array of integers specifying the array shape.
    @param type Created matrix type.
    */
    static MatExpr ones(int ndims, const int* sz, int type);

    /** @brief Returns an identity matrix of the specified size and type.

    The method returns a Matlab-style identity matrix initializer, similarly to Mat::zeros. Similarly to
    Mat::ones, you can use a scale operation to create a scaled identity matrix efficiently:
    @code
        // make a 4x4 diagonal matrix with 0.1's on the diagonal.
        Mat A = Mat::eye(4, 4, CV_32F)*0.1;
    @endcode
    @param rows Number of rows.
    @param cols Number of columns.
    @param type Created matrix type.
     */
    static MatExpr eye(int rows, int cols, int type);

    /** @overload
    @param size Alternative matrix size specification as Size(cols, rows) .
    @param type Created matrix type.
    */
    static MatExpr eye(Size size, int type);

    /** @brief Allocates new array data if needed.

    This is one of the key Mat methods. Most new-style OpenCV functions and methods that produce arrays
    call this method for each output array. The method uses the following algorithm:

    -# If the current array shape and the type match the new ones, return immediately. Otherwise,
       de-reference the previous data by calling Mat::release.
    -# Initialize the new header.
    -# Allocate the new data of total()\*elemSize() bytes.
    -# Allocate the new, associated with the data, reference counter and set it to 1.

    Such a scheme makes the memory management robust and efficient at the same time and helps avoid
    extra typing for you. This means that usually there is no need to explicitly allocate output arrays.
    That is, instead of writing:
    @code
        Mat color;
        ...
        Mat gray(color.rows, color.cols, color.depth());
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    you can simply write:
    @code
        Mat color;
        ...
        Mat gray;
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    because cvtColor, as well as the most of OpenCV functions, calls Mat::create() for the output array
    internally.
    @param rows New number of rows.
    @param cols New number of columns.
    @param type New matrix type.
     */
    void create(int rows, int cols, int type);

    /** @overload
    @param size Alternative new matrix size specification: Size(cols, rows)
    @param type New matrix type.
    */
    void create(Size size, int type);

    /** @overload
    @param ndims New array dimensionality.
    @param sizes Array of integers specifying a new array shape.
    @param type New matrix type.
    */
    void create(int ndims, const int* sizes, int type);

    /** @overload
    @param sizes Array of integers specifying a new array shape.
    @param type New matrix type.
    */
    void create(const std::vector<int>& sizes, int type);

    /** @brief Increments the reference counter.

    The method increments the reference counter associated with the matrix data. If the matrix header
    points to an external data set (see Mat::Mat ), the reference counter is NULL, and the method has no
    effect in this case. Normally, to avoid memory leaks, the method should not be called explicitly. It
    is called implicitly by the matrix assignment operator. The reference counter increment is an atomic
    operation on the platforms that support it. Thus, it is safe to operate on the same matrices
    asynchronously in different threads.
     */
    void addref();

    /** @brief Decrements the reference counter and deallocates the matrix if needed.

    The method decrements the reference counter associated with the matrix data. When the reference
    counter reaches 0, the matrix data is deallocated and the data and the reference counter pointers
    are set to NULL's. If the matrix header points to an external data set (see Mat::Mat ), the
    reference counter is NULL, and the method has no effect in this case.

    This method can be called manually to force the matrix data deallocation. But since this method is
    automatically called in the destructor, or by any other method that changes the data pointer, it is
    usually not needed. The reference counter decrement and check for 0 is an atomic operation on the
    platforms that support it. Thus, it is safe to operate on the same matrices asynchronously in
    different threads.
     */
    void release();

    //! internal use function, consider to use 'release' method instead; deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const Mat& m);

    /** @brief Reserves space for the certain number of rows.

    The method reserves space for sz rows. If the matrix already has enough space to store sz rows,
    nothing happens. If the matrix is reallocated, the first Mat::rows rows are preserved. The method
    emulates the corresponding method of the STL vector class.
    @param sz Number of rows.
     */
    void reserve(size_t sz);

    /** @brief Reserves space for the certain number of bytes.

    The method reserves space for sz bytes. If the matrix already has enough space to store sz bytes,
    nothing happens. If matrix has to be reallocated its previous content could be lost.
    @param sz Number of bytes.
    */
    void reserveBuffer(size_t sz);

    /** @brief Changes the number of matrix rows.

    The methods change the number of matrix rows. If the matrix is reallocated, the first
    min(Mat::rows, sz) rows are preserved. The methods emulate the corresponding methods of the STL
    vector class.
    @param sz New number of rows.
     */
    void resize(size_t sz);

    /** @overload
    @param sz New number of rows.
    @param s Value assigned to the newly added elements.
     */
    void resize(size_t sz, const Scalar& s);

    //! internal function
    void push_back_(const void* elem);

    /** @brief Adds elements to the bottom of the matrix.

    The methods add one or more elements to the bottom of the matrix. They emulate the corresponding
    method of the STL vector class. When elem is Mat , its type and the number of columns must be the
    same as in the container matrix.
    @param elem Added element(s).
     */
    template<typename _Tp> void push_back(const _Tp& elem);

    /** @overload
    @param elem Added element(s).
    */
    template<typename _Tp> void push_back(const Mat_<_Tp>& elem);

    /** @overload
    @param m Added line(s).
    */
    void push_back(const Mat& m);

    /** @brief Removes elements from the bottom of the matrix.

    The method removes one or more rows from the bottom of the matrix.
    @param nelems Number of removed rows. If it is greater than the total number of rows, an exception
    is thrown.
     */
    void pop_back(size_t nelems=1);

    /** @brief Locates the matrix header within a parent matrix.

    After you extracted a submatrix from a matrix using Mat::row, Mat::col, Mat::rowRange,
    Mat::colRange, and others, the resultant submatrix points just to the part of the original big
    matrix. However, each submatrix contains information (represented by datastart and dataend
    fields) that helps reconstruct the original matrix size and the position of the extracted
    submatrix within the original matrix. The method locateROI does exactly that.
    @param wholeSize Output parameter that contains the size of the whole matrix containing *this*
    as a part.
    @param ofs Output parameter that contains an offset of *this* inside the whole matrix.
     */
    void locateROI( Size& wholeSize, Point& ofs ) const;

    /** @brief Adjusts a submatrix size and position within the parent matrix.

    The method is complimentary to Mat::locateROI . The typical use of these functions is to determine
    the submatrix position within the parent matrix and then shift the position somehow. Typically, it
    can be required for filtering operations when pixels outside of the ROI should be taken into
    account. When all the method parameters are positive, the ROI needs to grow in all directions by the
    specified amount, for example:
    @code
        A.adjustROI(2, 2, 2, 2);
    @endcode
    In this example, the matrix size is increased by 4 elements in each direction. The matrix is shifted
    by 2 elements to the left and 2 elements up, which brings in all the necessary pixels for the
    filtering with the 5x5 kernel.

    adjustROI forces the adjusted ROI to be inside of the parent matrix that is boundaries of the
    adjusted ROI are constrained by boundaries of the parent matrix. For example, if the submatrix A is
    located in the first row of a parent matrix and you called A.adjustROI(2, 2, 2, 2) then A will not
    be increased in the upward direction.

    The function is used internally by the OpenCV filtering functions, like filter2D , morphological
    operations, and so on.
    @param dtop Shift of the top submatrix boundary upwards.
    @param dbottom Shift of the bottom submatrix boundary downwards.
    @param dleft Shift of the left submatrix boundary to the left.
    @param dright Shift of the right submatrix boundary to the right.
    @sa copyMakeBorder
     */
    Mat& adjustROI( int dtop, int dbottom, int dleft, int dright );

    /** @brief Extracts a rectangular submatrix.

    The operators make a new header for the specified sub-array of \*this . They are the most
    generalized forms of Mat::row, Mat::col, Mat::rowRange, and Mat::colRange . For example,
    `A(Range(0, 10), Range::all())` is equivalent to `A.rowRange(0, 10)`. Similarly to all of the above,
    the operators are O(1) operations, that is, no matrix data is copied.
    @param rowRange Start and end row of the extracted submatrix. The upper boundary is not included. To
    select all the rows, use Range::all().
    @param colRange Start and end column of the extracted submatrix. The upper boundary is not included.
    To select all the columns, use Range::all().
     */
    Mat operator()( Range rowRange, Range colRange ) const;

    /** @overload
    @param roi Extracted submatrix specified as a rectangle.
    */
    Mat operator()( const Rect& roi ) const;

    /** @overload
    @param ranges Array of selected ranges along each array dimension.
    */
    Mat operator()( const Range* ranges ) const;

    /** @overload
    @param ranges Array of selected ranges along each array dimension.
    */
    Mat operator()(const std::vector<Range>& ranges) const;

    // //! converts header to CvMat; no data is copied
    // operator CvMat() const;
    // //! converts header to CvMatND; no data is copied
    // operator CvMatND() const;
    // //! converts header to IplImage; no data is copied
    // operator IplImage() const;

    template<typename _Tp> operator std::vector<_Tp>() const;
    template<typename _Tp, int n> operator Vec<_Tp, n>() const;
    template<typename _Tp, int m, int n> operator Matx<_Tp, m, n>() const;

#ifdef CV_CXX_STD_ARRAY
    template<typename _Tp, std::size_t _Nm> operator std::array<_Tp, _Nm>() const;
#endif

    /** @brief Reports whether the matrix is continuous or not.

    The method returns true if the matrix elements are stored continuously without gaps at the end of
    each row. Otherwise, it returns false. Obviously, 1x1 or 1xN matrices are always continuous.
    Matrices created with Mat::create are always continuous. But if you extract a part of the matrix
    using Mat::col, Mat::diag, and so on, or constructed a matrix header for externally allocated data,
    such matrices may no longer have this property.

    The continuity flag is stored as a bit in the Mat::flags field and is computed automatically when
    you construct a matrix header. Thus, the continuity check is a very fast operation, though
    theoretically it could be done as follows:
    @code
        // alternative implementation of Mat::isContinuous()
        bool myCheckMatContinuity(const Mat& m)
        {
            //return (m.flags & Mat::CONTINUOUS_FLAG) != 0;
            return m.rows == 1 || m.step == m.cols*m.elemSize();
        }
    @endcode
    The method is used in quite a few of OpenCV functions. The point is that element-wise operations
    (such as arithmetic and logical operations, math functions, alpha blending, color space
    transformations, and others) do not depend on the image geometry. Thus, if all the input and output
    arrays are continuous, the functions can process them as very long single-row vectors. The example
    below illustrates how an alpha-blending function can be implemented:
    @code
        template<typename T>
        void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
        {
            const float alpha_scale = (float)std::numeric_limits<T>::max(),
                        inv_scale = 1.f/alpha_scale;

            CV_Assert( src1.type() == src2.type() &&
                       src1.type() == CV_MAKETYPE(DataType<T>::depth, 4) &&
                       src1.size() == src2.size());
            Size size = src1.size();
            dst.create(size, src1.type());

            // here is the idiom: check the arrays for continuity and,
            // if this is the case,
            // treat the arrays as 1D vectors
            if( src1.isContinuous() && src2.isContinuous() && dst.isContinuous() )
            {
                size.width *= size.height;
                size.height = 1;
            }
            size.width *= 4;

            for( int i = 0; i < size.height; i++ )
            {
                // when the arrays are continuous,
                // the outer loop is executed only once
                const T* ptr1 = src1.ptr<T>(i);
                const T* ptr2 = src2.ptr<T>(i);
                T* dptr = dst.ptr<T>(i);

                for( int j = 0; j < size.width; j += 4 )
                {
                    float alpha = ptr1[j+3]*inv_scale, beta = ptr2[j+3]*inv_scale;
                    dptr[j] = saturate_cast<T>(ptr1[j]*alpha + ptr2[j]*beta);
                    dptr[j+1] = saturate_cast<T>(ptr1[j+1]*alpha + ptr2[j+1]*beta);
                    dptr[j+2] = saturate_cast<T>(ptr1[j+2]*alpha + ptr2[j+2]*beta);
                    dptr[j+3] = saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale);
                }
            }
        }
    @endcode
    This approach, while being very simple, can boost the performance of a simple element-operation by
    10-20 percents, especially if the image is rather small and the operation is quite simple.

    Another OpenCV idiom in this function, a call of Mat::create for the destination array, that
    allocates the destination array unless it already has the proper size and type. And while the newly
    allocated arrays are always continuous, you still need to check the destination array because
    Mat::create does not always allocate a new matrix.
     */
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    /** @brief Returns the matrix element size in bytes.

    The method returns the matrix element size in bytes. For example, if the matrix type is CV_16SC3 ,
    the method returns 3\*sizeof(short) or 6.
     */
    size_t elemSize() const;

    /** @brief Returns the size of each matrix element channel in bytes.

    The method returns the matrix element channel size in bytes, that is, it ignores the number of
    channels. For example, if the matrix type is CV_16SC3 , the method returns sizeof(short) or 2.
     */
    size_t elemSize1() const;

    /** @brief Returns the type of a matrix element.

    The method returns a matrix element type. This is an identifier compatible with the CvMat type
    system, like CV_16SC3 or 16-bit signed 3-channel array, and so on.
     */
    int type() const;

    /** @brief Returns the depth of a matrix element.

    The method returns the identifier of the matrix element depth (the type of each individual channel).
    For example, for a 16-bit signed element array, the method returns CV_16S . A complete list of
    matrix types contains the following values:
    -   CV_8U - 8-bit unsigned integers ( 0..255 )
    -   CV_8S - 8-bit signed integers ( -128..127 )
    -   CV_16U - 16-bit unsigned integers ( 0..65535 )
    -   CV_16S - 16-bit signed integers ( -32768..32767 )
    -   CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
    -   CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    -   CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
     */
    int depth() const;

    /** @brief Returns the number of matrix channels.

    The method returns the number of matrix channels.
     */
    int channels() const;

    /** @brief Returns a normalized step.

    The method returns a matrix step divided by Mat::elemSize1() . It can be useful to quickly access an
    arbitrary matrix element.
     */
    size_t step1(int i=0) const;

    /** @brief Returns true if the array has no elements.

    The method returns true if Mat::total() is 0 or if Mat::data is NULL. Because of pop_back() and
    resize() methods `M.total() == 0` does not imply that `M.data == NULL`.
     */
    bool empty() const;

    /** @brief Returns the total number of array elements.

    The method returns the number of array elements (a number of pixels if the array represents an
    image).
     */
    size_t total() const;

    /** @brief Returns the total number of array elements.

     The method returns the number of elements within a certain sub-array slice with startDim <= dim < endDim
     */
    size_t total(int startDim, int endDim=INT_MAX) const;

    //! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
    int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

    /** @brief Returns a pointer to the specified matrix row.

    The methods return `uchar*` or typed pointer to the specified matrix row. See the sample in
    Mat::isContinuous to know how to use these methods.
    @param i0 A 0-based row index.
     */
    uchar* ptr(int i0=0);
    /** @overload */
    const uchar* ptr(int i0=0) const;

    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    uchar* ptr(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    const uchar* ptr(int row, int col) const;

    /** @overload */
    uchar* ptr(int i0, int i1, int i2);
    /** @overload */
    const uchar* ptr(int i0, int i1, int i2) const;

    /** @overload */
    uchar* ptr(const int* idx);
    /** @overload */
    const uchar* ptr(const int* idx) const;
    /** @overload */
    template<int n> uchar* ptr(const Vec<int, n>& idx);
    /** @overload */
    template<int n> const uchar* ptr(const Vec<int, n>& idx) const;

    /** @overload */
    template<typename _Tp> _Tp* ptr(int i0=0);
    /** @overload */
    template<typename _Tp> const _Tp* ptr(int i0=0) const;
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> _Tp* ptr(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> const _Tp* ptr(int row, int col) const;
    /** @overload */
    template<typename _Tp> _Tp* ptr(int i0, int i1, int i2);
    /** @overload */
    template<typename _Tp> const _Tp* ptr(int i0, int i1, int i2) const;
    /** @overload */
    template<typename _Tp> _Tp* ptr(const int* idx);
    /** @overload */
    template<typename _Tp> const _Tp* ptr(const int* idx) const;
    /** @overload */
    template<typename _Tp, int n> _Tp* ptr(const Vec<int, n>& idx);
    /** @overload */
    template<typename _Tp, int n> const _Tp* ptr(const Vec<int, n>& idx) const;

    /** @brief Returns a reference to the specified array element.

    The template methods return a reference to the specified array element. For the sake of higher
    performance, the index range checks are only performed in the Debug configuration.

    Note that the variants with a single index (i) can be used to access elements of single-row or
    single-column 2-dimensional arrays. That is, if, for example, A is a 1 x N floating-point matrix and
    B is an M x 1 integer matrix, you can simply write `A.at<float>(k+4)` and `B.at<int>(2*i+1)`
    instead of `A.at<float>(0,k+4)` and `B.at<int>(2*i+1,0)`, respectively.

    The example below initializes a Hilbert matrix:
    @code
        Mat H(100, 100, CV_64F);
        for(int i = 0; i < H.rows; i++)
            for(int j = 0; j < H.cols; j++)
                H.at<double>(i,j)=1./(i+j+1);
    @endcode

    Keep in mind that the size identifier used in the at operator cannot be chosen at random. It depends
    on the image from which you are trying to retrieve the data. The table below gives a better insight in this:
     - If matrix is of type `CV_8U` then use `Mat.at<uchar>(y,x)`.
     - If matrix is of type `CV_8S` then use `Mat.at<schar>(y,x)`.
     - If matrix is of type `CV_16U` then use `Mat.at<ushort>(y,x)`.
     - If matrix is of type `CV_16S` then use `Mat.at<short>(y,x)`.
     - If matrix is of type `CV_32S`  then use `Mat.at<int>(y,x)`.
     - If matrix is of type `CV_32F`  then use `Mat.at<float>(y,x)`.
     - If matrix is of type `CV_64F` then use `Mat.at<double>(y,x)`.

    @param i0 Index along the dimension 0
     */
    template<typename _Tp> _Tp& at(int i0=0);
    /** @overload
    @param i0 Index along the dimension 0
    */
    template<typename _Tp> const _Tp& at(int i0=0) const;
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> _Tp& at(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template<typename _Tp> const _Tp& at(int row, int col) const;

    /** @overload
    @param i0 Index along the dimension 0
    @param i1 Index along the dimension 1
    @param i2 Index along the dimension 2
    */
    template<typename _Tp> _Tp& at(int i0, int i1, int i2);
    /** @overload
    @param i0 Index along the dimension 0
    @param i1 Index along the dimension 1
    @param i2 Index along the dimension 2
    */
    template<typename _Tp> const _Tp& at(int i0, int i1, int i2) const;

    /** @overload
    @param idx Array of Mat::dims indices.
    */
    template<typename _Tp> _Tp& at(const int* idx);
    /** @overload
    @param idx Array of Mat::dims indices.
    */
    template<typename _Tp> const _Tp& at(const int* idx) const;

    /** @overload */
    template<typename _Tp, int n> _Tp& at(const Vec<int, n>& idx);
    /** @overload */
    template<typename _Tp, int n> const _Tp& at(const Vec<int, n>& idx) const;

    /** @overload
    special versions for 2D arrays (especially convenient for referencing image pixels)
    @param pt Element position specified as Point(j,i) .
    */
    template<typename _Tp> _Tp& at(Point pt);
    /** @overload
    special versions for 2D arrays (especially convenient for referencing image pixels)
    @param pt Element position specified as Point(j,i) .
    */
    template<typename _Tp> const _Tp& at(Point pt) const;

    /** @brief Returns the matrix iterator and sets it to the first matrix element.

    The methods return the matrix read-only or read-write iterators. The use of matrix iterators is very
    similar to the use of bi-directional STL iterators. In the example below, the alpha blending
    function is rewritten using the matrix iterators:
    @code
        template<typename T>
        void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
        {
            typedef Vec<T, 4> VT;

            const float alpha_scale = (float)std::numeric_limits<T>::max(),
                        inv_scale = 1.f/alpha_scale;

            CV_Assert( src1.type() == src2.type() &&
                       src1.type() == DataType<VT>::type &&
                       src1.size() == src2.size());
            Size size = src1.size();
            dst.create(size, src1.type());

            MatConstIterator_<VT> it1 = src1.begin<VT>(), it1_end = src1.end<VT>();
            MatConstIterator_<VT> it2 = src2.begin<VT>();
            MatIterator_<VT> dst_it = dst.begin<VT>();

            for( ; it1 != it1_end; ++it1, ++it2, ++dst_it )
            {
                VT pix1 = *it1, pix2 = *it2;
                float alpha = pix1[3]*inv_scale, beta = pix2[3]*inv_scale;
                *dst_it = VT(saturate_cast<T>(pix1[0]*alpha + pix2[0]*beta),
                             saturate_cast<T>(pix1[1]*alpha + pix2[1]*beta),
                             saturate_cast<T>(pix1[2]*alpha + pix2[2]*beta),
                             saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale));
            }
        }
    @endcode
     */
    template<typename _Tp> MatIterator_<_Tp> begin();
    template<typename _Tp> MatConstIterator_<_Tp> begin() const;

    /** @brief Returns the matrix iterator and sets it to the after-last matrix element.

    The methods return the matrix read-only or read-write iterators, set to the point following the last
    matrix element.
     */
    template<typename _Tp> MatIterator_<_Tp> end();
    template<typename _Tp> MatConstIterator_<_Tp> end() const;

    /** @brief Runs the given functor over all matrix elements in parallel.

    The operation passed as argument has to be a function pointer, a function object or a lambda(C++11).

    Example 1. All of the operations below put 0xFF the first channel of all matrix elements:
    @code
        Mat image(1920, 1080, CV_8UC3);
        typedef cv::Point3_<uint8_t> Pixel;

        // first. raw pointer access.
        for (int r = 0; r < image.rows; ++r) {
            Pixel* ptr = image.ptr<Pixel>(r, 0);
            const Pixel* ptr_end = ptr + image.cols;
            for (; ptr != ptr_end; ++ptr) {
                ptr->x = 255;
            }
        }

        // Using MatIterator. (Simple but there are a Iterator's overhead)
        for (Pixel &p : cv::Mat_<Pixel>(image)) {
            p.x = 255;
        }

        // Parallel execution with function object.
        struct Operator {
            void operator ()(Pixel &pixel, const int * position) {
                pixel.x = 255;
            }
        };
        image.forEach<Pixel>(Operator());

        // Parallel execution using C++11 lambda.
        image.forEach<Pixel>([](Pixel &p, const int * position) -> void {
            p.x = 255;
        });
    @endcode
    Example 2. Using the pixel's position:
    @code
        // Creating 3D matrix (255 x 255 x 255) typed uint8_t
        // and initialize all elements by the value which equals elements position.
        // i.e. pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).

        int sizes[] = { 255, 255, 255 };
        typedef cv::Point3_<uint8_t> Pixel;

        Mat_<Pixel> image = Mat::zeros(3, sizes, CV_8UC3);

        image.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
            pixel.x = position[0];
            pixel.y = position[1];
            pixel.z = position[2];
        });
    @endcode
     */
    template<typename _Tp, typename Functor> void forEach(const Functor& operation);
    /** @overload */
    template<typename _Tp, typename Functor> void forEach(const Functor& operation) const;

#ifdef CV_CXX_MOVE_SEMANTICS
    Mat(Mat&& m);
    Mat& operator = (Mat&& m);
#endif

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
    enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;
    //! pointer to the data
    uchar* data;

    //! helper fields used in locateROI and adjustROI
    const uchar* datastart;
    const uchar* dataend;
    const uchar* datalimit;

    //! custom allocator
    MatAllocator* allocator;
    //! and the standard allocator
    static MatAllocator* getStdAllocator();
    static MatAllocator* getDefaultAllocator();
    static void setDefaultAllocator(MatAllocator* allocator);

    //! interaction with UMat
    UMatData* u;

    MatSize size;
    MatStep step;

protected:
    template<typename _Tp, typename Functor> void forEach_impl(const Functor& operation);
};


///////////////////////////////// Mat_<_Tp> ////////////////////////////////////

/** @brief Template matrix class derived from Mat

@code{.cpp}
    template<typename _Tp> class Mat_ : public Mat
    {
    public:
        // ... some specific methods
        //         and
        // no new extra fields
    };
@endcode
The class `Mat_<_Tp>` is a *thin* template wrapper on top of the Mat class. It does not have any
extra data fields. Nor this class nor Mat has any virtual methods. Thus, references or pointers to
these two classes can be freely but carefully converted one to another. For example:
@code{.cpp}
    // create a 100x100 8-bit matrix
    Mat M(100,100,CV_8U);
    // this will be compiled fine. no any data conversion will be done.
    Mat_<float>& M1 = (Mat_<float>&)M;
    // the program is likely to crash at the statement below
    M1(99,99) = 1.f;
@endcode
While Mat is sufficient in most cases, Mat_ can be more convenient if you use a lot of element
access operations and if you know matrix type at the compilation time. Note that
`Mat::at(int y,int x)` and `Mat_::operator()(int y,int x)` do absolutely the same
and run at the same speed, but the latter is certainly shorter:
@code{.cpp}
    Mat_<double> M(20,20);
    for(int i = 0; i < M.rows; i++)
        for(int j = 0; j < M.cols; j++)
            M(i,j) = 1./(i+j+1);
    Mat E, V;
    eigen(M,E,V);
    cout << E.at<double>(0,0)/E.at<double>(M.rows-1,0);
@endcode
To use Mat_ for multi-channel images/matrices, pass Vec as a Mat_ parameter:
@code{.cpp}
    // allocate a 320x240 color image and fill it with green (in RGB space)
    Mat_<Vec3b> img(240, 320, Vec3b(0,255,0));
    // now draw a diagonal white line
    for(int i = 0; i < 100; i++)
        img(i,i)=Vec3b(255,255,255);
    // and now scramble the 2nd (red) channel of each pixel
    for(int i = 0; i < img.rows; i++)
        for(int j = 0; j < img.cols; j++)
            img(i,j)[2] ^= (uchar)(i ^ j);
@endcode
Mat_ is fully compatible with C++11 range-based for loop. For example such loop
can be used to safely apply look-up table:
@code{.cpp}
void applyTable(Mat_<uchar>& I, const uchar* const table)
{
    for(auto& pixel : I)
    {
        pixel = table[pixel];
    }
}
@endcode
 */
template<typename _Tp> class Mat_ : public Mat
{
public:
    typedef _Tp value_type;
    typedef typename DataType<_Tp>::channel_type channel_type;
    typedef MatIterator_<_Tp> iterator;
    typedef MatConstIterator_<_Tp> const_iterator;

    //! default constructor
    Mat_();
    //! equivalent to Mat(_rows, _cols, DataType<_Tp>::type)
    Mat_(int _rows, int _cols);
    //! constructor that sets each matrix element to specified value
    Mat_(int _rows, int _cols, const _Tp& value);
    //! equivalent to Mat(_size, DataType<_Tp>::type)
    explicit Mat_(Size _size);
    //! constructor that sets each matrix element to specified value
    Mat_(Size _size, const _Tp& value);
    //! n-dim array constructor
    Mat_(int _ndims, const int* _sizes);
    //! n-dim array constructor that sets each matrix element to specified value
    Mat_(int _ndims, const int* _sizes, const _Tp& value);
    //! copy/conversion contructor. If m is of different type, it's converted
    Mat_(const Mat& m);
    //! copy constructor
    Mat_(const Mat_& m);
    //! constructs a matrix on top of user-allocated data. step is in bytes(!!!), regardless of the type
    Mat_(int _rows, int _cols, _Tp* _data, size_t _step=AUTO_STEP);
    //! constructs n-dim matrix on top of user-allocated data. steps are in bytes(!!!), regardless of the type
    Mat_(int _ndims, const int* _sizes, _Tp* _data, const size_t* _steps=0);
    //! selects a submatrix
    Mat_(const Mat_& m, const Range& rowRange, const Range& colRange=Range::all());
    //! selects a submatrix
    Mat_(const Mat_& m, const Rect& roi);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_& m, const Range* ranges);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_& m, const std::vector<Range>& ranges);
    //! from a matrix expression
    explicit Mat_(const MatExpr& e);
    //! makes a matrix out of Vec, std::vector, Point_ or Point3_. The matrix will have a single column
    explicit Mat_(const std::vector<_Tp>& vec, bool copyData=false);
    template<int n> explicit Mat_(const Vec<typename DataType<_Tp>::channel_type, n>& vec, bool copyData=true);
    template<int m, int n> explicit Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n>& mtx, bool copyData=true);
    explicit Mat_(const Point_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const Point3_<typename DataType<_Tp>::channel_type>& pt, bool copyData=true);
    explicit Mat_(const MatCommaInitializer_<_Tp>& commaInitializer);

#ifdef CV_CXX11
    Mat_(std::initializer_list<_Tp> values);
#endif

#ifdef CV_CXX_STD_ARRAY
    template <std::size_t _Nm> explicit Mat_(const std::array<_Tp, _Nm>& arr, bool copyData=false);
#endif

    Mat_& operator = (const Mat& m);
    Mat_& operator = (const Mat_& m);
    //! set all the elements to s.
    Mat_& operator = (const _Tp& s);
    //! assign a matrix expression
    Mat_& operator = (const MatExpr& e);

    //! iterators; they are smart enough to skip gaps in the end of rows
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    //! template methods for for operation over all matrix elements.
    // the operations take care of skipping gaps in the end of rows (if any)
    template<typename Functor> void forEach(const Functor& operation);
    template<typename Functor> void forEach(const Functor& operation) const;

    //! equivalent to Mat::create(_rows, _cols, DataType<_Tp>::type)
    void create(int _rows, int _cols);
    //! equivalent to Mat::create(_size, DataType<_Tp>::type)
    void create(Size _size);
    //! equivalent to Mat::create(_ndims, _sizes, DatType<_Tp>::type)
    void create(int _ndims, const int* _sizes);
    //! equivalent to Mat::release()
    void release();
    //! cross-product
    Mat_ cross(const Mat_& m) const;
    //! data type conversion
    template<typename T2> operator Mat_<T2>() const;
    //! overridden forms of Mat::row() etc.
    Mat_ row(int y) const;
    Mat_ col(int x) const;
    Mat_ diag(int d=0) const;
    Mat_ clone() const;

    //! overridden forms of Mat::elemSize() etc.
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1(int i=0) const;
    //! returns step()/sizeof(_Tp)
    size_t stepT(int i=0) const;

    //! overridden forms of Mat::zeros() etc. Data type is omitted, of course
    static MatExpr zeros(int rows, int cols);
    static MatExpr zeros(Size size);
    static MatExpr zeros(int _ndims, const int* _sizes);
    static MatExpr ones(int rows, int cols);
    static MatExpr ones(Size size);
    static MatExpr ones(int _ndims, const int* _sizes);
    static MatExpr eye(int rows, int cols);
    static MatExpr eye(Size size);

    //! some more overriden methods
    Mat_& adjustROI( int dtop, int dbottom, int dleft, int dright );
    Mat_ operator()( const Range& rowRange, const Range& colRange ) const;
    Mat_ operator()( const Rect& roi ) const;
    Mat_ operator()( const Range* ranges ) const;
    Mat_ operator()(const std::vector<Range>& ranges) const;

    //! more convenient forms of row and element access operators
    _Tp* operator [](int y);
    const _Tp* operator [](int y) const;

    //! returns reference to the specified element
    _Tp& operator ()(const int* idx);
    //! returns read-only reference to the specified element
    const _Tp& operator ()(const int* idx) const;

    //! returns reference to the specified element
    template<int n> _Tp& operator ()(const Vec<int, n>& idx);
    //! returns read-only reference to the specified element
    template<int n> const _Tp& operator ()(const Vec<int, n>& idx) const;

    //! returns reference to the specified element (1D case)
    _Tp& operator ()(int idx0);
    //! returns read-only reference to the specified element (1D case)
    const _Tp& operator ()(int idx0) const;
    //! returns reference to the specified element (2D case)
    _Tp& operator ()(int row, int col);
    //! returns read-only reference to the specified element (2D case)
    const _Tp& operator ()(int row, int col) const;
    //! returns reference to the specified element (3D case)
    _Tp& operator ()(int idx0, int idx1, int idx2);
    //! returns read-only reference to the specified element (3D case)
    const _Tp& operator ()(int idx0, int idx1, int idx2) const;

    _Tp& operator ()(Point pt);
    const _Tp& operator ()(Point pt) const;

    //! conversion to vector.
    operator std::vector<_Tp>() const;

#ifdef CV_CXX_STD_ARRAY
    //! conversion to array.
    template<std::size_t _Nm> operator std::array<_Tp, _Nm>() const;
#endif

    //! conversion to Vec
    template<int n> operator Vec<typename DataType<_Tp>::channel_type, n>() const;
    //! conversion to Matx
    template<int m, int n> operator Matx<typename DataType<_Tp>::channel_type, m, n>() const;

#ifdef CV_CXX_MOVE_SEMANTICS
    Mat_(Mat_&& m);
    Mat_& operator = (Mat_&& m);

    Mat_(Mat&& m);
    Mat_& operator = (Mat&& m);

    Mat_(MatExpr&& e);
#endif
};

typedef Mat_<uchar> Mat1b;
typedef Mat_<Vec2b> Mat2b;
typedef Mat_<Vec3b> Mat3b;
typedef Mat_<Vec4b> Mat4b;

typedef Mat_<short> Mat1s;
typedef Mat_<Vec2s> Mat2s;
typedef Mat_<Vec3s> Mat3s;
typedef Mat_<Vec4s> Mat4s;

typedef Mat_<ushort> Mat1w;
typedef Mat_<Vec2w> Mat2w;
typedef Mat_<Vec3w> Mat3w;
typedef Mat_<Vec4w> Mat4w;

typedef Mat_<int>   Mat1i;
typedef Mat_<Vec2i> Mat2i;
typedef Mat_<Vec3i> Mat3i;
typedef Mat_<Vec4i> Mat4i;

typedef Mat_<float> Mat1f;
typedef Mat_<Vec2f> Mat2f;
typedef Mat_<Vec3f> Mat3f;
typedef Mat_<Vec4f> Mat4f;

typedef Mat_<double> Mat1d;
typedef Mat_<Vec2d> Mat2d;
typedef Mat_<Vec3d> Mat3d;
typedef Mat_<Vec4d> Mat4d;

/** @todo document */
class CV_EXPORTS UMat
{
public:
    //! default constructor
    UMat(UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constructs 2D matrix of the specified size and type
    // (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
    UMat(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    //! constucts 2D matrix and fills it with the specified value _s.
    UMat(int rows, int cols, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(Size size, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! constructs n-dimensional matrix
    UMat(int ndims, const int* sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    UMat(int ndims, const int* sizes, int type, const Scalar& s, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! copy constructor
    UMat(const UMat& m);

    //! creates a matrix header for a part of the bigger matrix
    UMat(const UMat& m, const Range& rowRange, const Range& colRange=Range::all());
    UMat(const UMat& m, const Rect& roi);
    UMat(const UMat& m, const Range* ranges);
    UMat(const UMat& m, const std::vector<Range>& ranges);
    //! builds matrix from std::vector with or without copying the data
    template<typename _Tp> explicit UMat(const std::vector<_Tp>& vec, bool copyData=false);

    //! builds matrix from cv::Vec; the data is copied by default
    template<typename _Tp, int n> explicit UMat(const Vec<_Tp, n>& vec, bool copyData=true);
    //! builds matrix from cv::Matx; the data is copied by default
    template<typename _Tp, int m, int n> explicit UMat(const Matx<_Tp, m, n>& mtx, bool copyData=true);
    //! builds matrix from a 2D point
    template<typename _Tp> explicit UMat(const Point_<_Tp>& pt, bool copyData=true);
    //! builds matrix from a 3D point
    template<typename _Tp> explicit UMat(const Point3_<_Tp>& pt, bool copyData=true);
    //! builds matrix from comma initializer
    template<typename _Tp> explicit UMat(const MatCommaInitializer_<_Tp>& commaInitializer);

    //! destructor - calls release()
    ~UMat();
    //! assignment operators
    UMat& operator = (const UMat& m);

    Mat getMat(int flags) const;

    //! returns a new matrix header for the specified row
    UMat row(int y) const;
    //! returns a new matrix header for the specified column
    UMat col(int x) const;
    //! ... for the specified row span
    UMat rowRange(int startrow, int endrow) const;
    UMat rowRange(const Range& r) const;
    //! ... for the specified column span
    UMat colRange(int startcol, int endcol) const;
    UMat colRange(const Range& r) const;
    //! ... for the specified diagonal
    //! (d=0 - the main diagonal,
    //!  >0 - a diagonal from the upper half,
    //!  <0 - a diagonal from the lower half)
    UMat diag(int d=0) const;
    //! constructs a square diagonal matrix which main diagonal is vector "d"
    static UMat diag(const UMat& d);

    //! returns deep copy of the matrix, i.e. the data is copied
    UMat clone() const;
    //! copies the matrix content to "m".
    // It calls m.create(this->size(), this->type()).
    void copyTo( OutputArray m ) const;
    //! copies those matrix elements to "m" that are marked with non-zero mask elements.
    void copyTo( OutputArray m, InputArray mask ) const;
    //! converts matrix to another datatype with optional scalng. See cvConvertScale.
    void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;

    void assignTo( UMat& m, int type=-1 ) const;

    //! sets every matrix element to s
    UMat& operator = (const Scalar& s);
    //! sets some of the matrix elements to s, according to the mask
    UMat& setTo(InputArray value, InputArray mask=noArray());
    //! creates alternative matrix header for the same data, with different
    // number of channels and/or different number of rows. see cvReshape.
    UMat reshape(int cn, int rows=0) const;
    UMat reshape(int cn, int newndims, const int* newsz) const;

    //! matrix transposition by means of matrix expressions
    UMat t() const;
    //! matrix inversion by means of matrix expressions
    UMat inv(int method=DECOMP_LU) const;
    //! per-element matrix multiplication by means of matrix expressions
    UMat mul(InputArray m, double scale=1) const;

    //! computes dot-product
    double dot(InputArray m) const;

    //! Matlab-style matrix initialization
    static UMat zeros(int rows, int cols, int type);
    static UMat zeros(Size size, int type);
    static UMat zeros(int ndims, const int* sz, int type);
    static UMat ones(int rows, int cols, int type);
    static UMat ones(Size size, int type);
    static UMat ones(int ndims, const int* sz, int type);
    static UMat eye(int rows, int cols, int type);
    static UMat eye(Size size, int type);

    //! allocates new matrix data unless the matrix already has specified size and type.
    // previous data is unreferenced if needed.
    void create(int rows, int cols, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(Size size, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(int ndims, const int* sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);
    void create(const std::vector<int>& sizes, int type, UMatUsageFlags usageFlags = USAGE_DEFAULT);

    //! increases the reference counter; use with care to avoid memleaks
    void addref();
    //! decreases reference counter;
    // deallocates the data when reference counter reaches 0.
    void release();

    //! deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const UMat& m);

    //! locates matrix header within a parent matrix. See below
    void locateROI( Size& wholeSize, Point& ofs ) const;
    //! moves/resizes the current matrix ROI inside the parent matrix.
    UMat& adjustROI( int dtop, int dbottom, int dleft, int dright );
    //! extracts a rectangular sub-matrix
    // (this is a generalized form of row, rowRange etc.)
    UMat operator()( Range rowRange, Range colRange ) const;
    UMat operator()( const Rect& roi ) const;
    UMat operator()( const Range* ranges ) const;
    UMat operator()(const std::vector<Range>& ranges) const;

    //! returns true iff the matrix data is continuous
    // (i.e. when there are no gaps between successive rows).
    // similar to CV_IS_MAT_CONT(cvmat->type)
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    //! returns element size in bytes,
    // similar to CV_ELEM_SIZE(cvmat->type)
    size_t elemSize() const;
    //! returns the size of element channel in bytes.
    size_t elemSize1() const;
    //! returns element type, similar to CV_MAT_TYPE(cvmat->type)
    int type() const;
    //! returns element type, similar to CV_MAT_DEPTH(cvmat->type)
    int depth() const;
    //! returns element type, similar to CV_MAT_CN(cvmat->type)
    int channels() const;
    //! returns step/elemSize1()
    size_t step1(int i=0) const;
    //! returns true if matrix data is NULL
    bool empty() const;
    //! returns the total number of matrix elements
    size_t total() const;

    //! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
    int checkVector(int elemChannels, int depth=-1, bool requireContinuous=true) const;

#ifdef CV_CXX_MOVE_SEMANTICS
    UMat(UMat&& m);
    UMat& operator = (UMat&& m);
#endif

    /*! Returns the OpenCL buffer handle on which UMat operates on.
        The UMat instance should be kept alive during the use of the handle to prevent the buffer to be
        returned to the OpenCV buffer pool.
     */
    void* handle(int accessFlags) const;
    void ndoffset(size_t* ofs) const;

    enum { MAGIC_VAL  = 0x42FF0000, AUTO_STEP = 0, CONTINUOUS_FLAG = CV_MAT_CONT_FLAG, SUBMATRIX_FLAG = CV_SUBMAT_FLAG };
    enum { MAGIC_MASK = 0xFFFF0000, TYPE_MASK = 0x00000FFF, DEPTH_MASK = 7 };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;

    //! custom allocator
    MatAllocator* allocator;
    UMatUsageFlags usageFlags; // usage flags for allocator
    //! and the standard allocator
    static MatAllocator* getStdAllocator();

    // black-box container of UMat data
    UMatData* u;

    // offset of the submatrix (or 0)
    size_t offset;

    MatSize size;
    MatStep step;

protected:
};


/////////////////////////// multi-dimensional sparse matrix //////////////////////////

/** @brief The class SparseMat represents multi-dimensional sparse numerical arrays.

Such a sparse array can store elements of any type that Mat can store. *Sparse* means that only
non-zero elements are stored (though, as a result of operations on a sparse matrix, some of its
stored elements can actually become 0. It is up to you to detect such elements and delete them
using SparseMat::erase ). The non-zero elements are stored in a hash table that grows when it is
filled so that the search time is O(1) in average (regardless of whether element is there or not).
Elements can be accessed using the following methods:
-   Query operations (SparseMat::ptr and the higher-level SparseMat::ref, SparseMat::value and
    SparseMat::find), for example:
    @code
        const int dims = 5;
        int size[5] = {10, 10, 10, 10, 10};
        SparseMat sparse_mat(dims, size, CV_32F);
        for(int i = 0; i < 1000; i++)
        {
            int idx[dims];
            for(int k = 0; k < dims; k++)
                idx[k] = rand() % size[k];
            sparse_mat.ref<float>(idx) += 1.f;
        }
        cout << "nnz = " << sparse_mat.nzcount() << endl;
    @endcode
-   Sparse matrix iterators. They are similar to MatIterator but different from NAryMatIterator.
    That is, the iteration loop is familiar to STL users:
    @code
        // prints elements of a sparse floating-point matrix
        // and the sum of elements.
        SparseMatConstIterator_<float>
            it = sparse_mat.begin<float>(),
            it_end = sparse_mat.end<float>();
        double s = 0;
        int dims = sparse_mat.dims();
        for(; it != it_end; ++it)
        {
            // print element indices and the element value
            const SparseMat::Node* n = it.node();
            printf("(");
            for(int i = 0; i < dims; i++)
                printf("%d%s", n->idx[i], i < dims-1 ? ", " : ")");
            printf(": %g\n", it.value<float>());
            s += *it;
        }
        printf("Element sum is %g\n", s);
    @endcode
    If you run this loop, you will notice that elements are not enumerated in a logical order
    (lexicographical, and so on). They come in the same order as they are stored in the hash table
    (semi-randomly). You may collect pointers to the nodes and sort them to get the proper ordering.
    Note, however, that pointers to the nodes may become invalid when you add more elements to the
    matrix. This may happen due to possible buffer reallocation.
-   Combination of the above 2 methods when you need to process 2 or more sparse matrices
    simultaneously. For example, this is how you can compute unnormalized cross-correlation of the 2
    floating-point sparse matrices:
    @code
        double cross_corr(const SparseMat& a, const SparseMat& b)
        {
            const SparseMat *_a = &a, *_b = &b;
            // if b contains less elements than a,
            // it is faster to iterate through b
            if(_a->nzcount() > _b->nzcount())
                std::swap(_a, _b);
            SparseMatConstIterator_<float> it = _a->begin<float>(),
                                           it_end = _a->end<float>();
            double ccorr = 0;
            for(; it != it_end; ++it)
            {
                // take the next element from the first matrix
                float avalue = *it;
                const Node* anode = it.node();
                // and try to find an element with the same index in the second matrix.
                // since the hash value depends only on the element index,
                // reuse the hash value stored in the node
                float bvalue = _b->value<float>(anode->idx,&anode->hashval);
                ccorr += avalue*bvalue;
            }
            return ccorr;
        }
    @endcode
 */
class CV_EXPORTS SparseMat
{
public:
    typedef SparseMatIterator iterator;
    typedef SparseMatConstIterator const_iterator;

    enum { MAGIC_VAL=0x42FD0000, MAX_DIM=32, HASH_SCALE=0x5bd1e995, HASH_BIT=0x80000000 };

    //! the sparse matrix header
    struct CV_EXPORTS Hdr
    {
        Hdr(int _dims, const int* _sizes, int _type);
        void clear();
        int refcount;
        int dims;
        int valueOffset;
        size_t nodeSize;
        size_t nodeCount;
        size_t freeList;
        std::vector<uchar> pool;
        std::vector<size_t> hashtab;
        int size[MAX_DIM];
    };

    //! sparse matrix node - element of a hash table
    struct CV_EXPORTS Node
    {
        //! hash value
        size_t hashval;
        //! index of the next node in the same hash table entry
        size_t next;
        //! index of the matrix element
        int idx[MAX_DIM];
    };

    /** @brief Various SparseMat constructors.
     */
    SparseMat();

    /** @overload
    @param dims Array dimensionality.
    @param _sizes Sparce matrix size on all dementions.
    @param _type Sparse matrix data type.
    */
    SparseMat(int dims, const int* _sizes, int _type);

    /** @overload
    @param m Source matrix for copy constructor. If m is dense matrix (ocvMat) then it will be converted
    to sparse representation.
    */
    SparseMat(const SparseMat& m);

    /** @overload
    @param m Source matrix for copy constructor. If m is dense matrix (ocvMat) then it will be converted
    to sparse representation.
    */
    explicit SparseMat(const Mat& m);

    //! the destructor
    ~SparseMat();

    //! assignment operator. This is O(1) operation, i.e. no data is copied
    SparseMat& operator = (const SparseMat& m);
    //! equivalent to the corresponding constructor
    SparseMat& operator = (const Mat& m);

    //! creates full copy of the matrix
    SparseMat clone() const;

    //! copies all the data to the destination matrix. All the previous content of m is erased
    void copyTo( SparseMat& m ) const;
    //! converts sparse matrix to dense matrix.
    void copyTo( Mat& m ) const;
    //! multiplies all the matrix elements by the specified scale factor alpha and converts the results to the specified data type
    void convertTo( SparseMat& m, int rtype, double alpha=1 ) const;
    //! converts sparse matrix to dense n-dim matrix with optional type conversion and scaling.
    /*!
        @param [out] m - output matrix; if it does not have a proper size or type before the operation,
            it is reallocated
        @param [in] rtype - desired output matrix type or, rather, the depth since the number of channels
            are the same as the input has; if rtype is negative, the output matrix will have the
            same type as the input.
        @param [in] alpha - optional scale factor
        @param [in] beta - optional delta added to the scaled values
    */
    void convertTo( Mat& m, int rtype, double alpha=1, double beta=0 ) const;

    // not used now
    void assignTo( SparseMat& m, int type=-1 ) const;

    //! reallocates sparse matrix.
    /*!
        If the matrix already had the proper size and type,
        it is simply cleared with clear(), otherwise,
        the old matrix is released (using release()) and the new one is allocated.
    */
    void create(int dims, const int* _sizes, int _type);
    //! sets all the sparse matrix elements to 0, which means clearing the hash table.
    void clear();
    //! manually increments the reference counter to the header.
    void addref();
    // decrements the header reference counter. When the counter reaches 0, the header and all the underlying data are deallocated.
    void release();

    //! converts sparse matrix to the old-style representation; all the elements are copied.
    //operator CvSparseMat*() const;
    //! returns the size of each element in bytes (not including the overhead - the space occupied by SparseMat::Node elements)
    size_t elemSize() const;
    //! returns elemSize()/channels()
    size_t elemSize1() const;

    //! returns type of sparse matrix elements
    int type() const;
    //! returns the depth of sparse matrix elements
    int depth() const;
    //! returns the number of channels
    int channels() const;

    //! returns the array of sizes, or NULL if the matrix is not allocated
    const int* size() const;
    //! returns the size of i-th matrix dimension (or 0)
    int size(int i) const;
    //! returns the matrix dimensionality
    int dims() const;
    //! returns the number of non-zero elements (=the number of hash table nodes)
    size_t nzcount() const;

    //! computes the element hash value (1D case)
    size_t hash(int i0) const;
    //! computes the element hash value (2D case)
    size_t hash(int i0, int i1) const;
    //! computes the element hash value (3D case)
    size_t hash(int i0, int i1, int i2) const;
    //! computes the element hash value (nD case)
    size_t hash(const int* idx) const;

    //!@{
    /*!
     specialized variants for 1D, 2D, 3D cases and the generic_type one for n-D case.
     return pointer to the matrix element.
      - if the element is there (it's non-zero), the pointer to it is returned
      - if it's not there and createMissing=false, NULL pointer is returned
      - if it's not there and createMissing=true, then the new element
        is created and initialized with 0. Pointer to it is returned
      - if the optional hashval pointer is not NULL, the element hash value is
        not computed, but *hashval is taken instead.
    */
    //! returns pointer to the specified element (1D case)
    uchar* ptr(int i0, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (2D case)
    uchar* ptr(int i0, int i1, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (3D case)
    uchar* ptr(int i0, int i1, int i2, bool createMissing, size_t* hashval=0);
    //! returns pointer to the specified element (nD case)
    uchar* ptr(const int* idx, bool createMissing, size_t* hashval=0);
    //!@}

    //!@{
    /*!
     return read-write reference to the specified sparse matrix element.

     `ref<_Tp>(i0,...[,hashval])` is equivalent to `*(_Tp*)ptr(i0,...,true[,hashval])`.
     The methods always return a valid reference.
     If the element did not exist, it is created and initialiazed with 0.
    */
    //! returns reference to the specified element (1D case)
    template<typename _Tp> _Tp& ref(int i0, size_t* hashval=0);
    //! returns reference to the specified element (2D case)
    template<typename _Tp> _Tp& ref(int i0, int i1, size_t* hashval=0);
    //! returns reference to the specified element (3D case)
    template<typename _Tp> _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    //! returns reference to the specified element (nD case)
    template<typename _Tp> _Tp& ref(const int* idx, size_t* hashval=0);
    //!@}

    //!@{
    /*!
     return value of the specified sparse matrix element.

     `value<_Tp>(i0,...[,hashval])` is equivalent to
     @code
     { const _Tp* p = find<_Tp>(i0,...[,hashval]); return p ? *p : _Tp(); }
     @endcode

     That is, if the element did not exist, the methods return 0.
     */
    //! returns value of the specified element (1D case)
    template<typename _Tp> _Tp value(int i0, size_t* hashval=0) const;
    //! returns value of the specified element (2D case)
    template<typename _Tp> _Tp value(int i0, int i1, size_t* hashval=0) const;
    //! returns value of the specified element (3D case)
    template<typename _Tp> _Tp value(int i0, int i1, int i2, size_t* hashval=0) const;
    //! returns value of the specified element (nD case)
    template<typename _Tp> _Tp value(const int* idx, size_t* hashval=0) const;
    //!@}

    //!@{
    /*!
     Return pointer to the specified sparse matrix element if it exists

     `find<_Tp>(i0,...[,hashval])` is equivalent to `(_const Tp*)ptr(i0,...false[,hashval])`.

     If the specified element does not exist, the methods return NULL.
    */
    //! returns pointer to the specified element (1D case)
    template<typename _Tp> const _Tp* find(int i0, size_t* hashval=0) const;
    //! returns pointer to the specified element (2D case)
    template<typename _Tp> const _Tp* find(int i0, int i1, size_t* hashval=0) const;
    //! returns pointer to the specified element (3D case)
    template<typename _Tp> const _Tp* find(int i0, int i1, int i2, size_t* hashval=0) const;
    //! returns pointer to the specified element (nD case)
    template<typename _Tp> const _Tp* find(const int* idx, size_t* hashval=0) const;
    //!@}

    //! erases the specified element (2D case)
    void erase(int i0, int i1, size_t* hashval=0);
    //! erases the specified element (3D case)
    void erase(int i0, int i1, int i2, size_t* hashval=0);
    //! erases the specified element (nD case)
    void erase(const int* idx, size_t* hashval=0);

    //!@{
    /*!
       return the sparse matrix iterator pointing to the first sparse matrix element
    */
    //! returns the sparse matrix iterator at the matrix beginning
    SparseMatIterator begin();
    //! returns the sparse matrix iterator at the matrix beginning
    template<typename _Tp> SparseMatIterator_<_Tp> begin();
    //! returns the read-only sparse matrix iterator at the matrix beginning
    SparseMatConstIterator begin() const;
    //! returns the read-only sparse matrix iterator at the matrix beginning
    template<typename _Tp> SparseMatConstIterator_<_Tp> begin() const;
    //!@}
    /*!
       return the sparse matrix iterator pointing to the element following the last sparse matrix element
    */
    //! returns the sparse matrix iterator at the matrix end
    SparseMatIterator end();
    //! returns the read-only sparse matrix iterator at the matrix end
    SparseMatConstIterator end() const;
    //! returns the typed sparse matrix iterator at the matrix end
    template<typename _Tp> SparseMatIterator_<_Tp> end();
    //! returns the typed read-only sparse matrix iterator at the matrix end
    template<typename _Tp> SparseMatConstIterator_<_Tp> end() const;

    //! returns the value stored in the sparse martix node
    template<typename _Tp> _Tp& value(Node* n);
    //! returns the value stored in the sparse martix node
    template<typename _Tp> const _Tp& value(const Node* n) const;

    ////////////// some internal-use methods ///////////////
    Node* node(size_t nidx);
    const Node* node(size_t nidx) const;

    uchar* newNode(const int* idx, size_t hashval);
    void removeNode(size_t hidx, size_t nidx, size_t previdx);
    void resizeHashTab(size_t newsize);

    int flags;
    Hdr* hdr;
};



///////////////////////////////// SparseMat_<_Tp> ////////////////////////////////////

/** @brief Template sparse n-dimensional array class derived from SparseMat

SparseMat_ is a thin wrapper on top of SparseMat created in the same way as Mat_ . It simplifies
notation of some operations:
@code
    int sz[] = {10, 20, 30};
    SparseMat_<double> M(3, sz);
    ...
    M.ref(1, 2, 3) = M(4, 5, 6) + M(7, 8, 9);
@endcode
 */
template<typename _Tp> class SparseMat_ : public SparseMat
{
public:
    typedef SparseMatIterator_<_Tp> iterator;
    typedef SparseMatConstIterator_<_Tp> const_iterator;

    //! the default constructor
    SparseMat_();
    //! the full constructor equivelent to SparseMat(dims, _sizes, DataType<_Tp>::type)
    SparseMat_(int dims, const int* _sizes);
    //! the copy constructor. If DataType<_Tp>.type != m.type(), the m elements are converted
    SparseMat_(const SparseMat& m);
    //! the copy constructor. This is O(1) operation - no data is copied
    SparseMat_(const SparseMat_& m);
    //! converts dense matrix to the sparse form
    SparseMat_(const Mat& m);
    //! converts the old-style sparse matrix to the C++ class. All the elements are copied
    //SparseMat_(const CvSparseMat* m);
    //! the assignment operator. If DataType<_Tp>.type != m.type(), the m elements are converted
    SparseMat_& operator = (const SparseMat& m);
    //! the assignment operator. This is O(1) operation - no data is copied
    SparseMat_& operator = (const SparseMat_& m);
    //! converts dense matrix to the sparse form
    SparseMat_& operator = (const Mat& m);

    //! makes full copy of the matrix. All the elements are duplicated
    SparseMat_ clone() const;
    //! equivalent to cv::SparseMat::create(dims, _sizes, DataType<_Tp>::type)
    void create(int dims, const int* _sizes);
    //! converts sparse matrix to the old-style CvSparseMat. All the elements are copied
    //operator CvSparseMat*() const;

    //! returns type of the matrix elements
    int type() const;
    //! returns depth of the matrix elements
    int depth() const;
    //! returns the number of channels in each matrix element
    int channels() const;

    //! equivalent to SparseMat::ref<_Tp>(i0, hashval)
    _Tp& ref(int i0, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(i0, i1, hashval)
    _Tp& ref(int i0, int i1, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(i0, i1, i2, hashval)
    _Tp& ref(int i0, int i1, int i2, size_t* hashval=0);
    //! equivalent to SparseMat::ref<_Tp>(idx, hashval)
    _Tp& ref(const int* idx, size_t* hashval=0);

    //! equivalent to SparseMat::value<_Tp>(i0, hashval)
    _Tp operator()(int i0, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(i0, i1, hashval)
    _Tp operator()(int i0, int i1, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(i0, i1, i2, hashval)
    _Tp operator()(int i0, int i1, int i2, size_t* hashval=0) const;
    //! equivalent to SparseMat::value<_Tp>(idx, hashval)
    _Tp operator()(const int* idx, size_t* hashval=0) const;

    //! returns sparse matrix iterator pointing to the first sparse matrix element
    SparseMatIterator_<_Tp> begin();
    //! returns read-only sparse matrix iterator pointing to the first sparse matrix element
    SparseMatConstIterator_<_Tp> begin() const;
    //! returns sparse matrix iterator pointing to the element following the last sparse matrix element
    SparseMatIterator_<_Tp> end();
    //! returns read-only sparse matrix iterator pointing to the element following the last sparse matrix element
    SparseMatConstIterator_<_Tp> end() const;
};



////////////////////////////////// MatConstIterator //////////////////////////////////

class CV_EXPORTS MatConstIterator
{
public:
    typedef uchar* value_type;
    typedef ptrdiff_t difference_type;
    typedef const uchar** pointer;
    typedef uchar* reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator(const Mat* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat* _m, const int* _idx);
    //! copy constructor
    MatConstIterator(const MatConstIterator& it);

    //! copy operator
    MatConstIterator& operator = (const MatConstIterator& it);
    //! returns the current matrix element
    const uchar* operator *() const;
    //! returns the i-th matrix element, relative to the current
    const uchar* operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator& operator --();
    //! decrements the iterator
    MatConstIterator operator --(int);
    //! increments the iterator
    MatConstIterator& operator ++();
    //! increments the iterator
    MatConstIterator operator ++(int);
    //! returns the current iterator position
    Point pos() const;
    //! returns the current iterator position
    void pos(int* _idx) const;

    ptrdiff_t lpos() const;
    void seek(ptrdiff_t ofs, bool relative = false);
    void seek(const int* _idx, bool relative = false);

    const Mat* m;
    size_t elemSize;
    const uchar* ptr;
    const uchar* sliceStart;
    const uchar* sliceEnd;
};



////////////////////////////////// MatConstIterator_ /////////////////////////////////

/** @brief Matrix read-only iterator
 */
template<typename _Tp>
class MatConstIterator_ : public MatConstIterator
{
public:
    typedef _Tp value_type;
    typedef ptrdiff_t difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatConstIterator_(const MatConstIterator_& it);

    //! copy operator
    MatConstIterator_& operator = (const MatConstIterator_& it);
    //! returns the current matrix element
    const _Tp& operator *() const;
    //! returns the i-th matrix element, relative to the current
    const _Tp& operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator_& operator --();
    //! decrements the iterator
    MatConstIterator_ operator --(int);
    //! increments the iterator
    MatConstIterator_& operator ++();
    //! increments the iterator
    MatConstIterator_ operator ++(int);
    //! returns the current iterator position
    Point pos() const;
};



//////////////////////////////////// MatIterator_ ////////////////////////////////////

/** @brief Matrix read-write iterator
*/
template<typename _Tp>
class MatIterator_ : public MatConstIterator_<_Tp>
{
public:
    typedef _Tp* pointer;
    typedef _Tp& reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! the default constructor
    MatIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatIterator_(Mat_<_Tp>* _m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, int _row, int _col=0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp>* _m, const int* _idx);
    //! copy constructor
    MatIterator_(const MatIterator_& it);
    //! copy operator
    MatIterator_& operator = (const MatIterator_<_Tp>& it );

    //! returns the current matrix element
    _Tp& operator *() const;
    //! returns the i-th matrix element, relative to the current
    _Tp& operator [](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatIterator_& operator += (ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatIterator_& operator -= (ptrdiff_t ofs);
    //! decrements the iterator
    MatIterator_& operator --();
    //! decrements the iterator
    MatIterator_ operator --(int);
    //! increments the iterator
    MatIterator_& operator ++();
    //! increments the iterator
    MatIterator_ operator ++(int);
};



/////////////////////////////// SparseMatConstIterator ///////////////////////////////

/**  @brief Read-Only Sparse Matrix Iterator.

 Here is how to use the iterator to compute the sum of floating-point sparse matrix elements:

 \code
 SparseMatConstIterator it = m.begin(), it_end = m.end();
 double s = 0;
 CV_Assert( m.type() == CV_32F );
 for( ; it != it_end; ++it )
    s += it.value<float>();
 \endcode
*/
class CV_EXPORTS SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatConstIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator(const SparseMatConstIterator& it);

    //! the assignment operator
    SparseMatConstIterator& operator = (const SparseMatConstIterator& it);

    //! template method returning the current matrix element
    template<typename _Tp> const _Tp& value() const;
    //! returns the current node of the sparse matrix. it.node->idx is the current element index
    const SparseMat::Node* node() const;

    //! moves iterator to the previous element
    SparseMatConstIterator& operator --();
    //! moves iterator to the previous element
    SparseMatConstIterator operator --(int);
    //! moves iterator to the next element
    SparseMatConstIterator& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator operator ++(int);

    //! moves iterator to the element after the last element
    void seekEnd();

    const SparseMat* m;
    size_t hashidx;
    uchar* ptr;
};



////////////////////////////////// SparseMatIterator /////////////////////////////////

/** @brief  Read-write Sparse Matrix Iterator

 The class is similar to cv::SparseMatConstIterator,
 but can be used for in-place modification of the matrix elements.
*/
class CV_EXPORTS SparseMatIterator : public SparseMatConstIterator
{
public:
    //! the default constructor
    SparseMatIterator();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator(SparseMat* _m);
    //! the full constructor setting the iterator to the specified sparse matrix element
    SparseMatIterator(SparseMat* _m, const int* idx);
    //! the copy constructor
    SparseMatIterator(const SparseMatIterator& it);

    //! the assignment operator
    SparseMatIterator& operator = (const SparseMatIterator& it);
    //! returns read-write reference to the current sparse matrix element
    template<typename _Tp> _Tp& value() const;
    //! returns pointer to the current sparse matrix node. it.node->idx is the index of the current element (do not modify it!)
    SparseMat::Node* node() const;

    //! moves iterator to the next element
    SparseMatIterator& operator ++();
    //! moves iterator to the next element
    SparseMatIterator operator ++(int);
};



/////////////////////////////// SparseMatConstIterator_ //////////////////////////////

/** @brief  Template Read-Only Sparse Matrix Iterator Class.

 This is the derived from SparseMatConstIterator class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class SparseMatConstIterator_ : public SparseMatConstIterator
{
public:

    typedef std::forward_iterator_tag iterator_category;

    //! the default constructor
    SparseMatConstIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatConstIterator_(const SparseMat_<_Tp>* _m);
    SparseMatConstIterator_(const SparseMat* _m);
    //! the copy constructor
    SparseMatConstIterator_(const SparseMatConstIterator_& it);

    //! the assignment operator
    SparseMatConstIterator_& operator = (const SparseMatConstIterator_& it);
    //! the element access operator
    const _Tp& operator *() const;

    //! moves iterator to the next element
    SparseMatConstIterator_& operator ++();
    //! moves iterator to the next element
    SparseMatConstIterator_ operator ++(int);
};



///////////////////////////////// SparseMatIterator_ /////////////////////////////////

/** @brief  Template Read-Write Sparse Matrix Iterator Class.

 This is the derived from cv::SparseMatConstIterator_ class that
 introduces more convenient operator *() for accessing the current element.
*/
template<typename _Tp> class SparseMatIterator_ : public SparseMatConstIterator_<_Tp>
{
public:

    typedef std::forward_iterator_tag iterator_category;

    //! the default constructor
    SparseMatIterator_();
    //! the full constructor setting the iterator to the first sparse matrix element
    SparseMatIterator_(SparseMat_<_Tp>* _m);
    SparseMatIterator_(SparseMat* _m);
    //! the copy constructor
    SparseMatIterator_(const SparseMatIterator_& it);

    //! the assignment operator
    SparseMatIterator_& operator = (const SparseMatIterator_& it);
    //! returns the reference to the current element
    _Tp& operator *() const;

    //! moves the iterator to the next element
    SparseMatIterator_& operator ++();
    //! moves the iterator to the next element
    SparseMatIterator_ operator ++(int);
};



/////////////////////////////////// NAryMatIterator //////////////////////////////////

/** @brief n-ary multi-dimensional array iterator.

Use the class to implement unary, binary, and, generally, n-ary element-wise operations on
multi-dimensional arrays. Some of the arguments of an n-ary function may be continuous arrays, some
may be not. It is possible to use conventional MatIterator 's for each array but incrementing all of
the iterators after each small operations may be a big overhead. In this case consider using
NAryMatIterator to iterate through several matrices simultaneously as long as they have the same
geometry (dimensionality and all the dimension sizes are the same). On each iteration `it.planes[0]`,
`it.planes[1]`,... will be the slices of the corresponding matrices.

The example below illustrates how you can compute a normalized and threshold 3D color histogram:
@code
    void computeNormalizedColorHist(const Mat& image, Mat& hist, int N, double minProb)
    {
        const int histSize[] = {N, N, N};

        // make sure that the histogram has a proper size and type
        hist.create(3, histSize, CV_32F);

        // and clear it
        hist = Scalar(0);

        // the loop below assumes that the image
        // is a 8-bit 3-channel. check it.
        CV_Assert(image.type() == CV_8UC3);
        MatConstIterator_<Vec3b> it = image.begin<Vec3b>(),
                                 it_end = image.end<Vec3b>();
        for( ; it != it_end; ++it )
        {
            const Vec3b& pix = *it;
            hist.at<float>(pix[0]*N/256, pix[1]*N/256, pix[2]*N/256) += 1.f;
        }

        minProb *= image.rows*image.cols;

        // initialize iterator (the style is different from STL).
        // after initialization the iterator will contain
        // the number of slices or planes the iterator will go through.
        // it simultaneously increments iterators for several matrices
        // supplied as a null terminated list of pointers
        const Mat* arrays[] = {&hist, 0};
        Mat planes[1];
        NAryMatIterator itNAry(arrays, planes, 1);
        double s = 0;
        // iterate through the matrix. on each iteration
        // itNAry.planes[i] (of type Mat) will be set to the current plane
        // of the i-th n-dim matrix passed to the iterator constructor.
        for(int p = 0; p < itNAry.nplanes; p++, ++itNAry)
        {
            threshold(itNAry.planes[0], itNAry.planes[0], minProb, 0, THRESH_TOZERO);
            s += sum(itNAry.planes[0])[0];
        }

        s = 1./s;
        itNAry = NAryMatIterator(arrays, planes, 1);
        for(int p = 0; p < itNAry.nplanes; p++, ++itNAry)
            itNAry.planes[0] *= s;
    }
@endcode
 */
class CV_EXPORTS NAryMatIterator
{
public:
    //! the default constructor
    NAryMatIterator();
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, uchar** ptrs, int narrays=-1);
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat** arrays, Mat* planes, int narrays=-1);
    //! the separate iterator initialization method
    void init(const Mat** arrays, Mat* planes, uchar** ptrs, int narrays=-1);

    //! proceeds to the next plane of every iterated matrix
    NAryMatIterator& operator ++();
    //! proceeds to the next plane of every iterated matrix (postfix increment operator)
    NAryMatIterator operator ++(int);

    //! the iterated arrays
    const Mat** arrays;
    //! the current planes
    Mat* planes;
    //! data pointers
    uchar** ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;
protected:
    int iterdepth;
    size_t idx;
};



///////////////////////////////// Matrix Expressions /////////////////////////////////

class CV_EXPORTS MatOp
{
public:
    MatOp();
    virtual ~MatOp();

    virtual bool elementWise(const MatExpr& expr) const;
    virtual void assign(const MatExpr& expr, Mat& m, int type=-1) const = 0;
    virtual void roi(const MatExpr& expr, const Range& rowRange,
                     const Range& colRange, MatExpr& res) const;
    virtual void diag(const MatExpr& expr, int d, MatExpr& res) const;
    virtual void augAssignAdd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignSubtract(const MatExpr& expr, Mat& m) const;
    virtual void augAssignMultiply(const MatExpr& expr, Mat& m) const;
    virtual void augAssignDivide(const MatExpr& expr, Mat& m) const;
    virtual void augAssignAnd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignOr(const MatExpr& expr, Mat& m) const;
    virtual void augAssignXor(const MatExpr& expr, Mat& m) const;

    virtual void add(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void add(const MatExpr& expr1, const Scalar& s, MatExpr& res) const;

    virtual void subtract(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void subtract(const Scalar& s, const MatExpr& expr, MatExpr& res) const;

    virtual void multiply(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void multiply(const MatExpr& expr1, double s, MatExpr& res) const;

    virtual void divide(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void divide(double s, const MatExpr& expr, MatExpr& res) const;

    virtual void abs(const MatExpr& expr, MatExpr& res) const;

    virtual void transpose(const MatExpr& expr, MatExpr& res) const;
    virtual void matmul(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void invert(const MatExpr& expr, int method, MatExpr& res) const;

    virtual Size size(const MatExpr& expr) const;
    virtual int type(const MatExpr& expr) const;
};

/** @brief Matrix expression representation
@anchor MatrixExpressions
This is a list of implemented matrix operations that can be combined in arbitrary complex
expressions (here A, B stand for matrices ( Mat ), s for a scalar ( Scalar ), alpha for a
real-valued scalar ( double )):
-   Addition, subtraction, negation: `A+B`, `A-B`, `A+s`, `A-s`, `s+A`, `s-A`, `-A`
-   Scaling: `A*alpha`
-   Per-element multiplication and division: `A.mul(B)`, `A/B`, `alpha/A`
-   Matrix multiplication: `A*B`
-   Transposition: `A.t()` (means A<sup>T</sup>)
-   Matrix inversion and pseudo-inversion, solving linear systems and least-squares problems:
    `A.inv([method]) (~ A<sup>-1</sup>)`,   `A.inv([method])*B (~ X: AX=B)`
-   Comparison: `A cmpop B`, `A cmpop alpha`, `alpha cmpop A`, where *cmpop* is one of
  `>`, `>=`, `==`, `!=`, `<=`, `<`. The result of comparison is an 8-bit single channel mask whose
    elements are set to 255 (if the particular element or pair of elements satisfy the condition) or
    0.
-   Bitwise logical operations: `A logicop B`, `A logicop s`, `s logicop A`, `~A`, where *logicop* is one of
  `&`, `|`, `^`.
-   Element-wise minimum and maximum: `min(A, B)`, `min(A, alpha)`, `max(A, B)`, `max(A, alpha)`
-   Element-wise absolute value: `abs(A)`
-   Cross-product, dot-product: `A.cross(B)`, `A.dot(B)`
-   Any function of matrix or matrices and scalars that returns a matrix or a scalar, such as norm,
    mean, sum, countNonZero, trace, determinant, repeat, and others.
-   Matrix initializers ( Mat::eye(), Mat::zeros(), Mat::ones() ), matrix comma-separated
    initializers, matrix constructors and operators that extract sub-matrices (see Mat description).
-   Mat_<destination_type>() constructors to cast the result to the proper type.
@note Comma-separated initializers and probably some other operations may require additional
explicit Mat() or Mat_<T>() constructor calls to resolve a possible ambiguity.

Here are examples of matrix expressions:
@code
    // compute pseudo-inverse of A, equivalent to A.inv(DECOMP_SVD)
    SVD svd(A);
    Mat pinvA = svd.vt.t()*Mat::diag(1./svd.w)*svd.u.t();

    // compute the new vector of parameters in the Levenberg-Marquardt algorithm
    x -= (A.t()*A + lambda*Mat::eye(A.cols,A.cols,A.type())).inv(DECOMP_CHOLESKY)*(A.t()*err);

    // sharpen image using "unsharp mask" algorithm
    Mat blurred; double sigma = 1, threshold = 5, amount = 1;
    GaussianBlur(img, blurred, Size(), sigma, sigma);
    Mat lowContrastMask = abs(img - blurred) < threshold;
    Mat sharpened = img*(1+amount) + blurred*(-amount);
    img.copyTo(sharpened, lowContrastMask);
@endcode
*/
class CV_EXPORTS MatExpr
{
public:
    MatExpr();
    explicit MatExpr(const Mat& m);

    MatExpr(const MatOp* _op, int _flags, const Mat& _a = Mat(), const Mat& _b = Mat(),
            const Mat& _c = Mat(), double _alpha = 1, double _beta = 1, const Scalar& _s = Scalar());

    operator Mat() const;
    template<typename _Tp> operator Mat_<_Tp>() const;

    Size size() const;
    int type() const;

    MatExpr row(int y) const;
    MatExpr col(int x) const;
    MatExpr diag(int d = 0) const;
    MatExpr operator()( const Range& rowRange, const Range& colRange ) const;
    MatExpr operator()( const Rect& roi ) const;

    MatExpr t() const;
    MatExpr inv(int method = DECOMP_LU) const;
    MatExpr mul(const MatExpr& e, double scale=1) const;
    MatExpr mul(const Mat& m, double scale=1) const;

    Mat cross(const Mat& m) const;
    double dot(const Mat& m) const;

    const MatOp* op;
    int flags;

    Mat a, b, c;
    double alpha, beta;
    Scalar s;
};

//! @} core_basic

//! @relates cv::MatExpr
//! @{
CV_EXPORTS MatExpr operator + (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator + (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator + (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator - (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator - (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator - (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator - (const Mat& m);
CV_EXPORTS MatExpr operator - (const MatExpr& e);

CV_EXPORTS MatExpr operator * (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator * (const Mat& a, double s);
CV_EXPORTS MatExpr operator * (double s, const Mat& a);
CV_EXPORTS MatExpr operator * (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator * (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator * (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator / (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator / (const Mat& a, double s);
CV_EXPORTS MatExpr operator / (double s, const Mat& a);
CV_EXPORTS MatExpr operator / (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator / (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator / (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator < (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator < (const Mat& a, double s);
CV_EXPORTS MatExpr operator < (double s, const Mat& a);

CV_EXPORTS MatExpr operator <= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator <= (const Mat& a, double s);
CV_EXPORTS MatExpr operator <= (double s, const Mat& a);

CV_EXPORTS MatExpr operator == (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator == (const Mat& a, double s);
CV_EXPORTS MatExpr operator == (double s, const Mat& a);

CV_EXPORTS MatExpr operator != (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator != (const Mat& a, double s);
CV_EXPORTS MatExpr operator != (double s, const Mat& a);

CV_EXPORTS MatExpr operator >= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator >= (const Mat& a, double s);
CV_EXPORTS MatExpr operator >= (double s, const Mat& a);

CV_EXPORTS MatExpr operator > (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator > (const Mat& a, double s);
CV_EXPORTS MatExpr operator > (double s, const Mat& a);

CV_EXPORTS MatExpr operator & (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator & (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator & (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator | (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator | (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator | (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator ^ (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator ^ (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator ^ (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator ~(const Mat& m);

CV_EXPORTS MatExpr min(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr min(const Mat& a, double s);
CV_EXPORTS MatExpr min(double s, const Mat& a);

CV_EXPORTS MatExpr max(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr max(const Mat& a, double s);
CV_EXPORTS MatExpr max(double s, const Mat& a);

/** @brief Calculates an absolute value of each matrix element.

abs is a meta-function that is expanded to one of absdiff or convertScaleAbs forms:
- C = abs(A-B) is equivalent to `absdiff(A, B, C)`
- C = abs(A) is equivalent to `absdiff(A, Scalar::all(0), C)`
- C = `Mat_<Vec<uchar,n> >(abs(A*alpha + beta))` is equivalent to `convertScaleAbs(A, C, alpha,
beta)`

The output matrix has the same size and the same type as the input one except for the last case,
where C is depth=CV_8U .
@param m matrix.
@sa @ref MatrixExpressions, absdiff, convertScaleAbs
 */
CV_EXPORTS MatExpr abs(const Mat& m);
/** @overload
@param e matrix expression.
*/
CV_EXPORTS MatExpr abs(const MatExpr& e);
//! @} relates cv::MatExpr

} // cv

#include "opencv2/core/mat.inl.hpp"

#endif // OPENCV_CORE_MAT_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_MATRIX_OPERATIONS_HPP
#define OPENCV_CORE_MATRIX_OPERATIONS_HPP

#ifndef __cplusplus
#  error mat.inl.hpp header must be compiled as C++
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif

namespace cv
{
CV__DEBUG_NS_BEGIN


//! @cond IGNORED

//////////////////////// Input/Output Arrays ////////////////////////

inline void _InputArray::init(int _flags, const void* _obj)
{ flags = _flags; obj = (void*)_obj; }

inline void _InputArray::init(int _flags, const void* _obj, Size _sz)
{ flags = _flags; obj = (void*)_obj; sz = _sz; }

inline void* _InputArray::getObj() const { return obj; }
inline int _InputArray::getFlags() const { return flags; }
inline Size _InputArray::getSz() const { return sz; }

inline _InputArray::_InputArray() { init(NONE, 0); }
inline _InputArray::_InputArray(int _flags, void* _obj) { init(_flags, _obj); }
inline _InputArray::_InputArray(const Mat& m) { init(MAT+ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<Mat>& vec) { init(STD_VECTOR_MAT+ACCESS_READ, &vec); }
inline _InputArray::_InputArray(const UMat& m) { init(UMAT+ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<UMat>& vec) { init(STD_VECTOR_UMAT+ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + DataType<_Tp>::type + ACCESS_READ, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_InputArray::_InputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + DataType<_Tp>::type + ACCESS_READ, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputArray::_InputArray(const std::array<Mat, _Nm>& arr)
{ init(STD_ARRAY_MAT + ACCESS_READ, arr.data(), Size(1, _Nm)); }
#endif

inline
_InputArray::_InputArray(const std::vector<bool>& vec)
{ init(FIXED_TYPE + STD_BOOL_VECTOR + DataType<bool>::type + ACCESS_READ, &vec); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_READ, &vec); }

inline
_InputArray::_InputArray(const std::vector<std::vector<bool> >&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<std::vector<bool> > is not supported!\n"); }

template<typename _Tp> inline
_InputArray::_InputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_READ, &vec); }

template<typename _Tp, int m, int n> inline
_InputArray::_InputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_READ, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputArray::_InputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_READ, vec, Size(n, 1)); }

template<typename _Tp> inline
_InputArray::_InputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + DataType<_Tp>::type + ACCESS_READ, &m); }

inline _InputArray::_InputArray(const double& val)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + CV_64F + ACCESS_READ, &val, Size(1,1)); }

inline _InputArray::_InputArray(const MatExpr& expr)
{ init(FIXED_TYPE + FIXED_SIZE + EXPR + ACCESS_READ, &expr); }

inline _InputArray::_InputArray(const cuda::GpuMat& d_mat)
{ init(CUDA_GPU_MAT + ACCESS_READ, &d_mat); }

inline _InputArray::_InputArray(const std::vector<cuda::GpuMat>& d_mat)
{	init(STD_VECTOR_CUDA_GPU_MAT + ACCESS_READ, &d_mat);}

inline _InputArray::_InputArray(const ogl::Buffer& buf)
{ init(OPENGL_BUFFER + ACCESS_READ, &buf); }

inline _InputArray::_InputArray(const cuda::HostMem& cuda_mem)
{ init(CUDA_HOST_MEM + ACCESS_READ, &cuda_mem); }

inline _InputArray::~_InputArray() {}

inline Mat _InputArray::getMat(int i) const
{
    if( kind() == MAT && i < 0 )
        return *(const Mat*)obj;
    return getMat_(i);
}

inline bool _InputArray::isMat() const { return kind() == _InputArray::MAT; }
inline bool _InputArray::isUMat() const  { return kind() == _InputArray::UMAT; }
inline bool _InputArray::isMatVector() const { return kind() == _InputArray::STD_VECTOR_MAT; }
inline bool _InputArray::isUMatVector() const  { return kind() == _InputArray::STD_VECTOR_UMAT; }
inline bool _InputArray::isMatx() const { return kind() == _InputArray::MATX; }
inline bool _InputArray::isVector() const { return kind() == _InputArray::STD_VECTOR ||
                                                   kind() == _InputArray::STD_BOOL_VECTOR ||
                                                   kind() == _InputArray::STD_ARRAY; }
inline bool _InputArray::isGpuMatVector() const { return kind() == _InputArray::STD_VECTOR_CUDA_GPU_MAT; }

////////////////////////////////////////////////////////////////////////////////////////

inline _OutputArray::_OutputArray() { init(ACCESS_WRITE, 0); }
inline _OutputArray::_OutputArray(int _flags, void* _obj) { init(_flags|ACCESS_WRITE, _obj); }
inline _OutputArray::_OutputArray(Mat& m) { init(MAT+ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<Mat>& vec) { init(STD_VECTOR_MAT+ACCESS_WRITE, &vec); }
inline _OutputArray::_OutputArray(UMat& m) { init(UMAT+ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<UMat>& vec) { init(STD_VECTOR_UMAT+ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_OutputArray::_OutputArray(std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + DataType<_Tp>::type + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_OutputArray::_OutputArray(std::array<Mat, _Nm>& arr)
{ init(STD_ARRAY_MAT + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }
#endif

inline
_OutputArray::_OutputArray(std::vector<bool>&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<bool> cannot be an output array\n"); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec); }

inline
_OutputArray::_OutputArray(std::vector<std::vector<bool> >&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<std::vector<bool> > cannot be an output array\n"); }

template<typename _Tp> inline
_OutputArray::_OutputArray(std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + DataType<_Tp>::type + ACCESS_WRITE, &m); }

template<typename _Tp, int m, int n> inline
_OutputArray::_OutputArray(Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, &mtx, Size(n, m)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(_Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, vec, Size(n, 1)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_OutputArray::_OutputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + DataType<_Tp>::type + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_OutputArray::_OutputArray(const std::array<Mat, _Nm>& arr)
{ init(FIXED_SIZE + STD_ARRAY_MAT + ACCESS_WRITE, arr.data(), Size(1, _Nm)); }
#endif

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_WRITE, &vec); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + DataType<_Tp>::type + ACCESS_WRITE, &m); }

template<typename _Tp, int m, int n> inline
_OutputArray::_OutputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, &mtx, Size(n, m)); }

template<typename _Tp> inline
_OutputArray::_OutputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, vec, Size(n, 1)); }

inline _OutputArray::_OutputArray(cuda::GpuMat& d_mat)
{ init(CUDA_GPU_MAT + ACCESS_WRITE, &d_mat); }

inline _OutputArray::_OutputArray(std::vector<cuda::GpuMat>& d_mat)
{	init(STD_VECTOR_CUDA_GPU_MAT + ACCESS_WRITE, &d_mat);}

inline _OutputArray::_OutputArray(ogl::Buffer& buf)
{ init(OPENGL_BUFFER + ACCESS_WRITE, &buf); }

inline _OutputArray::_OutputArray(cuda::HostMem& cuda_mem)
{ init(CUDA_HOST_MEM + ACCESS_WRITE, &cuda_mem); }

inline _OutputArray::_OutputArray(const Mat& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<Mat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_WRITE, &vec); }

inline _OutputArray::_OutputArray(const UMat& m)
{ init(FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<UMat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_UMAT + ACCESS_WRITE, &vec); }

inline _OutputArray::_OutputArray(const cuda::GpuMat& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_GPU_MAT + ACCESS_WRITE, &d_mat); }


inline _OutputArray::_OutputArray(const ogl::Buffer& buf)
{ init(FIXED_TYPE + FIXED_SIZE + OPENGL_BUFFER + ACCESS_WRITE, &buf); }

inline _OutputArray::_OutputArray(const cuda::HostMem& cuda_mem)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_HOST_MEM + ACCESS_WRITE, &cuda_mem); }

///////////////////////////////////////////////////////////////////////////////////////////

inline _InputOutputArray::_InputOutputArray() { init(ACCESS_RW, 0); }
inline _InputOutputArray::_InputOutputArray(int _flags, void* _obj) { init(_flags|ACCESS_RW, _obj); }
inline _InputOutputArray::_InputOutputArray(Mat& m) { init(MAT+ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<Mat>& vec) { init(STD_VECTOR_MAT+ACCESS_RW, &vec); }
inline _InputOutputArray::_InputOutputArray(UMat& m) { init(UMAT+ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<UMat>& vec) { init(STD_VECTOR_UMAT+ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<_Tp>& vec)
{ init(FIXED_TYPE + STD_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + DataType<_Tp>::type + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(std::array<Mat, _Nm>& arr)
{ init(STD_ARRAY_MAT + ACCESS_RW, arr.data(), Size(1, _Nm)); }
#endif

inline _InputOutputArray::_InputOutputArray(std::vector<bool>&)
{ CV_Error(Error::StsUnsupportedFormat, "std::vector<bool> cannot be an input/output array\n"); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(Mat_<_Tp>& m)
{ init(FIXED_TYPE + MAT + DataType<_Tp>::type + ACCESS_RW, &m); }

template<typename _Tp, int m, int n> inline
_InputOutputArray::_InputOutputArray(Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(_Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, vec, Size(n, 1)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<_Tp>& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec); }

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(const std::array<_Tp, _Nm>& arr)
{ init(FIXED_TYPE + FIXED_SIZE + STD_ARRAY + DataType<_Tp>::type + ACCESS_RW, arr.data(), Size(1, _Nm)); }

template<std::size_t _Nm> inline
_InputOutputArray::_InputOutputArray(const std::array<Mat, _Nm>& arr)
{ init(FIXED_SIZE + STD_ARRAY_MAT + ACCESS_RW, arr.data(), Size(1, _Nm)); }
#endif

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<std::vector<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const std::vector<Mat_<_Tp> >& vec)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_RW, &vec); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const Mat_<_Tp>& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + DataType<_Tp>::type + ACCESS_RW, &m); }

template<typename _Tp, int m, int n> inline
_InputOutputArray::_InputOutputArray(const Matx<_Tp, m, n>& mtx)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, &mtx, Size(n, m)); }

template<typename _Tp> inline
_InputOutputArray::_InputOutputArray(const _Tp* vec, int n)
{ init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, vec, Size(n, 1)); }

inline _InputOutputArray::_InputOutputArray(cuda::GpuMat& d_mat)
{ init(CUDA_GPU_MAT + ACCESS_RW, &d_mat); }

inline _InputOutputArray::_InputOutputArray(ogl::Buffer& buf)
{ init(OPENGL_BUFFER + ACCESS_RW, &buf); }

inline _InputOutputArray::_InputOutputArray(cuda::HostMem& cuda_mem)
{ init(CUDA_HOST_MEM + ACCESS_RW, &cuda_mem); }

inline _InputOutputArray::_InputOutputArray(const Mat& m)
{ init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<Mat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_RW, &vec); }

inline _InputOutputArray::_InputOutputArray(const UMat& m)
{ init(FIXED_TYPE + FIXED_SIZE + UMAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<UMat>& vec)
{ init(FIXED_SIZE + STD_VECTOR_UMAT + ACCESS_RW, &vec); }

inline _InputOutputArray::_InputOutputArray(const cuda::GpuMat& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_GPU_MAT + ACCESS_RW, &d_mat); }

inline _InputOutputArray::_InputOutputArray(const std::vector<cuda::GpuMat>& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_CUDA_GPU_MAT + ACCESS_RW, &d_mat);}

template<> inline _InputOutputArray::_InputOutputArray(std::vector<cuda::GpuMat>& d_mat)
{ init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_CUDA_GPU_MAT + ACCESS_RW, &d_mat);}

inline _InputOutputArray::_InputOutputArray(const ogl::Buffer& buf)
{ init(FIXED_TYPE + FIXED_SIZE + OPENGL_BUFFER + ACCESS_RW, &buf); }

inline _InputOutputArray::_InputOutputArray(const cuda::HostMem& cuda_mem)
{ init(FIXED_TYPE + FIXED_SIZE + CUDA_HOST_MEM + ACCESS_RW, &cuda_mem); }

CV__DEBUG_NS_END

//////////////////////////////////////////// Mat //////////////////////////////////////////

inline
Mat::Mat()
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{}

inline
Mat::Mat(int _rows, int _cols, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_rows, _cols, _type);
}

inline
Mat::Mat(int _rows, int _cols, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_rows, _cols, _type);
    *this = _s;
}

inline
Mat::Mat(Size _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create( _sz.height, _sz.width, _type );
}

inline
Mat::Mat(Size _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz.height, _sz.width, _type);
    *this = _s;
}

inline
Mat::Mat(int _dims, const int* _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_dims, _sz, _type);
}

inline
Mat::Mat(int _dims, const int* _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_dims, _sz, _type);
    *this = _s;
}

inline
Mat::Mat(const std::vector<int>& _sz, int _type)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz, _type);
}

inline
Mat::Mat(const std::vector<int>& _sz, int _type, const Scalar& _s)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0),
      datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    create(_sz, _type);
    *this = _s;
}

inline
Mat::Mat(const Mat& m)
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), data(m.data),
      datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit), allocator(m.allocator),
      u(m.u), size(&rows), step(0)
{
    if( u )
        CV_XADD(&u->refcount, 1);
    if( m.dims <= 2 )
    {
        step[0] = m.step[0]; step[1] = m.step[1];
    }
    else
    {
        dims = 0;
        copySize(m);
    }
}

inline
Mat::Mat(int _rows, int _cols, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_rows), cols(_cols),
      data((uchar*)_data), datastart((uchar*)_data), dataend(0), datalimit(0),
      allocator(0), u(0), size(&rows)
{
    CV_Assert(total() == 0 || data != NULL);

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = cols * esz;
    if( _step == AUTO_STEP )
    {
        _step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        if( rows == 1 ) _step = minstep;
        CV_DbgAssert( _step >= minstep );

        if (_step % esz1 != 0)
        {
            CV_Error(Error::BadStep, "Step must be a multiple of esz1");
        }

        flags |= _step == minstep ? CONTINUOUS_FLAG : 0;
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step * rows;
    dataend = datalimit - _step + minstep;
}

inline
Mat::Mat(Size _sz, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), dims(2), rows(_sz.height), cols(_sz.width),
      data((uchar*)_data), datastart((uchar*)_data), dataend(0), datalimit(0),
      allocator(0), u(0), size(&rows)
{
    CV_Assert(total() == 0 || data != NULL);

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = cols*esz;
    if( _step == AUTO_STEP )
    {
        _step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        if( rows == 1 ) _step = minstep;
        CV_DbgAssert( _step >= minstep );

        if (_step % esz1 != 0)
        {
            CV_Error(Error::BadStep, "Step must be a multiple of esz1");
        }

        flags |= _step == minstep ? CONTINUOUS_FLAG : 0;
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step*rows;
    dataend = datalimit - _step + minstep;
}

template<typename _Tp> inline
Mat::Mat(const std::vector<_Tp>& vec, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows((int)vec.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(vec.empty())
        return;
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&vec[0];
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)vec.size(), 1, DataType<_Tp>::type, (uchar*)&vec[0]).copyTo(*this);
}

#ifdef CV_CXX11
template<typename _Tp> inline
Mat::Mat(const std::initializer_list<_Tp> list)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows((int)list.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(list.size() == 0)
        return;
    Mat((int)list.size(), 1, DataType<_Tp>::type, (uchar*)list.begin()).copyTo(*this);
}
#endif

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
Mat::Mat(const std::array<_Tp, _Nm>& arr, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows((int)arr.size()),
      cols(1), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if(arr.empty())
        return;
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)arr.data();
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)arr.size(), 1, DataType<_Tp>::type, (uchar*)arr.data()).copyTo(*this);
}
#endif

template<typename _Tp, int n> inline
Mat::Mat(const Vec<_Tp, n>& vec, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows(n), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)vec.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(n, 1, DataType<_Tp>::type, (void*)vec.val).copyTo(*this);
}


template<typename _Tp, int m, int n> inline
Mat::Mat(const Matx<_Tp,m,n>& M, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows(m), cols(n), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = cols * sizeof(_Tp);
        step[1] = sizeof(_Tp);
        datastart = data = (uchar*)M.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(m, n, DataType<_Tp>::type, (uchar*)M.val).copyTo(*this);
}

template<typename _Tp> inline
Mat::Mat(const Point_<_Tp>& pt, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows(2), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(2, 1, DataType<_Tp>::type);
        ((_Tp*)data)[0] = pt.x;
        ((_Tp*)data)[1] = pt.y;
    }
}

template<typename _Tp> inline
Mat::Mat(const Point3_<_Tp>& pt, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows(3), cols(1), data(0),
      datastart(0), dataend(0), datalimit(0), allocator(0), u(0), size(&rows), step(0)
{
    if( !copyData )
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar*)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(3, 1, DataType<_Tp>::type);
        ((_Tp*)data)[0] = pt.x;
        ((_Tp*)data)[1] = pt.y;
        ((_Tp*)data)[2] = pt.z;
    }
}

template<typename _Tp> inline
Mat::Mat(const MatCommaInitializer_<_Tp>& commaInitializer)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(0), rows(0), cols(0), data(0),
      datastart(0), dataend(0), allocator(0), u(0), size(&rows)
{
    *this = commaInitializer.operator Mat_<_Tp>();
}

inline
Mat::~Mat()
{
    release();
    if( step.p != step.buf )
        fastFree(step.p);
}

inline
Mat& Mat::operator = (const Mat& m)
{
    if( this != &m )
    {
        if( m.u )
            CV_XADD(&m.u->refcount, 1);
        release();
        flags = m.flags;
        if( dims <= 2 && m.dims <= 2 )
        {
            dims = m.dims;
            rows = m.rows;
            cols = m.cols;
            step[0] = m.step[0];
            step[1] = m.step[1];
        }
        else
            copySize(m);
        data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        datalimit = m.datalimit;
        allocator = m.allocator;
        u = m.u;
    }
    return *this;
}

inline
Mat Mat::row(int y) const
{
    return Mat(*this, Range(y, y + 1), Range::all());
}

inline
Mat Mat::col(int x) const
{
    return Mat(*this, Range::all(), Range(x, x + 1));
}

inline
Mat Mat::rowRange(int startrow, int endrow) const
{
    return Mat(*this, Range(startrow, endrow), Range::all());
}

inline
Mat Mat::rowRange(const Range& r) const
{
    return Mat(*this, r, Range::all());
}

inline
Mat Mat::colRange(int startcol, int endcol) const
{
    return Mat(*this, Range::all(), Range(startcol, endcol));
}

inline
Mat Mat::colRange(const Range& r) const
{
    return Mat(*this, Range::all(), r);
}

inline
Mat Mat::clone() const
{
    Mat m;
    copyTo(m);
    return m;
}

inline
void Mat::assignTo( Mat& m, int _type ) const
{
    if( _type < 0 )
        m = *this;
    else
        convertTo(m, _type);
}

inline
void Mat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if( dims <= 2 && rows == _rows && cols == _cols && type() == _type && data )
        return;
    int sz[] = {_rows, _cols};
    create(2, sz, _type);
}

inline
void Mat::create(Size _sz, int _type)
{
    create(_sz.height, _sz.width, _type);
}

inline
void Mat::addref()
{
    if( u )
        CV_XADD(&u->refcount, 1);
}

inline
void Mat::release()
{
    if( u && CV_XADD(&u->refcount, -1) == 1 )
        deallocate();
    u = NULL;
    datastart = dataend = datalimit = data = 0;
    for(int i = 0; i < dims; i++)
        size.p[i] = 0;
#ifdef _DEBUG
    flags = MAGIC_VAL;
    dims = rows = cols = 0;
    if(step.p != step.buf)
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
#endif
}

inline
Mat Mat::operator()( Range _rowRange, Range _colRange ) const
{
    return Mat(*this, _rowRange, _colRange);
}

inline
Mat Mat::operator()( const Rect& roi ) const
{
    return Mat(*this, roi);
}

inline
Mat Mat::operator()(const Range* ranges) const
{
    return Mat(*this, ranges);
}

inline
Mat Mat::operator()(const std::vector<Range>& ranges) const
{
    return Mat(*this, ranges);
}

inline
bool Mat::isContinuous() const
{
    return (flags & CONTINUOUS_FLAG) != 0;
}

inline
bool Mat::isSubmatrix() const
{
    return (flags & SUBMATRIX_FLAG) != 0;
}

inline
size_t Mat::elemSize() const
{
    return dims > 0 ? step.p[dims - 1] : 0;
}

inline
size_t Mat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int Mat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int Mat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int Mat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t Mat::step1(int i) const
{
    return step.p[i] / elemSize1();
}

inline
bool Mat::empty() const
{
    return data == 0 || total() == 0 || dims == 0;
}

inline
size_t Mat::total() const
{
    if( dims <= 2 )
        return (size_t)rows * cols;
    size_t p = 1;
    for( int i = 0; i < dims; i++ )
        p *= size[i];
    return p;
}

inline
size_t Mat::total(int startDim, int endDim) const
{
    CV_Assert( 0 <= startDim && startDim <= endDim);
    size_t p = 1;
    int endDim_ = endDim <= dims ? endDim : dims;
    for( int i = startDim; i < endDim_; i++ )
        p *= size[i];
    return p;
}

inline
uchar* Mat::ptr(int y)
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return data + step.p[0] * y;
}

inline
const uchar* Mat::ptr(int y) const
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return data + step.p[0] * y;
}

template<typename _Tp> inline
_Tp* Mat::ptr(int y)
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]) );
    return (_Tp*)(data + step.p[0] * y);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int y) const
{
    CV_DbgAssert( y == 0 || (data && dims >= 1 && data && (unsigned)y < (unsigned)size.p[0]) );
    return (const _Tp*)(data + step.p[0] * y);
}

inline
uchar* Mat::ptr(int i0, int i1)
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

inline
const uchar* Mat::ptr(int i0, int i1) const
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

template<typename _Tp> inline
_Tp* Mat::ptr(int i0, int i1)
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return (_Tp*)(data + i0 * step.p[0] + i1 * step.p[1]);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int i0, int i1) const
{
    CV_DbgAssert(dims >= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return (const _Tp*)(data + i0 * step.p[0] + i1 * step.p[1]);
}

inline
uchar* Mat::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

inline
const uchar* Mat::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

template<typename _Tp> inline
_Tp* Mat::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return (_Tp*)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]);
}

template<typename _Tp> inline
const _Tp* Mat::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert(dims >= 3);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return (const _Tp*)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]);
}

inline
uchar* Mat::ptr(const int* idx)
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return p;
}

inline
const uchar* Mat::ptr(const int* idx) const
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( d >= 1 && p );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size.p[i] );
        p += idx[i] * step.p[i];
    }
    return p;
}

template<typename _Tp> inline
_Tp& Mat::at(int i0, int i1)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((_Tp*)(data + step.p[0] * i0))[i1];
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0, int i1) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((const _Tp*)(data + step.p[0] * i0))[i1];
}

template<typename _Tp> inline
_Tp& Mat::at(Point pt)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((_Tp*)(data + step.p[0] * pt.y))[pt.x];
}

template<typename _Tp> inline
const _Tp& Mat::at(Point pt) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    CV_DbgAssert(CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((const _Tp*)(data + step.p[0] * pt.y))[pt.x];
}

template<typename _Tp> inline
_Tp& Mat::at(int i0)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    CV_DbgAssert(elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type));
    if( isContinuous() || size.p[0] == 1 )
        return ((_Tp*)data)[i0];
    if( size.p[1] == 1 )
        return *(_Tp*)(data + step.p[0] * i0);
    int i = i0 / cols, j = i0 - i * cols;
    return ((_Tp*)(data + step.p[0] * i))[j];
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    CV_DbgAssert(elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type));
    if( isContinuous() || size.p[0] == 1 )
        return ((const _Tp*)data)[i0];
    if( size.p[1] == 1 )
        return *(const _Tp*)(data + step.p[0] * i0);
    int i = i0 / cols, j = i0 - i * cols;
    return ((const _Tp*)(data + step.p[0] * i))[j];
}

template<typename _Tp> inline
_Tp& Mat::at(int i0, int i1, int i2)
{
    CV_DbgAssert( elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type) );
    return *(_Tp*)ptr(i0, i1, i2);
}

template<typename _Tp> inline
const _Tp& Mat::at(int i0, int i1, int i2) const
{
    CV_DbgAssert( elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type) );
    return *(const _Tp*)ptr(i0, i1, i2);
}

template<typename _Tp> inline
_Tp& Mat::at(const int* idx)
{
    CV_DbgAssert( elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type) );
    return *(_Tp*)ptr(idx);
}

template<typename _Tp> inline
const _Tp& Mat::at(const int* idx) const
{
    CV_DbgAssert( elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type) );
    return *(const _Tp*)ptr(idx);
}

template<typename _Tp, int n> inline
_Tp& Mat::at(const Vec<int, n>& idx)
{
    CV_DbgAssert( elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type) );
    return *(_Tp*)ptr(idx.val);
}

template<typename _Tp, int n> inline
const _Tp& Mat::at(const Vec<int, n>& idx) const
{
    CV_DbgAssert( elemSize() == CV_ELEM_SIZE(DataType<_Tp>::type) );
    return *(const _Tp*)ptr(idx.val);
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat::begin() const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatConstIterator_<_Tp>((const Mat_<_Tp>*)this);
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat::end() const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatConstIterator_<_Tp> it((const Mat_<_Tp>*)this);
    it += total();
    return it;
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat::begin()
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatIterator_<_Tp>((Mat_<_Tp>*)this);
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat::end()
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatIterator_<_Tp> it((Mat_<_Tp>*)this);
    it += total();
    return it;
}

template<typename _Tp, typename Functor> inline
void Mat::forEach(const Functor& operation) {
    this->forEach_impl<_Tp>(operation);
}

template<typename _Tp, typename Functor> inline
void Mat::forEach(const Functor& operation) const {
    // call as not const
    (const_cast<Mat*>(this))->forEach<const _Tp>(operation);
}

template<typename _Tp> inline
Mat::operator std::vector<_Tp>() const
{
    std::vector<_Tp> v;
    copyTo(v);
    return v;
}

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp, std::size_t _Nm> inline
Mat::operator std::array<_Tp, _Nm>() const
{
    std::array<_Tp, _Nm> v;
    copyTo(v);
    return v;
}
#endif

template<typename _Tp, int n> inline
Mat::operator Vec<_Tp, n>() const
{
    CV_Assert( data && dims <= 2 && (rows == 1 || cols == 1) &&
               rows + cols - 1 == n && channels() == 1 );

    if( isContinuous() && type() == DataType<_Tp>::type )
        return Vec<_Tp, n>((_Tp*)data);
    Vec<_Tp, n> v;
    Mat tmp(rows, cols, DataType<_Tp>::type, v.val);
    convertTo(tmp, tmp.type());
    return v;
}

template<typename _Tp, int m, int n> inline
Mat::operator Matx<_Tp, m, n>() const
{
    CV_Assert( data && dims <= 2 && rows == m && cols == n && channels() == 1 );

    if( isContinuous() && type() == DataType<_Tp>::type )
        return Matx<_Tp, m, n>((_Tp*)data);
    Matx<_Tp, m, n> mtx;
    Mat tmp(rows, cols, DataType<_Tp>::type, mtx.val);
    convertTo(tmp, tmp.type());
    return mtx;
}

template<typename _Tp> inline
void Mat::push_back(const _Tp& elem)
{
    if( !data )
    {
        *this = Mat(1, 1, DataType<_Tp>::type, (void*)&elem).clone();
        return;
    }
    CV_Assert(DataType<_Tp>::type == type() && cols == 1
              /* && dims == 2 (cols == 1 implies dims == 2) */);
    const uchar* tmp = dataend + step[0];
    if( !isSubmatrix() && isContinuous() && tmp <= datalimit )
    {
        *(_Tp*)(data + (size.p[0]++) * step.p[0]) = elem;
        dataend = tmp;
    }
    else
        push_back_(&elem);
}

template<typename _Tp> inline
void Mat::push_back(const Mat_<_Tp>& m)
{
    push_back((const Mat&)m);
}

template<> inline
void Mat::push_back(const MatExpr& expr)
{
    push_back(static_cast<Mat>(expr));
}

#ifdef CV_CXX_MOVE_SEMANTICS

inline
Mat::Mat(Mat&& m)
    : flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), data(m.data),
      datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit), allocator(m.allocator),
      u(m.u), size(&rows)
{
    if (m.dims <= 2)  // move new step/size info
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        CV_DbgAssert(m.step.p != m.step.buf);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL; m.dims = m.rows = m.cols = 0;
    m.data = NULL; m.datastart = NULL; m.dataend = NULL; m.datalimit = NULL;
    m.allocator = NULL;
    m.u = NULL;
}

inline
Mat& Mat::operator = (Mat&& m)
{
    if (this == &m)
      return *this;

    release();
    flags = m.flags; dims = m.dims; rows = m.rows; cols = m.cols; data = m.data;
    datastart = m.datastart; dataend = m.dataend; datalimit = m.datalimit; allocator = m.allocator;
    u = m.u;
    if (step.p != step.buf) // release self step/size
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
    if (m.dims <= 2) // move new step/size info
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        CV_DbgAssert(m.step.p != m.step.buf);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL; m.dims = m.rows = m.cols = 0;
    m.data = NULL; m.datastart = NULL; m.dataend = NULL; m.datalimit = NULL;
    m.allocator = NULL;
    m.u = NULL;
    return *this;
}

#endif


///////////////////////////// MatSize ////////////////////////////

inline
MatSize::MatSize(int* _p)
    : p(_p) {}

inline
Size MatSize::operator()() const
{
    CV_DbgAssert(p[-1] <= 2);
    return Size(p[1], p[0]);
}

inline
const int& MatSize::operator[](int i) const
{
    return p[i];
}

inline
int& MatSize::operator[](int i)
{
    return p[i];
}

inline
MatSize::operator const int*() const
{
    return p;
}

inline
bool MatSize::operator == (const MatSize& sz) const
{
    int d = p[-1];
    int dsz = sz.p[-1];
    if( d != dsz )
        return false;
    if( d == 2 )
        return p[0] == sz.p[0] && p[1] == sz.p[1];

    for( int i = 0; i < d; i++ )
        if( p[i] != sz.p[i] )
            return false;
    return true;
}

inline
bool MatSize::operator != (const MatSize& sz) const
{
    return !(*this == sz);
}



///////////////////////////// MatStep ////////////////////////////

inline
MatStep::MatStep()
{
    p = buf; p[0] = p[1] = 0;
}

inline
MatStep::MatStep(size_t s)
{
    p = buf; p[0] = s; p[1] = 0;
}

inline
const size_t& MatStep::operator[](int i) const
{
    return p[i];
}

inline
size_t& MatStep::operator[](int i)
{
    return p[i];
}

inline MatStep::operator size_t() const
{
    CV_DbgAssert( p == buf );
    return buf[0];
}

inline MatStep& MatStep::operator = (size_t s)
{
    CV_DbgAssert( p == buf );
    buf[0] = s;
    return *this;
}



////////////////////////////// Mat_<_Tp> ////////////////////////////

template<typename _Tp> inline
Mat_<_Tp>::Mat_()
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols)
    : Mat(_rows, _cols, DataType<_Tp>::type)
{
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols, const _Tp& value)
    : Mat(_rows, _cols, DataType<_Tp>::type)
{
    *this = value;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Size _sz)
    : Mat(_sz.height, _sz.width, DataType<_Tp>::type)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Size _sz, const _Tp& value)
    : Mat(_sz.height, _sz.width, DataType<_Tp>::type)
{
    *this = value;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _dims, const int* _sz)
    : Mat(_dims, _sz, DataType<_Tp>::type)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _dims, const int* _sz, const _Tp& _s)
    : Mat(_dims, _sz, DataType<_Tp>::type, Scalar(_s))
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _dims, const int* _sz, _Tp* _data, const size_t* _steps)
    : Mat(_dims, _sz, DataType<_Tp>::type, _data, _steps)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_<_Tp>& m, const Range* ranges)
    : Mat(m, ranges)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_<_Tp>& m, const std::vector<Range>& ranges)
    : Mat(m, ranges)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat& m)
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type;
    *this = m;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_& m)
    : Mat(m)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(int _rows, int _cols, _Tp* _data, size_t steps)
    : Mat(_rows, _cols, DataType<_Tp>::type, _data, steps)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_& m, const Range& _rowRange, const Range& _colRange)
    : Mat(m, _rowRange, _colRange)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Mat_& m, const Rect& roi)
    : Mat(m, roi)
{}

template<typename _Tp> template<int n> inline
Mat_<_Tp>::Mat_(const Vec<typename DataType<_Tp>::channel_type, n>& vec, bool copyData)
    : Mat(n / DataType<_Tp>::channels, 1, DataType<_Tp>::type, (void*)&vec)
{
    CV_Assert(n%DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> template<int m, int n> inline
Mat_<_Tp>::Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n>& M, bool copyData)
    : Mat(m, n / DataType<_Tp>::channels, DataType<_Tp>::type, (void*)&M)
{
    CV_Assert(n % DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Point_<typename DataType<_Tp>::channel_type>& pt, bool copyData)
    : Mat(2 / DataType<_Tp>::channels, 1, DataType<_Tp>::type, (void*)&pt)
{
    CV_Assert(2 % DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const Point3_<typename DataType<_Tp>::channel_type>& pt, bool copyData)
    : Mat(3 / DataType<_Tp>::channels, 1, DataType<_Tp>::type, (void*)&pt)
{
    CV_Assert(3 % DataType<_Tp>::channels == 0);
    if( copyData )
        *this = clone();
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const MatCommaInitializer_<_Tp>& commaInitializer)
    : Mat(commaInitializer)
{}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const std::vector<_Tp>& vec, bool copyData)
    : Mat(vec, copyData)
{}

#ifdef CV_CXX11
template<typename _Tp> inline
Mat_<_Tp>::Mat_(std::initializer_list<_Tp> list)
    : Mat(list)
{}
#endif

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp> template<std::size_t _Nm> inline
Mat_<_Tp>::Mat_(const std::array<_Tp, _Nm>& arr, bool copyData)
    : Mat(arr, copyData)
{}
#endif

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const Mat& m)
{
    if( DataType<_Tp>::type == m.type() )
    {
        Mat::operator = (m);
        return *this;
    }
    if( DataType<_Tp>::depth == m.depth() )
    {
        return (*this = m.reshape(DataType<_Tp>::channels, m.dims, 0));
    }
    CV_DbgAssert(DataType<_Tp>::channels == m.channels());
    m.convertTo(*this, type());
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const Mat_& m)
{
    Mat::operator=(m);
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const _Tp& s)
{
    typedef typename DataType<_Tp>::vec_type VT;
    Mat::operator=(Scalar((const VT&)s));
    return *this;
}

template<typename _Tp> inline
void Mat_<_Tp>::create(int _rows, int _cols)
{
    Mat::create(_rows, _cols, DataType<_Tp>::type);
}

template<typename _Tp> inline
void Mat_<_Tp>::create(Size _sz)
{
    Mat::create(_sz, DataType<_Tp>::type);
}

template<typename _Tp> inline
void Mat_<_Tp>::create(int _dims, const int* _sz)
{
    Mat::create(_dims, _sz, DataType<_Tp>::type);
}

template<typename _Tp> inline
void Mat_<_Tp>::release()
{
    Mat::release();
#ifdef _DEBUG
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type;
#endif
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::cross(const Mat_& m) const
{
    return Mat_<_Tp>(Mat::cross(m));
}

template<typename _Tp> template<typename T2> inline
Mat_<_Tp>::operator Mat_<T2>() const
{
    return Mat_<T2>(*this);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::row(int y) const
{
    return Mat_(*this, Range(y, y+1), Range::all());
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::col(int x) const
{
    return Mat_(*this, Range::all(), Range(x, x+1));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::diag(int d) const
{
    return Mat_(Mat::diag(d));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::clone() const
{
    return Mat_(Mat::clone());
}

template<typename _Tp> inline
size_t Mat_<_Tp>::elemSize() const
{
    CV_DbgAssert( Mat::elemSize() == sizeof(_Tp) );
    return sizeof(_Tp);
}

template<typename _Tp> inline
size_t Mat_<_Tp>::elemSize1() const
{
    CV_DbgAssert( Mat::elemSize1() == sizeof(_Tp) / DataType<_Tp>::channels );
    return sizeof(_Tp) / DataType<_Tp>::channels;
}

template<typename _Tp> inline
int Mat_<_Tp>::type() const
{
    CV_DbgAssert( Mat::type() == DataType<_Tp>::type );
    return DataType<_Tp>::type;
}

template<typename _Tp> inline
int Mat_<_Tp>::depth() const
{
    CV_DbgAssert( Mat::depth() == DataType<_Tp>::depth );
    return DataType<_Tp>::depth;
}

template<typename _Tp> inline
int Mat_<_Tp>::channels() const
{
    CV_DbgAssert( Mat::channels() == DataType<_Tp>::channels );
    return DataType<_Tp>::channels;
}

template<typename _Tp> inline
size_t Mat_<_Tp>::stepT(int i) const
{
    return step.p[i] / elemSize();
}

template<typename _Tp> inline
size_t Mat_<_Tp>::step1(int i) const
{
    return step.p[i] / elemSize1();
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    return (Mat_<_Tp>&)(Mat::adjustROI(dtop, dbottom, dleft, dright));
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()( const Range& _rowRange, const Range& _colRange ) const
{
    return Mat_<_Tp>(*this, _rowRange, _colRange);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()( const Rect& roi ) const
{
    return Mat_<_Tp>(*this, roi);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()( const Range* ranges ) const
{
    return Mat_<_Tp>(*this, ranges);
}

template<typename _Tp> inline
Mat_<_Tp> Mat_<_Tp>::operator()(const std::vector<Range>& ranges) const
{
    return Mat_<_Tp>(*this, ranges);
}

template<typename _Tp> inline
_Tp* Mat_<_Tp>::operator [](int y)
{
    CV_DbgAssert( 0 <= y && y < size.p[0] );
    return (_Tp*)(data + y*step.p[0]);
}

template<typename _Tp> inline
const _Tp* Mat_<_Tp>::operator [](int y) const
{
    CV_DbgAssert( 0 <= y && y < size.p[0] );
    return (const _Tp*)(data + y*step.p[0]);
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(int i0, int i1)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert(type() == DataType<_Tp>::type);
    return ((_Tp*)(data + step.p[0] * i0))[i1];
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(int i0, int i1) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    CV_DbgAssert(type() == DataType<_Tp>::type);
    return ((const _Tp*)(data + step.p[0] * i0))[i1];
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(Point pt)
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)pt.x < (unsigned)size.p[1]);
    CV_DbgAssert(type() == DataType<_Tp>::type);
    return ((_Tp*)(data + step.p[0] * pt.y))[pt.x];
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(Point pt) const
{
    CV_DbgAssert(dims <= 2);
    CV_DbgAssert(data);
    CV_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    CV_DbgAssert((unsigned)pt.x < (unsigned)size.p[1]);
    CV_DbgAssert(type() == DataType<_Tp>::type);
    return ((const _Tp*)(data + step.p[0] * pt.y))[pt.x];
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(const int* idx)
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(const int* idx) const
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> template<int n> inline
_Tp& Mat_<_Tp>::operator ()(const Vec<int, n>& idx)
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> template<int n> inline
const _Tp& Mat_<_Tp>::operator ()(const Vec<int, n>& idx) const
{
    return Mat::at<_Tp>(idx);
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(int i0)
{
    return this->at<_Tp>(i0);
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(int i0) const
{
    return this->at<_Tp>(i0);
}

template<typename _Tp> inline
_Tp& Mat_<_Tp>::operator ()(int i0, int i1, int i2)
{
    return this->at<_Tp>(i0, i1, i2);
}

template<typename _Tp> inline
const _Tp& Mat_<_Tp>::operator ()(int i0, int i1, int i2) const
{
    return this->at<_Tp>(i0, i1, i2);
}

template<typename _Tp> inline
Mat_<_Tp>::operator std::vector<_Tp>() const
{
    std::vector<_Tp> v;
    copyTo(v);
    return v;
}

#ifdef CV_CXX_STD_ARRAY
template<typename _Tp> template<std::size_t _Nm> inline
Mat_<_Tp>::operator std::array<_Tp, _Nm>() const
{
    std::array<_Tp, _Nm> a;
    copyTo(a);
    return a;
}
#endif

template<typename _Tp> template<int n> inline
Mat_<_Tp>::operator Vec<typename DataType<_Tp>::channel_type, n>() const
{
    CV_Assert(n % DataType<_Tp>::channels == 0);

#if defined _MSC_VER
    const Mat* pMat = (const Mat*)this; // workaround for MSVS <= 2012 compiler bugs (but GCC 4.6 dislikes this workaround)
    return pMat->operator Vec<typename DataType<_Tp>::channel_type, n>();
#else
    return this->Mat::operator Vec<typename DataType<_Tp>::channel_type, n>();
#endif
}

template<typename _Tp> template<int m, int n> inline
Mat_<_Tp>::operator Matx<typename DataType<_Tp>::channel_type, m, n>() const
{
    CV_Assert(n % DataType<_Tp>::channels == 0);

#if defined _MSC_VER
    const Mat* pMat = (const Mat*)this; // workaround for MSVS <= 2012 compiler bugs (but GCC 4.6 dislikes this workaround)
    Matx<typename DataType<_Tp>::channel_type, m, n> res = pMat->operator Matx<typename DataType<_Tp>::channel_type, m, n>();
    return res;
#else
    Matx<typename DataType<_Tp>::channel_type, m, n> res = this->Mat::operator Matx<typename DataType<_Tp>::channel_type, m, n>();
    return res;
#endif
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat_<_Tp>::begin() const
{
    return Mat::begin<_Tp>();
}

template<typename _Tp> inline
MatConstIterator_<_Tp> Mat_<_Tp>::end() const
{
    return Mat::end<_Tp>();
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat_<_Tp>::begin()
{
    return Mat::begin<_Tp>();
}

template<typename _Tp> inline
MatIterator_<_Tp> Mat_<_Tp>::end()
{
    return Mat::end<_Tp>();
}

template<typename _Tp> template<typename Functor> inline
void Mat_<_Tp>::forEach(const Functor& operation) {
    Mat::forEach<_Tp, Functor>(operation);
}

template<typename _Tp> template<typename Functor> inline
void Mat_<_Tp>::forEach(const Functor& operation) const {
    Mat::forEach<_Tp, Functor>(operation);
}

#ifdef CV_CXX_MOVE_SEMANTICS

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Mat_&& m)
    : Mat(m)
{
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (Mat_&& m)
{
    Mat::operator = (std::move(m));
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(Mat&& m)
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type;
    *this = m;
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (Mat&& m)
{
    if( DataType<_Tp>::type == m.type() )
    {
        Mat::operator = ((Mat&&)m);
        return *this;
    }
    if( DataType<_Tp>::depth == m.depth() )
    {
        Mat::operator = ((Mat&&)m.reshape(DataType<_Tp>::channels, m.dims, 0));
        return *this;
    }
    CV_DbgAssert(DataType<_Tp>::channels == m.channels());
    m.convertTo(*this, type());
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(MatExpr&& e)
    : Mat()
{
    flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type;
    *this = Mat(e);
}

#endif

///////////////////////////// SparseMat /////////////////////////////

inline
SparseMat::SparseMat()
    : flags(MAGIC_VAL), hdr(0)
{}

inline
SparseMat::SparseMat(int _dims, const int* _sizes, int _type)
    : flags(MAGIC_VAL), hdr(0)
{
    create(_dims, _sizes, _type);
}

inline
SparseMat::SparseMat(const SparseMat& m)
    : flags(m.flags), hdr(m.hdr)
{
    addref();
}

inline
SparseMat::~SparseMat()
{
    release();
}

inline
SparseMat& SparseMat::operator = (const SparseMat& m)
{
    if( this != &m )
    {
        if( m.hdr )
            CV_XADD(&m.hdr->refcount, 1);
        release();
        flags = m.flags;
        hdr = m.hdr;
    }
    return *this;
}

inline
SparseMat& SparseMat::operator = (const Mat& m)
{
    return (*this = SparseMat(m));
}

inline
SparseMat SparseMat::clone() const
{
    SparseMat temp;
    this->copyTo(temp);
    return temp;
}

inline
void SparseMat::assignTo( SparseMat& m, int _type ) const
{
    if( _type < 0 )
        m = *this;
    else
        convertTo(m, _type);
}

inline
void SparseMat::addref()
{
    if( hdr )
        CV_XADD(&hdr->refcount, 1);
}

inline
void SparseMat::release()
{
    if( hdr && CV_XADD(&hdr->refcount, -1) == 1 )
        delete hdr;
    hdr = 0;
}

inline
size_t SparseMat::elemSize() const
{
    return CV_ELEM_SIZE(flags);
}

inline
size_t SparseMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int SparseMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int SparseMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int SparseMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
const int* SparseMat::size() const
{
    return hdr ? hdr->size : 0;
}

inline
int SparseMat::size(int i) const
{
    if( hdr )
    {
        CV_DbgAssert((unsigned)i < (unsigned)hdr->dims);
        return hdr->size[i];
    }
    return 0;
}

inline
int SparseMat::dims() const
{
    return hdr ? hdr->dims : 0;
}

inline
size_t SparseMat::nzcount() const
{
    return hdr ? hdr->nodeCount : 0;
}

inline
size_t SparseMat::hash(int i0) const
{
    return (size_t)i0;
}

inline
size_t SparseMat::hash(int i0, int i1) const
{
    return (size_t)(unsigned)i0 * HASH_SCALE + (unsigned)i1;
}

inline
size_t SparseMat::hash(int i0, int i1, int i2) const
{
    return ((size_t)(unsigned)i0 * HASH_SCALE + (unsigned)i1) * HASH_SCALE + (unsigned)i2;
}

inline
size_t SparseMat::hash(const int* idx) const
{
    size_t h = (unsigned)idx[0];
    if( !hdr )
        return 0;
    int d = hdr->dims;
    for(int i = 1; i < d; i++ )
        h = h * HASH_SCALE + (unsigned)idx[i];
    return h;
}

template<typename _Tp> inline
_Tp& SparseMat::ref(int i0, size_t* hashval)
{
    return *(_Tp*)((SparseMat*)this)->ptr(i0, true, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat::ref(int i0, int i1, size_t* hashval)
{
    return *(_Tp*)((SparseMat*)this)->ptr(i0, i1, true, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat::ref(int i0, int i1, int i2, size_t* hashval)
{
    return *(_Tp*)((SparseMat*)this)->ptr(i0, i1, i2, true, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat::ref(const int* idx, size_t* hashval)
{
    return *(_Tp*)((SparseMat*)this)->ptr(idx, true, hashval);
}

template<typename _Tp> inline
_Tp SparseMat::value(int i0, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(i0, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
_Tp SparseMat::value(int i0, int i1, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(i0, i1, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
_Tp SparseMat::value(int i0, int i1, int i2, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(i0, i1, i2, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
_Tp SparseMat::value(const int* idx, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(idx, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline
const _Tp* SparseMat::find(int i0, size_t* hashval) const
{
    return (const _Tp*)((SparseMat*)this)->ptr(i0, false, hashval);
}

template<typename _Tp> inline
const _Tp* SparseMat::find(int i0, int i1, size_t* hashval) const
{
    return (const _Tp*)((SparseMat*)this)->ptr(i0, i1, false, hashval);
}

template<typename _Tp> inline
const _Tp* SparseMat::find(int i0, int i1, int i2, size_t* hashval) const
{
    return (const _Tp*)((SparseMat*)this)->ptr(i0, i1, i2, false, hashval);
}

template<typename _Tp> inline
const _Tp* SparseMat::find(const int* idx, size_t* hashval) const
{
    return (const _Tp*)((SparseMat*)this)->ptr(idx, false, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat::value(Node* n)
{
    return *(_Tp*)((uchar*)n + hdr->valueOffset);
}

template<typename _Tp> inline
const _Tp& SparseMat::value(const Node* n) const
{
    return *(const _Tp*)((const uchar*)n + hdr->valueOffset);
}

inline
SparseMat::Node* SparseMat::node(size_t nidx)
{
    return (Node*)(void*)&hdr->pool[nidx];
}

inline
const SparseMat::Node* SparseMat::node(size_t nidx) const
{
    return (const Node*)(const void*)&hdr->pool[nidx];
}

inline
SparseMatIterator SparseMat::begin()
{
    return SparseMatIterator(this);
}

inline
SparseMatConstIterator SparseMat::begin() const
{
    return SparseMatConstIterator(this);
}

inline
SparseMatIterator SparseMat::end()
{
    SparseMatIterator it(this);
    it.seekEnd();
    return it;
}

inline
SparseMatConstIterator SparseMat::end() const
{
    SparseMatConstIterator it(this);
    it.seekEnd();
    return it;
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat::begin()
{
    return SparseMatIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat::begin() const
{
    return SparseMatConstIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat::end()
{
    SparseMatIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat::end() const
{
    SparseMatConstIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}



///////////////////////////// SparseMat_ ////////////////////////////

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_()
{
    flags = MAGIC_VAL | DataType<_Tp>::type;
}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(int _dims, const int* _sizes)
    : SparseMat(_dims, _sizes, DataType<_Tp>::type)
{}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(const SparseMat& m)
{
    if( m.type() == DataType<_Tp>::type )
        *this = (const SparseMat_<_Tp>&)m;
    else
        m.convertTo(*this, DataType<_Tp>::type);
}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(const SparseMat_<_Tp>& m)
{
    this->flags = m.flags;
    this->hdr = m.hdr;
    if( this->hdr )
        CV_XADD(&this->hdr->refcount, 1);
}

template<typename _Tp> inline
SparseMat_<_Tp>::SparseMat_(const Mat& m)
{
    SparseMat sm(m);
    *this = sm;
}

template<typename _Tp> inline
SparseMat_<_Tp>& SparseMat_<_Tp>::operator = (const SparseMat_<_Tp>& m)
{
    if( this != &m )
    {
        if( m.hdr ) CV_XADD(&m.hdr->refcount, 1);
        release();
        flags = m.flags;
        hdr = m.hdr;
    }
    return *this;
}

template<typename _Tp> inline
SparseMat_<_Tp>& SparseMat_<_Tp>::operator = (const SparseMat& m)
{
    if( m.type() == DataType<_Tp>::type )
        return (*this = (const SparseMat_<_Tp>&)m);
    m.convertTo(*this, DataType<_Tp>::type);
    return *this;
}

template<typename _Tp> inline
SparseMat_<_Tp>& SparseMat_<_Tp>::operator = (const Mat& m)
{
    return (*this = SparseMat(m));
}

template<typename _Tp> inline
SparseMat_<_Tp> SparseMat_<_Tp>::clone() const
{
    SparseMat_<_Tp> m;
    this->copyTo(m);
    return m;
}

template<typename _Tp> inline
void SparseMat_<_Tp>::create(int _dims, const int* _sizes)
{
    SparseMat::create(_dims, _sizes, DataType<_Tp>::type);
}

template<typename _Tp> inline
int SparseMat_<_Tp>::type() const
{
    return DataType<_Tp>::type;
}

template<typename _Tp> inline
int SparseMat_<_Tp>::depth() const
{
    return DataType<_Tp>::depth;
}

template<typename _Tp> inline
int SparseMat_<_Tp>::channels() const
{
    return DataType<_Tp>::channels;
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(int i0, size_t* hashval)
{
    return SparseMat::ref<_Tp>(i0, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(int i0, size_t* hashval) const
{
    return SparseMat::value<_Tp>(i0, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(int i0, int i1, size_t* hashval)
{
    return SparseMat::ref<_Tp>(i0, i1, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(int i0, int i1, size_t* hashval) const
{
    return SparseMat::value<_Tp>(i0, i1, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(int i0, int i1, int i2, size_t* hashval)
{
    return SparseMat::ref<_Tp>(i0, i1, i2, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(int i0, int i1, int i2, size_t* hashval) const
{
    return SparseMat::value<_Tp>(i0, i1, i2, hashval);
}

template<typename _Tp> inline
_Tp& SparseMat_<_Tp>::ref(const int* idx, size_t* hashval)
{
    return SparseMat::ref<_Tp>(idx, hashval);
}

template<typename _Tp> inline
_Tp SparseMat_<_Tp>::operator()(const int* idx, size_t* hashval) const
{
    return SparseMat::value<_Tp>(idx, hashval);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat_<_Tp>::begin()
{
    return SparseMatIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat_<_Tp>::begin() const
{
    return SparseMatConstIterator_<_Tp>(this);
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMat_<_Tp>::end()
{
    SparseMatIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMat_<_Tp>::end() const
{
    SparseMatConstIterator_<_Tp> it(this);
    it.seekEnd();
    return it;
}



////////////////////////// MatConstIterator /////////////////////////

inline
MatConstIterator::MatConstIterator()
    : m(0), elemSize(0), ptr(0), sliceStart(0), sliceEnd(0)
{}

inline
MatConstIterator::MatConstIterator(const Mat* _m)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    if( m && m->isContinuous() )
    {
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total()*elemSize;
    }
    seek((const int*)0);
}

inline
MatConstIterator::MatConstIterator(const Mat* _m, int _row, int _col)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    CV_Assert(m && m->dims <= 2);
    if( m->isContinuous() )
    {
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total()*elemSize;
    }
    int idx[] = {_row, _col};
    seek(idx);
}

inline
MatConstIterator::MatConstIterator(const Mat* _m, Point _pt)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    CV_Assert(m && m->dims <= 2);
    if( m->isContinuous() )
    {
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total()*elemSize;
    }
    int idx[] = {_pt.y, _pt.x};
    seek(idx);
}

inline
MatConstIterator::MatConstIterator(const MatConstIterator& it)
    : m(it.m), elemSize(it.elemSize), ptr(it.ptr), sliceStart(it.sliceStart), sliceEnd(it.sliceEnd)
{}

inline
MatConstIterator& MatConstIterator::operator = (const MatConstIterator& it )
{
    m = it.m; elemSize = it.elemSize; ptr = it.ptr;
    sliceStart = it.sliceStart; sliceEnd = it.sliceEnd;
    return *this;
}

inline
const uchar* MatConstIterator::operator *() const
{
    return ptr;
}

inline MatConstIterator& MatConstIterator::operator += (ptrdiff_t ofs)
{
    if( !m || ofs == 0 )
        return *this;
    ptrdiff_t ofsb = ofs*elemSize;
    ptr += ofsb;
    if( ptr < sliceStart || sliceEnd <= ptr )
    {
        ptr -= ofsb;
        seek(ofs, true);
    }
    return *this;
}

inline
MatConstIterator& MatConstIterator::operator -= (ptrdiff_t ofs)
{
    return (*this += -ofs);
}

inline
MatConstIterator& MatConstIterator::operator --()
{
    if( m && (ptr -= elemSize) < sliceStart )
    {
        ptr += elemSize;
        seek(-1, true);
    }
    return *this;
}

inline
MatConstIterator MatConstIterator::operator --(int)
{
    MatConstIterator b = *this;
    *this += -1;
    return b;
}

inline
MatConstIterator& MatConstIterator::operator ++()
{
    if( m && (ptr += elemSize) >= sliceEnd )
    {
        ptr -= elemSize;
        seek(1, true);
    }
    return *this;
}

inline MatConstIterator MatConstIterator::operator ++(int)
{
    MatConstIterator b = *this;
    *this += 1;
    return b;
}


static inline
bool operator == (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

static inline
bool operator != (const MatConstIterator& a, const MatConstIterator& b)
{
    return !(a == b);
}

static inline
bool operator < (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr < b.ptr;
}

static inline
bool operator > (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr > b.ptr;
}

static inline
bool operator <= (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr <= b.ptr;
}

static inline
bool operator >= (const MatConstIterator& a, const MatConstIterator& b)
{
    return a.ptr >= b.ptr;
}

static inline
ptrdiff_t operator - (const MatConstIterator& b, const MatConstIterator& a)
{
    if( a.m != b.m )
        return ((size_t)(-1) >> 1);
    if( a.sliceEnd == b.sliceEnd )
        return (b.ptr - a.ptr)/static_cast<ptrdiff_t>(b.elemSize);

    return b.lpos() - a.lpos();
}

static inline
MatConstIterator operator + (const MatConstIterator& a, ptrdiff_t ofs)
{
    MatConstIterator b = a;
    return b += ofs;
}

static inline
MatConstIterator operator + (ptrdiff_t ofs, const MatConstIterator& a)
{
    MatConstIterator b = a;
    return b += ofs;
}

static inline
MatConstIterator operator - (const MatConstIterator& a, ptrdiff_t ofs)
{
    MatConstIterator b = a;
    return b += -ofs;
}


inline
const uchar* MatConstIterator::operator [](ptrdiff_t i) const
{
    return *(*this + i);
}



///////////////////////// MatConstIterator_ /////////////////////////

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_()
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m)
    : MatConstIterator(_m)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col)
    : MatConstIterator(_m, _row, _col)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m, Point _pt)
    : MatConstIterator(_m, _pt)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>::MatConstIterator_(const MatConstIterator_& it)
    : MatConstIterator(it)
{}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator = (const MatConstIterator_& it )
{
    MatConstIterator::operator = (it);
    return *this;
}

template<typename _Tp> inline
const _Tp& MatConstIterator_<_Tp>::operator *() const
{
    return *(_Tp*)(this->ptr);
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator += (ptrdiff_t ofs)
{
    MatConstIterator::operator += (ofs);
    return *this;
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator -= (ptrdiff_t ofs)
{
    return (*this += -ofs);
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator --()
{
    MatConstIterator::operator --();
    return *this;
}

template<typename _Tp> inline
MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator --(int)
{
    MatConstIterator_ b = *this;
    MatConstIterator::operator --();
    return b;
}

template<typename _Tp> inline
MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator ++()
{
    MatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator ++(int)
{
    MatConstIterator_ b = *this;
    MatConstIterator::operator ++();
    return b;
}


template<typename _Tp> inline
Point MatConstIterator_<_Tp>::pos() const
{
    if( !m )
        return Point();
    CV_DbgAssert( m->dims <= 2 );
    if( m->isContinuous() )
    {
        ptrdiff_t ofs = (const _Tp*)ptr - (const _Tp*)m->data;
        int y = (int)(ofs / m->cols);
        int x = (int)(ofs - (ptrdiff_t)y * m->cols);
        return Point(x, y);
    }
    else
    {
        ptrdiff_t ofs = (uchar*)ptr - m->data;
        int y = (int)(ofs / m->step);
        int x = (int)((ofs - y * m->step)/sizeof(_Tp));
        return Point(x, y);
    }
}


template<typename _Tp> static inline
bool operator == (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

template<typename _Tp> static inline
bool operator != (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{
    return a.m != b.m || a.ptr != b.ptr;
}

template<typename _Tp> static inline
MatConstIterator_<_Tp> operator + (const MatConstIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatConstIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatConstIterator_<_Tp> operator + (ptrdiff_t ofs, const MatConstIterator_<_Tp>& a)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatConstIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatConstIterator_<_Tp> operator - (const MatConstIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a - ofs;
    return (MatConstIterator_<_Tp>&)t;
}

template<typename _Tp> inline
const _Tp& MatConstIterator_<_Tp>::operator [](ptrdiff_t i) const
{
    return *(_Tp*)MatConstIterator::operator [](i);
}



//////////////////////////// MatIterator_ ///////////////////////////

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_()
    : MatConstIterator_<_Tp>()
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m)
    : MatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, int _row, int _col)
    : MatConstIterator_<_Tp>(_m, _row, _col)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, Point _pt)
    : MatConstIterator_<_Tp>(_m, _pt)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, const int* _idx)
    : MatConstIterator_<_Tp>(_m, _idx)
{}

template<typename _Tp> inline
MatIterator_<_Tp>::MatIterator_(const MatIterator_& it)
    : MatConstIterator_<_Tp>(it)
{}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator = (const MatIterator_<_Tp>& it )
{
    MatConstIterator::operator = (it);
    return *this;
}

template<typename _Tp> inline
_Tp& MatIterator_<_Tp>::operator *() const
{
    return *(_Tp*)(this->ptr);
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator += (ptrdiff_t ofs)
{
    MatConstIterator::operator += (ofs);
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator -= (ptrdiff_t ofs)
{
    MatConstIterator::operator += (-ofs);
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator --()
{
    MatConstIterator::operator --();
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp> MatIterator_<_Tp>::operator --(int)
{
    MatIterator_ b = *this;
    MatConstIterator::operator --();
    return b;
}

template<typename _Tp> inline
MatIterator_<_Tp>& MatIterator_<_Tp>::operator ++()
{
    MatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
MatIterator_<_Tp> MatIterator_<_Tp>::operator ++(int)
{
    MatIterator_ b = *this;
    MatConstIterator::operator ++();
    return b;
}

template<typename _Tp> inline
_Tp& MatIterator_<_Tp>::operator [](ptrdiff_t i) const
{
    return *(*this + i);
}


template<typename _Tp> static inline
bool operator == (const MatIterator_<_Tp>& a, const MatIterator_<_Tp>& b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

template<typename _Tp> static inline
bool operator != (const MatIterator_<_Tp>& a, const MatIterator_<_Tp>& b)
{
    return a.m != b.m || a.ptr != b.ptr;
}

template<typename _Tp> static inline
MatIterator_<_Tp> operator + (const MatIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatIterator_<_Tp> operator + (ptrdiff_t ofs, const MatIterator_<_Tp>& a)
{
    MatConstIterator t = (const MatConstIterator&)a + ofs;
    return (MatIterator_<_Tp>&)t;
}

template<typename _Tp> static inline
MatIterator_<_Tp> operator - (const MatIterator_<_Tp>& a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator&)a - ofs;
    return (MatIterator_<_Tp>&)t;
}



/////////////////////// SparseMatConstIterator //////////////////////

inline
SparseMatConstIterator::SparseMatConstIterator()
    : m(0), hashidx(0), ptr(0)
{}

inline
SparseMatConstIterator::SparseMatConstIterator(const SparseMatConstIterator& it)
    : m(it.m), hashidx(it.hashidx), ptr(it.ptr)
{}

inline SparseMatConstIterator& SparseMatConstIterator::operator = (const SparseMatConstIterator& it)
{
    if( this != &it )
    {
        m = it.m;
        hashidx = it.hashidx;
        ptr = it.ptr;
    }
    return *this;
}

template<typename _Tp> inline
const _Tp& SparseMatConstIterator::value() const
{
    return *(const _Tp*)ptr;
}

inline
const SparseMat::Node* SparseMatConstIterator::node() const
{
    return (ptr && m && m->hdr) ? (const SparseMat::Node*)(const void*)(ptr - m->hdr->valueOffset) : 0;
}

inline
SparseMatConstIterator SparseMatConstIterator::operator ++(int)
{
    SparseMatConstIterator it = *this;
    ++*this;
    return it;
}

inline
void SparseMatConstIterator::seekEnd()
{
    if( m && m->hdr )
    {
        hashidx = m->hdr->hashtab.size();
        ptr = 0;
    }
}


static inline
bool operator == (const SparseMatConstIterator& it1, const SparseMatConstIterator& it2)
{
    return it1.m == it2.m && it1.ptr == it2.ptr;
}

static inline
bool operator != (const SparseMatConstIterator& it1, const SparseMatConstIterator& it2)
{
    return !(it1 == it2);
}



///////////////////////// SparseMatIterator /////////////////////////

inline
SparseMatIterator::SparseMatIterator()
{}

inline
SparseMatIterator::SparseMatIterator(SparseMat* _m)
    : SparseMatConstIterator(_m)
{}

inline
SparseMatIterator::SparseMatIterator(const SparseMatIterator& it)
    : SparseMatConstIterator(it)
{}

inline
SparseMatIterator& SparseMatIterator::operator = (const SparseMatIterator& it)
{
    (SparseMatConstIterator&)*this = it;
    return *this;
}

template<typename _Tp> inline
_Tp& SparseMatIterator::value() const
{
    return *(_Tp*)ptr;
}

inline
SparseMat::Node* SparseMatIterator::node() const
{
    return (SparseMat::Node*)SparseMatConstIterator::node();
}

inline
SparseMatIterator& SparseMatIterator::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

inline
SparseMatIterator SparseMatIterator::operator ++(int)
{
    SparseMatIterator it = *this;
    ++*this;
    return it;
}



////////////////////// SparseMatConstIterator_ //////////////////////

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_()
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMat_<_Tp>* _m)
    : SparseMatConstIterator(_m)
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMat* _m)
    : SparseMatConstIterator(_m)
{
    CV_Assert( _m->type() == DataType<_Tp>::type );
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMatConstIterator_<_Tp>& it)
    : SparseMatConstIterator(it)
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>& SparseMatConstIterator_<_Tp>::operator = (const SparseMatConstIterator_<_Tp>& it)
{
    return reinterpret_cast<SparseMatConstIterator_<_Tp>&>
         (*reinterpret_cast<SparseMatConstIterator*>(this) =
           reinterpret_cast<const SparseMatConstIterator&>(it));
}

template<typename _Tp> inline
const _Tp& SparseMatConstIterator_<_Tp>::operator *() const
{
    return *(const _Tp*)this->ptr;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>& SparseMatConstIterator_<_Tp>::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp> SparseMatConstIterator_<_Tp>::operator ++(int)
{
    SparseMatConstIterator_<_Tp> it = *this;
    SparseMatConstIterator::operator ++();
    return it;
}



///////////////////////// SparseMatIterator_ ////////////////////////

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_()
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(SparseMat_<_Tp>* _m)
    : SparseMatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(SparseMat* _m)
    : SparseMatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(const SparseMatIterator_<_Tp>& it)
    : SparseMatConstIterator_<_Tp>(it)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>& SparseMatIterator_<_Tp>::operator = (const SparseMatIterator_<_Tp>& it)
{
    return reinterpret_cast<SparseMatIterator_<_Tp>&>
         (*reinterpret_cast<SparseMatConstIterator*>(this) =
           reinterpret_cast<const SparseMatConstIterator&>(it));
}

template<typename _Tp> inline
_Tp& SparseMatIterator_<_Tp>::operator *() const
{
    return *(_Tp*)this->ptr;
}

template<typename _Tp> inline
SparseMatIterator_<_Tp>& SparseMatIterator_<_Tp>::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline
SparseMatIterator_<_Tp> SparseMatIterator_<_Tp>::operator ++(int)
{
    SparseMatIterator_<_Tp> it = *this;
    SparseMatConstIterator::operator ++();
    return it;
}



//////////////////////// MatCommaInitializer_ ///////////////////////

template<typename _Tp> inline
MatCommaInitializer_<_Tp>::MatCommaInitializer_(Mat_<_Tp>* _m)
    : it(_m)
{}

template<typename _Tp> template<typename T2> inline
MatCommaInitializer_<_Tp>& MatCommaInitializer_<_Tp>::operator , (T2 v)
{
    CV_DbgAssert( this->it < ((const Mat_<_Tp>*)this->it.m)->end() );
    *this->it = _Tp(v);
    ++this->it;
    return *this;
}

template<typename _Tp> inline
MatCommaInitializer_<_Tp>::operator Mat_<_Tp>() const
{
    CV_DbgAssert( this->it == ((const Mat_<_Tp>*)this->it.m)->end() );
    return Mat_<_Tp>(*this->it.m);
}


template<typename _Tp, typename T2> static inline
MatCommaInitializer_<_Tp> operator << (const Mat_<_Tp>& m, T2 val)
{
    MatCommaInitializer_<_Tp> commaInitializer((Mat_<_Tp>*)&m);
    return (commaInitializer, val);
}



///////////////////////// Matrix Expressions ////////////////////////

inline
Mat& Mat::operator = (const MatExpr& e)
{
    e.op->assign(e, *this);
    return *this;
}

template<typename _Tp> inline
Mat_<_Tp>::Mat_(const MatExpr& e)
{
    e.op->assign(e, *this, DataType<_Tp>::type);
}

template<typename _Tp> inline
Mat_<_Tp>& Mat_<_Tp>::operator = (const MatExpr& e)
{
    e.op->assign(e, *this, DataType<_Tp>::type);
    return *this;
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::zeros(int rows, int cols)
{
    return Mat::zeros(rows, cols, DataType<_Tp>::type);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::zeros(Size sz)
{
    return Mat::zeros(sz, DataType<_Tp>::type);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::ones(int rows, int cols)
{
    return Mat::ones(rows, cols, DataType<_Tp>::type);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::ones(Size sz)
{
    return Mat::ones(sz, DataType<_Tp>::type);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::eye(int rows, int cols)
{
    return Mat::eye(rows, cols, DataType<_Tp>::type);
}

template<typename _Tp> inline
MatExpr Mat_<_Tp>::eye(Size sz)
{
    return Mat::eye(sz, DataType<_Tp>::type);
}

inline
MatExpr::MatExpr()
    : op(0), flags(0), a(Mat()), b(Mat()), c(Mat()), alpha(0), beta(0), s()
{}

inline
MatExpr::MatExpr(const MatOp* _op, int _flags, const Mat& _a, const Mat& _b,
                 const Mat& _c, double _alpha, double _beta, const Scalar& _s)
    : op(_op), flags(_flags), a(_a), b(_b), c(_c), alpha(_alpha), beta(_beta), s(_s)
{}

inline
MatExpr::operator Mat() const
{
    Mat m;
    op->assign(*this, m);
    return m;
}

template<typename _Tp> inline
MatExpr::operator Mat_<_Tp>() const
{
    Mat_<_Tp> m;
    op->assign(*this, m, DataType<_Tp>::type);
    return m;
}


template<typename _Tp> static inline
MatExpr min(const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    return cv::min((const Mat&)a, (const Mat&)b);
}

template<typename _Tp> static inline
MatExpr min(const Mat_<_Tp>& a, double s)
{
    return cv::min((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr min(double s, const Mat_<_Tp>& a)
{
    return cv::min((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr max(const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    return cv::max((const Mat&)a, (const Mat&)b);
}

template<typename _Tp> static inline
MatExpr max(const Mat_<_Tp>& a, double s)
{
    return cv::max((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr max(double s, const Mat_<_Tp>& a)
{
    return cv::max((const Mat&)a, s);
}

template<typename _Tp> static inline
MatExpr abs(const Mat_<_Tp>& m)
{
    return cv::abs((const Mat&)m);
}


static inline
Mat& operator += (Mat& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, a);
    return a;
}

static inline
const Mat& operator += (const Mat& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator += (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator += (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, (Mat&)a);
    return a;
}

static inline
Mat& operator -= (Mat& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, a);
    return a;
}

static inline
const Mat& operator -= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator -= (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator -= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, (Mat&)a);
    return a;
}

static inline
Mat& operator *= (Mat& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, a);
    return a;
}

static inline
const Mat& operator *= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator *= (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator *= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, (Mat&)a);
    return a;
}

static inline
Mat& operator /= (Mat& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, a);
    return a;
}

static inline
const Mat& operator /= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, (Mat&)a);
    return a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator /= (Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, a);
    return a;
}

template<typename _Tp> static inline
const Mat_<_Tp>& operator /= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, (Mat&)a);
    return a;
}


//////////////////////////////// UMat ////////////////////////////////

inline
UMat::UMat(UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{}

inline
UMat::UMat(int _rows, int _cols, int _type, UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{
    create(_rows, _cols, _type);
}

inline
UMat::UMat(int _rows, int _cols, int _type, const Scalar& _s, UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{
    create(_rows, _cols, _type);
    *this = _s;
}

inline
UMat::UMat(Size _sz, int _type, UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{
    create( _sz.height, _sz.width, _type );
}

inline
UMat::UMat(Size _sz, int _type, const Scalar& _s, UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{
    create(_sz.height, _sz.width, _type);
    *this = _s;
}

inline
UMat::UMat(int _dims, const int* _sz, int _type, UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{
    create(_dims, _sz, _type);
}

inline
UMat::UMat(int _dims, const int* _sz, int _type, const Scalar& _s, UMatUsageFlags _usageFlags)
: flags(MAGIC_VAL), dims(0), rows(0), cols(0), allocator(0), usageFlags(_usageFlags), u(0), offset(0), size(&rows)
{
    create(_dims, _sz, _type);
    *this = _s;
}

inline
UMat::UMat(const UMat& m)
: flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), allocator(m.allocator),
  usageFlags(m.usageFlags), u(m.u), offset(m.offset), size(&rows)
{
    addref();
    if( m.dims <= 2 )
    {
        step[0] = m.step[0]; step[1] = m.step[1];
    }
    else
    {
        dims = 0;
        copySize(m);
    }
}


template<typename _Tp> inline
UMat::UMat(const std::vector<_Tp>& vec, bool copyData)
: flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG), dims(2), rows((int)vec.size()),
cols(1), allocator(0), usageFlags(USAGE_DEFAULT), u(0), offset(0), size(&rows)
{
    if(vec.empty())
        return;
    if( !copyData )
    {
        // !!!TODO!!!
        CV_Error(Error::StsNotImplemented, "");
    }
    else
        Mat((int)vec.size(), 1, DataType<_Tp>::type, (uchar*)&vec[0]).copyTo(*this);
}

inline
UMat& UMat::operator = (const UMat& m)
{
    if( this != &m )
    {
        const_cast<UMat&>(m).addref();
        release();
        flags = m.flags;
        if( dims <= 2 && m.dims <= 2 )
        {
            dims = m.dims;
            rows = m.rows;
            cols = m.cols;
            step[0] = m.step[0];
            step[1] = m.step[1];
        }
        else
            copySize(m);
        allocator = m.allocator;
        if (usageFlags == USAGE_DEFAULT)
            usageFlags = m.usageFlags;
        u = m.u;
        offset = m.offset;
    }
    return *this;
}

inline
UMat UMat::row(int y) const
{
    return UMat(*this, Range(y, y + 1), Range::all());
}

inline
UMat UMat::col(int x) const
{
    return UMat(*this, Range::all(), Range(x, x + 1));
}

inline
UMat UMat::rowRange(int startrow, int endrow) const
{
    return UMat(*this, Range(startrow, endrow), Range::all());
}

inline
UMat UMat::rowRange(const Range& r) const
{
    return UMat(*this, r, Range::all());
}

inline
UMat UMat::colRange(int startcol, int endcol) const
{
    return UMat(*this, Range::all(), Range(startcol, endcol));
}

inline
UMat UMat::colRange(const Range& r) const
{
    return UMat(*this, Range::all(), r);
}

inline
UMat UMat::clone() const
{
    UMat m;
    copyTo(m);
    return m;
}

inline
void UMat::assignTo( UMat& m, int _type ) const
{
    if( _type < 0 )
        m = *this;
    else
        convertTo(m, _type);
}

inline
void UMat::create(int _rows, int _cols, int _type, UMatUsageFlags _usageFlags)
{
    _type &= TYPE_MASK;
    if( dims <= 2 && rows == _rows && cols == _cols && type() == _type && u )
        return;
    int sz[] = {_rows, _cols};
    create(2, sz, _type, _usageFlags);
}

inline
void UMat::create(Size _sz, int _type, UMatUsageFlags _usageFlags)
{
    create(_sz.height, _sz.width, _type, _usageFlags);
}

inline
void UMat::addref()
{
    if( u )
        CV_XADD(&(u->urefcount), 1);
}

inline void UMat::release()
{
    if( u && CV_XADD(&(u->urefcount), -1) == 1 )
        deallocate();
    for(int i = 0; i < dims; i++)
        size.p[i] = 0;
    u = 0;
}

inline
UMat UMat::operator()( Range _rowRange, Range _colRange ) const
{
    return UMat(*this, _rowRange, _colRange);
}

inline
UMat UMat::operator()( const Rect& roi ) const
{
    return UMat(*this, roi);
}

inline
UMat UMat::operator()(const Range* ranges) const
{
    return UMat(*this, ranges);
}

inline
UMat UMat::operator()(const std::vector<Range>& ranges) const
{
    return UMat(*this, ranges);
}

inline
bool UMat::isContinuous() const
{
    return (flags & CONTINUOUS_FLAG) != 0;
}

inline
bool UMat::isSubmatrix() const
{
    return (flags & SUBMATRIX_FLAG) != 0;
}

inline
size_t UMat::elemSize() const
{
    return dims > 0 ? step.p[dims - 1] : 0;
}

inline
size_t UMat::elemSize1() const
{
    return CV_ELEM_SIZE1(flags);
}

inline
int UMat::type() const
{
    return CV_MAT_TYPE(flags);
}

inline
int UMat::depth() const
{
    return CV_MAT_DEPTH(flags);
}

inline
int UMat::channels() const
{
    return CV_MAT_CN(flags);
}

inline
size_t UMat::step1(int i) const
{
    return step.p[i] / elemSize1();
}

inline
bool UMat::empty() const
{
    return u == 0 || total() == 0 || dims == 0;
}

inline
size_t UMat::total() const
{
    if( dims <= 2 )
        return (size_t)rows * cols;
    size_t p = 1;
    for( int i = 0; i < dims; i++ )
        p *= size[i];
    return p;
}

#ifdef CV_CXX_MOVE_SEMANTICS

inline
UMat::UMat(UMat&& m)
: flags(m.flags), dims(m.dims), rows(m.rows), cols(m.cols), allocator(m.allocator),
  usageFlags(m.usageFlags), u(m.u), offset(m.offset), size(&rows)
{
    if (m.dims <= 2)  // move new step/size info
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        CV_DbgAssert(m.step.p != m.step.buf);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL; m.dims = m.rows = m.cols = 0;
    m.allocator = NULL;
    m.u = NULL;
    m.offset = 0;
}

inline
UMat& UMat::operator = (UMat&& m)
{
    if (this == &m)
      return *this;
    release();
    flags = m.flags; dims = m.dims; rows = m.rows; cols = m.cols;
    allocator = m.allocator; usageFlags = m.usageFlags;
    u = m.u;
    offset = m.offset;
    if (step.p != step.buf) // release self step/size
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
    if (m.dims <= 2) // move new step/size info
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        CV_DbgAssert(m.step.p != m.step.buf);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL; m.dims = m.rows = m.cols = 0;
    m.allocator = NULL;
    m.u = NULL;
    m.offset = 0;
    return *this;
}

#endif


inline bool UMatData::hostCopyObsolete() const { return (flags & HOST_COPY_OBSOLETE) != 0; }
inline bool UMatData::deviceCopyObsolete() const { return (flags & DEVICE_COPY_OBSOLETE) != 0; }
inline bool UMatData::deviceMemMapped() const { return (flags & DEVICE_MEM_MAPPED) != 0; }
inline bool UMatData::copyOnMap() const { return (flags & COPY_ON_MAP) != 0; }
inline bool UMatData::tempUMat() const { return (flags & TEMP_UMAT) != 0; }
inline bool UMatData::tempCopiedUMat() const { return (flags & TEMP_COPIED_UMAT) == TEMP_COPIED_UMAT; }

inline void UMatData::markDeviceMemMapped(bool flag)
{
  if(flag)
    flags |= DEVICE_MEM_MAPPED;
  else
    flags &= ~DEVICE_MEM_MAPPED;
}

inline void UMatData::markHostCopyObsolete(bool flag)
{
    if(flag)
        flags |= HOST_COPY_OBSOLETE;
    else
        flags &= ~HOST_COPY_OBSOLETE;
}
inline void UMatData::markDeviceCopyObsolete(bool flag)
{
    if(flag)
        flags |= DEVICE_COPY_OBSOLETE;
    else
        flags &= ~DEVICE_COPY_OBSOLETE;
}

inline UMatDataAutoLock::UMatDataAutoLock(UMatData* _u) : u(_u) { u->lock(); }
inline UMatDataAutoLock::~UMatDataAutoLock() { u->unlock(); }

//! @endcond

} //cv

#ifdef _MSC_VER
#pragma warning( pop )
#endif

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_MATX_HPP
#define OPENCV_CORE_MATX_HPP

#ifndef __cplusplus
#  error matx.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/saturate.hpp"

#ifdef CV_CXX11
#include <initializer_list>
#endif

namespace cv
{

//! @addtogroup core_basic
//! @{

////////////////////////////// Small Matrix ///////////////////////////

//! @cond IGNORED
struct CV_EXPORTS Matx_AddOp {};
struct CV_EXPORTS Matx_SubOp {};
struct CV_EXPORTS Matx_ScaleOp {};
struct CV_EXPORTS Matx_MulOp {};
struct CV_EXPORTS Matx_DivOp {};
struct CV_EXPORTS Matx_MatMulOp {};
struct CV_EXPORTS Matx_TOp {};
//! @endcond

/** @brief Template class for small matrices whose type and size are known at compilation time

If you need a more flexible type, use Mat . The elements of the matrix M are accessible using the
M(i,j) notation. Most of the common matrix operations (see also @ref MatrixExpressions ) are
available. To do an operation on Matx that is not implemented, you can easily convert the matrix to
Mat and backwards:
@code{.cpp}
    Matx33f m(1, 2, 3,
              4, 5, 6,
              7, 8, 9);
    cout << sum(Mat(m*m.t())) << endl;
@endcode
Except of the plain constructor which takes a list of elements, Matx can be initialized from a C-array:
@code{.cpp}
    float values[] = { 1, 2, 3};
    Matx31f m(values);
@endcode
In case if C++11 features are avaliable, std::initializer_list can be also used to initizlize Matx:
@code{.cpp}
    Matx31f m = { 1, 2, 3};
@endcode
 */
template<typename _Tp, int m, int n> class Matx
{
public:
    enum { depth    = DataType<_Tp>::depth,
           rows     = m,
           cols     = n,
           channels = rows*cols,
           type     = CV_MAKETYPE(depth, channels),
           shortdim = (m < n ? m : n)
         };

    typedef _Tp                           value_type;
    typedef Matx<_Tp, m, n>               mat_type;
    typedef Matx<_Tp, shortdim, 1> diag_type;

    //! default constructor
    Matx();

    Matx(_Tp v0); //!< 1x1 matrix
    Matx(_Tp v0, _Tp v1); //!< 1x2 or 2x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2); //!< 1x3 or 3x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 1x4, 2x2 or 4x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 1x5 or 5x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 1x6, 2x3, 3x2 or 6x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 1x7 or 7x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 1x8, 2x4, 4x2 or 8x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 1x9, 3x3 or 9x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 1x10, 2x5 or 5x2 or 10x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11); //!< 1x12, 2x6, 3x4, 4x3, 6x2 or 12x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11,
         _Tp v12, _Tp v13); //!< 1x14, 2x7, 7x2 or 14x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11,
         _Tp v12, _Tp v13, _Tp v14, _Tp v15); //!< 1x16, 4x4 or 16x1 matrix
    explicit Matx(const _Tp* vals); //!< initialize from a plain array

#ifdef CV_CXX11
    Matx(std::initializer_list<_Tp>); //!< initialize from an initializer list
#endif

    static Matx all(_Tp alpha);
    static Matx zeros();
    static Matx ones();
    static Matx eye();
    static Matx diag(const diag_type& d);
    static Matx randu(_Tp a, _Tp b);
    static Matx randn(_Tp a, _Tp b);

    //! dot product computed with the default precision
    _Tp dot(const Matx<_Tp, m, n>& v) const;

    //! dot product computed in double-precision arithmetics
    double ddot(const Matx<_Tp, m, n>& v) const;

    //! conversion to another data type
    template<typename T2> operator Matx<T2, m, n>() const;

    //! change the matrix shape
    template<int m1, int n1> Matx<_Tp, m1, n1> reshape() const;

    //! extract part of the matrix
    template<int m1, int n1> Matx<_Tp, m1, n1> get_minor(int i, int j) const;

    //! extract the matrix row
    Matx<_Tp, 1, n> row(int i) const;

    //! extract the matrix column
    Matx<_Tp, m, 1> col(int i) const;

    //! extract the matrix diagonal
    diag_type diag() const;

    //! transpose the matrix
    Matx<_Tp, n, m> t() const;

    //! invert the matrix
    Matx<_Tp, n, m> inv(int method=DECOMP_LU, bool *p_is_ok = NULL) const;

    //! solve linear system
    template<int l> Matx<_Tp, n, l> solve(const Matx<_Tp, m, l>& rhs, int flags=DECOMP_LU) const;
    Vec<_Tp, n> solve(const Vec<_Tp, m>& rhs, int method) const;

    //! multiply two matrices element-wise
    Matx<_Tp, m, n> mul(const Matx<_Tp, m, n>& a) const;

    //! divide two matrices element-wise
    Matx<_Tp, m, n> div(const Matx<_Tp, m, n>& a) const;

    //! element access
    const _Tp& operator ()(int i, int j) const;
    _Tp& operator ()(int i, int j);

    //! 1D element access
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);

    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_AddOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_SubOp);
    template<typename _T2> Matx(const Matx<_Tp, m, n>& a, _T2 alpha, Matx_ScaleOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_MulOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_DivOp);
    template<int l> Matx(const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b, Matx_MatMulOp);
    Matx(const Matx<_Tp, n, m>& a, Matx_TOp);

    _Tp val[m*n]; //< matrix elements
};

typedef Matx<float, 1, 2> Matx12f;
typedef Matx<double, 1, 2> Matx12d;
typedef Matx<float, 1, 3> Matx13f;
typedef Matx<double, 1, 3> Matx13d;
typedef Matx<float, 1, 4> Matx14f;
typedef Matx<double, 1, 4> Matx14d;
typedef Matx<float, 1, 6> Matx16f;
typedef Matx<double, 1, 6> Matx16d;

typedef Matx<float, 2, 1> Matx21f;
typedef Matx<double, 2, 1> Matx21d;
typedef Matx<float, 3, 1> Matx31f;
typedef Matx<double, 3, 1> Matx31d;
typedef Matx<float, 4, 1> Matx41f;
typedef Matx<double, 4, 1> Matx41d;
typedef Matx<float, 6, 1> Matx61f;
typedef Matx<double, 6, 1> Matx61d;

typedef Matx<float, 2, 2> Matx22f;
typedef Matx<double, 2, 2> Matx22d;
typedef Matx<float, 2, 3> Matx23f;
typedef Matx<double, 2, 3> Matx23d;
typedef Matx<float, 3, 2> Matx32f;
typedef Matx<double, 3, 2> Matx32d;

typedef Matx<float, 3, 3> Matx33f;
typedef Matx<double, 3, 3> Matx33d;

typedef Matx<float, 3, 4> Matx34f;
typedef Matx<double, 3, 4> Matx34d;
typedef Matx<float, 4, 3> Matx43f;
typedef Matx<double, 4, 3> Matx43d;

typedef Matx<float, 4, 4> Matx44f;
typedef Matx<double, 4, 4> Matx44d;
typedef Matx<float, 6, 6> Matx66f;
typedef Matx<double, 6, 6> Matx66d;

/*!
  traits
*/
template<typename _Tp, int m, int n> class DataType< Matx<_Tp, m, n> >
{
public:
    typedef Matx<_Tp, m, n>                               value_type;
    typedef Matx<typename DataType<_Tp>::work_type, m, n> work_type;
    typedef _Tp                                           channel_type;
    typedef value_type                                    vec_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = m * n,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };
};

/** @brief  Comma-separated Matrix Initializer
*/
template<typename _Tp, int m, int n> class MatxCommaInitializer
{
public:
    MatxCommaInitializer(Matx<_Tp, m, n>* _mtx);
    template<typename T2> MatxCommaInitializer<_Tp, m, n>& operator , (T2 val);
    Matx<_Tp, m, n> operator *() const;

    Matx<_Tp, m, n>* dst;
    int idx;
};

/*
 Utility methods
*/
template<typename _Tp, int m> static double determinant(const Matx<_Tp, m, m>& a);
template<typename _Tp, int m, int n> static double trace(const Matx<_Tp, m, n>& a);
template<typename _Tp, int m, int n> static double norm(const Matx<_Tp, m, n>& M);
template<typename _Tp, int m, int n> static double norm(const Matx<_Tp, m, n>& M, int normType);



/////////////////////// Vec (used as element of multi-channel images /////////////////////

/** @brief Template class for short numerical vectors, a partial case of Matx

This template class represents short numerical vectors (of 1, 2, 3, 4 ... elements) on which you
can perform basic arithmetical operations, access individual elements using [] operator etc. The
vectors are allocated on stack, as opposite to std::valarray, std::vector, cv::Mat etc., which
elements are dynamically allocated in the heap.

The template takes 2 parameters:
@tparam _Tp element type
@tparam cn the number of elements

In addition to the universal notation like Vec<float, 3>, you can use shorter aliases
for the most popular specialized variants of Vec, e.g. Vec3f ~ Vec<float, 3>.

It is possible to convert Vec\<T,2\> to/from Point_, Vec\<T,3\> to/from Point3_ , and Vec\<T,4\>
to CvScalar or Scalar_. Use operator[] to access the elements of Vec.

All the expected vector operations are also implemented:
-   v1 = v2 + v3
-   v1 = v2 - v3
-   v1 = v2 \* scale
-   v1 = scale \* v2
-   v1 = -v2
-   v1 += v2 and other augmenting operations
-   v1 == v2, v1 != v2
-   norm(v1) (euclidean norm)
The Vec class is commonly used to describe pixel types of multi-channel arrays. See Mat for details.
*/
template<typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1>
{
public:
    typedef _Tp value_type;
    enum { depth    = Matx<_Tp, cn, 1>::depth,
           channels = cn,
           type     = CV_MAKETYPE(depth, channels)
         };

    //! default constructor
    Vec();

    Vec(_Tp v0); //!< 1-element vector constructor
    Vec(_Tp v0, _Tp v1); //!< 2-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2); //!< 3-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 4-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 5-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 6-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 7-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 8-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 9-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 10-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13); //!< 14-element vector constructor
    explicit Vec(const _Tp* values);

#ifdef CV_CXX11
    Vec(std::initializer_list<_Tp>);
#endif

    Vec(const Vec<_Tp, cn>& v);

    static Vec all(_Tp alpha);

    //! per-element multiplication
    Vec mul(const Vec<_Tp, cn>& v) const;

    //! conjugation (makes sense for complex numbers and quaternions)
    Vec conj() const;

    /*!
      cross product of the two 3D vectors.

      For other dimensionalities the exception is raised
    */
    Vec cross(const Vec& v) const;
    //! conversion to another data type
    template<typename T2> operator Vec<T2, cn>() const;

    /*! element access */
    const _Tp& operator [](int i) const;
    _Tp& operator[](int i);
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);

    Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_AddOp);
    Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_SubOp);
    template<typename _T2> Vec(const Matx<_Tp, cn, 1>& a, _T2 alpha, Matx_ScaleOp);
};

/** @name Shorter aliases for the most popular specializations of Vec<T,n>
  @{
*/
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;

typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;

typedef Vec<ushort, 2> Vec2w;
typedef Vec<ushort, 3> Vec3w;
typedef Vec<ushort, 4> Vec4w;

typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;
typedef Vec<int, 6> Vec6i;
typedef Vec<int, 8> Vec8i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;
/** @} */

/*!
  traits
*/
template<typename _Tp, int cn> class DataType< Vec<_Tp, cn> >
{
public:
    typedef Vec<_Tp, cn>                               value_type;
    typedef Vec<typename DataType<_Tp>::work_type, cn> work_type;
    typedef _Tp                                        channel_type;
    typedef value_type                                 vec_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = cn,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };
};

/** @brief  Comma-separated Vec Initializer
*/
template<typename _Tp, int m> class VecCommaInitializer : public MatxCommaInitializer<_Tp, m, 1>
{
public:
    VecCommaInitializer(Vec<_Tp, m>* _vec);
    template<typename T2> VecCommaInitializer<_Tp, m>& operator , (T2 val);
    Vec<_Tp, m> operator *() const;
};

template<typename _Tp, int cn> static Vec<_Tp, cn> normalize(const Vec<_Tp, cn>& v);

//! @} core_basic

//! @cond IGNORED

///////////////////////////////////// helper classes /////////////////////////////////////
namespace internal
{

template<typename _Tp, int m> struct Matx_DetOp
{
    double operator ()(const Matx<_Tp, m, m>& a) const
    {
        Matx<_Tp, m, m> temp = a;
        double p = LU(temp.val, m*sizeof(_Tp), m, 0, 0, 0);
        if( p == 0 )
            return p;
        for( int i = 0; i < m; i++ )
            p *= temp(i, i);
        return p;
    }
};

template<typename _Tp> struct Matx_DetOp<_Tp, 1>
{
    double operator ()(const Matx<_Tp, 1, 1>& a) const
    {
        return a(0,0);
    }
};

template<typename _Tp> struct Matx_DetOp<_Tp, 2>
{
    double operator ()(const Matx<_Tp, 2, 2>& a) const
    {
        return a(0,0)*a(1,1) - a(0,1)*a(1,0);
    }
};

template<typename _Tp> struct Matx_DetOp<_Tp, 3>
{
    double operator ()(const Matx<_Tp, 3, 3>& a) const
    {
        return a(0,0)*(a(1,1)*a(2,2) - a(2,1)*a(1,2)) -
            a(0,1)*(a(1,0)*a(2,2) - a(2,0)*a(1,2)) +
            a(0,2)*(a(1,0)*a(2,1) - a(2,0)*a(1,1));
    }
};

template<typename _Tp> Vec<_Tp, 2> inline conjugate(const Vec<_Tp, 2>& v)
{
    return Vec<_Tp, 2>(v[0], -v[1]);
}

template<typename _Tp> Vec<_Tp, 4> inline conjugate(const Vec<_Tp, 4>& v)
{
    return Vec<_Tp, 4>(v[0], -v[1], -v[2], -v[3]);
}

} // internal



////////////////////////////////// Matx Implementation ///////////////////////////////////

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx()
{
    for(int i = 0; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0)
{
    val[0] = v0;
    for(int i = 1; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1)
{
    CV_StaticAssert(channels >= 2, "Matx should have at least 2 elements.");
    val[0] = v0; val[1] = v1;
    for(int i = 2; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2)
{
    CV_StaticAssert(channels >= 3, "Matx should have at least 3 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2;
    for(int i = 3; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
{
    CV_StaticAssert(channels >= 4, "Matx should have at least 4 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    for(int i = 4; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4)
{
    CV_StaticAssert(channels >= 5, "Matx should have at least 5 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3; val[4] = v4;
    for(int i = 5; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5)
{
    CV_StaticAssert(channels >= 6, "Matx should have at least 6 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5;
    for(int i = 6; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6)
{
    CV_StaticAssert(channels >= 7, "Matx should have at least 7 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6;
    for(int i = 7; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7)
{
    CV_StaticAssert(channels >= 8, "Matx should have at least 8 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
    for(int i = 8; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8)
{
    CV_StaticAssert(channels >= 9, "Matx should have at least 9 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
    val[8] = v8;
    for(int i = 9; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9)
{
    CV_StaticAssert(channels >= 10, "Matx should have at least 10 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
    val[8] = v8; val[9] = v9;
    for(int i = 10; i < channels; i++) val[i] = _Tp(0);
}


template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11)
{
    CV_StaticAssert(channels >= 12, "Matx should have at least 12 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
    val[8] = v8; val[9] = v9; val[10] = v10; val[11] = v11;
    for(int i = 12; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13)
{
    CV_StaticAssert(channels >= 14, "Matx should have at least 14 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
    val[8] = v8; val[9] = v9; val[10] = v10; val[11] = v11;
    val[12] = v12; val[13] = v13;
    for (int i = 14; i < channels; i++) val[i] = _Tp(0);
}


template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13, _Tp v14, _Tp v15)
{
    CV_StaticAssert(channels >= 16, "Matx should have at least 16 elements.");
    val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    val[4] = v4; val[5] = v5; val[6] = v6; val[7] = v7;
    val[8] = v8; val[9] = v9; val[10] = v10; val[11] = v11;
    val[12] = v12; val[13] = v13; val[14] = v14; val[15] = v15;
    for(int i = 16; i < channels; i++) val[i] = _Tp(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(const _Tp* values)
{
    for( int i = 0; i < channels; i++ ) val[i] = values[i];
}

#ifdef CV_CXX11
template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n>::Matx(std::initializer_list<_Tp> list)
{
    CV_DbgAssert(list.size() == channels);
    int i = 0;
    for(const auto& elem : list)
    {
        val[i++] = elem;
    }
}
#endif

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n> Matx<_Tp, m, n>::all(_Tp alpha)
{
    Matx<_Tp, m, n> M;
    for( int i = 0; i < m*n; i++ ) M.val[i] = alpha;
    return M;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::zeros()
{
    return all(0);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::ones()
{
    return all(1);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::eye()
{
    Matx<_Tp,m,n> M;
    for(int i = 0; i < shortdim; i++)
        M(i,i) = 1;
    return M;
}

template<typename _Tp, int m, int n> inline
_Tp Matx<_Tp, m, n>::dot(const Matx<_Tp, m, n>& M) const
{
    _Tp s = 0;
    for( int i = 0; i < channels; i++ ) s += val[i]*M.val[i];
    return s;
}

template<typename _Tp, int m, int n> inline
double Matx<_Tp, m, n>::ddot(const Matx<_Tp, m, n>& M) const
{
    double s = 0;
    for( int i = 0; i < channels; i++ ) s += (double)val[i]*M.val[i];
    return s;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::diag(const typename Matx<_Tp,m,n>::diag_type& d)
{
    Matx<_Tp,m,n> M;
    for(int i = 0; i < shortdim; i++)
        M(i,i) = d(i, 0);
    return M;
}

template<typename _Tp, int m, int n> template<typename T2>
inline Matx<_Tp, m, n>::operator Matx<T2, m, n>() const
{
    Matx<T2, m, n> M;
    for( int i = 0; i < m*n; i++ ) M.val[i] = saturate_cast<T2>(val[i]);
    return M;
}

template<typename _Tp, int m, int n> template<int m1, int n1> inline
Matx<_Tp, m1, n1> Matx<_Tp, m, n>::reshape() const
{
    CV_StaticAssert(m1*n1 == m*n, "Input and destnarion matrices must have the same number of elements");
    return (const Matx<_Tp, m1, n1>&)*this;
}

template<typename _Tp, int m, int n>
template<int m1, int n1> inline
Matx<_Tp, m1, n1> Matx<_Tp, m, n>::get_minor(int i, int j) const
{
    CV_DbgAssert(0 <= i && i+m1 <= m && 0 <= j && j+n1 <= n);
    Matx<_Tp, m1, n1> s;
    for( int di = 0; di < m1; di++ )
        for( int dj = 0; dj < n1; dj++ )
            s(di, dj) = (*this)(i+di, j+dj);
    return s;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, 1, n> Matx<_Tp, m, n>::row(int i) const
{
    CV_DbgAssert((unsigned)i < (unsigned)m);
    return Matx<_Tp, 1, n>(&val[i*n]);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, 1> Matx<_Tp, m, n>::col(int j) const
{
    CV_DbgAssert((unsigned)j < (unsigned)n);
    Matx<_Tp, m, 1> v;
    for( int i = 0; i < m; i++ )
        v.val[i] = val[i*n + j];
    return v;
}

template<typename _Tp, int m, int n> inline
typename Matx<_Tp, m, n>::diag_type Matx<_Tp, m, n>::diag() const
{
    diag_type d;
    for( int i = 0; i < shortdim; i++ )
        d.val[i] = val[i*n + i];
    return d;
}

template<typename _Tp, int m, int n> inline
const _Tp& Matx<_Tp, m, n>::operator()(int i, int j) const
{
    CV_DbgAssert( (unsigned)i < (unsigned)m && (unsigned)j < (unsigned)n );
    return this->val[i*n + j];
}

template<typename _Tp, int m, int n> inline
_Tp& Matx<_Tp, m, n>::operator ()(int i, int j)
{
    CV_DbgAssert( (unsigned)i < (unsigned)m && (unsigned)j < (unsigned)n );
    return val[i*n + j];
}

template<typename _Tp, int m, int n> inline
const _Tp& Matx<_Tp, m, n>::operator ()(int i) const
{
    CV_StaticAssert(m == 1 || n == 1, "Single index indexation requires matrix to be a column or a row");
    CV_DbgAssert( (unsigned)i < (unsigned)(m+n-1) );
    return val[i];
}

template<typename _Tp, int m, int n> inline
_Tp& Matx<_Tp, m, n>::operator ()(int i)
{
    CV_StaticAssert(m == 1 || n == 1, "Single index indexation requires matrix to be a column or a row");
    CV_DbgAssert( (unsigned)i < (unsigned)(m+n-1) );
    return val[i];
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_AddOp)
{
    for( int i = 0; i < channels; i++ )
        val[i] = saturate_cast<_Tp>(a.val[i] + b.val[i]);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_SubOp)
{
    for( int i = 0; i < channels; i++ )
        val[i] = saturate_cast<_Tp>(a.val[i] - b.val[i]);
}

template<typename _Tp, int m, int n> template<typename _T2> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, m, n>& a, _T2 alpha, Matx_ScaleOp)
{
    for( int i = 0; i < channels; i++ )
        val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_MulOp)
{
    for( int i = 0; i < channels; i++ )
        val[i] = saturate_cast<_Tp>(a.val[i] * b.val[i]);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_DivOp)
{
    for( int i = 0; i < channels; i++ )
        val[i] = saturate_cast<_Tp>(a.val[i] / b.val[i]);
}

template<typename _Tp, int m, int n> template<int l> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b, Matx_MatMulOp)
{
    for( int i = 0; i < m; i++ )
        for( int j = 0; j < n; j++ )
        {
            _Tp s = 0;
            for( int k = 0; k < l; k++ )
                s += a(i, k) * b(k, j);
            val[i*n + j] = s;
        }
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n>::Matx(const Matx<_Tp, n, m>& a, Matx_TOp)
{
    for( int i = 0; i < m; i++ )
        for( int j = 0; j < n; j++ )
            val[i*n + j] = a(j, i);
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n> Matx<_Tp, m, n>::mul(const Matx<_Tp, m, n>& a) const
{
    return Matx<_Tp, m, n>(*this, a, Matx_MulOp());
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n> Matx<_Tp, m, n>::div(const Matx<_Tp, m, n>& a) const
{
    return Matx<_Tp, m, n>(*this, a, Matx_DivOp());
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, n, m> Matx<_Tp, m, n>::t() const
{
    return Matx<_Tp, n, m>(*this, Matx_TOp());
}

template<typename _Tp, int m, int n> inline
Vec<_Tp, n> Matx<_Tp, m, n>::solve(const Vec<_Tp, m>& rhs, int method) const
{
    Matx<_Tp, n, 1> x = solve((const Matx<_Tp, m, 1>&)(rhs), method);
    return (Vec<_Tp, n>&)(x);
}

template<typename _Tp, int m> static inline
double determinant(const Matx<_Tp, m, m>& a)
{
    return cv::internal::Matx_DetOp<_Tp, m>()(a);
}

template<typename _Tp, int m, int n> static inline
double trace(const Matx<_Tp, m, n>& a)
{
    _Tp s = 0;
    for( int i = 0; i < std::min(m, n); i++ )
        s += a(i,i);
    return s;
}

template<typename _Tp, int m, int n> static inline
double norm(const Matx<_Tp, m, n>& M)
{
    return std::sqrt(normL2Sqr<_Tp, double>(M.val, m*n));
}

template<typename _Tp, int m, int n> static inline
double norm(const Matx<_Tp, m, n>& M, int normType)
{
    switch(normType) {
    case NORM_INF:
        return (double)normInf<_Tp, typename DataType<_Tp>::work_type>(M.val, m*n);
    case NORM_L1:
        return (double)normL1<_Tp, typename DataType<_Tp>::work_type>(M.val, m*n);
    case NORM_L2SQR:
        return (double)normL2Sqr<_Tp, typename DataType<_Tp>::work_type>(M.val, m*n);
    default:
    case NORM_L2:
        return std::sqrt((double)normL2Sqr<_Tp, typename DataType<_Tp>::work_type>(M.val, m*n));
    }
}



//////////////////////////////// matx comma initializer //////////////////////////////////

template<typename _Tp, typename _T2, int m, int n> static inline
MatxCommaInitializer<_Tp, m, n> operator << (const Matx<_Tp, m, n>& mtx, _T2 val)
{
    MatxCommaInitializer<_Tp, m, n> commaInitializer((Matx<_Tp, m, n>*)&mtx);
    return (commaInitializer, val);
}

template<typename _Tp, int m, int n> inline
MatxCommaInitializer<_Tp, m, n>::MatxCommaInitializer(Matx<_Tp, m, n>* _mtx)
    : dst(_mtx), idx(0)
{}

template<typename _Tp, int m, int n> template<typename _T2> inline
MatxCommaInitializer<_Tp, m, n>& MatxCommaInitializer<_Tp, m, n>::operator , (_T2 value)
{
    CV_DbgAssert( idx < m*n );
    dst->val[idx++] = saturate_cast<_Tp>(value);
    return *this;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, m, n> MatxCommaInitializer<_Tp, m, n>::operator *() const
{
    CV_DbgAssert( idx == n*m );
    return *dst;
}



/////////////////////////////////// Vec Implementation ///////////////////////////////////

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec() {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0)
    : Matx<_Tp, cn, 1>(v0) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1)
    : Matx<_Tp, cn, 1>(v0, v1) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2)
    : Matx<_Tp, cn, 1>(v0, v1, v2) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7, v8) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(const _Tp* values)
    : Matx<_Tp, cn, 1>(values) {}

#ifdef CV_CXX11
template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(std::initializer_list<_Tp> list)
    : Matx<_Tp, cn, 1>(list) {}
#endif

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(const Vec<_Tp, cn>& m)
    : Matx<_Tp, cn, 1>(m.val) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_AddOp op)
    : Matx<_Tp, cn, 1>(a, b, op) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn>::Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_SubOp op)
    : Matx<_Tp, cn, 1>(a, b, op) {}

template<typename _Tp, int cn> template<typename _T2> inline
Vec<_Tp, cn>::Vec(const Matx<_Tp, cn, 1>& a, _T2 alpha, Matx_ScaleOp op)
    : Matx<_Tp, cn, 1>(a, alpha, op) {}

template<typename _Tp, int cn> inline
Vec<_Tp, cn> Vec<_Tp, cn>::all(_Tp alpha)
{
    Vec v;
    for( int i = 0; i < cn; i++ ) v.val[i] = alpha;
    return v;
}

template<typename _Tp, int cn> inline
Vec<_Tp, cn> Vec<_Tp, cn>::mul(const Vec<_Tp, cn>& v) const
{
    Vec<_Tp, cn> w;
    for( int i = 0; i < cn; i++ ) w.val[i] = saturate_cast<_Tp>(this->val[i]*v.val[i]);
    return w;
}

template<> inline
Vec<float, 2> Vec<float, 2>::conj() const
{
    return cv::internal::conjugate(*this);
}

template<> inline
Vec<double, 2> Vec<double, 2>::conj() const
{
    return cv::internal::conjugate(*this);
}

template<> inline
Vec<float, 4> Vec<float, 4>::conj() const
{
    return cv::internal::conjugate(*this);
}

template<> inline
Vec<double, 4> Vec<double, 4>::conj() const
{
    return cv::internal::conjugate(*this);
}

template<typename _Tp, int cn> inline
Vec<_Tp, cn> Vec<_Tp, cn>::cross(const Vec<_Tp, cn>&) const
{
    CV_StaticAssert(cn == 3, "for arbitrary-size vector there is no cross-product defined");
    return Vec<_Tp, cn>();
}

template<> inline
Vec<float, 3> Vec<float, 3>::cross(const Vec<float, 3>& v) const
{
    return Vec<float,3>(this->val[1]*v.val[2] - this->val[2]*v.val[1],
                     this->val[2]*v.val[0] - this->val[0]*v.val[2],
                     this->val[0]*v.val[1] - this->val[1]*v.val[0]);
}

template<> inline
Vec<double, 3> Vec<double, 3>::cross(const Vec<double, 3>& v) const
{
    return Vec<double,3>(this->val[1]*v.val[2] - this->val[2]*v.val[1],
                     this->val[2]*v.val[0] - this->val[0]*v.val[2],
                     this->val[0]*v.val[1] - this->val[1]*v.val[0]);
}

template<typename _Tp, int cn> template<typename T2> inline
Vec<_Tp, cn>::operator Vec<T2, cn>() const
{
    Vec<T2, cn> v;
    for( int i = 0; i < cn; i++ ) v.val[i] = saturate_cast<T2>(this->val[i]);
    return v;
}

template<typename _Tp, int cn> inline
const _Tp& Vec<_Tp, cn>::operator [](int i) const
{
    CV_DbgAssert( (unsigned)i < (unsigned)cn );
    return this->val[i];
}

template<typename _Tp, int cn> inline
_Tp& Vec<_Tp, cn>::operator [](int i)
{
    CV_DbgAssert( (unsigned)i < (unsigned)cn );
    return this->val[i];
}

template<typename _Tp, int cn> inline
const _Tp& Vec<_Tp, cn>::operator ()(int i) const
{
    CV_DbgAssert( (unsigned)i < (unsigned)cn );
    return this->val[i];
}

template<typename _Tp, int cn> inline
_Tp& Vec<_Tp, cn>::operator ()(int i)
{
    CV_DbgAssert( (unsigned)i < (unsigned)cn );
    return this->val[i];
}

template<typename _Tp, int cn> inline
Vec<_Tp, cn> normalize(const Vec<_Tp, cn>& v)
{
    double nv = norm(v);
    return v * (nv ? 1./nv : 0.);
}



//////////////////////////////// vec comma initializer //////////////////////////////////


template<typename _Tp, typename _T2, int cn> static inline
VecCommaInitializer<_Tp, cn> operator << (const Vec<_Tp, cn>& vec, _T2 val)
{
    VecCommaInitializer<_Tp, cn> commaInitializer((Vec<_Tp, cn>*)&vec);
    return (commaInitializer, val);
}

template<typename _Tp, int cn> inline
VecCommaInitializer<_Tp, cn>::VecCommaInitializer(Vec<_Tp, cn>* _vec)
    : MatxCommaInitializer<_Tp, cn, 1>(_vec)
{}

template<typename _Tp, int cn> template<typename _T2> inline
VecCommaInitializer<_Tp, cn>& VecCommaInitializer<_Tp, cn>::operator , (_T2 value)
{
    CV_DbgAssert( this->idx < cn );
    this->dst->val[this->idx++] = saturate_cast<_Tp>(value);
    return *this;
}

template<typename _Tp, int cn> inline
Vec<_Tp, cn> VecCommaInitializer<_Tp, cn>::operator *() const
{
    CV_DbgAssert( this->idx == cn );
    return *this->dst;
}

//! @endcond

///////////////////////////// Matx out-of-class operators ////////////////////////////////

//! @relates cv::Matx
//! @{

template<typename _Tp1, typename _Tp2, int m, int n> static inline
Matx<_Tp1, m, n>& operator += (Matx<_Tp1, m, n>& a, const Matx<_Tp2, m, n>& b)
{
    for( int i = 0; i < m*n; i++ )
        a.val[i] = saturate_cast<_Tp1>(a.val[i] + b.val[i]);
    return a;
}

template<typename _Tp1, typename _Tp2, int m, int n> static inline
Matx<_Tp1, m, n>& operator -= (Matx<_Tp1, m, n>& a, const Matx<_Tp2, m, n>& b)
{
    for( int i = 0; i < m*n; i++ )
        a.val[i] = saturate_cast<_Tp1>(a.val[i] - b.val[i]);
    return a;
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator + (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b)
{
    return Matx<_Tp, m, n>(a, b, Matx_AddOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator - (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b)
{
    return Matx<_Tp, m, n>(a, b, Matx_SubOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator *= (Matx<_Tp, m, n>& a, int alpha)
{
    for( int i = 0; i < m*n; i++ )
        a.val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
    return a;
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator *= (Matx<_Tp, m, n>& a, float alpha)
{
    for( int i = 0; i < m*n; i++ )
        a.val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
    return a;
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator *= (Matx<_Tp, m, n>& a, double alpha)
{
    for( int i = 0; i < m*n; i++ )
        a.val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
    return a;
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, n>& a, int alpha)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, n>& a, float alpha)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, n>& a, double alpha)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (int alpha, const Matx<_Tp, m, n>& a)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (float alpha, const Matx<_Tp, m, n>& a)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (double alpha, const Matx<_Tp, m, n>& a)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator - (const Matx<_Tp, m, n>& a)
{
    return Matx<_Tp, m, n>(a, -1, Matx_ScaleOp());
}

template<typename _Tp, int m, int n, int l> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b)
{
    return Matx<_Tp, m, n>(a, b, Matx_MatMulOp());
}

template<typename _Tp, int m, int n> static inline
Vec<_Tp, m> operator * (const Matx<_Tp, m, n>& a, const Vec<_Tp, n>& b)
{
    Matx<_Tp, m, 1> c(a, b, Matx_MatMulOp());
    return (const Vec<_Tp, m>&)(c);
}

template<typename _Tp, int m, int n> static inline
bool operator == (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b)
{
    for( int i = 0; i < m*n; i++ )
        if( a.val[i] != b.val[i] ) return false;
    return true;
}

template<typename _Tp, int m, int n> static inline
bool operator != (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b)
{
    return !(a == b);
}

//! @}

////////////////////////////// Vec out-of-class operators ////////////////////////////////

//! @relates cv::Vec
//! @{

template<typename _Tp1, typename _Tp2, int cn> static inline
Vec<_Tp1, cn>& operator += (Vec<_Tp1, cn>& a, const Vec<_Tp2, cn>& b)
{
    for( int i = 0; i < cn; i++ )
        a.val[i] = saturate_cast<_Tp1>(a.val[i] + b.val[i]);
    return a;
}

template<typename _Tp1, typename _Tp2, int cn> static inline
Vec<_Tp1, cn>& operator -= (Vec<_Tp1, cn>& a, const Vec<_Tp2, cn>& b)
{
    for( int i = 0; i < cn; i++ )
        a.val[i] = saturate_cast<_Tp1>(a.val[i] - b.val[i]);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator + (const Vec<_Tp, cn>& a, const Vec<_Tp, cn>& b)
{
    return Vec<_Tp, cn>(a, b, Matx_AddOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator - (const Vec<_Tp, cn>& a, const Vec<_Tp, cn>& b)
{
    return Vec<_Tp, cn>(a, b, Matx_SubOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator *= (Vec<_Tp, cn>& a, int alpha)
{
    for( int i = 0; i < cn; i++ )
        a[i] = saturate_cast<_Tp>(a[i]*alpha);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator *= (Vec<_Tp, cn>& a, float alpha)
{
    for( int i = 0; i < cn; i++ )
        a[i] = saturate_cast<_Tp>(a[i]*alpha);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator *= (Vec<_Tp, cn>& a, double alpha)
{
    for( int i = 0; i < cn; i++ )
        a[i] = saturate_cast<_Tp>(a[i]*alpha);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator /= (Vec<_Tp, cn>& a, int alpha)
{
    double ialpha = 1./alpha;
    for( int i = 0; i < cn; i++ )
        a[i] = saturate_cast<_Tp>(a[i]*ialpha);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator /= (Vec<_Tp, cn>& a, float alpha)
{
    float ialpha = 1.f/alpha;
    for( int i = 0; i < cn; i++ )
        a[i] = saturate_cast<_Tp>(a[i]*ialpha);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator /= (Vec<_Tp, cn>& a, double alpha)
{
    double ialpha = 1./alpha;
    for( int i = 0; i < cn; i++ )
        a[i] = saturate_cast<_Tp>(a[i]*ialpha);
    return a;
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (const Vec<_Tp, cn>& a, int alpha)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (int alpha, const Vec<_Tp, cn>& a)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (const Vec<_Tp, cn>& a, float alpha)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (float alpha, const Vec<_Tp, cn>& a)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (const Vec<_Tp, cn>& a, double alpha)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (double alpha, const Vec<_Tp, cn>& a)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator / (const Vec<_Tp, cn>& a, int alpha)
{
    return Vec<_Tp, cn>(a, 1./alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator / (const Vec<_Tp, cn>& a, float alpha)
{
    return Vec<_Tp, cn>(a, 1.f/alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator / (const Vec<_Tp, cn>& a, double alpha)
{
    return Vec<_Tp, cn>(a, 1./alpha, Matx_ScaleOp());
}

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator - (const Vec<_Tp, cn>& a)
{
    Vec<_Tp,cn> t;
    for( int i = 0; i < cn; i++ ) t.val[i] = saturate_cast<_Tp>(-a.val[i]);
    return t;
}

template<typename _Tp> inline Vec<_Tp, 4> operator * (const Vec<_Tp, 4>& v1, const Vec<_Tp, 4>& v2)
{
    return Vec<_Tp, 4>(saturate_cast<_Tp>(v1[0]*v2[0] - v1[1]*v2[1] - v1[2]*v2[2] - v1[3]*v2[3]),
                       saturate_cast<_Tp>(v1[0]*v2[1] + v1[1]*v2[0] + v1[2]*v2[3] - v1[3]*v2[2]),
                       saturate_cast<_Tp>(v1[0]*v2[2] - v1[1]*v2[3] + v1[2]*v2[0] + v1[3]*v2[1]),
                       saturate_cast<_Tp>(v1[0]*v2[3] + v1[1]*v2[2] - v1[2]*v2[1] + v1[3]*v2[0]));
}

template<typename _Tp> inline Vec<_Tp, 4>& operator *= (Vec<_Tp, 4>& v1, const Vec<_Tp, 4>& v2)
{
    v1 = v1 * v2;
    return v1;
}

//! @}

} // cv

#endif // OPENCV_CORE_MATX_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HAL_NEON_UTILS_HPP
#define OPENCV_HAL_NEON_UTILS_HPP

#include "opencv2/core/cvdef.h"

//! @addtogroup core_utils_neon
//! @{

#if CV_NEON

inline int32x2_t cv_vrnd_s32_f32(float32x2_t v)
{
    static int32x2_t v_sign = vdup_n_s32(1 << 31),
        v_05 = vreinterpret_s32_f32(vdup_n_f32(0.5f));

    int32x2_t v_addition = vorr_s32(v_05, vand_s32(v_sign, vreinterpret_s32_f32(v)));
    return vcvt_s32_f32(vadd_f32(v, vreinterpret_f32_s32(v_addition)));
}

inline int32x4_t cv_vrndq_s32_f32(float32x4_t v)
{
    static int32x4_t v_sign = vdupq_n_s32(1 << 31),
        v_05 = vreinterpretq_s32_f32(vdupq_n_f32(0.5f));

    int32x4_t v_addition = vorrq_s32(v_05, vandq_s32(v_sign, vreinterpretq_s32_f32(v)));
    return vcvtq_s32_f32(vaddq_f32(v, vreinterpretq_f32_s32(v_addition)));
}

inline uint32x2_t cv_vrnd_u32_f32(float32x2_t v)
{
    static float32x2_t v_05 = vdup_n_f32(0.5f);
    return vcvt_u32_f32(vadd_f32(v, v_05));
}

inline uint32x4_t cv_vrndq_u32_f32(float32x4_t v)
{
    static float32x4_t v_05 = vdupq_n_f32(0.5f);
    return vcvtq_u32_f32(vaddq_f32(v, v_05));
}

inline float32x4_t cv_vrecpq_f32(float32x4_t val)
{
    float32x4_t reciprocal = vrecpeq_f32(val);
    reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(val, reciprocal), reciprocal);
    return reciprocal;
}

inline float32x2_t cv_vrecp_f32(float32x2_t val)
{
    float32x2_t reciprocal = vrecpe_f32(val);
    reciprocal = vmul_f32(vrecps_f32(val, reciprocal), reciprocal);
    reciprocal = vmul_f32(vrecps_f32(val, reciprocal), reciprocal);
    return reciprocal;
}

inline float32x4_t cv_vrsqrtq_f32(float32x4_t val)
{
    float32x4_t e = vrsqrteq_f32(val);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);
    e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(e, e), val), e);
    return e;
}

inline float32x2_t cv_vrsqrt_f32(float32x2_t val)
{
    float32x2_t e = vrsqrte_f32(val);
    e = vmul_f32(vrsqrts_f32(vmul_f32(e, e), val), e);
    e = vmul_f32(vrsqrts_f32(vmul_f32(e, e), val), e);
    return e;
}

inline float32x4_t cv_vsqrtq_f32(float32x4_t val)
{
    return cv_vrecpq_f32(cv_vrsqrtq_f32(val));
}

inline float32x2_t cv_vsqrt_f32(float32x2_t val)
{
    return cv_vrecp_f32(cv_vrsqrt_f32(val));
}

#endif

//! @}

#endif // OPENCV_HAL_NEON_UTILS_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OPENCL_GENBASE_HPP
#define OPENCV_OPENCL_GENBASE_HPP

//! @cond IGNORED

namespace cv {
namespace ocl {

class ProgramSource;

namespace internal {

struct CV_EXPORTS ProgramEntry
{
    const char* module;
    const char* name;
    const char* programCode;
    const char* programHash;
    ProgramSource* pProgramSource;

    operator ProgramSource& () const;
};

} } } // namespace

//! @endcond

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OPENCL_HPP
#define OPENCV_OPENCL_HPP

#include "opencv2/core.hpp"

namespace cv { namespace ocl {

//! @addtogroup core_opencl
//! @{

CV_EXPORTS_W bool haveOpenCL();
CV_EXPORTS_W bool useOpenCL();
CV_EXPORTS_W bool haveAmdBlas();
CV_EXPORTS_W bool haveAmdFft();
CV_EXPORTS_W void setUseOpenCL(bool flag);
CV_EXPORTS_W void finish();

CV_EXPORTS bool haveSVM();

class CV_EXPORTS Context;
class CV_EXPORTS Device;
class CV_EXPORTS Kernel;
class CV_EXPORTS Program;
class CV_EXPORTS ProgramSource;
class CV_EXPORTS Queue;
class CV_EXPORTS PlatformInfo;
class CV_EXPORTS Image2D;

class CV_EXPORTS Device
{
public:
    Device();
    explicit Device(void* d);
    Device(const Device& d);
    Device& operator = (const Device& d);
    ~Device();

    void set(void* d);

    enum
    {
        TYPE_DEFAULT     = (1 << 0),
        TYPE_CPU         = (1 << 1),
        TYPE_GPU         = (1 << 2),
        TYPE_ACCELERATOR = (1 << 3),
        TYPE_DGPU        = TYPE_GPU + (1 << 16),
        TYPE_IGPU        = TYPE_GPU + (1 << 17),
        TYPE_ALL         = 0xFFFFFFFF
    };

    String name() const;
    String extensions() const;
    String version() const;
    String vendorName() const;
    String OpenCL_C_Version() const;
    String OpenCLVersion() const;
    int deviceVersionMajor() const;
    int deviceVersionMinor() const;
    String driverVersion() const;
    void* ptr() const;

    int type() const;

    int addressBits() const;
    bool available() const;
    bool compilerAvailable() const;
    bool linkerAvailable() const;

    enum
    {
        FP_DENORM=(1 << 0),
        FP_INF_NAN=(1 << 1),
        FP_ROUND_TO_NEAREST=(1 << 2),
        FP_ROUND_TO_ZERO=(1 << 3),
        FP_ROUND_TO_INF=(1 << 4),
        FP_FMA=(1 << 5),
        FP_SOFT_FLOAT=(1 << 6),
        FP_CORRECTLY_ROUNDED_DIVIDE_SQRT=(1 << 7)
    };
    int doubleFPConfig() const;
    int singleFPConfig() const;
    int halfFPConfig() const;

    bool endianLittle() const;
    bool errorCorrectionSupport() const;

    enum
    {
        EXEC_KERNEL=(1 << 0),
        EXEC_NATIVE_KERNEL=(1 << 1)
    };
    int executionCapabilities() const;

    size_t globalMemCacheSize() const;

    enum
    {
        NO_CACHE=0,
        READ_ONLY_CACHE=1,
        READ_WRITE_CACHE=2
    };
    int globalMemCacheType() const;
    int globalMemCacheLineSize() const;
    size_t globalMemSize() const;

    size_t localMemSize() const;
    enum
    {
        NO_LOCAL_MEM=0,
        LOCAL_IS_LOCAL=1,
        LOCAL_IS_GLOBAL=2
    };
    int localMemType() const;
    bool hostUnifiedMemory() const;

    bool imageSupport() const;

    bool imageFromBufferSupport() const;
    uint imagePitchAlignment() const;
    uint imageBaseAddressAlignment() const;

    bool intelSubgroupsSupport() const;

    size_t image2DMaxWidth() const;
    size_t image2DMaxHeight() const;

    size_t image3DMaxWidth() const;
    size_t image3DMaxHeight() const;
    size_t image3DMaxDepth() const;

    size_t imageMaxBufferSize() const;
    size_t imageMaxArraySize() const;

    enum
    {
        UNKNOWN_VENDOR=0,
        VENDOR_AMD=1,
        VENDOR_INTEL=2,
        VENDOR_NVIDIA=3
    };
    int vendorID() const;
    // FIXIT
    // dev.isAMD() doesn't work for OpenCL CPU devices from AMD OpenCL platform.
    // This method should use platform name instead of vendor name.
    // After fix restore code in arithm.cpp: ocl_compare()
    inline bool isAMD() const { return vendorID() == VENDOR_AMD; }
    inline bool isIntel() const { return vendorID() == VENDOR_INTEL; }
    inline bool isNVidia() const { return vendorID() == VENDOR_NVIDIA; }

    int maxClockFrequency() const;
    int maxComputeUnits() const;
    int maxConstantArgs() const;
    size_t maxConstantBufferSize() const;

    size_t maxMemAllocSize() const;
    size_t maxParameterSize() const;

    int maxReadImageArgs() const;
    int maxWriteImageArgs() const;
    int maxSamplers() const;

    size_t maxWorkGroupSize() const;
    int maxWorkItemDims() const;
    void maxWorkItemSizes(size_t*) const;

    int memBaseAddrAlign() const;

    int nativeVectorWidthChar() const;
    int nativeVectorWidthShort() const;
    int nativeVectorWidthInt() const;
    int nativeVectorWidthLong() const;
    int nativeVectorWidthFloat() const;
    int nativeVectorWidthDouble() const;
    int nativeVectorWidthHalf() const;

    int preferredVectorWidthChar() const;
    int preferredVectorWidthShort() const;
    int preferredVectorWidthInt() const;
    int preferredVectorWidthLong() const;
    int preferredVectorWidthFloat() const;
    int preferredVectorWidthDouble() const;
    int preferredVectorWidthHalf() const;

    size_t printfBufferSize() const;
    size_t profilingTimerResolution() const;

    static const Device& getDefault();

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS Context
{
public:
    Context();
    explicit Context(int dtype);
    ~Context();
    Context(const Context& c);
    Context& operator = (const Context& c);

    bool create();
    bool create(int dtype);
    size_t ndevices() const;
    const Device& device(size_t idx) const;
    Program getProg(const ProgramSource& prog,
                    const String& buildopt, String& errmsg);

    static Context& getDefault(bool initialize = true);
    void* ptr() const;

    friend void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);

    bool useSVM() const;
    void setUseSVM(bool enabled);

    struct Impl;
    Impl* p;
};

class CV_EXPORTS Platform
{
public:
    Platform();
    ~Platform();
    Platform(const Platform& p);
    Platform& operator = (const Platform& p);

    void* ptr() const;
    static Platform& getDefault();

    friend void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);
protected:
    struct Impl;
    Impl* p;
};

/** @brief Attaches OpenCL context to OpenCV
@note
  OpenCV will check if available OpenCL platform has platformName name, then assign context to
  OpenCV and call `clRetainContext` function. The deviceID device will be used as target device and
  new command queue will be created.
@param platformName name of OpenCL platform to attach, this string is used to check if platform is available to OpenCV at runtime
@param platformID ID of platform attached context was created for
@param context OpenCL context to be attached to OpenCV
@param deviceID ID of device, must be created from attached context
*/
CV_EXPORTS void attachContext(const String& platformName, void* platformID, void* context, void* deviceID);

/** @brief Convert OpenCL buffer to UMat
@note
  OpenCL buffer (cl_mem_buffer) should contain 2D image data, compatible with OpenCV. Memory
  content is not copied from `clBuffer` to UMat. Instead, buffer handle assigned to UMat and
  `clRetainMemObject` is called.
@param cl_mem_buffer source clBuffer handle
@param step num of bytes in single row
@param rows number of rows
@param cols number of cols
@param type OpenCV type of image
@param dst destination UMat
*/
CV_EXPORTS void convertFromBuffer(void* cl_mem_buffer, size_t step, int rows, int cols, int type, UMat& dst);

/** @brief Convert OpenCL image2d_t to UMat
@note
  OpenCL `image2d_t` (cl_mem_image), should be compatible with OpenCV UMat formats. Memory content
  is copied from image to UMat with `clEnqueueCopyImageToBuffer` function.
@param cl_mem_image source image2d_t handle
@param dst destination UMat
*/
CV_EXPORTS void convertFromImage(void* cl_mem_image, UMat& dst);

// TODO Move to internal header
void initializeContextFromHandle(Context& ctx, void* platform, void* context, void* device);

class CV_EXPORTS Queue
{
public:
    Queue();
    explicit Queue(const Context& c, const Device& d=Device());
    ~Queue();
    Queue(const Queue& q);
    Queue& operator = (const Queue& q);

    bool create(const Context& c=Context(), const Device& d=Device());
    void finish();
    void* ptr() const;
    static Queue& getDefault();

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS KernelArg
{
public:
    enum { LOCAL=1, READ_ONLY=2, WRITE_ONLY=4, READ_WRITE=6, CONSTANT=8, PTR_ONLY = 16, NO_SIZE=256 };
    KernelArg(int _flags, UMat* _m, int wscale=1, int iwscale=1, const void* _obj=0, size_t _sz=0);
    KernelArg();

    static KernelArg Local() { return KernelArg(LOCAL, 0); }
    static KernelArg PtrWriteOnly(const UMat& m)
    { return KernelArg(PTR_ONLY+WRITE_ONLY, (UMat*)&m); }
    static KernelArg PtrReadOnly(const UMat& m)
    { return KernelArg(PTR_ONLY+READ_ONLY, (UMat*)&m); }
    static KernelArg PtrReadWrite(const UMat& m)
    { return KernelArg(PTR_ONLY+READ_WRITE, (UMat*)&m); }
    static KernelArg ReadWrite(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_WRITE, (UMat*)&m, wscale, iwscale); }
    static KernelArg ReadWriteNoSize(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_WRITE+NO_SIZE, (UMat*)&m, wscale, iwscale); }
    static KernelArg ReadOnly(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_ONLY, (UMat*)&m, wscale, iwscale); }
    static KernelArg WriteOnly(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(WRITE_ONLY, (UMat*)&m, wscale, iwscale); }
    static KernelArg ReadOnlyNoSize(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(READ_ONLY+NO_SIZE, (UMat*)&m, wscale, iwscale); }
    static KernelArg WriteOnlyNoSize(const UMat& m, int wscale=1, int iwscale=1)
    { return KernelArg(WRITE_ONLY+NO_SIZE, (UMat*)&m, wscale, iwscale); }
    static KernelArg Constant(const Mat& m);
    template<typename _Tp> static KernelArg Constant(const _Tp* arr, size_t n)
    { return KernelArg(CONSTANT, 0, 1, 1, (void*)arr, n); }

    int flags;
    UMat* m;
    const void* obj;
    size_t sz;
    int wscale, iwscale;
};


class CV_EXPORTS Kernel
{
public:
    Kernel();
    Kernel(const char* kname, const Program& prog);
    Kernel(const char* kname, const ProgramSource& prog,
           const String& buildopts = String(), String* errmsg=0);
    ~Kernel();
    Kernel(const Kernel& k);
    Kernel& operator = (const Kernel& k);

    bool empty() const;
    bool create(const char* kname, const Program& prog);
    bool create(const char* kname, const ProgramSource& prog,
                const String& buildopts, String* errmsg=0);

    int set(int i, const void* value, size_t sz);
    int set(int i, const Image2D& image2D);
    int set(int i, const UMat& m);
    int set(int i, const KernelArg& arg);
    template<typename _Tp> int set(int i, const _Tp& value)
    { return set(i, &value, sizeof(value)); }

    template<typename _Tp0>
    Kernel& args(const _Tp0& a0)
    {
        set(0, a0); return *this;
    }

    template<typename _Tp0, typename _Tp1>
    Kernel& args(const _Tp0& a0, const _Tp1& a1)
    {
        int i = set(0, a0); set(i, a1); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2)
    {
        int i = set(0, a0); i = set(i, a1); set(i, a2); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2,
                 const _Tp3& a3, const _Tp4& a4)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2);
        i = set(i, a3); set(i, a4); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2,
             typename _Tp3, typename _Tp4, typename _Tp5>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2,
                 const _Tp3& a3, const _Tp4& a4, const _Tp5& a5)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2);
        i = set(i, a3); i = set(i, a4); set(i, a5); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3);
        i = set(i, a4); i = set(i, a5); set(i, a6); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3);
        i = set(i, a4); i = set(i, a5); i = set(i, a6); set(i, a7); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4,
             typename _Tp5, typename _Tp6, typename _Tp7, typename _Tp8>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4);
        i = set(i, a5); i = set(i, a6); i = set(i, a7); set(i, a8); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3, typename _Tp4,
             typename _Tp5, typename _Tp6, typename _Tp7, typename _Tp8, typename _Tp9>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); set(i, a9); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); set(i, a10); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); set(i, a11); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        set(i, a12); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12,
             typename _Tp13>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12, const _Tp13& a13)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        i = set(i, a12); set(i, a13); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12,
             typename _Tp13, typename _Tp14>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12, const _Tp13& a13, const _Tp14& a14)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        i = set(i, a12); i = set(i, a13); set(i, a14); return *this;
    }

    template<typename _Tp0, typename _Tp1, typename _Tp2, typename _Tp3,
             typename _Tp4, typename _Tp5, typename _Tp6, typename _Tp7,
             typename _Tp8, typename _Tp9, typename _Tp10, typename _Tp11, typename _Tp12,
             typename _Tp13, typename _Tp14, typename _Tp15>
    Kernel& args(const _Tp0& a0, const _Tp1& a1, const _Tp2& a2, const _Tp3& a3,
                 const _Tp4& a4, const _Tp5& a5, const _Tp6& a6, const _Tp7& a7,
                 const _Tp8& a8, const _Tp9& a9, const _Tp10& a10, const _Tp11& a11,
                 const _Tp12& a12, const _Tp13& a13, const _Tp14& a14, const _Tp15& a15)
    {
        int i = set(0, a0); i = set(i, a1); i = set(i, a2); i = set(i, a3); i = set(i, a4); i = set(i, a5);
        i = set(i, a6); i = set(i, a7); i = set(i, a8); i = set(i, a9); i = set(i, a10); i = set(i, a11);
        i = set(i, a12); i = set(i, a13); i = set(i, a14); set(i, a15); return *this;
    }
    /** @brief Run the OpenCL kernel.
    @param dims the work problem dimensions. It is the length of globalsize and localsize. It can be either 1, 2 or 3.
    @param globalsize work items for each dimension. It is not the final globalsize passed to
      OpenCL. Each dimension will be adjusted to the nearest integer divisible by the corresponding
      value in localsize. If localsize is NULL, it will still be adjusted depending on dims. The
      adjusted values are greater than or equal to the original values.
    @param localsize work-group size for each dimension.
    @param sync specify whether to wait for OpenCL computation to finish before return.
    @param q command queue
    */
    bool run(int dims, size_t globalsize[],
             size_t localsize[], bool sync, const Queue& q=Queue());
    bool runTask(bool sync, const Queue& q=Queue());

    size_t workGroupSize() const;
    size_t preferedWorkGroupSizeMultiple() const;
    bool compileWorkGroupSize(size_t wsz[]) const;
    size_t localMemSize() const;

    void* ptr() const;
    struct Impl;

protected:
    Impl* p;
};

class CV_EXPORTS Program
{
public:
    Program();
    Program(const ProgramSource& src,
            const String& buildflags, String& errmsg);
    explicit Program(const String& buf);
    Program(const Program& prog);

    Program& operator = (const Program& prog);
    ~Program();

    bool create(const ProgramSource& src,
                const String& buildflags, String& errmsg);
    bool read(const String& buf, const String& buildflags);
    bool write(String& buf) const;

    const ProgramSource& source() const;
    void* ptr() const;

    String getPrefix() const;
    static String getPrefix(const String& buildflags);

protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS ProgramSource
{
public:
    typedef uint64 hash_t; // deprecated

    ProgramSource();
    explicit ProgramSource(const String& module, const String& name, const String& codeStr, const String& codeHash);
    explicit ProgramSource(const String& prog); // deprecated
    explicit ProgramSource(const char* prog); // deprecated
    ~ProgramSource();
    ProgramSource(const ProgramSource& prog);
    ProgramSource& operator = (const ProgramSource& prog);

    const String& source() const;
    hash_t hash() const; // deprecated

protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS PlatformInfo
{
public:
    PlatformInfo();
    explicit PlatformInfo(void* id);
    ~PlatformInfo();

    PlatformInfo(const PlatformInfo& i);
    PlatformInfo& operator =(const PlatformInfo& i);

    String name() const;
    String vendor() const;
    String version() const;
    int deviceNumber() const;
    void getDevice(Device& device, int d) const;

protected:
    struct Impl;
    Impl* p;
};

CV_EXPORTS const char* convertTypeStr(int sdepth, int ddepth, int cn, char* buf);
CV_EXPORTS const char* typeToStr(int t);
CV_EXPORTS const char* memopTypeToStr(int t);
CV_EXPORTS const char* vecopTypeToStr(int t);
CV_EXPORTS String kernelToStr(InputArray _kernel, int ddepth = -1, const char * name = NULL);
CV_EXPORTS void getPlatfomsInfo(std::vector<PlatformInfo>& platform_info);


enum OclVectorStrategy
{
    // all matrices have its own vector width
    OCL_VECTOR_OWN = 0,
    // all matrices have maximal vector width among all matrices
    // (useful for cases when matrices have different data types)
    OCL_VECTOR_MAX = 1,

    // default strategy
    OCL_VECTOR_DEFAULT = OCL_VECTOR_OWN
};

CV_EXPORTS int predictOptimalVectorWidth(InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                         InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                         InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray(),
                                         OclVectorStrategy strat = OCL_VECTOR_DEFAULT);

CV_EXPORTS int checkOptimalVectorWidth(const int *vectorWidths,
                                       InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                       InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                       InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray(),
                                       OclVectorStrategy strat = OCL_VECTOR_DEFAULT);

// with OCL_VECTOR_MAX strategy
CV_EXPORTS int predictOptimalVectorWidthMax(InputArray src1, InputArray src2 = noArray(), InputArray src3 = noArray(),
                                            InputArray src4 = noArray(), InputArray src5 = noArray(), InputArray src6 = noArray(),
                                            InputArray src7 = noArray(), InputArray src8 = noArray(), InputArray src9 = noArray());

CV_EXPORTS void buildOptionsAddMatrixDescription(String& buildOptions, const String& name, InputArray _m);

class CV_EXPORTS Image2D
{
public:
    Image2D();

    /**
    @param src UMat object from which to get image properties and data
    @param norm flag to enable the use of normalized channel data types
    @param alias flag indicating that the image should alias the src UMat. If true, changes to the
        image or src will be reflected in both objects.
    */
    explicit Image2D(const UMat &src, bool norm = false, bool alias = false);
    Image2D(const Image2D & i);
    ~Image2D();

    Image2D & operator = (const Image2D & i);

    /** Indicates if creating an aliased image should succeed.
    Depends on the underlying platform and the dimensions of the UMat.
    */
    static bool canCreateAlias(const UMat &u);

    /** Indicates if the image format is supported.
    */
    static bool isFormatSupported(int depth, int cn, bool norm);

    void* ptr() const;
protected:
    struct Impl;
    Impl* p;
};


CV_EXPORTS MatAllocator* getOpenCLAllocator();


#ifdef __OPENCV_BUILD
namespace internal {

CV_EXPORTS bool isOpenCLForced();
#define OCL_FORCE_CHECK(condition) (cv::ocl::internal::isOpenCLForced() || (condition))

CV_EXPORTS bool isPerformanceCheckBypassed();
#define OCL_PERFORMANCE_CHECK(condition) (cv::ocl::internal::isPerformanceCheckBypassed() || (condition))

CV_EXPORTS bool isCLBuffer(UMat& u);

} // namespace internal
#endif

//! @}

}}

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_OPENGL_HPP
#define OPENCV_CORE_OPENGL_HPP

#ifndef __cplusplus
#  error opengl.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "ocl.hpp"

namespace cv { namespace ogl {

/** @addtogroup core_opengl
This section describes OpenGL interoperability.

To enable OpenGL support, configure OpenCV using CMake with WITH_OPENGL=ON . Currently OpenGL is
supported only with WIN32, GTK and Qt backends on Windows and Linux (MacOS and Android are not
supported). For GTK backend gtkglext-1.0 library is required.

To use OpenGL functionality you should first create OpenGL context (window or frame buffer). You can
do this with namedWindow function or with other OpenGL toolkit (GLUT, for example).
*/
//! @{

/////////////////// OpenGL Objects ///////////////////

/** @brief Smart pointer for OpenGL buffer object with reference counting.

Buffer Objects are OpenGL objects that store an array of unformatted memory allocated by the OpenGL
context. These can be used to store vertex data, pixel data retrieved from images or the
framebuffer, and a variety of other things.

ogl::Buffer has interface similar with Mat interface and represents 2D array memory.

ogl::Buffer supports memory transfers between host and device and also can be mapped to CUDA memory.
 */
class CV_EXPORTS Buffer
{
public:
    /** @brief The target defines how you intend to use the buffer object.
    */
    enum Target
    {
        ARRAY_BUFFER         = 0x8892, //!< The buffer will be used as a source for vertex data
        ELEMENT_ARRAY_BUFFER = 0x8893, //!< The buffer will be used for indices (in glDrawElements, for example)
        PIXEL_PACK_BUFFER    = 0x88EB, //!< The buffer will be used for reading from OpenGL textures
        PIXEL_UNPACK_BUFFER  = 0x88EC  //!< The buffer will be used for writing to OpenGL textures
    };

    enum Access
    {
        READ_ONLY  = 0x88B8,
        WRITE_ONLY = 0x88B9,
        READ_WRITE = 0x88BA
    };

    /** @brief The constructors.

    Creates empty ogl::Buffer object, creates ogl::Buffer object from existed buffer ( abufId
    parameter), allocates memory for ogl::Buffer object or copies from host/device memory.
     */
    Buffer();

    /** @overload
    @param arows Number of rows in a 2D array.
    @param acols Number of columns in a 2D array.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param abufId Buffer object name.
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(int arows, int acols, int atype, unsigned int abufId, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param abufId Buffer object name.
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(Size asize, int atype, unsigned int abufId, bool autoRelease = false);

    /** @overload
    @param arows Number of rows in a 2D array.
    @param acols Number of columns in a 2D array.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Buffer(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload
    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or std::vector ).
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    explicit Buffer(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @brief Allocates memory for ogl::Buffer object.

    @param arows Number of rows in a 2D array.
    @param acols Number of columns in a 2D array.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void create(int arows, int acols, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param atype Array type ( CV_8UC1, ..., CV_64FC4 ). See Mat for details.
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    void create(Size asize, int atype, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @brief Decrements the reference counter and destroys the buffer object if needed.

    The function will call setAutoRelease(true) .
     */
    void release();

    /** @brief Sets auto release mode.

    The lifetime of the OpenGL object is tied to the lifetime of the context. If OpenGL context was
    bound to a window it could be released at any time (user can close a window). If object's destructor
    is called after destruction of the context it will cause an error. Thus ogl::Buffer doesn't destroy
    OpenGL object in destructor by default (all OpenGL resources will be released with OpenGL context).
    This function can force ogl::Buffer destructor to destroy OpenGL object.
    @param flag Auto release mode (if true, release will be called in object's destructor).
     */
    void setAutoRelease(bool flag);

    /** @brief Copies from host/device memory to OpenGL buffer.
    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or std::vector ).
    @param target Buffer usage. See cv::ogl::Buffer::Target .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void copyFrom(InputArray arr, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @overload */
    void copyFrom(InputArray arr, cuda::Stream& stream, Target target = ARRAY_BUFFER, bool autoRelease = false);

    /** @brief Copies from OpenGL buffer to host/device memory or another OpenGL buffer object.

    @param arr Destination array (host or device memory, can be Mat , cuda::GpuMat , std::vector or
    ogl::Buffer ).
     */
    void copyTo(OutputArray arr) const;

    /** @overload */
    void copyTo(OutputArray arr, cuda::Stream& stream) const;

    /** @brief Creates a full copy of the buffer object and the underlying data.

    @param target Buffer usage for destination buffer.
    @param autoRelease Auto release mode for destination buffer.
     */
    Buffer clone(Target target = ARRAY_BUFFER, bool autoRelease = false) const;

    /** @brief Binds OpenGL buffer to the specified buffer binding point.

    @param target Binding point. See cv::ogl::Buffer::Target .
     */
    void bind(Target target) const;

    /** @brief Unbind any buffers from the specified binding point.

    @param target Binding point. See cv::ogl::Buffer::Target .
     */
    static void unbind(Target target);

    /** @brief Maps OpenGL buffer to host memory.

    mapHost maps to the client's address space the entire data store of the buffer object. The data can
    then be directly read and/or written relative to the returned pointer, depending on the specified
    access policy.

    A mapped data store must be unmapped with ogl::Buffer::unmapHost before its buffer object is used.

    This operation can lead to memory transfers between host and device.

    Only one buffer object can be mapped at a time.
    @param access Access policy, indicating whether it will be possible to read from, write to, or both
    read from and write to the buffer object's mapped data store. The symbolic constant must be
    ogl::Buffer::READ_ONLY , ogl::Buffer::WRITE_ONLY or ogl::Buffer::READ_WRITE .
     */
    Mat mapHost(Access access);

    /** @brief Unmaps OpenGL buffer.
    */
    void unmapHost();

    //! map to device memory (blocking)
    cuda::GpuMat mapDevice();
    void unmapDevice();

    /** @brief Maps OpenGL buffer to CUDA device memory.

    This operatation doesn't copy data. Several buffer objects can be mapped to CUDA memory at a time.

    A mapped data store must be unmapped with ogl::Buffer::unmapDevice before its buffer object is used.
     */
    cuda::GpuMat mapDevice(cuda::Stream& stream);

    /** @brief Unmaps OpenGL buffer.
    */
    void unmapDevice(cuda::Stream& stream);

    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;

    int type() const;
    int depth() const;
    int channels() const;
    int elemSize() const;
    int elemSize1() const;

    //! get OpenGL opject id
    unsigned int bufId() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    int rows_;
    int cols_;
    int type_;
};

/** @brief Smart pointer for OpenGL 2D texture memory with reference counting.
 */
class CV_EXPORTS Texture2D
{
public:
    /** @brief An Image Format describes the way that the images in Textures store their data.
    */
    enum Format
    {
        NONE            = 0,
        DEPTH_COMPONENT = 0x1902, //!< Depth
        RGB             = 0x1907, //!< Red, Green, Blue
        RGBA            = 0x1908  //!< Red, Green, Blue, Alpha
    };

    /** @brief The constructors.

    Creates empty ogl::Texture2D object, allocates memory for ogl::Texture2D object or copies from
    host/device memory.
     */
    Texture2D();

    /** @overload */
    Texture2D(int arows, int acols, Format aformat, unsigned int atexId, bool autoRelease = false);

    /** @overload */
    Texture2D(Size asize, Format aformat, unsigned int atexId, bool autoRelease = false);

    /** @overload
    @param arows Number of rows.
    @param acols Number of columns.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Texture2D(int arows, int acols, Format aformat, bool autoRelease = false);

    /** @overload
    @param asize 2D array size.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    Texture2D(Size asize, Format aformat, bool autoRelease = false);

    /** @overload
    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or ogl::Buffer ).
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    explicit Texture2D(InputArray arr, bool autoRelease = false);

    /** @brief Allocates memory for ogl::Texture2D object.

    @param arows Number of rows.
    @param acols Number of columns.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void create(int arows, int acols, Format aformat, bool autoRelease = false);
    /** @overload
    @param asize 2D array size.
    @param aformat Image format. See cv::ogl::Texture2D::Format .
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
    */
    void create(Size asize, Format aformat, bool autoRelease = false);

    /** @brief Decrements the reference counter and destroys the texture object if needed.

    The function will call setAutoRelease(true) .
     */
    void release();

    /** @brief Sets auto release mode.

    @param flag Auto release mode (if true, release will be called in object's destructor).

    The lifetime of the OpenGL object is tied to the lifetime of the context. If OpenGL context was
    bound to a window it could be released at any time (user can close a window). If object's destructor
    is called after destruction of the context it will cause an error. Thus ogl::Texture2D doesn't
    destroy OpenGL object in destructor by default (all OpenGL resources will be released with OpenGL
    context). This function can force ogl::Texture2D destructor to destroy OpenGL object.
     */
    void setAutoRelease(bool flag);

    /** @brief Copies from host/device memory to OpenGL texture.

    @param arr Input array (host or device memory, it can be Mat , cuda::GpuMat or ogl::Buffer ).
    @param autoRelease Auto release mode (if true, release will be called in object's destructor).
     */
    void copyFrom(InputArray arr, bool autoRelease = false);

    /** @brief Copies from OpenGL texture to host/device memory or another OpenGL texture object.

    @param arr Destination array (host or device memory, can be Mat , cuda::GpuMat , ogl::Buffer or
    ogl::Texture2D ).
    @param ddepth Destination depth.
    @param autoRelease Auto release mode for destination buffer (if arr is OpenGL buffer or texture).
     */
    void copyTo(OutputArray arr, int ddepth = CV_32F, bool autoRelease = false) const;

    /** @brief Binds texture to current active texture unit for GL_TEXTURE_2D target.
    */
    void bind() const;

    int rows() const;
    int cols() const;
    Size size() const;
    bool empty() const;

    Format format() const;

    //! get OpenGL opject id
    unsigned int texId() const;

    class Impl;

private:
    Ptr<Impl> impl_;
    int rows_;
    int cols_;
    Format format_;
};

/** @brief Wrapper for OpenGL Client-Side Vertex arrays.

ogl::Arrays stores vertex data in ogl::Buffer objects.
 */
class CV_EXPORTS Arrays
{
public:
    /** @brief Default constructor
     */
    Arrays();

    /** @brief Sets an array of vertex coordinates.
    @param vertex array with vertex coordinates, can be both host and device memory.
    */
    void setVertexArray(InputArray vertex);

    /** @brief Resets vertex coordinates.
    */
    void resetVertexArray();

    /** @brief Sets an array of vertex colors.
    @param color array with vertex colors, can be both host and device memory.
     */
    void setColorArray(InputArray color);

    /** @brief Resets vertex colors.
    */
    void resetColorArray();

    /** @brief Sets an array of vertex normals.
    @param normal array with vertex normals, can be both host and device memory.
     */
    void setNormalArray(InputArray normal);

    /** @brief Resets vertex normals.
    */
    void resetNormalArray();

    /** @brief Sets an array of vertex texture coordinates.
    @param texCoord array with vertex texture coordinates, can be both host and device memory.
     */
    void setTexCoordArray(InputArray texCoord);

    /** @brief Resets vertex texture coordinates.
    */
    void resetTexCoordArray();

    /** @brief Releases all inner buffers.
    */
    void release();

    /** @brief Sets auto release mode all inner buffers.
    @param flag Auto release mode.
     */
    void setAutoRelease(bool flag);

    /** @brief Binds all vertex arrays.
    */
    void bind() const;

    /** @brief Returns the vertex count.
    */
    int size() const;
    bool empty() const;

private:
    int size_;
    Buffer vertex_;
    Buffer color_;
    Buffer normal_;
    Buffer texCoord_;
};

/////////////////// Render Functions ///////////////////

//! render mode
enum RenderModes {
    POINTS         = 0x0000,
    LINES          = 0x0001,
    LINE_LOOP      = 0x0002,
    LINE_STRIP     = 0x0003,
    TRIANGLES      = 0x0004,
    TRIANGLE_STRIP = 0x0005,
    TRIANGLE_FAN   = 0x0006,
    QUADS          = 0x0007,
    QUAD_STRIP     = 0x0008,
    POLYGON        = 0x0009
};

/** @brief Render OpenGL texture or primitives.
@param tex Texture to draw.
@param wndRect Region of window, where to draw a texture (normalized coordinates).
@param texRect Region of texture to draw (normalized coordinates).
 */
CV_EXPORTS void render(const Texture2D& tex,
    Rect_<double> wndRect = Rect_<double>(0.0, 0.0, 1.0, 1.0),
    Rect_<double> texRect = Rect_<double>(0.0, 0.0, 1.0, 1.0));

/** @overload
@param arr Array of privitives vertices.
@param mode Render mode. One of cv::ogl::RenderModes
@param color Color for all vertices. Will be used if arr doesn't contain color array.
*/
CV_EXPORTS void render(const Arrays& arr, int mode = POINTS, Scalar color = Scalar::all(255));

/** @overload
@param arr Array of privitives vertices.
@param indices Array of vertices indices (host or device memory).
@param mode Render mode. One of cv::ogl::RenderModes
@param color Color for all vertices. Will be used if arr doesn't contain color array.
*/
CV_EXPORTS void render(const Arrays& arr, InputArray indices, int mode = POINTS, Scalar color = Scalar::all(255));

/////////////////// CL-GL Interoperability Functions ///////////////////

namespace ocl {
using namespace cv::ocl;

// TODO static functions in the Context class
/** @brief Creates OpenCL context from GL.
@return Returns reference to OpenCL Context
 */
CV_EXPORTS Context& initializeContextFromGL();

} // namespace cv::ogl::ocl

/** @brief Converts InputArray to Texture2D object.
@param src     - source InputArray.
@param texture - destination Texture2D object.
 */
CV_EXPORTS void convertToGLTexture2D(InputArray src, Texture2D& texture);

/** @brief Converts Texture2D object to OutputArray.
@param texture - source Texture2D object.
@param dst     - destination OutputArray.
 */
CV_EXPORTS void convertFromGLTexture2D(const Texture2D& texture, OutputArray dst);

/** @brief Maps Buffer object to process on CL side (convert to UMat).

Function creates CL buffer from GL one, and then constructs UMat that can be used
to process buffer data with OpenCV functions. Note that in current implementation
UMat constructed this way doesn't own corresponding GL buffer object, so it is
the user responsibility to close down CL/GL buffers relationships by explicitly
calling unmapGLBuffer() function.
@param buffer      - source Buffer object.
@param accessFlags - data access flags (ACCESS_READ|ACCESS_WRITE).
@return Returns UMat object
 */
CV_EXPORTS UMat mapGLBuffer(const Buffer& buffer, int accessFlags = ACCESS_READ|ACCESS_WRITE);

/** @brief Unmaps Buffer object (releases UMat, previously mapped from Buffer).

Function must be called explicitly by the user for each UMat previously constructed
by the call to mapGLBuffer() function.
@param u           - source UMat, created by mapGLBuffer().
 */
CV_EXPORTS void unmapGLBuffer(UMat& u);

}} // namespace cv::ogl

namespace cv { namespace cuda {

//! @addtogroup cuda
//! @{

/** @brief Sets a CUDA device and initializes it for the current thread with OpenGL interoperability.

This function should be explicitly called after OpenGL context creation and before any CUDA calls.
@param device System index of a CUDA device starting with 0.
@ingroup core_opengl
 */
CV_EXPORTS void setGlDevice(int device = 0);

//! @}

}}

//! @cond IGNORED

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

inline
cv::ogl::Buffer::Buffer(int arows, int acols, int atype, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
    create(arows, acols, atype, target, autoRelease);
}

inline
cv::ogl::Buffer::Buffer(Size asize, int atype, Target target, bool autoRelease) : rows_(0), cols_(0), type_(0)
{
    create(asize, atype, target, autoRelease);
}

inline
void cv::ogl::Buffer::create(Size asize, int atype, Target target, bool autoRelease)
{
    create(asize.height, asize.width, atype, target, autoRelease);
}

inline
int cv::ogl::Buffer::rows() const
{
    return rows_;
}

inline
int cv::ogl::Buffer::cols() const
{
    return cols_;
}

inline
cv::Size cv::ogl::Buffer::size() const
{
    return Size(cols_, rows_);
}

inline
bool cv::ogl::Buffer::empty() const
{
    return rows_ == 0 || cols_ == 0;
}

inline
int cv::ogl::Buffer::type() const
{
    return type_;
}

inline
int cv::ogl::Buffer::depth() const
{
    return CV_MAT_DEPTH(type_);
}

inline
int cv::ogl::Buffer::channels() const
{
    return CV_MAT_CN(type_);
}

inline
int cv::ogl::Buffer::elemSize() const
{
    return CV_ELEM_SIZE(type_);
}

inline
int cv::ogl::Buffer::elemSize1() const
{
    return CV_ELEM_SIZE1(type_);
}

///////

inline
cv::ogl::Texture2D::Texture2D(int arows, int acols, Format aformat, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
    create(arows, acols, aformat, autoRelease);
}

inline
cv::ogl::Texture2D::Texture2D(Size asize, Format aformat, bool autoRelease) : rows_(0), cols_(0), format_(NONE)
{
    create(asize, aformat, autoRelease);
}

inline
void cv::ogl::Texture2D::create(Size asize, Format aformat, bool autoRelease)
{
    create(asize.height, asize.width, aformat, autoRelease);
}

inline
int cv::ogl::Texture2D::rows() const
{
    return rows_;
}

inline
int cv::ogl::Texture2D::cols() const
{
    return cols_;
}

inline
cv::Size cv::ogl::Texture2D::size() const
{
    return Size(cols_, rows_);
}

inline
bool cv::ogl::Texture2D::empty() const
{
    return rows_ == 0 || cols_ == 0;
}

inline
cv::ogl::Texture2D::Format cv::ogl::Texture2D::format() const
{
    return format_;
}

///////

inline
cv::ogl::Arrays::Arrays() : size_(0)
{
}

inline
int cv::ogl::Arrays::size() const
{
    return size_;
}

inline
bool cv::ogl::Arrays::empty() const
{
    return size_ == 0;
}

//! @endcond

#endif /* OPENCV_CORE_OPENGL_HPP */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_OPERATIONS_HPP
#define OPENCV_CORE_OPERATIONS_HPP

#ifndef __cplusplus
#  error operations.hpp header must be compiled as C++
#endif

#include <cstdio>

//! @cond IGNORED

namespace cv
{

////////////////////////////// Matx methods depending on core API /////////////////////////////

namespace internal
{

template<typename _Tp, int m> struct Matx_FastInvOp
{
    bool operator()(const Matx<_Tp, m, m>& a, Matx<_Tp, m, m>& b, int method) const
    {
        Matx<_Tp, m, m> temp = a;

        // assume that b is all 0's on input => make it a unity matrix
        for( int i = 0; i < m; i++ )
            b(i, i) = (_Tp)1;

        if( method == DECOMP_CHOLESKY )
            return Cholesky(temp.val, m*sizeof(_Tp), m, b.val, m*sizeof(_Tp), m);

        return LU(temp.val, m*sizeof(_Tp), m, b.val, m*sizeof(_Tp), m) != 0;
    }
};

template<typename _Tp> struct Matx_FastInvOp<_Tp, 2>
{
    bool operator()(const Matx<_Tp, 2, 2>& a, Matx<_Tp, 2, 2>& b, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        b(1,1) = a(0,0)*d;
        b(0,0) = a(1,1)*d;
        b(0,1) = -a(0,1)*d;
        b(1,0) = -a(1,0)*d;
        return true;
    }
};

template<typename _Tp> struct Matx_FastInvOp<_Tp, 3>
{
    bool operator()(const Matx<_Tp, 3, 3>& a, Matx<_Tp, 3, 3>& b, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        b(0,0) = (a(1,1) * a(2,2) - a(1,2) * a(2,1)) * d;
        b(0,1) = (a(0,2) * a(2,1) - a(0,1) * a(2,2)) * d;
        b(0,2) = (a(0,1) * a(1,2) - a(0,2) * a(1,1)) * d;

        b(1,0) = (a(1,2) * a(2,0) - a(1,0) * a(2,2)) * d;
        b(1,1) = (a(0,0) * a(2,2) - a(0,2) * a(2,0)) * d;
        b(1,2) = (a(0,2) * a(1,0) - a(0,0) * a(1,2)) * d;

        b(2,0) = (a(1,0) * a(2,1) - a(1,1) * a(2,0)) * d;
        b(2,1) = (a(0,1) * a(2,0) - a(0,0) * a(2,1)) * d;
        b(2,2) = (a(0,0) * a(1,1) - a(0,1) * a(1,0)) * d;
        return true;
    }
};


template<typename _Tp, int m, int n> struct Matx_FastSolveOp
{
    bool operator()(const Matx<_Tp, m, m>& a, const Matx<_Tp, m, n>& b,
                    Matx<_Tp, m, n>& x, int method) const
    {
        Matx<_Tp, m, m> temp = a;
        x = b;
        if( method == DECOMP_CHOLESKY )
            return Cholesky(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n);

        return LU(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n) != 0;
    }
};

template<typename _Tp> struct Matx_FastSolveOp<_Tp, 2, 1>
{
    bool operator()(const Matx<_Tp, 2, 2>& a, const Matx<_Tp, 2, 1>& b,
                    Matx<_Tp, 2, 1>& x, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        x(0) = (b(0)*a(1,1) - b(1)*a(0,1))*d;
        x(1) = (b(1)*a(0,0) - b(0)*a(1,0))*d;
        return true;
    }
};

template<typename _Tp> struct Matx_FastSolveOp<_Tp, 3, 1>
{
    bool operator()(const Matx<_Tp, 3, 3>& a, const Matx<_Tp, 3, 1>& b,
                    Matx<_Tp, 3, 1>& x, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        x(0) = d*(b(0)*(a(1,1)*a(2,2) - a(1,2)*a(2,1)) -
                a(0,1)*(b(1)*a(2,2) - a(1,2)*b(2)) +
                a(0,2)*(b(1)*a(2,1) - a(1,1)*b(2)));

        x(1) = d*(a(0,0)*(b(1)*a(2,2) - a(1,2)*b(2)) -
                b(0)*(a(1,0)*a(2,2) - a(1,2)*a(2,0)) +
                a(0,2)*(a(1,0)*b(2) - b(1)*a(2,0)));

        x(2) = d*(a(0,0)*(a(1,1)*b(2) - b(1)*a(2,1)) -
                a(0,1)*(a(1,0)*b(2) - b(1)*a(2,0)) +
                b(0)*(a(1,0)*a(2,1) - a(1,1)*a(2,0)));
        return true;
    }
};

} // internal

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::randu(_Tp a, _Tp b)
{
    Matx<_Tp,m,n> M;
    cv::randu(M, Scalar(a), Scalar(b));
    return M;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::randn(_Tp a, _Tp b)
{
    Matx<_Tp,m,n> M;
    cv::randn(M, Scalar(a), Scalar(b));
    return M;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, n, m> Matx<_Tp, m, n>::inv(int method, bool *p_is_ok /*= NULL*/) const
{
    Matx<_Tp, n, m> b;
    bool ok;
    if( method == DECOMP_LU || method == DECOMP_CHOLESKY )
        ok = cv::internal::Matx_FastInvOp<_Tp, m>()(*this, b, method);
    else
    {
        Mat A(*this, false), B(b, false);
        ok = (invert(A, B, method) != 0);
    }
    if( NULL != p_is_ok ) { *p_is_ok = ok; }
    return ok ? b : Matx<_Tp, n, m>::zeros();
}

template<typename _Tp, int m, int n> template<int l> inline
Matx<_Tp, n, l> Matx<_Tp, m, n>::solve(const Matx<_Tp, m, l>& rhs, int method) const
{
    Matx<_Tp, n, l> x;
    bool ok;
    if( method == DECOMP_LU || method == DECOMP_CHOLESKY )
        ok = cv::internal::Matx_FastSolveOp<_Tp, m, l>()(*this, rhs, x, method);
    else
    {
        Mat A(*this, false), B(rhs, false), X(x, false);
        ok = cv::solve(A, B, X, method);
    }

    return ok ? x : Matx<_Tp, n, l>::zeros();
}



////////////////////////// Augmenting algebraic & logical operations //////////////////////////

#define CV_MAT_AUG_OPERATOR1(op, cvop, A, B) \
    static inline A& operator op (A& a, const B& b) { cvop; return a; }

#define CV_MAT_AUG_OPERATOR(op, cvop, A, B)   \
    CV_MAT_AUG_OPERATOR1(op, cvop, A, B)      \
    CV_MAT_AUG_OPERATOR1(op, cvop, const A, B)

#define CV_MAT_AUG_OPERATOR_T(op, cvop, A, B)                   \
    template<typename _Tp> CV_MAT_AUG_OPERATOR1(op, cvop, A, B) \
    template<typename _Tp> CV_MAT_AUG_OPERATOR1(op, cvop, const A, B)

CV_MAT_AUG_OPERATOR  (+=, cv::add(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (+=, cv::add(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (-=, cv::subtract(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (-=, cv::subtract(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat, Mat)
CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR  (*=, a.convertTo(a, -1, b), Mat, double)
CV_MAT_AUG_OPERATOR_T(*=, a.convertTo(a, -1, b), Mat_<_Tp>, double)

CV_MAT_AUG_OPERATOR  (/=, cv::divide(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR_T(/=, cv::divide(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(/=, cv::divide(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR  (/=, a.convertTo((Mat&)a, -1, 1./b), Mat, double)
CV_MAT_AUG_OPERATOR_T(/=, a.convertTo((Mat&)a, -1, 1./b), Mat_<_Tp>, double)

CV_MAT_AUG_OPERATOR  (&=, cv::bitwise_and(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (&=, cv::bitwise_and(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (|=, cv::bitwise_or(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (|=, cv::bitwise_or(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (^=, cv::bitwise_xor(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (^=, cv::bitwise_xor(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

#undef CV_MAT_AUG_OPERATOR_T
#undef CV_MAT_AUG_OPERATOR
#undef CV_MAT_AUG_OPERATOR1



///////////////////////////////////////////// SVD /////////////////////////////////////////////

inline SVD::SVD() {}
inline SVD::SVD( InputArray m, int flags ) { operator ()(m, flags); }
inline void SVD::solveZ( InputArray m, OutputArray _dst )
{
    Mat mtx = m.getMat();
    SVD svd(mtx, (mtx.rows >= mtx.cols ? 0 : SVD::FULL_UV));
    _dst.create(svd.vt.cols, 1, svd.vt.type());
    Mat dst = _dst.getMat();
    svd.vt.row(svd.vt.rows-1).reshape(1,svd.vt.cols).copyTo(dst);
}

template<typename _Tp, int m, int n, int nm> inline void
    SVD::compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt )
{
    CV_StaticAssert( nm == MIN(m, n), "Invalid size of output vector.");
    Mat _a(a, false), _u(u, false), _w(w, false), _vt(vt, false);
    SVD::compute(_a, _w, _u, _vt);
    CV_Assert(_w.data == (uchar*)&w.val[0] && _u.data == (uchar*)&u.val[0] && _vt.data == (uchar*)&vt.val[0]);
}

template<typename _Tp, int m, int n, int nm> inline void
SVD::compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w )
{
    CV_StaticAssert( nm == MIN(m, n), "Invalid size of output vector.");
    Mat _a(a, false), _w(w, false);
    SVD::compute(_a, _w);
    CV_Assert(_w.data == (uchar*)&w.val[0]);
}

template<typename _Tp, int m, int n, int nm, int nb> inline void
SVD::backSubst( const Matx<_Tp, nm, 1>& w, const Matx<_Tp, m, nm>& u,
                const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs,
                Matx<_Tp, n, nb>& dst )
{
    CV_StaticAssert( nm == MIN(m, n), "Invalid size of output vector.");
    Mat _u(u, false), _w(w, false), _vt(vt, false), _rhs(rhs, false), _dst(dst, false);
    SVD::backSubst(_w, _u, _vt, _rhs, _dst);
    CV_Assert(_dst.data == (uchar*)&dst.val[0]);
}



/////////////////////////////////// Multiply-with-Carry RNG ///////////////////////////////////

inline RNG::RNG()              { state = 0xffffffff; }
inline RNG::RNG(uint64 _state) { state = _state ? _state : 0xffffffff; }

inline RNG::operator uchar()    { return (uchar)next(); }
inline RNG::operator schar()    { return (schar)next(); }
inline RNG::operator ushort()   { return (ushort)next(); }
inline RNG::operator short()    { return (short)next(); }
inline RNG::operator int()      { return (int)next(); }
inline RNG::operator unsigned() { return next(); }
inline RNG::operator float()    { return next()*2.3283064365386962890625e-10f; }
inline RNG::operator double()   { unsigned t = next(); return (((uint64)t << 32) | next()) * 5.4210108624275221700372640043497e-20; }

inline unsigned RNG::operator ()(unsigned N) { return (unsigned)uniform(0,N); }
inline unsigned RNG::operator ()()           { return next(); }

inline int    RNG::uniform(int a, int b)       { return a == b ? a : (int)(next() % (b - a) + a); }
inline float  RNG::uniform(float a, float b)   { return ((float)*this)*(b - a) + a; }
inline double RNG::uniform(double a, double b) { return ((double)*this)*(b - a) + a; }

inline bool RNG::operator ==(const RNG& other) const { return state == other.state; }

inline unsigned RNG::next()
{
    state = (uint64)(unsigned)state* /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
    return (unsigned)state;
}

//! returns the next unifomly-distributed random number of the specified type
template<typename _Tp> static inline _Tp randu()
{
  return (_Tp)theRNG();
}

///////////////////////////////// Formatted string generation /////////////////////////////////

CV_EXPORTS String format( const char* fmt, ... );

///////////////////////////////// Formatted output of cv::Mat /////////////////////////////////

static inline
Ptr<Formatted> format(InputArray mtx, int fmt)
{
    return Formatter::get(fmt)->format(mtx.getMat());
}

static inline
int print(Ptr<Formatted> fmtd, FILE* stream = stdout)
{
    int written = 0;
    fmtd->reset();
    for(const char* str = fmtd->next(); str; str = fmtd->next())
        written += fputs(str, stream);

    return written;
}

static inline
int print(const Mat& mtx, FILE* stream = stdout)
{
    return print(Formatter::get()->format(mtx), stream);
}

static inline
int print(const UMat& mtx, FILE* stream = stdout)
{
    return print(Formatter::get()->format(mtx.getMat(ACCESS_READ)), stream);
}

template<typename _Tp> static inline
int print(const std::vector<Point_<_Tp> >& vec, FILE* stream = stdout)
{
    return print(Formatter::get()->format(Mat(vec)), stream);
}

template<typename _Tp> static inline
int print(const std::vector<Point3_<_Tp> >& vec, FILE* stream = stdout)
{
    return print(Formatter::get()->format(Mat(vec)), stream);
}

template<typename _Tp, int m, int n> static inline
int print(const Matx<_Tp, m, n>& matx, FILE* stream = stdout)
{
    return print(Formatter::get()->format(cv::Mat(matx)), stream);
}

//! @endcond

/****************************************************************************************\
*                                  Auxiliary algorithms                                  *
\****************************************************************************************/

/** @brief Splits an element set into equivalency classes.

The generic function partition implements an \f$O(N^2)\f$ algorithm for splitting a set of \f$N\f$ elements
into one or more equivalency classes, as described in
<http://en.wikipedia.org/wiki/Disjoint-set_data_structure> . The function returns the number of
equivalency classes.
@param _vec Set of elements stored as a vector.
@param labels Output vector of labels. It contains as many elements as vec. Each label labels[i] is
a 0-based cluster index of `vec[i]`.
@param predicate Equivalence predicate (pointer to a boolean function of two arguments or an
instance of the class that has the method bool operator()(const _Tp& a, const _Tp& b) ). The
predicate returns true when the elements are certainly in the same class, and returns false if they
may or may not be in the same class.
@ingroup core_cluster
*/
template<typename _Tp, class _EqPredicate> int
partition( const std::vector<_Tp>& _vec, std::vector<int>& labels,
          _EqPredicate predicate=_EqPredicate())
{
    int i, j, N = (int)_vec.size();
    const _Tp* vec = &_vec[0];

    const int PARENT=0;
    const int RANK=1;

    std::vector<int> _nodes(N*2);
    int (*nodes)[2] = (int(*)[2])&_nodes[0];

    // The first O(N) pass: create N single-vertex trees
    for(i = 0; i < N; i++)
    {
        nodes[i][PARENT]=-1;
        nodes[i][RANK] = 0;
    }

    // The main O(N^2) pass: merge connected components
    for( i = 0; i < N; i++ )
    {
        int root = i;

        // find root
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];

        for( j = 0; j < N; j++ )
        {
            if( i == j || !predicate(vec[i], vec[j]))
                continue;
            int root2 = j;

            while( nodes[root2][PARENT] >= 0 )
                root2 = nodes[root2][PARENT];

            if( root2 != root )
            {
                // unite both trees
                int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                if( rank > rank2 )
                    nodes[root2][PARENT] = root;
                else
                {
                    nodes[root][PARENT] = root2;
                    nodes[root2][RANK] += rank == rank2;
                    root = root2;
                }
                CV_Assert( nodes[root][PARENT] < 0 );

                int k = j, parent;

                // compress the path from node2 to root
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }

                // compress the path from node to root
                k = i;
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
            }
        }
    }

    // Final O(N) pass: enumerate classes
    labels.resize(N);
    int nclasses = 0;

    for( i = 0; i < N; i++ )
    {
        int root = i;
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];
        // re-use the rank as the class label
        if( nodes[root][RANK] >= 0 )
            nodes[root][RANK] = ~nclasses++;
        labels[i] = ~nodes[root][RANK];
    }

    return nclasses;
}

} // cv

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OPTIM_HPP
#define OPENCV_OPTIM_HPP

#include "opencv2/core.hpp"

namespace cv
{

/** @addtogroup core_optim
The algorithms in this section minimize or maximize function value within specified constraints or
without any constraints.
@{
*/

/** @brief Basic interface for all solvers
 */
class CV_EXPORTS MinProblemSolver : public Algorithm
{
public:
    /** @brief Represents function being optimized
     */
    class CV_EXPORTS Function
    {
    public:
        virtual ~Function() {}
        virtual int getDims() const = 0;
        virtual double getGradientEps() const;
        virtual double calc(const double* x) const = 0;
        virtual void getGradient(const double* x,double* grad);
    };

    /** @brief Getter for the optimized function.

    The optimized function is represented by Function interface, which requires derivatives to
    implement the calc(double*) and getDim() methods to evaluate the function.

    @return Smart-pointer to an object that implements Function interface - it represents the
    function that is being optimized. It can be empty, if no function was given so far.
     */
    virtual Ptr<Function> getFunction() const = 0;

    /** @brief Setter for the optimized function.

    *It should be called at least once before the call to* minimize(), as default value is not usable.

    @param f The new function to optimize.
     */
    virtual void setFunction(const Ptr<Function>& f) = 0;

    /** @brief Getter for the previously set terminal criteria for this algorithm.

    @return Deep copy of the terminal criteria used at the moment.
     */
    virtual TermCriteria getTermCriteria() const = 0;

    /** @brief Set terminal criteria for solver.

    This method *is not necessary* to be called before the first call to minimize(), as the default
    value is sensible.

    Algorithm stops when the number of function evaluations done exceeds termcrit.maxCount, when
    the function values at the vertices of simplex are within termcrit.epsilon range or simplex
    becomes so small that it can enclosed in a box with termcrit.epsilon sides, whatever comes
    first.
    @param termcrit Terminal criteria to be used, represented as cv::TermCriteria structure.
     */
    virtual void setTermCriteria(const TermCriteria& termcrit) = 0;

    /** @brief actually runs the algorithm and performs the minimization.

    The sole input parameter determines the centroid of the starting simplex (roughly, it tells
    where to start), all the others (terminal criteria, initial step, function to be minimized) are
    supposed to be set via the setters before the call to this method or the default values (not
    always sensible) will be used.

    @param x The initial point, that will become a centroid of an initial simplex. After the algorithm
    will terminate, it will be setted to the point where the algorithm stops, the point of possible
    minimum.
    @return The value of a function at the point found.
     */
    virtual double minimize(InputOutputArray x) = 0;
};

/** @brief This class is used to perform the non-linear non-constrained minimization of a function,

defined on an `n`-dimensional Euclidean space, using the **Nelder-Mead method**, also known as
**downhill simplex method**. The basic idea about the method can be obtained from
<http://en.wikipedia.org/wiki/Nelder-Mead_method>.

It should be noted, that this method, although deterministic, is rather a heuristic and therefore
may converge to a local minima, not necessary a global one. It is iterative optimization technique,
which at each step uses an information about the values of a function evaluated only at `n+1`
points, arranged as a *simplex* in `n`-dimensional space (hence the second name of the method). At
each step new point is chosen to evaluate function at, obtained value is compared with previous
ones and based on this information simplex changes it's shape , slowly moving to the local minimum.
Thus this method is using *only* function values to make decision, on contrary to, say, Nonlinear
Conjugate Gradient method (which is also implemented in optim).

Algorithm stops when the number of function evaluations done exceeds termcrit.maxCount, when the
function values at the vertices of simplex are within termcrit.epsilon range or simplex becomes so
small that it can enclosed in a box with termcrit.epsilon sides, whatever comes first, for some
defined by user positive integer termcrit.maxCount and positive non-integer termcrit.epsilon.

@note DownhillSolver is a derivative of the abstract interface
cv::MinProblemSolver, which in turn is derived from the Algorithm interface and is used to
encapsulate the functionality, common to all non-linear optimization algorithms in the optim
module.

@note term criteria should meet following condition:
@code
    termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0
@endcode
 */
class CV_EXPORTS DownhillSolver : public MinProblemSolver
{
public:
    /** @brief Returns the initial step that will be used in downhill simplex algorithm.

    @param step Initial step that will be used in algorithm. Note, that although corresponding setter
    accepts column-vectors as well as row-vectors, this method will return a row-vector.
    @see DownhillSolver::setInitStep
     */
    virtual void getInitStep(OutputArray step) const=0;

    /** @brief Sets the initial step that will be used in downhill simplex algorithm.

    Step, together with initial point (givin in DownhillSolver::minimize) are two `n`-dimensional
    vectors that are used to determine the shape of initial simplex. Roughly said, initial point
    determines the position of a simplex (it will become simplex's centroid), while step determines the
    spread (size in each dimension) of a simplex. To be more precise, if \f$s,x_0\in\mathbb{R}^n\f$ are
    the initial step and initial point respectively, the vertices of a simplex will be:
    \f$v_0:=x_0-\frac{1}{2} s\f$ and \f$v_i:=x_0+s_i\f$ for \f$i=1,2,\dots,n\f$ where \f$s_i\f$ denotes
    projections of the initial step of *n*-th coordinate (the result of projection is treated to be
    vector given by \f$s_i:=e_i\cdot\left<e_i\cdot s\right>\f$, where \f$e_i\f$ form canonical basis)

    @param step Initial step that will be used in algorithm. Roughly said, it determines the spread
    (size in each dimension) of an initial simplex.
     */
    virtual void setInitStep(InputArray step)=0;

    /** @brief This function returns the reference to the ready-to-use DownhillSolver object.

    All the parameters are optional, so this procedure can be called even without parameters at
    all. In this case, the default values will be used. As default value for terminal criteria are
    the only sensible ones, MinProblemSolver::setFunction() and DownhillSolver::setInitStep()
    should be called upon the obtained object, if the respective parameters were not given to
    create(). Otherwise, the two ways (give parameters to createDownhillSolver() or miss them out
    and call the MinProblemSolver::setFunction() and DownhillSolver::setInitStep()) are absolutely
    equivalent (and will drop the same errors in the same way, should invalid input be detected).
    @param f Pointer to the function that will be minimized, similarly to the one you submit via
    MinProblemSolver::setFunction.
    @param initStep Initial step, that will be used to construct the initial simplex, similarly to the one
    you submit via MinProblemSolver::setInitStep.
    @param termcrit Terminal criteria to the algorithm, similarly to the one you submit via
    MinProblemSolver::setTermCriteria.
     */
    static Ptr<DownhillSolver> create(const Ptr<MinProblemSolver::Function>& f=Ptr<MinProblemSolver::Function>(),
                                      InputArray initStep=Mat_<double>(1,1,0.0),
                                      TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5000,0.000001));
};

/** @brief This class is used to perform the non-linear non-constrained minimization of a function
with known gradient,

defined on an *n*-dimensional Euclidean space, using the **Nonlinear Conjugate Gradient method**.
The implementation was done based on the beautifully clear explanatory article [An Introduction to
the Conjugate Gradient Method Without the Agonizing
Pain](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) by Jonathan Richard
Shewchuk. The method can be seen as an adaptation of a standard Conjugate Gradient method (see, for
example <http://en.wikipedia.org/wiki/Conjugate_gradient_method>) for numerically solving the
systems of linear equations.

It should be noted, that this method, although deterministic, is rather a heuristic method and
therefore may converge to a local minima, not necessary a global one. What is even more disastrous,
most of its behaviour is ruled by gradient, therefore it essentially cannot distinguish between
local minima and maxima. Therefore, if it starts sufficiently near to the local maximum, it may
converge to it. Another obvious restriction is that it should be possible to compute the gradient of
a function at any point, thus it is preferable to have analytic expression for gradient and
computational burden should be born by the user.

The latter responsibility is accompilished via the getGradient method of a
MinProblemSolver::Function interface (which represents function being optimized). This method takes
point a point in *n*-dimensional space (first argument represents the array of coordinates of that
point) and comput its gradient (it should be stored in the second argument as an array).

@note class ConjGradSolver thus does not add any new methods to the basic MinProblemSolver interface.

@note term criteria should meet following condition:
@code
    termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0
    // or
    termcrit.type == TermCriteria::MAX_ITER) && termcrit.maxCount > 0
@endcode
 */
class CV_EXPORTS ConjGradSolver : public MinProblemSolver
{
public:
    /** @brief This function returns the reference to the ready-to-use ConjGradSolver object.

    All the parameters are optional, so this procedure can be called even without parameters at
    all. In this case, the default values will be used. As default value for terminal criteria are
    the only sensible ones, MinProblemSolver::setFunction() should be called upon the obtained
    object, if the function was not given to create(). Otherwise, the two ways (submit it to
    create() or miss it out and call the MinProblemSolver::setFunction()) are absolutely equivalent
    (and will drop the same errors in the same way, should invalid input be detected).
    @param f Pointer to the function that will be minimized, similarly to the one you submit via
    MinProblemSolver::setFunction.
    @param termcrit Terminal criteria to the algorithm, similarly to the one you submit via
    MinProblemSolver::setTermCriteria.
    */
    static Ptr<ConjGradSolver> create(const Ptr<MinProblemSolver::Function>& f=Ptr<ConjGradSolver::Function>(),
                                      TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5000,0.000001));
};

//! return codes for cv::solveLP() function
enum SolveLPResult
{
    SOLVELP_UNBOUNDED    = -2, //!< problem is unbounded (target function can achieve arbitrary high values)
    SOLVELP_UNFEASIBLE    = -1, //!< problem is unfeasible (there are no points that satisfy all the constraints imposed)
    SOLVELP_SINGLE    = 0, //!< there is only one maximum for target function
    SOLVELP_MULTI    = 1 //!< there are multiple maxima for target function - the arbitrary one is returned
};

/** @brief Solve given (non-integer) linear programming problem using the Simplex Algorithm (Simplex Method).

What we mean here by "linear programming problem" (or LP problem, for short) can be formulated as:

\f[\mbox{Maximize } c\cdot x\\
 \mbox{Subject to:}\\
 Ax\leq b\\
 x\geq 0\f]

Where \f$c\f$ is fixed `1`-by-`n` row-vector, \f$A\f$ is fixed `m`-by-`n` matrix, \f$b\f$ is fixed `m`-by-`1`
column vector and \f$x\f$ is an arbitrary `n`-by-`1` column vector, which satisfies the constraints.

Simplex algorithm is one of many algorithms that are designed to handle this sort of problems
efficiently. Although it is not optimal in theoretical sense (there exist algorithms that can solve
any problem written as above in polynomial time, while simplex method degenerates to exponential
time for some special cases), it is well-studied, easy to implement and is shown to work well for
real-life purposes.

The particular implementation is taken almost verbatim from **Introduction to Algorithms, third
edition** by T. H. Cormen, C. E. Leiserson, R. L. Rivest and Clifford Stein. In particular, the
Bland's rule <http://en.wikipedia.org/wiki/Bland%27s_rule> is used to prevent cycling.

@param Func This row-vector corresponds to \f$c\f$ in the LP problem formulation (see above). It should
contain 32- or 64-bit floating point numbers. As a convenience, column-vector may be also submitted,
in the latter case it is understood to correspond to \f$c^T\f$.
@param Constr `m`-by-`n+1` matrix, whose rightmost column corresponds to \f$b\f$ in formulation above
and the remaining to \f$A\f$. It should containt 32- or 64-bit floating point numbers.
@param z The solution will be returned here as a column-vector - it corresponds to \f$c\f$ in the
formulation above. It will contain 64-bit floating point numbers.
@return One of cv::SolveLPResult
 */
CV_EXPORTS_W int solveLP(const Mat& Func, const Mat& Constr, Mat& z);

//! @}

}// cv

#endif
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// OpenVX related definitions and declarations

#pragma once
#ifndef OPENCV_OVX_HPP
#define OPENCV_OVX_HPP

#include "cvdef.h"

namespace cv
{
/// Check if use of OpenVX is possible
CV_EXPORTS_W bool haveOpenVX();

/// Check if use of OpenVX is enabled
CV_EXPORTS_W bool useOpenVX();

/// Enable/disable use of OpenVX
CV_EXPORTS_W void setUseOpenVX(bool flag);
} // namespace cv

#endif // OPENCV_OVX_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_PERSISTENCE_HPP
#define OPENCV_CORE_PERSISTENCE_HPP

#ifndef __cplusplus
#  error persistence.hpp header must be compiled as C++
#endif

//! @addtogroup core_c
//! @{

/** @brief "black box" representation of the file storage associated with a file on disk.

Several functions that are described below take CvFileStorage\* as inputs and allow the user to
save or to load hierarchical collections that consist of scalar values, standard CXCore objects
(such as matrices, sequences, graphs), and user-defined objects.

OpenCV can read and write data in XML (<http://www.w3c.org/XML>), YAML (<http://www.yaml.org>) or
JSON (<http://www.json.org/>) formats. Below is an example of 3x3 floating-point identity matrix A,
stored in XML and YAML files
using CXCore functions:
XML:
@code{.xml}
    <?xml version="1.0">
    <opencv_storage>
    <A type_id="opencv-matrix">
      <rows>3</rows>
      <cols>3</cols>
      <dt>f</dt>
      <data>1. 0. 0. 0. 1. 0. 0. 0. 1.</data>
    </A>
    </opencv_storage>
@endcode
YAML:
@code{.yaml}
    %YAML:1.0
    A: !!opencv-matrix
      rows: 3
      cols: 3
      dt: f
      data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1.]
@endcode
As it can be seen from the examples, XML uses nested tags to represent hierarchy, while YAML uses
indentation for that purpose (similar to the Python programming language).

The same functions can read and write data in both formats; the particular format is determined by
the extension of the opened file, ".xml" for XML files, ".yml" or ".yaml" for YAML and ".json" for
JSON.
 */
typedef struct CvFileStorage CvFileStorage;
typedef struct CvFileNode CvFileNode;
typedef struct CvMat CvMat;
typedef struct CvMatND CvMatND;

//! @} core_c

#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"

namespace cv {

/** @addtogroup core_xml

XML/YAML/JSON file storages.     {#xml_storage}
=======================
Writing to a file storage.
--------------------------
You can store and then restore various OpenCV data structures to/from XML (<http://www.w3c.org/XML>),
YAML (<http://www.yaml.org>) or JSON (<http://www.json.org/>) formats. Also, it is possible store
and load arbitrarily complex data structures, which include OpenCV data structures, as well as
primitive data types (integer and floating-point numbers and text strings) as their elements.

Use the following procedure to write something to XML, YAML or JSON:
-# Create new FileStorage and open it for writing. It can be done with a single call to
FileStorage::FileStorage constructor that takes a filename, or you can use the default constructor
and then call FileStorage::open. Format of the file (XML, YAML or JSON) is determined from the filename
extension (".xml", ".yml"/".yaml" and ".json", respectively)
-# Write all the data you want using the streaming operator `<<`, just like in the case of STL
streams.
-# Close the file using FileStorage::release. FileStorage destructor also closes the file.

Here is an example:
@code
    #include "opencv2/opencv.hpp"
    #include <time.h>

    using namespace cv;

    int main(int, char** argv)
    {
        FileStorage fs("test.yml", FileStorage::WRITE);

        fs << "frameCount" << 5;
        time_t rawtime; time(&rawtime);
        fs << "calibrationDate" << asctime(localtime(&rawtime));
        Mat cameraMatrix = (Mat_<double>(3,3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
        Mat distCoeffs = (Mat_<double>(5,1) << 0.1, 0.01, -0.001, 0, 0);
        fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
        fs << "features" << "[";
        for( int i = 0; i < 3; i++ )
        {
            int x = rand() % 640;
            int y = rand() % 480;
            uchar lbp = rand() % 256;

            fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
            for( int j = 0; j < 8; j++ )
                fs << ((lbp >> j) & 1);
            fs << "]" << "}";
        }
        fs << "]";
        fs.release();
        return 0;
    }
@endcode
The sample above stores to XML and integer, text string (calibration date), 2 matrices, and a custom
structure "feature", which includes feature coordinates and LBP (local binary pattern) value. Here
is output of the sample:
@code{.yaml}
%YAML:1.0
frameCount: 5
calibrationDate: "Fri Jun 17 14:09:29 2011\n"
cameraMatrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000., 0., 320., 0., 1000., 240., 0., 0., 1. ]
distCoeffs: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ 1.0000000000000001e-01, 1.0000000000000000e-02,
       -1.0000000000000000e-03, 0., 0. ]
features:
   - { x:167, y:49, lbp:[ 1, 0, 0, 1, 1, 0, 1, 1 ] }
   - { x:298, y:130, lbp:[ 0, 0, 0, 1, 0, 0, 1, 1 ] }
   - { x:344, y:158, lbp:[ 1, 1, 0, 0, 0, 0, 1, 0 ] }
@endcode

As an exercise, you can replace ".yml" with ".xml" or ".json" in the sample above and see, how the
corresponding XML file will look like.

Several things can be noted by looking at the sample code and the output:

-   The produced YAML (and XML/JSON) consists of heterogeneous collections that can be nested. There are
    2 types of collections: named collections (mappings) and unnamed collections (sequences). In mappings
    each element has a name and is accessed by name. This is similar to structures and std::map in
    C/C++ and dictionaries in Python. In sequences elements do not have names, they are accessed by
    indices. This is similar to arrays and std::vector in C/C++ and lists, tuples in Python.
    "Heterogeneous" means that elements of each single collection can have different types.

    Top-level collection in YAML/XML/JSON is a mapping. Each matrix is stored as a mapping, and the matrix
    elements are stored as a sequence. Then, there is a sequence of features, where each feature is
    represented a mapping, and lbp value in a nested sequence.

-   When you write to a mapping (a structure), you write element name followed by its value. When you
    write to a sequence, you simply write the elements one by one. OpenCV data structures (such as
    cv::Mat) are written in absolutely the same way as simple C data structures - using `<<`
    operator.

-   To write a mapping, you first write the special string `{` to the storage, then write the
    elements as pairs (`fs << <element_name> << <element_value>`) and then write the closing
    `}`.

-   To write a sequence, you first write the special string `[`, then write the elements, then
    write the closing `]`.

-   In YAML/JSON (but not XML), mappings and sequences can be written in a compact Python-like inline
    form. In the sample above matrix elements, as well as each feature, including its lbp value, is
    stored in such inline form. To store a mapping/sequence in a compact form, put `:` after the
    opening character, e.g. use `{:` instead of `{` and `[:` instead of `[`. When the
    data is written to XML, those extra `:` are ignored.

Reading data from a file storage.
---------------------------------
To read the previously written XML, YAML or JSON file, do the following:
-#  Open the file storage using FileStorage::FileStorage constructor or FileStorage::open method.
    In the current implementation the whole file is parsed and the whole representation of file
    storage is built in memory as a hierarchy of file nodes (see FileNode)

-#  Read the data you are interested in. Use FileStorage::operator [], FileNode::operator []
    and/or FileNodeIterator.

-#  Close the storage using FileStorage::release.

Here is how to read the file created by the code sample above:
@code
    FileStorage fs2("test.yml", FileStorage::READ);

    // first method: use (type) operator on FileNode.
    int frameCount = (int)fs2["frameCount"];

    String date;
    // second method: use FileNode::operator >>
    fs2["calibrationDate"] >> date;

    Mat cameraMatrix2, distCoeffs2;
    fs2["cameraMatrix"] >> cameraMatrix2;
    fs2["distCoeffs"] >> distCoeffs2;

    cout << "frameCount: " << frameCount << endl
         << "calibration date: " << date << endl
         << "camera matrix: " << cameraMatrix2 << endl
         << "distortion coeffs: " << distCoeffs2 << endl;

    FileNode features = fs2["features"];
    FileNodeIterator it = features.begin(), it_end = features.end();
    int idx = 0;
    std::vector<uchar> lbpval;

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it, idx++ )
    {
        cout << "feature #" << idx << ": ";
        cout << "x=" << (int)(*it)["x"] << ", y=" << (int)(*it)["y"] << ", lbp: (";
        // you can also easily read numerical arrays using FileNode >> std::vector operator.
        (*it)["lbp"] >> lbpval;
        for( int i = 0; i < (int)lbpval.size(); i++ )
            cout << " " << (int)lbpval[i];
        cout << ")" << endl;
    }
    fs2.release();
@endcode

Format specification    {#format_spec}
--------------------
`([count]{u|c|w|s|i|f|d})`... where the characters correspond to fundamental C++ types:
-   `u` 8-bit unsigned number
-   `c` 8-bit signed number
-   `w` 16-bit unsigned number
-   `s` 16-bit signed number
-   `i` 32-bit signed number
-   `f` single precision floating-point number
-   `d` double precision floating-point number
-   `r` pointer, 32 lower bits of which are written as a signed integer. The type can be used to
    store structures with links between the elements.

`count` is the optional counter of values of a given type. For example, `2if` means that each array
element is a structure of 2 integers, followed by a single-precision floating-point number. The
equivalent notations of the above specification are `iif`, `2i1f` and so forth. Other examples: `u`
means that the array consists of bytes, and `2d` means the array consists of pairs of doubles.

@see @ref filestorage.cpp
*/

//! @{

/** @example filestorage.cpp
A complete example using the FileStorage interface
*/

////////////////////////// XML & YAML I/O //////////////////////////

class CV_EXPORTS FileNode;
class CV_EXPORTS FileNodeIterator;

/** @brief XML/YAML/JSON file storage class that encapsulates all the information necessary for writing or
reading data to/from a file.
 */
class CV_EXPORTS_W FileStorage
{
public:
    //! file storage mode
    enum Mode
    {
        READ        = 0, //!< value, open the file for reading
        WRITE       = 1, //!< value, open the file for writing
        APPEND      = 2, //!< value, open the file for appending
        MEMORY      = 4, //!< flag, read data from source or write data to the internal buffer (which is
                         //!< returned by FileStorage::release)
        FORMAT_MASK = (7<<3), //!< mask for format flags
        FORMAT_AUTO = 0,      //!< flag, auto format
        FORMAT_XML  = (1<<3), //!< flag, XML format
        FORMAT_YAML = (2<<3), //!< flag, YAML format
        FORMAT_JSON = (3<<3), //!< flag, JSON format

        BASE64      = 64,     //!< flag, write rawdata in Base64 by default. (consider using WRITE_BASE64)
        WRITE_BASE64 = BASE64 | WRITE, //!< flag, enable both WRITE and BASE64
    };
    enum
    {
        UNDEFINED      = 0,
        VALUE_EXPECTED = 1,
        NAME_EXPECTED  = 2,
        INSIDE_MAP     = 4
    };

    /** @brief The constructors.

    The full constructor opens the file. Alternatively you can use the default constructor and then
    call FileStorage::open.
     */
    CV_WRAP FileStorage();

    /** @overload
    @param source Name of the file to open or the text string to read the data from. Extension of the
    file (.xml, .yml/.yaml, or .json) determines its format (XML, YAML or JSON respectively). Also you can
    append .gz to work with compressed files, for example myHugeMatrix.xml.gz. If both FileStorage::WRITE
    and FileStorage::MEMORY flags are specified, source is used just to specify the output file format (e.g.
    mydata.xml, .yml etc.).
    @param flags Mode of operation. See  FileStorage::Mode
    @param encoding Encoding of the file. Note that UTF-16 XML encoding is not supported currently and
    you should use 8-bit encoding instead of it.
    */
    CV_WRAP FileStorage(const String& source, int flags, const String& encoding=String());

    /** @overload */
    FileStorage(CvFileStorage* fs, bool owning=true);

    //! the destructor. calls release()
    virtual ~FileStorage();

    /** @brief Opens a file.

    See description of parameters in FileStorage::FileStorage. The method calls FileStorage::release
    before opening the file.
    @param filename Name of the file to open or the text string to read the data from.
       Extension of the file (.xml, .yml/.yaml or .json) determines its format (XML, YAML or JSON
        respectively). Also you can append .gz to work with compressed files, for example myHugeMatrix.xml.gz. If both
        FileStorage::WRITE and FileStorage::MEMORY flags are specified, source is used just to specify
        the output file format (e.g. mydata.xml, .yml etc.). A file name can also contain parameters.
        You can use this format, "*?base64" (e.g. "file.json?base64" (case sensitive)), as an alternative to
        FileStorage::BASE64 flag.
    @param flags Mode of operation. One of FileStorage::Mode
    @param encoding Encoding of the file. Note that UTF-16 XML encoding is not supported currently and
    you should use 8-bit encoding instead of it.
     */
    CV_WRAP virtual bool open(const String& filename, int flags, const String& encoding=String());

    /** @brief Checks whether the file is opened.

    @returns true if the object is associated with the current file and false otherwise. It is a
    good practice to call this method after you tried to open a file.
     */
    CV_WRAP virtual bool isOpened() const;

    /** @brief Closes the file and releases all the memory buffers.

    Call this method after all I/O operations with the storage are finished.
     */
    CV_WRAP virtual void release();

    /** @brief Closes the file and releases all the memory buffers.

    Call this method after all I/O operations with the storage are finished. If the storage was
    opened for writing data and FileStorage::WRITE was specified
     */
    CV_WRAP virtual String releaseAndGetString();

    /** @brief Returns the first element of the top-level mapping.
    @returns The first element of the top-level mapping.
     */
    CV_WRAP FileNode getFirstTopLevelNode() const;

    /** @brief Returns the top-level mapping
    @param streamidx Zero-based index of the stream. In most cases there is only one stream in the file.
    However, YAML supports multiple streams and so there can be several.
    @returns The top-level mapping.
     */
    CV_WRAP FileNode root(int streamidx=0) const;

    /** @brief Returns the specified element of the top-level mapping.
    @param nodename Name of the file node.
    @returns Node with the given name.
     */
    FileNode operator[](const String& nodename) const;

    /** @overload */
    CV_WRAP_AS(getNode) FileNode operator[](const char* nodename) const;

    /** @brief Returns the obsolete C FileStorage structure.
    @returns Pointer to the underlying C FileStorage structure
     */
    CvFileStorage* operator *() { return fs.get(); }

    /** @overload */
    const CvFileStorage* operator *() const { return fs.get(); }

    /** @brief Writes multiple numbers.

    Writes one or more numbers of the specified format to the currently written structure. Usually it is
    more convenient to use operator `<<` instead of this method.
    @param fmt Specification of each array element, see @ref format_spec "format specification"
    @param vec Pointer to the written array.
    @param len Number of the uchar elements to write.
     */
    void writeRaw( const String& fmt, const uchar* vec, size_t len );

    /** @brief Writes the registered C structure (CvMat, CvMatND, CvSeq).
    @param name Name of the written object.
    @param obj Pointer to the object.
    @see ocvWrite for details.
     */
    void writeObj( const String& name, const void* obj );

    /**
     * @brief Simplified writing API to use with bindings.
     * @param name Name of the written object
     * @param val Value of the written object
     */
    CV_WRAP void write(const String& name, double val);
    /// @overload
    CV_WRAP void write(const String& name, const String& val);
    /// @overload
    CV_WRAP void write(const String& name, InputArray val);

    /** @brief Writes a comment.

    The function writes a comment into file storage. The comments are skipped when the storage is read.
    @param comment The written comment, single-line or multi-line
    @param append If true, the function tries to put the comment at the end of current line.
    Else if the comment is multi-line, or if it does not fit at the end of the current
    line, the comment starts a new line.
     */
    CV_WRAP void writeComment(const String& comment, bool append = false);

    /** @brief Returns the normalized object name for the specified name of a file.
    @param filename Name of a file
    @returns The normalized object name.
     */
    static String getDefaultObjectName(const String& filename);

    /** @brief Returns the current format.
     * @returns The current format, see FileStorage::Mode
     */
    CV_WRAP int getFormat() const;

    Ptr<CvFileStorage> fs; //!< the underlying C FileStorage structure
    String elname; //!< the currently written element
    std::vector<char> structs; //!< the stack of written structures
    int state; //!< the writer state
};

template<> CV_EXPORTS void DefaultDeleter<CvFileStorage>::operator ()(CvFileStorage* obj) const;

/** @brief File Storage Node class.

The node is used to store each and every element of the file storage opened for reading. When
XML/YAML file is read, it is first parsed and stored in the memory as a hierarchical collection of
nodes. Each node can be a "leaf" that is contain a single number or a string, or be a collection of
other nodes. There can be named collections (mappings) where each element has a name and it is
accessed by a name, and ordered collections (sequences) where elements do not have names but rather
accessed by index. Type of the file node can be determined using FileNode::type method.

Note that file nodes are only used for navigating file storages opened for reading. When a file
storage is opened for writing, no data is stored in memory after it is written.
 */
class CV_EXPORTS_W_SIMPLE FileNode
{
public:
    //! type of the file storage node
    enum Type
    {
        NONE      = 0, //!< empty node
        INT       = 1, //!< an integer
        REAL      = 2, //!< floating-point number
        FLOAT     = REAL, //!< synonym or REAL
        STR       = 3, //!< text string in UTF-8 encoding
        STRING    = STR, //!< synonym for STR
        REF       = 4, //!< integer of size size_t. Typically used for storing complex dynamic structures where some elements reference the others
        SEQ       = 5, //!< sequence
        MAP       = 6, //!< mapping
        TYPE_MASK = 7,
        FLOW      = 8,  //!< compact representation of a sequence or mapping. Used only by YAML writer
        USER      = 16, //!< a registered object (e.g. a matrix)
        EMPTY     = 32, //!< empty structure (sequence or mapping)
        NAMED     = 64  //!< the node has a name (i.e. it is element of a mapping)
    };
    /** @brief The constructors.

    These constructors are used to create a default file node, construct it from obsolete structures or
    from the another file node.
     */
    CV_WRAP FileNode();

    /** @overload
    @param fs Pointer to the obsolete file storage structure.
    @param node File node to be used as initialization for the created file node.
    */
    FileNode(const CvFileStorage* fs, const CvFileNode* node);

    /** @overload
    @param node File node to be used as initialization for the created file node.
    */
    FileNode(const FileNode& node);

    /** @brief Returns element of a mapping node or a sequence node.
    @param nodename Name of an element in the mapping node.
    @returns Returns the element with the given identifier.
     */
    FileNode operator[](const String& nodename) const;

    /** @overload
    @param nodename Name of an element in the mapping node.
    */
    CV_WRAP_AS(getNode) FileNode operator[](const char* nodename) const;

    /** @overload
    @param i Index of an element in the sequence node.
    */
    CV_WRAP_AS(at) FileNode operator[](int i) const;

    /** @brief Returns type of the node.
    @returns Type of the node. See FileNode::Type
     */
    CV_WRAP int type() const;

    //! returns true if the node is empty
    CV_WRAP bool empty() const;
    //! returns true if the node is a "none" object
    CV_WRAP bool isNone() const;
    //! returns true if the node is a sequence
    CV_WRAP bool isSeq() const;
    //! returns true if the node is a mapping
    CV_WRAP bool isMap() const;
    //! returns true if the node is an integer
    CV_WRAP bool isInt() const;
    //! returns true if the node is a floating-point number
    CV_WRAP bool isReal() const;
    //! returns true if the node is a text string
    CV_WRAP bool isString() const;
    //! returns true if the node has a name
    CV_WRAP bool isNamed() const;
    //! returns the node name or an empty string if the node is nameless
    CV_WRAP String name() const;
    //! returns the number of elements in the node, if it is a sequence or mapping, or 1 otherwise.
    CV_WRAP size_t size() const;
    //! returns the node content as an integer. If the node stores floating-point number, it is rounded.
    operator int() const;
    //! returns the node content as float
    operator float() const;
    //! returns the node content as double
    operator double() const;
    //! returns the node content as text string
    operator String() const;
    operator std::string() const;

    //! returns pointer to the underlying file node
    CvFileNode* operator *();
    //! returns pointer to the underlying file node
    const CvFileNode* operator* () const;

    //! returns iterator pointing to the first node element
    FileNodeIterator begin() const;
    //! returns iterator pointing to the element following the last node element
    FileNodeIterator end() const;

    /** @brief Reads node elements to the buffer with the specified format.

    Usually it is more convenient to use operator `>>` instead of this method.
    @param fmt Specification of each array element. See @ref format_spec "format specification"
    @param vec Pointer to the destination array.
    @param len Number of elements to read. If it is greater than number of remaining elements then all
    of them will be read.
     */
    void readRaw( const String& fmt, uchar* vec, size_t len ) const;

    //! reads the registered object and returns pointer to it
    void* readObj() const;

    //! Simplified reading API to use with bindings.
    CV_WRAP double real() const;
    //! Simplified reading API to use with bindings.
    CV_WRAP String string() const;
    //! Simplified reading API to use with bindings.
    CV_WRAP Mat mat() const;

    // do not use wrapper pointer classes for better efficiency
    const CvFileStorage* fs;
    const CvFileNode* node;
};


/** @brief used to iterate through sequences and mappings.

A standard STL notation, with node.begin(), node.end() denoting the beginning and the end of a
sequence, stored in node. See the data reading sample in the beginning of the section.
 */
class CV_EXPORTS FileNodeIterator
{
public:
    /** @brief The constructors.

    These constructors are used to create a default iterator, set it to specific element in a file node
    or construct it from another iterator.
     */
    FileNodeIterator();

    /** @overload
    @param fs File storage for the iterator.
    @param node File node for the iterator.
    @param ofs Index of the element in the node. The created iterator will point to this element.
    */
    FileNodeIterator(const CvFileStorage* fs, const CvFileNode* node, size_t ofs=0);

    /** @overload
    @param it Iterator to be used as initialization for the created iterator.
    */
    FileNodeIterator(const FileNodeIterator& it);

    //! returns the currently observed element
    FileNode operator *() const;
    //! accesses the currently observed element methods
    FileNode operator ->() const;

    //! moves iterator to the next node
    FileNodeIterator& operator ++ ();
    //! moves iterator to the next node
    FileNodeIterator operator ++ (int);
    //! moves iterator to the previous node
    FileNodeIterator& operator -- ();
    //! moves iterator to the previous node
    FileNodeIterator operator -- (int);
    //! moves iterator forward by the specified offset (possibly negative)
    FileNodeIterator& operator += (int ofs);
    //! moves iterator backward by the specified offset (possibly negative)
    FileNodeIterator& operator -= (int ofs);

    /** @brief Reads node elements to the buffer with the specified format.

    Usually it is more convenient to use operator `>>` instead of this method.
    @param fmt Specification of each array element. See @ref format_spec "format specification"
    @param vec Pointer to the destination array.
    @param maxCount Number of elements to read. If it is greater than number of remaining elements then
    all of them will be read.
     */
    FileNodeIterator& readRaw( const String& fmt, uchar* vec,
                               size_t maxCount=(size_t)INT_MAX );

    struct SeqReader
    {
      int          header_size;
      void*        seq;        /* sequence, beign read; CvSeq      */
      void*        block;      /* current block;        CvSeqBlock */
      schar*       ptr;        /* pointer to element be read next */
      schar*       block_min;  /* pointer to the beginning of block */
      schar*       block_max;  /* pointer to the end of block */
      int          delta_index;/* = seq->first->start_index   */
      schar*       prev_elem;  /* pointer to previous element */
    };

    const CvFileStorage* fs;
    const CvFileNode* container;
    SeqReader reader;
    size_t remaining;
};

//! @} core_xml

/////////////////// XML & YAML I/O implementation //////////////////

//! @relates cv::FileStorage
//! @{

CV_EXPORTS void write( FileStorage& fs, const String& name, int value );
CV_EXPORTS void write( FileStorage& fs, const String& name, float value );
CV_EXPORTS void write( FileStorage& fs, const String& name, double value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const String& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const Mat& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const SparseMat& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const std::vector<KeyPoint>& value);
CV_EXPORTS void write( FileStorage& fs, const String& name, const std::vector<DMatch>& value);

CV_EXPORTS void writeScalar( FileStorage& fs, int value );
CV_EXPORTS void writeScalar( FileStorage& fs, float value );
CV_EXPORTS void writeScalar( FileStorage& fs, double value );
CV_EXPORTS void writeScalar( FileStorage& fs, const String& value );

//! @}

//! @relates cv::FileNode
//! @{

CV_EXPORTS void read(const FileNode& node, int& value, int default_value);
CV_EXPORTS void read(const FileNode& node, float& value, float default_value);
CV_EXPORTS void read(const FileNode& node, double& value, double default_value);
CV_EXPORTS void read(const FileNode& node, String& value, const String& default_value);
CV_EXPORTS void read(const FileNode& node, std::string& value, const std::string& default_value);
CV_EXPORTS void read(const FileNode& node, Mat& mat, const Mat& default_mat = Mat() );
CV_EXPORTS void read(const FileNode& node, SparseMat& mat, const SparseMat& default_mat = SparseMat() );
CV_EXPORTS void read(const FileNode& node, std::vector<KeyPoint>& keypoints);
CV_EXPORTS void read(const FileNode& node, std::vector<DMatch>& matches);

template<typename _Tp> static inline void read(const FileNode& node, Point_<_Tp>& value, const Point_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 2 ? default_value : Point_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]));
}

template<typename _Tp> static inline void read(const FileNode& node, Point3_<_Tp>& value, const Point3_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 3 ? default_value : Point3_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]),
                                                            saturate_cast<_Tp>(temp[2]));
}

template<typename _Tp> static inline void read(const FileNode& node, Size_<_Tp>& value, const Size_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 2 ? default_value : Size_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]));
}

template<typename _Tp> static inline void read(const FileNode& node, Complex<_Tp>& value, const Complex<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 2 ? default_value : Complex<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]));
}

template<typename _Tp> static inline void read(const FileNode& node, Rect_<_Tp>& value, const Rect_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 4 ? default_value : Rect_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]),
                                                          saturate_cast<_Tp>(temp[2]), saturate_cast<_Tp>(temp[3]));
}

template<typename _Tp, int cn> static inline void read(const FileNode& node, Vec<_Tp, cn>& value, const Vec<_Tp, cn>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != cn ? default_value : Vec<_Tp, cn>(&temp[0]);
}

template<typename _Tp> static inline void read(const FileNode& node, Scalar_<_Tp>& value, const Scalar_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 4 ? default_value : Scalar_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]),
                                                            saturate_cast<_Tp>(temp[2]), saturate_cast<_Tp>(temp[3]));
}

static inline void read(const FileNode& node, Range& value, const Range& default_value)
{
    Point2i temp(value.start, value.end); const Point2i default_temp = Point2i(default_value.start, default_value.end);
    read(node, temp, default_temp);
    value.start = temp.x; value.end = temp.y;
}

//! @}

/** @brief Writes string to a file storage.
@relates cv::FileStorage
 */
CV_EXPORTS FileStorage& operator << (FileStorage& fs, const String& str);

//! @cond IGNORED

namespace internal
{
    class CV_EXPORTS WriteStructContext
    {
    public:
        WriteStructContext(FileStorage& _fs, const String& name, int flags, const String& typeName = String());
        ~WriteStructContext();
    private:
        FileStorage* fs;
    };

    template<typename _Tp, int numflag> class VecWriterProxy
    {
    public:
        VecWriterProxy( FileStorage* _fs ) : fs(_fs) {}
        void operator()(const std::vector<_Tp>& vec) const
        {
            size_t count = vec.size();
            for (size_t i = 0; i < count; i++)
                write(*fs, vec[i]);
        }
    private:
        FileStorage* fs;
    };

    template<typename _Tp> class VecWriterProxy<_Tp, 1>
    {
    public:
        VecWriterProxy( FileStorage* _fs ) : fs(_fs) {}
        void operator()(const std::vector<_Tp>& vec) const
        {
            int _fmt = DataType<_Tp>::fmt;
            char fmt[] = { (char)((_fmt >> 8) + '1'), (char)_fmt, '\0' };
            fs->writeRaw(fmt, !vec.empty() ? (uchar*)&vec[0] : 0, vec.size() * sizeof(_Tp));
        }
    private:
        FileStorage* fs;
    };

    template<typename _Tp, int numflag> class VecReaderProxy
    {
    public:
        VecReaderProxy( FileNodeIterator* _it ) : it(_it) {}
        void operator()(std::vector<_Tp>& vec, size_t count) const
        {
            count = std::min(count, it->remaining);
            vec.resize(count);
            for (size_t i = 0; i < count; i++, ++(*it))
                read(**it, vec[i], _Tp());
        }
    private:
        FileNodeIterator* it;
    };

    template<typename _Tp> class VecReaderProxy<_Tp, 1>
    {
    public:
        VecReaderProxy( FileNodeIterator* _it ) : it(_it) {}
        void operator()(std::vector<_Tp>& vec, size_t count) const
        {
            size_t remaining = it->remaining;
            size_t cn = DataType<_Tp>::channels;
            int _fmt = DataType<_Tp>::fmt;
            char fmt[] = { (char)((_fmt >> 8)+'1'), (char)_fmt, '\0' };
            size_t remaining1 = remaining / cn;
            count = count < remaining1 ? count : remaining1;
            vec.resize(count);
            it->readRaw(fmt, !vec.empty() ? (uchar*)&vec[0] : 0, count*sizeof(_Tp));
        }
    private:
        FileNodeIterator* it;
    };

} // internal

//! @endcond

//! @relates cv::FileStorage
//! @{

template<typename _Tp> static inline
void write(FileStorage& fs, const _Tp& value)
{
    write(fs, String(), value);
}

template<> inline
void write( FileStorage& fs, const int& value )
{
    writeScalar(fs, value);
}

template<> inline
void write( FileStorage& fs, const float& value )
{
    writeScalar(fs, value);
}

template<> inline
void write( FileStorage& fs, const double& value )
{
    writeScalar(fs, value);
}

template<> inline
void write( FileStorage& fs, const String& value )
{
    writeScalar(fs, value);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Point_<_Tp>& pt )
{
    write(fs, pt.x);
    write(fs, pt.y);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Point3_<_Tp>& pt )
{
    write(fs, pt.x);
    write(fs, pt.y);
    write(fs, pt.z);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Size_<_Tp>& sz )
{
    write(fs, sz.width);
    write(fs, sz.height);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Complex<_Tp>& c )
{
    write(fs, c.re);
    write(fs, c.im);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Rect_<_Tp>& r )
{
    write(fs, r.x);
    write(fs, r.y);
    write(fs, r.width);
    write(fs, r.height);
}

template<typename _Tp, int cn> static inline
void write(FileStorage& fs, const Vec<_Tp, cn>& v )
{
    for(int i = 0; i < cn; i++)
        write(fs, v.val[i]);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Scalar_<_Tp>& s )
{
    write(fs, s.val[0]);
    write(fs, s.val[1]);
    write(fs, s.val[2]);
    write(fs, s.val[3]);
}

static inline
void write(FileStorage& fs, const KeyPoint& kpt )
{
    write(fs, kpt.pt.x);
    write(fs, kpt.pt.y);
    write(fs, kpt.size);
    write(fs, kpt.angle);
    write(fs, kpt.response);
    write(fs, kpt.octave);
    write(fs, kpt.class_id);
}

static inline
void write(FileStorage& fs, const DMatch& m )
{
    write(fs, m.queryIdx);
    write(fs, m.trainIdx);
    write(fs, m.imgIdx);
    write(fs, m.distance);
}

static inline
void write(FileStorage& fs, const Range& r )
{
    write(fs, r.start);
    write(fs, r.end);
}

static inline
void write( FileStorage& fs, const std::vector<KeyPoint>& vec )
{
    size_t npoints = vec.size();
    for(size_t i = 0; i < npoints; i++ )
    {
        write(fs, vec[i]);
    }
}

static inline
void write( FileStorage& fs, const std::vector<DMatch>& vec )
{
    size_t npoints = vec.size();
    for(size_t i = 0; i < npoints; i++ )
    {
        write(fs, vec[i]);
    }
}

template<typename _Tp> static inline
void write( FileStorage& fs, const std::vector<_Tp>& vec )
{
    cv::internal::VecWriterProxy<_Tp, DataType<_Tp>::fmt != 0> w(&fs);
    w(vec);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Point_<_Tp>& pt )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Point3_<_Tp>& pt )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Size_<_Tp>& sz )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, sz);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Complex<_Tp>& c )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, c);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Rect_<_Tp>& r )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

template<typename _Tp, int cn> static inline
void write(FileStorage& fs, const String& name, const Vec<_Tp, cn>& v )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, v);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Scalar_<_Tp>& s )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, s);
}

static inline
void write(FileStorage& fs, const String& name, const Range& r )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

static inline
void write(FileStorage& fs, const String& name, const KeyPoint& r )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

static inline
void write(FileStorage& fs, const String& name, const DMatch& r )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

template<typename _Tp> static inline
void write( FileStorage& fs, const String& name, const std::vector<_Tp>& vec )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+(DataType<_Tp>::fmt != 0 ? FileNode::FLOW : 0));
    write(fs, vec);
}

template<typename _Tp> static inline
void write( FileStorage& fs, const String& name, const std::vector< std::vector<_Tp> >& vec )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ);
    for(size_t i = 0; i < vec.size(); i++)
    {
        cv::internal::WriteStructContext ws_(fs, name, FileNode::SEQ+(DataType<_Tp>::fmt != 0 ? FileNode::FLOW : 0));
        write(fs, vec[i]);
    }
}

//! @} FileStorage

//! @relates cv::FileNode
//! @{

static inline
void read(const FileNode& node, bool& value, bool default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = temp != 0;
}

static inline
void read(const FileNode& node, uchar& value, uchar default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<uchar>(temp);
}

static inline
void read(const FileNode& node, schar& value, schar default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<schar>(temp);
}

static inline
void read(const FileNode& node, ushort& value, ushort default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<ushort>(temp);
}

static inline
void read(const FileNode& node, short& value, short default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<short>(temp);
}

template<typename _Tp> static inline
void read( FileNodeIterator& it, std::vector<_Tp>& vec, size_t maxCount = (size_t)INT_MAX )
{
    cv::internal::VecReaderProxy<_Tp, DataType<_Tp>::fmt != 0> r(&it);
    r(vec, maxCount);
}

template<typename _Tp> static inline
void read( const FileNode& node, std::vector<_Tp>& vec, const std::vector<_Tp>& default_value = std::vector<_Tp>() )
{
    if(!node.node)
        vec = default_value;
    else
    {
        FileNodeIterator it = node.begin();
        read( it, vec );
    }
}

static inline
void read( const FileNode& node, std::vector<KeyPoint>& vec, const std::vector<KeyPoint>& default_value )
{
    if(!node.node)
        vec = default_value;
    else
        read(node, vec);
}

static inline
void read( const FileNode& node, std::vector<DMatch>& vec, const std::vector<DMatch>& default_value )
{
    if(!node.node)
        vec = default_value;
    else
        read(node, vec);
}

//! @} FileNode

//! @relates cv::FileStorage
//! @{

/** @brief Writes data to a file storage.
 */
template<typename _Tp> static inline
FileStorage& operator << (FileStorage& fs, const _Tp& value)
{
    if( !fs.isOpened() )
        return fs;
    if( fs.state == FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP )
        CV_Error( Error::StsError, "No element name has been given" );
    write( fs, fs.elname, value );
    if( fs.state & FileStorage::INSIDE_MAP )
        fs.state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
    return fs;
}

/** @brief Writes data to a file storage.
 */
static inline
FileStorage& operator << (FileStorage& fs, const char* str)
{
    return (fs << String(str));
}

/** @brief Writes data to a file storage.
 */
static inline
FileStorage& operator << (FileStorage& fs, char* value)
{
    return (fs << String(value));
}

//! @} FileStorage

//! @relates cv::FileNodeIterator
//! @{

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, _Tp& value)
{
    read( *it, value, _Tp());
    return ++it;
}

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, std::vector<_Tp>& vec)
{
    cv::internal::VecReaderProxy<_Tp, DataType<_Tp>::fmt != 0> r(&it);
    r(vec, (size_t)INT_MAX);
    return it;
}

//! @} FileNodeIterator

//! @relates cv::FileNode
//! @{

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
void operator >> (const FileNode& n, _Tp& value)
{
    read( n, value, _Tp());
}

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
void operator >> (const FileNode& n, std::vector<_Tp>& vec)
{
    FileNodeIterator it = n.begin();
    it >> vec;
}

/** @brief Reads KeyPoint from a file storage.
*/
//It needs special handling because it contains two types of fields, int & float.
static inline
void operator >> (const FileNode& n, std::vector<KeyPoint>& vec)
{
    read(n, vec);
}

static inline
void operator >> (const FileNode& n, KeyPoint& kpt)
{
    FileNodeIterator it = n.begin();
    it >> kpt.pt.x >> kpt.pt.y >> kpt.size >> kpt.angle >> kpt.response >> kpt.octave >> kpt.class_id;
}

/** @brief Reads DMatch from a file storage.
*/
//It needs special handling because it contains two types of fields, int & float.
static inline
void operator >> (const FileNode& n, std::vector<DMatch>& vec)
{
    read(n, vec);
}

static inline
void operator >> (const FileNode& n, DMatch& m)
{
    FileNodeIterator it = n.begin();
    it >> m.queryIdx >> m.trainIdx >> m.imgIdx >> m.distance;
}

//! @} FileNode

//! @relates cv::FileNodeIterator
//! @{

static inline
bool operator == (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.fs == it2.fs && it1.container == it2.container &&
        it1.reader.ptr == it2.reader.ptr && it1.remaining == it2.remaining;
}

static inline
bool operator != (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return !(it1 == it2);
}

static inline
ptrdiff_t operator - (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it2.remaining - it1.remaining;
}

static inline
bool operator < (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.remaining > it2.remaining;
}

//! @} FileNodeIterator

//! @cond IGNORED

inline FileNode FileStorage::getFirstTopLevelNode() const { FileNode r = root(); FileNodeIterator it = r.begin(); return it != r.end() ? *it : FileNode(); }
inline FileNode::FileNode() : fs(0), node(0) {}
inline FileNode::FileNode(const CvFileStorage* _fs, const CvFileNode* _node) : fs(_fs), node(_node) {}
inline FileNode::FileNode(const FileNode& _node) : fs(_node.fs), node(_node.node) {}
inline bool FileNode::empty() const    { return node   == 0;    }
inline bool FileNode::isNone() const   { return type() == NONE; }
inline bool FileNode::isSeq() const    { return type() == SEQ;  }
inline bool FileNode::isMap() const    { return type() == MAP;  }
inline bool FileNode::isInt() const    { return type() == INT;  }
inline bool FileNode::isReal() const   { return type() == REAL; }
inline bool FileNode::isString() const { return type() == STR;  }
inline CvFileNode* FileNode::operator *() { return (CvFileNode*)node; }
inline const CvFileNode* FileNode::operator* () const { return node; }
inline FileNode::operator int() const    { int value;    read(*this, value, 0);     return value; }
inline FileNode::operator float() const  { float value;  read(*this, value, 0.f);   return value; }
inline FileNode::operator double() const { double value; read(*this, value, 0.);    return value; }
inline FileNode::operator String() const { String value; read(*this, value, value); return value; }
inline double FileNode::real() const  { return double(*this); }
inline String FileNode::string() const { return String(*this); }
inline Mat FileNode::mat() const { Mat value; read(*this, value, value);    return value; }
inline FileNodeIterator FileNode::begin() const { return FileNodeIterator(fs, node); }
inline FileNodeIterator FileNode::end() const   { return FileNodeIterator(fs, node, size()); }
inline void FileNode::readRaw( const String& fmt, uchar* vec, size_t len ) const { begin().readRaw( fmt, vec, len ); }
inline FileNode FileNodeIterator::operator *() const  { return FileNode(fs, (const CvFileNode*)(const void*)reader.ptr); }
inline FileNode FileNodeIterator::operator ->() const { return FileNode(fs, (const CvFileNode*)(const void*)reader.ptr); }
inline String::String(const FileNode& fn): cstr_(0), len_(0) { read(fn, *this, *this); }

//! @endcond


CV_EXPORTS void cvStartWriteRawData_Base64(::CvFileStorage * fs, const char* name, int len, const char* dt);

CV_EXPORTS void cvWriteRawData_Base64(::CvFileStorage * fs, const void* _data, int len);

CV_EXPORTS void cvEndWriteRawData_Base64(::CvFileStorage * fs);

CV_EXPORTS void cvWriteMat_Base64(::CvFileStorage* fs, const char* name, const ::CvMat* mat);

CV_EXPORTS void cvWriteMatND_Base64(::CvFileStorage* fs, const char* name, const ::CvMatND* mat);

} // cv

#endif // OPENCV_CORE_PERSISTENCE_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_PTR_INL_HPP
#define OPENCV_CORE_PTR_INL_HPP

#include <algorithm>

//! @cond IGNORED

namespace cv {

template<typename Y>
void DefaultDeleter<Y>::operator () (Y* p) const
{
    delete p;
}

namespace detail
{

struct PtrOwner
{
    PtrOwner() : refCount(1)
    {}

    void incRef()
    {
        CV_XADD(&refCount, 1);
    }

    void decRef()
    {
        if (CV_XADD(&refCount, -1) == 1) deleteSelf();
    }

protected:
    /* This doesn't really need to be virtual, since PtrOwner is never deleted
       directly, but it doesn't hurt and it helps avoid warnings. */
    virtual ~PtrOwner()
    {}

    virtual void deleteSelf() = 0;

private:
    unsigned int refCount;

    // noncopyable
    PtrOwner(const PtrOwner&);
    PtrOwner& operator = (const PtrOwner&);
};

template<typename Y, typename D>
struct PtrOwnerImpl : PtrOwner
{
    PtrOwnerImpl(Y* p, D d) : owned(p), deleter(d)
    {}

    void deleteSelf()
    {
        deleter(owned);
        delete this;
    }

private:
    Y* owned;
    D deleter;
};


}

template<typename T>
Ptr<T>::Ptr() : owner(NULL), stored(NULL)
{}

template<typename T>
template<typename Y>
Ptr<T>::Ptr(Y* p)
  : owner(p
      ? new detail::PtrOwnerImpl<Y, DefaultDeleter<Y> >(p, DefaultDeleter<Y>())
      : NULL),
    stored(p)
{}

template<typename T>
template<typename Y, typename D>
Ptr<T>::Ptr(Y* p, D d)
  : owner(p
      ? new detail::PtrOwnerImpl<Y, D>(p, d)
      : NULL),
    stored(p)
{}

template<typename T>
Ptr<T>::Ptr(const Ptr& o) : owner(o.owner), stored(o.stored)
{
    if (owner) owner->incRef();
}

template<typename T>
template<typename Y>
Ptr<T>::Ptr(const Ptr<Y>& o) : owner(o.owner), stored(o.stored)
{
    if (owner) owner->incRef();
}

template<typename T>
template<typename Y>
Ptr<T>::Ptr(const Ptr<Y>& o, T* p) : owner(o.owner), stored(p)
{
    if (owner) owner->incRef();
}

template<typename T>
Ptr<T>::~Ptr()
{
    release();
}

template<typename T>
Ptr<T>& Ptr<T>::operator = (const Ptr<T>& o)
{
    Ptr(o).swap(*this);
    return *this;
}

template<typename T>
template<typename Y>
Ptr<T>& Ptr<T>::operator = (const Ptr<Y>& o)
{
    Ptr(o).swap(*this);
    return *this;
}

template<typename T>
void Ptr<T>::release()
{
    if (owner) owner->decRef();
    owner = NULL;
    stored = NULL;
}

template<typename T>
template<typename Y>
void Ptr<T>::reset(Y* p)
{
    Ptr(p).swap(*this);
}

template<typename T>
template<typename Y, typename D>
void Ptr<T>::reset(Y* p, D d)
{
    Ptr(p, d).swap(*this);
}

template<typename T>
void Ptr<T>::swap(Ptr<T>& o)
{
    std::swap(owner, o.owner);
    std::swap(stored, o.stored);
}

template<typename T>
T* Ptr<T>::get() const
{
    return stored;
}

template<typename T>
typename detail::RefOrVoid<T>::type Ptr<T>::operator * () const
{
    return *stored;
}

template<typename T>
T* Ptr<T>::operator -> () const
{
    return stored;
}

template<typename T>
Ptr<T>::operator T* () const
{
    return stored;
}


template<typename T>
bool Ptr<T>::empty() const
{
    return !stored;
}

template<typename T>
template<typename Y>
Ptr<Y> Ptr<T>::staticCast() const
{
    return Ptr<Y>(*this, static_cast<Y*>(stored));
}

template<typename T>
template<typename Y>
Ptr<Y> Ptr<T>::constCast() const
{
    return Ptr<Y>(*this, const_cast<Y*>(stored));
}

template<typename T>
template<typename Y>
Ptr<Y> Ptr<T>::dynamicCast() const
{
    return Ptr<Y>(*this, dynamic_cast<Y*>(stored));
}

#ifdef CV_CXX_MOVE_SEMANTICS

template<typename T>
Ptr<T>::Ptr(Ptr&& o) : owner(o.owner), stored(o.stored)
{
    o.owner = NULL;
    o.stored = NULL;
}

template<typename T>
Ptr<T>& Ptr<T>::operator = (Ptr<T>&& o)
{
    if (this == &o)
        return *this;

    release();
    owner = o.owner;
    stored = o.stored;
    o.owner = NULL;
    o.stored = NULL;
    return *this;
}

#endif


template<typename T>
void swap(Ptr<T>& ptr1, Ptr<T>& ptr2){
    ptr1.swap(ptr2);
}

template<typename T>
bool operator == (const Ptr<T>& ptr1, const Ptr<T>& ptr2)
{
    return ptr1.get() == ptr2.get();
}

template<typename T>
bool operator != (const Ptr<T>& ptr1, const Ptr<T>& ptr2)
{
    return ptr1.get() != ptr2.get();
}

template<typename T>
Ptr<T> makePtr()
{
    return Ptr<T>(new T());
}

template<typename T, typename A1>
Ptr<T> makePtr(const A1& a1)
{
    return Ptr<T>(new T(a1));
}

template<typename T, typename A1, typename A2>
Ptr<T> makePtr(const A1& a1, const A2& a2)
{
    return Ptr<T>(new T(a1, a2));
}

template<typename T, typename A1, typename A2, typename A3>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3)
{
    return Ptr<T>(new T(a1, a2, a3));
}

template<typename T, typename A1, typename A2, typename A3, typename A4>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4)
{
    return Ptr<T>(new T(a1, a2, a3, a4));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10, const A11& a11)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11));
}

template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12>
Ptr<T> makePtr(const A1& a1, const A2& a2, const A3& a3, const A4& a4, const A5& a5, const A6& a6, const A7& a7, const A8& a8, const A9& a9, const A10& a10, const A11& a11, const A12& a12)
{
    return Ptr<T>(new T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12));
}
} // namespace cv

//! @endcond

#endif // OPENCV_CORE_PTR_INL_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_SATURATE_HPP
#define OPENCV_CORE_SATURATE_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/fast_math.hpp"

namespace cv
{

//! @addtogroup core_utils
//! @{

/////////////// saturate_cast (used in image & signal processing) ///////////////////

/** @brief Template function for accurate conversion from one primitive type to another.

 The functions saturate_cast resemble the standard C++ cast operations, such as static_cast\<T\>()
 and others. They perform an efficient and accurate conversion from one primitive type to another
 (see the introduction chapter). saturate in the name means that when the input value v is out of the
 range of the target type, the result is not formed just by taking low bits of the input, but instead
 the value is clipped. For example:
 @code
 uchar a = saturate_cast<uchar>(-100); // a = 0 (UCHAR_MIN)
 short b = saturate_cast<short>(33333.33333); // b = 32767 (SHRT_MAX)
 @endcode
 Such clipping is done when the target type is unsigned char , signed char , unsigned short or
 signed short . For 32-bit integers, no clipping is done.

 When the parameter is a floating-point value and the target type is an integer (8-, 16- or 32-bit),
 the floating-point value is first rounded to the nearest integer and then clipped if needed (when
 the target type is 8- or 16-bit).

 This operation is used in the simplest or most complex image processing functions in OpenCV.

 @param v Function parameter.
 @sa add, subtract, multiply, divide, Mat::convertTo
 */
template<typename _Tp> static inline _Tp saturate_cast(uchar v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(schar v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(ushort v)   { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(short v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(unsigned v) { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(int v)      { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(float v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(double v)   { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(int64 v)    { return _Tp(v); }
/** @overload */
template<typename _Tp> static inline _Tp saturate_cast(uint64 v)   { return _Tp(v); }

template<> inline uchar saturate_cast<uchar>(schar v)        { return (uchar)std::max((int)v, 0); }
template<> inline uchar saturate_cast<uchar>(ushort v)       { return (uchar)std::min((unsigned)v, (unsigned)UCHAR_MAX); }
template<> inline uchar saturate_cast<uchar>(int v)          { return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline uchar saturate_cast<uchar>(short v)        { return saturate_cast<uchar>((int)v); }
template<> inline uchar saturate_cast<uchar>(unsigned v)     { return (uchar)std::min(v, (unsigned)UCHAR_MAX); }
template<> inline uchar saturate_cast<uchar>(float v)        { int iv = cvRound(v); return saturate_cast<uchar>(iv); }
template<> inline uchar saturate_cast<uchar>(double v)       { int iv = cvRound(v); return saturate_cast<uchar>(iv); }
template<> inline uchar saturate_cast<uchar>(int64 v)        { return (uchar)((uint64)v <= (uint64)UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }
template<> inline uchar saturate_cast<uchar>(uint64 v)       { return (uchar)std::min(v, (uint64)UCHAR_MAX); }

template<> inline schar saturate_cast<schar>(uchar v)        { return (schar)std::min((int)v, SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(ushort v)       { return (schar)std::min((unsigned)v, (unsigned)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(int v)          { return (schar)((unsigned)(v-SCHAR_MIN) <= (unsigned)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline schar saturate_cast<schar>(short v)        { return saturate_cast<schar>((int)v); }
template<> inline schar saturate_cast<schar>(unsigned v)     { return (schar)std::min(v, (unsigned)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(float v)        { int iv = cvRound(v); return saturate_cast<schar>(iv); }
template<> inline schar saturate_cast<schar>(double v)       { int iv = cvRound(v); return saturate_cast<schar>(iv); }
template<> inline schar saturate_cast<schar>(int64 v)        { return (schar)((uint64)((int64)v-SCHAR_MIN) <= (uint64)UCHAR_MAX ? v : v > 0 ? SCHAR_MAX : SCHAR_MIN); }
template<> inline schar saturate_cast<schar>(uint64 v)       { return (schar)std::min(v, (uint64)SCHAR_MAX); }

template<> inline ushort saturate_cast<ushort>(schar v)      { return (ushort)std::max((int)v, 0); }
template<> inline ushort saturate_cast<ushort>(short v)      { return (ushort)std::max((int)v, 0); }
template<> inline ushort saturate_cast<ushort>(int v)        { return (ushort)((unsigned)v <= (unsigned)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline ushort saturate_cast<ushort>(unsigned v)   { return (ushort)std::min(v, (unsigned)USHRT_MAX); }
template<> inline ushort saturate_cast<ushort>(float v)      { int iv = cvRound(v); return saturate_cast<ushort>(iv); }
template<> inline ushort saturate_cast<ushort>(double v)     { int iv = cvRound(v); return saturate_cast<ushort>(iv); }
template<> inline ushort saturate_cast<ushort>(int64 v)      { return (ushort)((uint64)v <= (uint64)USHRT_MAX ? v : v > 0 ? USHRT_MAX : 0); }
template<> inline ushort saturate_cast<ushort>(uint64 v)     { return (ushort)std::min(v, (uint64)USHRT_MAX); }

template<> inline short saturate_cast<short>(ushort v)       { return (short)std::min((int)v, SHRT_MAX); }
template<> inline short saturate_cast<short>(int v)          { return (short)((unsigned)(v - SHRT_MIN) <= (unsigned)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline short saturate_cast<short>(unsigned v)     { return (short)std::min(v, (unsigned)SHRT_MAX); }
template<> inline short saturate_cast<short>(float v)        { int iv = cvRound(v); return saturate_cast<short>(iv); }
template<> inline short saturate_cast<short>(double v)       { int iv = cvRound(v); return saturate_cast<short>(iv); }
template<> inline short saturate_cast<short>(int64 v)        { return (short)((uint64)((int64)v - SHRT_MIN) <= (uint64)USHRT_MAX ? v : v > 0 ? SHRT_MAX : SHRT_MIN); }
template<> inline short saturate_cast<short>(uint64 v)       { return (short)std::min(v, (uint64)SHRT_MAX); }

template<> inline int saturate_cast<int>(float v)            { return cvRound(v); }
template<> inline int saturate_cast<int>(double v)           { return cvRound(v); }

// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template<> inline unsigned saturate_cast<unsigned>(float v)  { return cvRound(v); }
template<> inline unsigned saturate_cast<unsigned>(double v) { return cvRound(v); }

//! @}

} // cv

#endif // OPENCV_CORE_SATURATE_HPP
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on files from package issued with the following license:

/*============================================================================

This C header file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#pragma once
#ifndef softfloat_h
#define softfloat_h 1

#include "cvdef.h"

// int32_t / uint32_t
#if defined(_MSC_VER) && _MSC_VER < 1600 /* MSVS 2010 */
namespace cv {
typedef signed int int32_t;
typedef unsigned int uint32_t;
}
#elif defined(_MSC_VER) || __cplusplus >= 201103L
#include <cstdint>
#else
#include <stdint.h>
#endif

namespace cv
{

struct softfloat;
struct softdouble;

struct CV_EXPORTS softfloat
{
public:
    softfloat() { v = 0; }
    softfloat( const softfloat& c) { v = c.v; }
    softfloat& operator=( const softfloat& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    static const softfloat fromRaw( const uint32_t a ) { softfloat x; x.v = a; return x; }

    explicit softfloat( const uint32_t );
    explicit softfloat( const uint64_t );
    explicit softfloat( const int32_t );
    explicit softfloat( const int64_t );
    explicit softfloat( const float a ) { Cv32suf s; s.f = a; v = s.u; }

    operator softdouble() const;
    operator float() const { Cv32suf s; s.u = v; return s.f; }

    softfloat operator + (const softfloat&) const;
    softfloat operator - (const softfloat&) const;
    softfloat operator * (const softfloat&) const;
    softfloat operator / (const softfloat&) const;
    softfloat operator % (const softfloat&) const;
    softfloat operator - () const { softfloat x; x.v = v ^ (1U << 31); return x; }

    softfloat& operator += (const softfloat& a) { *this = *this + a; return *this; }
    softfloat& operator -= (const softfloat& a) { *this = *this - a; return *this; }
    softfloat& operator *= (const softfloat& a) { *this = *this * a; return *this; }
    softfloat& operator /= (const softfloat& a) { *this = *this / a; return *this; }
    softfloat& operator %= (const softfloat& a) { *this = *this % a; return *this; }

    bool operator == ( const softfloat& ) const;
    bool operator != ( const softfloat& ) const;
    bool operator >  ( const softfloat& ) const;
    bool operator >= ( const softfloat& ) const;
    bool operator <  ( const softfloat& ) const;
    bool operator <= ( const softfloat& ) const;

    bool isNaN() const { return (v & 0x7fffffff)  > 0x7f800000; }
    bool isInf() const { return (v & 0x7fffffff) == 0x7f800000; }

    static softfloat zero() { return softfloat::fromRaw( 0 ); }
    static softfloat  inf() { return softfloat::fromRaw( 0xFF << 23 ); }
    static softfloat  nan() { return softfloat::fromRaw( 0x7fffffff ); }
    static softfloat  one() { return softfloat::fromRaw(  127 << 23 ); }

    uint32_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

struct CV_EXPORTS softdouble
{
public:
    softdouble() : v(0) { }
    softdouble( const softdouble& c) { v = c.v; }
    softdouble& operator=( const softdouble& c )
    {
        if(&c != this) v = c.v;
        return *this;
    }
    static softdouble fromRaw( const uint64_t a ) { softdouble x; x.v = a; return x; }

    explicit softdouble( const uint32_t );
    explicit softdouble( const uint64_t );
    explicit softdouble( const  int32_t );
    explicit softdouble( const  int64_t );
    explicit softdouble( const double a ) { Cv64suf s; s.f = a; v = s.u; }

    operator softfloat() const;
    operator double() const { Cv64suf s; s.u = v; return s.f; }

    softdouble operator + (const softdouble&) const;
    softdouble operator - (const softdouble&) const;
    softdouble operator * (const softdouble&) const;
    softdouble operator / (const softdouble&) const;
    softdouble operator % (const softdouble&) const;
    softdouble operator - () const { softdouble x; x.v = v ^ (1ULL << 63); return x; }

    softdouble& operator += (const softdouble& a) { *this = *this + a; return *this; }
    softdouble& operator -= (const softdouble& a) { *this = *this - a; return *this; }
    softdouble& operator *= (const softdouble& a) { *this = *this * a; return *this; }
    softdouble& operator /= (const softdouble& a) { *this = *this / a; return *this; }
    softdouble& operator %= (const softdouble& a) { *this = *this % a; return *this; }

    bool operator == ( const softdouble& ) const;
    bool operator != ( const softdouble& ) const;
    bool operator >  ( const softdouble& ) const;
    bool operator >= ( const softdouble& ) const;
    bool operator <  ( const softdouble& ) const;
    bool operator <= ( const softdouble& ) const;

    bool isNaN() const { return (v & 0x7fffffffffffffff)  > 0x7ff0000000000000; }
    bool isInf() const { return (v & 0x7fffffffffffffff) == 0x7ff0000000000000; }

    static softdouble zero() { return softdouble::fromRaw( 0 ); }
    static softdouble  inf() { return softdouble::fromRaw( (uint_fast64_t)(0x7FF) << 52 ); }
    static softdouble  nan() { return softdouble::fromRaw( CV_BIG_INT(0x7FFFFFFFFFFFFFFF) ); }
    static softdouble  one() { return softdouble::fromRaw( (uint_fast64_t)( 1023) << 52 ); }

    uint64_t v;
};

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

CV_EXPORTS softfloat  mulAdd( const softfloat&  a, const softfloat&  b, const softfloat & c);
CV_EXPORTS softdouble mulAdd( const softdouble& a, const softdouble& b, const softdouble& c);

CV_EXPORTS softfloat  sqrt( const softfloat&  a );
CV_EXPORTS softdouble sqrt( const softdouble& a );
}

/*----------------------------------------------------------------------------
| Ported from OpenCV and added for usability
*----------------------------------------------------------------------------*/

CV_EXPORTS int cvTrunc(const cv::softfloat&  a);
CV_EXPORTS int cvTrunc(const cv::softdouble& a);

CV_EXPORTS int cvRound(const cv::softfloat&  a);
CV_EXPORTS int cvRound(const cv::softdouble& a);

CV_EXPORTS int cvFloor(const cv::softfloat&  a);
CV_EXPORTS int cvFloor(const cv::softdouble& a);

CV_EXPORTS int  cvCeil(const cv::softfloat&  a);
CV_EXPORTS int  cvCeil(const cv::softdouble& a);

namespace cv
{
template<typename _Tp> static inline _Tp saturate_cast(softfloat  a) { return _Tp(a); }
template<typename _Tp> static inline _Tp saturate_cast(softdouble a) { return _Tp(a); }

template<> inline uchar saturate_cast<uchar>(softfloat  a) { return (uchar)std::max(std::min(cvRound(a), (int)UCHAR_MAX), 0); }
template<> inline uchar saturate_cast<uchar>(softdouble a) { return (uchar)std::max(std::min(cvRound(a), (int)UCHAR_MAX), 0); }

template<> inline schar saturate_cast<schar>(softfloat  a) { return (schar)std::min(std::max(cvRound(a), (int)SCHAR_MIN), (int)SCHAR_MAX); }
template<> inline schar saturate_cast<schar>(softdouble a) { return (schar)std::min(std::max(cvRound(a), (int)SCHAR_MIN), (int)SCHAR_MAX); }

template<> inline ushort saturate_cast<ushort>(softfloat  a) { return (ushort)std::max(std::min(cvRound(a), (int)USHRT_MAX), 0); }
template<> inline ushort saturate_cast<ushort>(softdouble a) { return (ushort)std::max(std::min(cvRound(a), (int)USHRT_MAX), 0); }

template<> inline short saturate_cast<short>(softfloat  a) { return (short)std::min(std::max(cvRound(a), (int)SHRT_MIN), (int)SHRT_MAX); }
template<> inline short saturate_cast<short>(softdouble a) { return (short)std::min(std::max(cvRound(a), (int)SHRT_MIN), (int)SHRT_MAX); }

template<> inline int saturate_cast<int>(softfloat  a) { return cvRound(a); }
template<> inline int saturate_cast<int>(softdouble a) { return cvRound(a); }

// we intentionally do not clip negative numbers, to make -1 become 0xffffffff etc.
template<> inline unsigned saturate_cast<unsigned>(softfloat  a) { return cvRound(a); }
template<> inline unsigned saturate_cast<unsigned>(softdouble a) { return cvRound(a); }

inline softfloat  min(const softfloat&  a, const softfloat&  b) { return (a > b) ? b : a; }
inline softdouble min(const softdouble& a, const softdouble& b) { return (a > b) ? b : a; }

inline softfloat  max(const softfloat&  a, const softfloat&  b) { return (a > b) ? a : b; }
inline softdouble max(const softdouble& a, const softdouble& b) { return (a > b) ? a : b; }

inline softfloat  abs( softfloat  a) { softfloat  x; x.v = a.v & ((1U   << 31) - 1); return x; }
inline softdouble abs( softdouble a) { softdouble x; x.v = a.v & ((1ULL << 63) - 1); return x; }

CV_EXPORTS softfloat  exp( const softfloat&  a);
CV_EXPORTS softdouble exp( const softdouble& a);

CV_EXPORTS softfloat  log( const softfloat&  a );
CV_EXPORTS softdouble log( const softdouble& a );

CV_EXPORTS softfloat  pow( const softfloat&  a, const softfloat&  b);
CV_EXPORTS softdouble pow( const softdouble& a, const softdouble& b);

CV_EXPORTS softfloat cbrt(const softfloat& a);

}

#endif
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_SSE_UTILS_HPP
#define OPENCV_CORE_SSE_UTILS_HPP

#ifndef __cplusplus
#  error sse_utils.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"

//! @addtogroup core_utils_sse
//! @{

#if CV_SSE2

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_g0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_g0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_g1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_g1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk3);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk3);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk3);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk3);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk2);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk2);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk3);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk3);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk2);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk2);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk3);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk3);
}

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                  __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_g1);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_g1);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_b0);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_b0);
    __m128i layer1_chunk4 = _mm_unpacklo_epi8(v_g0, v_b1);
    __m128i layer1_chunk5 = _mm_unpackhi_epi8(v_g0, v_b1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk4 = _mm_unpacklo_epi8(layer1_chunk2, layer1_chunk5);
    __m128i layer2_chunk5 = _mm_unpackhi_epi8(layer1_chunk2, layer1_chunk5);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk4 = _mm_unpacklo_epi8(layer2_chunk2, layer2_chunk5);
    __m128i layer3_chunk5 = _mm_unpackhi_epi8(layer2_chunk2, layer2_chunk5);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk3);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk3);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk4);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk4);
    __m128i layer4_chunk4 = _mm_unpacklo_epi8(layer3_chunk2, layer3_chunk5);
    __m128i layer4_chunk5 = _mm_unpackhi_epi8(layer3_chunk2, layer3_chunk5);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk3);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk3);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk4);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk4);
    v_b0 = _mm_unpacklo_epi8(layer4_chunk2, layer4_chunk5);
    v_b1 = _mm_unpackhi_epi8(layer4_chunk2, layer4_chunk5);
}

inline void _mm_deinterleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                  __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi8(v_r0, v_b0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi8(v_r0, v_b0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi8(v_r1, v_b1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi8(v_r1, v_b1);
    __m128i layer1_chunk4 = _mm_unpacklo_epi8(v_g0, v_a0);
    __m128i layer1_chunk5 = _mm_unpackhi_epi8(v_g0, v_a0);
    __m128i layer1_chunk6 = _mm_unpacklo_epi8(v_g1, v_a1);
    __m128i layer1_chunk7 = _mm_unpackhi_epi8(v_g1, v_a1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi8(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk1 = _mm_unpackhi_epi8(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk2 = _mm_unpacklo_epi8(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk3 = _mm_unpackhi_epi8(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk4 = _mm_unpacklo_epi8(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk5 = _mm_unpackhi_epi8(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk6 = _mm_unpacklo_epi8(layer1_chunk3, layer1_chunk7);
    __m128i layer2_chunk7 = _mm_unpackhi_epi8(layer1_chunk3, layer1_chunk7);

    __m128i layer3_chunk0 = _mm_unpacklo_epi8(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk1 = _mm_unpackhi_epi8(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk2 = _mm_unpacklo_epi8(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk3 = _mm_unpackhi_epi8(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk4 = _mm_unpacklo_epi8(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk5 = _mm_unpackhi_epi8(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk6 = _mm_unpacklo_epi8(layer2_chunk3, layer2_chunk7);
    __m128i layer3_chunk7 = _mm_unpackhi_epi8(layer2_chunk3, layer2_chunk7);

    __m128i layer4_chunk0 = _mm_unpacklo_epi8(layer3_chunk0, layer3_chunk4);
    __m128i layer4_chunk1 = _mm_unpackhi_epi8(layer3_chunk0, layer3_chunk4);
    __m128i layer4_chunk2 = _mm_unpacklo_epi8(layer3_chunk1, layer3_chunk5);
    __m128i layer4_chunk3 = _mm_unpackhi_epi8(layer3_chunk1, layer3_chunk5);
    __m128i layer4_chunk4 = _mm_unpacklo_epi8(layer3_chunk2, layer3_chunk6);
    __m128i layer4_chunk5 = _mm_unpackhi_epi8(layer3_chunk2, layer3_chunk6);
    __m128i layer4_chunk6 = _mm_unpacklo_epi8(layer3_chunk3, layer3_chunk7);
    __m128i layer4_chunk7 = _mm_unpackhi_epi8(layer3_chunk3, layer3_chunk7);

    v_r0 = _mm_unpacklo_epi8(layer4_chunk0, layer4_chunk4);
    v_r1 = _mm_unpackhi_epi8(layer4_chunk0, layer4_chunk4);
    v_g0 = _mm_unpacklo_epi8(layer4_chunk1, layer4_chunk5);
    v_g1 = _mm_unpackhi_epi8(layer4_chunk1, layer4_chunk5);
    v_b0 = _mm_unpacklo_epi8(layer4_chunk2, layer4_chunk6);
    v_b1 = _mm_unpackhi_epi8(layer4_chunk2, layer4_chunk6);
    v_a0 = _mm_unpacklo_epi8(layer4_chunk3, layer4_chunk7);
    v_a1 = _mm_unpackhi_epi8(layer4_chunk3, layer4_chunk7);
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_g1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk4 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer4_chunk5 = _mm_packus_epi16(_mm_srli_epi16(v_b0, 8), _mm_srli_epi16(v_b1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_and_si128(layer4_chunk4, v_mask), _mm_and_si128(layer4_chunk5, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk4, 8), _mm_srli_epi16(layer4_chunk5, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk4, 8), _mm_srli_epi16(layer3_chunk5, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk4, 8), _mm_srli_epi16(layer2_chunk5, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
    v_g0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_b1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk4, 8), _mm_srli_epi16(layer1_chunk5, 8));
}

inline void _mm_interleave_epi8(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i v_mask = _mm_set1_epi16(0x00ff);

    __m128i layer4_chunk0 = _mm_packus_epi16(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer4_chunk4 = _mm_packus_epi16(_mm_srli_epi16(v_r0, 8), _mm_srli_epi16(v_r1, 8));
    __m128i layer4_chunk1 = _mm_packus_epi16(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer4_chunk5 = _mm_packus_epi16(_mm_srli_epi16(v_g0, 8), _mm_srli_epi16(v_g1, 8));
    __m128i layer4_chunk2 = _mm_packus_epi16(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer4_chunk6 = _mm_packus_epi16(_mm_srli_epi16(v_b0, 8), _mm_srli_epi16(v_b1, 8));
    __m128i layer4_chunk3 = _mm_packus_epi16(_mm_and_si128(v_a0, v_mask), _mm_and_si128(v_a1, v_mask));
    __m128i layer4_chunk7 = _mm_packus_epi16(_mm_srli_epi16(v_a0, 8), _mm_srli_epi16(v_a1, 8));

    __m128i layer3_chunk0 = _mm_packus_epi16(_mm_and_si128(layer4_chunk0, v_mask), _mm_and_si128(layer4_chunk1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk0, 8), _mm_srli_epi16(layer4_chunk1, 8));
    __m128i layer3_chunk1 = _mm_packus_epi16(_mm_and_si128(layer4_chunk2, v_mask), _mm_and_si128(layer4_chunk3, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk2, 8), _mm_srli_epi16(layer4_chunk3, 8));
    __m128i layer3_chunk2 = _mm_packus_epi16(_mm_and_si128(layer4_chunk4, v_mask), _mm_and_si128(layer4_chunk5, v_mask));
    __m128i layer3_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk4, 8), _mm_srli_epi16(layer4_chunk5, 8));
    __m128i layer3_chunk3 = _mm_packus_epi16(_mm_and_si128(layer4_chunk6, v_mask), _mm_and_si128(layer4_chunk7, v_mask));
    __m128i layer3_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer4_chunk6, 8), _mm_srli_epi16(layer4_chunk7, 8));

    __m128i layer2_chunk0 = _mm_packus_epi16(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk0, 8), _mm_srli_epi16(layer3_chunk1, 8));
    __m128i layer2_chunk1 = _mm_packus_epi16(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk2, 8), _mm_srli_epi16(layer3_chunk3, 8));
    __m128i layer2_chunk2 = _mm_packus_epi16(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk4, 8), _mm_srli_epi16(layer3_chunk5, 8));
    __m128i layer2_chunk3 = _mm_packus_epi16(_mm_and_si128(layer3_chunk6, v_mask), _mm_and_si128(layer3_chunk7, v_mask));
    __m128i layer2_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer3_chunk6, 8), _mm_srli_epi16(layer3_chunk7, 8));

    __m128i layer1_chunk0 = _mm_packus_epi16(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk0, 8), _mm_srli_epi16(layer2_chunk1, 8));
    __m128i layer1_chunk1 = _mm_packus_epi16(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk2, 8), _mm_srli_epi16(layer2_chunk3, 8));
    __m128i layer1_chunk2 = _mm_packus_epi16(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk6 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk4, 8), _mm_srli_epi16(layer2_chunk5, 8));
    __m128i layer1_chunk3 = _mm_packus_epi16(_mm_and_si128(layer2_chunk6, v_mask), _mm_and_si128(layer2_chunk7, v_mask));
    __m128i layer1_chunk7 = _mm_packus_epi16(_mm_srli_epi16(layer2_chunk6, 8), _mm_srli_epi16(layer2_chunk7, 8));

    v_r0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_b0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk0, 8), _mm_srli_epi16(layer1_chunk1, 8));
    v_r1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk2, 8), _mm_srli_epi16(layer1_chunk3, 8));
    v_g0 = _mm_packus_epi16(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_a0 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk4, 8), _mm_srli_epi16(layer1_chunk5, 8));
    v_g1 = _mm_packus_epi16(_mm_and_si128(layer1_chunk6, v_mask), _mm_and_si128(layer1_chunk7, v_mask));
    v_a1 = _mm_packus_epi16(_mm_srli_epi16(layer1_chunk6, 8), _mm_srli_epi16(layer1_chunk7, 8));
}

inline void _mm_deinterleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi16(v_r0, v_g0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi16(v_r0, v_g0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi16(v_r1, v_g1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi16(v_r1, v_g1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi16(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk1 = _mm_unpackhi_epi16(layer1_chunk0, layer1_chunk2);
    __m128i layer2_chunk2 = _mm_unpacklo_epi16(layer1_chunk1, layer1_chunk3);
    __m128i layer2_chunk3 = _mm_unpackhi_epi16(layer1_chunk1, layer1_chunk3);

    __m128i layer3_chunk0 = _mm_unpacklo_epi16(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk1 = _mm_unpackhi_epi16(layer2_chunk0, layer2_chunk2);
    __m128i layer3_chunk2 = _mm_unpacklo_epi16(layer2_chunk1, layer2_chunk3);
    __m128i layer3_chunk3 = _mm_unpackhi_epi16(layer2_chunk1, layer2_chunk3);

    v_r0 = _mm_unpacklo_epi16(layer3_chunk0, layer3_chunk2);
    v_r1 = _mm_unpackhi_epi16(layer3_chunk0, layer3_chunk2);
    v_g0 = _mm_unpacklo_epi16(layer3_chunk1, layer3_chunk3);
    v_g1 = _mm_unpackhi_epi16(layer3_chunk1, layer3_chunk3);
}

inline void _mm_deinterleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                   __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi16(v_r0, v_g1);
    __m128i layer1_chunk1 = _mm_unpackhi_epi16(v_r0, v_g1);
    __m128i layer1_chunk2 = _mm_unpacklo_epi16(v_r1, v_b0);
    __m128i layer1_chunk3 = _mm_unpackhi_epi16(v_r1, v_b0);
    __m128i layer1_chunk4 = _mm_unpacklo_epi16(v_g0, v_b1);
    __m128i layer1_chunk5 = _mm_unpackhi_epi16(v_g0, v_b1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi16(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk1 = _mm_unpackhi_epi16(layer1_chunk0, layer1_chunk3);
    __m128i layer2_chunk2 = _mm_unpacklo_epi16(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk3 = _mm_unpackhi_epi16(layer1_chunk1, layer1_chunk4);
    __m128i layer2_chunk4 = _mm_unpacklo_epi16(layer1_chunk2, layer1_chunk5);
    __m128i layer2_chunk5 = _mm_unpackhi_epi16(layer1_chunk2, layer1_chunk5);

    __m128i layer3_chunk0 = _mm_unpacklo_epi16(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk1 = _mm_unpackhi_epi16(layer2_chunk0, layer2_chunk3);
    __m128i layer3_chunk2 = _mm_unpacklo_epi16(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk3 = _mm_unpackhi_epi16(layer2_chunk1, layer2_chunk4);
    __m128i layer3_chunk4 = _mm_unpacklo_epi16(layer2_chunk2, layer2_chunk5);
    __m128i layer3_chunk5 = _mm_unpackhi_epi16(layer2_chunk2, layer2_chunk5);

    v_r0 = _mm_unpacklo_epi16(layer3_chunk0, layer3_chunk3);
    v_r1 = _mm_unpackhi_epi16(layer3_chunk0, layer3_chunk3);
    v_g0 = _mm_unpacklo_epi16(layer3_chunk1, layer3_chunk4);
    v_g1 = _mm_unpackhi_epi16(layer3_chunk1, layer3_chunk4);
    v_b0 = _mm_unpacklo_epi16(layer3_chunk2, layer3_chunk5);
    v_b1 = _mm_unpackhi_epi16(layer3_chunk2, layer3_chunk5);
}

inline void _mm_deinterleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                   __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i layer1_chunk0 = _mm_unpacklo_epi16(v_r0, v_b0);
    __m128i layer1_chunk1 = _mm_unpackhi_epi16(v_r0, v_b0);
    __m128i layer1_chunk2 = _mm_unpacklo_epi16(v_r1, v_b1);
    __m128i layer1_chunk3 = _mm_unpackhi_epi16(v_r1, v_b1);
    __m128i layer1_chunk4 = _mm_unpacklo_epi16(v_g0, v_a0);
    __m128i layer1_chunk5 = _mm_unpackhi_epi16(v_g0, v_a0);
    __m128i layer1_chunk6 = _mm_unpacklo_epi16(v_g1, v_a1);
    __m128i layer1_chunk7 = _mm_unpackhi_epi16(v_g1, v_a1);

    __m128i layer2_chunk0 = _mm_unpacklo_epi16(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk1 = _mm_unpackhi_epi16(layer1_chunk0, layer1_chunk4);
    __m128i layer2_chunk2 = _mm_unpacklo_epi16(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk3 = _mm_unpackhi_epi16(layer1_chunk1, layer1_chunk5);
    __m128i layer2_chunk4 = _mm_unpacklo_epi16(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk5 = _mm_unpackhi_epi16(layer1_chunk2, layer1_chunk6);
    __m128i layer2_chunk6 = _mm_unpacklo_epi16(layer1_chunk3, layer1_chunk7);
    __m128i layer2_chunk7 = _mm_unpackhi_epi16(layer1_chunk3, layer1_chunk7);

    __m128i layer3_chunk0 = _mm_unpacklo_epi16(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk1 = _mm_unpackhi_epi16(layer2_chunk0, layer2_chunk4);
    __m128i layer3_chunk2 = _mm_unpacklo_epi16(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk3 = _mm_unpackhi_epi16(layer2_chunk1, layer2_chunk5);
    __m128i layer3_chunk4 = _mm_unpacklo_epi16(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk5 = _mm_unpackhi_epi16(layer2_chunk2, layer2_chunk6);
    __m128i layer3_chunk6 = _mm_unpacklo_epi16(layer2_chunk3, layer2_chunk7);
    __m128i layer3_chunk7 = _mm_unpackhi_epi16(layer2_chunk3, layer2_chunk7);

    v_r0 = _mm_unpacklo_epi16(layer3_chunk0, layer3_chunk4);
    v_r1 = _mm_unpackhi_epi16(layer3_chunk0, layer3_chunk4);
    v_g0 = _mm_unpacklo_epi16(layer3_chunk1, layer3_chunk5);
    v_g1 = _mm_unpackhi_epi16(layer3_chunk1, layer3_chunk5);
    v_b0 = _mm_unpacklo_epi16(layer3_chunk2, layer3_chunk6);
    v_b1 = _mm_unpackhi_epi16(layer3_chunk2, layer3_chunk6);
    v_a0 = _mm_unpacklo_epi16(layer3_chunk3, layer3_chunk7);
    v_a1 = _mm_unpackhi_epi16(layer3_chunk3, layer3_chunk7);
}

#if CV_SSE4_1

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_g1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
}

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0,
                                 __m128i & v_g1, __m128i & v_b0, __m128i & v_b1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi32(_mm_srli_epi32(v_b0, 16), _mm_srli_epi32(v_b1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk4, 16), _mm_srli_epi32(layer3_chunk5, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk4, 16), _mm_srli_epi32(layer2_chunk5, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_g1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
    v_g0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_b1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk4, 16), _mm_srli_epi32(layer1_chunk5, 16));
}

inline void _mm_interleave_epi16(__m128i & v_r0, __m128i & v_r1, __m128i & v_g0, __m128i & v_g1,
                                 __m128i & v_b0, __m128i & v_b1, __m128i & v_a0, __m128i & v_a1)
{
    __m128i v_mask = _mm_set1_epi32(0x0000ffff);

    __m128i layer3_chunk0 = _mm_packus_epi32(_mm_and_si128(v_r0, v_mask), _mm_and_si128(v_r1, v_mask));
    __m128i layer3_chunk4 = _mm_packus_epi32(_mm_srli_epi32(v_r0, 16), _mm_srli_epi32(v_r1, 16));
    __m128i layer3_chunk1 = _mm_packus_epi32(_mm_and_si128(v_g0, v_mask), _mm_and_si128(v_g1, v_mask));
    __m128i layer3_chunk5 = _mm_packus_epi32(_mm_srli_epi32(v_g0, 16), _mm_srli_epi32(v_g1, 16));
    __m128i layer3_chunk2 = _mm_packus_epi32(_mm_and_si128(v_b0, v_mask), _mm_and_si128(v_b1, v_mask));
    __m128i layer3_chunk6 = _mm_packus_epi32(_mm_srli_epi32(v_b0, 16), _mm_srli_epi32(v_b1, 16));
    __m128i layer3_chunk3 = _mm_packus_epi32(_mm_and_si128(v_a0, v_mask), _mm_and_si128(v_a1, v_mask));
    __m128i layer3_chunk7 = _mm_packus_epi32(_mm_srli_epi32(v_a0, 16), _mm_srli_epi32(v_a1, 16));

    __m128i layer2_chunk0 = _mm_packus_epi32(_mm_and_si128(layer3_chunk0, v_mask), _mm_and_si128(layer3_chunk1, v_mask));
    __m128i layer2_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk0, 16), _mm_srli_epi32(layer3_chunk1, 16));
    __m128i layer2_chunk1 = _mm_packus_epi32(_mm_and_si128(layer3_chunk2, v_mask), _mm_and_si128(layer3_chunk3, v_mask));
    __m128i layer2_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk2, 16), _mm_srli_epi32(layer3_chunk3, 16));
    __m128i layer2_chunk2 = _mm_packus_epi32(_mm_and_si128(layer3_chunk4, v_mask), _mm_and_si128(layer3_chunk5, v_mask));
    __m128i layer2_chunk6 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk4, 16), _mm_srli_epi32(layer3_chunk5, 16));
    __m128i layer2_chunk3 = _mm_packus_epi32(_mm_and_si128(layer3_chunk6, v_mask), _mm_and_si128(layer3_chunk7, v_mask));
    __m128i layer2_chunk7 = _mm_packus_epi32(_mm_srli_epi32(layer3_chunk6, 16), _mm_srli_epi32(layer3_chunk7, 16));

    __m128i layer1_chunk0 = _mm_packus_epi32(_mm_and_si128(layer2_chunk0, v_mask), _mm_and_si128(layer2_chunk1, v_mask));
    __m128i layer1_chunk4 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk0, 16), _mm_srli_epi32(layer2_chunk1, 16));
    __m128i layer1_chunk1 = _mm_packus_epi32(_mm_and_si128(layer2_chunk2, v_mask), _mm_and_si128(layer2_chunk3, v_mask));
    __m128i layer1_chunk5 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk2, 16), _mm_srli_epi32(layer2_chunk3, 16));
    __m128i layer1_chunk2 = _mm_packus_epi32(_mm_and_si128(layer2_chunk4, v_mask), _mm_and_si128(layer2_chunk5, v_mask));
    __m128i layer1_chunk6 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk4, 16), _mm_srli_epi32(layer2_chunk5, 16));
    __m128i layer1_chunk3 = _mm_packus_epi32(_mm_and_si128(layer2_chunk6, v_mask), _mm_and_si128(layer2_chunk7, v_mask));
    __m128i layer1_chunk7 = _mm_packus_epi32(_mm_srli_epi32(layer2_chunk6, 16), _mm_srli_epi32(layer2_chunk7, 16));

    v_r0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk0, v_mask), _mm_and_si128(layer1_chunk1, v_mask));
    v_b0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk0, 16), _mm_srli_epi32(layer1_chunk1, 16));
    v_r1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk2, v_mask), _mm_and_si128(layer1_chunk3, v_mask));
    v_b1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk2, 16), _mm_srli_epi32(layer1_chunk3, 16));
    v_g0 = _mm_packus_epi32(_mm_and_si128(layer1_chunk4, v_mask), _mm_and_si128(layer1_chunk5, v_mask));
    v_a0 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk4, 16), _mm_srli_epi32(layer1_chunk5, 16));
    v_g1 = _mm_packus_epi32(_mm_and_si128(layer1_chunk6, v_mask), _mm_and_si128(layer1_chunk7, v_mask));
    v_a1 = _mm_packus_epi32(_mm_srli_epi32(layer1_chunk6, 16), _mm_srli_epi32(layer1_chunk7, 16));
}

#endif // CV_SSE4_1

inline void _mm_deinterleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1)
{
    __m128 layer1_chunk0 = _mm_unpacklo_ps(v_r0, v_g0);
    __m128 layer1_chunk1 = _mm_unpackhi_ps(v_r0, v_g0);
    __m128 layer1_chunk2 = _mm_unpacklo_ps(v_r1, v_g1);
    __m128 layer1_chunk3 = _mm_unpackhi_ps(v_r1, v_g1);

    __m128 layer2_chunk0 = _mm_unpacklo_ps(layer1_chunk0, layer1_chunk2);
    __m128 layer2_chunk1 = _mm_unpackhi_ps(layer1_chunk0, layer1_chunk2);
    __m128 layer2_chunk2 = _mm_unpacklo_ps(layer1_chunk1, layer1_chunk3);
    __m128 layer2_chunk3 = _mm_unpackhi_ps(layer1_chunk1, layer1_chunk3);

    v_r0 = _mm_unpacklo_ps(layer2_chunk0, layer2_chunk2);
    v_r1 = _mm_unpackhi_ps(layer2_chunk0, layer2_chunk2);
    v_g0 = _mm_unpacklo_ps(layer2_chunk1, layer2_chunk3);
    v_g1 = _mm_unpackhi_ps(layer2_chunk1, layer2_chunk3);
}

inline void _mm_deinterleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0,
                                __m128 & v_g1, __m128 & v_b0, __m128 & v_b1)
{
    __m128 layer1_chunk0 = _mm_unpacklo_ps(v_r0, v_g1);
    __m128 layer1_chunk1 = _mm_unpackhi_ps(v_r0, v_g1);
    __m128 layer1_chunk2 = _mm_unpacklo_ps(v_r1, v_b0);
    __m128 layer1_chunk3 = _mm_unpackhi_ps(v_r1, v_b0);
    __m128 layer1_chunk4 = _mm_unpacklo_ps(v_g0, v_b1);
    __m128 layer1_chunk5 = _mm_unpackhi_ps(v_g0, v_b1);

    __m128 layer2_chunk0 = _mm_unpacklo_ps(layer1_chunk0, layer1_chunk3);
    __m128 layer2_chunk1 = _mm_unpackhi_ps(layer1_chunk0, layer1_chunk3);
    __m128 layer2_chunk2 = _mm_unpacklo_ps(layer1_chunk1, layer1_chunk4);
    __m128 layer2_chunk3 = _mm_unpackhi_ps(layer1_chunk1, layer1_chunk4);
    __m128 layer2_chunk4 = _mm_unpacklo_ps(layer1_chunk2, layer1_chunk5);
    __m128 layer2_chunk5 = _mm_unpackhi_ps(layer1_chunk2, layer1_chunk5);

    v_r0 = _mm_unpacklo_ps(layer2_chunk0, layer2_chunk3);
    v_r1 = _mm_unpackhi_ps(layer2_chunk0, layer2_chunk3);
    v_g0 = _mm_unpacklo_ps(layer2_chunk1, layer2_chunk4);
    v_g1 = _mm_unpackhi_ps(layer2_chunk1, layer2_chunk4);
    v_b0 = _mm_unpacklo_ps(layer2_chunk2, layer2_chunk5);
    v_b1 = _mm_unpackhi_ps(layer2_chunk2, layer2_chunk5);
}

inline void _mm_deinterleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1,
                                __m128 & v_b0, __m128 & v_b1, __m128 & v_a0, __m128 & v_a1)
{
    __m128 layer1_chunk0 = _mm_unpacklo_ps(v_r0, v_b0);
    __m128 layer1_chunk1 = _mm_unpackhi_ps(v_r0, v_b0);
    __m128 layer1_chunk2 = _mm_unpacklo_ps(v_r1, v_b1);
    __m128 layer1_chunk3 = _mm_unpackhi_ps(v_r1, v_b1);
    __m128 layer1_chunk4 = _mm_unpacklo_ps(v_g0, v_a0);
    __m128 layer1_chunk5 = _mm_unpackhi_ps(v_g0, v_a0);
    __m128 layer1_chunk6 = _mm_unpacklo_ps(v_g1, v_a1);
    __m128 layer1_chunk7 = _mm_unpackhi_ps(v_g1, v_a1);

    __m128 layer2_chunk0 = _mm_unpacklo_ps(layer1_chunk0, layer1_chunk4);
    __m128 layer2_chunk1 = _mm_unpackhi_ps(layer1_chunk0, layer1_chunk4);
    __m128 layer2_chunk2 = _mm_unpacklo_ps(layer1_chunk1, layer1_chunk5);
    __m128 layer2_chunk3 = _mm_unpackhi_ps(layer1_chunk1, layer1_chunk5);
    __m128 layer2_chunk4 = _mm_unpacklo_ps(layer1_chunk2, layer1_chunk6);
    __m128 layer2_chunk5 = _mm_unpackhi_ps(layer1_chunk2, layer1_chunk6);
    __m128 layer2_chunk6 = _mm_unpacklo_ps(layer1_chunk3, layer1_chunk7);
    __m128 layer2_chunk7 = _mm_unpackhi_ps(layer1_chunk3, layer1_chunk7);

    v_r0 = _mm_unpacklo_ps(layer2_chunk0, layer2_chunk4);
    v_r1 = _mm_unpackhi_ps(layer2_chunk0, layer2_chunk4);
    v_g0 = _mm_unpacklo_ps(layer2_chunk1, layer2_chunk5);
    v_g1 = _mm_unpackhi_ps(layer2_chunk1, layer2_chunk5);
    v_b0 = _mm_unpacklo_ps(layer2_chunk2, layer2_chunk6);
    v_b1 = _mm_unpackhi_ps(layer2_chunk2, layer2_chunk6);
    v_a0 = _mm_unpacklo_ps(layer2_chunk3, layer2_chunk7);
    v_a1 = _mm_unpackhi_ps(layer2_chunk3, layer2_chunk7);
}

inline void _mm_interleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 layer2_chunk0 = _mm_shuffle_ps(v_r0, v_r1, mask_lo);
    __m128 layer2_chunk2 = _mm_shuffle_ps(v_r0, v_r1, mask_hi);
    __m128 layer2_chunk1 = _mm_shuffle_ps(v_g0, v_g1, mask_lo);
    __m128 layer2_chunk3 = _mm_shuffle_ps(v_g0, v_g1, mask_hi);

    __m128 layer1_chunk0 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_lo);
    __m128 layer1_chunk2 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_hi);
    __m128 layer1_chunk1 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_lo);
    __m128 layer1_chunk3 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_hi);

    v_r0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_lo);
    v_g0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_hi);
    v_r1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_lo);
    v_g1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_hi);
}

inline void _mm_interleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0,
                              __m128 & v_g1, __m128 & v_b0, __m128 & v_b1)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 layer2_chunk0 = _mm_shuffle_ps(v_r0, v_r1, mask_lo);
    __m128 layer2_chunk3 = _mm_shuffle_ps(v_r0, v_r1, mask_hi);
    __m128 layer2_chunk1 = _mm_shuffle_ps(v_g0, v_g1, mask_lo);
    __m128 layer2_chunk4 = _mm_shuffle_ps(v_g0, v_g1, mask_hi);
    __m128 layer2_chunk2 = _mm_shuffle_ps(v_b0, v_b1, mask_lo);
    __m128 layer2_chunk5 = _mm_shuffle_ps(v_b0, v_b1, mask_hi);

    __m128 layer1_chunk0 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_lo);
    __m128 layer1_chunk3 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_hi);
    __m128 layer1_chunk1 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_lo);
    __m128 layer1_chunk4 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_hi);
    __m128 layer1_chunk2 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_lo);
    __m128 layer1_chunk5 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_hi);

    v_r0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_lo);
    v_g1 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_hi);
    v_r1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_lo);
    v_b0 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_hi);
    v_g0 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_lo);
    v_b1 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_hi);
}

inline void _mm_interleave_ps(__m128 & v_r0, __m128 & v_r1, __m128 & v_g0, __m128 & v_g1,
                              __m128 & v_b0, __m128 & v_b1, __m128 & v_a0, __m128 & v_a1)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 layer2_chunk0 = _mm_shuffle_ps(v_r0, v_r1, mask_lo);
    __m128 layer2_chunk4 = _mm_shuffle_ps(v_r0, v_r1, mask_hi);
    __m128 layer2_chunk1 = _mm_shuffle_ps(v_g0, v_g1, mask_lo);
    __m128 layer2_chunk5 = _mm_shuffle_ps(v_g0, v_g1, mask_hi);
    __m128 layer2_chunk2 = _mm_shuffle_ps(v_b0, v_b1, mask_lo);
    __m128 layer2_chunk6 = _mm_shuffle_ps(v_b0, v_b1, mask_hi);
    __m128 layer2_chunk3 = _mm_shuffle_ps(v_a0, v_a1, mask_lo);
    __m128 layer2_chunk7 = _mm_shuffle_ps(v_a0, v_a1, mask_hi);

    __m128 layer1_chunk0 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_lo);
    __m128 layer1_chunk4 = _mm_shuffle_ps(layer2_chunk0, layer2_chunk1, mask_hi);
    __m128 layer1_chunk1 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_lo);
    __m128 layer1_chunk5 = _mm_shuffle_ps(layer2_chunk2, layer2_chunk3, mask_hi);
    __m128 layer1_chunk2 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_lo);
    __m128 layer1_chunk6 = _mm_shuffle_ps(layer2_chunk4, layer2_chunk5, mask_hi);
    __m128 layer1_chunk3 = _mm_shuffle_ps(layer2_chunk6, layer2_chunk7, mask_lo);
    __m128 layer1_chunk7 = _mm_shuffle_ps(layer2_chunk6, layer2_chunk7, mask_hi);

    v_r0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_lo);
    v_b0 = _mm_shuffle_ps(layer1_chunk0, layer1_chunk1, mask_hi);
    v_r1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_lo);
    v_b1 = _mm_shuffle_ps(layer1_chunk2, layer1_chunk3, mask_hi);
    v_g0 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_lo);
    v_a0 = _mm_shuffle_ps(layer1_chunk4, layer1_chunk5, mask_hi);
    v_g1 = _mm_shuffle_ps(layer1_chunk6, layer1_chunk7, mask_lo);
    v_a1 = _mm_shuffle_ps(layer1_chunk6, layer1_chunk7, mask_hi);
}

#endif // CV_SSE2

//! @}

#endif //OPENCV_CORE_SSE_UTILS_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_TRAITS_HPP
#define OPENCV_CORE_TRAITS_HPP

#include "opencv2/core/cvdef.h"

namespace cv
{

//! @addtogroup core_basic
//! @{

/** @brief Template "trait" class for OpenCV primitive data types.

A primitive OpenCV data type is one of unsigned char, bool, signed char, unsigned short, signed
short, int, float, double, or a tuple of values of one of these types, where all the values in the
tuple have the same type. Any primitive type from the list can be defined by an identifier in the
form CV_\<bit-depth\>{U|S|F}C(\<number_of_channels\>), for example: uchar \~ CV_8UC1, 3-element
floating-point tuple \~ CV_32FC3, and so on. A universal OpenCV structure that is able to store a
single instance of such a primitive data type is Vec. Multiple instances of such a type can be
stored in a std::vector, Mat, Mat_, SparseMat, SparseMat_, or any other container that is able to
store Vec instances.

The DataType class is basically used to provide a description of such primitive data types without
adding any fields or methods to the corresponding classes (and it is actually impossible to add
anything to primitive C/C++ data types). This technique is known in C++ as class traits. It is not
DataType itself that is used but its specialized versions, such as:
@code
    template<> class DataType<uchar>
    {
        typedef uchar value_type;
        typedef int work_type;
        typedef uchar channel_type;
        enum { channel_type = CV_8U, channels = 1, fmt='u', type = CV_8U };
    };
    ...
    template<typename _Tp> DataType<std::complex<_Tp> >
    {
        typedef std::complex<_Tp> value_type;
        typedef std::complex<_Tp> work_type;
        typedef _Tp channel_type;
        // DataDepth is another helper trait class
        enum { depth = DataDepth<_Tp>::value, channels=2,
            fmt=(channels-1)*256+DataDepth<_Tp>::fmt,
            type=CV_MAKETYPE(depth, channels) };
    };
    ...
@endcode
The main purpose of this class is to convert compilation-time type information to an
OpenCV-compatible data type identifier, for example:
@code
    // allocates a 30x40 floating-point matrix
    Mat A(30, 40, DataType<float>::type);

    Mat B = Mat_<std::complex<double> >(3, 3);
    // the statement below will print 6, 2 , that is depth == CV_64F, channels == 2
    cout << B.depth() << ", " << B.channels() << endl;
@endcode
So, such traits are used to tell OpenCV which data type you are working with, even if such a type is
not native to OpenCV. For example, the matrix B initialization above is compiled because OpenCV
defines the proper specialized template class DataType\<complex\<_Tp\> \> . This mechanism is also
useful (and used in OpenCV this way) for generic algorithms implementations.
*/
template<typename _Tp> class DataType
{
public:
    typedef _Tp         value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 1,
           depth        = -1,
           channels     = 1,
           fmt          = 0,
           type = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<bool>
{
public:
    typedef bool        value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8U,
           channels     = 1,
           fmt          = (int)'u',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<uchar>
{
public:
    typedef uchar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8U,
           channels     = 1,
           fmt          = (int)'u',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<schar>
{
public:
    typedef schar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8S,
           channels     = 1,
           fmt          = (int)'c',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<char>
{
public:
    typedef schar       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_8S,
           channels     = 1,
           fmt          = (int)'c',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<ushort>
{
public:
    typedef ushort      value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_16U,
           channels     = 1,
           fmt          = (int)'w',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<short>
{
public:
    typedef short       value_type;
    typedef int         work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_16S,
           channels     = 1,
           fmt          = (int)'s',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<int>
{
public:
    typedef int         value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32S,
           channels     = 1,
           fmt          = (int)'i',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<float>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 1,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

template<> class DataType<double>
{
public:
    typedef double      value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_64F,
           channels     = 1,
           fmt          = (int)'d',
           type         = CV_MAKETYPE(depth, channels)
         };
};


/** @brief A helper class for cv::DataType

The class is specialized for each fundamental numerical data type supported by OpenCV. It provides
DataDepth<T>::value constant.
*/
template<typename _Tp> class DataDepth
{
public:
    enum
    {
        value = DataType<_Tp>::depth,
        fmt   = DataType<_Tp>::fmt
    };
};



template<int _depth> class TypeDepth
{
    enum { depth = CV_USRTYPE1 };
    typedef void value_type;
};

template<> class TypeDepth<CV_8U>
{
    enum { depth = CV_8U };
    typedef uchar value_type;
};

template<> class TypeDepth<CV_8S>
{
    enum { depth = CV_8S };
    typedef schar value_type;
};

template<> class TypeDepth<CV_16U>
{
    enum { depth = CV_16U };
    typedef ushort value_type;
};

template<> class TypeDepth<CV_16S>
{
    enum { depth = CV_16S };
    typedef short value_type;
};

template<> class TypeDepth<CV_32S>
{
    enum { depth = CV_32S };
    typedef int value_type;
};

template<> class TypeDepth<CV_32F>
{
    enum { depth = CV_32F };
    typedef float value_type;
};

template<> class TypeDepth<CV_64F>
{
    enum { depth = CV_64F };
    typedef double value_type;
};

//! @}

} // cv

#endif // OPENCV_CORE_TRAITS_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_TYPES_HPP
#define OPENCV_CORE_TYPES_HPP

#ifndef __cplusplus
#  error types.hpp header must be compiled as C++
#endif

#include <climits>
#include <cfloat>
#include <vector>
#include <limits>

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/matx.hpp"

namespace cv
{

//! @addtogroup core_basic
//! @{

//////////////////////////////// Complex //////////////////////////////

/** @brief  A complex number class.

  The template class is similar and compatible with std::complex, however it provides slightly
  more convenient access to the real and imaginary parts using through the simple field access, as opposite
  to std::complex::real() and std::complex::imag().
*/
template<typename _Tp> class Complex
{
public:

    //! constructors
    Complex();
    Complex( _Tp _re, _Tp _im = 0 );

    //! conversion to another data type
    template<typename T2> operator Complex<T2>() const;
    //! conjugation
    Complex conj() const;

    _Tp re, im; //< the real and the imaginary parts
};

typedef Complex<float> Complexf;
typedef Complex<double> Complexd;

template<typename _Tp> class DataType< Complex<_Tp> >
{
public:
    typedef Complex<_Tp> value_type;
    typedef value_type   work_type;
    typedef _Tp          channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels) };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// Point_ ////////////////////////////////

/** @brief Template class for 2D points specified by its coordinates `x` and `y`.

An instance of the class is interchangeable with C structures, CvPoint and CvPoint2D32f . There is
also a cast operator to convert point coordinates to the specified type. The conversion from
floating-point coordinates to integer coordinates is done by rounding. Commonly, the conversion
uses this operation for each of the coordinates. Besides the class members listed in the
declaration above, the following operations on points are implemented:
@code
    pt1 = pt2 + pt3;
    pt1 = pt2 - pt3;
    pt1 = pt2 * a;
    pt1 = a * pt2;
    pt1 = pt2 / a;
    pt1 += pt2;
    pt1 -= pt2;
    pt1 *= a;
    pt1 /= a;
    double value = norm(pt); // L2 norm
    pt1 == pt2;
    pt1 != pt2;
@endcode
For your convenience, the following type aliases are defined:
@code
    typedef Point_<int> Point2i;
    typedef Point2i Point;
    typedef Point_<float> Point2f;
    typedef Point_<double> Point2d;
@endcode
Example:
@code
    Point2f a(0.3f, 0.f), b(0.f, 0.4f);
    Point pt = (a + b)*10.f;
    cout << pt.x << ", " << pt.y << endl;
@endcode
*/
template<typename _Tp> class Point_
{
public:
    typedef _Tp value_type;

    // various constructors
    Point_();
    Point_(_Tp _x, _Tp _y);
    Point_(const Point_& pt);
    Point_(const Size_<_Tp>& sz);
    Point_(const Vec<_Tp, 2>& v);

    Point_& operator = (const Point_& pt);
    //! conversion to another data type
    template<typename _Tp2> operator Point_<_Tp2>() const;

    //! conversion to the old-style C structures
    operator Vec<_Tp, 2>() const;

    //! dot product
    _Tp dot(const Point_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point_& pt) const;
    //! cross-product
    double cross(const Point_& pt) const;
    //! checks whether the point is inside the specified rectangle
    bool inside(const Rect_<_Tp>& r) const;

    _Tp x, y; //< the point coordinates
};

typedef Point_<int> Point2i;
typedef Point_<int64> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

template<typename _Tp> class DataType< Point_<_Tp> >
{
public:
    typedef Point_<_Tp>                               value_type;
    typedef Point_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp                                       channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// Point3_ ////////////////////////////////

/** @brief Template class for 3D points specified by its coordinates `x`, `y` and `z`.

An instance of the class is interchangeable with the C structure CvPoint2D32f . Similarly to
Point_ , the coordinates of 3D points can be converted to another type. The vector arithmetic and
comparison operations are also supported.

The following Point3_\<\> aliases are available:
@code
    typedef Point3_<int> Point3i;
    typedef Point3_<float> Point3f;
    typedef Point3_<double> Point3d;
@endcode
@see cv::Point3i, cv::Point3f and cv::Point3d
*/
template<typename _Tp> class Point3_
{
public:
    typedef _Tp value_type;

    // various constructors
    Point3_();
    Point3_(_Tp _x, _Tp _y, _Tp _z);
    Point3_(const Point3_& pt);
    explicit Point3_(const Point_<_Tp>& pt);
    Point3_(const Vec<_Tp, 3>& v);

    Point3_& operator = (const Point3_& pt);
    //! conversion to another data type
    template<typename _Tp2> operator Point3_<_Tp2>() const;
    //! conversion to cv::Vec<>
#if OPENCV_ABI_COMPATIBILITY > 300
    template<typename _Tp2> operator Vec<_Tp2, 3>() const;
#else
    operator Vec<_Tp, 3>() const;
#endif

    //! dot product
    _Tp dot(const Point3_& pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point3_& pt) const;
    //! cross product of the 2 3D points
    Point3_ cross(const Point3_& pt) const;

    _Tp x, y, z; //< the point coordinates
};

typedef Point3_<int> Point3i;
typedef Point3_<float> Point3f;
typedef Point3_<double> Point3d;

template<typename _Tp> class DataType< Point3_<_Tp> >
{
public:
    typedef Point3_<_Tp>                               value_type;
    typedef Point3_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp                                        channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 3,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// Size_ ////////////////////////////////

/** @brief Template class for specifying the size of an image or rectangle.

The class includes two members called width and height. The structure can be converted to and from
the old OpenCV structures CvSize and CvSize2D32f . The same set of arithmetic and comparison
operations as for Point_ is available.

OpenCV defines the following Size_\<\> aliases:
@code
    typedef Size_<int> Size2i;
    typedef Size2i Size;
    typedef Size_<float> Size2f;
@endcode
*/
template<typename _Tp> class Size_
{
public:
    typedef _Tp value_type;

    //! various constructors
    Size_();
    Size_(_Tp _width, _Tp _height);
    Size_(const Size_& sz);
    Size_(const Point_<_Tp>& pt);

    Size_& operator = (const Size_& sz);
    //! the area (width*height)
    _Tp area() const;
    //! true if empty
    bool empty() const;

    //! conversion of another data type.
    template<typename _Tp2> operator Size_<_Tp2>() const;

    _Tp width, height; // the width and the height
};

typedef Size_<int> Size2i;
typedef Size_<int64> Size2l;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

template<typename _Tp> class DataType< Size_<_Tp> >
{
public:
    typedef Size_<_Tp>                               value_type;
    typedef Size_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp                                      channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// Rect_ ////////////////////////////////

/** @brief Template class for 2D rectangles

described by the following parameters:
-   Coordinates of the top-left corner. This is a default interpretation of Rect_::x and Rect_::y
    in OpenCV. Though, in your algorithms you may count x and y from the bottom-left corner.
-   Rectangle width and height.

OpenCV typically assumes that the top and left boundary of the rectangle are inclusive, while the
right and bottom boundaries are not. For example, the method Rect_::contains returns true if

\f[x  \leq pt.x < x+width,
      y  \leq pt.y < y+height\f]

Virtually every loop over an image ROI in OpenCV (where ROI is specified by Rect_\<int\> ) is
implemented as:
@code
    for(int y = roi.y; y < roi.y + roi.height; y++)
        for(int x = roi.x; x < roi.x + roi.width; x++)
        {
            // ...
        }
@endcode
In addition to the class members, the following operations on rectangles are implemented:
-   \f$\texttt{rect} = \texttt{rect} \pm \texttt{point}\f$ (shifting a rectangle by a certain offset)
-   \f$\texttt{rect} = \texttt{rect} \pm \texttt{size}\f$ (expanding or shrinking a rectangle by a
    certain amount)
-   rect += point, rect -= point, rect += size, rect -= size (augmenting operations)
-   rect = rect1 & rect2 (rectangle intersection)
-   rect = rect1 | rect2 (minimum area rectangle containing rect1 and rect2 )
-   rect &= rect1, rect |= rect1 (and the corresponding augmenting operations)
-   rect == rect1, rect != rect1 (rectangle comparison)

This is an example how the partial ordering on rectangles can be established (rect1 \f$\subseteq\f$
rect2):
@code
    template<typename _Tp> inline bool
    operator <= (const Rect_<_Tp>& r1, const Rect_<_Tp>& r2)
    {
        return (r1 & r2) == r1;
    }
@endcode
For your convenience, the Rect_\<\> alias is available: cv::Rect
*/
template<typename _Tp> class Rect_
{
public:
    typedef _Tp value_type;

    //! various constructors
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
    Rect_(const Rect_& r);
    Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz);
    Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2);

    Rect_& operator = ( const Rect_& r );
    //! the top-left corner
    Point_<_Tp> tl() const;
    //! the bottom-right corner
    Point_<_Tp> br() const;

    //! size (width, height) of the rectangle
    Size_<_Tp> size() const;
    //! area (width*height) of the rectangle
    _Tp area() const;
    //! true if empty
    bool empty() const;

    //! conversion to another data type
    template<typename _Tp2> operator Rect_<_Tp2>() const;

    //! checks whether the rectangle contains the point
    bool contains(const Point_<_Tp>& pt) const;

    _Tp x, y, width, height; //< the top-left corner, as well as width and height of the rectangle
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;

template<typename _Tp> class DataType< Rect_<_Tp> >
{
public:
    typedef Rect_<_Tp>                               value_type;
    typedef Rect_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp                                      channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 4,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



///////////////////////////// RotatedRect /////////////////////////////

/** @brief The class represents rotated (i.e. not up-right) rectangles on a plane.

Each rectangle is specified by the center point (mass center), length of each side (represented by
cv::Size2f structure) and the rotation angle in degrees.

The sample below demonstrates how to use RotatedRect:
@code
    Mat image(200, 200, CV_8UC3, Scalar(0));
    RotatedRect rRect = RotatedRect(Point2f(100,100), Size2f(100,50), 30);

    Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));

    Rect brect = rRect.boundingRect();
    rectangle(image, brect, Scalar(255,0,0));

    imshow("rectangles", image);
    waitKey(0);
@endcode
![image](pics/rotatedrect.png)

@sa CamShift, fitEllipse, minAreaRect, CvBox2D
*/
class CV_EXPORTS RotatedRect
{
public:
    //! various constructors
    RotatedRect();
    /**
    @param center The rectangle mass center.
    @param size Width and height of the rectangle.
    @param angle The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc.,
    the rectangle becomes an up-right rectangle.
    */
    RotatedRect(const Point2f& center, const Size2f& size, float angle);
    /**
    Any 3 end points of the RotatedRect. They must be given in order (either clockwise or
    anticlockwise).
     */
    RotatedRect(const Point2f& point1, const Point2f& point2, const Point2f& point3);

    /** returns 4 vertices of the rectangle
    @param pts The points array for storing rectangle vertices.
    */
    void points(Point2f pts[]) const;
    //! returns the minimal up-right integer rectangle containing the rotated rectangle
    Rect boundingRect() const;
    //! returns the minimal (exact) floating point rectangle containing the rotated rectangle, not intended for use with images
    Rect_<float> boundingRect2f() const;

    Point2f center; //< the rectangle mass center
    Size2f size;    //< width and height of the rectangle
    float angle;    //< the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
};

template<> class DataType< RotatedRect >
{
public:
    typedef RotatedRect  value_type;
    typedef value_type   work_type;
    typedef float        channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = (int)sizeof(value_type)/sizeof(channel_type), // 5
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// Range /////////////////////////////////

/** @brief Template class specifying a continuous subsequence (slice) of a sequence.

The class is used to specify a row or a column span in a matrix ( Mat ) and for many other purposes.
Range(a,b) is basically the same as a:b in Matlab or a..b in Python. As in Python, start is an
inclusive left boundary of the range and end is an exclusive right boundary of the range. Such a
half-opened interval is usually denoted as \f$[start,end)\f$ .

The static method Range::all() returns a special variable that means "the whole sequence" or "the
whole range", just like " : " in Matlab or " ... " in Python. All the methods and functions in
OpenCV that take Range support this special Range::all() value. But, of course, in case of your own
custom processing, you will probably have to check and handle it explicitly:
@code
    void my_function(..., const Range& r, ....)
    {
        if(r == Range::all()) {
            // process all the data
        }
        else {
            // process [r.start, r.end)
        }
    }
@endcode
*/
class CV_EXPORTS Range
{
public:
    Range();
    Range(int _start, int _end);
    int size() const;
    bool empty() const;
    static Range all();

    int start, end;
};

template<> class DataType<Range>
{
public:
    typedef Range      value_type;
    typedef value_type work_type;
    typedef int        channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 2,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// Scalar_ ///////////////////////////////

/** @brief Template class for a 4-element vector derived from Vec.

Being derived from Vec\<_Tp, 4\> , Scalar\_ and Scalar can be used just as typical 4-element
vectors. In addition, they can be converted to/from CvScalar . The type Scalar is widely used in
OpenCV to pass pixel values.
*/
template<typename _Tp> class Scalar_ : public Vec<_Tp, 4>
{
public:
    //! various constructors
    Scalar_();
    Scalar_(_Tp v0, _Tp v1, _Tp v2=0, _Tp v3=0);
    Scalar_(_Tp v0);

    template<typename _Tp2, int cn>
    Scalar_(const Vec<_Tp2, cn>& v);

    //! returns a scalar with all elements set to v0
    static Scalar_<_Tp> all(_Tp v0);

    //! conversion to another data type
    template<typename T2> operator Scalar_<T2>() const;

    //! per-element product
    Scalar_<_Tp> mul(const Scalar_<_Tp>& a, double scale=1 ) const;

    // returns (v0, -v1, -v2, -v3)
    Scalar_<_Tp> conj() const;

    // returns true iff v1 == v2 == v3 == 0
    bool isReal() const;
};

typedef Scalar_<double> Scalar;

template<typename _Tp> class DataType< Scalar_<_Tp> >
{
public:
    typedef Scalar_<_Tp>                               value_type;
    typedef Scalar_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp                                        channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = 4,
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



/////////////////////////////// KeyPoint ////////////////////////////////

/** @brief Data structure for salient point detectors.

The class instance stores a keypoint, i.e. a point feature found by one of many available keypoint
detectors, such as Harris corner detector, cv::FAST, cv::StarDetector, cv::SURF, cv::SIFT,
cv::LDetector etc.

The keypoint is characterized by the 2D position, scale (proportional to the diameter of the
neighborhood that needs to be taken into account), orientation and some other parameters. The
keypoint neighborhood is then analyzed by another algorithm that builds a descriptor (usually
represented as a feature vector). The keypoints representing the same object in different images
can then be matched using cv::KDTree or another method.
*/
class CV_EXPORTS_W_SIMPLE KeyPoint
{
public:
    //! the default constructor
    CV_WRAP KeyPoint();
    /**
    @param _pt x & y coordinates of the keypoint
    @param _size keypoint diameter
    @param _angle keypoint orientation
    @param _response keypoint detector response on the keypoint (that is, strength of the keypoint)
    @param _octave pyramid octave in which the keypoint has been detected
    @param _class_id object id
     */
    KeyPoint(Point2f _pt, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1);
    /**
    @param x x-coordinate of the keypoint
    @param y y-coordinate of the keypoint
    @param _size keypoint diameter
    @param _angle keypoint orientation
    @param _response keypoint detector response on the keypoint (that is, strength of the keypoint)
    @param _octave pyramid octave in which the keypoint has been detected
    @param _class_id object id
     */
    CV_WRAP KeyPoint(float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1);

    size_t hash() const;

    /**
    This method converts vector of keypoints to vector of points or the reverse, where each keypoint is
    assigned the same size and the same orientation.

    @param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
    @param points2f Array of (x,y) coordinates of each keypoint
    @param keypointIndexes Array of indexes of keypoints to be converted to points. (Acts like a mask to
    convert only specified keypoints)
    */
    CV_WRAP static void convert(const std::vector<KeyPoint>& keypoints,
                                CV_OUT std::vector<Point2f>& points2f,
                                const std::vector<int>& keypointIndexes=std::vector<int>());
    /** @overload
    @param points2f Array of (x,y) coordinates of each keypoint
    @param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
    @param size keypoint diameter
    @param response keypoint detector response on the keypoint (that is, strength of the keypoint)
    @param octave pyramid octave in which the keypoint has been detected
    @param class_id object id
    */
    CV_WRAP static void convert(const std::vector<Point2f>& points2f,
                                CV_OUT std::vector<KeyPoint>& keypoints,
                                float size=1, float response=1, int octave=0, int class_id=-1);

    /**
    This method computes overlap for pair of keypoints. Overlap is the ratio between area of keypoint
    regions' intersection and area of keypoint regions' union (considering keypoint region as circle).
    If they don't overlap, we get zero. If they coincide at same location with same size, we get 1.
    @param kp1 First keypoint
    @param kp2 Second keypoint
    */
    CV_WRAP static float overlap(const KeyPoint& kp1, const KeyPoint& kp2);

    CV_PROP_RW Point2f pt; //!< coordinates of the keypoints
    CV_PROP_RW float size; //!< diameter of the meaningful keypoint neighborhood
    CV_PROP_RW float angle; //!< computed orientation of the keypoint (-1 if not applicable);
                            //!< it's in [0,360) degrees and measured relative to
                            //!< image coordinate system, ie in clockwise.
    CV_PROP_RW float response; //!< the response by which the most strong keypoints have been selected. Can be used for the further sorting or subsampling
    CV_PROP_RW int octave; //!< octave (pyramid layer) from which the keypoint has been extracted
    CV_PROP_RW int class_id; //!< object class (if the keypoints need to be clustered by an object they belong to)
};

template<> class DataType<KeyPoint>
{
public:
    typedef KeyPoint      value_type;
    typedef float         work_type;
    typedef float         channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = (int)(sizeof(value_type)/sizeof(channel_type)), // 7
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



//////////////////////////////// DMatch /////////////////////////////////

/** @brief Class for matching keypoint descriptors

query descriptor index, train descriptor index, train image index, and distance between
descriptors.
*/
class CV_EXPORTS_W_SIMPLE DMatch
{
public:
    CV_WRAP DMatch();
    CV_WRAP DMatch(int _queryIdx, int _trainIdx, float _distance);
    CV_WRAP DMatch(int _queryIdx, int _trainIdx, int _imgIdx, float _distance);

    CV_PROP_RW int queryIdx; // query descriptor index
    CV_PROP_RW int trainIdx; // train descriptor index
    CV_PROP_RW int imgIdx;   // train image index

    CV_PROP_RW float distance;

    // less is better
    bool operator<(const DMatch &m) const;
};

template<> class DataType<DMatch>
{
public:
    typedef DMatch      value_type;
    typedef int         work_type;
    typedef int         channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = (int)(sizeof(value_type)/sizeof(channel_type)), // 4
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};



///////////////////////////// TermCriteria //////////////////////////////

/** @brief The class defining termination criteria for iterative algorithms.

You can initialize it by default constructor and then override any parameters, or the structure may
be fully initialized using the advanced variant of the constructor.
*/
class CV_EXPORTS TermCriteria
{
public:
    /**
      Criteria type, can be one of: COUNT, EPS or COUNT + EPS
    */
    enum Type
    {
        COUNT=1, //!< the maximum number of iterations or elements to compute
        MAX_ITER=COUNT, //!< ditto
        EPS=2 //!< the desired accuracy or change in parameters at which the iterative algorithm stops
    };

    //! default constructor
    TermCriteria();
    /**
    @param type The type of termination criteria, one of TermCriteria::Type
    @param maxCount The maximum number of iterations or elements to compute.
    @param epsilon The desired accuracy or change in parameters at which the iterative algorithm stops.
    */
    TermCriteria(int type, int maxCount, double epsilon);

    int type; //!< the type of termination criteria: COUNT, EPS or COUNT + EPS
    int maxCount; // the maximum number of iterations/elements
    double epsilon; // the desired accuracy
};


//! @} core_basic

///////////////////////// raster image moments //////////////////////////

//! @addtogroup imgproc_shape
//! @{

/** @brief struct returned by cv::moments

The spatial moments \f$\texttt{Moments::m}_{ji}\f$ are computed as:

\f[\texttt{m} _{ji}= \sum _{x,y}  \left ( \texttt{array} (x,y)  \cdot x^j  \cdot y^i \right )\f]

The central moments \f$\texttt{Moments::mu}_{ji}\f$ are computed as:

\f[\texttt{mu} _{ji}= \sum _{x,y}  \left ( \texttt{array} (x,y)  \cdot (x -  \bar{x} )^j  \cdot (y -  \bar{y} )^i \right )\f]

where \f$(\bar{x}, \bar{y})\f$ is the mass center:

\f[\bar{x} = \frac{\texttt{m}_{10}}{\texttt{m}_{00}} , \; \bar{y} = \frac{\texttt{m}_{01}}{\texttt{m}_{00}}\f]

The normalized central moments \f$\texttt{Moments::nu}_{ij}\f$ are computed as:

\f[\texttt{nu} _{ji}= \frac{\texttt{mu}_{ji}}{\texttt{m}_{00}^{(i+j)/2+1}} .\f]

@note
\f$\texttt{mu}_{00}=\texttt{m}_{00}\f$, \f$\texttt{nu}_{00}=1\f$
\f$\texttt{nu}_{10}=\texttt{mu}_{10}=\texttt{mu}_{01}=\texttt{mu}_{10}=0\f$ , hence the values are not
stored.

The moments of a contour are defined in the same way but computed using the Green's formula (see
<http://en.wikipedia.org/wiki/Green_theorem>). So, due to a limited raster resolution, the moments
computed for a contour are slightly different from the moments computed for the same rasterized
contour.

@note
Since the contour moments are computed using Green formula, you may get seemingly odd results for
contours with self-intersections, e.g. a zero area (m00) for butterfly-shaped contours.
 */
class CV_EXPORTS_W_MAP Moments
{
public:
    //! the default constructor
    Moments();
    //! the full constructor
    Moments(double m00, double m10, double m01, double m20, double m11,
            double m02, double m30, double m21, double m12, double m03 );
    ////! the conversion from CvMoments
    //Moments( const CvMoments& moments );
    ////! the conversion to CvMoments
    //operator CvMoments() const;

    //! @name spatial moments
    //! @{
    CV_PROP_RW double  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    //! @}

    //! @name central moments
    //! @{
    CV_PROP_RW double  mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    //! @}

    //! @name central normalized moments
    //! @{
    CV_PROP_RW double  nu20, nu11, nu02, nu30, nu21, nu12, nu03;
    //! @}
};

template<> class DataType<Moments>
{
public:
    typedef Moments     value_type;
    typedef double      work_type;
    typedef double      channel_type;

    enum { generic_type = 0,
           depth        = DataType<channel_type>::depth,
           channels     = (int)(sizeof(value_type)/sizeof(channel_type)), // 24
           fmt          = DataType<channel_type>::fmt + ((channels - 1) << 8),
           type         = CV_MAKETYPE(depth, channels)
         };

    typedef Vec<channel_type, channels> vec_type;
};

//! @} imgproc_shape

//! @cond IGNORED

/////////////////////////////////////////////////////////////////////////
///////////////////////////// Implementation ////////////////////////////
/////////////////////////////////////////////////////////////////////////

//////////////////////////////// Complex ////////////////////////////////

template<typename _Tp> inline
Complex<_Tp>::Complex()
    : re(0), im(0) {}

template<typename _Tp> inline
Complex<_Tp>::Complex( _Tp _re, _Tp _im )
    : re(_re), im(_im) {}

template<typename _Tp> template<typename T2> inline
Complex<_Tp>::operator Complex<T2>() const
{
    return Complex<T2>(saturate_cast<T2>(re), saturate_cast<T2>(im));
}

template<typename _Tp> inline
Complex<_Tp> Complex<_Tp>::conj() const
{
    return Complex<_Tp>(re, -im);
}


template<typename _Tp> static inline
bool operator == (const Complex<_Tp>& a, const Complex<_Tp>& b)
{
    return a.re == b.re && a.im == b.im;
}

template<typename _Tp> static inline
bool operator != (const Complex<_Tp>& a, const Complex<_Tp>& b)
{
    return a.re != b.re || a.im != b.im;
}

template<typename _Tp> static inline
Complex<_Tp> operator + (const Complex<_Tp>& a, const Complex<_Tp>& b)
{
    return Complex<_Tp>( a.re + b.re, a.im + b.im );
}

template<typename _Tp> static inline
Complex<_Tp>& operator += (Complex<_Tp>& a, const Complex<_Tp>& b)
{
    a.re += b.re; a.im += b.im;
    return a;
}

template<typename _Tp> static inline
Complex<_Tp> operator - (const Complex<_Tp>& a, const Complex<_Tp>& b)
{
    return Complex<_Tp>( a.re - b.re, a.im - b.im );
}

template<typename _Tp> static inline
Complex<_Tp>& operator -= (Complex<_Tp>& a, const Complex<_Tp>& b)
{
    a.re -= b.re; a.im -= b.im;
    return a;
}

template<typename _Tp> static inline
Complex<_Tp> operator - (const Complex<_Tp>& a)
{
    return Complex<_Tp>(-a.re, -a.im);
}

template<typename _Tp> static inline
Complex<_Tp> operator * (const Complex<_Tp>& a, const Complex<_Tp>& b)
{
    return Complex<_Tp>( a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re );
}

template<typename _Tp> static inline
Complex<_Tp> operator * (const Complex<_Tp>& a, _Tp b)
{
    return Complex<_Tp>( a.re*b, a.im*b );
}

template<typename _Tp> static inline
Complex<_Tp> operator * (_Tp b, const Complex<_Tp>& a)
{
    return Complex<_Tp>( a.re*b, a.im*b );
}

template<typename _Tp> static inline
Complex<_Tp> operator + (const Complex<_Tp>& a, _Tp b)
{
    return Complex<_Tp>( a.re + b, a.im );
}

template<typename _Tp> static inline
Complex<_Tp> operator - (const Complex<_Tp>& a, _Tp b)
{ return Complex<_Tp>( a.re - b, a.im ); }

template<typename _Tp> static inline
Complex<_Tp> operator + (_Tp b, const Complex<_Tp>& a)
{
    return Complex<_Tp>( a.re + b, a.im );
}

template<typename _Tp> static inline
Complex<_Tp> operator - (_Tp b, const Complex<_Tp>& a)
{
    return Complex<_Tp>( b - a.re, -a.im );
}

template<typename _Tp> static inline
Complex<_Tp>& operator += (Complex<_Tp>& a, _Tp b)
{
    a.re += b; return a;
}

template<typename _Tp> static inline
Complex<_Tp>& operator -= (Complex<_Tp>& a, _Tp b)
{
    a.re -= b; return a;
}

template<typename _Tp> static inline
Complex<_Tp>& operator *= (Complex<_Tp>& a, _Tp b)
{
    a.re *= b; a.im *= b; return a;
}

template<typename _Tp> static inline
double abs(const Complex<_Tp>& a)
{
    return std::sqrt( (double)a.re*a.re + (double)a.im*a.im);
}

template<typename _Tp> static inline
Complex<_Tp> operator / (const Complex<_Tp>& a, const Complex<_Tp>& b)
{
    double t = 1./((double)b.re*b.re + (double)b.im*b.im);
    return Complex<_Tp>( (_Tp)((a.re*b.re + a.im*b.im)*t),
                        (_Tp)((-a.re*b.im + a.im*b.re)*t) );
}

template<typename _Tp> static inline
Complex<_Tp>& operator /= (Complex<_Tp>& a, const Complex<_Tp>& b)
{
    a = a / b;
    return a;
}

template<typename _Tp> static inline
Complex<_Tp> operator / (const Complex<_Tp>& a, _Tp b)
{
    _Tp t = (_Tp)1/b;
    return Complex<_Tp>( a.re*t, a.im*t );
}

template<typename _Tp> static inline
Complex<_Tp> operator / (_Tp b, const Complex<_Tp>& a)
{
    return Complex<_Tp>(b)/a;
}

template<typename _Tp> static inline
Complex<_Tp> operator /= (const Complex<_Tp>& a, _Tp b)
{
    _Tp t = (_Tp)1/b;
    a.re *= t; a.im *= t; return a;
}



//////////////////////////////// 2D Point ///////////////////////////////

template<typename _Tp> inline
Point_<_Tp>::Point_()
    : x(0), y(0) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(_Tp _x, _Tp _y)
    : x(_x), y(_y) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(const Point_& pt)
    : x(pt.x), y(pt.y) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(const Size_<_Tp>& sz)
    : x(sz.width), y(sz.height) {}

template<typename _Tp> inline
Point_<_Tp>::Point_(const Vec<_Tp,2>& v)
    : x(v[0]), y(v[1]) {}

template<typename _Tp> inline
Point_<_Tp>& Point_<_Tp>::operator = (const Point_& pt)
{
    x = pt.x; y = pt.y;
    return *this;
}

template<typename _Tp> template<typename _Tp2> inline
Point_<_Tp>::operator Point_<_Tp2>() const
{
    return Point_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
}

template<typename _Tp> inline
Point_<_Tp>::operator Vec<_Tp, 2>() const
{
    return Vec<_Tp, 2>(x, y);
}

template<typename _Tp> inline
_Tp Point_<_Tp>::dot(const Point_& pt) const
{
    return saturate_cast<_Tp>(x*pt.x + y*pt.y);
}

template<typename _Tp> inline
double Point_<_Tp>::ddot(const Point_& pt) const
{
    return (double)x*pt.x + (double)y*pt.y;
}

template<typename _Tp> inline
double Point_<_Tp>::cross(const Point_& pt) const
{
    return (double)x*pt.y - (double)y*pt.x;
}

template<typename _Tp> inline bool
Point_<_Tp>::inside( const Rect_<_Tp>& r ) const
{
    return r.contains(*this);
}


template<typename _Tp> static inline
Point_<_Tp>& operator += (Point_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator -= (Point_<_Tp>& a, const Point_<_Tp>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator *= (Point_<_Tp>& a, int b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator *= (Point_<_Tp>& a, float b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator *= (Point_<_Tp>& a, double b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator /= (Point_<_Tp>& a, int b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator /= (Point_<_Tp>& a, float b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template<typename _Tp> static inline
Point_<_Tp>& operator /= (Point_<_Tp>& a, double b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template<typename _Tp> static inline
double norm(const Point_<_Tp>& pt)
{
    return std::sqrt((double)pt.x*pt.x + (double)pt.y*pt.y);
}

template<typename _Tp> static inline
bool operator == (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return a.x == b.x && a.y == b.y;
}

template<typename _Tp> static inline
bool operator != (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return a.x != b.x || a.y != b.y;
}

template<typename _Tp> static inline
Point_<_Tp> operator + (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x + b.x), saturate_cast<_Tp>(a.y + b.y) );
}

template<typename _Tp> static inline
Point_<_Tp> operator - (const Point_<_Tp>& a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x - b.x), saturate_cast<_Tp>(a.y - b.y) );
}

template<typename _Tp> static inline
Point_<_Tp> operator - (const Point_<_Tp>& a)
{
    return Point_<_Tp>( saturate_cast<_Tp>(-a.x), saturate_cast<_Tp>(-a.y) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Point_<_Tp>& a, int b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (int a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(b.x*a), saturate_cast<_Tp>(b.y*a) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Point_<_Tp>& a, float b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (float a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(b.x*a), saturate_cast<_Tp>(b.y*a) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Point_<_Tp>& a, double b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (double a, const Point_<_Tp>& b)
{
    return Point_<_Tp>( saturate_cast<_Tp>(b.x*a), saturate_cast<_Tp>(b.y*a) );
}

template<typename _Tp> static inline
Point_<_Tp> operator * (const Matx<_Tp, 2, 2>& a, const Point_<_Tp>& b)
{
    Matx<_Tp, 2, 1> tmp = a * Vec<_Tp,2>(b.x, b.y);
    return Point_<_Tp>(tmp.val[0], tmp.val[1]);
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (const Matx<_Tp, 3, 3>& a, const Point_<_Tp>& b)
{
    Matx<_Tp, 3, 1> tmp = a * Vec<_Tp,3>(b.x, b.y, 1);
    return Point3_<_Tp>(tmp.val[0], tmp.val[1], tmp.val[2]);
}

template<typename _Tp> static inline
Point_<_Tp> operator / (const Point_<_Tp>& a, int b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp> static inline
Point_<_Tp> operator / (const Point_<_Tp>& a, float b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp> static inline
Point_<_Tp> operator / (const Point_<_Tp>& a, double b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}



//////////////////////////////// 3D Point ///////////////////////////////

template<typename _Tp> inline
Point3_<_Tp>::Point3_()
    : x(0), y(0), z(0) {}

template<typename _Tp> inline
Point3_<_Tp>::Point3_(_Tp _x, _Tp _y, _Tp _z)
    : x(_x), y(_y), z(_z) {}

template<typename _Tp> inline
Point3_<_Tp>::Point3_(const Point3_& pt)
    : x(pt.x), y(pt.y), z(pt.z) {}

template<typename _Tp> inline
Point3_<_Tp>::Point3_(const Point_<_Tp>& pt)
    : x(pt.x), y(pt.y), z(_Tp()) {}

template<typename _Tp> inline
Point3_<_Tp>::Point3_(const Vec<_Tp, 3>& v)
    : x(v[0]), y(v[1]), z(v[2]) {}

template<typename _Tp> template<typename _Tp2> inline
Point3_<_Tp>::operator Point3_<_Tp2>() const
{
    return Point3_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y), saturate_cast<_Tp2>(z));
}

#if OPENCV_ABI_COMPATIBILITY > 300
template<typename _Tp> template<typename _Tp2> inline
Point3_<_Tp>::operator Vec<_Tp2, 3>() const
{
    return Vec<_Tp2, 3>(x, y, z);
}
#else
template<typename _Tp> inline
Point3_<_Tp>::operator Vec<_Tp, 3>() const
{
    return Vec<_Tp, 3>(x, y, z);
}
#endif

template<typename _Tp> inline
Point3_<_Tp>& Point3_<_Tp>::operator = (const Point3_& pt)
{
    x = pt.x; y = pt.y; z = pt.z;
    return *this;
}

template<typename _Tp> inline
_Tp Point3_<_Tp>::dot(const Point3_& pt) const
{
    return saturate_cast<_Tp>(x*pt.x + y*pt.y + z*pt.z);
}

template<typename _Tp> inline
double Point3_<_Tp>::ddot(const Point3_& pt) const
{
    return (double)x*pt.x + (double)y*pt.y + (double)z*pt.z;
}

template<typename _Tp> inline
Point3_<_Tp> Point3_<_Tp>::cross(const Point3_<_Tp>& pt) const
{
    return Point3_<_Tp>(y*pt.z - z*pt.y, z*pt.x - x*pt.z, x*pt.y - y*pt.x);
}


template<typename _Tp> static inline
Point3_<_Tp>& operator += (Point3_<_Tp>& a, const Point3_<_Tp>& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator -= (Point3_<_Tp>& a, const Point3_<_Tp>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator *= (Point3_<_Tp>& a, int b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    a.z = saturate_cast<_Tp>(a.z * b);
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator *= (Point3_<_Tp>& a, float b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    a.z = saturate_cast<_Tp>(a.z * b);
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator *= (Point3_<_Tp>& a, double b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    a.z = saturate_cast<_Tp>(a.z * b);
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator /= (Point3_<_Tp>& a, int b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    a.z = saturate_cast<_Tp>(a.z / b);
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator /= (Point3_<_Tp>& a, float b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    a.z = saturate_cast<_Tp>(a.z / b);
    return a;
}

template<typename _Tp> static inline
Point3_<_Tp>& operator /= (Point3_<_Tp>& a, double b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    a.z = saturate_cast<_Tp>(a.z / b);
    return a;
}

template<typename _Tp> static inline
double norm(const Point3_<_Tp>& pt)
{
    return std::sqrt((double)pt.x*pt.x + (double)pt.y*pt.y + (double)pt.z*pt.z);
}

template<typename _Tp> static inline
bool operator == (const Point3_<_Tp>& a, const Point3_<_Tp>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

template<typename _Tp> static inline
bool operator != (const Point3_<_Tp>& a, const Point3_<_Tp>& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

template<typename _Tp> static inline
Point3_<_Tp> operator + (const Point3_<_Tp>& a, const Point3_<_Tp>& b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(a.x + b.x), saturate_cast<_Tp>(a.y + b.y), saturate_cast<_Tp>(a.z + b.z));
}

template<typename _Tp> static inline
Point3_<_Tp> operator - (const Point3_<_Tp>& a, const Point3_<_Tp>& b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(a.x - b.x), saturate_cast<_Tp>(a.y - b.y), saturate_cast<_Tp>(a.z - b.z));
}

template<typename _Tp> static inline
Point3_<_Tp> operator - (const Point3_<_Tp>& a)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(-a.x), saturate_cast<_Tp>(-a.y), saturate_cast<_Tp>(-a.z) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (const Point3_<_Tp>& a, int b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(a.x*b), saturate_cast<_Tp>(a.y*b), saturate_cast<_Tp>(a.z*b) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (int a, const Point3_<_Tp>& b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(b.x * a), saturate_cast<_Tp>(b.y * a), saturate_cast<_Tp>(b.z * a) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (const Point3_<_Tp>& a, float b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(a.x * b), saturate_cast<_Tp>(a.y * b), saturate_cast<_Tp>(a.z * b) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (float a, const Point3_<_Tp>& b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(b.x * a), saturate_cast<_Tp>(b.y * a), saturate_cast<_Tp>(b.z * a) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (const Point3_<_Tp>& a, double b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(a.x * b), saturate_cast<_Tp>(a.y * b), saturate_cast<_Tp>(a.z * b) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (double a, const Point3_<_Tp>& b)
{
    return Point3_<_Tp>( saturate_cast<_Tp>(b.x * a), saturate_cast<_Tp>(b.y * a), saturate_cast<_Tp>(b.z * a) );
}

template<typename _Tp> static inline
Point3_<_Tp> operator * (const Matx<_Tp, 3, 3>& a, const Point3_<_Tp>& b)
{
    Matx<_Tp, 3, 1> tmp = a * Vec<_Tp,3>(b.x, b.y, b.z);
    return Point3_<_Tp>(tmp.val[0], tmp.val[1], tmp.val[2]);
}

template<typename _Tp> static inline
Matx<_Tp, 4, 1> operator * (const Matx<_Tp, 4, 4>& a, const Point3_<_Tp>& b)
{
    return a * Matx<_Tp, 4, 1>(b.x, b.y, b.z, 1);
}

template<typename _Tp> static inline
Point3_<_Tp> operator / (const Point3_<_Tp>& a, int b)
{
    Point3_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp> static inline
Point3_<_Tp> operator / (const Point3_<_Tp>& a, float b)
{
    Point3_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp> static inline
Point3_<_Tp> operator / (const Point3_<_Tp>& a, double b)
{
    Point3_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}



////////////////////////////////// Size /////////////////////////////////

template<typename _Tp> inline
Size_<_Tp>::Size_()
    : width(0), height(0) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(_Tp _width, _Tp _height)
    : width(_width), height(_height) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(const Size_& sz)
    : width(sz.width), height(sz.height) {}

template<typename _Tp> inline
Size_<_Tp>::Size_(const Point_<_Tp>& pt)
    : width(pt.x), height(pt.y) {}

template<typename _Tp> template<typename _Tp2> inline
Size_<_Tp>::operator Size_<_Tp2>() const
{
    return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
}

template<typename _Tp> inline
Size_<_Tp>& Size_<_Tp>::operator = (const Size_<_Tp>& sz)
{
    width = sz.width; height = sz.height;
    return *this;
}

template<typename _Tp> inline
_Tp Size_<_Tp>::area() const
{
    const _Tp result = width * height;
    CV_DbgAssert(!std::numeric_limits<_Tp>::is_integer
        || width == 0 || result / width == height); // make sure the result fits in the return value
    return result;
}

template<typename _Tp> inline
bool Size_<_Tp>::empty() const
{
    return width <= 0 || height <= 0;
}


template<typename _Tp> static inline
Size_<_Tp>& operator *= (Size_<_Tp>& a, _Tp b)
{
    a.width *= b;
    a.height *= b;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator * (const Size_<_Tp>& a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp *= b;
    return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator /= (Size_<_Tp>& a, _Tp b)
{
    a.width /= b;
    a.height /= b;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator / (const Size_<_Tp>& a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator += (Size_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator + (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    Size_<_Tp> tmp(a);
    tmp += b;
    return tmp;
}

template<typename _Tp> static inline
Size_<_Tp>& operator -= (Size_<_Tp>& a, const Size_<_Tp>& b)
{
    a.width -= b.width;
    a.height -= b.height;
    return a;
}

template<typename _Tp> static inline
Size_<_Tp> operator - (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    Size_<_Tp> tmp(a);
    tmp -= b;
    return tmp;
}

template<typename _Tp> static inline
bool operator == (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    return a.width == b.width && a.height == b.height;
}

template<typename _Tp> static inline
bool operator != (const Size_<_Tp>& a, const Size_<_Tp>& b)
{
    return !(a == b);
}



////////////////////////////////// Rect /////////////////////////////////

template<typename _Tp> inline
Rect_<_Tp>::Rect_()
    : x(0), y(0), width(0), height(0) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
    : x(_x), y(_y), width(_width), height(_height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Rect_<_Tp>& r)
    : x(r.x), y(r.y), width(r.width), height(r.height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz)
    : x(org.x), y(org.y), width(sz.width), height(sz.height) {}

template<typename _Tp> inline
Rect_<_Tp>::Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2)
{
    x = std::min(pt1.x, pt2.x);
    y = std::min(pt1.y, pt2.y);
    width = std::max(pt1.x, pt2.x) - x;
    height = std::max(pt1.y, pt2.y) - y;
}

template<typename _Tp> inline
Rect_<_Tp>& Rect_<_Tp>::operator = ( const Rect_<_Tp>& r )
{
    x = r.x;
    y = r.y;
    width = r.width;
    height = r.height;
    return *this;
}

template<typename _Tp> inline
Point_<_Tp> Rect_<_Tp>::tl() const
{
    return Point_<_Tp>(x,y);
}

template<typename _Tp> inline
Point_<_Tp> Rect_<_Tp>::br() const
{
    return Point_<_Tp>(x + width, y + height);
}

template<typename _Tp> inline
Size_<_Tp> Rect_<_Tp>::size() const
{
    return Size_<_Tp>(width, height);
}

template<typename _Tp> inline
_Tp Rect_<_Tp>::area() const
{
    const _Tp result = width * height;
    CV_DbgAssert(!std::numeric_limits<_Tp>::is_integer
        || width == 0 || result / width == height); // make sure the result fits in the return value
    return result;
}

template<typename _Tp> inline
bool Rect_<_Tp>::empty() const
{
    return width <= 0 || height <= 0;
}

template<typename _Tp> template<typename _Tp2> inline
Rect_<_Tp>::operator Rect_<_Tp2>() const
{
    return Rect_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y), saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
}

template<typename _Tp> inline
bool Rect_<_Tp>::contains(const Point_<_Tp>& pt) const
{
    return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
}


template<typename _Tp> static inline
Rect_<_Tp>& operator += ( Rect_<_Tp>& a, const Point_<_Tp>& b )
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator -= ( Rect_<_Tp>& a, const Point_<_Tp>& b )
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator += ( Rect_<_Tp>& a, const Size_<_Tp>& b )
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator -= ( Rect_<_Tp>& a, const Size_<_Tp>& b )
{
    a.width -= b.width;
    a.height -= b.height;
    return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator &= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
{
    _Tp x1 = std::max(a.x, b.x);
    _Tp y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if( a.width <= 0 || a.height <= 0 )
        a = Rect();
    return a;
}

template<typename _Tp> static inline
Rect_<_Tp>& operator |= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
{
    if (a.empty()) {
        a = b;
    }
    else if (!b.empty()) {
        _Tp x1 = std::min(a.x, b.x);
        _Tp y1 = std::min(a.y, b.y);
        a.width = std::max(a.x + a.width, b.x + b.width) - x1;
        a.height = std::max(a.y + a.height, b.y + b.height) - y1;
        a.x = x1;
        a.y = y1;
    }
    return a;
}

template<typename _Tp> static inline
bool operator == (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template<typename _Tp> static inline
bool operator != (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

template<typename _Tp> static inline
Rect_<_Tp> operator + (const Rect_<_Tp>& a, const Point_<_Tp>& b)
{
    return Rect_<_Tp>( a.x + b.x, a.y + b.y, a.width, a.height );
}

template<typename _Tp> static inline
Rect_<_Tp> operator - (const Rect_<_Tp>& a, const Point_<_Tp>& b)
{
    return Rect_<_Tp>( a.x - b.x, a.y - b.y, a.width, a.height );
}

template<typename _Tp> static inline
Rect_<_Tp> operator + (const Rect_<_Tp>& a, const Size_<_Tp>& b)
{
    return Rect_<_Tp>( a.x, a.y, a.width + b.width, a.height + b.height );
}

template<typename _Tp> static inline
Rect_<_Tp> operator & (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

template<typename _Tp> static inline
Rect_<_Tp> operator | (const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c |= b;
}

/**
 * @brief measure dissimilarity between two sample sets
 *
 * computes the complement of the Jaccard Index as described in <https://en.wikipedia.org/wiki/Jaccard_index>.
 * For rectangles this reduces to computing the intersection over the union.
 */
template<typename _Tp> static inline
double jaccardDistance(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

////////////////////////////// RotatedRect //////////////////////////////

inline
RotatedRect::RotatedRect()
    : center(), size(), angle(0) {}

inline
RotatedRect::RotatedRect(const Point2f& _center, const Size2f& _size, float _angle)
    : center(_center), size(_size), angle(_angle) {}



///////////////////////////////// Range /////////////////////////////////

inline
Range::Range()
    : start(0), end(0) {}

inline
Range::Range(int _start, int _end)
    : start(_start), end(_end) {}

inline
int Range::size() const
{
    return end - start;
}

inline
bool Range::empty() const
{
    return start == end;
}

inline
Range Range::all()
{
    return Range(INT_MIN, INT_MAX);
}


static inline
bool operator == (const Range& r1, const Range& r2)
{
    return r1.start == r2.start && r1.end == r2.end;
}

static inline
bool operator != (const Range& r1, const Range& r2)
{
    return !(r1 == r2);
}

static inline
bool operator !(const Range& r)
{
    return r.start == r.end;
}

static inline
Range operator & (const Range& r1, const Range& r2)
{
    Range r(std::max(r1.start, r2.start), std::min(r1.end, r2.end));
    r.end = std::max(r.end, r.start);
    return r;
}

static inline
Range& operator &= (Range& r1, const Range& r2)
{
    r1 = r1 & r2;
    return r1;
}

static inline
Range operator + (const Range& r1, int delta)
{
    return Range(r1.start + delta, r1.end + delta);
}

static inline
Range operator + (int delta, const Range& r1)
{
    return Range(r1.start + delta, r1.end + delta);
}

static inline
Range operator - (const Range& r1, int delta)
{
    return r1 + (-delta);
}



///////////////////////////////// Scalar ////////////////////////////////

template<typename _Tp> inline
Scalar_<_Tp>::Scalar_()
{
    this->val[0] = this->val[1] = this->val[2] = this->val[3] = 0;
}

template<typename _Tp> inline
Scalar_<_Tp>::Scalar_(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
{
    this->val[0] = v0;
    this->val[1] = v1;
    this->val[2] = v2;
    this->val[3] = v3;
}

template<typename _Tp> template<typename _Tp2, int cn> inline
Scalar_<_Tp>::Scalar_(const Vec<_Tp2, cn>& v)
{
    int i;
    for( i = 0; i < (cn < 4 ? cn : 4); i++ )
        this->val[i] = cv::saturate_cast<_Tp>(v.val[i]);
    for( ; i < 4; i++ )
        this->val[i] = 0;
}

template<typename _Tp> inline
Scalar_<_Tp>::Scalar_(_Tp v0)
{
    this->val[0] = v0;
    this->val[1] = this->val[2] = this->val[3] = 0;
}

template<typename _Tp> inline
Scalar_<_Tp> Scalar_<_Tp>::all(_Tp v0)
{
    return Scalar_<_Tp>(v0, v0, v0, v0);
}


template<typename _Tp> inline
Scalar_<_Tp> Scalar_<_Tp>::mul(const Scalar_<_Tp>& a, double scale ) const
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(this->val[0] * a.val[0] * scale),
                        saturate_cast<_Tp>(this->val[1] * a.val[1] * scale),
                        saturate_cast<_Tp>(this->val[2] * a.val[2] * scale),
                        saturate_cast<_Tp>(this->val[3] * a.val[3] * scale));
}

template<typename _Tp> inline
Scalar_<_Tp> Scalar_<_Tp>::conj() const
{
    return Scalar_<_Tp>(saturate_cast<_Tp>( this->val[0]),
                        saturate_cast<_Tp>(-this->val[1]),
                        saturate_cast<_Tp>(-this->val[2]),
                        saturate_cast<_Tp>(-this->val[3]));
}

template<typename _Tp> inline
bool Scalar_<_Tp>::isReal() const
{
    return this->val[1] == 0 && this->val[2] == 0 && this->val[3] == 0;
}


template<typename _Tp> template<typename T2> inline
Scalar_<_Tp>::operator Scalar_<T2>() const
{
    return Scalar_<T2>(saturate_cast<T2>(this->val[0]),
                       saturate_cast<T2>(this->val[1]),
                       saturate_cast<T2>(this->val[2]),
                       saturate_cast<T2>(this->val[3]));
}


template<typename _Tp> static inline
Scalar_<_Tp>& operator += (Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    a.val[0] += b.val[0];
    a.val[1] += b.val[1];
    a.val[2] += b.val[2];
    a.val[3] += b.val[3];
    return a;
}

template<typename _Tp> static inline
Scalar_<_Tp>& operator -= (Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    a.val[0] -= b.val[0];
    a.val[1] -= b.val[1];
    a.val[2] -= b.val[2];
    a.val[3] -= b.val[3];
    return a;
}

template<typename _Tp> static inline
Scalar_<_Tp>& operator *= ( Scalar_<_Tp>& a, _Tp v )
{
    a.val[0] *= v;
    a.val[1] *= v;
    a.val[2] *= v;
    a.val[3] *= v;
    return a;
}

template<typename _Tp> static inline
bool operator == ( const Scalar_<_Tp>& a, const Scalar_<_Tp>& b )
{
    return a.val[0] == b.val[0] && a.val[1] == b.val[1] &&
           a.val[2] == b.val[2] && a.val[3] == b.val[3];
}

template<typename _Tp> static inline
bool operator != ( const Scalar_<_Tp>& a, const Scalar_<_Tp>& b )
{
    return a.val[0] != b.val[0] || a.val[1] != b.val[1] ||
           a.val[2] != b.val[2] || a.val[3] != b.val[3];
}

template<typename _Tp> static inline
Scalar_<_Tp> operator + (const Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    return Scalar_<_Tp>(a.val[0] + b.val[0],
                        a.val[1] + b.val[1],
                        a.val[2] + b.val[2],
                        a.val[3] + b.val[3]);
}

template<typename _Tp> static inline
Scalar_<_Tp> operator - (const Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(a.val[0] - b.val[0]),
                        saturate_cast<_Tp>(a.val[1] - b.val[1]),
                        saturate_cast<_Tp>(a.val[2] - b.val[2]),
                        saturate_cast<_Tp>(a.val[3] - b.val[3]));
}

template<typename _Tp> static inline
Scalar_<_Tp> operator * (const Scalar_<_Tp>& a, _Tp alpha)
{
    return Scalar_<_Tp>(a.val[0] * alpha,
                        a.val[1] * alpha,
                        a.val[2] * alpha,
                        a.val[3] * alpha);
}

template<typename _Tp> static inline
Scalar_<_Tp> operator * (_Tp alpha, const Scalar_<_Tp>& a)
{
    return a*alpha;
}

template<typename _Tp> static inline
Scalar_<_Tp> operator - (const Scalar_<_Tp>& a)
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(-a.val[0]),
                        saturate_cast<_Tp>(-a.val[1]),
                        saturate_cast<_Tp>(-a.val[2]),
                        saturate_cast<_Tp>(-a.val[3]));
}


template<typename _Tp> static inline
Scalar_<_Tp> operator * (const Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]),
                        saturate_cast<_Tp>(a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]),
                        saturate_cast<_Tp>(a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]),
                        saturate_cast<_Tp>(a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]));
}

template<typename _Tp> static inline
Scalar_<_Tp>& operator *= (Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    a = a * b;
    return a;
}

template<typename _Tp> static inline
Scalar_<_Tp> operator / (const Scalar_<_Tp>& a, _Tp alpha)
{
    return Scalar_<_Tp>(a.val[0] / alpha,
                        a.val[1] / alpha,
                        a.val[2] / alpha,
                        a.val[3] / alpha);
}

template<typename _Tp> static inline
Scalar_<float> operator / (const Scalar_<float>& a, float alpha)
{
    float s = 1 / alpha;
    return Scalar_<float>(a.val[0] * s, a.val[1] * s, a.val[2] * s, a.val[3] * s);
}

template<typename _Tp> static inline
Scalar_<double> operator / (const Scalar_<double>& a, double alpha)
{
    double s = 1 / alpha;
    return Scalar_<double>(a.val[0] * s, a.val[1] * s, a.val[2] * s, a.val[3] * s);
}

template<typename _Tp> static inline
Scalar_<_Tp>& operator /= (Scalar_<_Tp>& a, _Tp alpha)
{
    a = a / alpha;
    return a;
}

template<typename _Tp> static inline
Scalar_<_Tp> operator / (_Tp a, const Scalar_<_Tp>& b)
{
    _Tp s = a / (b[0]*b[0] + b[1]*b[1] + b[2]*b[2] + b[3]*b[3]);
    return b.conj() * s;
}

template<typename _Tp> static inline
Scalar_<_Tp> operator / (const Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    return a * ((_Tp)1 / b);
}

template<typename _Tp> static inline
Scalar_<_Tp>& operator /= (Scalar_<_Tp>& a, const Scalar_<_Tp>& b)
{
    a = a / b;
    return a;
}

template<typename _Tp> static inline
Scalar operator * (const Matx<_Tp, 4, 4>& a, const Scalar& b)
{
    Matx<double, 4, 1> c((Matx<double, 4, 4>)a, b, Matx_MatMulOp());
    return reinterpret_cast<const Scalar&>(c);
}

template<> inline
Scalar operator * (const Matx<double, 4, 4>& a, const Scalar& b)
{
    Matx<double, 4, 1> c(a, b, Matx_MatMulOp());
    return reinterpret_cast<const Scalar&>(c);
}



//////////////////////////////// KeyPoint ///////////////////////////////

inline
KeyPoint::KeyPoint()
    : pt(0,0), size(0), angle(-1), response(0), octave(0), class_id(-1) {}

inline
KeyPoint::KeyPoint(Point2f _pt, float _size, float _angle, float _response, int _octave, int _class_id)
    : pt(_pt), size(_size), angle(_angle), response(_response), octave(_octave), class_id(_class_id) {}

inline
KeyPoint::KeyPoint(float x, float y, float _size, float _angle, float _response, int _octave, int _class_id)
    : pt(x, y), size(_size), angle(_angle), response(_response), octave(_octave), class_id(_class_id) {}



///////////////////////////////// DMatch ////////////////////////////////

inline
DMatch::DMatch()
    : queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(FLT_MAX) {}

inline
DMatch::DMatch(int _queryIdx, int _trainIdx, float _distance)
    : queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}

inline
DMatch::DMatch(int _queryIdx, int _trainIdx, int _imgIdx, float _distance)
    : queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}

inline
bool DMatch::operator < (const DMatch &m) const
{
    return distance < m.distance;
}



////////////////////////////// TermCriteria /////////////////////////////

inline
TermCriteria::TermCriteria()
    : type(0), maxCount(0), epsilon(0) {}

inline
TermCriteria::TermCriteria(int _type, int _maxCount, double _epsilon)
    : type(_type), maxCount(_maxCount), epsilon(_epsilon) {}

//! @endcond

} // cv

#endif //OPENCV_CORE_TYPES_HPP
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_UTILITY_H
#define OPENCV_CORE_UTILITY_H

#ifndef __cplusplus
#  error utility.hpp header must be compiled as C++
#endif

#if defined(check)
#  warning Detected Apple 'check' macro definition, it can cause build conflicts. Please, include this header before any Apple headers.
#endif

#include "opencv2/core.hpp"
#include <ostream>

#ifdef CV_CXX11
#include <functional>
#endif

namespace cv
{

#ifdef CV_COLLECT_IMPL_DATA
CV_EXPORTS void setImpl(int flags); // set implementation flags and reset storage arrays
CV_EXPORTS void addImpl(int flag, const char* func = 0); // add implementation and function name to storage arrays
// Get stored implementation flags and functions names arrays
// Each implementation entry correspond to function name entry, so you can find which implementation was executed in which function
CV_EXPORTS int getImpl(std::vector<int> &impl, std::vector<String> &funName);

CV_EXPORTS bool useCollection(); // return implementation collection state
CV_EXPORTS void setUseCollection(bool flag); // set implementation collection state

#define CV_IMPL_PLAIN  0x01 // native CPU OpenCV implementation
#define CV_IMPL_OCL    0x02 // OpenCL implementation
#define CV_IMPL_IPP    0x04 // IPP implementation
#define CV_IMPL_MT     0x10 // multithreaded implementation

#define CV_IMPL_ADD(impl)                                                   \
    if(cv::useCollection())                                                 \
    {                                                                       \
        cv::addImpl(impl, CV_Func);                                         \
    }
#else
#define CV_IMPL_ADD(impl)
#endif

//! @addtogroup core_utils
//! @{

/** @brief  Automatically Allocated Buffer Class

 The class is used for temporary buffers in functions and methods.
 If a temporary buffer is usually small (a few K's of memory),
 but its size depends on the parameters, it makes sense to create a small
 fixed-size array on stack and use it if it's large enough. If the required buffer size
 is larger than the fixed size, another buffer of sufficient size is allocated dynamically
 and released after the processing. Therefore, in typical cases, when the buffer size is small,
 there is no overhead associated with malloc()/free().
 At the same time, there is no limit on the size of processed data.

 This is what AutoBuffer does. The template takes 2 parameters - type of the buffer elements and
 the number of stack-allocated elements. Here is how the class is used:

 \code
 void my_func(const cv::Mat& m)
 {
    cv::AutoBuffer<float> buf(1000); // create automatic buffer containing 1000 floats

    buf.allocate(m.rows); // if m.rows <= 1000, the pre-allocated buffer is used,
                          // otherwise the buffer of "m.rows" floats will be allocated
                          // dynamically and deallocated in cv::AutoBuffer destructor
    ...
 }
 \endcode
*/
template<typename _Tp, size_t fixed_size = 1024/sizeof(_Tp)+8> class AutoBuffer
{
public:
    typedef _Tp value_type;

    //! the default constructor
    AutoBuffer();
    //! constructor taking the real buffer size
    AutoBuffer(size_t _size);

    //! the copy constructor
    AutoBuffer(const AutoBuffer<_Tp, fixed_size>& buf);
    //! the assignment operator
    AutoBuffer<_Tp, fixed_size>& operator = (const AutoBuffer<_Tp, fixed_size>& buf);

    //! destructor. calls deallocate()
    ~AutoBuffer();

    //! allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
    void allocate(size_t _size);
    //! deallocates the buffer if it was dynamically allocated
    void deallocate();
    //! resizes the buffer and preserves the content
    void resize(size_t _size);
    //! returns the current buffer size
    size_t size() const;
    //! returns pointer to the real buffer, stack-allocated or heap-allocated
    operator _Tp* ();
    //! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    operator const _Tp* () const;

protected:
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    //! size of the real buffer
    size_t sz;
    //! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
    _Tp buf[(fixed_size > 0) ? fixed_size : 1];
};

/**  @brief Sets/resets the break-on-error mode.

When the break-on-error mode is set, the default error handler issues a hardware exception, which
can make debugging more convenient.

\return the previous state
 */
CV_EXPORTS bool setBreakOnError(bool flag);

extern "C" typedef int (*ErrorCallback)( int status, const char* func_name,
                                       const char* err_msg, const char* file_name,
                                       int line, void* userdata );


/** @brief Sets the new error handler and the optional user data.

  The function sets the new error handler, called from cv::error().

  \param errCallback the new error handler. If NULL, the default error handler is used.
  \param userdata the optional user data pointer, passed to the callback.
  \param prevUserdata the optional output parameter where the previous user data pointer is stored

  \return the previous error handler
*/
CV_EXPORTS ErrorCallback redirectError( ErrorCallback errCallback, void* userdata=0, void** prevUserdata=0);

/** @brief Returns a text string formatted using the printf-like expression.

The function acts like sprintf but forms and returns an STL string. It can be used to form an error
message in the Exception constructor.
@param fmt printf-compatible formatting specifiers.
 */
CV_EXPORTS String format( const char* fmt, ... );
CV_EXPORTS String tempfile( const char* suffix = 0);
CV_EXPORTS void glob(String pattern, std::vector<String>& result, bool recursive = false);

/** @brief OpenCV will try to set the number of threads for the next parallel region.

If threads == 0, OpenCV will disable threading optimizations and run all it's functions
sequentially. Passing threads \< 0 will reset threads number to system default. This function must
be called outside of parallel region.

OpenCV will try to run it's functions with specified threads number, but some behaviour differs from
framework:
-   `TBB` - User-defined parallel constructions will run with the same threads number, if
    another does not specified. If later on user creates own scheduler, OpenCV will use it.
-   `OpenMP` - No special defined behaviour.
-   `Concurrency` - If threads == 1, OpenCV will disable threading optimizations and run it's
    functions sequentially.
-   `GCD` - Supports only values \<= 0.
-   `C=` - No special defined behaviour.
@param nthreads Number of threads used by OpenCV.
@sa getNumThreads, getThreadNum
 */
CV_EXPORTS_W void setNumThreads(int nthreads);

/** @brief Returns the number of threads used by OpenCV for parallel regions.

Always returns 1 if OpenCV is built without threading support.

The exact meaning of return value depends on the threading framework used by OpenCV library:
- `TBB` - The number of threads, that OpenCV will try to use for parallel regions. If there is
  any tbb::thread_scheduler_init in user code conflicting with OpenCV, then function returns
  default number of threads used by TBB library.
- `OpenMP` - An upper bound on the number of threads that could be used to form a new team.
- `Concurrency` - The number of threads, that OpenCV will try to use for parallel regions.
- `GCD` - Unsupported; returns the GCD thread pool limit (512) for compatibility.
- `C=` - The number of threads, that OpenCV will try to use for parallel regions, if before
  called setNumThreads with threads \> 0, otherwise returns the number of logical CPUs,
  available for the process.
@sa setNumThreads, getThreadNum
 */
CV_EXPORTS_W int getNumThreads();

/** @brief Returns the index of the currently executed thread within the current parallel region. Always
returns 0 if called outside of parallel region.

The exact meaning of return value depends on the threading framework used by OpenCV library:
- `TBB` - Unsupported with current 4.1 TBB release. Maybe will be supported in future.
- `OpenMP` - The thread number, within the current team, of the calling thread.
- `Concurrency` - An ID for the virtual processor that the current context is executing on (0
  for master thread and unique number for others, but not necessary 1,2,3,...).
- `GCD` - System calling thread's ID. Never returns 0 inside parallel region.
- `C=` - The index of the current parallel task.
@sa setNumThreads, getNumThreads
 */
CV_EXPORTS_W int getThreadNum();

/** @brief Returns full configuration time cmake output.

Returned value is raw cmake output including version control system revision, compiler version,
compiler flags, enabled modules and third party libraries, etc. Output format depends on target
architecture.
 */
CV_EXPORTS_W const String& getBuildInformation();

/** @brief Returns the number of ticks.

The function returns the number of ticks after the certain event (for example, when the machine was
turned on). It can be used to initialize RNG or to measure a function execution time by reading the
tick count before and after the function call.
@sa getTickFrequency, TickMeter
 */
CV_EXPORTS_W int64 getTickCount();

/** @brief Returns the number of ticks per second.

The function returns the number of ticks per second. That is, the following code computes the
execution time in seconds:
@code
    double t = (double)getTickCount();
    // do something ...
    t = ((double)getTickCount() - t)/getTickFrequency();
@endcode
@sa getTickCount, TickMeter
 */
CV_EXPORTS_W double getTickFrequency();

/** @brief a Class to measure passing time.

The class computes passing time by counting the number of ticks per second. That is, the following code computes the
execution time in seconds:
@code
TickMeter tm;
tm.start();
// do something ...
tm.stop();
std::cout << tm.getTimeSec();
@endcode
@sa getTickCount, getTickFrequency
*/

class CV_EXPORTS_W TickMeter
{
public:
    //! the default constructor
    CV_WRAP TickMeter()
    {
    reset();
    }

    /**
    starts counting ticks.
    */
    CV_WRAP void start()
    {
    startTime = cv::getTickCount();
    }

    /**
    stops counting ticks.
    */
    CV_WRAP void stop()
    {
    int64 time = cv::getTickCount();
    if (startTime == 0)
    return;
    ++counter;
    sumTime += (time - startTime);
    startTime = 0;
    }

    /**
    returns counted ticks.
    */
    CV_WRAP int64 getTimeTicks() const
    {
    return sumTime;
    }

    /**
    returns passed time in microseconds.
    */
    CV_WRAP double getTimeMicro() const
    {
    return getTimeMilli()*1e3;
    }

    /**
    returns passed time in milliseconds.
    */
    CV_WRAP double getTimeMilli() const
    {
    return getTimeSec()*1e3;
    }

    /**
    returns passed time in seconds.
    */
    CV_WRAP double getTimeSec()   const
    {
    return (double)getTimeTicks() / getTickFrequency();
    }

    /**
    returns internal counter value.
    */
    CV_WRAP int64 getCounter() const
    {
    return counter;
    }

    /**
    resets internal values.
    */
    CV_WRAP void reset()
    {
    startTime = 0;
    sumTime = 0;
    counter = 0;
    }

private:
    int64 counter;
    int64 sumTime;
    int64 startTime;
};

/** @brief output operator
@code
TickMeter tm;
tm.start();
// do something ...
tm.stop();
std::cout << tm;
@endcode
*/

static inline
std::ostream& operator << (std::ostream& out, const TickMeter& tm)
{
    return out << tm.getTimeSec() << "sec";
}

/** @brief Returns the number of CPU ticks.

The function returns the current number of CPU ticks on some architectures (such as x86, x64,
PowerPC). On other platforms the function is equivalent to getTickCount. It can also be used for
very accurate time measurements, as well as for RNG initialization. Note that in case of multi-CPU
systems a thread, from which getCPUTickCount is called, can be suspended and resumed at another CPU
with its own counter. So, theoretically (and practically) the subsequent calls to the function do
not necessary return the monotonously increasing values. Also, since a modern CPU varies the CPU
frequency depending on the load, the number of CPU clocks spent in some code cannot be directly
converted to time units. Therefore, getTickCount is generally a preferable solution for measuring
execution time.
 */
CV_EXPORTS_W int64 getCPUTickCount();

/** @brief Returns true if the specified feature is supported by the host hardware.

The function returns true if the host hardware supports the specified feature. When user calls
setUseOptimized(false), the subsequent calls to checkHardwareSupport() will return false until
setUseOptimized(true) is called. This way user can dynamically switch on and off the optimized code
in OpenCV.
@param feature The feature of interest, one of cv::CpuFeatures
 */
CV_EXPORTS_W bool checkHardwareSupport(int feature);

/** @brief Returns the number of logical CPUs available for the process.
 */
CV_EXPORTS_W int getNumberOfCPUs();


/** @brief Aligns a pointer to the specified number of bytes.

The function returns the aligned pointer of the same type as the input pointer:
\f[\texttt{(_Tp*)(((size_t)ptr + n-1) & -n)}\f]
@param ptr Aligned pointer.
@param n Alignment size that must be a power of two.
 */
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

/** @brief Aligns a buffer size to the specified number of bytes.

The function returns the minimum number that is greater or equal to sz and is divisible by n :
\f[\texttt{(sz + n-1) & -n}\f]
@param sz Buffer size to align.
@param n Alignment size that must be a power of two.
 */
static inline size_t alignSize(size_t sz, int n)
{
    CV_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n-1) & -n;
}

/** @brief Integer division with result round up.

Use this function instead of `ceil((float)a / b)` expressions.

@sa alignSize
*/
static inline int divUp(int a, unsigned int b)
{
    CV_DbgAssert(a >= 0);
    return (a + b - 1) / b;
}
/** @overload */
static inline size_t divUp(size_t a, unsigned int b)
{
    return (a + b - 1) / b;
}

/** @brief Enables or disables the optimized code.

The function can be used to dynamically turn on and off optimized code (code that uses SSE2, AVX,
and other instructions on the platforms that support it). It sets a global flag that is further
checked by OpenCV functions. Since the flag is not checked in the inner OpenCV loops, it is only
safe to call the function on the very top level in your application where you can be sure that no
other OpenCV function is currently executed.

By default, the optimized code is enabled unless you disable it in CMake. The current status can be
retrieved using useOptimized.
@param onoff The boolean flag specifying whether the optimized code should be used (onoff=true)
or not (onoff=false).
 */
CV_EXPORTS_W void setUseOptimized(bool onoff);

/** @brief Returns the status of optimized code usage.

The function returns true if the optimized code is enabled. Otherwise, it returns false.
 */
CV_EXPORTS_W bool useOptimized();

static inline size_t getElemSize(int type) { return CV_ELEM_SIZE(type); }

/////////////////////////////// Parallel Primitives //////////////////////////////////

/** @brief Base class for parallel data processors
*/
class CV_EXPORTS ParallelLoopBody
{
public:
    virtual ~ParallelLoopBody();
    virtual void operator() (const Range& range) const = 0;
};

/** @brief Parallel data processor
*/
CV_EXPORTS void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes=-1.);

#ifdef CV_CXX11
class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
{
private:
    std::function<void(const Range&)> m_functor;
public:
    ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor) :
        m_functor(functor)
    { }

    virtual void operator() (const cv::Range& range) const
    {
        m_functor(range);
    }
};

inline void parallel_for_(const Range& range, std::function<void(const Range&)> functor, double nstripes=-1.)
{
    parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
}
#endif

/////////////////////////////// forEach method of cv::Mat ////////////////////////////
template<typename _Tp, typename Functor> inline
void Mat::forEach_impl(const Functor& operation) {
    if (false) {
        operation(*reinterpret_cast<_Tp*>(0), reinterpret_cast<int*>(0));
        // If your compiler fail in this line.
        // Please check that your functor signature is
        //     (_Tp&, const int*)   <- multidimential
        //  or (_Tp&, void*)        <- in case of you don't need current idx.
    }

    CV_Assert(this->total() / this->size[this->dims - 1] <= INT_MAX);
    const int LINES = static_cast<int>(this->total() / this->size[this->dims - 1]);

    class PixelOperationWrapper :public ParallelLoopBody
    {
    public:
        PixelOperationWrapper(Mat_<_Tp>* const frame, const Functor& _operation)
            : mat(frame), op(_operation) {}
        virtual ~PixelOperationWrapper(){}
        // ! Overloaded virtual operator
        // convert range call to row call.
        virtual void operator()(const Range &range) const {
            const int DIMS = mat->dims;
            const int COLS = mat->size[DIMS - 1];
            if (DIMS <= 2) {
                for (int row = range.start; row < range.end; ++row) {
                    this->rowCall2(row, COLS);
                }
            } else {
                std::vector<int> idx(DIMS); /// idx is modified in this->rowCall
                idx[DIMS - 2] = range.start - 1;

                for (int line_num = range.start; line_num < range.end; ++line_num) {
                    idx[DIMS - 2]++;
                    for (int i = DIMS - 2; i >= 0; --i) {
                        if (idx[i] >= mat->size[i]) {
                            idx[i - 1] += idx[i] / mat->size[i];
                            idx[i] %= mat->size[i];
                            continue; // carry-over;
                        }
                        else {
                            break;
                        }
                    }
                    this->rowCall(&idx[0], COLS, DIMS);
                }
            }
        }
    private:
        Mat_<_Tp>* const mat;
        const Functor op;
        // ! Call operator for each elements in this row.
        inline void rowCall(int* const idx, const int COLS, const int DIMS) const {
            int &col = idx[DIMS - 1];
            col = 0;
            _Tp* pixel = &(mat->template at<_Tp>(idx));

            while (col < COLS) {
                op(*pixel, const_cast<const int*>(idx));
                pixel++; col++;
            }
            col = 0;
        }
        // ! Call operator for each elements in this row. 2d mat special version.
        inline void rowCall2(const int row, const int COLS) const {
            union Index{
                int body[2];
                operator const int*() const {
                    return reinterpret_cast<const int*>(this);
                }
                int& operator[](const int i) {
                    return body[i];
                }
            } idx = {{row, 0}};
            // Special union is needed to avoid
            // "error: array subscript is above array bounds [-Werror=array-bounds]"
            // when call the functor `op` such that access idx[3].

            _Tp* pixel = &(mat->template at<_Tp>(idx));
            const _Tp* const pixel_end = pixel + COLS;
            while(pixel < pixel_end) {
                op(*pixel++, static_cast<const int*>(idx));
                idx[1]++;
            }
        }
        PixelOperationWrapper& operator=(const PixelOperationWrapper &) {
            CV_Assert(false);
            // We can not remove this implementation because Visual Studio warning C4822.
            return *this;
        }
    };

    parallel_for_(cv::Range(0, LINES), PixelOperationWrapper(reinterpret_cast<Mat_<_Tp>*>(this), operation));
}

/////////////////////////// Synchronization Primitives ///////////////////////////////

class CV_EXPORTS Mutex
{
public:
    Mutex();
    ~Mutex();
    Mutex(const Mutex& m);
    Mutex& operator = (const Mutex& m);

    void lock();
    bool trylock();
    void unlock();

    struct Impl;
protected:
    Impl* impl;
};

class CV_EXPORTS AutoLock
{
public:
    AutoLock(Mutex& m) : mutex(&m) { mutex->lock(); }
    ~AutoLock() { mutex->unlock(); }
protected:
    Mutex* mutex;
private:
    AutoLock(const AutoLock&);
    AutoLock& operator = (const AutoLock&);
};

// TLS interface
class CV_EXPORTS TLSDataContainer
{
protected:
    TLSDataContainer();
    virtual ~TLSDataContainer();

    void  gatherData(std::vector<void*> &data) const;
#if OPENCV_ABI_COMPATIBILITY > 300
    void* getData() const;
    void  release();

private:
#else
    void  release();

public:
    void* getData() const;
#endif
    virtual void* createDataInstance() const = 0;
    virtual void  deleteDataInstance(void* pData) const = 0;

    int key_;

public:
    void cleanup(); //! Release created TLS data container objects. It is similar to release() call, but it keeps TLS container valid.
};

// Main TLS data class
template <typename T>
class TLSData : protected TLSDataContainer
{
public:
    inline TLSData()        {}
    inline ~TLSData()       { release();            } // Release key and delete associated data
    inline T* get() const   { return (T*)getData(); } // Get data associated with key
    inline T& getRef() const { T* ptr = (T*)getData(); CV_Assert(ptr); return *ptr; } // Get data associated with key

    // Get data from all threads
    inline void gather(std::vector<T*> &data) const
    {
        std::vector<void*> &dataVoid = reinterpret_cast<std::vector<void*>&>(data);
        gatherData(dataVoid);
    }

    inline void cleanup() { TLSDataContainer::cleanup(); }

private:
    virtual void* createDataInstance() const {return new T;}                // Wrapper to allocate data by template
    virtual void  deleteDataInstance(void* pData) const {delete (T*)pData;} // Wrapper to release data by template

    // Disable TLS copy operations
    TLSData(TLSData &) {}
    TLSData& operator =(const TLSData &) {return *this;}
};

/** @brief Designed for command line parsing

The sample below demonstrates how to use CommandLineParser:
@code
    CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int N = parser.get<int>("N");
    double fps = parser.get<double>("fps");
    String path = parser.get<String>("path");

    use_time_stamp = parser.has("timestamp");

    String img1 = parser.get<String>(0);
    String img2 = parser.get<String>(1);

    int repeat = parser.get<int>(2);

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
@endcode

### Keys syntax

The keys parameter is a string containing several blocks, each one is enclosed in curly braces and
describes one argument. Each argument contains three parts separated by the `|` symbol:

-# argument names is a space-separated list of option synonyms (to mark argument as positional, prefix it with the `@` symbol)
-# default value will be used if the argument was not provided (can be empty)
-# help message (can be empty)

For example:

@code{.cpp}
    const String keys =
        "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1 for compare   }"
        "{@image2        |<none>| image2 for compare   }"
        "{@repeat        |1     | number               }"
        "{path           |.     | path to file         }"
        "{fps            | -1.0 | fps for output video }"
        "{N count        |100   | count of objects     }"
        "{ts timestamp   |      | use time stamp       }"
        ;
}
@endcode

Note that there are no default values for `help` and `timestamp` so we can check their presence using the `has()` method.
Arguments with default values are considered to be always present. Use the `get()` method in these cases to check their
actual value instead.

String keys like `get<String>("@image1")` return the empty string `""` by default - even with an empty default value.
Use the special `<none>` default value to enforce that the returned string must not be empty. (like in `get<String>("@image2")`)

### Usage

For the described keys:

@code{.sh}
    # Good call (3 positional parameters: image1, image2 and repeat; N is 200, ts is true)
    $ ./app -N=200 1.png 2.jpg 19 -ts

    # Bad call
    $ ./app -fps=aaa
    ERRORS:
    Parameter 'fps': can not convert: [aaa] to [double]
@endcode
 */
class CV_EXPORTS CommandLineParser
{
public:

    /** @brief Constructor

    Initializes command line parser object

    @param argc number of command line arguments (from main())
    @param argv array of command line arguments (from main())
    @param keys string describing acceptable command line parameters (see class description for syntax)
    */
    CommandLineParser(int argc, const char* const argv[], const String& keys);

    /** @brief Copy constructor */
    CommandLineParser(const CommandLineParser& parser);

    /** @brief Assignment operator */
    CommandLineParser& operator = (const CommandLineParser& parser);

    /** @brief Destructor */
    ~CommandLineParser();

    /** @brief Returns application path

    This method returns the path to the executable from the command line (`argv[0]`).

    For example, if the application has been started with such command:
    @code{.sh}
    $ ./bin/my-executable
    @endcode
    this method will return `./bin`.
    */
    String getPathToApplication() const;

    /** @brief Access arguments by name

    Returns argument converted to selected type. If the argument is not known or can not be
    converted to selected type, the error flag is set (can be checked with @ref check).

    For example, define:
    @code{.cpp}
    String keys = "{N count||}";
    @endcode

    Call:
    @code{.sh}
    $ ./my-app -N=20
    # or
    $ ./my-app --count=20
    @endcode

    Access:
    @code{.cpp}
    int N = parser.get<int>("N");
    @endcode

    @param name name of the argument
    @param space_delete remove spaces from the left and right of the string
    @tparam T the argument will be converted to this type if possible

    @note You can access positional arguments by their `@`-prefixed name:
    @code{.cpp}
    parser.get<String>("@image");
    @endcode
     */
    template <typename T>
    T get(const String& name, bool space_delete = true) const
    {
        T val = T();
        getByName(name, space_delete, ParamType<T>::type, (void*)&val);
        return val;
    }

    /** @brief Access positional arguments by index

    Returns argument converted to selected type. Indexes are counted from zero.

    For example, define:
    @code{.cpp}
    String keys = "{@arg1||}{@arg2||}"
    @endcode

    Call:
    @code{.sh}
    ./my-app abc qwe
    @endcode

    Access arguments:
    @code{.cpp}
    String val_1 = parser.get<String>(0); // returns "abc", arg1
    String val_2 = parser.get<String>(1); // returns "qwe", arg2
    @endcode

    @param index index of the argument
    @param space_delete remove spaces from the left and right of the string
    @tparam T the argument will be converted to this type if possible
     */
    template <typename T>
    T get(int index, bool space_delete = true) const
    {
        T val = T();
        getByIndex(index, space_delete, ParamType<T>::type, (void*)&val);
        return val;
    }

    /** @brief Check if field was provided in the command line

    @param name argument name to check
    */
    bool has(const String& name) const;

    /** @brief Check for parsing errors

    Returns true if error occurred while accessing the parameters (bad conversion, missing arguments,
    etc.). Call @ref printErrors to print error messages list.
     */
    bool check() const;

    /** @brief Set the about message

    The about message will be shown when @ref printMessage is called, right before arguments table.
     */
    void about(const String& message);

    /** @brief Print help message

    This method will print standard help message containing the about message and arguments description.

    @sa about
    */
    void printMessage() const;

    /** @brief Print list of errors occured

    @sa check
    */
    void printErrors() const;

protected:
    void getByName(const String& name, bool space_delete, int type, void* dst) const;
    void getByIndex(int index, bool space_delete, int type, void* dst) const;

    struct Impl;
    Impl* impl;
};

//! @} core_utils

//! @cond IGNORED

/////////////////////////////// AutoBuffer implementation ////////////////////////////////////////

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer()
{
    ptr = buf;
    sz = fixed_size;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer(size_t _size)
{
    ptr = buf;
    sz = fixed_size;
    allocate(_size);
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer(const AutoBuffer<_Tp, fixed_size>& abuf )
{
    ptr = buf;
    sz = fixed_size;
    allocate(abuf.size());
    for( size_t i = 0; i < sz; i++ )
        ptr[i] = abuf.ptr[i];
}

template<typename _Tp, size_t fixed_size> inline AutoBuffer<_Tp, fixed_size>&
AutoBuffer<_Tp, fixed_size>::operator = (const AutoBuffer<_Tp, fixed_size>& abuf)
{
    if( this != &abuf )
    {
        deallocate();
        allocate(abuf.size());
        for( size_t i = 0; i < sz; i++ )
            ptr[i] = abuf.ptr[i];
    }
    return *this;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::~AutoBuffer()
{ deallocate(); }

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::allocate(size_t _size)
{
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    deallocate();
    sz = _size;
    if(_size > fixed_size)
    {
        ptr = new _Tp[_size];
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::deallocate()
{
    if( ptr != buf )
    {
        delete[] ptr;
        ptr = buf;
        sz = fixed_size;
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::resize(size_t _size)
{
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    size_t i, prevsize = sz, minsize = MIN(prevsize, _size);
    _Tp* prevptr = ptr;

    ptr = _size > fixed_size ? new _Tp[_size] : buf;
    sz = _size;

    if( ptr != prevptr )
        for( i = 0; i < minsize; i++ )
            ptr[i] = prevptr[i];
    for( i = prevsize; i < _size; i++ )
        ptr[i] = _Tp();

    if( prevptr != buf )
        delete[] prevptr;
}

template<typename _Tp, size_t fixed_size> inline size_t
AutoBuffer<_Tp, fixed_size>::size() const
{ return sz; }

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::operator _Tp* ()
{ return ptr; }

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::operator const _Tp* () const
{ return ptr; }

template<> inline std::string CommandLineParser::get<std::string>(int index, bool space_delete) const
{
    return get<String>(index, space_delete);
}
template<> inline std::string CommandLineParser::get<std::string>(const String& name, bool space_delete) const
{
    return get<String>(name, space_delete);
}

//! @endcond


// Basic Node class for tree building
template<class OBJECT>
class CV_EXPORTS Node
{
public:
    Node()
    {
        m_pParent  = 0;
    }
    Node(OBJECT& payload) : m_payload(payload)
    {
        m_pParent  = 0;
    }
    ~Node()
    {
        removeChilds();
        if (m_pParent)
        {
            int idx = m_pParent->findChild(this);
            if (idx >= 0)
                m_pParent->m_childs.erase(m_pParent->m_childs.begin() + idx);
        }
    }

    Node<OBJECT>* findChild(OBJECT& payload) const
    {
        for(size_t i = 0; i < this->m_childs.size(); i++)
        {
            if(this->m_childs[i]->m_payload == payload)
                return this->m_childs[i];
        }
        return NULL;
    }

    int findChild(Node<OBJECT> *pNode) const
    {
        for (size_t i = 0; i < this->m_childs.size(); i++)
        {
            if(this->m_childs[i] == pNode)
                return (int)i;
        }
        return -1;
    }

    void addChild(Node<OBJECT> *pNode)
    {
        if(!pNode)
            return;

        CV_Assert(pNode->m_pParent == 0);
        pNode->m_pParent = this;
        this->m_childs.push_back(pNode);
    }

    void removeChilds()
    {
        for(size_t i = 0; i < m_childs.size(); i++)
        {
            m_childs[i]->m_pParent = 0; // avoid excessive parent vector trimming
            delete m_childs[i];
        }
        m_childs.clear();
    }

    int getDepth()
    {
        int   count   = 0;
        Node *pParent = m_pParent;
        while(pParent) count++, pParent = pParent->m_pParent;
        return count;
    }

public:
    OBJECT                     m_payload;
    Node<OBJECT>*              m_pParent;
    std::vector<Node<OBJECT>*> m_childs;
};

// Instrumentation external interface
namespace instr
{

#if !defined OPENCV_ABI_CHECK

enum TYPE
{
    TYPE_GENERAL = 0,   // OpenCV API function, e.g. exported function
    TYPE_MARKER,        // Information marker
    TYPE_WRAPPER,       // Wrapper function for implementation
    TYPE_FUN,           // Simple function call
};

enum IMPL
{
    IMPL_PLAIN = 0,
    IMPL_IPP,
    IMPL_OPENCL,
};

struct NodeDataTls
{
    NodeDataTls()
    {
        m_ticksTotal = 0;
    }
    uint64      m_ticksTotal;
};

class CV_EXPORTS NodeData
{
public:
    NodeData(const char* funName = 0, const char* fileName = NULL, int lineNum = 0, void* retAddress = NULL, bool alwaysExpand = false, cv::instr::TYPE instrType = TYPE_GENERAL, cv::instr::IMPL implType = IMPL_PLAIN);
    NodeData(NodeData &ref);
    ~NodeData();
    NodeData& operator=(const NodeData&);

    cv::String          m_funName;
    cv::instr::TYPE     m_instrType;
    cv::instr::IMPL     m_implType;
    const char*         m_fileName;
    int                 m_lineNum;
    void*               m_retAddress;
    bool                m_alwaysExpand;
    bool                m_funError;

    volatile int         m_counter;
    volatile uint64      m_ticksTotal;
    TLSData<NodeDataTls> m_tls;
    int                  m_threads;

    // No synchronization
    double getTotalMs()   const { return ((double)m_ticksTotal / cv::getTickFrequency()) * 1000; }
    double getMeanMs()    const { return (((double)m_ticksTotal/m_counter) / cv::getTickFrequency()) * 1000; }
};
bool operator==(const NodeData& lhs, const NodeData& rhs);

typedef Node<NodeData> InstrNode;

CV_EXPORTS InstrNode* getTrace();

#endif // !defined OPENCV_ABI_CHECK


CV_EXPORTS bool       useInstrumentation();
CV_EXPORTS void       setUseInstrumentation(bool flag);
CV_EXPORTS void       resetTrace();

enum FLAGS
{
    FLAGS_NONE              = 0,
    FLAGS_MAPPING           = 0x01,
    FLAGS_EXPAND_SAME_NAMES = 0x02,
};

CV_EXPORTS void       setFlags(FLAGS modeFlags);
static inline void    setFlags(int modeFlags) { setFlags((FLAGS)modeFlags); }
CV_EXPORTS FLAGS      getFlags();
}

namespace utils {

CV_EXPORTS int getThreadID();

} // namespace

} //namespace cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/core/core_c.h"
#endif

#endif //OPENCV_CORE_UTILITY_H
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2015, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CORE_VA_INTEL_HPP
#define OPENCV_CORE_VA_INTEL_HPP

#ifndef __cplusplus
#  error va_intel.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"
#include "ocl.hpp"

#if defined(HAVE_VA)
# include "va/va.h"
#else  // HAVE_VA
# if !defined(_VA_H_)
    typedef void* VADisplay;
    typedef unsigned int VASurfaceID;
# endif // !_VA_H_
#endif // HAVE_VA

namespace cv { namespace va_intel {

/** @addtogroup core_va_intel
This section describes Intel VA-API/OpenCL (CL-VA) interoperability.

To enable CL-VA interoperability support, configure OpenCV using CMake with WITH_VA_INTEL=ON . Currently VA-API is
supported on Linux only. You should also install Intel Media Server Studio (MSS) to use this feature. You may
have to specify the path(s) to MSS components for cmake in environment variables: VA_INTEL_MSDK_ROOT for Media SDK
(default is "/opt/intel/mediasdk"), and VA_INTEL_IOCL_ROOT for Intel OpenCL (default is "/opt/intel/opencl").

To use CL-VA interoperability you should first create VADisplay (libva), and then call initializeContextFromVA()
function to create OpenCL context and set up interoperability.
*/
//! @{

/////////////////// CL-VA Interoperability Functions ///////////////////

namespace ocl {
using namespace cv::ocl;

// TODO static functions in the Context class
/** @brief Creates OpenCL context from VA.
@param display    - VADisplay for which CL interop should be established.
@param tryInterop - try to set up for interoperability, if true; set up for use slow copy if false.
@return Returns reference to OpenCL Context
 */
CV_EXPORTS Context& initializeContextFromVA(VADisplay display, bool tryInterop = true);

} // namespace cv::va_intel::ocl

/** @brief Converts InputArray to VASurfaceID object.
@param display - VADisplay object.
@param src     - source InputArray.
@param surface - destination VASurfaceID object.
@param size    - size of image represented by VASurfaceID object.
 */
CV_EXPORTS void convertToVASurface(VADisplay display, InputArray src, VASurfaceID surface, Size size);

/** @brief Converts VASurfaceID object to OutputArray.
@param display - VADisplay object.
@param surface - source VASurfaceID object.
@param size    - size of image represented by VASurfaceID object.
@param dst     - destination OutputArray.
 */
CV_EXPORTS void convertFromVASurface(VADisplay display, VASurfaceID surface, Size size, OutputArray dst);

//! @}

}} // namespace cv::va_intel

#endif /* OPENCV_CORE_VA_INTEL_HPP */
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright( C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2011-2013, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
  definition of the current version of OpenCV
  Usefull to test in user programs
*/

#ifndef OPENCV_VERSION_HPP
#define OPENCV_VERSION_HPP

#define CV_VERSION_MAJOR    3
#define CV_VERSION_MINOR    3
#define CV_VERSION_REVISION 0
#define CV_VERSION_STATUS   ""

#define CVAUX_STR_EXP(__A)  #__A
#define CVAUX_STR(__A)      CVAUX_STR_EXP(__A)

#define CVAUX_STRW_EXP(__A)  L ## #__A
#define CVAUX_STRW(__A)      CVAUX_STRW_EXP(__A)

#define CV_VERSION          CVAUX_STR(CV_VERSION_MAJOR) "." CVAUX_STR(CV_VERSION_MINOR) "." CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS

/* old  style version constants*/
#define CV_MAJOR_VERSION    CV_VERSION_MAJOR
#define CV_MINOR_VERSION    CV_VERSION_MINOR
#define CV_SUBMINOR_VERSION CV_VERSION_REVISION

#endif
/*M//////////////////////////////////////////////////////////////////////////////
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to
//  this license.  If you do not agree to this license, do not download,
//  install, copy or use the software.
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008, Google, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//  * The name of Intel Corporation or contributors may not be used to endorse
//     or promote products derived from this software without specific
//     prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
/////////////////////////////////////////////////////////////////////////////////
//M*/

#ifndef OPENCV_CORE_WIMAGE_HPP
#define OPENCV_CORE_WIMAGE_HPP

#include "opencv2/core/core_c.h"

#ifdef __cplusplus

namespace cv {

//! @addtogroup core
//! @{

template <typename T> class WImage;
template <typename T> class WImageBuffer;
template <typename T> class WImageView;

template<typename T, int C> class WImageC;
template<typename T, int C> class WImageBufferC;
template<typename T, int C> class WImageViewC;

// Commonly used typedefs.
typedef WImage<uchar>            WImage_b;
typedef WImageView<uchar>        WImageView_b;
typedef WImageBuffer<uchar>      WImageBuffer_b;

typedef WImageC<uchar, 1>        WImage1_b;
typedef WImageViewC<uchar, 1>    WImageView1_b;
typedef WImageBufferC<uchar, 1>  WImageBuffer1_b;

typedef WImageC<uchar, 3>        WImage3_b;
typedef WImageViewC<uchar, 3>    WImageView3_b;
typedef WImageBufferC<uchar, 3>  WImageBuffer3_b;

typedef WImage<float>            WImage_f;
typedef WImageView<float>        WImageView_f;
typedef WImageBuffer<float>      WImageBuffer_f;

typedef WImageC<float, 1>        WImage1_f;
typedef WImageViewC<float, 1>    WImageView1_f;
typedef WImageBufferC<float, 1>  WImageBuffer1_f;

typedef WImageC<float, 3>        WImage3_f;
typedef WImageViewC<float, 3>    WImageView3_f;
typedef WImageBufferC<float, 3>  WImageBuffer3_f;

// There isn't a standard for signed and unsigned short so be more
// explicit in the typename for these cases.
typedef WImage<short>            WImage_16s;
typedef WImageView<short>        WImageView_16s;
typedef WImageBuffer<short>      WImageBuffer_16s;

typedef WImageC<short, 1>        WImage1_16s;
typedef WImageViewC<short, 1>    WImageView1_16s;
typedef WImageBufferC<short, 1>  WImageBuffer1_16s;

typedef WImageC<short, 3>        WImage3_16s;
typedef WImageViewC<short, 3>    WImageView3_16s;
typedef WImageBufferC<short, 3>  WImageBuffer3_16s;

typedef WImage<ushort>            WImage_16u;
typedef WImageView<ushort>        WImageView_16u;
typedef WImageBuffer<ushort>      WImageBuffer_16u;

typedef WImageC<ushort, 1>        WImage1_16u;
typedef WImageViewC<ushort, 1>    WImageView1_16u;
typedef WImageBufferC<ushort, 1>  WImageBuffer1_16u;

typedef WImageC<ushort, 3>        WImage3_16u;
typedef WImageViewC<ushort, 3>    WImageView3_16u;
typedef WImageBufferC<ushort, 3>  WImageBuffer3_16u;

/** @brief Image class which provides a thin layer around an IplImage.

The goals of the class design are:

    -# All the data has explicit ownership to avoid memory leaks
    -# No hidden allocations or copies for performance.
    -# Easy access to OpenCV methods (which will access IPP if available)
    -# Can easily treat external data as an image
    -# Easy to create images which are subsets of other images
    -# Fast pixel access which can take advantage of number of channels if known at compile time.

The WImage class is the image class which provides the data accessors. The 'W' comes from the fact
that it is also a wrapper around the popular but inconvenient IplImage class. A WImage can be
constructed either using a WImageBuffer class which allocates and frees the data, or using a
WImageView class which constructs a subimage or a view into external data. The view class does no
memory management. Each class actually has two versions, one when the number of channels is known
at compile time and one when it isn't. Using the one with the number of channels specified can
provide some compile time optimizations by using the fact that the number of channels is a
constant.

We use the convention (c,r) to refer to column c and row r with (0,0) being the upper left corner.
This is similar to standard Euclidean coordinates with the first coordinate varying in the
horizontal direction and the second coordinate varying in the vertical direction. Thus (c,r) is
usually in the domain [0, width) X [0, height)

Example usage:
@code
WImageBuffer3_b  im(5,7);  // Make a 5X7 3 channel image of type uchar
WImageView3_b  sub_im(im, 2,2, 3,3); // 3X3 submatrix
vector<float> vec(10, 3.0f);
WImageView1_f user_im(&vec[0], 2, 5);  // 2X5 image w/ supplied data

im.SetZero();  // same as cvSetZero(im.Ipl())
*im(2, 3) = 15;  // Modify the element at column 2, row 3
MySetRand(&sub_im);

// Copy the second row into the first.  This can be done with no memory
// allocation and will use SSE if IPP is available.
int w = im.Width();
im.View(0,0, w,1).CopyFrom(im.View(0,1, w,1));

// Doesn't care about source of data since using WImage
void MySetRand(WImage_b* im) { // Works with any number of channels
for (int r = 0; r < im->Height(); ++r) {
 float* row = im->Row(r);
 for (int c = 0; c < im->Width(); ++c) {
    for (int ch = 0; ch < im->Channels(); ++ch, ++row) {
      *row = uchar(rand() & 255);
    }
 }
}
}
@endcode

Functions that are not part of the basic image allocation, viewing, and access should come from
OpenCV, except some useful functions that are not part of OpenCV can be found in wimage_util.h
*/
template<typename T>
class WImage
{
public:
    typedef T BaseType;

    // WImage is an abstract class with no other virtual methods so make the
    // destructor virtual.
    virtual ~WImage() = 0;

    // Accessors
    IplImage* Ipl() {return image_; }
    const IplImage* Ipl() const {return image_; }
    T* ImageData() { return reinterpret_cast<T*>(image_->imageData); }
    const T* ImageData() const {
        return reinterpret_cast<const T*>(image_->imageData);
    }

    int Width() const {return image_->width; }
    int Height() const {return image_->height; }

    // WidthStep is the number of bytes to go to the pixel with the next y coord
    int WidthStep() const {return image_->widthStep; }

    int Channels() const {return image_->nChannels; }
    int ChannelSize() const {return sizeof(T); }  // number of bytes per channel

    // Number of bytes per pixel
    int PixelSize() const {return Channels() * ChannelSize(); }

    // Return depth type (e.g. IPL_DEPTH_8U, IPL_DEPTH_32F) which is the number
    // of bits per channel and with the signed bit set.
    // This is known at compile time using specializations.
    int Depth() const;

    inline const T* Row(int r) const {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep);
    }

    inline T* Row(int r) {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep);
    }

    // Pixel accessors which returns a pointer to the start of the channel
    inline T* operator() (int c, int r)  {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep) +
            c*Channels();
    }

    inline const T* operator() (int c, int r) const  {
        return reinterpret_cast<T*>(image_->imageData + r*image_->widthStep) +
            c*Channels();
    }

    // Copy the contents from another image which is just a convenience to cvCopy
    void CopyFrom(const WImage<T>& src) { cvCopy(src.Ipl(), image_); }

    // Set contents to zero which is just a convenient to cvSetZero
    void SetZero() { cvSetZero(image_); }

    // Construct a view into a region of this image
    WImageView<T> View(int c, int r, int width, int height);

protected:
    // Disallow copy and assignment
    WImage(const WImage&);
    void operator=(const WImage&);

    explicit WImage(IplImage* img) : image_(img) {
        assert(!img || img->depth == Depth());
    }

    void SetIpl(IplImage* image) {
        assert(!image || image->depth == Depth());
        image_ = image;
    }

    IplImage* image_;
};


/** Image class when both the pixel type and number of channels
are known at compile time.  This wrapper will speed up some of the operations
like accessing individual pixels using the () operator.
*/
template<typename T, int C>
class WImageC : public WImage<T>
{
public:
    typedef typename WImage<T>::BaseType BaseType;
    enum { kChannels = C };

    explicit WImageC(IplImage* img) : WImage<T>(img) {
        assert(!img || img->nChannels == Channels());
    }

    // Construct a view into a region of this image
    WImageViewC<T, C> View(int c, int r, int width, int height);

    // Copy the contents from another image which is just a convenience to cvCopy
    void CopyFrom(const WImageC<T, C>& src) {
        cvCopy(src.Ipl(), WImage<T>::image_);
    }

    // WImageC is an abstract class with no other virtual methods so make the
    // destructor virtual.
    virtual ~WImageC() = 0;

    int Channels() const {return C; }

protected:
    // Disallow copy and assignment
    WImageC(const WImageC&);
    void operator=(const WImageC&);

    void SetIpl(IplImage* image) {
        assert(!image || image->depth == WImage<T>::Depth());
        WImage<T>::SetIpl(image);
    }
};

/** Image class which owns the data, so it can be allocated and is always
freed.  It cannot be copied but can be explicity cloned.
*/
template<typename T>
class WImageBuffer : public WImage<T>
{
public:
    typedef typename WImage<T>::BaseType BaseType;

    // Default constructor which creates an object that can be
    WImageBuffer() : WImage<T>(0) {}

    WImageBuffer(int width, int height, int nchannels) : WImage<T>(0) {
        Allocate(width, height, nchannels);
    }

    // Constructor which takes ownership of a given IplImage so releases
    // the image on destruction.
    explicit WImageBuffer(IplImage* img) : WImage<T>(img) {}

    // Allocate an image.  Does nothing if current size is the same as
    // the new size.
    void Allocate(int width, int height, int nchannels);

    // Set the data to point to an image, releasing the old data
    void SetIpl(IplImage* img) {
        ReleaseImage();
        WImage<T>::SetIpl(img);
    }

    // Clone an image which reallocates the image if of a different dimension.
    void CloneFrom(const WImage<T>& src) {
        Allocate(src.Width(), src.Height(), src.Channels());
        CopyFrom(src);
    }

    ~WImageBuffer() {
        ReleaseImage();
    }

    // Release the image if it isn't null.
    void ReleaseImage() {
        if (WImage<T>::image_) {
            IplImage* image = WImage<T>::image_;
            cvReleaseImage(&image);
            WImage<T>::SetIpl(0);
        }
    }

    bool IsNull() const {return WImage<T>::image_ == NULL; }

private:
    // Disallow copy and assignment
    WImageBuffer(const WImageBuffer&);
    void operator=(const WImageBuffer&);
};

/** Like a WImageBuffer class but when the number of channels is known at compile time.
*/
template<typename T, int C>
class WImageBufferC : public WImageC<T, C>
{
public:
    typedef typename WImage<T>::BaseType BaseType;
    enum { kChannels = C };

    // Default constructor which creates an object that can be
    WImageBufferC() : WImageC<T, C>(0) {}

    WImageBufferC(int width, int height) : WImageC<T, C>(0) {
        Allocate(width, height);
    }

    // Constructor which takes ownership of a given IplImage so releases
    // the image on destruction.
    explicit WImageBufferC(IplImage* img) : WImageC<T, C>(img) {}

    // Allocate an image.  Does nothing if current size is the same as
    // the new size.
    void Allocate(int width, int height);

    // Set the data to point to an image, releasing the old data
    void SetIpl(IplImage* img) {
        ReleaseImage();
        WImageC<T, C>::SetIpl(img);
    }

    // Clone an image which reallocates the image if of a different dimension.
    void CloneFrom(const WImageC<T, C>& src) {
        Allocate(src.Width(), src.Height());
        CopyFrom(src);
    }

    ~WImageBufferC() {
        ReleaseImage();
    }

    // Release the image if it isn't null.
    void ReleaseImage() {
        if (WImage<T>::image_) {
            IplImage* image = WImage<T>::image_;
            cvReleaseImage(&image);
            WImageC<T, C>::SetIpl(0);
        }
    }

    bool IsNull() const {return WImage<T>::image_ == NULL; }

private:
    // Disallow copy and assignment
    WImageBufferC(const WImageBufferC&);
    void operator=(const WImageBufferC&);
};

/** View into an image class which allows treating a subimage as an image or treating external data
as an image
*/
template<typename T> class WImageView : public WImage<T>
{
public:
    typedef typename WImage<T>::BaseType BaseType;

    // Construct a subimage.  No checks are done that the subimage lies
    // completely inside the original image.
    WImageView(WImage<T>* img, int c, int r, int width, int height);

    // Refer to external data.
    // If not given width_step assumed to be same as width.
    WImageView(T* data, int width, int height, int channels, int width_step = -1);

    // Refer to external data.  This does NOT take ownership
    // of the supplied IplImage.
    WImageView(IplImage* img) : WImage<T>(img) {}

    // Copy constructor
    WImageView(const WImage<T>& img) : WImage<T>(0) {
        header_ = *(img.Ipl());
        WImage<T>::SetIpl(&header_);
    }

    WImageView& operator=(const WImage<T>& img) {
        header_ = *(img.Ipl());
        WImage<T>::SetIpl(&header_);
        return *this;
    }

protected:
    IplImage header_;
};


template<typename T, int C>
class WImageViewC : public WImageC<T, C>
{
public:
    typedef typename WImage<T>::BaseType BaseType;
    enum { kChannels = C };

    // Default constructor needed for vectors of views.
    WImageViewC();

    virtual ~WImageViewC() {}

    // Construct a subimage.  No checks are done that the subimage lies
    // completely inside the original image.
    WImageViewC(WImageC<T, C>* img,
        int c, int r, int width, int height);

    // Refer to external data
    WImageViewC(T* data, int width, int height, int width_step = -1);

    // Refer to external data.  This does NOT take ownership
    // of the supplied IplImage.
    WImageViewC(IplImage* img) : WImageC<T, C>(img) {}

    // Copy constructor which does a shallow copy to allow multiple views
    // of same data.  gcc-4.1.1 gets confused if both versions of
    // the constructor and assignment operator are not provided.
    WImageViewC(const WImageC<T, C>& img) : WImageC<T, C>(0) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
    }
    WImageViewC(const WImageViewC<T, C>& img) : WImageC<T, C>(0) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
    }

    WImageViewC& operator=(const WImageC<T, C>& img) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
        return *this;
    }
    WImageViewC& operator=(const WImageViewC<T, C>& img) {
        header_ = *(img.Ipl());
        WImageC<T, C>::SetIpl(&header_);
        return *this;
    }

protected:
    IplImage header_;
};


// Specializations for depth
template<>
inline int WImage<uchar>::Depth() const {return IPL_DEPTH_8U; }
template<>
inline int WImage<signed char>::Depth() const {return IPL_DEPTH_8S; }
template<>
inline int WImage<short>::Depth() const {return IPL_DEPTH_16S; }
template<>
inline int WImage<ushort>::Depth() const {return IPL_DEPTH_16U; }
template<>
inline int WImage<int>::Depth() const {return IPL_DEPTH_32S; }
template<>
inline int WImage<float>::Depth() const {return IPL_DEPTH_32F; }
template<>
inline int WImage<double>::Depth() const {return IPL_DEPTH_64F; }

template<typename T> inline WImage<T>::~WImage() {}
template<typename T, int C> inline WImageC<T, C>::~WImageC() {}

template<typename T>
inline void WImageBuffer<T>::Allocate(int width, int height, int nchannels)
{
    if (IsNull() || WImage<T>::Width() != width ||
        WImage<T>::Height() != height || WImage<T>::Channels() != nchannels) {
        ReleaseImage();
        WImage<T>::image_ = cvCreateImage(cvSize(width, height),
            WImage<T>::Depth(), nchannels);
    }
}

template<typename T, int C>
inline void WImageBufferC<T, C>::Allocate(int width, int height)
{
    if (IsNull() || WImage<T>::Width() != width || WImage<T>::Height() != height) {
        ReleaseImage();
        WImageC<T, C>::SetIpl(cvCreateImage(cvSize(width, height),WImage<T>::Depth(), C));
    }
}

template<typename T>
WImageView<T>::WImageView(WImage<T>* img, int c, int r, int width, int height)
        : WImage<T>(0)
{
    header_ = *(img->Ipl());
    header_.imageData = reinterpret_cast<char*>((*img)(c, r));
    header_.width = width;
    header_.height = height;
    WImage<T>::SetIpl(&header_);
}

template<typename T>
WImageView<T>::WImageView(T* data, int width, int height, int nchannels, int width_step)
          : WImage<T>(0)
{
    cvInitImageHeader(&header_, cvSize(width, height), WImage<T>::Depth(), nchannels);
    header_.imageData = reinterpret_cast<char*>(data);
    if (width_step > 0) {
        header_.widthStep = width_step;
    }
    WImage<T>::SetIpl(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC(WImageC<T, C>* img, int c, int r, int width, int height)
        : WImageC<T, C>(0)
{
    header_ = *(img->Ipl());
    header_.imageData = reinterpret_cast<char*>((*img)(c, r));
    header_.width = width;
    header_.height = height;
    WImageC<T, C>::SetIpl(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC() : WImageC<T, C>(0) {
    cvInitImageHeader(&header_, cvSize(0, 0), WImage<T>::Depth(), C);
    header_.imageData = reinterpret_cast<char*>(0);
    WImageC<T, C>::SetIpl(&header_);
}

template<typename T, int C>
WImageViewC<T, C>::WImageViewC(T* data, int width, int height, int width_step)
    : WImageC<T, C>(0)
{
    cvInitImageHeader(&header_, cvSize(width, height), WImage<T>::Depth(), C);
    header_.imageData = reinterpret_cast<char*>(data);
    if (width_step > 0) {
        header_.widthStep = width_step;
    }
    WImageC<T, C>::SetIpl(&header_);
}

// Construct a view into a region of an image
template<typename T>
WImageView<T> WImage<T>::View(int c, int r, int width, int height) {
    return WImageView<T>(this, c, r, width, height);
}

template<typename T, int C>
WImageViewC<T, C> WImageC<T, C>::View(int c, int r, int width, int height) {
    return WImageViewC<T, C>(this, c, r, width, height);
}

//! @} core

}  // end of namespace

#endif // __cplusplus

#endif
