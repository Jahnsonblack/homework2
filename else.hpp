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

#ifndef OPENCV_CALIB3D_HPP
#define OPENCV_CALIB3D_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"

/**
  @defgroup calib3d Camera Calibration and 3D Reconstruction

The functions in this section use a so-called pinhole camera model. In this model, a scene view is
formed by projecting 3D points into the image plane using a perspective transformation.

\f[s  \; m' = A [R|t] M'\f]

or

\f[s  \vecthree{u}{v}{1} = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} & t_1  \\
r_{21} & r_{22} & r_{23} & t_2  \\
r_{31} & r_{32} & r_{33} & t_3
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}\f]

where:

-   \f$(X, Y, Z)\f$ are the coordinates of a 3D point in the world coordinate space
-   \f$(u, v)\f$ are the coordinates of the projection point in pixels
-   \f$A\f$ is a camera matrix, or a matrix of intrinsic parameters
-   \f$(cx, cy)\f$ is a principal point that is usually at the image center
-   \f$fx, fy\f$ are the focal lengths expressed in pixel units.

Thus, if an image from the camera is scaled by a factor, all of these parameters should be scaled
(multiplied/divided, respectively) by the same factor. The matrix of intrinsic parameters does not
depend on the scene viewed. So, once estimated, it can be re-used as long as the focal length is
fixed (in case of zoom lens). The joint rotation-translation matrix \f$[R|t]\f$ is called a matrix of
extrinsic parameters. It is used to describe the camera motion around a static scene, or vice versa,
rigid motion of an object in front of a still camera. That is, \f$[R|t]\f$ translates coordinates of a
point \f$(X, Y, Z)\f$ to a coordinate system, fixed with respect to the camera. The transformation above
is equivalent to the following (when \f$z \ne 0\f$ ):

\f[\begin{array}{l}
\vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\
x' = x/z \\
y' = y/z \\
u = f_x*x' + c_x \\
v = f_y*y' + c_y
\end{array}\f]

The following figure illustrates the pinhole camera model.

![Pinhole camera model](pics/pinhole_camera_model.png)

Real lenses usually have some distortion, mostly radial distortion and slight tangential distortion.
So, the above model is extended as:

\f[\begin{array}{l}
\vecthree{x}{y}{z} = R  \vecthree{X}{Y}{Z} + t \\
x' = x/z \\
y' = y/z \\
x'' = x'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2) + s_1 r^2 + s_2 r^4 \\
y'' = y'  \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
\text{where} \quad r^2 = x'^2 + y'^2  \\
u = f_x*x'' + c_x \\
v = f_y*y'' + c_y
\end{array}\f]

\f$k_1\f$, \f$k_2\f$, \f$k_3\f$, \f$k_4\f$, \f$k_5\f$, and \f$k_6\f$ are radial distortion coefficients. \f$p_1\f$ and \f$p_2\f$ are
tangential distortion coefficients. \f$s_1\f$, \f$s_2\f$, \f$s_3\f$, and \f$s_4\f$, are the thin prism distortion
coefficients. Higher-order coefficients are not considered in OpenCV.

The next figure shows two common types of radial distortion: barrel distortion (typically \f$ k_1 > 0 \f$ and pincushion distortion (typically \f$ k_1 < 0 \f$).

![](pics/distortion_examples.png)

In some cases the image sensor may be tilted in order to focus an oblique plane in front of the
camera (Scheimpfug condition). This can be useful for particle image velocimetry (PIV) or
triangulation with a laser fan. The tilt causes a perspective distortion of \f$x''\f$ and
\f$y''\f$. This distortion can be modelled in the following way, see e.g. @cite Louhichi07.

\f[\begin{array}{l}
s\vecthree{x'''}{y'''}{1} =
\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}(\tau_x, \tau_y)}
{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\
u = f_x*x''' + c_x \\
v = f_y*y''' + c_y
\end{array}\f]

where the matrix \f$R(\tau_x, \tau_y)\f$ is defined by two rotations with angular parameter \f$\tau_x\f$
and \f$\tau_y\f$, respectively,

\f[
R(\tau_x, \tau_y) =
\vecthreethree{\cos(\tau_y)}{0}{-\sin(\tau_y)}{0}{1}{0}{\sin(\tau_y)}{0}{\cos(\tau_y)}
\vecthreethree{1}{0}{0}{0}{\cos(\tau_x)}{\sin(\tau_x)}{0}{-\sin(\tau_x)}{\cos(\tau_x)} =
\vecthreethree{\cos(\tau_y)}{\sin(\tau_y)\sin(\tau_x)}{-\sin(\tau_y)\cos(\tau_x)}
{0}{\cos(\tau_x)}{\sin(\tau_x)}
{\sin(\tau_y)}{-\cos(\tau_y)\sin(\tau_x)}{\cos(\tau_y)\cos(\tau_x)}.
\f]

In the functions below the coefficients are passed or returned as

\f[(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f]

vector. That is, if the vector contains four elements, it means that \f$k_3=0\f$ . The distortion
coefficients do not depend on the scene viewed. Thus, they also belong to the intrinsic camera
parameters. And they remain the same regardless of the captured image resolution. If, for example, a
camera has been calibrated on images of 320 x 240 resolution, absolutely the same distortion
coefficients can be used for 640 x 480 images from the same camera while \f$f_x\f$, \f$f_y\f$, \f$c_x\f$, and
\f$c_y\f$ need to be scaled appropriately.

The functions below use the above model to do the following:

-   Project 3D points to the image plane given intrinsic and extrinsic parameters.
-   Compute extrinsic parameters given intrinsic parameters, a few 3D points, and their
projections.
-   Estimate intrinsic and extrinsic camera parameters from several views of a known calibration
pattern (every view is described by several 3D-2D point correspondences).
-   Estimate the relative position and orientation of the stereo camera "heads" and compute the
*rectification* transformation that makes the camera optical axes parallel.

@note
   -   A calibration sample for 3 cameras in horizontal position can be found at
        opencv_source_code/samples/cpp/3calibration.cpp
    -   A calibration sample based on a sequence of images can be found at
        opencv_source_code/samples/cpp/calibration.cpp
    -   A calibration sample in order to do 3D reconstruction can be found at
        opencv_source_code/samples/cpp/build3dmodel.cpp
    -   A calibration sample of an artificially generated camera and chessboard patterns can be
        found at opencv_source_code/samples/cpp/calibration_artificial.cpp
    -   A calibration example on stereo calibration can be found at
        opencv_source_code/samples/cpp/stereo_calib.cpp
    -   A calibration example on stereo matching can be found at
        opencv_source_code/samples/cpp/stereo_match.cpp
    -   (Python) A camera calibration sample can be found at
        opencv_source_code/samples/python/calibrate.py

  @{
    @defgroup calib3d_fisheye Fisheye camera model

    Definitions: Let P be a point in 3D of coordinates X in the world reference frame (stored in the
    matrix X) The coordinate vector of P in the camera reference frame is:

    \f[Xc = R X + T\f]

    where R is the rotation matrix corresponding to the rotation vector om: R = rodrigues(om); call x, y
    and z the 3 coordinates of Xc:

    \f[x = Xc_1 \\ y = Xc_2 \\ z = Xc_3\f]

    The pinhole projection coordinates of P is [a; b] where

    \f[a = x / z \ and \ b = y / z \\ r^2 = a^2 + b^2 \\ \theta = atan(r)\f]

    Fisheye distortion:

    \f[\theta_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8)\f]

    The distorted point coordinates are [x'; y'] where

    \f[x' = (\theta_d / r) a \\ y' = (\theta_d / r) b \f]

    Finally, conversion into pixel coordinates: The final pixel coordinates vector [u; v] where:

    \f[u = f_x (x' + \alpha y') + c_x \\
    v = f_y y' + c_y\f]

    @defgroup calib3d_c C API

  @}
 */

namespace cv
{

//! @addtogroup calib3d
//! @{

//! type of the robust estimation algorithm
enum { LMEDS  = 4, //!< least-median algorithm
       RANSAC = 8, //!< RANSAC algorithm
       RHO    = 16 //!< RHO algorithm
     };

enum { SOLVEPNP_ITERATIVE = 0,
       SOLVEPNP_EPNP      = 1, //!< EPnP: Efficient Perspective-n-Point Camera Pose Estimation @cite lepetit2009epnp
       SOLVEPNP_P3P       = 2, //!< Complete Solution Classification for the Perspective-Three-Point Problem @cite gao2003complete
       SOLVEPNP_DLS       = 3, //!< A Direct Least-Squares (DLS) Method for PnP  @cite hesch2011direct
       SOLVEPNP_UPNP      = 4, //!< Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation @cite penate2013exhaustive
       SOLVEPNP_AP3P      = 5, //!< An Efficient Algebraic Solution to the Perspective-Three-Point Problem @cite Ke17
       SOLVEPNP_MAX_COUNT      //!< Used for count
};

enum { CALIB_CB_ADAPTIVE_THRESH = 1,
       CALIB_CB_NORMALIZE_IMAGE = 2,
       CALIB_CB_FILTER_QUADS    = 4,
       CALIB_CB_FAST_CHECK      = 8
     };

enum { CALIB_CB_SYMMETRIC_GRID  = 1,
       CALIB_CB_ASYMMETRIC_GRID = 2,
       CALIB_CB_CLUSTERING      = 4
     };

enum { CALIB_USE_INTRINSIC_GUESS = 0x00001,
       CALIB_FIX_ASPECT_RATIO    = 0x00002,
       CALIB_FIX_PRINCIPAL_POINT = 0x00004,
       CALIB_ZERO_TANGENT_DIST   = 0x00008,
       CALIB_FIX_FOCAL_LENGTH    = 0x00010,
       CALIB_FIX_K1              = 0x00020,
       CALIB_FIX_K2              = 0x00040,
       CALIB_FIX_K3              = 0x00080,
       CALIB_FIX_K4              = 0x00800,
       CALIB_FIX_K5              = 0x01000,
       CALIB_FIX_K6              = 0x02000,
       CALIB_RATIONAL_MODEL      = 0x04000,
       CALIB_THIN_PRISM_MODEL    = 0x08000,
       CALIB_FIX_S1_S2_S3_S4     = 0x10000,
       CALIB_TILTED_MODEL        = 0x40000,
       CALIB_FIX_TAUX_TAUY       = 0x80000,
       CALIB_USE_QR              = 0x100000, //!< use QR instead of SVD decomposition for solving. Faster but potentially less precise
       CALIB_FIX_TANGENT_DIST    = 0x200000,
       // only for stereo
       CALIB_FIX_INTRINSIC       = 0x00100,
       CALIB_SAME_FOCAL_LENGTH   = 0x00200,
       // for stereo rectification
       CALIB_ZERO_DISPARITY      = 0x00400,
       CALIB_USE_LU              = (1 << 17), //!< use LU instead of SVD decomposition for solving. much faster but potentially less precise
     };

//! the algorithm for finding fundamental matrix
enum { FM_7POINT = 1, //!< 7-point algorithm
       FM_8POINT = 2, //!< 8-point algorithm
       FM_LMEDS  = 4, //!< least-median algorithm
       FM_RANSAC = 8  //!< RANSAC algorithm
     };



/** @brief Converts a rotation matrix to a rotation vector or vice versa.

@param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).
@param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.
@param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial
derivatives of the output array components with respect to the input array components.

\f[\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos{\theta} I + (1- \cos{\theta} ) r r^T +  \sin{\theta} \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\f]

Inverse transformation can be also done easily, since

\f[\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\f]

A rotation vector is a convenient and most compact representation of a rotation matrix (since any
rotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry
optimization procedures like calibrateCamera, stereoCalibrate, or solvePnP .
 */
CV_EXPORTS_W void Rodrigues( InputArray src, OutputArray dst, OutputArray jacobian = noArray() );

/** @brief Finds a perspective transformation between two planes.

@param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2
or vector\<Point2f\> .
@param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or
a vector\<Point2f\> .
@param method Method used to computed a homography matrix. The following methods are possible:
-   **0** - a regular method using all the points
-   **RANSAC** - RANSAC-based robust method
-   **LMEDS** - Least-Median robust method
-   **RHO**    - PROSAC-based robust method
@param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier
(used in the RANSAC and RHO methods only). That is, if
\f[\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|  >  \texttt{ransacReprojThreshold}\f]
then the point \f$i\f$ is considered an outlier. If srcPoints and dstPoints are measured in pixels,
it usually makes sense to set this parameter somewhere in the range of 1 to 10.
@param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input
mask values are ignored.
@param maxIters The maximum number of RANSAC iterations, 2000 is the maximum it can be.
@param confidence Confidence level, between 0 and 1.

The function finds and returns the perspective transformation \f$H\f$ between the source and the
destination planes:

\f[s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\f]

so that the back-projection error

\f[\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\f]

is minimized. If the parameter method is set to the default value 0, the function uses all the point
pairs to compute an initial homography estimate with a simple least-squares scheme.

However, if not all of the point pairs ( \f$srcPoints_i\f$, \f$dstPoints_i\f$ ) fit the rigid perspective
transformation (that is, there are some outliers), this initial estimate will be poor. In this case,
you can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different
random subsets of the corresponding point pairs (of four pairs each), estimate the homography matrix
using this subset and a simple least-square algorithm, and then compute the quality/goodness of the
computed homography (which is the number of inliers for RANSAC or the median re-projection error for
LMeDs). The best subset is then used to produce the initial estimate of the homography matrix and
the mask of inliers/outliers.

Regardless of the method, robust or not, the computed homography matrix is refined further (using
inliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the
re-projection error even more.

The methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers. Finally, if there are no outliers and the
noise is rather small, use the default method (method=0).

The function is used to find initial intrinsic and extrinsic matrices. Homography matrix is
determined up to a scale. Thus, it is normalized so that \f$h_{33}=1\f$. Note that whenever an H matrix
cannot be estimated, an empty one will be returned.

@sa
getAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,
perspectiveTransform


@note
   -   A example on calculating a homography for image matching can be found at
        opencv_source_code/samples/cpp/video_homography.cpp

 */
CV_EXPORTS_W Mat findHomography( InputArray srcPoints, InputArray dstPoints,
                                 int method = 0, double ransacReprojThreshold = 3,
                                 OutputArray mask=noArray(), const int maxIters = 2000,
                                 const double confidence = 0.995);

/** @overload */
CV_EXPORTS Mat findHomography( InputArray srcPoints, InputArray dstPoints,
                               OutputArray mask, int method = 0, double ransacReprojThreshold = 3 );

/** @brief Computes an RQ decomposition of 3x3 matrices.

@param src 3x3 input matrix.
@param mtxR Output 3x3 upper-triangular matrix.
@param mtxQ Output 3x3 orthogonal matrix.
@param Qx Optional output 3x3 rotation matrix around x-axis.
@param Qy Optional output 3x3 rotation matrix around y-axis.
@param Qz Optional output 3x3 rotation matrix around z-axis.

The function computes a RQ decomposition using the given rotations. This function is used in
decomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera
and a rotation matrix.

It optionally returns three rotation matrices, one for each axis, and the three Euler angles in
degrees (as the return value) that could be used in OpenGL. Note, there is always more than one
sequence of rotations about the three principal axes that results in the same orientation of an
object, eg. see @cite Slabaugh . Returned tree rotation matrices and corresponding three Euler angules
are only one of the possible solutions.
 */
CV_EXPORTS_W Vec3d RQDecomp3x3( InputArray src, OutputArray mtxR, OutputArray mtxQ,
                                OutputArray Qx = noArray(),
                                OutputArray Qy = noArray(),
                                OutputArray Qz = noArray());

/** @brief Decomposes a projection matrix into a rotation matrix and a camera matrix.

@param projMatrix 3x4 input projection matrix P.
@param cameraMatrix Output 3x3 camera matrix K.
@param rotMatrix Output 3x3 external rotation matrix R.
@param transVect Output 4x1 translation vector T.
@param rotMatrixX Optional 3x3 rotation matrix around x-axis.
@param rotMatrixY Optional 3x3 rotation matrix around y-axis.
@param rotMatrixZ Optional 3x3 rotation matrix around z-axis.
@param eulerAngles Optional three-element vector containing three Euler angles of rotation in
degrees.

The function computes a decomposition of a projection matrix into a calibration and a rotation
matrix and the position of a camera.

It optionally returns three rotation matrices, one for each axis, and three Euler angles that could
be used in OpenGL. Note, there is always more than one sequence of rotations about the three
principal axes that results in the same orientation of an object, eg. see @cite Slabaugh . Returned
tree rotation matrices and corresponding three Euler angules are only one of the possible solutions.

The function is based on RQDecomp3x3 .
 */
CV_EXPORTS_W void decomposeProjectionMatrix( InputArray projMatrix, OutputArray cameraMatrix,
                                             OutputArray rotMatrix, OutputArray transVect,
                                             OutputArray rotMatrixX = noArray(),
                                             OutputArray rotMatrixY = noArray(),
                                             OutputArray rotMatrixZ = noArray(),
                                             OutputArray eulerAngles =noArray() );

/** @brief Computes partial derivatives of the matrix product for each multiplied matrix.

@param A First multiplied matrix.
@param B Second multiplied matrix.
@param dABdA First output derivative matrix d(A\*B)/dA of size
\f$\texttt{A.rows*B.cols} \times {A.rows*A.cols}\f$ .
@param dABdB Second output derivative matrix d(A\*B)/dB of size
\f$\texttt{A.rows*B.cols} \times {B.rows*B.cols}\f$ .

The function computes partial derivatives of the elements of the matrix product \f$A*B\f$ with regard to
the elements of each of the two input matrices. The function is used to compute the Jacobian
matrices in stereoCalibrate but can also be used in any other similar optimization function.
 */
CV_EXPORTS_W void matMulDeriv( InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB );

/** @brief Combines two rotation-and-shift transformations.

@param rvec1 First rotation vector.
@param tvec1 First translation vector.
@param rvec2 Second rotation vector.
@param tvec2 Second translation vector.
@param rvec3 Output rotation vector of the superposition.
@param tvec3 Output translation vector of the superposition.
@param dr3dr1
@param dr3dt1
@param dr3dr2
@param dr3dt2
@param dt3dr1
@param dt3dt1
@param dt3dr2
@param dt3dt2 Optional output derivatives of rvec3 or tvec3 with regard to rvec1, rvec2, tvec1 and
tvec2, respectively.

The functions compute:

\f[\begin{array}{l} \texttt{rvec3} =  \mathrm{rodrigues} ^{-1} \left ( \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \mathrm{rodrigues} ( \texttt{rvec1} ) \right )  \\ \texttt{tvec3} =  \mathrm{rodrigues} ( \texttt{rvec2} )  \cdot \texttt{tvec1} +  \texttt{tvec2} \end{array} ,\f]

where \f$\mathrm{rodrigues}\f$ denotes a rotation vector to a rotation matrix transformation, and
\f$\mathrm{rodrigues}^{-1}\f$ denotes the inverse transformation. See Rodrigues for details.

Also, the functions can compute the derivatives of the output vectors with regards to the input
vectors (see matMulDeriv ). The functions are used inside stereoCalibrate but can also be used in
your own code where Levenberg-Marquardt or another gradient-based solver is used to optimize a
function that contains a matrix multiplication.
 */
CV_EXPORTS_W void composeRT( InputArray rvec1, InputArray tvec1,
                             InputArray rvec2, InputArray tvec2,
                             OutputArray rvec3, OutputArray tvec3,
                             OutputArray dr3dr1 = noArray(), OutputArray dr3dt1 = noArray(),
                             OutputArray dr3dr2 = noArray(), OutputArray dr3dt2 = noArray(),
                             OutputArray dt3dr1 = noArray(), OutputArray dt3dt1 = noArray(),
                             OutputArray dt3dr2 = noArray(), OutputArray dt3dt2 = noArray() );

/** @brief Projects 3D points to an image plane.

@param objectPoints Array of object points, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel (or
vector\<Point3f\> ), where N is the number of points in the view.
@param rvec Rotation vector. See Rodrigues for details.
@param tvec Translation vector.
@param cameraMatrix Camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.
@param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or
vector\<Point2f\> .
@param jacobian Optional output 2Nx(10+\<numDistCoeffs\>) jacobian matrix of derivatives of image
points with respect to components of the rotation vector, translation vector, focal lengths,
coordinates of the principal point and the distortion coefficients. In the old interface different
components of the jacobian are returned via different output parameters.
@param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the
function assumes that the aspect ratio (*fx/fy*) is fixed and correspondingly adjusts the jacobian
matrix.

The function computes projections of 3D points to the image plane given intrinsic and extrinsic
camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
image points coordinates (as functions of all the input parameters) with respect to the particular
parameters, intrinsic and/or extrinsic. The Jacobians are used during the global optimization in
calibrateCamera, solvePnP, and stereoCalibrate . The function itself can also be used to compute a
re-projection error given the current intrinsic and extrinsic parameters.

@note By setting rvec=tvec=(0,0,0) or by setting cameraMatrix to a 3x3 identity matrix, or by
passing zero distortion coefficients, you can get various useful partial cases of the function. This
means that you can compute the distorted coordinates for a sparse set of points or apply a
perspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.
 */
CV_EXPORTS_W void projectPoints( InputArray objectPoints,
                                 InputArray rvec, InputArray tvec,
                                 InputArray cameraMatrix, InputArray distCoeffs,
                                 OutputArray imagePoints,
                                 OutputArray jacobian = noArray(),
                                 double aspectRatio = 0 );

/** @brief Finds an object pose from 3D-2D point correspondences.

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec , brings points from
the model coordinate system to the camera coordinate system.
@param tvec Output translation vector.
@param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param flags Method for solving a PnP problem:
-   **SOLVEPNP_ITERATIVE** Iterative method is based on Levenberg-Marquardt optimization. In
this case the function finds such a pose that minimizes reprojection error, that is the sum
of squared distances between the observed projections imagePoints and the projected (using
projectPoints ) objectPoints .
-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).
In this case the function requires exactly four object and image points.
-   **SOLVEPNP_AP3P** Method is based on the paper of T. Ke, S. Roumeliotis
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).
In this case the function requires exactly four object and image points.
-   **SOLVEPNP_EPNP** Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the
paper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (@cite lepetit2009epnp).
-   **SOLVEPNP_DLS** Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.
"A Direct Least-Squares (DLS) Method for PnP" (@cite hesch2011direct).
-   **SOLVEPNP_UPNP** Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,
F.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length
Estimation" (@cite penate2013exhaustive). In this case the function also estimates the parameters \f$f_x\f$ and \f$f_y\f$
assuming that both have the same value. Then the cameraMatrix is updated with the estimated
focal length.
-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem". In this case the
function requires exactly four object and image points.

The function estimates the object pose given a set of object points, their corresponding image
projections, as well as the camera matrix and the distortion coefficients.

@note
   -   An example of how to use solvePnP for planar augmented reality can be found at
        opencv_source_code/samples/python/plane_ar.py
   -   If you are using Python:
        - Numpy array slices won't work as input because solvePnP requires contiguous
        arrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of
        modules/calib3d/src/solvepnp.cpp version 2.4.9)
        - The P3P algorithm requires image points to be in an array of shape (N,1,2) due
        to its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)
        which requires 2-channel information.
        - Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of
        it as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =
        np.ascontiguousarray(D[:,:2]).reshape((N,1,2))
   -   The methods **SOLVEPNP_DLS** and **SOLVEPNP_UPNP** cannot be used as the current implementations are
       unstable and sometimes give completly wrong results. If you pass one of these two
       flags, **SOLVEPNP_EPNP** method will be used instead.
   -   The minimum number of points is 4. In the case of **SOLVEPNP_P3P** and **SOLVEPNP_AP3P**
       methods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions
       of the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).
 */
CV_EXPORTS_W bool solvePnP( InputArray objectPoints, InputArray imagePoints,
                            InputArray cameraMatrix, InputArray distCoeffs,
                            OutputArray rvec, OutputArray tvec,
                            bool useExtrinsicGuess = false, int flags = SOLVEPNP_ITERATIVE );

/** @brief Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or
1xN/Nx1 3-channel, where N is the number of points. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,
where N is the number of points. vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvec Output rotation vector (see Rodrigues ) that, together with tvec , brings points from
the model coordinate system to the camera coordinate system.
@param tvec Output translation vector.
@param useExtrinsicGuess Parameter used for SOLVEPNP_ITERATIVE. If true (1), the function uses
the provided rvec and tvec values as initial approximations of the rotation and translation
vectors, respectively, and further optimizes them.
@param iterationsCount Number of iterations.
@param reprojectionError Inlier threshold value used by the RANSAC procedure. The parameter value
is the maximum allowed distance between the observed and computed point projections to consider it
an inlier.
@param confidence The probability that the algorithm produces a useful result.
@param inliers Output vector that contains indices of inliers in objectPoints and imagePoints .
@param flags Method for solving a PnP problem (see solvePnP ).

The function estimates an object pose given a set of object points, their corresponding image
projections, as well as the camera matrix and the distortion coefficients. This function finds such
a pose that minimizes reprojection error, that is, the sum of squared distances between the observed
projections imagePoints and the projected (using projectPoints ) objectPoints. The use of RANSAC
makes the function resistant to outliers.

@note
   -   An example of how to use solvePNPRansac for object detection can be found at
        opencv_source_code/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/
   -   The default method used to estimate the camera pose for the Minimal Sample Sets step
       is #SOLVEPNP_EPNP. Exceptions are:
         - if you choose #SOLVEPNP_P3P or #SOLVEPNP_AP3P, these methods will be used.
         - if the number of input points is equal to 4, #SOLVEPNP_P3P is used.
   -   The method used to estimate the camera pose using all the inliers is defined by the
       flags parameters unless it is equal to #SOLVEPNP_P3P or #SOLVEPNP_AP3P. In this case,
       the method #SOLVEPNP_EPNP will be used instead.
 */
CV_EXPORTS_W bool solvePnPRansac( InputArray objectPoints, InputArray imagePoints,
                                  InputArray cameraMatrix, InputArray distCoeffs,
                                  OutputArray rvec, OutputArray tvec,
                                  bool useExtrinsicGuess = false, int iterationsCount = 100,
                                  float reprojectionError = 8.0, double confidence = 0.99,
                                  OutputArray inliers = noArray(), int flags = SOLVEPNP_ITERATIVE );
/** @brief Finds an object pose from 3 3D-2D point correspondences.

@param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or
1x3/3x1 3-channel. vector\<Point3f\> can be also passed here.
@param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.
 vector\<Point2f\> can be also passed here.
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param rvecs Output rotation vectors (see Rodrigues ) that, together with tvecs , brings points from
the model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.
@param tvecs Output translation vectors.
@param flags Method for solving a P3P problem:
-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang
"Complete Solution Classification for the Perspective-Three-Point Problem".
-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.
"An Efficient Algebraic Solution to the Perspective-Three-Point Problem".

The function estimates the object pose given 3 object points, their corresponding image
projections, as well as the camera matrix and the distortion coefficients.
 */
CV_EXPORTS_W int solveP3P( InputArray objectPoints, InputArray imagePoints,
                           InputArray cameraMatrix, InputArray distCoeffs,
                           OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                           int flags );

/** @brief Finds an initial camera matrix from 3D-2D point correspondences.

@param objectPoints Vector of vectors of the calibration pattern points in the calibration pattern
coordinate space. In the old interface all the per-view vectors are concatenated. See
calibrateCamera for details.
@param imagePoints Vector of vectors of the projections of the calibration pattern points. In the
old interface all the per-view vectors are concatenated.
@param imageSize Image size in pixels used to initialize the principal point.
@param aspectRatio If it is zero or negative, both \f$f_x\f$ and \f$f_y\f$ are estimated independently.
Otherwise, \f$f_x = f_y * \texttt{aspectRatio}\f$ .

The function estimates and returns an initial camera matrix for the camera calibration process.
Currently, the function only supports planar calibration patterns, which are patterns where each
object point has z-coordinate =0.
 */
CV_EXPORTS_W Mat initCameraMatrix2D( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints,
                                     Size imageSize, double aspectRatio = 1.0 );

/** @brief Finds the positions of internal corners of the chessboard.

@param image Source chessboard view. It must be an 8-bit grayscale or color image.
@param patternSize Number of inner corners per a chessboard row and column
( patternSize = cvSize(points_per_row,points_per_colum) = cvSize(columns,rows) ).
@param corners Output array of detected corners.
@param flags Various operation flags that can be zero or a combination of the following values:
-   **CALIB_CB_ADAPTIVE_THRESH** Use adaptive thresholding to convert the image to black
and white, rather than a fixed threshold level (computed from the average image brightness).
-   **CALIB_CB_NORMALIZE_IMAGE** Normalize the image gamma with equalizeHist before
applying fixed or adaptive thresholding.
-   **CALIB_CB_FILTER_QUADS** Use additional criteria (like contour area, perimeter,
square-like shape) to filter out false quads extracted at the contour retrieval stage.
-   **CALIB_CB_FAST_CHECK** Run a fast check on the image that looks for chessboard corners,
and shortcut the call if none is found. This can drastically speed up the call in the
degenerate condition when no chessboard is observed.

The function attempts to determine whether the input image is a view of the chessboard pattern and
locate the internal chessboard corners. The function returns a non-zero value if all of the corners
are found and they are placed in a certain order (row by row, left to right in every row).
Otherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,
a regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black
squares touch each other. The detected coordinates are approximate, and to determine their positions
more accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with
different parameters if returned coordinates are not accurate enough.

Sample usage of detecting and drawing chessboard corners: :
@code
    Size patternsize(8,6); //interior number of corners
    Mat gray = ....; //source image
    vector<Point2f> corners; //this will be filled by the detected corners

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    if(patternfound)
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

    drawChessboardCorners(img, patternsize, Mat(corners), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments. Otherwise, if there is no
border and the background is dark, the outer black squares cannot be segmented properly and so the
square grouping and ordering algorithm fails.
 */
CV_EXPORTS_W bool findChessboardCorners( InputArray image, Size patternSize, OutputArray corners,
                                         int flags = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE );

//! finds subpixel-accurate positions of the chessboard corners
CV_EXPORTS bool find4QuadCornerSubpix( InputArray img, InputOutputArray corners, Size region_size );

/** @brief Renders the detected chessboard corners.

@param image Destination image. It must be an 8-bit color image.
@param patternSize Number of inner corners per a chessboard row and column
(patternSize = cv::Size(points_per_row,points_per_column)).
@param corners Array of detected corners, the output of findChessboardCorners.
@param patternWasFound Parameter indicating whether the complete board was found or not. The
return value of findChessboardCorners should be passed here.

The function draws individual chessboard corners detected either as red circles if the board was not
found, or as colored corners connected with lines if the board was found.
 */
CV_EXPORTS_W void drawChessboardCorners( InputOutputArray image, Size patternSize,
                                         InputArray corners, bool patternWasFound );

struct CV_EXPORTS_W_SIMPLE CirclesGridFinderParameters
{
    CV_WRAP CirclesGridFinderParameters();
    CV_PROP_RW cv::Size2f densityNeighborhoodSize;
    CV_PROP_RW float minDensity;
    CV_PROP_RW int kmeansAttempts;
    CV_PROP_RW int minDistanceToAddKeypoint;
    CV_PROP_RW int keypointScale;
    CV_PROP_RW float minGraphConfidence;
    CV_PROP_RW float vertexGain;
    CV_PROP_RW float vertexPenalty;
    CV_PROP_RW float existingVertexGain;
    CV_PROP_RW float edgeGain;
    CV_PROP_RW float edgePenalty;
    CV_PROP_RW float convexHullFactor;
    CV_PROP_RW float minRNGEdgeSwitchDist;

    enum GridType
    {
      SYMMETRIC_GRID, ASYMMETRIC_GRID
    };
    GridType gridType;
};

/** @brief Finds centers in the grid of circles.

@param image grid view of input circles; it must be an 8-bit grayscale or color image.
@param patternSize number of circles per row and column
( patternSize = Size(points_per_row, points_per_colum) ).
@param centers output array of detected centers.
@param flags various operation flags that can be one of the following values:
-   **CALIB_CB_SYMMETRIC_GRID** uses symmetric pattern of circles.
-   **CALIB_CB_ASYMMETRIC_GRID** uses asymmetric pattern of circles.
-   **CALIB_CB_CLUSTERING** uses a special algorithm for grid detection. It is more robust to
perspective distortions but much more sensitive to background clutter.
@param blobDetector feature detector that finds blobs like dark circles on light background.
@param parameters struct for finding circles in a grid pattern.

The function attempts to determine whether the input image contains a grid of circles. If it is, the
function locates centers of the circles. The function returns a non-zero value if all of the centers
have been found and they have been placed in a certain order (row by row, left to right in every
row). Otherwise, if the function fails to find all the corners or reorder them, it returns 0.

Sample usage of detecting and drawing the centers of circles: :
@code
    Size patternsize(7,7); //number of centers
    Mat gray = ....; //source image
    vector<Point2f> centers; //this will be filled by the detected centers

    bool patternfound = findCirclesGrid(gray, patternsize, centers);

    drawChessboardCorners(img, patternsize, Mat(centers), patternfound);
@endcode
@note The function requires white space (like a square-thick border, the wider the better) around
the board to make the detection more robust in various environments.
 */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags,
                                   const Ptr<FeatureDetector> &blobDetector,
                                   CirclesGridFinderParameters parameters);

/** @overload */
CV_EXPORTS_W bool findCirclesGrid( InputArray image, Size patternSize,
                                   OutputArray centers, int flags = CALIB_CB_SYMMETRIC_GRID,
                                   const Ptr<FeatureDetector> &blobDetector = SimpleBlobDetector::create());

/** @brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

@param objectPoints In the new interface it is a vector of vectors of calibration pattern points in
the calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer
vector contains as many elements as the number of the pattern views. If the same calibration pattern
is shown in each view and it is fully visible, all the vectors will be the same. Although, it is
possible to use partially occluded patterns, or even different patterns in different views. Then,
the vectors will be different. The points are 3D, but since they are in a pattern coordinate system,
then, if the rig is planar, it may make sense to put the model to a XY coordinate plane so that
Z-coordinate of each input object point is 0.
In the old interface all the vectors of object points from different views are concatenated
together.
@param imagePoints In the new interface it is a vector of vectors of the projections of calibration
pattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and
objectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.
In the old interface all the vectors of object points from different views are concatenated
together.
@param imageSize Size of the image used only to initialize the intrinsic camera matrix.
@param cameraMatrix Output 3x3 floating-point camera matrix
\f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
and/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
initialized before calling the function.
@param distCoeffs Output vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements.
@param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view
(e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
k-th translation vector (see the next output parameter description) brings the calibration pattern
from the model coordinate space (in which object points are specified) to the world coordinate
space, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
@param tvecs Output vector of translation vectors estimated for each pattern view.
@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
 Order of deviations values:
\f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
 s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
 Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
 \f$R_i, T_i\f$ are concatenated 1x3 vectors.
 @param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.
@param flags Different flags that may be zero or a combination of the following values:
-   **CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid initial values of
fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
center ( imageSize is used), and focal distances are computed in a least-squares fashion.
Note, that if intrinsic parameters are known, there is no need to use this function just to
estimate extrinsic parameters. Use solvePnP instead.
-   **CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when
CALIB_USE_INTRINSIC_GUESS is set too.
-   **CALIB_FIX_ASPECT_RATIO** The functions considers only fy as a free parameter. The
ratio fx/fy stays the same as in the input cameraMatrix . When
CALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are
ignored, only their ratio is computed and used further.
-   **CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients \f$(p_1, p_2)\f$ are set
to zeros and stay zero.
-   **CALIB_FIX_K1,...,CALIB_FIX_K6** The corresponding radial distortion
coefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is
set, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_RATIONAL_MODEL** Coefficients k4, k5, and k6 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the rational model and return 8 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion coefficients are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted sensor model are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

@return the overall RMS re-projection error.

The function estimates the intrinsic camera parameters and extrinsic parameters for each of the
views. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object
points and their corresponding 2D projections in each view must be specified. That may be achieved
by using an object with a known geometry and easily detectable feature points. Such an object is
called a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as
a calibration rig (see findChessboardCorners ). Currently, initialization of intrinsic parameters
(when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration
patterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also
be used as long as initial cameraMatrix is provided.

The algorithm performs the following steps:

-   Compute the initial intrinsic parameters (the option only available for planar calibration
    patterns) or read them from the input parameters. The distortion coefficients are all set to
    zeros initially unless some of CALIB_FIX_K? are specified.

-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is
    done using solvePnP .

-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,
    that is, the total sum of squared distances between the observed feature points imagePoints and
    the projected (using the current estimates for camera parameters and the poses) object points
    objectPoints. See projectPoints for details.

@note
   If you use a non-square (=non-NxN) grid and findChessboardCorners for calibration, and
    calibrateCamera returns bad values (zero distortion coefficients, an image center very far from
    (w/2-0.5,h/2-0.5), and/or large differences between \f$f_x\f$ and \f$f_y\f$ (ratios of 10:1 or more)),
    then you have probably used patternSize=cvSize(rows,cols) instead of using
    patternSize=cvSize(cols,rows) in findChessboardCorners .

@sa
   findChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort
 */
CV_EXPORTS_AS(calibrateCameraExtended) double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviationsIntrinsics,
                                     OutputArray stdDeviationsExtrinsics,
                                     OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @overload double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     OutputArray stdDeviations, OutputArray perViewErrors,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) )
 */
CV_EXPORTS_W double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

/** @brief Computes useful camera characteristics from the camera matrix.

@param cameraMatrix Input camera matrix that can be estimated by calibrateCamera or
stereoCalibrate .
@param imageSize Input image size in pixels.
@param apertureWidth Physical width in mm of the sensor.
@param apertureHeight Physical height in mm of the sensor.
@param fovx Output field of view in degrees along the horizontal sensor axis.
@param fovy Output field of view in degrees along the vertical sensor axis.
@param focalLength Focal length of the lens in mm.
@param principalPoint Principal point in mm.
@param aspectRatio \f$f_y/f_x\f$

The function computes various useful camera characteristics from the previously estimated camera
matrix.

@note
   Do keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for
    the chessboard pitch (it can thus be any value).
 */
CV_EXPORTS_W void calibrationMatrixValues( InputArray cameraMatrix, Size imageSize,
                                           double apertureWidth, double apertureHeight,
                                           CV_OUT double& fovx, CV_OUT double& fovy,
                                           CV_OUT double& focalLength, CV_OUT Point2d& principalPoint,
                                           CV_OUT double& aspectRatio );

/** @brief Calibrates the stereo camera.

@param objectPoints Vector of vectors of the calibration pattern points.
@param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
observed by the first camera.
@param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
observed by the second camera.
@param cameraMatrix1 Input/output first camera matrix:
\f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
any of CALIB_USE_INTRINSIC_GUESS , CALIB_FIX_ASPECT_RATIO ,
CALIB_FIX_INTRINSIC , or CALIB_FIX_FOCAL_LENGTH are specified, some or all of the
matrix components must be initialized. See the flags description for details.
@param distCoeffs1 Input/output vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. The output vector length depends on the flags.
@param cameraMatrix2 Input/output second camera matrix. The parameter is similar to cameraMatrix1
@param distCoeffs2 Input/output lens distortion coefficients for the second camera. The parameter
is similar to distCoeffs1 .
@param imageSize Size of the image used only to initialize intrinsic camera matrix.
@param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
@param T Output translation vector between the coordinate systems of the cameras.
@param E Output essential matrix.
@param F Output fundamental matrix.
@param flags Different flags that may be zero or a combination of the following values:
-   **CALIB_FIX_INTRINSIC** Fix cameraMatrix? and distCoeffs? so that only R, T, E , and F
matrices are estimated.
-   **CALIB_USE_INTRINSIC_GUESS** Optimize some or all of the intrinsic parameters
according to the specified flags. Initial values are provided by the user.
-   **CALIB_FIX_PRINCIPAL_POINT** Fix the principal points during the optimization.
-   **CALIB_FIX_FOCAL_LENGTH** Fix \f$f^{(j)}_x\f$ and \f$f^{(j)}_y\f$ .
-   **CALIB_FIX_ASPECT_RATIO** Optimize \f$f^{(j)}_y\f$ . Fix the ratio \f$f^{(j)}_x/f^{(j)}_y\f$
.
-   **CALIB_SAME_FOCAL_LENGTH** Enforce \f$f^{(0)}_x=f^{(1)}_x\f$ and \f$f^{(0)}_y=f^{(1)}_y\f$ .
-   **CALIB_ZERO_TANGENT_DIST** Set tangential distortion coefficients for each camera to
zeros and fix there.
-   **CALIB_FIX_K1,...,CALIB_FIX_K6** Do not change the corresponding radial
distortion coefficient during the optimization. If CALIB_USE_INTRINSIC_GUESS is set,
the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_RATIONAL_MODEL** Enable coefficients k4, k5, and k6. To provide the backward
compatibility, this extra flag should be explicitly specified to make the calibration
function use the rational model and return 8 coefficients. If the flag is not set, the
function computes and returns only 5 distortion coefficients.
-   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4 are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the thin prism model and return 12 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion coefficients are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
-   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are enabled. To provide the
backward compatibility, this extra flag should be explicitly specified to make the
calibration function use the tilted sensor model and return 14 coefficients. If the flag is not
set, the function computes and returns only 5 distortion coefficients.
-   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted sensor model are not changed during
the optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
supplied distCoeffs matrix is used. Otherwise, it is set to 0.
@param criteria Termination criteria for the iterative optimization algorithm.

The function estimates transformation between two cameras making a stereo pair. If you have a stereo
camera where the relative position and orientation of two cameras is fixed, and if you computed
poses of an object relative to the first camera and to the second camera, (R1, T1) and (R2, T2),
respectively (this can be done with solvePnP ), then those poses definitely relate to each other.
This means that, given ( \f$R_1\f$,\f$T_1\f$ ), it should be possible to compute ( \f$R_2\f$,\f$T_2\f$ ). You only
need to know the position and orientation of the second camera relative to the first camera. This is
what the described function does. It computes ( \f$R\f$,\f$T\f$ ) so that:

\f[R_2=R*R_1\f]
\f[T_2=R*T_1 + T,\f]

Optionally, it computes the essential matrix E:

\f[E= \vecthreethree{0}{-T_2}{T_1}{T_2}{0}{-T_0}{-T_1}{T_0}{0} *R\f]

where \f$T_i\f$ are components of the translation vector \f$T\f$ : \f$T=[T_0, T_1, T_2]^T\f$ . And the function
can also compute the fundamental matrix F:

\f[F = cameraMatrix2^{-T} E cameraMatrix1^{-1}\f]

Besides the stereo-related information, the function can also perform a full calibration of each of
two cameras. However, due to the high dimensionality of the parameter space and noise in the input
data, the function can diverge from the correct solution. If the intrinsic parameters can be
estimated with high accuracy for each of the cameras individually (for example, using
calibrateCamera ), you are recommended to do so and then pass CALIB_FIX_INTRINSIC flag to the
function along with the computed intrinsic parameters. Otherwise, if all the parameters are
estimated at once, it makes sense to restrict some parameters, for example, pass
CALIB_SAME_FOCAL_LENGTH and CALIB_ZERO_TANGENT_DIST flags, which is usually a
reasonable assumption.

Similarly to calibrateCamera , the function minimizes the total re-projection error for all the
points in all the available views from both cameras. The function returns the final value of the
re-projection error.
 */
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, OutputArray R,OutputArray T, OutputArray E, OutputArray F,
                                     int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );


/** @brief Computes rectification transforms for each head of a calibrated stereo camera.

@param cameraMatrix1 First camera matrix.
@param distCoeffs1 First camera distortion parameters.
@param cameraMatrix2 Second camera matrix.
@param distCoeffs2 Second camera distortion parameters.
@param imageSize Size of the image used for stereo calibration.
@param R Rotation matrix between the coordinate systems of the first and the second cameras.
@param T Translation vector between coordinate systems of the cameras.
@param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
@param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
@param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
camera.
@param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
camera.
@param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
@param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
the function makes the principal points of each camera have the same pixel coordinates in the
rectified views. And if the flag is not set, the function may still shift the images in the
horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
useful image area.
@param alpha Free scaling parameter. If it is -1 or absent, the function performs the default
scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified
images are zoomed and shifted so that only valid pixels are visible (no black areas after
rectification). alpha=1 means that the rectified image is decimated and shifted so that all the
pixels from the original images from the cameras are retained in the rectified images (no source
image pixels are lost). Obviously, any intermediate value yields an intermediate result between
those two extreme cases.
@param newImageSize New image resolution after rectification. The same size should be passed to
initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
is passed (default), it is set to the original imageSize . Setting it to larger value can help you
preserve details in the original image, especially when there is a big radial distortion.
@param validPixROI1 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).
@param validPixROI2 Optional output rectangles inside the rectified images where all the pixels
are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller
(see the picture below).

The function computes the rotation matrices for each camera that (virtually) make both camera image
planes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies
the dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate
as input. As output, it provides two rotation matrices and also two projection matrices in the new
coordinates. The function distinguishes the following two cases:

-   **Horizontal stereo**: the first and the second camera views are shifted relative to each other
    mainly along the x axis (with possible small vertical shift). In the rectified images, the
    corresponding epipolar lines in the left and right cameras are horizontal and have the same
    y-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix} f & 0 & cx_1 & 0 \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix} f & 0 & cx_2 & T_x*f \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} ,\f]

    where \f$T_x\f$ is a horizontal shift between the cameras and \f$cx_1=cx_2\f$ if
    CALIB_ZERO_DISPARITY is set.

-   **Vertical stereo**: the first and the second camera views are shifted relative to each other
    mainly in vertical direction (and probably a bit in the horizontal direction too). The epipolar
    lines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:

    \f[\texttt{P1} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_1 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}\f]

    \f[\texttt{P2} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_2 & T_y*f \\ 0 & 0 & 1 & 0 \end{bmatrix} ,\f]

    where \f$T_y\f$ is a vertical shift between the cameras and \f$cy_1=cy_2\f$ if CALIB_ZERO_DISPARITY is
    set.

As you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera
matrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to
initialize the rectification map for each camera.

See below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through
the corresponding image regions. This means that the images are well rectified, which is what most
stereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that
their interiors are all valid pixels.

![image](pics/stereo_undistort.jpg)
 */
CV_EXPORTS_W void stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 Size imageSize, InputArray R, InputArray T,
                                 OutputArray R1, OutputArray R2,
                                 OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags = CALIB_ZERO_DISPARITY,
                                 double alpha = -1, Size newImageSize = Size(),
                                 CV_OUT Rect* validPixROI1 = 0, CV_OUT Rect* validPixROI2 = 0 );

/** @brief Computes a rectification transform for an uncalibrated stereo camera.

@param points1 Array of feature points in the first image.
@param points2 The corresponding points in the second image. The same formats as in
findFundamentalMat are supported.
@param F Input fundamental matrix. It can be computed from the same set of point pairs using
findFundamentalMat .
@param imgSize Size of the image.
@param H1 Output rectification homography matrix for the first image.
@param H2 Output rectification homography matrix for the second image.
@param threshold Optional threshold used to filter out the outliers. If the parameter is greater
than zero, all the point pairs that do not comply with the epipolar geometry (that is, the points
for which \f$|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|>\texttt{threshold}\f$ ) are
rejected prior to computing the homographies. Otherwise,all the points are considered inliers.

The function computes the rectification transformations without knowing intrinsic parameters of the
cameras and their relative position in the space, which explains the suffix "uncalibrated". Another
related difference from stereoRectify is that the function outputs not the rectification
transformations in the object (3D) space, but the planar perspective transformations encoded by the
homography matrices H1 and H2 . The function implements the algorithm @cite Hartley99 .

@note
   While the algorithm does not need to know the intrinsic parameters of the cameras, it heavily
    depends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion,
    it would be better to correct it before computing the fundamental matrix and calling this
    function. For example, distortion coefficients can be estimated for each head of stereo camera
    separately by using calibrateCamera . Then, the images can be corrected using undistort , or
    just the point coordinates can be corrected with undistortPoints .
 */
CV_EXPORTS_W bool stereoRectifyUncalibrated( InputArray points1, InputArray points2,
                                             InputArray F, Size imgSize,
                                             OutputArray H1, OutputArray H2,
                                             double threshold = 5 );

//! computes the rectification transformations for 3-head camera, where all the heads are on the same line.
CV_EXPORTS_W float rectify3Collinear( InputArray cameraMatrix1, InputArray distCoeffs1,
                                      InputArray cameraMatrix2, InputArray distCoeffs2,
                                      InputArray cameraMatrix3, InputArray distCoeffs3,
                                      InputArrayOfArrays imgpt1, InputArrayOfArrays imgpt3,
                                      Size imageSize, InputArray R12, InputArray T12,
                                      InputArray R13, InputArray T13,
                                      OutputArray R1, OutputArray R2, OutputArray R3,
                                      OutputArray P1, OutputArray P2, OutputArray P3,
                                      OutputArray Q, double alpha, Size newImgSize,
                                      CV_OUT Rect* roi1, CV_OUT Rect* roi2, int flags );

/** @brief Returns the new camera matrix based on the free scaling parameter.

@param cameraMatrix Input camera matrix.
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of
4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are
assumed.
@param imageSize Original image size.
@param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are
valid) and 1 (when all the source image pixels are retained in the undistorted image). See
stereoRectify for details.
@param newImgSize Image size after rectification. By default,it is set to imageSize .
@param validPixROI Optional output rectangle that outlines all-good-pixels region in the
undistorted image. See roi1, roi2 description in stereoRectify .
@param centerPrincipalPoint Optional flag that indicates whether in the new camera matrix the
principal point should be at the image center or not. By default, the principal point is chosen to
best fit a subset of the source image (determined by alpha) to the corrected image.
@return new_camera_matrix Output new camera matrix.

The function computes and returns the optimal new camera matrix based on the free scaling parameter.
By varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original
image pixels if there is valuable information in the corners alpha=1 , or get something in between.
When alpha\>0 , the undistortion result is likely to have some black pixels corresponding to
"virtual" pixels outside of the captured distorted image. The original camera matrix, distortion
coefficients, the computed new camera matrix, and newImageSize should be passed to
initUndistortRectifyMap to produce the maps for remap .
 */
CV_EXPORTS_W Mat getOptimalNewCameraMatrix( InputArray cameraMatrix, InputArray distCoeffs,
                                            Size imageSize, double alpha, Size newImgSize = Size(),
                                            CV_OUT Rect* validPixROI = 0,
                                            bool centerPrincipalPoint = false);

/** @brief Converts points from Euclidean to homogeneous space.

@param src Input vector of N-dimensional points.
@param dst Output vector of N+1-dimensional points.

The function converts points from Euclidean to homogeneous space by appending 1's to the tuple of
point coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).
 */
CV_EXPORTS_W void convertPointsToHomogeneous( InputArray src, OutputArray dst );

/** @brief Converts points from homogeneous to Euclidean space.

@param src Input vector of N-dimensional points.
@param dst Output vector of N-1-dimensional points.

The function converts points homogeneous to Euclidean space using perspective projection. That is,
each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the
output point coordinates will be (0,0,0,...).
 */
CV_EXPORTS_W void convertPointsFromHomogeneous( InputArray src, OutputArray dst );

/** @brief Converts points to/from homogeneous coordinates.

@param src Input array or vector of 2D, 3D, or 4D points.
@param dst Output vector of 2D, 3D, or 4D points.

The function converts 2D or 3D points from/to homogeneous coordinates by calling either
convertPointsToHomogeneous or convertPointsFromHomogeneous.

@note The function is obsolete. Use one of the previous two functions instead.
 */
CV_EXPORTS void convertPointsHomogeneous( InputArray src, OutputArray dst );

/** @brief Calculates a fundamental matrix from the corresponding points in two images.

@param points1 Array of N points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param method Method for computing a fundamental matrix.
-   **CV_FM_7POINT** for a 7-point algorithm. \f$N = 7\f$
-   **CV_FM_8POINT** for an 8-point algorithm. \f$N \ge 8\f$
-   **CV_FM_RANSAC** for the RANSAC algorithm. \f$N \ge 8\f$
-   **CV_FM_LMEDS** for the LMedS algorithm. \f$N \ge 8\f$
@param param1 Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param param2 Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level
of confidence (probability) that the estimated matrix is correct.
@param mask

The epipolar geometry is described by the following equation:

\f[[p_2; 1]^T F [p_1; 1] = 0\f]

where \f$F\f$ is a fundamental matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the
second images, respectively.

The function calculates the fundamental matrix using one of four methods listed above and returns
the found fundamental matrix. Normally just one matrix is found. But in case of the 7-point
algorithm, the function may return up to 3 solutions ( \f$9 \times 3\f$ matrix that stores all 3
matrices sequentially).

The calculated fundamental matrix may be passed further to computeCorrespondEpilines that finds the
epipolar lines corresponding to the specified points. It can also be passed to
stereoRectifyUncalibrated to compute the rectification transformation. :
@code
    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    Mat fundamental_matrix =
     findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
@endcode
 */
CV_EXPORTS_W Mat findFundamentalMat( InputArray points1, InputArray points2,
                                     int method = FM_RANSAC,
                                     double param1 = 3., double param2 = 0.99,
                                     OutputArray mask = noArray() );

/** @overload */
CV_EXPORTS Mat findFundamentalMat( InputArray points1, InputArray points2,
                                   OutputArray mask, int method = FM_RANSAC,
                                   double param1 = 3., double param2 = 0.99 );

/** @brief Calculates an essential matrix from the corresponding points in two images.

@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should
be floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera matrix.
@param method Method for computing a fundamental matrix.
-   **RANSAC** for the RANSAC algorithm.
-   **MEDS** for the LMedS algorithm.
@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
confidence (probability) that the estimated matrix is correct.
@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
for the other points. The array is computed only in the RANSAC and LMedS methods.

This function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 .
@cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation:

\f[[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\f]

where \f$E\f$ is an essential matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the
second images, respectively. The result of this function may be passed further to
decomposeEssentialMat or recoverPose to recover the relative pose between cameras.
 */
CV_EXPORTS_W Mat findEssentialMat( InputArray points1, InputArray points2,
                                 InputArray cameraMatrix, int method = RANSAC,
                                 double prob = 0.999, double threshold = 1.0,
                                 OutputArray mask = noArray() );

/** @overload
@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should
be floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param focal focal length of the camera. Note that this function assumes that points1 and points2
are feature points from cameras with same focal length and principal point.
@param pp principal point of the camera.
@param method Method for computing a fundamental matrix.
-   **RANSAC** for the RANSAC algorithm.
-   **LMEDS** for the LMedS algorithm.
@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar
line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the
point localization, image resolution, and the image noise.
@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of
confidence (probability) that the estimated matrix is correct.
@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1
for the other points. The array is computed only in the RANSAC and LMedS methods.

This function differs from the one above that it computes camera matrix from focal length and
principal point:

\f[K =
\begin{bmatrix}
f & 0 & x_{pp}  \\
0 & f & y_{pp}  \\
0 & 0 & 1
\end{bmatrix}\f]
 */
CV_EXPORTS_W Mat findEssentialMat( InputArray points1, InputArray points2,
                                 double focal = 1.0, Point2d pp = Point2d(0, 0),
                                 int method = RANSAC, double prob = 0.999,
                                 double threshold = 1.0, OutputArray mask = noArray() );

/** @brief Decompose an essential matrix to possible rotations and translation.

@param E The input essential matrix.
@param R1 One possible rotation matrix.
@param R2 Another possible rotation matrix.
@param t One possible translation.

This function decompose an essential matrix E using svd decomposition @cite HartleyZ00 . Generally 4
possible poses exists for a given E. They are \f$[R_1, t]\f$, \f$[R_1, -t]\f$, \f$[R_2, t]\f$, \f$[R_2, -t]\f$. By
decomposing E, you can only get the direction of the translation, so the function returns unit t.
 */
CV_EXPORTS_W void decomposeEssentialMat( InputArray E, OutputArray R1, OutputArray R2, OutputArray t );

/** @brief Recover relative camera rotation and translation from an estimated essential matrix and the
corresponding points in two images, using cheirality check. Returns the number of inliers which pass
the check.

@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera matrix.
@param R Recovered relative rotation.
@param t Recoverd relative translation.
@param mask Input/output mask for inliers in points1 and points2.
:   If it is not empty, then it marks inliers in points1 and points2 for then given essential
matrix E. Only these inliers will be used to recover pose. In the output mask only inliers
which pass the cheirality check.
This function decomposes an essential matrix using decomposeEssentialMat and then verifies possible
pose hypotheses by doing cheirality check. The cheirality check basically means that the
triangulated 3D points should have positive depth. Some details can be found in @cite Nister03 .

This function can be used to process output E and mask from findEssentialMat. In this scenario,
points1 and points2 are the same input for findEssentialMat. :
@code
    // Example. Estimation of fundamental matrix using the RANSAC algorithm
    int point_count = 100;
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = ...;
        points2[i] = ...;
    }

    // cametra matrix with both focal lengths = 1, and principal point = (0, 0)
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);

    Mat E, R, t, mask;

    E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
@endcode
 */
CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            InputArray cameraMatrix, OutputArray R, OutputArray t,
                            InputOutputArray mask = noArray() );

/** @overload
@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1 .
@param R Recovered relative rotation.
@param t Recoverd relative translation.
@param focal Focal length of the camera. Note that this function assumes that points1 and points2
are feature points from cameras with same focal length and principal point.
@param pp principal point of the camera.
@param mask Input/output mask for inliers in points1 and points2.
:   If it is not empty, then it marks inliers in points1 and points2 for then given essential
matrix E. Only these inliers will be used to recover pose. In the output mask only inliers
which pass the cheirality check.

This function differs from the one above that it computes camera matrix from focal length and
principal point:

\f[K =
\begin{bmatrix}
f & 0 & x_{pp}  \\
0 & f & y_{pp}  \\
0 & 0 & 1
\end{bmatrix}\f]
 */
CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            OutputArray R, OutputArray t,
                            double focal = 1.0, Point2d pp = Point2d(0, 0),
                            InputOutputArray mask = noArray() );

/** @overload
@param E The input essential matrix.
@param points1 Array of N 2D points from the first image. The point coordinates should be
floating-point (single or double precision).
@param points2 Array of the second image points of the same size and format as points1.
@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
Note that this function assumes that points1 and points2 are feature points from cameras with the
same camera matrix.
@param R Recovered relative rotation.
@param t Recoverd relative translation.
@param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite points).
@param mask Input/output mask for inliers in points1 and points2.
:   If it is not empty, then it marks inliers in points1 and points2 for then given essential
matrix E. Only these inliers will be used to recover pose. In the output mask only inliers
which pass the cheirality check.
@param triangulatedPoints 3d points which were reconstructed by triangulation.
 */

CV_EXPORTS_W int recoverPose( InputArray E, InputArray points1, InputArray points2,
                            InputArray cameraMatrix, OutputArray R, OutputArray t, double distanceThresh, InputOutputArray mask = noArray(),
                            OutputArray triangulatedPoints = noArray());

/** @brief For points in an image of a stereo pair, computes the corresponding epilines in the other image.

@param points Input points. \f$N \times 1\f$ or \f$1 \times N\f$ matrix of type CV_32FC2 or
vector\<Point2f\> .
@param whichImage Index of the image (1 or 2) that contains the points .
@param F Fundamental matrix that can be estimated using findFundamentalMat or stereoRectify .
@param lines Output vector of the epipolar lines corresponding to the points in the other image.
Each line \f$ax + by + c=0\f$ is encoded by 3 numbers \f$(a, b, c)\f$ .

For every point in one of the two images of a stereo pair, the function finds the equation of the
corresponding epipolar line in the other image.

From the fundamental matrix definition (see findFundamentalMat ), line \f$l^{(2)}_i\f$ in the second
image for the point \f$p^{(1)}_i\f$ in the first image (when whichImage=1 ) is computed as:

\f[l^{(2)}_i = F p^{(1)}_i\f]

And vice versa, when whichImage=2, \f$l^{(1)}_i\f$ is computed from \f$p^{(2)}_i\f$ as:

\f[l^{(1)}_i = F^T p^{(2)}_i\f]

Line coefficients are defined up to a scale. They are normalized so that \f$a_i^2+b_i^2=1\f$ .
 */
CV_EXPORTS_W void computeCorrespondEpilines( InputArray points, int whichImage,
                                             InputArray F, OutputArray lines );

/** @brief Reconstructs points by triangulation.

@param projMatr1 3x4 projection matrix of the first camera.
@param projMatr2 3x4 projection matrix of the second camera.
@param projPoints1 2xN array of feature points in the first image. In case of c++ version it can
be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
@param projPoints2 2xN array of corresponding points in the second image. In case of c++ version
it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
@param points4D 4xN array of reconstructed points in homogeneous coordinates.

The function reconstructs 3-dimensional points (in homogeneous coordinates) by using their
observations with a stereo camera. Projections matrices can be obtained from stereoRectify.

@note
   Keep in mind that all input data should be of float type in order for this function to work.

@sa
   reprojectImageTo3D
 */
CV_EXPORTS_W void triangulatePoints( InputArray projMatr1, InputArray projMatr2,
                                     InputArray projPoints1, InputArray projPoints2,
                                     OutputArray points4D );

/** @brief Refines coordinates of corresponding points.

@param F 3x3 fundamental matrix.
@param points1 1xN array containing the first set of points.
@param points2 1xN array containing the second set of points.
@param newPoints1 The optimized points1.
@param newPoints2 The optimized points2.

The function implements the Optimal Triangulation Method (see Multiple View Geometry for details).
For each given point correspondence points1[i] \<-\> points2[i], and a fundamental matrix F, it
computes the corrected correspondences newPoints1[i] \<-\> newPoints2[i] that minimize the geometric
error \f$d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2\f$ (where \f$d(a,b)\f$ is the
geometric distance between points \f$a\f$ and \f$b\f$ ) subject to the epipolar constraint
\f$newPoints2^T * F * newPoints1 = 0\f$ .
 */
CV_EXPORTS_W void correctMatches( InputArray F, InputArray points1, InputArray points2,
                                  OutputArray newPoints1, OutputArray newPoints2 );

/** @brief Filters off small noise blobs (speckles) in the disparity map

@param img The input 16-bit signed disparity image
@param newVal The disparity value used to paint-off the speckles
@param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not
affected by the algorithm
@param maxDiff Maximum difference between neighbor disparity pixels to put them into the same
blob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point
disparity map, where disparity values are multiplied by 16, this scale factor should be taken into
account when specifying this parameter value.
@param buf The optional temporary buffer to avoid memory allocation within the function.
 */
CV_EXPORTS_W void filterSpeckles( InputOutputArray img, double newVal,
                                  int maxSpeckleSize, double maxDiff,
                                  InputOutputArray buf = noArray() );

//! computes valid disparity ROI from the valid ROIs of the rectified images (that are returned by cv::stereoRectify())
CV_EXPORTS_W Rect getValidDisparityROI( Rect roi1, Rect roi2,
                                        int minDisparity, int numberOfDisparities,
                                        int SADWindowSize );

//! validates disparity using the left-right check. The matrix "cost" should be computed by the stereo correspondence algorithm
CV_EXPORTS_W void validateDisparity( InputOutputArray disparity, InputArray cost,
                                     int minDisparity, int numberOfDisparities,
                                     int disp12MaxDisp = 1 );

/** @brief Reprojects a disparity image to 3D space.

@param disparity Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit
floating-point disparity image. If 16-bit signed format is used, the values are assumed to have no
fractional bits.
@param _3dImage Output 3-channel floating-point image of the same size as disparity . Each
element of _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity
map.
@param Q \f$4 \times 4\f$ perspective transformation matrix that can be obtained with stereoRectify.
@param handleMissingValues Indicates, whether the function should handle missing values (i.e.
points where the disparity was not computed). If handleMissingValues=true, then pixels with the
minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed
to 3D points with a very large Z value (currently set to 10000).
@param ddepth The optional output array depth. If it is -1, the output image will have CV_32F
depth. ddepth can also be set to CV_16S, CV_32S or CV_32F.

The function transforms a single-channel disparity map to a 3-channel image representing a 3D
surface. That is, for each pixel (x,y) andthe corresponding disparity d=disparity(x,y) , it
computes:

\f[\begin{array}{l} [X \; Y \; Z \; W]^T =  \texttt{Q} *[x \; y \; \texttt{disparity} (x,y) \; 1]^T  \\ \texttt{\_3dImage} (x,y) = (X/W, \; Y/W, \; Z/W) \end{array}\f]

The matrix Q can be an arbitrary \f$4 \times 4\f$ matrix (for example, the one computed by
stereoRectify). To reproject a sparse set of points {(x,y,d),...} to 3D space, use
perspectiveTransform .
 */
CV_EXPORTS_W void reprojectImageTo3D( InputArray disparity,
                                      OutputArray _3dImage, InputArray Q,
                                      bool handleMissingValues = false,
                                      int ddepth = -1 );

/** @brief Calculates the Sampson Distance between two points.

The function sampsonDistance calculates and returns the first order approximation of the geometric error as:
\f[sd( \texttt{pt1} , \texttt{pt2} )= \frac{(\texttt{pt2}^t \cdot \texttt{F} \cdot \texttt{pt1})^2}{(\texttt{F} \cdot \texttt{pt1})(0) + (\texttt{F} \cdot \texttt{pt1})(1) + (\texttt{F}^t \cdot \texttt{pt2})(0) + (\texttt{F}^t \cdot \texttt{pt2})(1)}\f]
The fundamental matrix may be calculated using the cv::findFundamentalMat function. See HZ 11.4.3 for details.
@param pt1 first homogeneous 2d point
@param pt2 second homogeneous 2d point
@param F fundamental matrix
*/
CV_EXPORTS_W double sampsonDistance(InputArray pt1, InputArray pt2, InputArray F);

/** @brief Computes an optimal affine transformation between two 3D point sets.

@param src First input 3D point set.
@param dst Second input 3D point set.
@param out Output 3D affine transformation matrix \f$3 \times 4\f$ .
@param inliers Output vector indicating which points are inliers.
@param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as
an inlier.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.

The function estimates an optimal 3D affine transformation between two 3D point sets using the
RANSAC algorithm.
 */
CV_EXPORTS_W  int estimateAffine3D(InputArray src, InputArray dst,
                                   OutputArray out, OutputArray inliers,
                                   double ransacThreshold = 3, double confidence = 0.99);

/** @brief Computes an optimal affine transformation between two 2D point sets.

@param from First input 2D point set.
@param to Second input 2D point set.
@param inliers Output vector indicating which points are inliers.
@param method Robust method used to compute tranformation. The following methods are possible:
-   cv::RANSAC - RANSAC-based robust method
-   cv::LMEDS - Least-Median robust method
RANSAC is the default method.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations, 2000 is the maximum it can be.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output 2D affine transformation matrix \f$2 \times 3\f$ or empty matrix if transformation
could not be estimated.

The function estimates an optimal 2D affine transformation between two 2D point sets using the
selected robust algorithm.

The computed transformation is then refined further (using only inliers) with the
Levenberg-Marquardt method to reduce the re-projection error even more.

@note
The RANSAC method can handle practically any ratio of outliers but need a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers.

@sa estimateAffinePartial2D, getAffineTransform
*/
CV_EXPORTS_W cv::Mat estimateAffine2D(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = RANSAC, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

/** @brief Computes an optimal limited affine transformation with 4 degrees of freedom between
two 2D point sets.

@param from First input 2D point set.
@param to Second input 2D point set.
@param inliers Output vector indicating which points are inliers.
@param method Robust method used to compute tranformation. The following methods are possible:
-   cv::RANSAC - RANSAC-based robust method
-   cv::LMEDS - Least-Median robust method
RANSAC is the default method.
@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.
@param maxIters The maximum number of robust method iterations, 2000 is the maximum it can be.
@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything
between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation
significantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.
@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

@return Output 2D affine transformation (4 degrees of freedom) matrix \f$2 \times 3\f$ or
empty matrix if transformation could not be estimated.

The function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to
combinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust
estimation.

The computed transformation is then refined further (using only inliers) with the
Levenberg-Marquardt method to reduce the re-projection error even more.

Estimated transformation matrix is:
\f[ \begin{bmatrix} \cos(\theta)s & -\sin(\theta)s & tx \\
                \sin(\theta)s & \cos(\theta)s & ty
\end{bmatrix} \f]
Where \f$ \theta \f$ is the rotation angle, \f$ s \f$ the scaling factor and \f$ tx, ty \f$ are
translations in \f$ x, y \f$ axes respectively.

@note
The RANSAC method can handle practically any ratio of outliers but need a threshold to
distinguish inliers from outliers. The method LMeDS does not need any threshold but it works
correctly only when there are more than 50% of inliers.

@sa estimateAffine2D, getAffineTransform
*/
CV_EXPORTS_W cv::Mat estimateAffinePartial2D(InputArray from, InputArray to, OutputArray inliers = noArray(),
                                  int method = RANSAC, double ransacReprojThreshold = 3,
                                  size_t maxIters = 2000, double confidence = 0.99,
                                  size_t refineIters = 10);

/** @brief Decompose a homography matrix to rotation(s), translation(s) and plane normal(s).

@param H The input homography matrix between two images.
@param K The input intrinsic camera calibration matrix.
@param rotations Array of rotation matrices.
@param translations Array of translation matrices.
@param normals Array of plane normal matrices.

This function extracts relative camera motion between two views observing a planar object from the
homography H induced by the plane. The intrinsic camera matrix K must also be provided. The function
may return up to four mathematical solution sets. At least two of the solutions may further be
invalidated if point correspondences are available by applying positive depth constraint (all points
must be in front of the camera). The decomposition method is described in detail in @cite Malis .
 */
CV_EXPORTS_W int decomposeHomographyMat(InputArray H,
                                        InputArray K,
                                        OutputArrayOfArrays rotations,
                                        OutputArrayOfArrays translations,
                                        OutputArrayOfArrays normals);

/** @brief The base class for stereo correspondence algorithms.
 */
class CV_EXPORTS_W StereoMatcher : public Algorithm
{
public:
    enum { DISP_SHIFT = 4,
           DISP_SCALE = (1 << DISP_SHIFT)
         };

    /** @brief Computes disparity map for the specified stereo pair

    @param left Left 8-bit single-channel image.
    @param right Right image of the same size and the same type as the left one.
    @param disparity Output disparity map. It has the same size as the input images. Some algorithms,
    like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value
    has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map.
     */
    CV_WRAP virtual void compute( InputArray left, InputArray right,
                                  OutputArray disparity ) = 0;

    CV_WRAP virtual int getMinDisparity() const = 0;
    CV_WRAP virtual void setMinDisparity(int minDisparity) = 0;

    CV_WRAP virtual int getNumDisparities() const = 0;
    CV_WRAP virtual void setNumDisparities(int numDisparities) = 0;

    CV_WRAP virtual int getBlockSize() const = 0;
    CV_WRAP virtual void setBlockSize(int blockSize) = 0;

    CV_WRAP virtual int getSpeckleWindowSize() const = 0;
    CV_WRAP virtual void setSpeckleWindowSize(int speckleWindowSize) = 0;

    CV_WRAP virtual int getSpeckleRange() const = 0;
    CV_WRAP virtual void setSpeckleRange(int speckleRange) = 0;

    CV_WRAP virtual int getDisp12MaxDiff() const = 0;
    CV_WRAP virtual void setDisp12MaxDiff(int disp12MaxDiff) = 0;
};


/** @brief Class for computing stereo correspondence using the block matching algorithm, introduced and
contributed to OpenCV by K. Konolige.
 */
class CV_EXPORTS_W StereoBM : public StereoMatcher
{
public:
    enum { PREFILTER_NORMALIZED_RESPONSE = 0,
           PREFILTER_XSOBEL              = 1
         };

    CV_WRAP virtual int getPreFilterType() const = 0;
    CV_WRAP virtual void setPreFilterType(int preFilterType) = 0;

    CV_WRAP virtual int getPreFilterSize() const = 0;
    CV_WRAP virtual void setPreFilterSize(int preFilterSize) = 0;

    CV_WRAP virtual int getPreFilterCap() const = 0;
    CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;

    CV_WRAP virtual int getTextureThreshold() const = 0;
    CV_WRAP virtual void setTextureThreshold(int textureThreshold) = 0;

    CV_WRAP virtual int getUniquenessRatio() const = 0;
    CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;

    CV_WRAP virtual int getSmallerBlockSize() const = 0;
    CV_WRAP virtual void setSmallerBlockSize(int blockSize) = 0;

    CV_WRAP virtual Rect getROI1() const = 0;
    CV_WRAP virtual void setROI1(Rect roi1) = 0;

    CV_WRAP virtual Rect getROI2() const = 0;
    CV_WRAP virtual void setROI2(Rect roi2) = 0;

    /** @brief Creates StereoBM object

    @param numDisparities the disparity search range. For each pixel algorithm will find the best
    disparity from 0 (default minimum disparity) to numDisparities. The search range can then be
    shifted by changing the minimum disparity.
    @param blockSize the linear size of the blocks compared by the algorithm. The size should be odd
    (as the block is centered at the current pixel). Larger block size implies smoother, though less
    accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher
    chance for algorithm to find a wrong correspondence.

    The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for
    a specific stereo pair.
     */
    CV_WRAP static Ptr<StereoBM> create(int numDisparities = 0, int blockSize = 21);
};

/** @brief The class implements the modified H. Hirschmuller algorithm @cite HH08 that differs from the original
one as follows:

-   By default, the algorithm is single-pass, which means that you consider only 5 directions
instead of 8. Set mode=StereoSGBM::MODE_HH in createStereoSGBM to run the full variant of the
algorithm but beware that it may consume a lot of memory.
-   The algorithm matches blocks, not individual pixels. Though, setting blockSize=1 reduces the
blocks to single pixels.
-   Mutual information cost function is not implemented. Instead, a simpler Birchfield-Tomasi
sub-pixel metric from @cite BT98 is used. Though, the color images are supported as well.
-   Some pre- and post- processing steps from K. Konolige algorithm StereoBM are included, for
example: pre-filtering (StereoBM::PREFILTER_XSOBEL type) and post-filtering (uniqueness
check, quadratic interpolation and speckle filtering).

@note
   -   (Python) An example illustrating the use of the StereoSGBM matching algorithm can be found
        at opencv_source_code/samples/python/stereo_match.py
 */
class CV_EXPORTS_W StereoSGBM : public StereoMatcher
{
public:
    enum
    {
        MODE_SGBM = 0,
        MODE_HH   = 1,
        MODE_SGBM_3WAY = 2,
        MODE_HH4  = 3
    };

    CV_WRAP virtual int getPreFilterCap() const = 0;
    CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;

    CV_WRAP virtual int getUniquenessRatio() const = 0;
    CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;

    CV_WRAP virtual int getP1() const = 0;
    CV_WRAP virtual void setP1(int P1) = 0;

    CV_WRAP virtual int getP2() const = 0;
    CV_WRAP virtual void setP2(int P2) = 0;

    CV_WRAP virtual int getMode() const = 0;
    CV_WRAP virtual void setMode(int mode) = 0;

    /** @brief Creates StereoSGBM object

    @param minDisparity Minimum possible disparity value. Normally, it is zero but sometimes
    rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    @param numDisparities Maximum disparity minus minimum disparity. The value is always greater than
    zero. In the current implementation, this parameter must be divisible by 16.
    @param blockSize Matched block size. It must be an odd number \>=1 . Normally, it should be
    somewhere in the 3..11 range.
    @param P1 The first parameter controlling the disparity smoothness. See below.
    @param P2 The second parameter controlling the disparity smoothness. The larger the values are,
    the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1
    between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor
    pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good
    P1 and P2 values are shown (like 8\*number_of_image_channels\*SADWindowSize\*SADWindowSize and
    32\*number_of_image_channels\*SADWindowSize\*SADWindowSize , respectively).
    @param disp12MaxDiff Maximum allowed difference (in integer pixel units) in the left-right
    disparity check. Set it to a non-positive value to disable the check.
    @param preFilterCap Truncation value for the prefiltered image pixels. The algorithm first
    computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.
    The result values are passed to the Birchfield-Tomasi pixel cost function.
    @param uniquenessRatio Margin in percentage by which the best (minimum) computed cost function
    value should "win" the second best value to consider the found match correct. Normally, a value
    within the 5-15 range is good enough.
    @param speckleWindowSize Maximum size of smooth disparity regions to consider their noise speckles
    and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the
    50-200 range.
    @param speckleRange Maximum disparity variation within each connected component. If you do speckle
    filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    Normally, 1 or 2 is good enough.
    @param mode Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming
    algorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and
    huge for HD-size pictures. By default, it is set to false .

    The first constructor initializes StereoSGBM with all the default parameters. So, you only have to
    set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter
    to a custom value.
     */
    CV_WRAP static Ptr<StereoSGBM> create(int minDisparity = 0, int numDisparities = 16, int blockSize = 3,
                                          int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
                                          int preFilterCap = 0, int uniquenessRatio = 0,
                                          int speckleWindowSize = 0, int speckleRange = 0,
                                          int mode = StereoSGBM::MODE_SGBM);
};

//! @} calib3d

/** @brief The methods in this namespace use a so-called fisheye camera model.
  @ingroup calib3d_fisheye
*/
namespace fisheye
{
//! @addtogroup calib3d_fisheye
//! @{

    enum{
        CALIB_USE_INTRINSIC_GUESS   = 1 << 0,
        CALIB_RECOMPUTE_EXTRINSIC   = 1 << 1,
        CALIB_CHECK_COND            = 1 << 2,
        CALIB_FIX_SKEW              = 1 << 3,
        CALIB_FIX_K1                = 1 << 4,
        CALIB_FIX_K2                = 1 << 5,
        CALIB_FIX_K3                = 1 << 6,
        CALIB_FIX_K4                = 1 << 7,
        CALIB_FIX_INTRINSIC         = 1 << 8,
        CALIB_FIX_PRINCIPAL_POINT   = 1 << 9
    };

    /** @brief Projects points using fisheye model

    @param objectPoints Array of object points, 1xN/Nx1 3-channel (or vector\<Point3f\> ), where N is
    the number of points in the view.
    @param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or
    vector\<Point2f\>.
    @param affine
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param alpha The skew coefficient.
    @param jacobian Optional output 2Nx15 jacobian matrix of derivatives of image points with respect
    to components of the focal lengths, coordinates of the principal point, distortion coefficients,
    rotation vector, translation vector, and the skew. In the old interface different components of
    the jacobian are returned via different output parameters.

    The function computes projections of 3D points to the image plane given intrinsic and extrinsic
    camera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of
    image points coordinates (as functions of all the input parameters) with respect to the particular
    parameters, intrinsic and/or extrinsic.
     */
    CV_EXPORTS void projectPoints(InputArray objectPoints, OutputArray imagePoints, const Affine3d& affine,
        InputArray K, InputArray D, double alpha = 0, OutputArray jacobian = noArray());

    /** @overload */
    CV_EXPORTS_W void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec,
        InputArray K, InputArray D, double alpha = 0, OutputArray jacobian = noArray());

    /** @brief Distorts 2D points using fisheye model.

    @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is
    the number of points in the view.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param alpha The skew coefficient.
    @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .

    Note that the function assumes the camera matrix of the undistorted points to be indentity.
    This means if you want to transform back points undistorted with undistortPoints() you have to
    multiply them with \f$P^{-1}\f$.
     */
    CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double alpha = 0);

    /** @brief Undistorts 2D points using fisheye model

    @param distorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is the
    number of points in the view.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
     */
    CV_EXPORTS_W void undistortPoints(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray R = noArray(), InputArray P  = noArray());

    /** @brief Computes undistortion and rectification maps for image transform by cv::remap(). If D is empty zero
    distortion is used, if R or P is empty identity matrixes are used.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
    for details.
    @param map1 The first output map.
    @param map2 The second output map.
     */
    CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, InputArray R, InputArray P,
        const cv::Size& size, int m1type, OutputArray map1, OutputArray map2);

    /** @brief Transforms an image to compensate for fisheye lens distortion.

    @param distorted image with fisheye lens distortion.
    @param undistorted Output image with compensated fisheye lens distortion.
    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param Knew Camera matrix of the distorted image. By default, it is the identity matrix but you
    may additionally scale and shift the result by using a different matrix.
    @param new_size

    The function transforms an image to compensate radial and tangential lens distortion.

    The function is simply a combination of fisheye::initUndistortRectifyMap (with unity R ) and remap
    (with bilinear interpolation). See the former function for details of the transformation being
    performed.

    See below the results of undistortImage.
       -   a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
            k_4, k_5, k_6) of distortion were optimized under calibration)
        -   b\) result of fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
            k_3, k_4) of fisheye distortion were optimized under calibration)
        -   c\) original image was captured with fisheye lens

    Pictures a) and b) almost the same. But if we consider points of image located far from the center
    of image, we can notice that on image a) these points are distorted.

    ![image](pics/fisheye_undistorted.jpg)
     */
    CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, InputArray Knew = cv::noArray(), const Size& new_size = Size());

    /** @brief Estimates new camera matrix for undistortion or rectification.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param image_size
    @param D Input vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param balance Sets the new focal length in range between the min focal length and the max focal
    length. Balance is in range of [0, 1].
    @param new_size
    @param fov_scale Divisor for new focal length.
     */
    CV_EXPORTS_W void estimateNewCameraMatrixForUndistortRectify(InputArray K, InputArray D, const Size &image_size, InputArray R,
        OutputArray P, double balance = 0.0, const Size& new_size = Size(), double fov_scale = 1.0);

    /** @brief Performs camera calibaration

    @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
    coordinate space.
    @param imagePoints vector of vectors of the projections of calibration pattern points.
    imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
    objectPoints[i].size() for each i.
    @param image_size Size of the image used only to initialize the intrinsic camera matrix.
    @param K Output 3x3 floating-point camera matrix
    \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If
    fisheye::CALIB_USE_INTRINSIC_GUESS/ is specified, some or all of fx, fy, cx, cy must be
    initialized before calling the function.
    @param D Output vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$.
    @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
    That is, each k-th rotation vector together with the corresponding k-th translation vector (see
    the next output parameter description) brings the calibration pattern from the model coordinate
    space (in which object points are specified) to the world coordinate space, that is, a real
    position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
    @param tvecs Output vector of translation vectors estimated for each pattern view.
    @param flags Different flags that may be zero or a combination of the following values:
    -   **fisheye::CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center ( imageSize is used), and focal distances are computed in a least-squares fashion.
    -   **fisheye::CALIB_RECOMPUTE_EXTRINSIC** Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -   **fisheye::CALIB_CHECK_COND** The functions will check validity of condition number.
    -   **fisheye::CALIB_FIX_SKEW** Skew coefficient (alpha) is set to zero and stay zero.
    -   **fisheye::CALIB_FIX_K1..fisheye::CALIB_FIX_K4** Selected distortion coefficients
    are set to zeros and stay zero.
    -   **fisheye::CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global
optimization. It stays at the center or at a different location specified when CALIB_USE_INTRINSIC_GUESS is set too.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, const Size& image_size,
        InputOutputArray K, InputOutputArray D, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags = 0,
            TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

    /** @brief Stereo rectification for fisheye camera model

    @param K1 First camera matrix.
    @param D1 First camera distortion parameters.
    @param K2 Second camera matrix.
    @param D2 Second camera distortion parameters.
    @param imageSize Size of the image used for stereo calibration.
    @param R Rotation matrix between the coordinate systems of the first and the second
    cameras.
    @param tvec Translation vector between coordinate systems of the cameras.
    @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
    camera.
    @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
    camera.
    @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
    @param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,
    the function makes the principal points of each camera have the same pixel coordinates in the
    rectified views. And if the flag is not set, the function may still shift the images in the
    horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
    useful image area.
    @param newImageSize New image resolution after rectification. The same size should be passed to
    initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
    is passed (default), it is set to the original imageSize . Setting it to larger value can help you
    preserve details in the original image, especially when there is a big radial distortion.
    @param balance Sets the new focal length in range between the min focal length and the max focal
    length. Balance is in range of [0, 1].
    @param fov_scale Divisor for new focal length.
     */
    CV_EXPORTS_W void stereoRectify(InputArray K1, InputArray D1, InputArray K2, InputArray D2, const Size &imageSize, InputArray R, InputArray tvec,
        OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags, const Size &newImageSize = Size(),
        double balance = 0.0, double fov_scale = 1.0);

    /** @brief Performs stereo calibration

    @param objectPoints Vector of vectors of the calibration pattern points.
    @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
    observed by the first camera.
    @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
    observed by the second camera.
    @param K1 Input/output first camera matrix:
    \f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
    any of fisheye::CALIB_USE_INTRINSIC_GUESS , fisheye::CALIB_FIX_INTRINSIC are specified,
    some or all of the matrix components must be initialized.
    @param D1 Input/output vector of distortion coefficients \f$(k_1, k_2, k_3, k_4)\f$ of 4 elements.
    @param K2 Input/output second camera matrix. The parameter is similar to K1 .
    @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
    similar to D1 .
    @param imageSize Size of the image used only to initialize intrinsic camera matrix.
    @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
    @param T Output translation vector between the coordinate systems of the cameras.
    @param flags Different flags that may be zero or a combination of the following values:
    -   **fisheye::CALIB_FIX_INTRINSIC** Fix K1, K2? and D1, D2? so that only R, T matrices
    are estimated.
    -   **fisheye::CALIB_USE_INTRINSIC_GUESS** K1, K2 contains valid initial values of
    fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    center (imageSize is used), and focal distances are computed in a least-squares fashion.
    -   **fisheye::CALIB_RECOMPUTE_EXTRINSIC** Extrinsic will be recomputed after each iteration
    of intrinsic optimization.
    -   **fisheye::CALIB_CHECK_COND** The functions will check validity of condition number.
    -   **fisheye::CALIB_FIX_SKEW** Skew coefficient (alpha) is set to zero and stay zero.
    -   **fisheye::CALIB_FIX_K1..4** Selected distortion coefficients are set to zeros and stay
    zero.
    @param criteria Termination criteria for the iterative optimization algorithm.
     */
    CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                  InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                  OutputArray R, OutputArray T, int flags = fisheye::CALIB_FIX_INTRINSIC,
                                  TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON));

//! @} calib3d_fisheye
}

} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/calib3d/calib3d_c.h"
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
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CORE_HPP
#define OPENCV_CORE_HPP

#ifndef __cplusplus
#  error core.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/version.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/persistence.hpp"

/**
@defgroup core Core functionality
@{
    @defgroup core_basic Basic structures
    @defgroup core_c C structures and operations
    @{
        @defgroup core_c_glue Connections with C++
    @}
    @defgroup core_array Operations on arrays
    @defgroup core_xml XML/YAML Persistence
    @defgroup core_cluster Clustering
    @defgroup core_utils Utility and system functions and macros
    @{
        @defgroup core_utils_sse SSE utilities
        @defgroup core_utils_neon NEON utilities
    @}
    @defgroup core_opengl OpenGL interoperability
    @defgroup core_ipp Intel IPP Asynchronous C/C++ Converters
    @defgroup core_optim Optimization Algorithms
    @defgroup core_directx DirectX interoperability
    @defgroup core_eigen Eigen support
    @defgroup core_opencl OpenCL support
    @defgroup core_va_intel Intel VA-API/OpenCL (CL-VA) interoperability
    @defgroup core_hal Hardware Acceleration Layer
    @{
        @defgroup core_hal_functions Functions
        @defgroup core_hal_interface Interface
        @defgroup core_hal_intrin Universal intrinsics
        @{
            @defgroup core_hal_intrin_impl Private implementation helpers
        @}
    @}
@}
 */

namespace cv {

//! @addtogroup core_utils
//! @{

/*! @brief Class passed to an error.

This class encapsulates all or almost all necessary
information about the error happened in the program. The exception is
usually constructed and thrown implicitly via CV_Error and CV_Error_ macros.
@see error
 */
class CV_EXPORTS Exception : public std::exception
{
public:
    /*!
     Default constructor
     */
    Exception();
    /*!
     Full constructor. Normally the constructor is not called explicitly.
     Instead, the macros CV_Error(), CV_Error_() and CV_Assert() are used.
    */
    Exception(int _code, const String& _err, const String& _func, const String& _file, int _line);
    virtual ~Exception() throw();

    /*!
     \return the error description and the context as a text string.
    */
    virtual const char *what() const throw();
    void formatMessage();

    String msg; ///< the formatted error message

    int code; ///< error code @see CVStatus
    String err; ///< error description
    String func; ///< function name. Available only when the compiler supports getting it
    String file; ///< source file name where the error has occurred
    int line; ///< line number in the source file where the error has occurred
};

/*! @brief Signals an error and raises the exception.

By default the function prints information about the error to stderr,
then it either stops if cv::setBreakOnError() had been called before or raises the exception.
It is possible to alternate error processing by using cv::redirectError().
@param exc the exception raisen.
@deprecated drop this version
 */
CV_EXPORTS void error( const Exception& exc );

enum SortFlags { SORT_EVERY_ROW    = 0, //!< each matrix row is sorted independently
                 SORT_EVERY_COLUMN = 1, //!< each matrix column is sorted
                                        //!< independently; this flag and the previous one are
                                        //!< mutually exclusive.
                 SORT_ASCENDING    = 0, //!< each matrix row is sorted in the ascending
                                        //!< order.
                 SORT_DESCENDING   = 16 //!< each matrix row is sorted in the
                                        //!< descending order; this flag and the previous one are also
                                        //!< mutually exclusive.
               };

//! @} core_utils

//! @addtogroup core
//! @{

//! Covariation flags
enum CovarFlags {
    /** The output covariance matrix is calculated as:
       \f[\texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]^T  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...],\f]
       The covariance matrix will be nsamples x nsamples. Such an unusual covariance matrix is used
       for fast PCA of a set of very large vectors (see, for example, the EigenFaces technique for
       face recognition). Eigenvalues of this "scrambled" matrix match the eigenvalues of the true
       covariance matrix. The "true" eigenvectors can be easily calculated from the eigenvectors of
       the "scrambled" covariance matrix. */
    COVAR_SCRAMBLED = 0,
    /**The output covariance matrix is calculated as:
        \f[\texttt{scale}   \cdot  [  \texttt{vects}  [0]-  \texttt{mean}  , \texttt{vects}  [1]-  \texttt{mean}  ,...]  \cdot  [ \texttt{vects}  [0]- \texttt{mean}  , \texttt{vects}  [1]- \texttt{mean}  ,...]^T,\f]
        covar will be a square matrix of the same size as the total number of elements in each input
        vector. One and only one of COVAR_SCRAMBLED and COVAR_NORMAL must be specified.*/
    COVAR_NORMAL    = 1,
    /** If the flag is specified, the function does not calculate mean from
        the input vectors but, instead, uses the passed mean vector. This is useful if mean has been
        pre-calculated or known in advance, or if the covariance matrix is calculated by parts. In
        this case, mean is not a mean vector of the input sub-set of vectors but rather the mean
        vector of the whole set.*/
    COVAR_USE_AVG   = 2,
    /** If the flag is specified, the covariance matrix is scaled. In the
        "normal" mode, scale is 1./nsamples . In the "scrambled" mode, scale is the reciprocal of the
        total number of elements in each input vector. By default (if the flag is not specified), the
        covariance matrix is not scaled ( scale=1 ).*/
    COVAR_SCALE     = 4,
    /** If the flag is
        specified, all the input vectors are stored as rows of the samples matrix. mean should be a
        single-row vector in this case.*/
    COVAR_ROWS      = 8,
    /** If the flag is
        specified, all the input vectors are stored as columns of the samples matrix. mean should be a
        single-column vector in this case.*/
    COVAR_COLS      = 16
};

//! k-Means flags
enum KmeansFlags {
    /** Select random initial centers in each attempt.*/
    KMEANS_RANDOM_CENTERS     = 0,
    /** Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].*/
    KMEANS_PP_CENTERS         = 2,
    /** During the first (and possibly the only) attempt, use the
        user-supplied labels instead of computing them from the initial centers. For the second and
        further attempts, use the random or semi-random centers. Use one of KMEANS_\*_CENTERS flag
        to specify the exact method.*/
    KMEANS_USE_INITIAL_LABELS = 1
};

//! type of line
enum LineTypes {
    FILLED  = -1,
    LINE_4  = 4, //!< 4-connected line
    LINE_8  = 8, //!< 8-connected line
    LINE_AA = 16 //!< antialiased line
};

//! Only a subset of Hershey fonts
//! <http://sources.isc.org/utils/misc/hershey-font.txt> are supported
enum HersheyFonts {
    FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_ITALIC                 = 16 //!< flag for italic font
};

enum ReduceTypes { REDUCE_SUM = 0, //!< the output is the sum of all rows/columns of the matrix.
                   REDUCE_AVG = 1, //!< the output is the mean vector of all rows/columns of the matrix.
                   REDUCE_MAX = 2, //!< the output is the maximum (column/row-wise) of all rows/columns of the matrix.
                   REDUCE_MIN = 3  //!< the output is the minimum (column/row-wise) of all rows/columns of the matrix.
                 };


/** @brief Swaps two matrices
*/
CV_EXPORTS void swap(Mat& a, Mat& b);
/** @overload */
CV_EXPORTS void swap( UMat& a, UMat& b );

//! @} core

//! @addtogroup core_array
//! @{

/** @brief Computes the source location of an extrapolated pixel.

The function computes and returns the coordinate of a donor pixel corresponding to the specified
extrapolated pixel when using the specified extrapolation border mode. For example, if you use
cv::BORDER_WRAP mode in the horizontal direction, cv::BORDER_REFLECT_101 in the vertical direction and
want to compute value of the "virtual" pixel Point(-5, 100) in a floating-point image img , it
looks like:
@code{.cpp}
    float val = img.at<float>(borderInterpolate(100, img.rows, cv::BORDER_REFLECT_101),
                              borderInterpolate(-5, img.cols, cv::BORDER_WRAP));
@endcode
Normally, the function is not called directly. It is used inside filtering functions and also in
copyMakeBorder.
@param p 0-based coordinate of the extrapolated pixel along one of the axes, likely \<0 or \>= len
@param len Length of the array along the corresponding axis.
@param borderType Border type, one of the cv::BorderTypes, except for cv::BORDER_TRANSPARENT and
cv::BORDER_ISOLATED . When borderType==cv::BORDER_CONSTANT , the function always returns -1, regardless
of p and len.

@sa copyMakeBorder
*/
CV_EXPORTS_W int borderInterpolate(int p, int len, int borderType);

/** @example copyMakeBorder_demo.cpp
An example using copyMakeBorder function
 */
/** @brief Forms a border around an image.

The function copies the source image into the middle of the destination image. The areas to the
left, to the right, above and below the copied source image will be filled with extrapolated
pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but
what other more complex functions, including your own, may do to simplify image boundary handling.

The function supports the mode when src is already in the middle of dst . In this case, the
function does not copy src itself but simply constructs the border, for example:

@code{.cpp}
    // let border be the same in all directions
    int border=2;
    // constructs a larger image to fit both the image and the border
    Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());
    // select the middle part of it w/o copying data
    Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));
    // convert image from RGB to grayscale
    cvtColor(rgb, gray, COLOR_RGB2GRAY);
    // form a border in-place
    copyMakeBorder(gray, gray_buf, border, border,
                   border, border, BORDER_REPLICATE);
    // now do some custom filtering ...
    ...
@endcode
@note When the source image is a part (ROI) of a bigger image, the function will try to use the
pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as
if src was not a ROI, use borderType | BORDER_ISOLATED.

@param src Source image.
@param dst Destination image of the same type as src and the size Size(src.cols+left+right,
src.rows+top+bottom) .
@param top
@param bottom
@param left
@param right Parameter specifying how many pixels in each direction from the source image rectangle
to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
to be built.
@param borderType Border type. See borderInterpolate for details.
@param value Border value if borderType==BORDER_CONSTANT .

@sa  borderInterpolate
*/
CV_EXPORTS_W void copyMakeBorder(InputArray src, OutputArray dst,
                                 int top, int bottom, int left, int right,
                                 int borderType, const Scalar& value = Scalar() );

/** @brief Calculates the per-element sum of two arrays or an array and a scalar.

The function add calculates:
- Sum of two arrays when both input arrays have the same size and the same number of channels:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]
- Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of
elements as `src1.channels()`:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]
- Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of
elements as `src2.channels()`:
\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]
where `I` is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

The first function in the list above can be replaced with matrix expressions:
@code{.cpp}
    dst = src1 + src2;
    dst += src1; // equivalent to add(dst, src1, dst);
@endcode
The input arrays and the output array can all have the same or different depths. For example, you
can add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit
floating-point array. Depth of the output array is determined by the dtype parameter. In the second
and third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can
be set to the default -1. In this case, the output array will have the same depth as the input
array, be it src1, src2 or both.
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and number of channels as the input array(s); the
depth is defined by dtype or src1/src2.
@param mask optional operation mask - 8-bit single channel array, that specifies elements of the
output array to be changed.
@param dtype optional depth of the output array (see the discussion below).
@sa subtract, addWeighted, scaleAdd, Mat::convertTo
*/
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,
                      InputArray mask = noArray(), int dtype = -1);

/** @brief Calculates the per-element difference between two arrays or array and a scalar.

The function subtract calculates:
- Difference between two arrays, when both input arrays have the same size and the same number of
channels:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]
- Difference between an array and a scalar, when src2 is constructed from Scalar or has the same
number of elements as `src1.channels()`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]
- Difference between a scalar and an array, when src1 is constructed from Scalar or has the same
number of elements as `src2.channels()`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]
- The reverse difference between a scalar and an array in the case of `SubRS`:
    \f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\f]
where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.

The first function in the list above can be replaced with matrix expressions:
@code{.cpp}
    dst = src1 - src2;
    dst -= src1; // equivalent to subtract(dst, src1, dst);
@endcode
The input arrays and the output array can all have the same or different depths. For example, you
can subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of
the output array is determined by dtype parameter. In the second and third cases above, as well as
in the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this
case the output array will have the same depth as the input array, be it src1, src2 or both.
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array of the same size and the same number of channels as the input array.
@param mask optional operation mask; this is an 8-bit single channel array that specifies elements
of the output array to be changed.
@param dtype optional depth of the output array
@sa  add, addWeighted, scaleAdd, Mat::convertTo
  */
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,
                           InputArray mask = noArray(), int dtype = -1);


/** @brief Calculates the per-element scaled product of two arrays.

The function multiply calculates the per-element product of two arrays:

\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\f]

There is also a @ref MatrixExpressions -friendly variant of the first function. See Mat::mul .

For a not-per-element matrix product, see gemm .

@note Saturation is not applied when the output array has the depth
CV_32S. You may even get result of an incorrect sign in the case of
overflow.
@param src1 first input array.
@param src2 second input array of the same size and the same type as src1.
@param dst output array of the same size and type as src1.
@param scale optional scale factor.
@param dtype optional depth of the output array
@sa add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,
Mat::convertTo
*/
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,
                           OutputArray dst, double scale = 1, int dtype = -1);

/** @brief Performs per-element division of two arrays or a scalar by an array.

The function cv::divide divides one array by another:
\f[\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\f]
or a scalar by an array when there is no src1 :
\f[\texttt{dst(I) = saturate(scale/src2(I))}\f]

When src2(I) is zero, dst(I) will also be zero. Different channels of
multi-channel arrays are processed independently.

@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array.
@param src2 second input array of the same size and type as src1.
@param scale scalar factor.
@param dst output array of the same size and type as src2.
@param dtype optional depth of the output array; if -1, dst will have depth src2.depth(), but in
case of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().
@sa  multiply, add, subtract
*/
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst,
                         double scale = 1, int dtype = -1);

/** @overload */
CV_EXPORTS_W void divide(double scale, InputArray src2,
                         OutputArray dst, int dtype = -1);

/** @brief Calculates the sum of a scaled array and another array.

The function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY
or SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates
the sum of a scaled array and another array:
\f[\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)\f]
The function can also be emulated with a matrix expression, for example:
@code{.cpp}
    Mat A(3, 3, CV_64F);
    ...
    A.row(0) = A.row(1)*2 + A.row(2);
@endcode
@param src1 first input array.
@param alpha scale factor for the first array.
@param src2 second input array of the same size and type as src1.
@param dst output array of the same size and type as src1.
@sa add, addWeighted, subtract, Mat::dot, Mat::convertTo
*/
CV_EXPORTS_W void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst);

/** @example AddingImagesTrackbar.cpp

 */
/** @brief Calculates the weighted sum of two arrays.

The function addWeighted calculates the weighted sum of two arrays as follows:
\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\f]
where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each
channel is processed independently.
The function can be replaced with a matrix expression:
@code{.cpp}
    dst = src1*alpha + src2*beta + gamma;
@endcode
@note Saturation is not applied when the output array has the depth CV_32S. You may even get
result of an incorrect sign in the case of overflow.
@param src1 first input array.
@param alpha weight of the first array elements.
@param src2 second input array of the same size and channel number as src1.
@param beta weight of the second array elements.
@param gamma scalar added to each sum.
@param dst output array that has the same size and number of channels as the input arrays.
@param dtype optional depth of the output array; when both input arrays have the same depth, dtype
can be set to -1, which will be equivalent to src1.depth().
@sa  add, subtract, scaleAdd, Mat::convertTo
*/
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);

/** @brief Scales, calculates absolute values, and converts the result to 8-bit.

On each element of the input array, the function convertScaleAbs
performs three operations sequentially: scaling, taking an absolute
value, conversion to an unsigned 8-bit type:
\f[\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\f]
In case of multi-channel arrays, the function processes each channel
independently. When the output is not 8-bit, the operation can be
emulated by calling the Mat::convertTo method (or by using matrix
expressions) and then by calculating an absolute value of the result.
For example:
@code{.cpp}
    Mat_<float> A(30,30);
    randu(A, Scalar(-100), Scalar(100));
    Mat_<float> B = A*5 + 3;
    B = abs(B);
    // Mat_<float> B = abs(A*5+3) will also do the job,
    // but it will allocate a temporary matrix
@endcode
@param src input array.
@param dst output array.
@param alpha optional scale factor.
@param beta optional delta added to the scaled values.
@sa  Mat::convertTo, cv::abs(const Mat&)
*/
CV_EXPORTS_W void convertScaleAbs(InputArray src, OutputArray dst,
                                  double alpha = 1, double beta = 0);

/** @brief Converts an array to half precision floating number.

This function converts FP32 (single precision floating point) from/to FP16 (half precision floating point).  The input array has to have type of CV_32F or
CV_16S to represent the bit depth.  If the input array is neither of them, the function will raise an error.
The format of half precision floating point is defined in IEEE 754-2008.

@param src input array.
@param dst output array.
*/
CV_EXPORTS_W void convertFp16(InputArray src, OutputArray dst);

/** @brief Performs a look-up table transform of an array.

The function LUT fills the output array with values from the look-up table. Indices of the entries
are taken from the input array. That is, the function processes each element of src as follows:
\f[\texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}\f]
where
\f[d =  \fork{0}{if \(\texttt{src}\) has depth \(\texttt{CV_8U}\)}{128}{if \(\texttt{src}\) has depth \(\texttt{CV_8S}\)}\f]
@param src input array of 8-bit elements.
@param lut look-up table of 256 elements; in case of multi-channel input array, the table should
either have a single channel (in this case the same table is used for all channels) or the same
number of channels as in the input array.
@param dst output array of the same size and number of channels as src, and the same depth as lut.
@sa  convertScaleAbs, Mat::convertTo
*/
CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst);

/** @brief Calculates the sum of array elements.

The function cv::sum calculates and returns the sum of array elements,
independently for each channel.
@param src input array that must have from 1 to 4 channels.
@sa  countNonZero, mean, meanStdDev, norm, minMaxLoc, reduce
*/
CV_EXPORTS_AS(sumElems) Scalar sum(InputArray src);

/** @brief Counts non-zero array elements.

The function returns the number of non-zero elements in src :
\f[\sum _{I: \; \texttt{src} (I) \ne0 } 1\f]
@param src single-channel array.
@sa  mean, meanStdDev, norm, minMaxLoc, calcCovarMatrix
*/
CV_EXPORTS_W int countNonZero( InputArray src );

/** @brief Returns the list of locations of non-zero pixels

Given a binary matrix (likely returned from an operation such
as threshold(), compare(), >, ==, etc, return all of
the non-zero indices as a cv::Mat or std::vector<cv::Point> (x,y)
For example:
@code{.cpp}
    cv::Mat binaryImage; // input, binary image
    cv::Mat locations;   // output, locations of non-zero pixels
    cv::findNonZero(binaryImage, locations);

    // access pixel coordinates
    Point pnt = locations.at<Point>(i);
@endcode
or
@code{.cpp}
    cv::Mat binaryImage; // input, binary image
    vector<Point> locations;   // output, locations of non-zero pixels
    cv::findNonZero(binaryImage, locations);

    // access pixel coordinates
    Point pnt = locations[i];
@endcode
@param src single-channel array (type CV_8UC1)
@param idx the output array, type of cv::Mat or std::vector<Point>, corresponding to non-zero indices in the input
*/
CV_EXPORTS_W void findNonZero( InputArray src, OutputArray idx );

/** @brief Calculates an average (mean) of array elements.

The function cv::mean calculates the mean value M of array elements,
independently for each channel, and return it:
\f[\begin{array}{l} N =  \sum _{I: \; \texttt{mask} (I) \ne 0} 1 \\ M_c =  \left ( \sum _{I: \; \texttt{mask} (I) \ne 0}{ \texttt{mtx} (I)_c} \right )/N \end{array}\f]
When all the mask elements are 0's, the function returns Scalar::all(0)
@param src input array that should have from 1 to 4 channels so that the result can be stored in
Scalar_ .
@param mask optional operation mask.
@sa  countNonZero, meanStdDev, norm, minMaxLoc
*/
CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask = noArray());

/** Calculates a mean and standard deviation of array elements.

The function cv::meanStdDev calculates the mean and the standard deviation M
of array elements independently for each channel and returns it via the
output parameters:
\f[\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\f]
When all the mask elements are 0's, the function returns
mean=stddev=Scalar::all(0).
@note The calculated standard deviation is only the diagonal of the
complete normalized covariance matrix. If the full matrix is needed, you
can reshape the multi-channel array M x N to the single-channel array
M\*N x mtx.channels() (only possible when the matrix is continuous) and
then pass the matrix to calcCovarMatrix .
@param src input array that should have from 1 to 4 channels so that the results can be stored in
Scalar_ 's.
@param mean output parameter: calculated mean value.
@param stddev output parameter: calculated standard deviation.
@param mask optional operation mask.
@sa  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix
*/
CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev,
                             InputArray mask=noArray());

/** @brief Calculates an absolute array norm, an absolute difference norm, or a
relative difference norm.

The function cv::norm calculates an absolute norm of src1 (when there is no
src2 ):

\f[norm =  \forkthree{\|\texttt{src1}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
{ \| \texttt{src1} \| _{L_1} =  \sum _I | \texttt{src1} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
{ \| \texttt{src1} \| _{L_2} =  \sqrt{\sum_I \texttt{src1}(I)^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }\f]

or an absolute or relative difference norm if src2 is there:

\f[norm =  \forkthree{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}} =  \max _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_INF}\) }
{ \| \texttt{src1} - \texttt{src2} \| _{L_1} =  \sum _I | \texttt{src1} (I) -  \texttt{src2} (I)|}{if  \(\texttt{normType} = \texttt{NORM_L1}\) }
{ \| \texttt{src1} - \texttt{src2} \| _{L_2} =  \sqrt{\sum_I (\texttt{src1}(I) - \texttt{src2}(I))^2} }{if  \(\texttt{normType} = \texttt{NORM_L2}\) }\f]

or

\f[norm =  \forkthree{\frac{\|\texttt{src1}-\texttt{src2}\|_{L_{\infty}}    }{\|\texttt{src2}\|_{L_{\infty}} }}{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_INF}\) }
{ \frac{\|\texttt{src1}-\texttt{src2}\|_{L_1} }{\|\texttt{src2}\|_{L_1}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_L1}\) }
{ \frac{\|\texttt{src1}-\texttt{src2}\|_{L_2} }{\|\texttt{src2}\|_{L_2}} }{if  \(\texttt{normType} = \texttt{NORM_RELATIVE_L2}\) }\f]

The function cv::norm returns the calculated norm.

When the mask parameter is specified and it is not empty, the norm is
calculated only over the region specified by the mask.

A multi-channel input arrays are treated as a single-channel, that is,
the results for all channels are combined.

@param src1 first input array.
@param normType type of the norm (see cv::NormTypes).
@param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
*/
CV_EXPORTS_W double norm(InputArray src1, int normType = NORM_L2, InputArray mask = noArray());

/** @overload
@param src1 first input array.
@param src2 second input array of the same size and the same type as src1.
@param normType type of the norm (cv::NormTypes).
@param mask optional operation mask; it must have the same size as src1 and CV_8UC1 type.
*/
CV_EXPORTS_W double norm(InputArray src1, InputArray src2,
                         int normType = NORM_L2, InputArray mask = noArray());
/** @overload
@param src first input array.
@param normType type of the norm (see cv::NormTypes).
*/
CV_EXPORTS double norm( const SparseMat& src, int normType );

/** @brief computes PSNR image/video quality metric

see http://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio for details
@todo document
  */
CV_EXPORTS_W double PSNR(InputArray src1, InputArray src2);

/** @brief naive nearest neighbor finder

see http://en.wikipedia.org/wiki/Nearest_neighbor_search
@todo document
  */
CV_EXPORTS_W void batchDistance(InputArray src1, InputArray src2,
                                OutputArray dist, int dtype, OutputArray nidx,
                                int normType = NORM_L2, int K = 0,
                                InputArray mask = noArray(), int update = 0,
                                bool crosscheck = false);

/** @brief Normalizes the norm or value range of an array.

The function cv::normalize normalizes scale and shift the input array elements so that
\f[\| \texttt{dst} \| _{L_p}= \texttt{alpha}\f]
(where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that
\f[\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\f]

when normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be
normalized. This means that the norm or min-n-max are calculated over the sub-array, and then this
sub-array is modified to be normalized. If you want to only use the mask to calculate the norm or
min-max but modify the whole array, you can use norm and Mat::convertTo.

In case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,
the range transformation for sparse matrices is not allowed since it can shift the zero level.

Possible usage with some positive example data:
@code{.cpp}
    vector<double> positiveData = { 2.0, 8.0, 10.0 };
    vector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;

    // Norm to probability (total count)
    // sum(numbers) = 20.0
    // 2.0      0.1     (2.0/20.0)
    // 8.0      0.4     (8.0/20.0)
    // 10.0     0.5     (10.0/20.0)
    normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);

    // Norm to unit vector: ||positiveData|| = 1.0
    // 2.0      0.15
    // 8.0      0.62
    // 10.0     0.77
    normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);

    // Norm to max element
    // 2.0      0.2     (2.0/10.0)
    // 8.0      0.8     (8.0/10.0)
    // 10.0     1.0     (10.0/10.0)
    normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);

    // Norm to range [0.0;1.0]
    // 2.0      0.0     (shift to left border)
    // 8.0      0.75    (6.0/8.0)
    // 10.0     1.0     (shift to right border)
    normalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
@endcode

@param src input array.
@param dst output array of the same size as src .
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param beta upper range boundary in case of the range normalization; it is not used for the norm
normalization.
@param norm_type normalization type (see cv::NormTypes).
@param dtype when negative, the output array has the same type as src; otherwise, it has the same
number of channels as src and the depth =CV_MAT_DEPTH(dtype).
@param mask optional operation mask.
@sa norm, Mat::convertTo, SparseMat::convertTo
*/
CV_EXPORTS_W void normalize( InputArray src, InputOutputArray dst, double alpha = 1, double beta = 0,
                             int norm_type = NORM_L2, int dtype = -1, InputArray mask = noArray());

/** @overload
@param src input array.
@param dst output array of the same size as src .
@param alpha norm value to normalize to or the lower range boundary in case of the range
normalization.
@param normType normalization type (see cv::NormTypes).
*/
CV_EXPORTS void normalize( const SparseMat& src, SparseMat& dst, double alpha, int normType );

/** @brief Finds the global minimum and maximum in an array.

The function cv::minMaxLoc finds the minimum and maximum element values and their positions. The
extremums are searched across the whole array or, if mask is not an empty array, in the specified
array region.

The function do not work with multi-channel arrays. If you need to find minimum or maximum
elements across all the channels, use Mat::reshape first to reinterpret the array as
single-channel. Or you may extract the particular channel using either extractImageCOI , or
mixChannels , or split .
@param src input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minLoc pointer to the returned minimum location (in 2D case); NULL is used if not required.
@param maxLoc pointer to the returned maximum location (in 2D case); NULL is used if not required.
@param mask optional mask used to select a sub-array.
@sa max, min, compare, inRange, extractImageCOI, mixChannels, split, Mat::reshape
*/
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                            CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                            CV_OUT Point* maxLoc = 0, InputArray mask = noArray());


/** @brief Finds the global minimum and maximum in an array

The function cv::minMaxIdx finds the minimum and maximum element values and their positions. The
extremums are searched across the whole array or, if mask is not an empty array, in the specified
array region. The function does not work with multi-channel arrays. If you need to find minimum or
maximum elements across all the channels, use Mat::reshape first to reinterpret the array as
single-channel. Or you may extract the particular channel using either extractImageCOI , or
mixChannels , or split . In case of a sparse matrix, the minimum is found among non-zero elements
only.
@note When minIdx is not NULL, it must have at least 2 elements (as well as maxIdx), even if src is
a single-row or single-column matrix. In OpenCV (following MATLAB) each array has at least 2
dimensions, i.e. single-column matrix is Mx1 matrix (and therefore minIdx/maxIdx will be
(i1,0)/(i2,0)) and single-row matrix is 1xN matrix (and therefore minIdx/maxIdx will be
(0,j1)/(0,j2)).
@param src input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minIdx pointer to the returned minimum location (in nD case); NULL is used if not required;
Otherwise, it must point to an array of src.dims elements, the coordinates of the minimum element
in each dimension are stored there sequentially.
@param maxIdx pointer to the returned maximum location (in nD case). NULL is used if not required.
@param mask specified array region
*/
CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal = 0,
                          int* minIdx = 0, int* maxIdx = 0, InputArray mask = noArray());

/** @overload
@param a input single-channel array.
@param minVal pointer to the returned minimum value; NULL is used if not required.
@param maxVal pointer to the returned maximum value; NULL is used if not required.
@param minIdx pointer to the returned minimum location (in nD case); NULL is used if not required;
Otherwise, it must point to an array of src.dims elements, the coordinates of the minimum element
in each dimension are stored there sequentially.
@param maxIdx pointer to the returned maximum location (in nD case). NULL is used if not required.
*/
CV_EXPORTS void minMaxLoc(const SparseMat& a, double* minVal,
                          double* maxVal, int* minIdx = 0, int* maxIdx = 0);

/** @brief Reduces a matrix to a vector.

The function cv::reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of
1D vectors and performing the specified operation on the vectors until a single row/column is
obtained. For example, the function can be used to compute horizontal and vertical projections of a
raster image. In case of REDUCE_MAX and REDUCE_MIN , the output image should have the same type as the source one.
In case of REDUCE_SUM and REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy.
And multi-channel arrays are also supported in these two reduction modes.
@param src input 2D matrix.
@param dst output vector. Its size and type is defined by dim and dtype parameters.
@param dim dimension index along which the matrix is reduced. 0 means that the matrix is reduced to
a single row. 1 means that the matrix is reduced to a single column.
@param rtype reduction operation that could be one of cv::ReduceTypes
@param dtype when negative, the output vector will have the same type as the input matrix,
otherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()).
@sa repeat
*/
CV_EXPORTS_W void reduce(InputArray src, OutputArray dst, int dim, int rtype, int dtype = -1);

/** @brief Creates one multi-channel array out of several single-channel ones.

The function cv::merge merges several arrays to make a single multi-channel array. That is, each
element of the output array will be a concatenation of the elements of the input arrays, where
elements of i-th input array are treated as mv[i].channels()-element vectors.

The function cv::split does the reverse operation. If you need to shuffle channels in some other
advanced way, use cv::mixChannels.
@param mv input array of matrices to be merged; all the matrices in mv must have the same
size and the same depth.
@param count number of input matrices when mv is a plain C array; it must be greater than zero.
@param dst output array of the same size and the same depth as mv[0]; The number of channels will
be equal to the parameter count.
@sa  mixChannels, split, Mat::reshape
*/
CV_EXPORTS void merge(const Mat* mv, size_t count, OutputArray dst);

/** @overload
@param mv input vector of matrices to be merged; all the matrices in mv must have the same
size and the same depth.
@param dst output array of the same size and the same depth as mv[0]; The number of channels will
be the total number of channels in the matrix array.
  */
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);

/** @brief Divides a multi-channel array into several single-channel arrays.

The function cv::split splits a multi-channel array into separate single-channel arrays:
\f[\texttt{mv} [c](I) =  \texttt{src} (I)_c\f]
If you need to extract a single channel or do some other sophisticated channel permutation, use
mixChannels .
@param src input multi-channel array.
@param mvbegin output array; the number of arrays must match src.channels(); the arrays themselves are
reallocated, if needed.
@sa merge, mixChannels, cvtColor
*/
CV_EXPORTS void split(const Mat& src, Mat* mvbegin);

/** @overload
@param m input multi-channel array.
@param mv output vector of arrays; the arrays themselves are reallocated, if needed.
*/
CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);

/** @brief Copies specified channels from input arrays to the specified channels of
output arrays.

The function cv::mixChannels provides an advanced mechanism for shuffling image channels.

cv::split,cv::merge,cv::extractChannel,cv::insertChannel and some forms of cv::cvtColor are partial cases of cv::mixChannels.

In the example below, the code splits a 4-channel BGRA image into a 3-channel BGR (with B and R
channels swapped) and a separate alpha-channel image:
@code{.cpp}
    Mat bgra( 100, 100, CV_8UC4, Scalar(255,0,0,255) );
    Mat bgr( bgra.rows, bgra.cols, CV_8UC3 );
    Mat alpha( bgra.rows, bgra.cols, CV_8UC1 );

    // forming an array of matrices is a quite efficient operation,
    // because the matrix data is not copied, only the headers
    Mat out[] = { bgr, alpha };
    // bgra[0] -> bgr[2], bgra[1] -> bgr[1],
    // bgra[2] -> bgr[0], bgra[3] -> alpha[0]
    int from_to[] = { 0,2, 1,1, 2,0, 3,3 };
    mixChannels( &bgra, 1, out, 2, from_to, 4 );
@endcode
@note Unlike many other new-style C++ functions in OpenCV (see the introduction section and
Mat::create ), cv::mixChannels requires the output arrays to be pre-allocated before calling the
function.
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param nsrcs number of matrices in `src`.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in `src[0]`.
@param ndsts number of matrices in `dst`.
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
@param npairs number of index pairs in `fromTo`.
@sa split, merge, extractChannel, insertChannel, cvtColor
*/
CV_EXPORTS void mixChannels(const Mat* src, size_t nsrcs, Mat* dst, size_t ndsts,
                            const int* fromTo, size_t npairs);

/** @overload
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in src[0].
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
@param npairs number of index pairs in fromTo.
*/
CV_EXPORTS void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                            const int* fromTo, size_t npairs);

/** @overload
@param src input array or vector of matrices; all of the matrices must have the same size and the
same depth.
@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and
depth must be the same as in src[0].
@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is
a 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in
dst; the continuous channel numbering is used: the first input image channels are indexed from 0 to
src[0].channels()-1, the second input image channels are indexed from src[0].channels() to
src[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image
channels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is
filled with zero .
*/
CV_EXPORTS_W void mixChannels(InputArrayOfArrays src, InputOutputArrayOfArrays dst,
                              const std::vector<int>& fromTo);

/** @brief Extracts a single channel from src (coi is 0-based index)
@param src input array
@param dst output array
@param coi index of channel to extract
@sa mixChannels, split
*/
CV_EXPORTS_W void extractChannel(InputArray src, OutputArray dst, int coi);

/** @brief Inserts a single channel to dst (coi is 0-based index)
@param src input array
@param dst output array
@param coi index of channel for insertion
@sa mixChannels, merge
*/
CV_EXPORTS_W void insertChannel(InputArray src, InputOutputArray dst, int coi);

/** @brief Flips a 2D array around vertical, horizontal, or both axes.

The function cv::flip flips the array in one of three different ways (row
and column indices are 0-based):
\f[\texttt{dst} _{ij} =
\left\{
\begin{array}{l l}
\texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\
\texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\
\texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\
\end{array}
\right.\f]
The example scenarios of using the function are the following:
*   Vertical flipping of the image (flipCode == 0) to switch between
    top-left and bottom-left image origin. This is a typical operation
    in video processing on Microsoft Windows\* OS.
*   Horizontal flipping of the image with the subsequent horizontal
    shift and absolute difference calculation to check for a
    vertical-axis symmetry (flipCode \> 0).
*   Simultaneous horizontal and vertical flipping of the image with
    the subsequent shift and absolute difference calculation to check
    for a central symmetry (flipCode \< 0).
*   Reversing the order of point arrays (flipCode \> 0 or
    flipCode == 0).
@param src input array.
@param dst output array of the same size and type as src.
@param flipCode a flag to specify how to flip the array; 0 means
flipping around the x-axis and positive value (for example, 1) means
flipping around y-axis. Negative value (for example, -1) means flipping
around both axes.
@sa transpose , repeat , completeSymm
*/
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);

enum RotateFlags {
    ROTATE_90_CLOCKWISE = 0, //Rotate 90 degrees clockwise
    ROTATE_180 = 1, //Rotate 180 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, //Rotate 270 degrees clockwise
};
/** @brief Rotates a 2D array in multiples of 90 degrees.
The function rotate rotates the array in one of three different ways:
*   Rotate by 90 degrees clockwise (rotateCode = ROTATE_90).
*   Rotate by 180 degrees clockwise (rotateCode = ROTATE_180).
*   Rotate by 270 degrees clockwise (rotateCode = ROTATE_270).
@param src input array.
@param dst output array of the same type as src.  The size is the same with ROTATE_180,
and the rows and cols are switched for ROTATE_90 and ROTATE_270.
@param rotateCode an enum to specify how to rotate the array; see the enum RotateFlags
@sa transpose , repeat , completeSymm, flip, RotateFlags
*/
CV_EXPORTS_W void rotate(InputArray src, OutputArray dst, int rotateCode);

/** @brief Fills the output array with repeated copies of the input array.

The function cv::repeat duplicates the input array one or more times along each of the two axes:
\f[\texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }\f]
The second variant of the function is more convenient to use with @ref MatrixExpressions.
@param src input array to replicate.
@param ny Flag to specify how many times the `src` is repeated along the
vertical axis.
@param nx Flag to specify how many times the `src` is repeated along the
horizontal axis.
@param dst output array of the same type as `src`.
@sa cv::reduce
*/
CV_EXPORTS_W void repeat(InputArray src, int ny, int nx, OutputArray dst);

/** @overload
@param src input array to replicate.
@param ny Flag to specify how many times the `src` is repeated along the
vertical axis.
@param nx Flag to specify how many times the `src` is repeated along the
horizontal axis.
  */
CV_EXPORTS Mat repeat(const Mat& src, int ny, int nx);

/** @brief Applies horizontal concatenation to given matrices.

The function horizontally concatenates two or more cv::Mat matrices (with the same number of rows).
@code{.cpp}
    cv::Mat matArray[] = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
                           cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
                           cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::hconcat( matArray, 3, out );
    //out:
    //[1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3]
@endcode
@param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
@param nsrc number of matrices in src.
@param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
@sa cv::vconcat(const Mat*, size_t, OutputArray), @sa cv::vconcat(InputArrayOfArrays, OutputArray) and @sa cv::vconcat(InputArray, InputArray, OutputArray)
*/
CV_EXPORTS void hconcat(const Mat* src, size_t nsrc, OutputArray dst);
/** @overload
 @code{.cpp}
    cv::Mat_<float> A = (cv::Mat_<float>(3, 2) << 1, 4,
                                                  2, 5,
                                                  3, 6);
    cv::Mat_<float> B = (cv::Mat_<float>(3, 2) << 7, 10,
                                                  8, 11,
                                                  9, 12);

    cv::Mat C;
    cv::hconcat(A, B, C);
    //C:
    //[1, 4, 7, 10;
    // 2, 5, 8, 11;
    // 3, 6, 9, 12]
 @endcode
 @param src1 first input array to be considered for horizontal concatenation.
 @param src2 second input array to be considered for horizontal concatenation.
 @param dst output array. It has the same number of rows and depth as the src1 and src2, and the sum of cols of the src1 and src2.
 */
CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
 @code{.cpp}
    std::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),
                                      cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),
                                      cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::hconcat( matrices, out );
    //out:
    //[1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3;
    // 1, 2, 3]
 @endcode
 @param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.
 @param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.
same depth.
 */
CV_EXPORTS_W void hconcat(InputArrayOfArrays src, OutputArray dst);

/** @brief Applies vertical concatenation to given matrices.

The function vertically concatenates two or more cv::Mat matrices (with the same number of cols).
@code{.cpp}
    cv::Mat matArray[] = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
                           cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
                           cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::vconcat( matArray, 3, out );
    //out:
    //[1,   1,   1,   1;
    // 2,   2,   2,   2;
    // 3,   3,   3,   3]
@endcode
@param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth.
@param nsrc number of matrices in src.
@param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
@sa cv::hconcat(const Mat*, size_t, OutputArray), @sa cv::hconcat(InputArrayOfArrays, OutputArray) and @sa cv::hconcat(InputArray, InputArray, OutputArray)
*/
CV_EXPORTS void vconcat(const Mat* src, size_t nsrc, OutputArray dst);
/** @overload
 @code{.cpp}
    cv::Mat_<float> A = (cv::Mat_<float>(3, 2) << 1, 7,
                                                  2, 8,
                                                  3, 9);
    cv::Mat_<float> B = (cv::Mat_<float>(3, 2) << 4, 10,
                                                  5, 11,
                                                  6, 12);

    cv::Mat C;
    cv::vconcat(A, B, C);
    //C:
    //[1, 7;
    // 2, 8;
    // 3, 9;
    // 4, 10;
    // 5, 11;
    // 6, 12]
 @endcode
 @param src1 first input array to be considered for vertical concatenation.
 @param src2 second input array to be considered for vertical concatenation.
 @param dst output array. It has the same number of cols and depth as the src1 and src2, and the sum of rows of the src1 and src2.
 */
CV_EXPORTS void vconcat(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
 @code{.cpp}
    std::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),
                                      cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),
                                      cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};

    cv::Mat out;
    cv::vconcat( matrices, out );
    //out:
    //[1,   1,   1,   1;
    // 2,   2,   2,   2;
    // 3,   3,   3,   3]
 @endcode
 @param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth
 @param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.
same depth.
 */
CV_EXPORTS_W void vconcat(InputArrayOfArrays src, OutputArray dst);

/** @brief computes bitwise conjunction of the two arrays (dst = src1 & src2)
Calculates the per-element bit-wise conjunction of two arrays or an
array and a scalar.

The function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the second and third cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

/** @brief Calculates the per-element bit-wise disjunction of two arrays or an
array and a scalar.

The function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the second and third cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2,
                             OutputArray dst, InputArray mask = noArray());

/** @brief Calculates the per-element bit-wise "exclusive or" operation on two
arrays or an array and a scalar.

The function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"
operation for:
*   Two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
*   An array and a scalar when src2 is constructed from Scalar or has
    the same number of elements as `src1.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]
*   A scalar and an array when src1 is constructed from Scalar or has
    the same number of elements as `src2.channels()`:
    \f[\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]
In case of floating-point arrays, their machine-specific bit
representations (usually IEEE754-compliant) are used for the operation.
In case of multi-channel arrays, each channel is processed
independently. In the 2nd and 3rd cases above, the scalar is first
converted to the array type.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as the input
arrays.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2,
                              OutputArray dst, InputArray mask = noArray());

/** @brief  Inverts every bit of an array.

The function cv::bitwise_not calculates per-element bit-wise inversion of the input
array:
\f[\texttt{dst} (I) =  \neg \texttt{src} (I)\f]
In case of a floating-point input array, its machine-specific bit
representation (usually IEEE754-compliant) is used for the operation. In
case of multi-channel arrays, each channel is processed independently.
@param src input array.
@param dst output array that has the same size and type as the input
array.
@param mask optional operation mask, 8-bit single channel array, that
specifies elements of the output array to be changed.
*/
CV_EXPORTS_W void bitwise_not(InputArray src, OutputArray dst,
                              InputArray mask = noArray());

/** @brief Calculates the per-element absolute difference between two arrays or between an array and a scalar.

The function cv::absdiff calculates:
*   Absolute difference between two arrays when they have the same
    size and type:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\f]
*   Absolute difference between an array and a scalar when the second
    array is constructed from Scalar or has as many elements as the
    number of channels in `src1`:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)\f]
*   Absolute difference between a scalar and an array when the first
    array is constructed from Scalar or has as many elements as the
    number of channels in `src2`:
    \f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)\f]
    where I is a multi-dimensional index of array elements. In case of
    multi-channel arrays, each channel is processed independently.
@note Saturation is not applied when the arrays have the depth CV_32S.
You may even get a negative value in the case of overflow.
@param src1 first input array or a scalar.
@param src2 second input array or a scalar.
@param dst output array that has the same size and type as input arrays.
@sa cv::abs(const Mat&)
*/
CV_EXPORTS_W void absdiff(InputArray src1, InputArray src2, OutputArray dst);

/** @brief  Checks if array elements lie between the elements of two other arrays.

The function checks the range as follows:
-   For every element of a single-channel input array:
    \f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0\f]
-   For two-channel arrays:
    \f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 \leq  \texttt{upperb} (I)_1\f]
-   and so forth.

That is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the
specified 1D, 2D, 3D, ... box and 0 otherwise.

When the lower and/or upper boundary parameters are scalars, the indexes
(I) at lowerb and upperb in the above formulas should be omitted.
@param src first input array.
@param lowerb inclusive lower boundary array or a scalar.
@param upperb inclusive upper boundary array or a scalar.
@param dst output array of the same size as src and CV_8U type.
*/
CV_EXPORTS_W void inRange(InputArray src, InputArray lowerb,
                          InputArray upperb, OutputArray dst);

/** @brief Performs the per-element comparison of two arrays or an array and scalar value.

The function compares:
*   Elements of two arrays when src1 and src2 have the same size:
    \f[\texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)\f]
*   Elements of src1 with a scalar src2 when src2 is constructed from
    Scalar or has a single element:
    \f[\texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}\f]
*   src1 with elements of src2 when src1 is constructed from Scalar or
    has a single element:
    \f[\texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)\f]
When the comparison result is true, the corresponding element of output
array is set to 255. The comparison operations can be replaced with the
equivalent matrix expressions:
@code{.cpp}
    Mat dst1 = src1 >= src2;
    Mat dst2 = src1 < 8;
    ...
@endcode
@param src1 first input array or a scalar; when it is an array, it must have a single channel.
@param src2 second input array or a scalar; when it is an array, it must have a single channel.
@param dst output array of type ref CV_8U that has the same size and the same number of channels as
    the input arrays.
@param cmpop a flag, that specifies correspondence between the arrays (cv::CmpTypes)
@sa checkRange, min, max, threshold
*/
CV_EXPORTS_W void compare(InputArray src1, InputArray src2, OutputArray dst, int cmpop);

/** @brief Calculates per-element minimum of two arrays or an array and a scalar.

The function cv::min calculates the per-element minimum of two arrays:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\f]
or array and a scalar:
\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )\f]
@param src1 first input array.
@param src2 second input array of the same size and type as src1.
@param dst output array of the same size and type as src1.
@sa max, compare, inRange, minMaxLoc
*/
CV_EXPORTS_W void min(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void min(const Mat& src1, const Mat& src2, Mat& dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void min(const UMat& src1, const UMat& src2, UMat& dst);

/** @brief Calculates per-element maximum of two arrays or an array and a scalar.

The function cv::max calculates the per-element maximum of two arrays:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\f]
or array and a scalar:
\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )\f]
@param src1 first input array.
@param src2 second input array of the same size and type as src1 .
@param dst output array of the same size and type as src1.
@sa  min, compare, inRange, minMaxLoc, @ref MatrixExpressions
*/
CV_EXPORTS_W void max(InputArray src1, InputArray src2, OutputArray dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void max(const Mat& src1, const Mat& src2, Mat& dst);
/** @overload
needed to avoid conflicts with const _Tp& std::min(const _Tp&, const _Tp&, _Compare)
*/
CV_EXPORTS void max(const UMat& src1, const UMat& src2, UMat& dst);

/** @brief Calculates a square root of array elements.

The function cv::sqrt calculates a square root of each input array element.
In case of multi-channel arrays, each channel is processed
independently. The accuracy is approximately the same as of the built-in
std::sqrt .
@param src input floating-point array.
@param dst output array of the same size and type as src.
*/
CV_EXPORTS_W void sqrt(InputArray src, OutputArray dst);

/** @brief Raises every array element to a power.

The function cv::pow raises every element of the input array to power :
\f[\texttt{dst} (I) =  \fork{\texttt{src}(I)^{power}}{if \(\texttt{power}\) is integer}{|\texttt{src}(I)|^{power}}{otherwise}\f]

So, for a non-integer power exponent, the absolute values of input array
elements are used. However, it is possible to get true values for
negative values using some extra operations. In the example below,
computing the 5th root of array src shows:
@code{.cpp}
    Mat mask = src < 0;
    pow(src, 1./5, dst);
    subtract(Scalar::all(0), dst, dst, mask);
@endcode
For some values of power, such as integer values, 0.5 and -0.5,
specialized faster algorithms are used.

Special values (NaN, Inf) are not handled.
@param src input array.
@param power exponent of power.
@param dst output array of the same size and type as src.
@sa sqrt, exp, log, cartToPolar, polarToCart
*/
CV_EXPORTS_W void pow(InputArray src, double power, OutputArray dst);

/** @brief Calculates the exponent of every array element.

The function cv::exp calculates the exponent of every element of the input
array:
\f[\texttt{dst} [I] = e^{ src(I) }\f]

The maximum relative error is about 7e-6 for single-precision input and
less than 1e-10 for double-precision input. Currently, the function
converts denormalized values to zeros on output. Special values (NaN,
Inf) are not handled.
@param src input array.
@param dst output array of the same size and type as src.
@sa log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude
*/
CV_EXPORTS_W void exp(InputArray src, OutputArray dst);

/** @brief Calculates the natural logarithm of every array element.

The function cv::log calculates the natural logarithm of every element of the input array:
\f[\texttt{dst} (I) =  \log (\texttt{src}(I)) \f]

Output on zero, negative and special (NaN, Inf) values is undefined.

@param src input array.
@param dst output array of the same size and type as src .
@sa exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude
*/
CV_EXPORTS_W void log(InputArray src, OutputArray dst);

/** @brief Calculates x and y coordinates of 2D vectors from their magnitude and angle.

The function cv::polarToCart calculates the Cartesian coordinates of each 2D
vector represented by the corresponding elements of magnitude and angle:
\f[\begin{array}{l} \texttt{x} (I) =  \texttt{magnitude} (I) \cos ( \texttt{angle} (I)) \\ \texttt{y} (I) =  \texttt{magnitude} (I) \sin ( \texttt{angle} (I)) \\ \end{array}\f]

The relative accuracy of the estimated coordinates is about 1e-6.
@param magnitude input floating-point array of magnitudes of 2D vectors;
it can be an empty matrix (=Mat()), in this case, the function assumes
that all the magnitudes are =1; if it is not empty, it must have the
same size and type as angle.
@param angle input floating-point array of angles of 2D vectors.
@param x output array of x-coordinates of 2D vectors; it has the same
size and type as angle.
@param y output array of y-coordinates of 2D vectors; it has the same
size and type as angle.
@param angleInDegrees when true, the input angles are measured in
degrees, otherwise, they are measured in radians.
@sa cartToPolar, magnitude, phase, exp, log, pow, sqrt
*/
CV_EXPORTS_W void polarToCart(InputArray magnitude, InputArray angle,
                              OutputArray x, OutputArray y, bool angleInDegrees = false);

/** @brief Calculates the magnitude and angle of 2D vectors.

The function cv::cartToPolar calculates either the magnitude, angle, or both
for every 2D vector (x(I),y(I)):
\f[\begin{array}{l} \texttt{magnitude} (I)= \sqrt{\texttt{x}(I)^2+\texttt{y}(I)^2} , \\ \texttt{angle} (I)= \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))[ \cdot180 / \pi ] \end{array}\f]

The angles are calculated with accuracy about 0.3 degrees. For the point
(0,0), the angle is set to 0.
@param x array of x-coordinates; this must be a single-precision or
double-precision floating-point array.
@param y array of y-coordinates, that must have the same size and same type as x.
@param magnitude output array of magnitudes of the same size and type as x.
@param angle output array of angles that has the same size and type as
x; the angles are measured in radians (from 0 to 2\*Pi) or in degrees (0 to 360 degrees).
@param angleInDegrees a flag, indicating whether the angles are measured
in radians (which is by default), or in degrees.
@sa Sobel, Scharr
*/
CV_EXPORTS_W void cartToPolar(InputArray x, InputArray y,
                              OutputArray magnitude, OutputArray angle,
                              bool angleInDegrees = false);

/** @brief Calculates the rotation angle of 2D vectors.

The function cv::phase calculates the rotation angle of each 2D vector that
is formed from the corresponding elements of x and y :
\f[\texttt{angle} (I) =  \texttt{atan2} ( \texttt{y} (I), \texttt{x} (I))\f]

The angle estimation accuracy is about 0.3 degrees. When x(I)=y(I)=0 ,
the corresponding angle(I) is set to 0.
@param x input floating-point array of x-coordinates of 2D vectors.
@param y input array of y-coordinates of 2D vectors; it must have the
same size and the same type as x.
@param angle output array of vector angles; it has the same size and
same type as x .
@param angleInDegrees when true, the function calculates the angle in
degrees, otherwise, they are measured in radians.
*/
CV_EXPORTS_W void phase(InputArray x, InputArray y, OutputArray angle,
                        bool angleInDegrees = false);

/** @brief Calculates the magnitude of 2D vectors.

The function cv::magnitude calculates the magnitude of 2D vectors formed
from the corresponding elements of x and y arrays:
\f[\texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}\f]
@param x floating-point array of x-coordinates of the vectors.
@param y floating-point array of y-coordinates of the vectors; it must
have the same size as x.
@param magnitude output array of the same size and type as x.
@sa cartToPolar, polarToCart, phase, sqrt
*/
CV_EXPORTS_W void magnitude(InputArray x, InputArray y, OutputArray magnitude);

/** @brief Checks every element of an input array for invalid values.

The function cv::checkRange checks that every array element is neither NaN nor infinite. When minVal \>
-DBL_MAX and maxVal \< DBL_MAX, the function also checks that each value is between minVal and
maxVal. In case of multi-channel arrays, each channel is processed independently. If some values
are out of range, position of the first outlier is stored in pos (when pos != NULL). Then, the
function either returns false (when quiet=true) or throws an exception.
@param a input array.
@param quiet a flag, indicating whether the functions quietly return false when the array elements
are out of range or they throw an exception.
@param pos optional output parameter, when not NULL, must be a pointer to array of src.dims
elements.
@param minVal inclusive lower boundary of valid values range.
@param maxVal exclusive upper boundary of valid values range.
*/
CV_EXPORTS_W bool checkRange(InputArray a, bool quiet = true, CV_OUT Point* pos = 0,
                            double minVal = -DBL_MAX, double maxVal = DBL_MAX);

/** @brief converts NaN's to the given number
*/
CV_EXPORTS_W void patchNaNs(InputOutputArray a, double val = 0);

/** @brief Performs generalized matrix multiplication.

The function cv::gemm performs generalized matrix multiplication similar to the
gemm functions in BLAS level 3. For example,
`gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T)`
corresponds to
\f[\texttt{dst} =  \texttt{alpha} \cdot \texttt{src1} ^T  \cdot \texttt{src2} +  \texttt{beta} \cdot \texttt{src3} ^T\f]

In case of complex (two-channel) data, performed a complex matrix
multiplication.

The function can be replaced with a matrix expression. For example, the
above call can be replaced with:
@code{.cpp}
    dst = alpha*src1.t()*src2 + beta*src3.t();
@endcode
@param src1 first multiplied input matrix that could be real(CV_32FC1,
CV_64FC1) or complex(CV_32FC2, CV_64FC2).
@param src2 second multiplied input matrix of the same type as src1.
@param alpha weight of the matrix product.
@param src3 third optional delta matrix added to the matrix product; it
should have the same type as src1 and src2.
@param beta weight of src3.
@param dst output matrix; it has the proper size and the same type as
input matrices.
@param flags operation flags (cv::GemmFlags)
@sa mulTransposed , transform
*/
CV_EXPORTS_W void gemm(InputArray src1, InputArray src2, double alpha,
                       InputArray src3, double beta, OutputArray dst, int flags = 0);

/** @brief Calculates the product of a matrix and its transposition.

The function cv::mulTransposed calculates the product of src and its
transposition:
\f[\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} )^T ( \texttt{src} - \texttt{delta} )\f]
if aTa=true , and
\f[\texttt{dst} = \texttt{scale} ( \texttt{src} - \texttt{delta} ) ( \texttt{src} - \texttt{delta} )^T\f]
otherwise. The function is used to calculate the covariance matrix. With
zero delta, it can be used as a faster substitute for general matrix
product A\*B when B=A'
@param src input single-channel matrix. Note that unlike gemm, the
function can multiply not only floating-point matrices.
@param dst output square matrix.
@param aTa Flag specifying the multiplication ordering. See the
description below.
@param delta Optional delta matrix subtracted from src before the
multiplication. When the matrix is empty ( delta=noArray() ), it is
assumed to be zero, that is, nothing is subtracted. If it has the same
size as src , it is simply subtracted. Otherwise, it is "repeated" (see
repeat ) to cover the full src and then subtracted. Type of the delta
matrix, when it is not empty, must be the same as the type of created
output matrix. See the dtype parameter description below.
@param scale Optional scale factor for the matrix product.
@param dtype Optional type of the output matrix. When it is negative,
the output matrix will have the same type as src . Otherwise, it will be
type=CV_MAT_DEPTH(dtype) that should be either CV_32F or CV_64F .
@sa calcCovarMatrix, gemm, repeat, reduce
*/
CV_EXPORTS_W void mulTransposed( InputArray src, OutputArray dst, bool aTa,
                                 InputArray delta = noArray(),
                                 double scale = 1, int dtype = -1 );

/** @brief Transposes a matrix.

The function cv::transpose transposes the matrix src :
\f[\texttt{dst} (i,j) =  \texttt{src} (j,i)\f]
@note No complex conjugation is done in case of a complex matrix. It
should be done separately if needed.
@param src input array.
@param dst output array of the same type as src.
*/
CV_EXPORTS_W void transpose(InputArray src, OutputArray dst);

/** @brief Performs the matrix transformation of every array element.

The function cv::transform performs the matrix transformation of every
element of the array src and stores the results in dst :
\f[\texttt{dst} (I) =  \texttt{m} \cdot \texttt{src} (I)\f]
(when m.cols=src.channels() ), or
\f[\texttt{dst} (I) =  \texttt{m} \cdot [ \texttt{src} (I); 1]\f]
(when m.cols=src.channels()+1 )

Every element of the N -channel array src is interpreted as N -element
vector that is transformed using the M x N or M x (N+1) matrix m to
M-element vector - the corresponding element of the output array dst .

The function may be used for geometrical transformation of
N -dimensional points, arbitrary linear color space transformation (such
as various kinds of RGB to YUV transforms), shuffling the image
channels, and so forth.
@param src input array that must have as many channels (1 to 4) as
m.cols or m.cols-1.
@param dst output array of the same size and depth as src; it has as
many channels as m.rows.
@param m transformation 2x2 or 2x3 floating-point matrix.
@sa perspectiveTransform, getAffineTransform, estimateAffine2D, warpAffine, warpPerspective
*/
CV_EXPORTS_W void transform(InputArray src, OutputArray dst, InputArray m );

/** @brief Performs the perspective matrix transformation of vectors.

The function cv::perspectiveTransform transforms every element of src by
treating it as a 2D or 3D vector, in the following way:
\f[(x, y, z)  \rightarrow (x'/w, y'/w, z'/w)\f]
where
\f[(x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}\f]
and
\f[w =  \fork{w'}{if \(w' \ne 0\)}{\infty}{otherwise}\f]

Here a 3D vector transformation is shown. In case of a 2D vector
transformation, the z component is omitted.

@note The function transforms a sparse set of 2D or 3D vectors. If you
want to transform an image using perspective transformation, use
warpPerspective . If you have an inverse problem, that is, you want to
compute the most probable perspective transformation out of several
pairs of corresponding points, you can use getPerspectiveTransform or
findHomography .
@param src input two-channel or three-channel floating-point array; each
element is a 2D/3D vector to be transformed.
@param dst output array of the same size and type as src.
@param m 3x3 or 4x4 floating-point transformation matrix.
@sa  transform, warpPerspective, getPerspectiveTransform, findHomography
*/
CV_EXPORTS_W void perspectiveTransform(InputArray src, OutputArray dst, InputArray m );

/** @brief Copies the lower or the upper half of a square matrix to another half.

The function cv::completeSymm copies the lower half of a square matrix to
its another half. The matrix diagonal remains unchanged:
*   \f$\texttt{mtx}_{ij}=\texttt{mtx}_{ji}\f$ for \f$i > j\f$ if
    lowerToUpper=false
*   \f$\texttt{mtx}_{ij}=\texttt{mtx}_{ji}\f$ for \f$i < j\f$ if
    lowerToUpper=true
@param mtx input-output floating-point square matrix.
@param lowerToUpper operation flag; if true, the lower half is copied to
the upper half. Otherwise, the upper half is copied to the lower half.
@sa flip, transpose
*/
CV_EXPORTS_W void completeSymm(InputOutputArray mtx, bool lowerToUpper = false);

/** @brief Initializes a scaled identity matrix.

The function cv::setIdentity initializes a scaled identity matrix:
\f[\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}\f]

The function can also be emulated using the matrix initializers and the
matrix expressions:
@code
    Mat A = Mat::eye(4, 3, CV_32F)*5;
    // A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]
@endcode
@param mtx matrix to initialize (not necessarily square).
@param s value to assign to diagonal elements.
@sa Mat::zeros, Mat::ones, Mat::setTo, Mat::operator=
*/
CV_EXPORTS_W void setIdentity(InputOutputArray mtx, const Scalar& s = Scalar(1));

/** @brief Returns the determinant of a square floating-point matrix.

The function cv::determinant calculates and returns the determinant of the
specified matrix. For small matrices ( mtx.cols=mtx.rows\<=3 ), the
direct method is used. For larger matrices, the function uses LU
factorization with partial pivoting.

For symmetric positively-determined matrices, it is also possible to use
eigen decomposition to calculate the determinant.
@param mtx input matrix that must have CV_32FC1 or CV_64FC1 type and
square size.
@sa trace, invert, solve, eigen, @ref MatrixExpressions
*/
CV_EXPORTS_W double determinant(InputArray mtx);

/** @brief Returns the trace of a matrix.

The function cv::trace returns the sum of the diagonal elements of the
matrix mtx .
\f[\mathrm{tr} ( \texttt{mtx} ) =  \sum _i  \texttt{mtx} (i,i)\f]
@param mtx input matrix.
*/
CV_EXPORTS_W Scalar trace(InputArray mtx);

/** @brief Finds the inverse or pseudo-inverse of a matrix.

The function cv::invert inverts the matrix src and stores the result in dst
. When the matrix src is singular or non-square, the function calculates
the pseudo-inverse matrix (the dst matrix) so that norm(src\*dst - I) is
minimal, where I is an identity matrix.

In case of the DECOMP_LU method, the function returns non-zero value if
the inverse has been successfully calculated and 0 if src is singular.

In case of the DECOMP_SVD method, the function returns the inverse
condition number of src (the ratio of the smallest singular value to the
largest singular value) and 0 if src is singular. The SVD method
calculates a pseudo-inverse matrix if src is singular.

Similarly to DECOMP_LU, the method DECOMP_CHOLESKY works only with
non-singular square matrices that should also be symmetrical and
positively defined. In this case, the function stores the inverted
matrix in dst and returns non-zero. Otherwise, it returns 0.

@param src input floating-point M x N matrix.
@param dst output matrix of N x M size and the same type as src.
@param flags inversion method (cv::DecompTypes)
@sa solve, SVD
*/
CV_EXPORTS_W double invert(InputArray src, OutputArray dst, int flags = DECOMP_LU);

/** @brief Solves one or more linear systems or least-squares problems.

The function cv::solve solves a linear system or least-squares problem (the
latter is possible with SVD or QR methods, or by specifying the flag
DECOMP_NORMAL ):
\f[\texttt{dst} =  \arg \min _X \| \texttt{src1} \cdot \texttt{X} -  \texttt{src2} \|\f]

If DECOMP_LU or DECOMP_CHOLESKY method is used, the function returns 1
if src1 (or \f$\texttt{src1}^T\texttt{src1}\f$ ) is non-singular. Otherwise,
it returns 0. In the latter case, dst is not valid. Other methods find a
pseudo-solution in case of a singular left-hand side part.

@note If you want to find a unity-norm solution of an under-defined
singular system \f$\texttt{src1}\cdot\texttt{dst}=0\f$ , the function solve
will not do the work. Use SVD::solveZ instead.

@param src1 input matrix on the left-hand side of the system.
@param src2 input matrix on the right-hand side of the system.
@param dst output solution.
@param flags solution (matrix inversion) method (cv::DecompTypes)
@sa invert, SVD, eigen
*/
CV_EXPORTS_W bool solve(InputArray src1, InputArray src2,
                        OutputArray dst, int flags = DECOMP_LU);

/** @brief Sorts each row or each column of a matrix.

The function cv::sort sorts each matrix row or each matrix column in
ascending or descending order. So you should pass two operation flags to
get desired behaviour. If you want to sort matrix rows or columns
lexicographically, you can use STL std::sort generic function with the
proper comparison predicate.

@param src input single-channel array.
@param dst output array of the same size and type as src.
@param flags operation flags, a combination of cv::SortFlags
@sa sortIdx, randShuffle
*/
CV_EXPORTS_W void sort(InputArray src, OutputArray dst, int flags);

/** @brief Sorts each row or each column of a matrix.

The function cv::sortIdx sorts each matrix row or each matrix column in the
ascending or descending order. So you should pass two operation flags to
get desired behaviour. Instead of reordering the elements themselves, it
stores the indices of sorted elements in the output array. For example:
@code
    Mat A = Mat::eye(3,3,CV_32F), B;
    sortIdx(A, B, SORT_EVERY_ROW + SORT_ASCENDING);
    // B will probably contain
    // (because of equal elements in A some permutations are possible):
    // [[1, 2, 0], [0, 2, 1], [0, 1, 2]]
@endcode
@param src input single-channel array.
@param dst output integer array of the same size as src.
@param flags operation flags that could be a combination of cv::SortFlags
@sa sort, randShuffle
*/
CV_EXPORTS_W void sortIdx(InputArray src, OutputArray dst, int flags);

/** @brief Finds the real roots of a cubic equation.

The function solveCubic finds the real roots of a cubic equation:
-   if coeffs is a 4-element vector:
\f[\texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0\f]
-   if coeffs is a 3-element vector:
\f[x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0\f]

The roots are stored in the roots array.
@param coeffs equation coefficients, an array of 3 or 4 elements.
@param roots output array of real roots that has 1 or 3 elements.
*/
CV_EXPORTS_W int solveCubic(InputArray coeffs, OutputArray roots);

/** @brief Finds the real or complex roots of a polynomial equation.

The function cv::solvePoly finds real and complex roots of a polynomial equation:
\f[\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\f]
@param coeffs array of polynomial coefficients.
@param roots output (complex) array of roots.
@param maxIters maximum number of iterations the algorithm does.
*/
CV_EXPORTS_W double solvePoly(InputArray coeffs, OutputArray roots, int maxIters = 300);

/** @brief Calculates eigenvalues and eigenvectors of a symmetric matrix.

The function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric
matrix src:
@code
    src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
@endcode
@note in the new and the old interfaces different ordering of eigenvalues and eigenvectors
parameters is used.
@param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical
(src ^T^ == src).
@param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored
in the descending order.
@param eigenvectors output matrix of eigenvectors; it has the same size and type as src; the
eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding
eigenvalues.
@sa completeSymm , PCA
*/
CV_EXPORTS_W bool eigen(InputArray src, OutputArray eigenvalues,
                        OutputArray eigenvectors = noArray());

/** @brief Calculates the covariance matrix of a set of vectors.

The function cv::calcCovarMatrix calculates the covariance matrix and, optionally, the mean vector of
the set of input vectors.
@param samples samples stored as separate matrices
@param nsamples number of samples
@param covar output covariance matrix of the type ctype and square size.
@param mean input or output (depending on the flags) array as the average value of the input vectors.
@param flags operation flags as a combination of cv::CovarFlags
@param ctype type of the matrixl; it equals 'CV_64F' by default.
@sa PCA, mulTransposed, Mahalanobis
@todo InputArrayOfArrays
*/
CV_EXPORTS void calcCovarMatrix( const Mat* samples, int nsamples, Mat& covar, Mat& mean,
                                 int flags, int ctype = CV_64F);

/** @overload
@note use cv::COVAR_ROWS or cv::COVAR_COLS flag
@param samples samples stored as rows/columns of a single matrix.
@param covar output covariance matrix of the type ctype and square size.
@param mean input or output (depending on the flags) array as the average value of the input vectors.
@param flags operation flags as a combination of cv::CovarFlags
@param ctype type of the matrixl; it equals 'CV_64F' by default.
*/
CV_EXPORTS_W void calcCovarMatrix( InputArray samples, OutputArray covar,
                                   InputOutputArray mean, int flags, int ctype = CV_64F);

/** wrap PCA::operator() */
CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, int maxComponents = 0);

/** wrap PCA::operator() */
CV_EXPORTS_W void PCACompute(InputArray data, InputOutputArray mean,
                             OutputArray eigenvectors, double retainedVariance);

/** wrap PCA::project */
CV_EXPORTS_W void PCAProject(InputArray data, InputArray mean,
                             InputArray eigenvectors, OutputArray result);

/** wrap PCA::backProject */
CV_EXPORTS_W void PCABackProject(InputArray data, InputArray mean,
                                 InputArray eigenvectors, OutputArray result);

/** wrap SVD::compute */
CV_EXPORTS_W void SVDecomp( InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags = 0 );

/** wrap SVD::backSubst */
CV_EXPORTS_W void SVBackSubst( InputArray w, InputArray u, InputArray vt,
                               InputArray rhs, OutputArray dst );

/** @brief Calculates the Mahalanobis distance between two vectors.

The function cv::Mahalanobis calculates and returns the weighted distance between two vectors:
\f[d( \texttt{vec1} , \texttt{vec2} )= \sqrt{\sum_{i,j}{\texttt{icovar(i,j)}\cdot(\texttt{vec1}(I)-\texttt{vec2}(I))\cdot(\texttt{vec1(j)}-\texttt{vec2(j)})} }\f]
The covariance matrix may be calculated using the cv::calcCovarMatrix function and then inverted using
the invert function (preferably using the cv::DECOMP_SVD method, as the most accurate).
@param v1 first 1D input vector.
@param v2 second 1D input vector.
@param icovar inverse covariance matrix.
*/
CV_EXPORTS_W double Mahalanobis(InputArray v1, InputArray v2, InputArray icovar);

/** @brief Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array.

The function cv::dft performs one of the following:
-   Forward the Fourier transform of a 1D vector of N elements:
    \f[Y = F^{(N)}  \cdot X,\f]
    where \f$F^{(N)}_{jk}=\exp(-2\pi i j k/N)\f$ and \f$i=\sqrt{-1}\f$
-   Inverse the Fourier transform of a 1D vector of N elements:
    \f[\begin{array}{l} X'=  \left (F^{(N)} \right )^{-1}  \cdot Y =  \left (F^{(N)} \right )^*  \cdot y  \\ X = (1/N)  \cdot X, \end{array}\f]
    where \f$F^*=\left(\textrm{Re}(F^{(N)})-\textrm{Im}(F^{(N)})\right)^T\f$
-   Forward the 2D Fourier transform of a M x N matrix:
    \f[Y = F^{(M)}  \cdot X  \cdot F^{(N)}\f]
-   Inverse the 2D Fourier transform of a M x N matrix:
    \f[\begin{array}{l} X'=  \left (F^{(M)} \right )^*  \cdot Y  \cdot \left (F^{(N)} \right )^* \\ X =  \frac{1}{M \cdot N} \cdot X' \end{array}\f]

In case of real (single-channel) data, the output spectrum of the forward Fourier transform or input
spectrum of the inverse Fourier transform can be represented in a packed format called *CCS*
(complex-conjugate-symmetrical). It was borrowed from IPL (Intel\* Image Processing Library). Here
is how 2D *CCS* spectrum looks:
\f[\begin{bmatrix} Re Y_{0,0} & Re Y_{0,1} & Im Y_{0,1} & Re Y_{0,2} & Im Y_{0,2} &  \cdots & Re Y_{0,N/2-1} & Im Y_{0,N/2-1} & Re Y_{0,N/2}  \\ Re Y_{1,0} & Re Y_{1,1} & Im Y_{1,1} & Re Y_{1,2} & Im Y_{1,2} &  \cdots & Re Y_{1,N/2-1} & Im Y_{1,N/2-1} & Re Y_{1,N/2}  \\ Im Y_{1,0} & Re Y_{2,1} & Im Y_{2,1} & Re Y_{2,2} & Im Y_{2,2} &  \cdots & Re Y_{2,N/2-1} & Im Y_{2,N/2-1} & Im Y_{1,N/2}  \\ \hdotsfor{9} \\ Re Y_{M/2-1,0} &  Re Y_{M-3,1}  & Im Y_{M-3,1} &  \hdotsfor{3} & Re Y_{M-3,N/2-1} & Im Y_{M-3,N/2-1}& Re Y_{M/2-1,N/2}  \\ Im Y_{M/2-1,0} &  Re Y_{M-2,1}  & Im Y_{M-2,1} &  \hdotsfor{3} & Re Y_{M-2,N/2-1} & Im Y_{M-2,N/2-1}& Im Y_{M/2-1,N/2}  \\ Re Y_{M/2,0}  &  Re Y_{M-1,1} &  Im Y_{M-1,1} &  \hdotsfor{3} & Re Y_{M-1,N/2-1} & Im Y_{M-1,N/2-1}& Re Y_{M/2,N/2} \end{bmatrix}\f]

In case of 1D transform of a real vector, the output looks like the first row of the matrix above.

So, the function chooses an operation mode depending on the flags and size of the input array:
-   If DFT_ROWS is set or the input array has a single row or single column, the function
    performs a 1D forward or inverse transform of each row of a matrix when DFT_ROWS is set.
    Otherwise, it performs a 2D transform.
-   If the input array is real and DFT_INVERSE is not set, the function performs a forward 1D or
    2D transform:
    -   When DFT_COMPLEX_OUTPUT is set, the output is a complex matrix of the same size as
        input.
    -   When DFT_COMPLEX_OUTPUT is not set, the output is a real matrix of the same size as
        input. In case of 2D transform, it uses the packed format as shown above. In case of a
        single 1D transform, it looks like the first row of the matrix above. In case of
        multiple 1D transforms (when using the DFT_ROWS flag), each row of the output matrix
        looks like the first row of the matrix above.
-   If the input array is complex and either DFT_INVERSE or DFT_REAL_OUTPUT are not set, the
    output is a complex array of the same size as input. The function performs a forward or
    inverse 1D or 2D transform of the whole input array or each row of the input array
    independently, depending on the flags DFT_INVERSE and DFT_ROWS.
-   When DFT_INVERSE is set and the input array is real, or it is complex but DFT_REAL_OUTPUT
    is set, the output is a real array of the same size as input. The function performs a 1D or 2D
    inverse transformation of the whole input array or each individual row, depending on the flags
    DFT_INVERSE and DFT_ROWS.

If DFT_SCALE is set, the scaling is done after the transformation.

Unlike dct , the function supports arrays of arbitrary size. But only those arrays are processed
efficiently, whose sizes can be factorized in a product of small prime numbers (2, 3, and 5 in the
current implementation). Such an efficient DFT size can be calculated using the getOptimalDFTSize
method.

The sample below illustrates how to calculate a DFT-based convolution of two 2D real arrays:
@code
    void convolveDFT(InputArray A, InputArray B, OutputArray C)
    {
        // reallocate the output array if needed
        C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
        Size dftSize;
        // calculate the size of DFT transform
        dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
        dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

        // allocate temporary buffers and initialize them with 0's
        Mat tempA(dftSize, A.type(), Scalar::all(0));
        Mat tempB(dftSize, B.type(), Scalar::all(0));

        // copy A and B to the top-left corners of tempA and tempB, respectively
        Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
        A.copyTo(roiA);
        Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
        B.copyTo(roiB);

        // now transform the padded A & B in-place;
        // use "nonzeroRows" hint for faster processing
        dft(tempA, tempA, 0, A.rows);
        dft(tempB, tempB, 0, B.rows);

        // multiply the spectrums;
        // the function handles packed spectrum representations well
        mulSpectrums(tempA, tempB, tempA);

        // transform the product back from the frequency domain.
        // Even though all the result rows will be non-zero,
        // you need only the first C.rows of them, and thus you
        // pass nonzeroRows == C.rows
        dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);

        // now copy the result back to C.
        tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);

        // all the temporary buffers will be deallocated automatically
    }
@endcode
To optimize this sample, consider the following approaches:
-   Since nonzeroRows != 0 is passed to the forward transform calls and since A and B are copied to
    the top-left corners of tempA and tempB, respectively, it is not necessary to clear the whole
    tempA and tempB. It is only necessary to clear the tempA.cols - A.cols ( tempB.cols - B.cols)
    rightmost columns of the matrices.
-   This DFT-based convolution does not have to be applied to the whole big arrays, especially if B
    is significantly smaller than A or vice versa. Instead, you can calculate convolution by parts.
    To do this, you need to split the output array C into multiple tiles. For each tile, estimate
    which parts of A and B are required to calculate convolution in this tile. If the tiles in C are
    too small, the speed will decrease a lot because of repeated work. In the ultimate case, when
    each tile in C is a single pixel, the algorithm becomes equivalent to the naive convolution
    algorithm. If the tiles are too big, the temporary arrays tempA and tempB become too big and
    there is also a slowdown because of bad cache locality. So, there is an optimal tile size
    somewhere in the middle.
-   If different tiles in C can be calculated in parallel and, thus, the convolution is done by
    parts, the loop can be threaded.

All of the above improvements have been implemented in matchTemplate and filter2D . Therefore, by
using them, you can get the performance even better than with the above theoretically optimal
implementation. Though, those two functions actually calculate cross-correlation, not convolution,
so you need to "flip" the second convolution operand B vertically and horizontally using flip .
@note
-   An example using the discrete fourier transform can be found at
    opencv_source_code/samples/cpp/dft.cpp
-   (Python) An example using the dft functionality to perform Wiener deconvolution can be found
    at opencv_source/samples/python/deconvolution.py
-   (Python) An example rearranging the quadrants of a Fourier image can be found at
    opencv_source/samples/python/dft.py
@param src input array that could be real or complex.
@param dst output array whose size and type depends on the flags .
@param flags transformation flags, representing a combination of the cv::DftFlags
@param nonzeroRows when the parameter is not zero, the function assumes that only the first
nonzeroRows rows of the input array (DFT_INVERSE is not set) or only the first nonzeroRows of the
output array (DFT_INVERSE is set) contain non-zeros, thus, the function can handle the rest of the
rows more efficiently and save some time; this technique is very useful for calculating array
cross-correlation or convolution using DFT.
@sa dct , getOptimalDFTSize , mulSpectrums, filter2D , matchTemplate , flip , cartToPolar ,
magnitude , phase
*/
CV_EXPORTS_W void dft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);

/** @brief Calculates the inverse Discrete Fourier Transform of a 1D or 2D array.

idft(src, dst, flags) is equivalent to dft(src, dst, flags | DFT_INVERSE) .
@note None of dft and idft scales the result by default. So, you should pass DFT_SCALE to one of
dft or idft explicitly to make these transforms mutually inverse.
@sa dft, dct, idct, mulSpectrums, getOptimalDFTSize
@param src input floating-point real or complex array.
@param dst output array whose size and type depend on the flags.
@param flags operation flags (see dft and cv::DftFlags).
@param nonzeroRows number of dst rows to process; the rest of the rows have undefined content (see
the convolution sample in dft description.
*/
CV_EXPORTS_W void idft(InputArray src, OutputArray dst, int flags = 0, int nonzeroRows = 0);

/** @brief Performs a forward or inverse discrete Cosine transform of 1D or 2D array.

The function cv::dct performs a forward or inverse discrete Cosine transform (DCT) of a 1D or 2D
floating-point array:
-   Forward Cosine transform of a 1D vector of N elements:
    \f[Y = C^{(N)}  \cdot X\f]
    where
    \f[C^{(N)}_{jk}= \sqrt{\alpha_j/N} \cos \left ( \frac{\pi(2k+1)j}{2N} \right )\f]
    and
    \f$\alpha_0=1\f$, \f$\alpha_j=2\f$ for *j \> 0*.
-   Inverse Cosine transform of a 1D vector of N elements:
    \f[X =  \left (C^{(N)} \right )^{-1}  \cdot Y =  \left (C^{(N)} \right )^T  \cdot Y\f]
    (since \f$C^{(N)}\f$ is an orthogonal matrix, \f$C^{(N)} \cdot \left(C^{(N)}\right)^T = I\f$ )
-   Forward 2D Cosine transform of M x N matrix:
    \f[Y = C^{(N)}  \cdot X  \cdot \left (C^{(N)} \right )^T\f]
-   Inverse 2D Cosine transform of M x N matrix:
    \f[X =  \left (C^{(N)} \right )^T  \cdot X  \cdot C^{(N)}\f]

The function chooses the mode of operation by looking at the flags and size of the input array:
-   If (flags & DCT_INVERSE) == 0 , the function does a forward 1D or 2D transform. Otherwise, it
    is an inverse 1D or 2D transform.
-   If (flags & DCT_ROWS) != 0 , the function performs a 1D transform of each row.
-   If the array is a single column or a single row, the function performs a 1D transform.
-   If none of the above is true, the function performs a 2D transform.

@note Currently dct supports even-size arrays (2, 4, 6 ...). For data analysis and approximation, you
can pad the array when necessary.
Also, the function performance depends very much, and not monotonically, on the array size (see
getOptimalDFTSize ). In the current implementation DCT of a vector of size N is calculated via DFT
of a vector of size N/2 . Thus, the optimal DCT size N1 \>= N can be calculated as:
@code
    size_t getOptimalDCTSize(size_t N) { return 2*getOptimalDFTSize((N+1)/2); }
    N1 = getOptimalDCTSize(N);
@endcode
@param src input floating-point array.
@param dst output array of the same size and type as src .
@param flags transformation flags as a combination of cv::DftFlags (DCT_*)
@sa dft , getOptimalDFTSize , idct
*/
CV_EXPORTS_W void dct(InputArray src, OutputArray dst, int flags = 0);

/** @brief Calculates the inverse Discrete Cosine Transform of a 1D or 2D array.

idct(src, dst, flags) is equivalent to dct(src, dst, flags | DCT_INVERSE).
@param src input floating-point single-channel array.
@param dst output array of the same size and type as src.
@param flags operation flags.
@sa  dct, dft, idft, getOptimalDFTSize
*/
CV_EXPORTS_W void idct(InputArray src, OutputArray dst, int flags = 0);

/** @brief Performs the per-element multiplication of two Fourier spectrums.

The function cv::mulSpectrums performs the per-element multiplication of the two CCS-packed or complex
matrices that are results of a real or complex Fourier transform.

The function, together with dft and idft , may be used to calculate convolution (pass conjB=false )
or correlation (pass conjB=true ) of two arrays rapidly. When the arrays are complex, they are
simply multiplied (per element) with an optional conjugation of the second-array elements. When the
arrays are real, they are assumed to be CCS-packed (see dft for details).
@param a first input array.
@param b second input array of the same size and type as src1 .
@param c output array of the same size and type as src1 .
@param flags operation flags; currently, the only supported flag is cv::DFT_ROWS, which indicates that
each row of src1 and src2 is an independent 1D Fourier spectrum. If you do not want to use this flag, then simply add a `0` as value.
@param conjB optional flag that conjugates the second input array before the multiplication (true)
or not (false).
*/
CV_EXPORTS_W void mulSpectrums(InputArray a, InputArray b, OutputArray c,
                               int flags, bool conjB = false);

/** @brief Returns the optimal DFT size for a given vector size.

DFT performance is not a monotonic function of a vector size. Therefore, when you calculate
convolution of two arrays or perform the spectral analysis of an array, it usually makes sense to
pad the input data with zeros to get a bit larger array that can be transformed much faster than the
original one. Arrays whose size is a power-of-two (2, 4, 8, 16, 32, ...) are the fastest to process.
Though, the arrays whose size is a product of 2's, 3's, and 5's (for example, 300 = 5\*5\*3\*2\*2)
are also processed quite efficiently.

The function cv::getOptimalDFTSize returns the minimum number N that is greater than or equal to vecsize
so that the DFT of a vector of size N can be processed efficiently. In the current implementation N
= 2 ^p^ \* 3 ^q^ \* 5 ^r^ for some integer p, q, r.

The function returns a negative number if vecsize is too large (very close to INT_MAX ).

While the function cannot be used directly to estimate the optimal vector size for DCT transform
(since the current DCT implementation supports only even-size vectors), it can be easily processed
as getOptimalDFTSize((vecsize+1)/2)\*2.
@param vecsize vector size.
@sa dft , dct , idft , idct , mulSpectrums
*/
CV_EXPORTS_W int getOptimalDFTSize(int vecsize);

/** @brief Returns the default random number generator.

The function cv::theRNG returns the default random number generator. For each thread, there is a
separate random number generator, so you can use the function safely in multi-thread environments.
If you just need to get a single random number using this generator or initialize an array, you can
use randu or randn instead. But if you are going to generate many random numbers inside a loop, it
is much faster to use this function to retrieve the generator and then use RNG::operator _Tp() .
@sa RNG, randu, randn
*/
CV_EXPORTS RNG& theRNG();

/** @brief Sets state of default random number generator.

The function cv::setRNGSeed sets state of default random number generator to custom value.
@param seed new state for default random number generator
@sa RNG, randu, randn
*/
CV_EXPORTS_W void setRNGSeed(int seed);

/** @brief Generates a single uniformly-distributed random number or an array of random numbers.

Non-template variant of the function fills the matrix dst with uniformly-distributed
random numbers from the specified range:
\f[\texttt{low} _c  \leq \texttt{dst} (I)_c <  \texttt{high} _c\f]
@param dst output array of random numbers; the array must be pre-allocated.
@param low inclusive lower boundary of the generated random numbers.
@param high exclusive upper boundary of the generated random numbers.
@sa RNG, randn, theRNG
*/
CV_EXPORTS_W void randu(InputOutputArray dst, InputArray low, InputArray high);

/** @brief Fills the array with normally distributed random numbers.

The function cv::randn fills the matrix dst with normally distributed random numbers with the specified
mean vector and the standard deviation matrix. The generated random numbers are clipped to fit the
value range of the output array data type.
@param dst output array of random numbers; the array must be pre-allocated and have 1 to 4 channels.
@param mean mean value (expectation) of the generated random numbers.
@param stddev standard deviation of the generated random numbers; it can be either a vector (in
which case a diagonal standard deviation matrix is assumed) or a square matrix.
@sa RNG, randu
*/
CV_EXPORTS_W void randn(InputOutputArray dst, InputArray mean, InputArray stddev);

/** @brief Shuffles the array elements randomly.

The function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and
swapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor .
@param dst input/output numerical 1D array.
@param iterFactor scale factor that determines the number of random swap operations (see the details
below).
@param rng optional random number generator used for shuffling; if it is zero, theRNG () is used
instead.
@sa RNG, sort
*/
CV_EXPORTS_W void randShuffle(InputOutputArray dst, double iterFactor = 1., RNG* rng = 0);

/** @brief Principal Component Analysis

The class is used to calculate a special basis for a set of vectors. The
basis will consist of eigenvectors of the covariance matrix calculated
from the input set of vectors. The class %PCA can also transform
vectors to/from the new coordinate space defined by the basis. Usually,
in this new coordinate system, each vector from the original set (and
any linear combination of such vectors) can be quite accurately
approximated by taking its first few components, corresponding to the
eigenvectors of the largest eigenvalues of the covariance matrix.
Geometrically it means that you calculate a projection of the vector to
a subspace formed by a few eigenvectors corresponding to the dominant
eigenvalues of the covariance matrix. And usually such a projection is
very close to the original vector. So, you can represent the original
vector from a high-dimensional space with a much shorter vector
consisting of the projected vector's coordinates in the subspace. Such a
transformation is also known as Karhunen-Loeve Transform, or KLT.
See http://en.wikipedia.org/wiki/Principal_component_analysis

The sample below is the function that takes two matrices. The first
function stores a set of vectors (a row per vector) that is used to
calculate PCA. The second function stores another "test" set of vectors
(a row per vector). First, these vectors are compressed with PCA, then
reconstructed back, and then the reconstruction error norm is computed
and printed for each vector. :

@code{.cpp}
using namespace cv;

PCA compressPCA(const Mat& pcaset, int maxComponents,
                const Mat& testset, Mat& compressed)
{
    PCA pca(pcaset, // pass the data
            Mat(), // we do not have a pre-computed mean vector,
                   // so let the PCA engine to compute it
            PCA::DATA_AS_ROW, // indicate that the vectors
                                // are stored as matrix rows
                                // (use PCA::DATA_AS_COL if the vectors are
                                // the matrix columns)
            maxComponents // specify, how many principal components to retain
            );
    // if there is no test data, just return the computed basis, ready-to-use
    if( !testset.data )
        return pca;
    CV_Assert( testset.cols == pcaset.cols );

    compressed.create(testset.rows, maxComponents, testset.type());

    Mat reconstructed;
    for( int i = 0; i < testset.rows; i++ )
    {
        Mat vec = testset.row(i), coeffs = compressed.row(i), reconstructed;
        // compress the vector, the result will be stored
        // in the i-th row of the output matrix
        pca.project(vec, coeffs);
        // and then reconstruct it
        pca.backProject(coeffs, reconstructed);
        // and measure the error
        printf("%d. diff = %g\n", i, norm(vec, reconstructed, NORM_L2));
    }
    return pca;
}
@endcode
@sa calcCovarMatrix, mulTransposed, SVD, dft, dct
*/
class CV_EXPORTS PCA
{
public:
    enum Flags { DATA_AS_ROW = 0, //!< indicates that the input samples are stored as matrix rows
                 DATA_AS_COL = 1, //!< indicates that the input samples are stored as matrix columns
                 USE_AVG     = 2  //!
               };

    /** @brief default constructor

    The default constructor initializes an empty %PCA structure. The other
    constructors initialize the structure and call PCA::operator()().
    */
    PCA();

    /** @overload
    @param data input samples stored as matrix rows or matrix columns.
    @param mean optional mean value; if the matrix is empty (@c noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout (PCA::Flags)
    @param maxComponents maximum number of components that %PCA should
    retain; by default, all the components are retained.
    */
    PCA(InputArray data, InputArray mean, int flags, int maxComponents = 0);

    /** @overload
    @param data input samples stored as matrix rows or matrix columns.
    @param mean optional mean value; if the matrix is empty (noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout (PCA::Flags)
    @param retainedVariance Percentage of variance that PCA should retain.
    Using this parameter will let the PCA decided how many components to
    retain but it will always keep at least 2.
    */
    PCA(InputArray data, InputArray mean, int flags, double retainedVariance);

    /** @brief performs %PCA

    The operator performs %PCA of the supplied dataset. It is safe to reuse
    the same PCA structure for multiple datasets. That is, if the structure
    has been previously used with another dataset, the existing internal
    data is reclaimed and the new @ref eigenvalues, @ref eigenvectors and @ref
    mean are allocated and computed.

    The computed @ref eigenvalues are sorted from the largest to the smallest and
    the corresponding @ref eigenvectors are stored as eigenvectors rows.

    @param data input samples stored as the matrix rows or as the matrix
    columns.
    @param mean optional mean value; if the matrix is empty (noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout. (Flags)
    @param maxComponents maximum number of components that PCA should
    retain; by default, all the components are retained.
    */
    PCA& operator()(InputArray data, InputArray mean, int flags, int maxComponents = 0);

    /** @overload
    @param data input samples stored as the matrix rows or as the matrix
    columns.
    @param mean optional mean value; if the matrix is empty (noArray()),
    the mean is computed from the data.
    @param flags operation flags; currently the parameter is only used to
    specify the data layout. (PCA::Flags)
    @param retainedVariance Percentage of variance that %PCA should retain.
    Using this parameter will let the %PCA decided how many components to
    retain but it will always keep at least 2.
     */
    PCA& operator()(InputArray data, InputArray mean, int flags, double retainedVariance);

    /** @brief Projects vector(s) to the principal component subspace.

    The methods project one or more vectors to the principal component
    subspace, where each vector projection is represented by coefficients in
    the principal component basis. The first form of the method returns the
    matrix that the second form writes to the result. So the first form can
    be used as a part of expression while the second form can be more
    efficient in a processing loop.
    @param vec input vector(s); must have the same dimensionality and the
    same layout as the input data used at %PCA phase, that is, if
    DATA_AS_ROW are specified, then `vec.cols==data.cols`
    (vector dimensionality) and `vec.rows` is the number of vectors to
    project, and the same is true for the PCA::DATA_AS_COL case.
    */
    Mat project(InputArray vec) const;

    /** @overload
    @param vec input vector(s); must have the same dimensionality and the
    same layout as the input data used at PCA phase, that is, if
    DATA_AS_ROW are specified, then `vec.cols==data.cols`
    (vector dimensionality) and `vec.rows` is the number of vectors to
    project, and the same is true for the PCA::DATA_AS_COL case.
    @param result output vectors; in case of PCA::DATA_AS_COL, the
    output matrix has as many columns as the number of input vectors, this
    means that `result.cols==vec.cols` and the number of rows match the
    number of principal components (for example, `maxComponents` parameter
    passed to the constructor).
     */
    void project(InputArray vec, OutputArray result) const;

    /** @brief Reconstructs vectors from their PC projections.

    The methods are inverse operations to PCA::project. They take PC
    coordinates of projected vectors and reconstruct the original vectors.
    Unless all the principal components have been retained, the
    reconstructed vectors are different from the originals. But typically,
    the difference is small if the number of components is large enough (but
    still much smaller than the original vector dimensionality). As a
    result, PCA is used.
    @param vec coordinates of the vectors in the principal component
    subspace, the layout and size are the same as of PCA::project output
    vectors.
     */
    Mat backProject(InputArray vec) const;

    /** @overload
    @param vec coordinates of the vectors in the principal component
    subspace, the layout and size are the same as of PCA::project output
    vectors.
    @param result reconstructed vectors; the layout and size are the same as
    of PCA::project input vectors.
     */
    void backProject(InputArray vec, OutputArray result) const;

    /** @brief write PCA objects

    Writes @ref eigenvalues @ref eigenvectors and @ref mean to specified FileStorage
     */
    void write(FileStorage& fs) const;

    /** @brief load PCA objects

    Loads @ref eigenvalues @ref eigenvectors and @ref mean from specified FileNode
     */
    void read(const FileNode& fn);

    Mat eigenvectors; //!< eigenvectors of the covariation matrix
    Mat eigenvalues; //!< eigenvalues of the covariation matrix
    Mat mean; //!< mean value subtracted before the projection and added after the back projection
};

/** @example pca.cpp
  An example using %PCA for dimensionality reduction while maintaining an amount of variance
 */

/**
   @brief Linear Discriminant Analysis
   @todo document this class
 */
class CV_EXPORTS LDA
{
public:
    /** @brief constructor
    Initializes a LDA with num_components (default 0).
    */
    explicit LDA(int num_components = 0);

    /** Initializes and performs a Discriminant Analysis with Fisher's
     Optimization Criterion on given data in src and corresponding labels
     in labels. If 0 (or less) number of components are given, they are
     automatically determined for given data in computation.
    */
    LDA(InputArrayOfArrays src, InputArray labels, int num_components = 0);

    /** Serializes this object to a given filename.
      */
    void save(const String& filename) const;

    /** Deserializes this object from a given filename.
      */
    void load(const String& filename);

    /** Serializes this object to a given cv::FileStorage.
      */
    void save(FileStorage& fs) const;

    /** Deserializes this object from a given cv::FileStorage.
      */
    void load(const FileStorage& node);

    /** destructor
      */
    ~LDA();

    /** Compute the discriminants for data in src (row aligned) and labels.
      */
    void compute(InputArrayOfArrays src, InputArray labels);

    /** Projects samples into the LDA subspace.
        src may be one or more row aligned samples.
      */
    Mat project(InputArray src);

    /** Reconstructs projections from the LDA subspace.
        src may be one or more row aligned projections.
      */
    Mat reconstruct(InputArray src);

    /** Returns the eigenvectors of this LDA.
      */
    Mat eigenvectors() const { return _eigenvectors; }

    /** Returns the eigenvalues of this LDA.
      */
    Mat eigenvalues() const { return _eigenvalues; }

    static Mat subspaceProject(InputArray W, InputArray mean, InputArray src);
    static Mat subspaceReconstruct(InputArray W, InputArray mean, InputArray src);

protected:
    bool _dataAsRow; // unused, but needed for 3.0 ABI compatibility.
    int _num_components;
    Mat _eigenvectors;
    Mat _eigenvalues;
    void lda(InputArrayOfArrays src, InputArray labels);
};

/** @brief Singular Value Decomposition

Class for computing Singular Value Decomposition of a floating-point
matrix. The Singular Value Decomposition is used to solve least-square
problems, under-determined linear systems, invert matrices, compute
condition numbers, and so on.

If you want to compute a condition number of a matrix or an absolute value of
its determinant, you do not need `u` and `vt`. You can pass
flags=SVD::NO_UV|... . Another flag SVD::FULL_UV indicates that full-size u
and vt must be computed, which is not necessary most of the time.

@sa invert, solve, eigen, determinant
*/
class CV_EXPORTS SVD
{
public:
    enum Flags {
        /** allow the algorithm to modify the decomposed matrix; it can save space and speed up
            processing. currently ignored. */
        MODIFY_A = 1,
        /** indicates that only a vector of singular values `w` is to be processed, while u and vt
            will be set to empty matrices */
        NO_UV    = 2,
        /** when the matrix is not square, by default the algorithm produces u and vt matrices of
            sufficiently large size for the further A reconstruction; if, however, FULL_UV flag is
            specified, u and vt will be full-size square orthogonal matrices.*/
        FULL_UV  = 4
    };

    /** @brief the default constructor

    initializes an empty SVD structure
      */
    SVD();

    /** @overload
    initializes an empty SVD structure and then calls SVD::operator()
    @param src decomposed matrix.
    @param flags operation flags (SVD::Flags)
      */
    SVD( InputArray src, int flags = 0 );

    /** @brief the operator that performs SVD. The previously allocated u, w and vt are released.

    The operator performs the singular value decomposition of the supplied
    matrix. The u,`vt` , and the vector of singular values w are stored in
    the structure. The same SVD structure can be reused many times with
    different matrices. Each time, if needed, the previous u,`vt` , and w
    are reclaimed and the new matrices are created, which is all handled by
    Mat::create.
    @param src decomposed matrix.
    @param flags operation flags (SVD::Flags)
      */
    SVD& operator ()( InputArray src, int flags = 0 );

    /** @brief decomposes matrix and stores the results to user-provided matrices

    The methods/functions perform SVD of matrix. Unlike SVD::SVD constructor
    and SVD::operator(), they store the results to the user-provided
    matrices:

    @code{.cpp}
    Mat A, w, u, vt;
    SVD::compute(A, w, u, vt);
    @endcode

    @param src decomposed matrix
    @param w calculated singular values
    @param u calculated left singular vectors
    @param vt transposed matrix of right singular values
    @param flags operation flags - see SVD::SVD.
      */
    static void compute( InputArray src, OutputArray w,
                         OutputArray u, OutputArray vt, int flags = 0 );

    /** @overload
    computes singular values of a matrix
    @param src decomposed matrix
    @param w calculated singular values
    @param flags operation flags - see SVD::Flags.
      */
    static void compute( InputArray src, OutputArray w, int flags = 0 );

    /** @brief performs back substitution
      */
    static void backSubst( InputArray w, InputArray u,
                           InputArray vt, InputArray rhs,
                           OutputArray dst );

    /** @brief solves an under-determined singular linear system

    The method finds a unit-length solution x of a singular linear system
    A\*x = 0. Depending on the rank of A, there can be no solutions, a
    single solution or an infinite number of solutions. In general, the
    algorithm solves the following problem:
    \f[dst =  \arg \min _{x:  \| x \| =1}  \| src  \cdot x  \|\f]
    @param src left-hand-side matrix.
    @param dst found solution.
      */
    static void solveZ( InputArray src, OutputArray dst );

    /** @brief performs a singular value back substitution.

    The method calculates a back substitution for the specified right-hand
    side:

    \f[\texttt{x} =  \texttt{vt} ^T  \cdot diag( \texttt{w} )^{-1}  \cdot \texttt{u} ^T  \cdot \texttt{rhs} \sim \texttt{A} ^{-1}  \cdot \texttt{rhs}\f]

    Using this technique you can either get a very accurate solution of the
    convenient linear system, or the best (in the least-squares terms)
    pseudo-solution of an overdetermined linear system.

    @param rhs right-hand side of a linear system (u\*w\*v')\*dst = rhs to
    be solved, where A has been previously decomposed.

    @param dst found solution of the system.

    @note Explicit SVD with the further back substitution only makes sense
    if you need to solve many linear systems with the same left-hand side
    (for example, src ). If all you need is to solve a single system
    (possibly with multiple rhs immediately available), simply call solve
    add pass DECOMP_SVD there. It does absolutely the same thing.
      */
    void backSubst( InputArray rhs, OutputArray dst ) const;

    /** @todo document */
    template<typename _Tp, int m, int n, int nm> static
    void compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt );

    /** @todo document */
    template<typename _Tp, int m, int n, int nm> static
    void compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w );

    /** @todo document */
    template<typename _Tp, int m, int n, int nm, int nb> static
    void backSubst( const Matx<_Tp, nm, 1>& w, const Matx<_Tp, m, nm>& u, const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs, Matx<_Tp, n, nb>& dst );

    Mat u, w, vt;
};

/** @brief Random Number Generator

Random number generator. It encapsulates the state (currently, a 64-bit
integer) and has methods to return scalar random values and to fill
arrays with random values. Currently it supports uniform and Gaussian
(normal) distributions. The generator uses Multiply-With-Carry
algorithm, introduced by G. Marsaglia (
<http://en.wikipedia.org/wiki/Multiply-with-carry> ).
Gaussian-distribution random numbers are generated using the Ziggurat
algorithm ( <http://en.wikipedia.org/wiki/Ziggurat_algorithm> ),
introduced by G. Marsaglia and W. W. Tsang.
*/
class CV_EXPORTS RNG
{
public:
    enum { UNIFORM = 0,
           NORMAL  = 1
         };

    /** @brief constructor

    These are the RNG constructors. The first form sets the state to some
    pre-defined value, equal to 2\*\*32-1 in the current implementation. The
    second form sets the state to the specified value. If you passed state=0
    , the constructor uses the above default value instead to avoid the
    singular random number sequence, consisting of all zeros.
    */
    RNG();
    /** @overload
    @param state 64-bit value used to initialize the RNG.
    */
    RNG(uint64 state);
    /**The method updates the state using the MWC algorithm and returns the
    next 32-bit random number.*/
    unsigned next();

    /**Each of the methods updates the state using the MWC algorithm and
    returns the next random number of the specified type. In case of integer
    types, the returned number is from the available value range for the
    specified type. In case of floating-point types, the returned value is
    from [0,1) range.
    */
    operator uchar();
    /** @overload */
    operator schar();
    /** @overload */
    operator ushort();
    /** @overload */
    operator short();
    /** @overload */
    operator unsigned();
    /** @overload */
    operator int();
    /** @overload */
    operator float();
    /** @overload */
    operator double();

    /** @brief returns a random integer sampled uniformly from [0, N).

    The methods transform the state using the MWC algorithm and return the
    next random number. The first form is equivalent to RNG::next . The
    second form returns the random number modulo N , which means that the
    result is in the range [0, N) .
    */
    unsigned operator ()();
    /** @overload
    @param N upper non-inclusive boundary of the returned random number.
    */
    unsigned operator ()(unsigned N);

    /** @brief returns uniformly distributed integer random number from [a,b) range

    The methods transform the state using the MWC algorithm and return the
    next uniformly-distributed random number of the specified type, deduced
    from the input parameter type, from the range [a, b) . There is a nuance
    illustrated by the following sample:

    @code{.cpp}
    RNG rng;

    // always produces 0
    double a = rng.uniform(0, 1);

    // produces double from [0, 1)
    double a1 = rng.uniform((double)0, (double)1);

    // produces float from [0, 1)
    float b = rng.uniform(0.f, 1.f);

    // produces double from [0, 1)
    double c = rng.uniform(0., 1.);

    // may cause compiler error because of ambiguity:
    //  RNG::uniform(0, (int)0.999999)? or RNG::uniform((double)0, 0.99999)?
    double d = rng.uniform(0, 0.999999);
    @endcode

    The compiler does not take into account the type of the variable to
    which you assign the result of RNG::uniform . The only thing that
    matters to the compiler is the type of a and b parameters. So, if you
    want a floating-point random number, but the range boundaries are
    integer numbers, either put dots in the end, if they are constants, or
    use explicit type cast operators, as in the a1 initialization above.
    @param a lower inclusive boundary of the returned random number.
    @param b upper non-inclusive boundary of the returned random number.
      */
    int uniform(int a, int b);
    /** @overload */
    float uniform(float a, float b);
    /** @overload */
    double uniform(double a, double b);

    /** @brief Fills arrays with random numbers.

    @param mat 2D or N-dimensional matrix; currently matrices with more than
    4 channels are not supported by the methods, use Mat::reshape as a
    possible workaround.
    @param distType distribution type, RNG::UNIFORM or RNG::NORMAL.
    @param a first distribution parameter; in case of the uniform
    distribution, this is an inclusive lower boundary, in case of the normal
    distribution, this is a mean value.
    @param b second distribution parameter; in case of the uniform
    distribution, this is a non-inclusive upper boundary, in case of the
    normal distribution, this is a standard deviation (diagonal of the
    standard deviation matrix or the full standard deviation matrix).
    @param saturateRange pre-saturation flag; for uniform distribution only;
    if true, the method will first convert a and b to the acceptable value
    range (according to the mat datatype) and then will generate uniformly
    distributed random numbers within the range [saturate(a), saturate(b)),
    if saturateRange=false, the method will generate uniformly distributed
    random numbers in the original range [a, b) and then will saturate them,
    it means, for example, that
    <tt>theRNG().fill(mat_8u, RNG::UNIFORM, -DBL_MAX, DBL_MAX)</tt> will likely
    produce array mostly filled with 0's and 255's, since the range (0, 255)
    is significantly smaller than [-DBL_MAX, DBL_MAX).

    Each of the methods fills the matrix with the random values from the
    specified distribution. As the new numbers are generated, the RNG state
    is updated accordingly. In case of multiple-channel images, every
    channel is filled independently, which means that RNG cannot generate
    samples from the multi-dimensional Gaussian distribution with
    non-diagonal covariance matrix directly. To do that, the method
    generates samples from multi-dimensional standard Gaussian distribution
    with zero mean and identity covariation matrix, and then transforms them
    using transform to get samples from the specified Gaussian distribution.
    */
    void fill( InputOutputArray mat, int distType, InputArray a, InputArray b, bool saturateRange = false );

    /** @brief Returns the next random number sampled from the Gaussian distribution
    @param sigma standard deviation of the distribution.

    The method transforms the state using the MWC algorithm and returns the
    next random number from the Gaussian distribution N(0,sigma) . That is,
    the mean value of the returned random numbers is zero and the standard
    deviation is the specified sigma .
    */
    double gaussian(double sigma);

    uint64 state;

    bool operator ==(const RNG& other) const;
};

/** @brief Mersenne Twister random number generator

Inspired by http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/CODES/mt19937ar.c
@todo document
 */
class CV_EXPORTS RNG_MT19937
{
public:
    RNG_MT19937();
    RNG_MT19937(unsigned s);
    void seed(unsigned s);

    unsigned next();

    operator int();
    operator unsigned();
    operator float();
    operator double();

    unsigned operator ()(unsigned N);
    unsigned operator ()();

    /** @brief returns uniformly distributed integer random number from [a,b) range

*/
    int uniform(int a, int b);
    /** @brief returns uniformly distributed floating-point random number from [a,b) range

*/
    float uniform(float a, float b);
    /** @brief returns uniformly distributed double-precision floating-point random number from [a,b) range

*/
    double uniform(double a, double b);

private:
    enum PeriodParameters {N = 624, M = 397};
    unsigned state[N];
    int mti;
};

//! @} core_array

//! @addtogroup core_cluster
//!  @{

/** @example kmeans.cpp
  An example on K-means clustering
*/

/** @brief Finds centers of clusters and groups input samples around the clusters.

The function kmeans implements a k-means algorithm that finds the centers of cluster_count clusters
and groups the input samples around the clusters. As an output, \f$\texttt{labels}_i\f$ contains a
0-based cluster index for the sample stored in the \f$i^{th}\f$ row of the samples matrix.

@note
-   (Python) An example on K-means clustering can be found at
    opencv_source_code/samples/python/kmeans.py
@param data Data for clustering. An array of N-Dimensional points with float coordinates is needed.
Examples of this array can be:
-   Mat points(count, 2, CV_32F);
-   Mat points(count, 1, CV_32FC2);
-   Mat points(1, count, CV_32FC2);
-   std::vector\<cv::Point2f\> points(sampleCount);
@param K Number of clusters to split the set by.
@param bestLabels Input/output integer array that stores the cluster indices for every sample.
@param criteria The algorithm termination criteria, that is, the maximum number of iterations and/or
the desired accuracy. The accuracy is specified as criteria.epsilon. As soon as each of the cluster
centers moves by less than criteria.epsilon on some iteration, the algorithm stops.
@param attempts Flag to specify the number of times the algorithm is executed using different
initial labellings. The algorithm returns the labels that yield the best compactness (see the last
function parameter).
@param flags Flag that can take values of cv::KmeansFlags
@param centers Output matrix of the cluster centers, one row per each cluster center.
@return The function returns the compactness measure that is computed as
\f[\sum _i  \| \texttt{samples} _i -  \texttt{centers} _{ \texttt{labels} _i} \| ^2\f]
after every attempt. The best (minimum) value is chosen and the corresponding labels and the
compactness value are returned by the function. Basically, you can use only the core of the
function, set the number of attempts to 1, initialize labels each time using a custom algorithm,
pass them with the ( flags = KMEANS_USE_INITIAL_LABELS ) flag, and then choose the best
(most-compact) clustering.
*/
CV_EXPORTS_W double kmeans( InputArray data, int K, InputOutputArray bestLabels,
                            TermCriteria criteria, int attempts,
                            int flags, OutputArray centers = noArray() );

//! @} core_cluster

//! @addtogroup core_basic
//! @{

/////////////////////////////// Formatted output of cv::Mat ///////////////////////////

/** @todo document */
class CV_EXPORTS Formatted
{
public:
    virtual const char* next() = 0;
    virtual void reset() = 0;
    virtual ~Formatted();
};

/** @todo document */
class CV_EXPORTS Formatter
{
public:
    enum { FMT_DEFAULT = 0,
           FMT_MATLAB  = 1,
           FMT_CSV     = 2,
           FMT_PYTHON  = 3,
           FMT_NUMPY   = 4,
           FMT_C       = 5
         };

    virtual ~Formatter();

    virtual Ptr<Formatted> format(const Mat& mtx) const = 0;

    virtual void set32fPrecision(int p = 8) = 0;
    virtual void set64fPrecision(int p = 16) = 0;
    virtual void setMultiline(bool ml = true) = 0;

    static Ptr<Formatter> get(int fmt = FMT_DEFAULT);

};

static inline
String& operator << (String& out, Ptr<Formatted> fmtd)
{
    fmtd->reset();
    for(const char* str = fmtd->next(); str; str = fmtd->next())
        out += cv::String(str);
    return out;
}

static inline
String& operator << (String& out, const Mat& mtx)
{
    return out << Formatter::get()->format(mtx);
}

//////////////////////////////////////// Algorithm ////////////////////////////////////

class CV_EXPORTS Algorithm;

template<typename _Tp> struct ParamType {};


/** @brief This is a base class for all more or less complex algorithms in OpenCV

especially for classes of algorithms, for which there can be multiple implementations. The examples
are stereo correspondence (for which there are algorithms like block matching, semi-global block
matching, graph-cut etc.), background subtraction (which can be done using mixture-of-gaussians
models, codebook-based algorithm etc.), optical flow (block matching, Lucas-Kanade, Horn-Schunck
etc.).

Here is example of SIFT use in your application via Algorithm interface:
@code
    #include "opencv2/opencv.hpp"
    #include "opencv2/xfeatures2d.hpp"
    using namespace cv::xfeatures2d;

    Ptr<Feature2D> sift = SIFT::create();
    FileStorage fs("sift_params.xml", FileStorage::READ);
    if( fs.isOpened() ) // if we have file with parameters, read them
    {
        sift->read(fs["sift_params"]);
        fs.release();
    }
    else // else modify the parameters and store them; user can later edit the file to use different parameters
    {
        sift->setContrastThreshold(0.01f); // lower the contrast threshold, compared to the default value
        {
            WriteStructContext ws(fs, "sift_params", CV_NODE_MAP);
            sift->write(fs);
        }
    }
    Mat image = imread("myimage.png", 0), descriptors;
    vector<KeyPoint> keypoints;
    sift->detectAndCompute(image, noArray(), keypoints, descriptors);
@endcode
 */
class CV_EXPORTS_W Algorithm
{
public:
    Algorithm();
    virtual ~Algorithm();

    /** @brief Clears the algorithm state
    */
    CV_WRAP virtual void clear() {}

    /** @brief Stores algorithm parameters in a file storage
    */
    virtual void write(FileStorage& fs) const { (void)fs; }

    /** @brief Reads algorithm parameters from a file storage
    */
    virtual void read(const FileNode& fn) { (void)fn; }

    /** @brief Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
     */
    virtual bool empty() const { return false; }

    /** @brief Reads algorithm from the file node

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     cv::FileStorage fsRead("example.xml", FileStorage::READ);
     Ptr<SVM> svm = Algorithm::read<SVM>(fsRead.root());
     @endcode
     In order to make this method work, the derived class must overwrite Algorithm::read(const
     FileNode& fn) and also have static create() method without parameters
     (or with all the optional parameters)
     */
    template<typename _Tp> static Ptr<_Tp> read(const FileNode& fn)
    {
        Ptr<_Tp> obj = _Tp::create();
        obj->read(fn);
        return !obj->empty() ? obj : Ptr<_Tp>();
    }

    /** @brief Loads algorithm from the file

     @param filename Name of the file to read.
     @param objname The optional name of the node to read (if empty, the first top-level node will be used)

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     Ptr<SVM> svm = Algorithm::load<SVM>("my_svm_model.xml");
     @endcode
     In order to make this method work, the derived class must overwrite Algorithm::read(const
     FileNode& fn).
     */
    template<typename _Tp> static Ptr<_Tp> load(const String& filename, const String& objname=String())
    {
        FileStorage fs(filename, FileStorage::READ);
        FileNode fn = objname.empty() ? fs.getFirstTopLevelNode() : fs[objname];
        if (fn.empty()) return Ptr<_Tp>();
        Ptr<_Tp> obj = _Tp::create();
        obj->read(fn);
        return !obj->empty() ? obj : Ptr<_Tp>();
    }

    /** @brief Loads algorithm from a String

     @param strModel The string variable containing the model you want to load.
     @param objname The optional name of the node to read (if empty, the first top-level node will be used)

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     Ptr<SVM> svm = Algorithm::loadFromString<SVM>(myStringModel);
     @endcode
     */
    template<typename _Tp> static Ptr<_Tp> loadFromString(const String& strModel, const String& objname=String())
    {
        FileStorage fs(strModel, FileStorage::READ + FileStorage::MEMORY);
        FileNode fn = objname.empty() ? fs.getFirstTopLevelNode() : fs[objname];
        Ptr<_Tp> obj = _Tp::create();
        obj->read(fn);
        return !obj->empty() ? obj : Ptr<_Tp>();
    }

    /** Saves the algorithm to a file.
     In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs). */
    CV_WRAP virtual void save(const String& filename) const;

    /** Returns the algorithm string identifier.
     This string is used as top level xml/yml node tag when the object is saved to a file or string. */
    CV_WRAP virtual String getDefaultName() const;

protected:
    void writeFormat(FileStorage& fs) const;
};

struct Param {
    enum { INT=0, BOOLEAN=1, REAL=2, STRING=3, MAT=4, MAT_VECTOR=5, ALGORITHM=6, FLOAT=7,
           UNSIGNED_INT=8, UINT64=9, UCHAR=11 };
};



template<> struct ParamType<bool>
{
    typedef bool const_param_type;
    typedef bool member_type;

    enum { type = Param::BOOLEAN };
};

template<> struct ParamType<int>
{
    typedef int const_param_type;
    typedef int member_type;

    enum { type = Param::INT };
};

template<> struct ParamType<double>
{
    typedef double const_param_type;
    typedef double member_type;

    enum { type = Param::REAL };
};

template<> struct ParamType<String>
{
    typedef const String& const_param_type;
    typedef String member_type;

    enum { type = Param::STRING };
};

template<> struct ParamType<Mat>
{
    typedef const Mat& const_param_type;
    typedef Mat member_type;

    enum { type = Param::MAT };
};

template<> struct ParamType<std::vector<Mat> >
{
    typedef const std::vector<Mat>& const_param_type;
    typedef std::vector<Mat> member_type;

    enum { type = Param::MAT_VECTOR };
};

template<> struct ParamType<Algorithm>
{
    typedef const Ptr<Algorithm>& const_param_type;
    typedef Ptr<Algorithm> member_type;

    enum { type = Param::ALGORITHM };
};

template<> struct ParamType<float>
{
    typedef float const_param_type;
    typedef float member_type;

    enum { type = Param::FLOAT };
};

template<> struct ParamType<unsigned>
{
    typedef unsigned const_param_type;
    typedef unsigned member_type;

    enum { type = Param::UNSIGNED_INT };
};

template<> struct ParamType<uint64>
{
    typedef uint64 const_param_type;
    typedef uint64 member_type;

    enum { type = Param::UINT64 };
};

template<> struct ParamType<uchar>
{
    typedef uchar const_param_type;
    typedef uchar member_type;

    enum { type = Param::UCHAR };
};

//! @} core_basic

} //namespace cv

#include "opencv2/core/operations.hpp"
#include "opencv2/core/cvstd.inl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/optim.hpp"
#include "opencv2/core/ovx.hpp"

#endif /*OPENCV_CORE_HPP*/
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_DNN_HPP
#define OPENCV_DNN_HPP

// This is an umbrealla header to include into you project.
// We are free to change headers layout in dnn subfolder, so please include
// this header for future compatibility


/** @defgroup dnn Deep Neural Network module
  @{
    This module contains:
        - API for new layers creation, layers are building bricks of neural networks;
        - set of built-in most-useful Layers;
        - API to constuct and modify comprehensive neural networks from layers;
        - functionality for loading serialized networks models from differnet frameworks.

    Functionality of this module is designed only for forward pass computations (i. e. network testing).
    A network training is in principle not supported.
  @}
*/
#include <opencv2/dnn/dnn.hpp>

#endif /* OPENCV_DNN_HPP */
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

#ifndef OPENCV_FEATURES_2D_HPP
#define OPENCV_FEATURES_2D_HPP

#include "opencv2/core.hpp"
#include "opencv2/flann/miniflann.hpp"

/**
  @defgroup features2d 2D Features Framework
  @{
    @defgroup features2d_main Feature Detection and Description
    @defgroup features2d_match Descriptor Matchers

Matchers of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to
easily switch between different algorithms solving the same problem. This section is devoted to
matching descriptors that are represented as vectors in a multidimensional space. All objects that
implement vector descriptor matchers inherit the DescriptorMatcher interface.

@note
   -   An example explaining keypoint matching can be found at
        opencv_source_code/samples/cpp/descriptor_extractor_matcher.cpp
    -   An example on descriptor matching evaluation can be found at
        opencv_source_code/samples/cpp/detector_descriptor_matcher_evaluation.cpp
    -   An example on one to many image matching can be found at
        opencv_source_code/samples/cpp/matching_to_many_images.cpp

    @defgroup features2d_draw Drawing Function of Keypoints and Matches
    @defgroup features2d_category Object Categorization

This section describes approaches based on local 2D features and used to categorize objects.

@note
   -   A complete Bag-Of-Words sample can be found at
        opencv_source_code/samples/cpp/bagofwords_classification.cpp
    -   (Python) An example using the features2D framework to perform object categorization can be
        found at opencv_source_code/samples/python/find_obj.py

  @}
 */

namespace cv
{

//! @addtogroup features2d
//! @{

// //! writes vector of keypoints to the file storage
// CV_EXPORTS void write(FileStorage& fs, const String& name, const std::vector<KeyPoint>& keypoints);
// //! reads vector of keypoints from the specified file storage node
// CV_EXPORTS void read(const FileNode& node, CV_OUT std::vector<KeyPoint>& keypoints);

/** @brief A class filters a vector of keypoints.

 Because now it is difficult to provide a convenient interface for all usage scenarios of the
 keypoints filter class, it has only several needed by now static methods.
 */
class CV_EXPORTS KeyPointsFilter
{
public:
    KeyPointsFilter(){}

    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void runByImageBorder( std::vector<KeyPoint>& keypoints, Size imageSize, int borderSize );
    /*
     * Remove keypoints of sizes out of range.
     */
    static void runByKeypointSize( std::vector<KeyPoint>& keypoints, float minSize,
                                   float maxSize=FLT_MAX );
    /*
     * Remove keypoints from some image by mask for pixels of this image.
     */
    static void runByPixelsMask( std::vector<KeyPoint>& keypoints, const Mat& mask );
    /*
     * Remove duplicated keypoints.
     */
    static void removeDuplicated( std::vector<KeyPoint>& keypoints );

    /*
     * Retain the specified number of the best keypoints (according to the response)
     */
    static void retainBest( std::vector<KeyPoint>& keypoints, int npoints );
};


/************************************ Base Classes ************************************/

/** @brief Abstract base class for 2D image feature detectors and descriptor extractors
*/
class CV_EXPORTS_W Feature2D : public virtual Algorithm
{
public:
    virtual ~Feature2D();

    /** @brief Detects keypoints in an image (first variant) or image set (second variant).

    @param image Image.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
    matrix with non-zero values in the region of interest.
     */
    CV_WRAP virtual void detect( InputArray image,
                                 CV_OUT std::vector<KeyPoint>& keypoints,
                                 InputArray mask=noArray() );

    /** @overload
    @param images Image set.
    @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
    of keypoints detected in images[i] .
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    masks[i] is a mask for images[i].
    */
    CV_WRAP virtual void detect( InputArrayOfArrays images,
                         CV_OUT std::vector<std::vector<KeyPoint> >& keypoints,
                         InputArrayOfArrays masks=noArray() );

    /** @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
    (second variant).

    @param image Image.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
     */
    CV_WRAP virtual void compute( InputArray image,
                                  CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors );

    /** @overload

    @param images Image set.
    @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
    computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
    with several dominant orientations (for each orientation).
    @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
    descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
    descriptor for keypoint j-th keypoint.
    */
    CV_WRAP virtual void compute( InputArrayOfArrays images,
                          CV_OUT CV_IN_OUT std::vector<std::vector<KeyPoint> >& keypoints,
                          OutputArrayOfArrays descriptors );

    /** Detects keypoints and computes the descriptors */
    CV_WRAP virtual void detectAndCompute( InputArray image, InputArray mask,
                                           CV_OUT std::vector<KeyPoint>& keypoints,
                                           OutputArray descriptors,
                                           bool useProvidedKeypoints=false );

    CV_WRAP virtual int descriptorSize() const;
    CV_WRAP virtual int descriptorType() const;
    CV_WRAP virtual int defaultNorm() const;

    CV_WRAP void write( const String& fileName ) const;

    CV_WRAP void read( const String& fileName );

    virtual void write( FileStorage&) const;

    virtual void read( const FileNode&);

    //! Return true if detector object is empty
    CV_WRAP virtual bool empty() const;
};

/** Feature detectors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. All objects that implement keypoint detectors
inherit the FeatureDetector interface. */
typedef Feature2D FeatureDetector;

/** Extractors of keypoint descriptors in OpenCV have wrappers with a common interface that enables you
to easily switch between different algorithms solving the same problem. This section is devoted to
computing descriptors represented as vectors in a multidimensional space. All objects that implement
the vector descriptor extractors inherit the DescriptorExtractor interface.
 */
typedef Feature2D DescriptorExtractor;

//! @addtogroup features2d_main
//! @{

/** @brief Class implementing the BRISK keypoint detector and descriptor extractor, described in @cite LCS11 .
 */
class CV_EXPORTS_W BRISK : public Feature2D
{
public:
    /** @brief The BRISK constructor

    @param thresh AGAST detection threshold score.
    @param octaves detection octaves. Use 0 to do single scale.
    @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a
    keypoint.
     */
    CV_WRAP static Ptr<BRISK> create(int thresh=30, int octaves=3, float patternScale=1.0f);

    /** @brief The BRISK constructor for a custom pattern

    @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for
    keypoint scale 1).
    @param numberList defines the number of sampling points on the sampling circle. Must be the same
    size as radiusList..
    @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint
    scale 1).
    @param dMin threshold for the long pairings used for orientation determination (in pixels for
    keypoint scale 1).
    @param indexChange index remapping of the bits. */
    CV_WRAP static Ptr<BRISK> create(const std::vector<float> &radiusList, const std::vector<int> &numberList,
        float dMax=5.85f, float dMin=8.2f, const std::vector<int>& indexChange=std::vector<int>());
};

/** @brief Class implementing the ORB (*oriented BRIEF*) keypoint detector and descriptor extractor

described in @cite RRKB11 . The algorithm uses FAST in pyramids to detect stable keypoints, selects
the strongest features using FAST or Harris response, finds their orientation using first-order
moments and computes the descriptors using BRIEF (where the coordinates of random point pairs (or
k-tuples) are rotated according to the measured orientation).
 */
class CV_EXPORTS_W ORB : public Feature2D
{
public:
    enum { kBytes = 32, HARRIS_SCORE=0, FAST_SCORE=1 };

    /** @brief The ORB constructor

    @param nfeatures The maximum number of features to retain.
    @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
    pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
    will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
    will mean that to cover certain scale range you will need more pyramid levels and so the speed
    will suffer.
    @param nlevels The number of pyramid levels. The smallest level will have linear size equal to
    input_image_linear_size/pow(scaleFactor, nlevels).
    @param edgeThreshold This is size of the border where the features are not detected. It should
    roughly match the patchSize parameter.
    @param firstLevel It should be 0 in the current implementation.
    @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
    default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
    so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
    random points (of course, those point coordinates are random, but they are generated from the
    pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
    rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
    output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
    denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
    bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
    @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
    (the score is written to KeyPoint::score and is used to retain best nfeatures features);
    FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
    but it is a little faster to compute.
    @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
    pyramid layers the perceived image area covered by a feature will be larger.
    @param fastThreshold
     */
    CV_WRAP static Ptr<ORB> create(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31,
        int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20);

    CV_WRAP virtual void setMaxFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getMaxFeatures() const = 0;

    CV_WRAP virtual void setScaleFactor(double scaleFactor) = 0;
    CV_WRAP virtual double getScaleFactor() const = 0;

    CV_WRAP virtual void setNLevels(int nlevels) = 0;
    CV_WRAP virtual int getNLevels() const = 0;

    CV_WRAP virtual void setEdgeThreshold(int edgeThreshold) = 0;
    CV_WRAP virtual int getEdgeThreshold() const = 0;

    CV_WRAP virtual void setFirstLevel(int firstLevel) = 0;
    CV_WRAP virtual int getFirstLevel() const = 0;

    CV_WRAP virtual void setWTA_K(int wta_k) = 0;
    CV_WRAP virtual int getWTA_K() const = 0;

    CV_WRAP virtual void setScoreType(int scoreType) = 0;
    CV_WRAP virtual int getScoreType() const = 0;

    CV_WRAP virtual void setPatchSize(int patchSize) = 0;
    CV_WRAP virtual int getPatchSize() const = 0;

    CV_WRAP virtual void setFastThreshold(int fastThreshold) = 0;
    CV_WRAP virtual int getFastThreshold() const = 0;
};

/** @brief Maximally stable extremal region extractor

The class encapsulates all the parameters of the %MSER extraction algorithm (see [wiki
article](http://en.wikipedia.org/wiki/Maximally_stable_extremal_regions)).

- there are two different implementation of %MSER: one for grey image, one for color image

- the grey image algorithm is taken from: @cite nister2008linear ;  the paper claims to be faster
than union-find method; it actually get 1.5~2m/s on my centrino L7200 1.2GHz laptop.

- the color image algorithm is taken from: @cite forssen2007maximally ; it should be much slower
than grey image method ( 3~4 times ); the chi_table.h file is taken directly from paper's source
code which is distributed under GPL.

- (Python) A complete example showing the use of the %MSER detector can be found at samples/python/mser.py
*/
class CV_EXPORTS_W MSER : public Feature2D
{
public:
    /** @brief Full consturctor for %MSER detector

    @param _delta it compares \f$(size_{i}-size_{i-delta})/size_{i-delta}\f$
    @param _min_area prune the area which smaller than minArea
    @param _max_area prune the area which bigger than maxArea
    @param _max_variation prune the area have simliar size to its children
    @param _min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
    @param _max_evolution  for color image, the evolution steps
    @param _area_threshold for color image, the area threshold to cause re-initialize
    @param _min_margin for color image, ignore too small margin
    @param _edge_blur_size for color image, the aperture size for edge blur
     */
    CV_WRAP static Ptr<MSER> create( int _delta=5, int _min_area=60, int _max_area=14400,
          double _max_variation=0.25, double _min_diversity=.2,
          int _max_evolution=200, double _area_threshold=1.01,
          double _min_margin=0.003, int _edge_blur_size=5 );

    /** @brief Detect %MSER regions

    @param image input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
    @param msers resulting list of point sets
    @param bboxes resulting bounding boxes
    */
    CV_WRAP virtual void detectRegions( InputArray image,
                                        CV_OUT std::vector<std::vector<Point> >& msers,
                                        CV_OUT std::vector<Rect>& bboxes ) = 0;

    CV_WRAP virtual void setDelta(int delta) = 0;
    CV_WRAP virtual int getDelta() const = 0;

    CV_WRAP virtual void setMinArea(int minArea) = 0;
    CV_WRAP virtual int getMinArea() const = 0;

    CV_WRAP virtual void setMaxArea(int maxArea) = 0;
    CV_WRAP virtual int getMaxArea() const = 0;

    CV_WRAP virtual void setPass2Only(bool f) = 0;
    CV_WRAP virtual bool getPass2Only() const = 0;
};

/** @overload */
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true );

/** @brief Detects corners using the FAST algorithm

@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the three neighborhoods as defined in the paper:
FastFeatureDetector::TYPE_9_16, FastFeatureDetector::TYPE_7_12,
FastFeatureDetector::TYPE_5_8

Detects corners using the FAST algorithm by @cite Rosten06 .

@note In Python API, types are given as cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
cv2.FAST_FEATURE_DETECTOR_TYPE_7_12 and cv2.FAST_FEATURE_DETECTOR_TYPE_9_16. For corner
detection, use cv2.FAST.detect() method.
 */
CV_EXPORTS void FAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression, int type );

//! @} features2d_main

//! @addtogroup features2d_main
//! @{

/** @brief Wrapping class for feature detection using the FAST method. :
 */
class CV_EXPORTS_W FastFeatureDetector : public Feature2D
{
public:
    enum
    {
        TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2,
        THRESHOLD = 10000, NONMAX_SUPPRESSION=10001, FAST_N=10002,
    };

    CV_WRAP static Ptr<FastFeatureDetector> create( int threshold=10,
                                                    bool nonmaxSuppression=true,
                                                    int type=FastFeatureDetector::TYPE_9_16 );

    CV_WRAP virtual void setThreshold(int threshold) = 0;
    CV_WRAP virtual int getThreshold() const = 0;

    CV_WRAP virtual void setNonmaxSuppression(bool f) = 0;
    CV_WRAP virtual bool getNonmaxSuppression() const = 0;

    CV_WRAP virtual void setType(int type) = 0;
    CV_WRAP virtual int getType() const = 0;
};

/** @overload */
CV_EXPORTS void AGAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression=true );

/** @brief Detects corners using the AGAST algorithm

@param image grayscale image where keypoints (corners) are detected.
@param keypoints keypoints detected on the image.
@param threshold threshold on difference between intensity of the central pixel and pixels of a
circle around this pixel.
@param nonmaxSuppression if true, non-maximum suppression is applied to detected corners
(keypoints).
@param type one of the four neighborhoods as defined in the paper:
AgastFeatureDetector::AGAST_5_8, AgastFeatureDetector::AGAST_7_12d,
AgastFeatureDetector::AGAST_7_12s, AgastFeatureDetector::OAST_9_16

For non-Intel platforms, there is a tree optimised variant of AGAST with same numerical results.
The 32-bit binary tree tables were generated automatically from original code using perl script.
The perl script and examples of tree generation are placed in features2d/doc folder.
Detects corners using the AGAST algorithm by @cite mair2010_agast .

 */
CV_EXPORTS void AGAST( InputArray image, CV_OUT std::vector<KeyPoint>& keypoints,
                      int threshold, bool nonmaxSuppression, int type );
//! @} features2d_main

//! @addtogroup features2d_main
//! @{

/** @brief Wrapping class for feature detection using the AGAST method. :
 */
class CV_EXPORTS_W AgastFeatureDetector : public Feature2D
{
public:
    enum
    {
        AGAST_5_8 = 0, AGAST_7_12d = 1, AGAST_7_12s = 2, OAST_9_16 = 3,
        THRESHOLD = 10000, NONMAX_SUPPRESSION = 10001,
    };

    CV_WRAP static Ptr<AgastFeatureDetector> create( int threshold=10,
                                                     bool nonmaxSuppression=true,
                                                     int type=AgastFeatureDetector::OAST_9_16 );

    CV_WRAP virtual void setThreshold(int threshold) = 0;
    CV_WRAP virtual int getThreshold() const = 0;

    CV_WRAP virtual void setNonmaxSuppression(bool f) = 0;
    CV_WRAP virtual bool getNonmaxSuppression() const = 0;

    CV_WRAP virtual void setType(int type) = 0;
    CV_WRAP virtual int getType() const = 0;
};

/** @brief Wrapping class for feature detection using the goodFeaturesToTrack function. :
 */
class CV_EXPORTS_W GFTTDetector : public Feature2D
{
public:
    CV_WRAP static Ptr<GFTTDetector> create( int maxCorners=1000, double qualityLevel=0.01, double minDistance=1,
                                             int blockSize=3, bool useHarrisDetector=false, double k=0.04 );
    CV_WRAP virtual void setMaxFeatures(int maxFeatures) = 0;
    CV_WRAP virtual int getMaxFeatures() const = 0;

    CV_WRAP virtual void setQualityLevel(double qlevel) = 0;
    CV_WRAP virtual double getQualityLevel() const = 0;

    CV_WRAP virtual void setMinDistance(double minDistance) = 0;
    CV_WRAP virtual double getMinDistance() const = 0;

    CV_WRAP virtual void setBlockSize(int blockSize) = 0;
    CV_WRAP virtual int getBlockSize() const = 0;

    CV_WRAP virtual void setHarrisDetector(bool val) = 0;
    CV_WRAP virtual bool getHarrisDetector() const = 0;

    CV_WRAP virtual void setK(double k) = 0;
    CV_WRAP virtual double getK() const = 0;
};

/** @brief Class for extracting blobs from an image. :

The class implements a simple algorithm for extracting blobs from an image:

1.  Convert the source image to binary images by applying thresholding with several thresholds from
    minThreshold (inclusive) to maxThreshold (exclusive) with distance thresholdStep between
    neighboring thresholds.
2.  Extract connected components from every binary image by findContours and calculate their
    centers.
3.  Group centers from several binary images by their coordinates. Close centers form one group that
    corresponds to one blob, which is controlled by the minDistBetweenBlobs parameter.
4.  From the groups, estimate final centers of blobs and their radiuses and return as locations and
    sizes of keypoints.

This class performs several filtrations of returned blobs. You should set filterBy\* to true/false
to turn on/off corresponding filtration. Available filtrations:

-   **By color**. This filter compares the intensity of a binary image at the center of a blob to
blobColor. If they differ, the blob is filtered out. Use blobColor = 0 to extract dark blobs
and blobColor = 255 to extract light blobs.
-   **By area**. Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
-   **By circularity**. Extracted blobs have circularity
(\f$\frac{4*\pi*Area}{perimeter * perimeter}\f$) between minCircularity (inclusive) and
maxCircularity (exclusive).
-   **By ratio of the minimum inertia to maximum inertia**. Extracted blobs have this ratio
between minInertiaRatio (inclusive) and maxInertiaRatio (exclusive).
-   **By convexity**. Extracted blobs have convexity (area / area of blob convex hull) between
minConvexity (inclusive) and maxConvexity (exclusive).

Default values of parameters are tuned to extract dark circular blobs.
 */
class CV_EXPORTS_W SimpleBlobDetector : public Feature2D
{
public:
  struct CV_EXPORTS_W_SIMPLE Params
  {
      CV_WRAP Params();
      CV_PROP_RW float thresholdStep;
      CV_PROP_RW float minThreshold;
      CV_PROP_RW float maxThreshold;
      CV_PROP_RW size_t minRepeatability;
      CV_PROP_RW float minDistBetweenBlobs;

      CV_PROP_RW bool filterByColor;
      CV_PROP_RW uchar blobColor;

      CV_PROP_RW bool filterByArea;
      CV_PROP_RW float minArea, maxArea;

      CV_PROP_RW bool filterByCircularity;
      CV_PROP_RW float minCircularity, maxCircularity;

      CV_PROP_RW bool filterByInertia;
      CV_PROP_RW float minInertiaRatio, maxInertiaRatio;

      CV_PROP_RW bool filterByConvexity;
      CV_PROP_RW float minConvexity, maxConvexity;

      void read( const FileNode& fn );
      void write( FileStorage& fs ) const;
  };

  CV_WRAP static Ptr<SimpleBlobDetector>
    create(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());
};

//! @} features2d_main

//! @addtogroup features2d_main
//! @{

/** @brief Class implementing the KAZE keypoint detector and descriptor extractor, described in @cite ABD12 .

@note AKAZE descriptor can only be used with KAZE or AKAZE keypoints .. [ABD12] KAZE Features. Pablo
F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. In European Conference on Computer Vision
(ECCV), Fiorenze, Italy, October 2012.
*/
class CV_EXPORTS_W KAZE : public Feature2D
{
public:
    enum
    {
        DIFF_PM_G1 = 0,
        DIFF_PM_G2 = 1,
        DIFF_WEICKERT = 2,
        DIFF_CHARBONNIER = 3
    };

    /** @brief The KAZE constructor

    @param extended Set to enable extraction of extended (128-byte) descriptor.
    @param upright Set to enable use of upright descriptors (non rotation-invariant).
    @param threshold Detector response threshold to accept point
    @param nOctaves Maximum octave evolution of the image
    @param nOctaveLayers Default number of sublevels per scale level
    @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
    DIFF_CHARBONNIER
     */
    CV_WRAP static Ptr<KAZE> create(bool extended=false, bool upright=false,
                                    float threshold = 0.001f,
                                    int nOctaves = 4, int nOctaveLayers = 4,
                                    int diffusivity = KAZE::DIFF_PM_G2);

    CV_WRAP virtual void setExtended(bool extended) = 0;
    CV_WRAP virtual bool getExtended() const = 0;

    CV_WRAP virtual void setUpright(bool upright) = 0;
    CV_WRAP virtual bool getUpright() const = 0;

    CV_WRAP virtual void setThreshold(double threshold) = 0;
    CV_WRAP virtual double getThreshold() const = 0;

    CV_WRAP virtual void setNOctaves(int octaves) = 0;
    CV_WRAP virtual int getNOctaves() const = 0;

    CV_WRAP virtual void setNOctaveLayers(int octaveLayers) = 0;
    CV_WRAP virtual int getNOctaveLayers() const = 0;

    CV_WRAP virtual void setDiffusivity(int diff) = 0;
    CV_WRAP virtual int getDiffusivity() const = 0;
};

/** @brief Class implementing the AKAZE keypoint detector and descriptor extractor, described in @cite ANB13 . :

@note AKAZE descriptors can only be used with KAZE or AKAZE keypoints. Try to avoid using *extract*
and *detect* instead of *operator()* due to performance reasons. .. [ANB13] Fast Explicit Diffusion
for Accelerated Features in Nonlinear Scale Spaces. Pablo F. Alcantarilla, Jesús Nuevo and Adrien
Bartoli. In British Machine Vision Conference (BMVC), Bristol, UK, September 2013.
 */
class CV_EXPORTS_W AKAZE : public Feature2D
{
public:
    // AKAZE descriptor type
    enum
    {
        DESCRIPTOR_KAZE_UPRIGHT = 2, ///< Upright descriptors, not invariant to rotation
        DESCRIPTOR_KAZE = 3,
        DESCRIPTOR_MLDB_UPRIGHT = 4, ///< Upright descriptors, not invariant to rotation
        DESCRIPTOR_MLDB = 5
    };

    /** @brief The AKAZE constructor

    @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE,
    DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    @param descriptor_size Size of the descriptor in bits. 0 -\> Full size
    @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
    @param threshold Detector response threshold to accept point
    @param nOctaves Maximum octave evolution of the image
    @param nOctaveLayers Default number of sublevels per scale level
    @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or
    DIFF_CHARBONNIER
     */
    CV_WRAP static Ptr<AKAZE> create(int descriptor_type=AKAZE::DESCRIPTOR_MLDB,
                                     int descriptor_size = 0, int descriptor_channels = 3,
                                     float threshold = 0.001f, int nOctaves = 4,
                                     int nOctaveLayers = 4, int diffusivity = KAZE::DIFF_PM_G2);

    CV_WRAP virtual void setDescriptorType(int dtype) = 0;
    CV_WRAP virtual int getDescriptorType() const = 0;

    CV_WRAP virtual void setDescriptorSize(int dsize) = 0;
    CV_WRAP virtual int getDescriptorSize() const = 0;

    CV_WRAP virtual void setDescriptorChannels(int dch) = 0;
    CV_WRAP virtual int getDescriptorChannels() const = 0;

    CV_WRAP virtual void setThreshold(double threshold) = 0;
    CV_WRAP virtual double getThreshold() const = 0;

    CV_WRAP virtual void setNOctaves(int octaves) = 0;
    CV_WRAP virtual int getNOctaves() const = 0;

    CV_WRAP virtual void setNOctaveLayers(int octaveLayers) = 0;
    CV_WRAP virtual int getNOctaveLayers() const = 0;

    CV_WRAP virtual void setDiffusivity(int diff) = 0;
    CV_WRAP virtual int getDiffusivity() const = 0;
};

//! @} features2d_main

/****************************************************************************************\
*                                      Distance                                          *
\****************************************************************************************/

template<typename T>
struct CV_EXPORTS Accumulator
{
    typedef T Type;
};

template<> struct Accumulator<unsigned char>  { typedef float Type; };
template<> struct Accumulator<unsigned short> { typedef float Type; };
template<> struct Accumulator<char>   { typedef float Type; };
template<> struct Accumulator<short>  { typedef float Type; };

/*
 * Squared Euclidean distance functor
 */
template<class T>
struct CV_EXPORTS SL2
{
    enum { normType = NORM_L2SQR };
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return normL2Sqr<ValueType, ResultType>(a, b, size);
    }
};

/*
 * Euclidean distance functor
 */
template<class T>
struct CV_EXPORTS L2
{
    enum { normType = NORM_L2 };
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return (ResultType)std::sqrt((double)normL2Sqr<ValueType, ResultType>(a, b, size));
    }
};

/*
 * Manhattan distance (city block distance) functor
 */
template<class T>
struct CV_EXPORTS L1
{
    enum { normType = NORM_L1 };
    typedef T ValueType;
    typedef typename Accumulator<T>::Type ResultType;

    ResultType operator()( const T* a, const T* b, int size ) const
    {
        return normL1<ValueType, ResultType>(a, b, size);
    }
};

/****************************************************************************************\
*                                  DescriptorMatcher                                     *
\****************************************************************************************/

//! @addtogroup features2d_match
//! @{

/** @brief Abstract base class for matching keypoint descriptors.

It has two groups of match methods: for matching descriptors of an image with another image or with
an image set.
 */
class CV_EXPORTS_W DescriptorMatcher : public Algorithm
{
public:
   enum
    {
        FLANNBASED            = 1,
        BRUTEFORCE            = 2,
        BRUTEFORCE_L1         = 3,
        BRUTEFORCE_HAMMING    = 4,
        BRUTEFORCE_HAMMINGLUT = 5,
        BRUTEFORCE_SL2        = 6
    };
    virtual ~DescriptorMatcher();

    /** @brief Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptor
    collection.

    If the collection is not empty, the new descriptors are added to existing train descriptors.

    @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same
    train image.
     */
    CV_WRAP virtual void add( InputArrayOfArrays descriptors );

    /** @brief Returns a constant link to the train descriptor collection trainDescCollection .
     */
    CV_WRAP const std::vector<Mat>& getTrainDescriptors() const;

    /** @brief Clears the train descriptor collections.
     */
    CV_WRAP virtual void clear();

    /** @brief Returns true if there are no train descriptors in the both collections.
     */
    CV_WRAP virtual bool empty() const;

    /** @brief Returns true if the descriptor matcher supports masking permissible matches.
     */
    CV_WRAP virtual bool isMaskSupported() const = 0;

    /** @brief Trains a descriptor matcher

    Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
    train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher)
    have an empty implementation of this method. Other matchers really train their inner structures (for
    example, FlannBasedMatcher trains flann::Index ).
     */
    CV_WRAP virtual void train();

    /** @brief Finds the best match for each descriptor from a query set.

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
    descriptor. So, matches size may be smaller than the query descriptors count.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.

    In the first variant of this method, the train descriptors are passed as an input argument. In the
    second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is
    used. Optional mask (or masks) can be passed to specify which query and training descriptors can be
    matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if
    mask.at\<uchar\>(i,j) is non-zero.
     */
    CV_WRAP void match( InputArray queryDescriptors, InputArray trainDescriptors,
                CV_OUT std::vector<DMatch>& matches, InputArray mask=noArray() ) const;

    /** @brief Finds the k best matches for each descriptor from a query set.

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.

    These extended variants of DescriptorMatcher::match methods find several best matches for each query
    descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match
    for the details about query and train descriptors.
     */
    CV_WRAP void knnMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                   CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                   InputArray mask=noArray(), bool compactResult=false ) const;

    /** @brief For each query descriptor, finds the training descriptors not farther than the specified distance.

    @param queryDescriptors Query set of descriptors.
    @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors
    collection stored in the class object.
    @param matches Found matches.
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param mask Mask specifying permissible matches between an input query and train matrices of
    descriptors.

    For each query descriptor, the methods find such training descriptors that the distance between the
    query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are
    returned in the distance increasing order.
     */
    CV_WRAP void radiusMatch( InputArray queryDescriptors, InputArray trainDescriptors,
                      CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArray mask=noArray(), bool compactResult=false ) const;

    /** @overload
    @param queryDescriptors Query set of descriptors.
    @param matches Matches. If a query descriptor is masked out in mask , no match is added for this
    descriptor. So, matches size may be smaller than the query descriptors count.
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    */
    CV_WRAP void match( InputArray queryDescriptors, CV_OUT std::vector<DMatch>& matches,
                        InputArrayOfArrays masks=noArray() );
    /** @overload
    @param queryDescriptors Query set of descriptors.
    @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
    @param k Count of best matches found per each query descriptor or less if a query descriptor has
    less than k possible matches in total.
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    */
    CV_WRAP void knnMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, int k,
                           InputArrayOfArrays masks=noArray(), bool compactResult=false );
    /** @overload
    @param queryDescriptors Query set of descriptors.
    @param matches Found matches.
    @param maxDistance Threshold for the distance between matched descriptors. Distance means here
    metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured
    in Pixels)!
    @param masks Set of masks. Each masks[i] specifies permissible matches between the input query
    descriptors and stored train descriptors from the i-th image trainDescCollection[i].
    @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is
    false, the matches vector has the same size as queryDescriptors rows. If compactResult is true,
    the matches vector does not contain matches for fully masked-out query descriptors.
    */
    CV_WRAP void radiusMatch( InputArray queryDescriptors, CV_OUT std::vector<std::vector<DMatch> >& matches, float maxDistance,
                      InputArrayOfArrays masks=noArray(), bool compactResult=false );


    CV_WRAP void write( const String& fileName ) const
    {
        FileStorage fs(fileName, FileStorage::WRITE);
        write(fs);
    }

    CV_WRAP void read( const String& fileName )
    {
        FileStorage fs(fileName, FileStorage::READ);
        read(fs.root());
    }
    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    /** @brief Clones the matcher.

    @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object,
    that is, copies both parameters and train data. If emptyTrainData is true, the method creates an
    object copy with the current parameters but with empty train data.
     */
    CV_WRAP virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const = 0;

    /** @brief Creates a descriptor matcher of a given type with the default parameters (using default
    constructor).

    @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are
    supported:
    -   `BruteForce` (it uses L2 )
    -   `BruteForce-L1`
    -   `BruteForce-Hamming`
    -   `BruteForce-Hamming(2)`
    -   `FlannBased`
     */
    CV_WRAP static Ptr<DescriptorMatcher> create( const String& descriptorMatcherType );

    CV_WRAP static Ptr<DescriptorMatcher> create( int matcherType );

protected:
    /**
     * Class to work with descriptors from several images as with one merged matrix.
     * It is used e.g. in FlannBasedMatcher.
     */
    class CV_EXPORTS DescriptorCollection
    {
    public:
        DescriptorCollection();
        DescriptorCollection( const DescriptorCollection& collection );
        virtual ~DescriptorCollection();

        // Vector of matrices "descriptors" will be merged to one matrix "mergedDescriptors" here.
        void set( const std::vector<Mat>& descriptors );
        virtual void clear();

        const Mat& getDescriptors() const;
        const Mat getDescriptor( int imgIdx, int localDescIdx ) const;
        const Mat getDescriptor( int globalDescIdx ) const;
        void getLocalIdx( int globalDescIdx, int& imgIdx, int& localDescIdx ) const;

        int size() const;

    protected:
        Mat mergedDescriptors;
        std::vector<int> startIdxs;
    };

    //! In fact the matching is implemented only by the following two methods. These methods suppose
    //! that the class object has been trained already. Public match methods call these methods
    //! after calling train().
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) = 0;
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false ) = 0;

    static bool isPossibleMatch( InputArray mask, int queryIdx, int trainIdx );
    static bool isMaskedOut( InputArrayOfArrays masks, int queryIdx );

    static Mat clone_op( Mat m ) { return m.clone(); }
    void checkMasks( InputArrayOfArrays masks, int queryDescriptorsCount ) const;

    //! Collection of descriptors from train images.
    std::vector<Mat> trainDescCollection;
    std::vector<UMat> utrainDescCollection;
};

/** @brief Brute-force descriptor matcher.

For each descriptor in the first set, this matcher finds the closest descriptor in the second set
by trying each one. This descriptor matcher supports masking permissible matches of descriptor
sets.
 */
class CV_EXPORTS_W BFMatcher : public DescriptorMatcher
{
public:
    /** @brief Brute-force matcher constructor (obsolete). Please use BFMatcher.create()
     *
     *
    */
    CV_WRAP BFMatcher( int normType=NORM_L2, bool crossCheck=false );

    virtual ~BFMatcher() {}

    virtual bool isMaskSupported() const { return true; }

    /* @brief Brute-force matcher create method.
    @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are
    preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and
    BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor
    description).
    @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k
    nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with
    k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the
    matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent
    pairs. Such technique usually produces best results with minimal number of outliers when there are
    enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
     */
    CV_WRAP static Ptr<BFMatcher> create( int normType=NORM_L2, bool crossCheck=false ) ;

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;
protected:
    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );

    int normType;
    bool crossCheck;
};


/** @brief Flann-based descriptor matcher.

This matcher trains cv::flann::Index on a train descriptor collection and calls its nearest search
methods to find the best matches. So, this matcher may be faster when matching a large train
collection than the brute force matcher. FlannBasedMatcher does not support masking permissible
matches of descriptor sets because flann::Index does not support this. :
 */
class CV_EXPORTS_W FlannBasedMatcher : public DescriptorMatcher
{
public:
    CV_WRAP FlannBasedMatcher( const Ptr<flann::IndexParams>& indexParams=makePtr<flann::KDTreeIndexParams>(),
                       const Ptr<flann::SearchParams>& searchParams=makePtr<flann::SearchParams>() );

    virtual void add( InputArrayOfArrays descriptors );
    virtual void clear();

    // Reads matcher object from a file node
    virtual void read( const FileNode& );
    // Writes matcher object to a file storage
    virtual void write( FileStorage& ) const;

    virtual void train();
    virtual bool isMaskSupported() const;

    CV_WRAP static Ptr<FlannBasedMatcher> create();

    virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;
protected:
    static void convertToDMatches( const DescriptorCollection& descriptors,
                                   const Mat& indices, const Mat& distances,
                                   std::vector<std::vector<DMatch> >& matches );

    virtual void knnMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );
    virtual void radiusMatchImpl( InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
        InputArrayOfArrays masks=noArray(), bool compactResult=false );

    Ptr<flann::IndexParams> indexParams;
    Ptr<flann::SearchParams> searchParams;
    Ptr<flann::Index> flannIndex;

    DescriptorCollection mergedDescriptors;
    int addedDescCount;
};

//! @} features2d_match

/****************************************************************************************\
*                                   Drawing functions                                    *
\****************************************************************************************/

//! @addtogroup features2d_draw
//! @{

struct CV_EXPORTS DrawMatchesFlags
{
    enum{ DEFAULT = 0, //!< Output image matrix will be created (Mat::create),
                       //!< i.e. existing memory of output image may be reused.
                       //!< Two source image, matches and single keypoints will be drawn.
                       //!< For each keypoint only the center point will be drawn (without
                       //!< the circle around keypoint with keypoint size and orientation).
          DRAW_OVER_OUTIMG = 1, //!< Output image matrix will not be created (Mat::create).
                                //!< Matches will be drawn on existing content of output image.
          NOT_DRAW_SINGLE_POINTS = 2, //!< Single keypoints will not be drawn.
          DRAW_RICH_KEYPOINTS = 4 //!< For each keypoint the circle around keypoint with keypoint size and
                                  //!< orientation will be drawn.
        };
};

/** @brief Draws keypoints.

@param image Source image.
@param keypoints Keypoints from the source image.
@param outImage Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param color Color of keypoints.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags. See details above in drawMatches .

@note
For Python API, flags are modified as cv2.DRAW_MATCHES_FLAGS_DEFAULT,
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
 */
CV_EXPORTS_W void drawKeypoints( InputArray image, const std::vector<KeyPoint>& keypoints, InputOutputArray outImage,
                               const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT );

/** @brief Draws the found matches of keypoints from two images.

@param img1 First source image.
@param keypoints1 Keypoints from the first source image.
@param img2 Second source image.
@param keypoints2 Keypoints from the second source image.
@param matches1to2 Matches from the first image to the second one, which means that keypoints1[i]
has a corresponding point in keypoints2[matches[i]] .
@param outImg Output image. Its content depends on the flags value defining what is drawn in the
output image. See possible flags bit values below.
@param matchColor Color of matches (lines and connected keypoints). If matchColor==Scalar::all(-1)
, the color is generated randomly.
@param singlePointColor Color of single keypoints (circles), which means that keypoints do not
have the matches. If singlePointColor==Scalar::all(-1) , the color is generated randomly.
@param matchesMask Mask determining which matches are drawn. If the mask is empty, all matches are
drawn.
@param flags Flags setting drawing features. Possible flags bit values are defined by
DrawMatchesFlags.

This function draws matches of keypoints from two images in the output image. Match is a line
connecting two keypoints (circles). See cv::DrawMatchesFlags.
 */
CV_EXPORTS_W void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<char>& matchesMask=std::vector<char>(), int flags=DrawMatchesFlags::DEFAULT );

/** @overload */
CV_EXPORTS_AS(drawMatchesKnn) void drawMatches( InputArray img1, const std::vector<KeyPoint>& keypoints1,
                             InputArray img2, const std::vector<KeyPoint>& keypoints2,
                             const std::vector<std::vector<DMatch> >& matches1to2, InputOutputArray outImg,
                             const Scalar& matchColor=Scalar::all(-1), const Scalar& singlePointColor=Scalar::all(-1),
                             const std::vector<std::vector<char> >& matchesMask=std::vector<std::vector<char> >(), int flags=DrawMatchesFlags::DEFAULT );

//! @} features2d_draw

/****************************************************************************************\
*   Functions to evaluate the feature detectors and [generic] descriptor extractors      *
\****************************************************************************************/

CV_EXPORTS void evaluateFeatureDetector( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                         std::vector<KeyPoint>* keypoints1, std::vector<KeyPoint>* keypoints2,
                                         float& repeatability, int& correspCount,
                                         const Ptr<FeatureDetector>& fdetector=Ptr<FeatureDetector>() );

CV_EXPORTS void computeRecallPrecisionCurve( const std::vector<std::vector<DMatch> >& matches1to2,
                                             const std::vector<std::vector<uchar> >& correctMatches1to2Mask,
                                             std::vector<Point2f>& recallPrecisionCurve );

CV_EXPORTS float getRecall( const std::vector<Point2f>& recallPrecisionCurve, float l_precision );
CV_EXPORTS int getNearestPoint( const std::vector<Point2f>& recallPrecisionCurve, float l_precision );

/****************************************************************************************\
*                                     Bag of visual words                                *
\****************************************************************************************/

//! @addtogroup features2d_category
//! @{

/** @brief Abstract base class for training the *bag of visual words* vocabulary from a set of descriptors.

For details, see, for example, *Visual Categorization with Bags of Keypoints* by Gabriella Csurka,
Christopher R. Dance, Lixin Fan, Jutta Willamowski, Cedric Bray, 2004. :
 */
class CV_EXPORTS_W BOWTrainer
{
public:
    BOWTrainer();
    virtual ~BOWTrainer();

    /** @brief Adds descriptors to a training set.

    @param descriptors Descriptors to add to a training set. Each row of the descriptors matrix is a
    descriptor.

    The training set is clustered using clustermethod to construct the vocabulary.
     */
    CV_WRAP void add( const Mat& descriptors );

    /** @brief Returns a training set of descriptors.
    */
    CV_WRAP const std::vector<Mat>& getDescriptors() const;

    /** @brief Returns the count of all descriptors stored in the training set.
    */
    CV_WRAP int descriptorsCount() const;

    CV_WRAP virtual void clear();

    /** @overload */
    CV_WRAP virtual Mat cluster() const = 0;

    /** @brief Clusters train descriptors.

    @param descriptors Descriptors to cluster. Each row of the descriptors matrix is a descriptor.
    Descriptors are not added to the inner train descriptor set.

    The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first
    variant of the method, train descriptors stored in the object are clustered. In the second variant,
    input descriptors are clustered.
     */
    CV_WRAP virtual Mat cluster( const Mat& descriptors ) const = 0;

protected:
    std::vector<Mat> descriptors;
    int size;
};

/** @brief kmeans -based class to train visual vocabulary using the *bag of visual words* approach. :
 */
class CV_EXPORTS_W BOWKMeansTrainer : public BOWTrainer
{
public:
    /** @brief The constructor.

    @see cv::kmeans
    */
    CV_WRAP BOWKMeansTrainer( int clusterCount, const TermCriteria& termcrit=TermCriteria(),
                      int attempts=3, int flags=KMEANS_PP_CENTERS );
    virtual ~BOWKMeansTrainer();

    // Returns trained vocabulary (i.e. cluster centers).
    CV_WRAP virtual Mat cluster() const;
    CV_WRAP virtual Mat cluster( const Mat& descriptors ) const;

protected:

    int clusterCount;
    TermCriteria termcrit;
    int attempts;
    int flags;
};

/** @brief Class to compute an image descriptor using the *bag of visual words*.

Such a computation consists of the following steps:

1.  Compute descriptors for a given image and its keypoints set.
2.  Find the nearest visual words from the vocabulary for each keypoint descriptor.
3.  Compute the bag-of-words image descriptor as is a normalized histogram of vocabulary words
encountered in the image. The i-th bin of the histogram is a frequency of i-th word of the
vocabulary in the given image.
 */
class CV_EXPORTS_W BOWImgDescriptorExtractor
{
public:
    /** @brief The constructor.

    @param dextractor Descriptor extractor that is used to compute descriptors for an input image and
    its keypoints.
    @param dmatcher Descriptor matcher that is used to find the nearest word of the trained vocabulary
    for each keypoint descriptor of the image.
     */
    CV_WRAP BOWImgDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor,
                               const Ptr<DescriptorMatcher>& dmatcher );
    /** @overload */
    BOWImgDescriptorExtractor( const Ptr<DescriptorMatcher>& dmatcher );
    virtual ~BOWImgDescriptorExtractor();

    /** @brief Sets a visual vocabulary.

    @param vocabulary Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of the
    vocabulary is a visual word (cluster center).
     */
    CV_WRAP void setVocabulary( const Mat& vocabulary );

    /** @brief Returns the set vocabulary.
    */
    CV_WRAP const Mat& getVocabulary() const;

    /** @brief Computes an image descriptor using the set visual vocabulary.

    @param image Image, for which the descriptor is computed.
    @param keypoints Keypoints detected in the input image.
    @param imgDescriptor Computed output image descriptor.
    @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that
    pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
    returned if it is non-zero.
    @param descriptors Descriptors of the image keypoints that are returned if they are non-zero.
     */
    void compute( InputArray image, std::vector<KeyPoint>& keypoints, OutputArray imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0, Mat* descriptors=0 );
    /** @overload
    @param keypointDescriptors Computed descriptors to match with vocabulary.
    @param imgDescriptor Computed output image descriptor.
    @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that
    pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary)
    returned if it is non-zero.
    */
    void compute( InputArray keypointDescriptors, OutputArray imgDescriptor,
                  std::vector<std::vector<int> >* pointIdxsOfClusters=0 );
    // compute() is not constant because DescriptorMatcher::match is not constant

    CV_WRAP_AS(compute) void compute2( const Mat& image, std::vector<KeyPoint>& keypoints, CV_OUT Mat& imgDescriptor )
    { compute(image,keypoints,imgDescriptor); }

    /** @brief Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.
    */
    CV_WRAP int descriptorSize() const;

    /** @brief Returns an image descriptor type.
     */
    CV_WRAP int descriptorType() const;

protected:
    Mat vocabulary;
    Ptr<DescriptorExtractor> dextractor;
    Ptr<DescriptorMatcher> dmatcher;
};

//! @} features2d_category

//! @} features2d

} /* namespace cv */

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

#ifndef OPENCV_FLANN_HPP
#define OPENCV_FLANN_HPP

#include "opencv2/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/flann/flann_base.hpp"

/**
@defgroup flann Clustering and Search in Multi-Dimensional Spaces

This section documents OpenCV's interface to the FLANN library. FLANN (Fast Library for Approximate
Nearest Neighbors) is a library that contains a collection of algorithms optimized for fast nearest
neighbor search in large datasets and for high dimensional features. More information about FLANN
can be found in @cite Muja2009 .
*/

namespace cvflann
{
    CV_EXPORTS flann_distance_t flann_distance_type();
    CV_DEPRECATED CV_EXPORTS void set_distance_type(flann_distance_t distance_type, int order);
}


namespace cv
{
namespace flann
{


//! @addtogroup flann
//! @{

template <typename T> struct CvType {};
template <> struct CvType<unsigned char> { static int type() { return CV_8U; } };
template <> struct CvType<char> { static int type() { return CV_8S; } };
template <> struct CvType<unsigned short> { static int type() { return CV_16U; } };
template <> struct CvType<short> { static int type() { return CV_16S; } };
template <> struct CvType<int> { static int type() { return CV_32S; } };
template <> struct CvType<float> { static int type() { return CV_32F; } };
template <> struct CvType<double> { static int type() { return CV_64F; } };


// bring the flann parameters into this namespace
using ::cvflann::get_param;
using ::cvflann::print_params;

// bring the flann distances into this namespace
using ::cvflann::L2_Simple;
using ::cvflann::L2;
using ::cvflann::L1;
using ::cvflann::MinkowskiDistance;
using ::cvflann::MaxDistance;
using ::cvflann::HammingLUT;
using ::cvflann::Hamming;
using ::cvflann::Hamming2;
using ::cvflann::HistIntersectionDistance;
using ::cvflann::HellingerDistance;
using ::cvflann::ChiSquareDistance;
using ::cvflann::KL_Divergence;


/** @brief The FLANN nearest neighbor index class. This class is templated with the type of elements for which
the index is built.
 */
template <typename Distance>
class GenericIndex
{
public:
        typedef typename Distance::ElementType ElementType;
        typedef typename Distance::ResultType DistanceType;

        /** @brief Constructs a nearest neighbor search index for a given dataset.

        @param features Matrix of containing the features(points) to index. The size of the matrix is
        num_features x feature_dimensionality and the data type of the elements in the matrix must
        coincide with the type of the index.
        @param params Structure containing the index parameters. The type of index that will be
        constructed depends on the type of this parameter. See the description.
        @param distance

        The method constructs a fast search structure from a set of features using the specified algorithm
        with specified parameters, as defined by params. params is a reference to one of the following class
        IndexParams descendants:

        - **LinearIndexParams** When passing an object of this type, the index will perform a linear,
        brute-force search. :
        @code
        struct LinearIndexParams : public IndexParams
        {
        };
        @endcode
        - **KDTreeIndexParams** When passing an object of this type the index constructed will consist of
        a set of randomized kd-trees which will be searched in parallel. :
        @code
        struct KDTreeIndexParams : public IndexParams
        {
            KDTreeIndexParams( int trees = 4 );
        };
        @endcode
        - **KMeansIndexParams** When passing an object of this type the index constructed will be a
        hierarchical k-means tree. :
        @code
        struct KMeansIndexParams : public IndexParams
        {
            KMeansIndexParams(
                int branching = 32,
                int iterations = 11,
                flann_centers_init_t centers_init = CENTERS_RANDOM,
                float cb_index = 0.2 );
        };
        @endcode
        - **CompositeIndexParams** When using a parameters object of this type the index created
        combines the randomized kd-trees and the hierarchical k-means tree. :
        @code
        struct CompositeIndexParams : public IndexParams
        {
            CompositeIndexParams(
                int trees = 4,
                int branching = 32,
                int iterations = 11,
                flann_centers_init_t centers_init = CENTERS_RANDOM,
                float cb_index = 0.2 );
        };
        @endcode
        - **LshIndexParams** When using a parameters object of this type the index created uses
        multi-probe LSH (by Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
        by Qin Lv, William Josephson, Zhe Wang, Moses Charikar, Kai Li., Proceedings of the 33rd
        International Conference on Very Large Data Bases (VLDB). Vienna, Austria. September 2007) :
        @code
        struct LshIndexParams : public IndexParams
        {
            LshIndexParams(
                unsigned int table_number,
                unsigned int key_size,
                unsigned int multi_probe_level );
        };
        @endcode
        - **AutotunedIndexParams** When passing an object of this type the index created is
        automatically tuned to offer the best performance, by choosing the optimal index type
        (randomized kd-trees, hierarchical kmeans, linear) and parameters for the dataset provided. :
        @code
        struct AutotunedIndexParams : public IndexParams
        {
            AutotunedIndexParams(
                float target_precision = 0.9,
                float build_weight = 0.01,
                float memory_weight = 0,
                float sample_fraction = 0.1 );
        };
        @endcode
        - **SavedIndexParams** This object type is used for loading a previously saved index from the
        disk. :
        @code
        struct SavedIndexParams : public IndexParams
        {
            SavedIndexParams( String filename );
        };
        @endcode
         */
        GenericIndex(const Mat& features, const ::cvflann::IndexParams& params, Distance distance = Distance());

        ~GenericIndex();

        /** @brief Performs a K-nearest neighbor search for a given query point using the index.

        @param query The query point
        @param indices Vector that will contain the indices of the K-nearest neighbors found. It must have
        at least knn size.
        @param dists Vector that will contain the distances to the K-nearest neighbors found. It must have
        at least knn size.
        @param knn Number of nearest neighbors to search for.
        @param params SearchParams
         */
        void knnSearch(const std::vector<ElementType>& query, std::vector<int>& indices,
                       std::vector<DistanceType>& dists, int knn, const ::cvflann::SearchParams& params);
        void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const ::cvflann::SearchParams& params);

        int radiusSearch(const std::vector<ElementType>& query, std::vector<int>& indices,
                         std::vector<DistanceType>& dists, DistanceType radius, const ::cvflann::SearchParams& params);
        int radiusSearch(const Mat& query, Mat& indices, Mat& dists,
                         DistanceType radius, const ::cvflann::SearchParams& params);

        void save(String filename) { nnIndex->save(filename); }

        int veclen() const { return nnIndex->veclen(); }

        int size() const { return nnIndex->size(); }

        ::cvflann::IndexParams getParameters() { return nnIndex->getParameters(); }

        CV_DEPRECATED const ::cvflann::IndexParams* getIndexParameters() { return nnIndex->getIndexParameters(); }

private:
        ::cvflann::Index<Distance>* nnIndex;
};

//! @cond IGNORED

#define FLANN_DISTANCE_CHECK \
    if ( ::cvflann::flann_distance_type() != cvflann::FLANN_DIST_L2) { \
        printf("[WARNING] You are using cv::flann::Index (or cv::flann::GenericIndex) and have also changed "\
        "the distance using cvflann::set_distance_type. This is no longer working as expected "\
        "(cv::flann::Index always uses L2). You should create the index templated on the distance, "\
        "for example for L1 distance use: GenericIndex< L1<float> > \n"); \
    }


template <typename Distance>
GenericIndex<Distance>::GenericIndex(const Mat& dataset, const ::cvflann::IndexParams& params, Distance distance)
{
    CV_Assert(dataset.type() == CvType<ElementType>::type());
    CV_Assert(dataset.isContinuous());
    ::cvflann::Matrix<ElementType> m_dataset((ElementType*)dataset.ptr<ElementType>(0), dataset.rows, dataset.cols);

    nnIndex = new ::cvflann::Index<Distance>(m_dataset, params, distance);

    FLANN_DISTANCE_CHECK

    nnIndex->buildIndex();
}

template <typename Distance>
GenericIndex<Distance>::~GenericIndex()
{
    delete nnIndex;
}

template <typename Distance>
void GenericIndex<Distance>::knnSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, int knn, const ::cvflann::SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

    FLANN_DISTANCE_CHECK

    nnIndex->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
}


template <typename Distance>
void GenericIndex<Distance>::knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const ::cvflann::SearchParams& searchParams)
{
    CV_Assert(queries.type() == CvType<ElementType>::type());
    CV_Assert(queries.isContinuous());
    ::cvflann::Matrix<ElementType> m_queries((ElementType*)queries.ptr<ElementType>(0), queries.rows, queries.cols);

    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);

    CV_Assert(dists.type() == CvType<DistanceType>::type());
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);

    FLANN_DISTANCE_CHECK

    nnIndex->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
}

template <typename Distance>
int GenericIndex<Distance>::radiusSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
{
    ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
    ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
    ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

    FLANN_DISTANCE_CHECK

    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

template <typename Distance>
int GenericIndex<Distance>::radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
{
    CV_Assert(query.type() == CvType<ElementType>::type());
    CV_Assert(query.isContinuous());
    ::cvflann::Matrix<ElementType> m_query((ElementType*)query.ptr<ElementType>(0), query.rows, query.cols);

    CV_Assert(indices.type() == CV_32S);
    CV_Assert(indices.isContinuous());
    ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);

    CV_Assert(dists.type() == CvType<DistanceType>::type());
    CV_Assert(dists.isContinuous());
    ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);

    FLANN_DISTANCE_CHECK

    return nnIndex->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
}

//! @endcond

/**
 * @deprecated Use GenericIndex class instead
 */
template <typename T>
class Index_
{
public:
    typedef typename L2<T>::ElementType ElementType;
    typedef typename L2<T>::ResultType DistanceType;

    CV_DEPRECATED Index_(const Mat& dataset, const ::cvflann::IndexParams& params)
    {
        printf("[WARNING] The cv::flann::Index_<T> class is deperecated, use cv::flann::GenericIndex<Distance> instead\n");

        CV_Assert(dataset.type() == CvType<ElementType>::type());
        CV_Assert(dataset.isContinuous());
        ::cvflann::Matrix<ElementType> m_dataset((ElementType*)dataset.ptr<ElementType>(0), dataset.rows, dataset.cols);

        if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L2 ) {
            nnIndex_L1 = NULL;
            nnIndex_L2 = new ::cvflann::Index< L2<ElementType> >(m_dataset, params);
        }
        else if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L1 ) {
            nnIndex_L1 = new ::cvflann::Index< L1<ElementType> >(m_dataset, params);
            nnIndex_L2 = NULL;
        }
        else {
            printf("[ERROR] cv::flann::Index_<T> only provides backwards compatibility for the L1 and L2 distances. "
                   "For other distance types you must use cv::flann::GenericIndex<Distance>\n");
            CV_Assert(0);
        }
        if (nnIndex_L1) nnIndex_L1->buildIndex();
        if (nnIndex_L2) nnIndex_L2->buildIndex();
    }
    CV_DEPRECATED ~Index_()
    {
        if (nnIndex_L1) delete nnIndex_L1;
        if (nnIndex_L2) delete nnIndex_L2;
    }

    CV_DEPRECATED void knnSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, int knn, const ::cvflann::SearchParams& searchParams)
    {
        ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
        ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
        ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

        if (nnIndex_L1) nnIndex_L1->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
        if (nnIndex_L2) nnIndex_L2->knnSearch(m_query,m_indices,m_dists,knn,searchParams);
    }
    CV_DEPRECATED void knnSearch(const Mat& queries, Mat& indices, Mat& dists, int knn, const ::cvflann::SearchParams& searchParams)
    {
        CV_Assert(queries.type() == CvType<ElementType>::type());
        CV_Assert(queries.isContinuous());
        ::cvflann::Matrix<ElementType> m_queries((ElementType*)queries.ptr<ElementType>(0), queries.rows, queries.cols);

        CV_Assert(indices.type() == CV_32S);
        CV_Assert(indices.isContinuous());
        ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);

        CV_Assert(dists.type() == CvType<DistanceType>::type());
        CV_Assert(dists.isContinuous());
        ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);

        if (nnIndex_L1) nnIndex_L1->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
        if (nnIndex_L2) nnIndex_L2->knnSearch(m_queries,m_indices,m_dists,knn, searchParams);
    }

    CV_DEPRECATED int radiusSearch(const std::vector<ElementType>& query, std::vector<int>& indices, std::vector<DistanceType>& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
    {
        ::cvflann::Matrix<ElementType> m_query((ElementType*)&query[0], 1, query.size());
        ::cvflann::Matrix<int> m_indices(&indices[0], 1, indices.size());
        ::cvflann::Matrix<DistanceType> m_dists(&dists[0], 1, dists.size());

        if (nnIndex_L1) return nnIndex_L1->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
        if (nnIndex_L2) return nnIndex_L2->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
    }

    CV_DEPRECATED int radiusSearch(const Mat& query, Mat& indices, Mat& dists, DistanceType radius, const ::cvflann::SearchParams& searchParams)
    {
        CV_Assert(query.type() == CvType<ElementType>::type());
        CV_Assert(query.isContinuous());
        ::cvflann::Matrix<ElementType> m_query((ElementType*)query.ptr<ElementType>(0), query.rows, query.cols);

        CV_Assert(indices.type() == CV_32S);
        CV_Assert(indices.isContinuous());
        ::cvflann::Matrix<int> m_indices((int*)indices.ptr<int>(0), indices.rows, indices.cols);

        CV_Assert(dists.type() == CvType<DistanceType>::type());
        CV_Assert(dists.isContinuous());
        ::cvflann::Matrix<DistanceType> m_dists((DistanceType*)dists.ptr<DistanceType>(0), dists.rows, dists.cols);

        if (nnIndex_L1) return nnIndex_L1->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
        if (nnIndex_L2) return nnIndex_L2->radiusSearch(m_query,m_indices,m_dists,radius,searchParams);
    }

    CV_DEPRECATED void save(String filename)
    {
        if (nnIndex_L1) nnIndex_L1->save(filename);
        if (nnIndex_L2) nnIndex_L2->save(filename);
    }

    CV_DEPRECATED int veclen() const
    {
        if (nnIndex_L1) return nnIndex_L1->veclen();
        if (nnIndex_L2) return nnIndex_L2->veclen();
    }

    CV_DEPRECATED int size() const
    {
        if (nnIndex_L1) return nnIndex_L1->size();
        if (nnIndex_L2) return nnIndex_L2->size();
    }

    CV_DEPRECATED ::cvflann::IndexParams getParameters()
    {
        if (nnIndex_L1) return nnIndex_L1->getParameters();
        if (nnIndex_L2) return nnIndex_L2->getParameters();

    }

    CV_DEPRECATED const ::cvflann::IndexParams* getIndexParameters()
    {
        if (nnIndex_L1) return nnIndex_L1->getIndexParameters();
        if (nnIndex_L2) return nnIndex_L2->getIndexParameters();
    }

private:
    // providing backwards compatibility for L2 and L1 distances (most common)
    ::cvflann::Index< L2<ElementType> >* nnIndex_L2;
    ::cvflann::Index< L1<ElementType> >* nnIndex_L1;
};


/** @brief Clusters features using hierarchical k-means algorithm.

@param features The points to be clustered. The matrix must have elements of type
Distance::ElementType.
@param centers The centers of the clusters obtained. The matrix must have type
Distance::ResultType. The number of rows in this matrix represents the number of clusters desired,
however, because of the way the cut in the hierarchical tree is chosen, the number of clusters
computed will be the highest number of the form (branching-1)\*k+1 that's lower than the number of
clusters desired, where branching is the tree's branching factor (see description of the
KMeansIndexParams).
@param params Parameters used in the construction of the hierarchical k-means tree.
@param d Distance to be used for clustering.

The method clusters the given feature vectors by constructing a hierarchical k-means tree and
choosing a cut in the tree that minimizes the cluster's variance. It returns the number of clusters
found.
 */
template <typename Distance>
int hierarchicalClustering(const Mat& features, Mat& centers, const ::cvflann::KMeansIndexParams& params,
                           Distance d = Distance())
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    CV_Assert(features.type() == CvType<ElementType>::type());
    CV_Assert(features.isContinuous());
    ::cvflann::Matrix<ElementType> m_features((ElementType*)features.ptr<ElementType>(0), features.rows, features.cols);

    CV_Assert(centers.type() == CvType<DistanceType>::type());
    CV_Assert(centers.isContinuous());
    ::cvflann::Matrix<DistanceType> m_centers((DistanceType*)centers.ptr<DistanceType>(0), centers.rows, centers.cols);

    return ::cvflann::hierarchicalClustering<Distance>(m_features, m_centers, params, d);
}

/** @deprecated
*/
template <typename ELEM_TYPE, typename DIST_TYPE>
CV_DEPRECATED int hierarchicalClustering(const Mat& features, Mat& centers, const ::cvflann::KMeansIndexParams& params)
{
    printf("[WARNING] cv::flann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE> is deprecated, use "
        "cv::flann::hierarchicalClustering<Distance> instead\n");

    if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L2 ) {
        return hierarchicalClustering< L2<ELEM_TYPE> >(features, centers, params);
    }
    else if ( ::cvflann::flann_distance_type() == cvflann::FLANN_DIST_L1 ) {
        return hierarchicalClustering< L1<ELEM_TYPE> >(features, centers, params);
    }
    else {
        printf("[ERROR] cv::flann::hierarchicalClustering<ELEM_TYPE,DIST_TYPE> only provides backwards "
        "compatibility for the L1 and L2 distances. "
        "For other distance types you must use cv::flann::hierarchicalClustering<Distance>\n");
        CV_Assert(0);
    }
}

//! @} flann

} } // namespace cv::flann

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

#ifndef OPENCV_HIGHGUI_HPP
#define OPENCV_HIGHGUI_HPP

#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_IMGCODECS
#include "opencv2/imgcodecs.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOIO
#include "opencv2/videoio.hpp"
#endif

/**
@defgroup highgui High-level GUI

While OpenCV was designed for use in full-scale applications and can be used within functionally
rich UI frameworks (such as Qt\*, WinForms\*, or Cocoa\*) or without any UI at all, sometimes there
it is required to try functionality quickly and visualize the results. This is what the HighGUI
module has been designed for.

It provides easy interface to:

-   Create and manipulate windows that can display images and "remember" their content (no need to
    handle repaint events from OS).
-   Add trackbars to the windows, handle simple mouse events as well as keyboard commands.

@{
    @defgroup highgui_opengl OpenGL support
    @defgroup highgui_qt Qt New Functions

    ![image](pics/qtgui.png)

    This figure explains new functionality implemented with Qt\* GUI. The new GUI provides a statusbar,
    a toolbar, and a control panel. The control panel can have trackbars and buttonbars attached to it.
    If you cannot see the control panel, press Ctrl+P or right-click any Qt window and select **Display
    properties window**.

    -   To attach a trackbar, the window name parameter must be NULL.

    -   To attach a buttonbar, a button must be created. If the last bar attached to the control panel
        is a buttonbar, the new button is added to the right of the last button. If the last bar
        attached to the control panel is a trackbar, or the control panel is empty, a new buttonbar is
        created. Then, a new button is attached to it.

    See below the example used to generate the figure:
    @code
        int main(int argc, char *argv[])
        {

            int value = 50;
            int value2 = 0;


            namedWindow("main1",WINDOW_NORMAL);
            namedWindow("main2",WINDOW_AUTOSIZE | CV_GUI_NORMAL);
            createTrackbar( "track1", "main1", &value, 255,  NULL);

            String nameb1 = "button1";
            String nameb2 = "button2";

            createButton(nameb1,callbackButton,&nameb1,QT_CHECKBOX,1);
            createButton(nameb2,callbackButton,NULL,QT_CHECKBOX,0);
            createTrackbar( "track2", NULL, &value2, 255, NULL);
            createButton("button5",callbackButton1,NULL,QT_RADIOBOX,0);
            createButton("button6",callbackButton2,NULL,QT_RADIOBOX,1);

            setMouseCallback( "main2",on_mouse,NULL );

            Mat img1 = imread("files/flower.jpg");
            VideoCapture video;
            video.open("files/hockey.avi");

            Mat img2,img3;

            while( waitKey(33) != 27 )
            {
                img1.convertTo(img2,-1,1,value);
                video >> img3;

                imshow("main1",img2);
                imshow("main2",img3);
            }

            destroyAllWindows();

            return 0;
        }
    @endcode


    @defgroup highgui_winrt WinRT support

    This figure explains new functionality implemented with WinRT GUI. The new GUI provides an Image control,
    and a slider panel. Slider panel holds trackbars attached to it.

    Sliders are attached below the image control. Every new slider is added below the previous one.

    See below the example used to generate the figure:
    @code
        void sample_app::MainPage::ShowWindow()
        {
            static cv::String windowName("sample");
            cv::winrt_initContainer(this->cvContainer);
            cv::namedWindow(windowName); // not required

            cv::Mat image = cv::imread("Assets/sample.jpg");
            cv::Mat converted = cv::Mat(image.rows, image.cols, CV_8UC4);
            cv::cvtColor(image, converted, COLOR_BGR2BGRA);
            cv::imshow(windowName, converted); // this will create window if it hasn't been created before

            int state = 42;
            cv::TrackbarCallback callback = [](int pos, void* userdata)
            {
                if (pos == 0) {
                    cv::destroyWindow(windowName);
                }
            };
            cv::TrackbarCallback callbackTwin = [](int pos, void* userdata)
            {
                if (pos >= 70) {
                    cv::destroyAllWindows();
                }
            };
            cv::createTrackbar("Sample trackbar", windowName, &state, 100, callback);
            cv::createTrackbar("Twin brother", windowName, &state, 100, callbackTwin);
        }
    @endcode

    @defgroup highgui_c C API
@}
*/

///////////////////////// graphical user interface //////////////////////////
namespace cv
{

//! @addtogroup highgui
//! @{

//! Flags for cv::namedWindow
enum WindowFlags {
       WINDOW_NORMAL     = 0x00000000, //!< the user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size.
       WINDOW_AUTOSIZE   = 0x00000001, //!< the user cannot resize the window, the size is constrainted by the image displayed.
       WINDOW_OPENGL     = 0x00001000, //!< window with opengl support.

       WINDOW_FULLSCREEN = 1,          //!< change the window to fullscreen.
       WINDOW_FREERATIO  = 0x00000100, //!< the image expends as much as it can (no ratio constraint).
       WINDOW_KEEPRATIO  = 0x00000000, //!< the ratio of the image is respected.
       WINDOW_GUI_EXPANDED=0x00000000, //!< status bar and tool bar
       WINDOW_GUI_NORMAL = 0x00000010, //!< old fashious way
    };

//! Flags for cv::setWindowProperty / cv::getWindowProperty
enum WindowPropertyFlags {
       WND_PROP_FULLSCREEN   = 0, //!< fullscreen property    (can be WINDOW_NORMAL or WINDOW_FULLSCREEN).
       WND_PROP_AUTOSIZE     = 1, //!< autosize property      (can be WINDOW_NORMAL or WINDOW_AUTOSIZE).
       WND_PROP_ASPECT_RATIO = 2, //!< window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO).
       WND_PROP_OPENGL       = 3, //!< opengl support.
       WND_PROP_VISIBLE      = 4  //!< checks whether the window exists and is visible
     };

//! Mouse Events see cv::MouseCallback
enum MouseEventTypes {
       EVENT_MOUSEMOVE      = 0, //!< indicates that the mouse pointer has moved over the window.
       EVENT_LBUTTONDOWN    = 1, //!< indicates that the left mouse button is pressed.
       EVENT_RBUTTONDOWN    = 2, //!< indicates that the right mouse button is pressed.
       EVENT_MBUTTONDOWN    = 3, //!< indicates that the middle mouse button is pressed.
       EVENT_LBUTTONUP      = 4, //!< indicates that left mouse button is released.
       EVENT_RBUTTONUP      = 5, //!< indicates that right mouse button is released.
       EVENT_MBUTTONUP      = 6, //!< indicates that middle mouse button is released.
       EVENT_LBUTTONDBLCLK  = 7, //!< indicates that left mouse button is double clicked.
       EVENT_RBUTTONDBLCLK  = 8, //!< indicates that right mouse button is double clicked.
       EVENT_MBUTTONDBLCLK  = 9, //!< indicates that middle mouse button is double clicked.
       EVENT_MOUSEWHEEL     = 10,//!< positive and negative values mean forward and backward scrolling, respectively.
       EVENT_MOUSEHWHEEL    = 11 //!< positive and negative values mean right and left scrolling, respectively.
     };

//! Mouse Event Flags see cv::MouseCallback
enum MouseEventFlags {
       EVENT_FLAG_LBUTTON   = 1, //!< indicates that the left mouse button is down.
       EVENT_FLAG_RBUTTON   = 2, //!< indicates that the right mouse button is down.
       EVENT_FLAG_MBUTTON   = 4, //!< indicates that the middle mouse button is down.
       EVENT_FLAG_CTRLKEY   = 8, //!< indicates that CTRL Key is pressed.
       EVENT_FLAG_SHIFTKEY  = 16,//!< indicates that SHIFT Key is pressed.
       EVENT_FLAG_ALTKEY    = 32 //!< indicates that ALT Key is pressed.
     };

//! Qt font weight
enum QtFontWeights {
        QT_FONT_LIGHT           = 25, //!< Weight of 25
        QT_FONT_NORMAL          = 50, //!< Weight of 50
        QT_FONT_DEMIBOLD        = 63, //!< Weight of 63
        QT_FONT_BOLD            = 75, //!< Weight of 75
        QT_FONT_BLACK           = 87  //!< Weight of 87
     };

//! Qt font style
enum QtFontStyles {
        QT_STYLE_NORMAL         = 0, //!< Normal font.
        QT_STYLE_ITALIC         = 1, //!< Italic font.
        QT_STYLE_OBLIQUE        = 2  //!< Oblique font.
     };

//! Qt "button" type
enum QtButtonTypes {
       QT_PUSH_BUTTON   = 0,    //!< Push button.
       QT_CHECKBOX      = 1,    //!< Checkbox button.
       QT_RADIOBOX      = 2,    //!< Radiobox button.
       QT_NEW_BUTTONBAR = 1024  //!< Button should create a new buttonbar
     };

/** @brief Callback function for mouse events. see cv::setMouseCallback
@param event one of the cv::MouseEventTypes constants.
@param x The x-coordinate of the mouse event.
@param y The y-coordinate of the mouse event.
@param flags one of the cv::MouseEventFlags constants.
@param userdata The optional parameter.
 */
typedef void (*MouseCallback)(int event, int x, int y, int flags, void* userdata);

/** @brief Callback function for Trackbar see cv::createTrackbar
@param pos current position of the specified trackbar.
@param userdata The optional parameter.
 */
typedef void (*TrackbarCallback)(int pos, void* userdata);

/** @brief Callback function defined to be called every frame. See cv::setOpenGlDrawCallback
@param userdata The optional parameter.
 */
typedef void (*OpenGlDrawCallback)(void* userdata);

/** @brief Callback function for a button created by cv::createButton
@param state current state of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button.
@param userdata The optional parameter.
 */
typedef void (*ButtonCallback)(int state, void* userdata);

/** @brief Creates a window.

The function namedWindow creates a window that can be used as a placeholder for images and
trackbars. Created windows are referred to by their names.

If a window with the same name already exists, the function does nothing.

You can call cv::destroyWindow or cv::destroyAllWindows to close the window and de-allocate any associated
memory usage. For a simple program, you do not really have to call these functions because all the
resources and windows of the application are closed automatically by the operating system upon exit.

@note

Qt backend supports additional flags:
 -   **WINDOW_NORMAL or WINDOW_AUTOSIZE:** WINDOW_NORMAL enables you to resize the
     window, whereas WINDOW_AUTOSIZE adjusts automatically the window size to fit the
     displayed image (see imshow ), and you cannot change the window size manually.
 -   **WINDOW_FREERATIO or WINDOW_KEEPRATIO:** WINDOW_FREERATIO adjusts the image
     with no respect to its ratio, whereas WINDOW_KEEPRATIO keeps the image ratio.
 -   **WINDOW_GUI_NORMAL or WINDOW_GUI_EXPANDED:** WINDOW_GUI_NORMAL is the old way to draw the window
     without statusbar and toolbar, whereas WINDOW_GUI_EXPANDED is a new enhanced GUI.
By default, flags == WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED

@param winname Name of the window in the window caption that may be used as a window identifier.
@param flags Flags of the window. The supported flags are: (cv::WindowFlags)
 */
CV_EXPORTS_W void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);

/** @brief Destroys the specified window.

The function destroyWindow destroys the window with the given name.

@param winname Name of the window to be destroyed.
 */
CV_EXPORTS_W void destroyWindow(const String& winname);

/** @brief Destroys all of the HighGUI windows.

The function destroyAllWindows destroys all of the opened HighGUI windows.
 */
CV_EXPORTS_W void destroyAllWindows();

CV_EXPORTS_W int startWindowThread();

/** @brief Similar to #waitKey, but returns full key code.

@note

Key code is implementation specific and depends on used backend: QT/GTK/Win32/etc

*/
CV_EXPORTS_W int waitKeyEx(int delay = 0);

/** @brief Waits for a pressed key.

The function waitKey waits for a key event infinitely (when \f$\texttt{delay}\leq 0\f$ ) or for delay
milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the
function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is
running on your computer at that time. It returns the code of the pressed key or -1 if no key was
pressed before the specified time had elapsed.

@note

This function is the only method in HighGUI that can fetch and handle events, so it needs to be
called periodically for normal event processing unless HighGUI is used within an environment that
takes care of event processing.

@note

The function only works if there is at least one HighGUI window created and the window is active.
If there are several HighGUI windows, any of them can be active.

@param delay Delay in milliseconds. 0 is the special value that means "forever".
 */
CV_EXPORTS_W int waitKey(int delay = 0);

/** @brief Displays an image in the specified window.

The function imshow displays an image in the specified window. If the window was created with the
cv::WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution.
Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:

-   If the image is 8-bit unsigned, it is displayed as is.
-   If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the
    value range [0,255\*256] is mapped to [0,255].
-   If the image is 32-bit or 64-bit floating-point, the pixel values are multiplied by 255. That is, the
    value range [0,1] is mapped to [0,255].

If window was created with OpenGL support, cv::imshow also support ogl::Buffer , ogl::Texture2D and
cuda::GpuMat as input.

If the window was not created before this function, it is assumed creating a window with cv::WINDOW_AUTOSIZE.

If you need to show an image that is bigger than the screen resolution, you will need to call namedWindow("", WINDOW_NORMAL) before the imshow.

@note This function should be followed by cv::waitKey function which displays the image for specified
milliseconds. Otherwise, it won't display the image. For example, **waitKey(0)** will display the window
infinitely until any keypress (it is suitable for image display). **waitKey(25)** will display a frame
for 25 ms, after which display will be automatically closed. (If you put it in a loop to read
videos, it will display the video frame-by-frame)

@note

[__Windows Backend Only__] Pressing Ctrl+C will copy the image to the clipboard.

[__Windows Backend Only__] Pressing Ctrl+S will show a dialog to save the image.

@param winname Name of the window.
@param mat Image to be shown.
 */
CV_EXPORTS_W void imshow(const String& winname, InputArray mat);

/** @brief Resizes window to the specified size

@note

-   The specified window size is for the image area. Toolbars are not counted.
-   Only windows created without cv::WINDOW_AUTOSIZE flag can be resized.

@param winname Window name.
@param width The new window width.
@param height The new window height.
 */
CV_EXPORTS_W void resizeWindow(const String& winname, int width, int height);

/** @brief Moves window to the specified position

@param winname Name of the window.
@param x The new x-coordinate of the window.
@param y The new y-coordinate of the window.
 */
CV_EXPORTS_W void moveWindow(const String& winname, int x, int y);

/** @brief Changes parameters of a window dynamically.

The function setWindowProperty enables changing properties of a window.

@param winname Name of the window.
@param prop_id Window property to edit. The supported operation flags are: (cv::WindowPropertyFlags)
@param prop_value New value of the window property. The supported flags are: (cv::WindowFlags)
 */
CV_EXPORTS_W void setWindowProperty(const String& winname, int prop_id, double prop_value);

/** @brief Updates window title
@param winname Name of the window.
@param title New title.
*/
CV_EXPORTS_W void setWindowTitle(const String& winname, const String& title);

/** @brief Provides parameters of a window.

The function getWindowProperty returns properties of a window.

@param winname Name of the window.
@param prop_id Window property to retrieve. The following operation flags are available: (cv::WindowPropertyFlags)

@sa setWindowProperty
 */
CV_EXPORTS_W double getWindowProperty(const String& winname, int prop_id);

/** @brief Sets mouse handler for the specified window

@param winname Name of the window.
@param onMouse Mouse callback. See OpenCV samples, such as
<https://github.com/opencv/opencv/tree/master/samples/cpp/ffilldemo.cpp>, on how to specify and
use the callback.
@param userdata The optional parameter passed to the callback.
 */
CV_EXPORTS void setMouseCallback(const String& winname, MouseCallback onMouse, void* userdata = 0);

/** @brief Gets the mouse-wheel motion delta, when handling mouse-wheel events cv::EVENT_MOUSEWHEEL and
cv::EVENT_MOUSEHWHEEL.

For regular mice with a scroll-wheel, delta will be a multiple of 120. The value 120 corresponds to
a one notch rotation of the wheel or the threshold for action to be taken and one such action should
occur for each delta. Some high-precision mice with higher-resolution freely-rotating wheels may
generate smaller values.

For cv::EVENT_MOUSEWHEEL positive and negative values mean forward and backward scrolling,
respectively. For cv::EVENT_MOUSEHWHEEL, where available, positive and negative values mean right and
left scrolling, respectively.

With the C API, the macro CV_GET_WHEEL_DELTA(flags) can be used alternatively.

@note

Mouse-wheel events are currently supported only on Windows.

@param flags The mouse callback flags parameter.
 */
CV_EXPORTS int getMouseWheelDelta(int flags);

/** @brief Selects ROI on the given image.
Function creates a window and allows user to select a ROI using mouse.
Controls: use `space` or `enter` to finish selection, use key `c` to cancel selection (function will return the zero cv::Rect).

@param windowName name of the window where selection process will be shown.
@param img image to select a ROI.
@param showCrosshair if true crosshair of selection rectangle will be shown.
@param fromCenter if true center of selection will match initial mouse position. In opposite case a corner of
selection rectangle will correspont to the initial mouse position.
@return selected ROI or empty rect if selection canceled.

@note The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).
After finish of work an empty callback will be set for the used window.
 */
CV_EXPORTS_W Rect selectROI(const String& windowName, InputArray img, bool showCrosshair = true, bool fromCenter = false);

/** @overload
 */
CV_EXPORTS_W Rect selectROI(InputArray img, bool showCrosshair = true, bool fromCenter = false);

/** @brief Selects ROIs on the given image.
Function creates a window and allows user to select a ROIs using mouse.
Controls: use `space` or `enter` to finish current selection and start a new one,
use `esc` to terminate multiple ROI selection process.

@param windowName name of the window where selection process will be shown.
@param img image to select a ROI.
@param boundingBoxes selected ROIs.
@param showCrosshair if true crosshair of selection rectangle will be shown.
@param fromCenter if true center of selection will match initial mouse position. In opposite case a corner of
selection rectangle will correspont to the initial mouse position.

@note The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).
After finish of work an empty callback will be set for the used window.
 */
CV_EXPORTS_W void selectROIs(const String& windowName, InputArray img,
                             CV_OUT std::vector<Rect>& boundingBoxes, bool showCrosshair = true, bool fromCenter = false);

/** @brief Creates a trackbar and attaches it to the specified window.

The function createTrackbar creates a trackbar (a slider or range control) with the specified name
and range, assigns a variable value to be a position synchronized with the trackbar and specifies
the callback function onChange to be called on the trackbar position change. The created trackbar is
displayed in the specified window winname.

@note

[__Qt Backend Only__] winname can be empty (or NULL) if the trackbar should be attached to the
control panel.

Clicking the label of each trackbar enables editing the trackbar values manually.

@param trackbarname Name of the created trackbar.
@param winname Name of the window that will be used as a parent of the created trackbar.
@param value Optional pointer to an integer variable whose value reflects the position of the
slider. Upon creation, the slider position is defined by this variable.
@param count Maximal position of the slider. The minimal position is always 0.
@param onChange Pointer to the function to be called every time the slider changes position. This
function should be prototyped as void Foo(int,void\*); , where the first parameter is the trackbar
position and the second parameter is the user data (see the next parameter). If the callback is
the NULL pointer, no callbacks are called, but only value is updated.
@param userdata User data that is passed as is to the callback. It can be used to handle trackbar
events without using global variables.
 */
CV_EXPORTS int createTrackbar(const String& trackbarname, const String& winname,
                              int* value, int count,
                              TrackbarCallback onChange = 0,
                              void* userdata = 0);

/** @brief Returns the trackbar position.

The function returns the current position of the specified trackbar.

@note

[__Qt Backend Only__] winname can be empty (or NULL) if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of the trackbar.
 */
CV_EXPORTS_W int getTrackbarPos(const String& trackbarname, const String& winname);

/** @brief Sets the trackbar position.

The function sets the position of the specified trackbar in the specified window.

@note

[__Qt Backend Only__] winname can be empty (or NULL) if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param pos New position.
 */
CV_EXPORTS_W void setTrackbarPos(const String& trackbarname, const String& winname, int pos);

/** @brief Sets the trackbar maximum position.

The function sets the maximum position of the specified trackbar in the specified window.

@note

[__Qt Backend Only__] winname can be empty (or NULL) if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param maxval New maximum position.
 */
CV_EXPORTS_W void setTrackbarMax(const String& trackbarname, const String& winname, int maxval);

/** @brief Sets the trackbar minimum position.

The function sets the minimum position of the specified trackbar in the specified window.

@note

[__Qt Backend Only__] winname can be empty (or NULL) if the trackbar is attached to the control
panel.

@param trackbarname Name of the trackbar.
@param winname Name of the window that is the parent of trackbar.
@param minval New maximum position.
 */
CV_EXPORTS_W void setTrackbarMin(const String& trackbarname, const String& winname, int minval);

//! @addtogroup highgui_opengl OpenGL support
//! @{

/** @brief Displays OpenGL 2D texture in the specified window.

@param winname Name of the window.
@param tex OpenGL 2D texture data.
 */
CV_EXPORTS void imshow(const String& winname, const ogl::Texture2D& tex);

/** @brief Sets a callback function to be called to draw on top of displayed image.

The function setOpenGlDrawCallback can be used to draw 3D data on the window. See the example of
callback function below:
@code
    void on_opengl(void* param)
    {
        glLoadIdentity();

        glTranslated(0.0, 0.0, -1.0);

        glRotatef( 55, 1, 0, 0 );
        glRotatef( 45, 0, 1, 0 );
        glRotatef( 0, 0, 0, 1 );

        static const int coords[6][4][3] = {
            { { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
            { { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
            { { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
            { { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
            { { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
            { { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
        };

        for (int i = 0; i < 6; ++i) {
                    glColor3ub( i*20, 100+i*10, i*42 );
                    glBegin(GL_QUADS);
                    for (int j = 0; j < 4; ++j) {
                            glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
                    }
                    glEnd();
        }
    }
@endcode

@param winname Name of the window.
@param onOpenGlDraw Pointer to the function to be called every frame. This function should be
prototyped as void Foo(void\*) .
@param userdata Pointer passed to the callback function.(__Optional__)
 */
CV_EXPORTS void setOpenGlDrawCallback(const String& winname, OpenGlDrawCallback onOpenGlDraw, void* userdata = 0);

/** @brief Sets the specified window as current OpenGL context.

@param winname Name of the window.
 */
CV_EXPORTS void setOpenGlContext(const String& winname);

/** @brief Force window to redraw its context and call draw callback ( See cv::setOpenGlDrawCallback ).

@param winname Name of the window.
 */
CV_EXPORTS void updateWindow(const String& winname);

//! @} highgui_opengl

//! @addtogroup highgui_qt
//! @{

/** @brief QtFont available only for Qt. See cv::fontQt
 */
struct QtFont
{
    const char* nameFont;  //!< Name of the font
    Scalar      color;     //!< Color of the font. Scalar(blue_component, green_component, red_component[, alpha_component])
    int         font_face; //!< See cv::QtFontStyles
    const int*  ascii;     //!< font data and metrics
    const int*  greek;
    const int*  cyrillic;
    float       hscale, vscale;
    float       shear;     //!< slope coefficient: 0 - normal, >0 - italic
    int         thickness; //!< See cv::QtFontWeights
    float       dx;        //!< horizontal interval between letters
    int         line_type; //!< PointSize
};

/** @brief Creates the font to draw a text on an image.

The function fontQt creates a cv::QtFont object. This cv::QtFont is not compatible with putText .

A basic usage of this function is the following: :
@code
    QtFont font = fontQt("Times");
    addText( img1, "Hello World !", Point(50,50), font);
@endcode

@param nameFont Name of the font. The name should match the name of a system font (such as
*Times*). If the font is not found, a default one is used.
@param pointSize Size of the font. If not specified, equal zero or negative, the point size of the
font is set to a system-dependent default value. Generally, this is 12 points.
@param color Color of the font in BGRA where A = 255 is fully transparent. Use the macro CV_RGB
for simplicity.
@param weight Font weight. Available operation flags are : cv::QtFontWeights You can also specify a positive integer for better control.
@param style Font style. Available operation flags are : cv::QtFontStyles
@param spacing Spacing between characters. It can be negative or positive.
 */
CV_EXPORTS QtFont fontQt(const String& nameFont, int pointSize = -1,
                         Scalar color = Scalar::all(0), int weight = QT_FONT_NORMAL,
                         int style = QT_STYLE_NORMAL, int spacing = 0);

/** @brief Draws a text on the image.

The function addText draws *text* on the image *img* using a specific font *font* (see example cv::fontQt
)

@param img 8-bit 3-channel image where the text should be drawn.
@param text Text to write on an image.
@param org Point(x,y) where the text should start on an image.
@param font Font to use to draw a text.
 */
CV_EXPORTS void addText( const Mat& img, const String& text, Point org, const QtFont& font);

/** @brief Draws a text on the image.

@param img 8-bit 3-channel image where the text should be drawn.
@param text Text to write on an image.
@param org Point(x,y) where the text should start on an image.
@param nameFont Name of the font. The name should match the name of a system font (such as
*Times*). If the font is not found, a default one is used.
@param pointSize Size of the font. If not specified, equal zero or negative, the point size of the
font is set to a system-dependent default value. Generally, this is 12 points.
@param color Color of the font in BGRA where A = 255 is fully transparent.
@param weight Font weight. Available operation flags are : cv::QtFontWeights You can also specify a positive integer for better control.
@param style Font style. Available operation flags are : cv::QtFontStyles
@param spacing Spacing between characters. It can be negative or positive.
 */
CV_EXPORTS_W void addText(const Mat& img, const String& text, Point org, const String& nameFont, int pointSize = -1, Scalar color = Scalar::all(0),
        int weight = QT_FONT_NORMAL, int style = QT_STYLE_NORMAL, int spacing = 0);

/** @brief Displays a text on a window image as an overlay for a specified duration.

The function displayOverlay displays useful information/tips on top of the window for a certain
amount of time *delayms*. The function does not modify the image, displayed in the window, that is,
after the specified delay the original content of the window is restored.

@param winname Name of the window.
@param text Overlay text to write on a window image.
@param delayms The period (in milliseconds), during which the overlay text is displayed. If this
function is called before the previous overlay text timed out, the timer is restarted and the text
is updated. If this value is zero, the text never disappears.
 */
CV_EXPORTS_W void displayOverlay(const String& winname, const String& text, int delayms = 0);

/** @brief Displays a text on the window statusbar during the specified period of time.

The function displayStatusBar displays useful information/tips on top of the window for a certain
amount of time *delayms* . This information is displayed on the window statusbar (the window must be
created with the CV_GUI_EXPANDED flags).

@param winname Name of the window.
@param text Text to write on the window statusbar.
@param delayms Duration (in milliseconds) to display the text. If this function is called before
the previous text timed out, the timer is restarted and the text is updated. If this value is
zero, the text never disappears.
 */
CV_EXPORTS_W void displayStatusBar(const String& winname, const String& text, int delayms = 0);

/** @brief Saves parameters of the specified window.

The function saveWindowParameters saves size, location, flags, trackbars value, zoom and panning
location of the window windowName.

@param windowName Name of the window.
 */
CV_EXPORTS void saveWindowParameters(const String& windowName);

/** @brief Loads parameters of the specified window.

The function loadWindowParameters loads size, location, flags, trackbars value, zoom and panning
location of the window windowName.

@param windowName Name of the window.
 */
CV_EXPORTS void loadWindowParameters(const String& windowName);

CV_EXPORTS  int startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[]);

CV_EXPORTS  void stopLoop();

/** @brief Attaches a button to the control panel.

The function createButton attaches a button to the control panel. Each button is added to a
buttonbar to the right of the last button. A new buttonbar is created if nothing was attached to the
control panel before, or if the last element attached to the control panel was a trackbar or if the
QT_NEW_BUTTONBAR flag is added to the type.

See below various examples of the cv::createButton function call: :
@code
    createButton(NULL,callbackButton);//create a push button "button 0", that will call callbackButton.
    createButton("button2",callbackButton,NULL,QT_CHECKBOX,0);
    createButton("button3",callbackButton,&value);
    createButton("button5",callbackButton1,NULL,QT_RADIOBOX);
    createButton("button6",callbackButton2,NULL,QT_PUSH_BUTTON,1);
    createButton("button6",callbackButton2,NULL,QT_PUSH_BUTTON|QT_NEW_BUTTONBAR);// create a push button in a new row
@endcode

@param  bar_name Name of the button.
@param on_change Pointer to the function to be called every time the button changes its state.
This function should be prototyped as void Foo(int state,\*void); . *state* is the current state
of the button. It could be -1 for a push button, 0 or 1 for a check/radio box button.
@param userdata Pointer passed to the callback function.
@param type Optional type of the button. Available types are: (cv::QtButtonTypes)
@param initial_button_state Default state of the button. Use for checkbox and radiobox. Its
value could be 0 or 1. (__Optional__)
*/
CV_EXPORTS int createButton( const String& bar_name, ButtonCallback on_change,
                             void* userdata = 0, int type = QT_PUSH_BUTTON,
                             bool initial_button_state = false);

//! @} highgui_qt

//! @} highgui

} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/highgui/highgui_c.h"
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

#ifndef OPENCV_IMGCODECS_HPP
#define OPENCV_IMGCODECS_HPP

#include "opencv2/core.hpp"

/**
  @defgroup imgcodecs Image file reading and writing
  @{
    @defgroup imgcodecs_c C API
    @defgroup imgcodecs_ios iOS glue
  @}
*/

//////////////////////////////// image codec ////////////////////////////////
namespace cv
{

//! @addtogroup imgcodecs
//! @{

//! Imread flags
enum ImreadModes {
       IMREAD_UNCHANGED            = -1, //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
       IMREAD_GRAYSCALE            = 0,  //!< If set, always convert image to the single channel grayscale image.
       IMREAD_COLOR                = 1,  //!< If set, always convert image to the 3 channel BGR color image.
       IMREAD_ANYDEPTH             = 2,  //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
       IMREAD_ANYCOLOR             = 4,  //!< If set, the image is read in any possible color format.
       IMREAD_LOAD_GDAL            = 8,  //!< If set, use the gdal driver for loading the image.
       IMREAD_REDUCED_GRAYSCALE_2  = 16, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
       IMREAD_REDUCED_COLOR_2      = 17, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
       IMREAD_REDUCED_GRAYSCALE_4  = 32, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
       IMREAD_REDUCED_COLOR_4      = 33, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
       IMREAD_REDUCED_GRAYSCALE_8  = 64, //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
       IMREAD_REDUCED_COLOR_8      = 65, //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
       IMREAD_IGNORE_ORIENTATION   = 128 //!< If set, do not rotate the image according to EXIF's orientation flag.
     };

//! Imwrite flags
enum ImwriteFlags {
       IMWRITE_JPEG_QUALITY        = 1,  //!< For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
       IMWRITE_JPEG_PROGRESSIVE    = 2,  //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_OPTIMIZE       = 3,  //!< Enable JPEG features, 0 or 1, default is False.
       IMWRITE_JPEG_RST_INTERVAL   = 4,  //!< JPEG restart interval, 0 - 65535, default is 0 - no restart.
       IMWRITE_JPEG_LUMA_QUALITY   = 5,  //!< Separate luma quality level, 0 - 100, default is 0 - don't use.
       IMWRITE_JPEG_CHROMA_QUALITY = 6,  //!< Separate chroma quality level, 0 - 100, default is 0 - don't use.
       IMWRITE_PNG_COMPRESSION     = 16, //!< For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).
       IMWRITE_PNG_STRATEGY        = 17, //!< One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
       IMWRITE_PNG_BILEVEL         = 18, //!< Binary level PNG, 0 or 1, default is 0.
       IMWRITE_PXM_BINARY          = 32, //!< For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
       IMWRITE_WEBP_QUALITY        = 64, //!< For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default (without any parameter) and for quality above 100 the lossless compression is used.
       IMWRITE_PAM_TUPLETYPE       = 128,//!< For PAM, sets the TUPLETYPE field to the corresponding string value that is defined for the format
     };

//! Imwrite PNG specific flags used to tune the compression algorithm.
/** These flags will be modify the way of PNG image compression and will be passed to the underlying zlib processing stage.

-   The effect of IMWRITE_PNG_STRATEGY_FILTERED is to force more Huffman coding and less string matching; it is somewhat intermediate between IMWRITE_PNG_STRATEGY_DEFAULT and IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY.
-   IMWRITE_PNG_STRATEGY_RLE is designed to be almost as fast as IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, but give better compression for PNG image data.
-   The strategy parameter only affects the compression ratio but not the correctness of the compressed output even if it is not set appropriately.
-   IMWRITE_PNG_STRATEGY_FIXED prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.
*/
enum ImwritePNGFlags {
       IMWRITE_PNG_STRATEGY_DEFAULT      = 0, //!< Use this value for normal data.
       IMWRITE_PNG_STRATEGY_FILTERED     = 1, //!< Use this value for data produced by a filter (or predictor).Filtered data consists mostly of small values with a somewhat random distribution. In this case, the compression algorithm is tuned to compress them better.
       IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2, //!< Use this value to force Huffman encoding only (no string match).
       IMWRITE_PNG_STRATEGY_RLE          = 3, //!< Use this value to limit match distances to one (run-length encoding).
       IMWRITE_PNG_STRATEGY_FIXED        = 4  //!< Using this value prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.
     };

//! Imwrite PAM specific tupletype flags used to define the 'TUPETYPE' field of a PAM file.
enum ImwritePAMFlags {
       IMWRITE_PAM_FORMAT_NULL = 0,
       IMWRITE_PAM_FORMAT_BLACKANDWHITE = 1,
       IMWRITE_PAM_FORMAT_GRAYSCALE = 2,
       IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3,
       IMWRITE_PAM_FORMAT_RGB = 4,
       IMWRITE_PAM_FORMAT_RGB_ALPHA = 5,
     };

/** @brief Loads an image from a file.

@anchor imread

The function imread loads an image from the specified file and returns it. If the image cannot be
read (because of missing file, improper permissions, unsupported or invalid format), the function
returns an empty matrix ( Mat::data==NULL ).

Currently, the following file formats are supported:

-   Windows bitmaps - \*.bmp, \*.dib (always supported)
-   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Notes* section)
-   JPEG 2000 files - \*.jp2 (see the *Notes* section)
-   Portable Network Graphics - \*.png (see the *Notes* section)
-   WebP - \*.webp (see the *Notes* section)
-   Portable image format - \*.pbm, \*.pgm, \*.ppm \*.pxm, \*.pnm (always supported)
-   Sun rasters - \*.sr, \*.ras (always supported)
-   TIFF files - \*.tiff, \*.tif (see the *Notes* section)
-   OpenEXR Image files - \*.exr (see the *Notes* section)
-   Radiance HDR - \*.hdr, \*.pic (always supported)
-   Raster and Vector geospatial data supported by Gdal (see the *Notes* section)

@note

-   The function determines the type of an image by the content, not by the file extension.
-   In the case of color images, the decoded images will have the channels stored in **B G R** order.
-   On Microsoft Windows\* OS and MacOSX\*, the codecs shipped with an OpenCV image (libjpeg,
    libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs,
    and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware
    that currently these native image loaders give images with different pixel values because of
    the color management embedded into MacOSX.
-   On Linux\*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for
    codecs supplied with an OS image. Install the relevant packages (do not forget the development
    files, for example, "libjpeg-dev", in Debian\* and Ubuntu\*) to get the codec support or turn
    on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.
-   In the case you set *WITH_GDAL* flag to true in CMake and @ref IMREAD_LOAD_GDAL to load the image,
    then [GDAL](http://www.gdal.org) driver will be used in order to decode the image by supporting
    the following formats: [Raster](http://www.gdal.org/formats_list.html),
    [Vector](http://www.gdal.org/ogr_formats.html).
-   If EXIF information are embedded in the image file, the EXIF orientation will be taken into account
    and thus the image will be rotated accordingly except if the flag @ref IMREAD_IGNORE_ORIENTATION is passed.
@param filename Name of file to be loaded.
@param flags Flag that can take values of cv::ImreadModes
*/
CV_EXPORTS_W Mat imread( const String& filename, int flags = IMREAD_COLOR );

/** @brief Loads a multi-page image from a file.

The function imreadmulti loads a multi-page image from the specified file into a vector of Mat objects.
@param filename Name of file to be loaded.
@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.
@param mats A vector of Mat objects holding each page, if more than one.
@sa cv::imread
*/
CV_EXPORTS_W bool imreadmulti(const String& filename, std::vector<Mat>& mats, int flags = IMREAD_ANYCOLOR);

/** @brief Saves an image to a specified file.

The function imwrite saves the image to the specified file. The image format is chosen based on the
filename extension (see cv::imread for the list of extensions). Only 8-bit (or 16-bit unsigned (CV_16U)
in case of PNG, JPEG 2000, and TIFF) single-channel or 3-channel (with 'BGR' channel order) images
can be saved using this function. If the format, depth or channel order is different, use
Mat::convertTo , and cv::cvtColor to convert it before saving. Or, use the universal FileStorage I/O
functions to save the image to XML or YAML format.

It is possible to store PNG images with an alpha channel using this function. To do this, create
8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels
should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535.

The sample below shows how to create such a BGRA image and store to PNG file. It also demonstrates how to set custom
compression parameters :
@code
    #include <opencv2/opencv.hpp>

    using namespace cv;
    using namespace std;

    void createAlphaMat(Mat &mat)
    {
        CV_Assert(mat.channels() == 4);
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                Vec4b& bgra = mat.at<Vec4b>(i, j);
                bgra[0] = UCHAR_MAX; // Blue
                bgra[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX); // Green
                bgra[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX); // Red
                bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2])); // Alpha
            }
        }
    }

    int main(int argv, char **argc)
    {
        // Create mat with alpha channel
        Mat mat(480, 640, CV_8UC4);
        createAlphaMat(mat);

        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        try {
            imwrite("alpha.png", mat, compression_params);
        }
        catch (cv::Exception& ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            return 1;
        }

        fprintf(stdout, "Saved PNG file with alpha data.\n");
        return 0;
    }
@endcode
@param filename Name of the file.
@param img Image to be saved.
@param params Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see cv::ImwriteFlags
*/
CV_EXPORTS_W bool imwrite( const String& filename, InputArray img,
              const std::vector<int>& params = std::vector<int>());

/** @brief Reads an image from a buffer in memory.

The function imdecode reads an image from the specified buffer in the memory. If the buffer is too short or
contains invalid data, the function returns an empty matrix ( Mat::data==NULL ).

See cv::imread for the list of supported formats and flags description.

@note In the case of color images, the decoded images will have the channels stored in **B G R** order.
@param buf Input array or vector of bytes.
@param flags The same flags as in cv::imread, see cv::ImreadModes.
*/
CV_EXPORTS_W Mat imdecode( InputArray buf, int flags );

/** @overload
@param buf
@param flags
@param dst The optional output placeholder for the decoded matrix. It can save the image
reallocations when the function is called repeatedly for images of the same size.
*/
CV_EXPORTS Mat imdecode( InputArray buf, int flags, Mat* dst);

/** @brief Encodes an image into a memory buffer.

The function imencode compresses the image and stores it in the memory buffer that is resized to fit the
result. See cv::imwrite for the list of supported formats and flags description.

@param ext File extension that defines the output format.
@param img Image to be written.
@param buf Output buffer resized to fit the compressed image.
@param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.
*/
CV_EXPORTS_W bool imencode( const String& ext, InputArray img,
                            CV_OUT std::vector<uchar>& buf,
                            const std::vector<int>& params = std::vector<int>());

//! @} imgcodecs

} // cv

#endif //OPENCV_IMGCODECS_HPP
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

#ifndef OPENCV_IMGPROC_HPP
#define OPENCV_IMGPROC_HPP

#include "opencv2/core.hpp"

/**
  @defgroup imgproc Image processing
  @{
    @defgroup imgproc_filter Image Filtering

Functions and classes described in this section are used to perform various linear or non-linear
filtering operations on 2D images (represented as Mat's). It means that for each pixel location
\f$(x,y)\f$ in the source image (normally, rectangular), its neighborhood is considered and used to
compute the response. In case of a linear filter, it is a weighted sum of pixel values. In case of
morphological operations, it is the minimum or maximum values, and so on. The computed response is
stored in the destination image at the same location \f$(x,y)\f$. It means that the output image
will be of the same size as the input image. Normally, the functions support multi-channel arrays,
in which case every channel is processed independently. Therefore, the output image will also have
the same number of channels as the input one.

Another common feature of the functions and classes described in this section is that, unlike
simple arithmetic functions, they need to extrapolate values of some non-existing pixels. For
example, if you want to smooth an image using a Gaussian \f$3 \times 3\f$ filter, then, when
processing the left-most pixels in each row, you need pixels to the left of them, that is, outside
of the image. You can let these pixels be the same as the left-most image pixels ("replicated
border" extrapolation method), or assume that all the non-existing pixels are zeros ("constant
border" extrapolation method), and so on. OpenCV enables you to specify the extrapolation method.
For details, see cv::BorderTypes

@anchor filter_depths
### Depth combinations
Input depth (src.depth()) | Output depth (ddepth)
--------------------------|----------------------
CV_8U                     | -1/CV_16S/CV_32F/CV_64F
CV_16U/CV_16S             | -1/CV_32F/CV_64F
CV_32F                    | -1/CV_32F/CV_64F
CV_64F                    | -1/CV_64F

@note when ddepth=-1, the output image will have the same depth as the source.

    @defgroup imgproc_transform Geometric Image Transformations

The functions in this section perform various geometrical transformations of 2D images. They do not
change the image content but deform the pixel grid and map this deformed grid to the destination
image. In fact, to avoid sampling artifacts, the mapping is done in the reverse order, from
destination to the source. That is, for each pixel \f$(x, y)\f$ of the destination image, the
functions compute coordinates of the corresponding "donor" pixel in the source image and copy the
pixel value:

\f[\texttt{dst} (x,y)= \texttt{src} (f_x(x,y), f_y(x,y))\f]

In case when you specify the forward mapping \f$\left<g_x, g_y\right>: \texttt{src} \rightarrow
\texttt{dst}\f$, the OpenCV functions first compute the corresponding inverse mapping
\f$\left<f_x, f_y\right>: \texttt{dst} \rightarrow \texttt{src}\f$ and then use the above formula.

The actual implementations of the geometrical transformations, from the most generic remap and to
the simplest and the fastest resize, need to solve two main problems with the above formula:

- Extrapolation of non-existing pixels. Similarly to the filtering functions described in the
previous section, for some \f$(x,y)\f$, either one of \f$f_x(x,y)\f$, or \f$f_y(x,y)\f$, or both
of them may fall outside of the image. In this case, an extrapolation method needs to be used.
OpenCV provides the same selection of extrapolation methods as in the filtering functions. In
addition, it provides the method BORDER_TRANSPARENT. This means that the corresponding pixels in
the destination image will not be modified at all.

- Interpolation of pixel values. Usually \f$f_x(x,y)\f$ and \f$f_y(x,y)\f$ are floating-point
numbers. This means that \f$\left<f_x, f_y\right>\f$ can be either an affine or perspective
transformation, or radial lens distortion correction, and so on. So, a pixel value at fractional
coordinates needs to be retrieved. In the simplest case, the coordinates can be just rounded to the
nearest integer coordinates and the corresponding pixel can be used. This is called a
nearest-neighbor interpolation. However, a better result can be achieved by using more
sophisticated [interpolation methods](http://en.wikipedia.org/wiki/Multivariate_interpolation) ,
where a polynomial function is fit into some neighborhood of the computed pixel \f$(f_x(x,y),
f_y(x,y))\f$, and then the value of the polynomial at \f$(f_x(x,y), f_y(x,y))\f$ is taken as the
interpolated pixel value. In OpenCV, you can choose between several interpolation methods. See
resize for details.

    @defgroup imgproc_misc Miscellaneous Image Transformations
    @defgroup imgproc_draw Drawing Functions

Drawing functions work with matrices/images of arbitrary depth. The boundaries of the shapes can be
rendered with antialiasing (implemented only for 8-bit images for now). All the functions include
the parameter color that uses an RGB value (that may be constructed with the Scalar constructor )
for color images and brightness for grayscale images. For color images, the channel ordering is
normally *Blue, Green, Red*. This is what imshow, imread, and imwrite expect. So, if you form a
color using the Scalar constructor, it should look like:

\f[\texttt{Scalar} (blue \_ component, green \_ component, red \_ component[, alpha \_ component])\f]

If you are using your own image rendering and I/O functions, you can use any channel ordering. The
drawing functions process each channel independently and do not depend on the channel order or even
on the used color space. The whole image can be converted from BGR to RGB or to a different color
space using cvtColor .

If a drawn figure is partially or completely outside the image, the drawing functions clip it. Also,
many drawing functions can handle pixel coordinates specified with sub-pixel accuracy. This means
that the coordinates can be passed as fixed-point numbers encoded as integers. The number of
fractional bits is specified by the shift parameter and the real point coordinates are calculated as
\f$\texttt{Point}(x,y)\rightarrow\texttt{Point2f}(x*2^{-shift},y*2^{-shift})\f$ . This feature is
especially effective when rendering antialiased shapes.

@note The functions do not support alpha-transparency when the target image is 4-channel. In this
case, the color[3] is simply copied to the repainted pixels. Thus, if you want to paint
semi-transparent shapes, you can paint them in a separate buffer and then blend it with the main
image.

    @defgroup imgproc_colormap ColorMaps in OpenCV

The human perception isn't built for observing fine changes in grayscale images. Human eyes are more
sensitive to observing changes between colors, so you often need to recolor your grayscale images to
get a clue about them. OpenCV now comes with various colormaps to enhance the visualization in your
computer vision application.

In OpenCV you only need applyColorMap to apply a colormap on a given image. The following sample
code reads the path to an image from command line, applies a Jet colormap on it and shows the
result:

@code
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

#include <iostream>
using namespace std;

int main(int argc, const char *argv[])
{
    // We need an input image. (can be grayscale or color)
    if (argc < 2)
    {
        cerr << "We need an image to process here. Please run: colorMap [path_to_image]" << endl;
        return -1;
    }
    Mat img_in = imread(argv[1]);
    if(img_in.empty())
    {
        cerr << "Sample image (" << argv[1] << ") is empty. Please adjust your path, so it points to a valid input image!" << endl;
        return -1;
    }
    // Holds the colormap version of the image:
    Mat img_color;
    // Apply the colormap:
    applyColorMap(img_in, img_color, COLORMAP_JET);
    // Show the result:
    imshow("colorMap", img_color);
    waitKey(0);
    return 0;
}
@endcode

@see cv::ColormapTypes

    @defgroup imgproc_subdiv2d Planar Subdivision

The Subdiv2D class described in this section is used to perform various planar subdivision on
a set of 2D points (represented as vector of Point2f). OpenCV subdivides a plane into triangles
using the Delaunay's algorithm, which corresponds to the dual graph of the Voronoi diagram.
In the figure below, the Delaunay's triangulation is marked with black lines and the Voronoi
diagram with red lines.

![Delaunay triangulation (black) and Voronoi (red)](pics/delaunay_voronoi.png)

The subdivisions can be used for the 3D piece-wise transformation of a plane, morphing, fast
location of points on the plane, building special graphs (such as NNG,RNG), and so forth.

    @defgroup imgproc_hist Histograms
    @defgroup imgproc_shape Structural Analysis and Shape Descriptors
    @defgroup imgproc_motion Motion Analysis and Object Tracking
    @defgroup imgproc_feature Feature Detection
    @defgroup imgproc_object Object Detection
    @defgroup imgproc_c C API
    @defgroup imgproc_hal Hardware Acceleration Layer
    @{
        @defgroup imgproc_hal_functions Functions
        @defgroup imgproc_hal_interface Interface
    @}
  @}
*/

namespace cv
{

/** @addtogroup imgproc
@{
*/

//! @addtogroup imgproc_filter
//! @{

//! type of morphological operation
enum MorphTypes{
    MORPH_ERODE    = 0, //!< see cv::erode
    MORPH_DILATE   = 1, //!< see cv::dilate
    MORPH_OPEN     = 2, //!< an opening operation
                        //!< \f[\texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))\f]
    MORPH_CLOSE    = 3, //!< a closing operation
                        //!< \f[\texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))\f]
    MORPH_GRADIENT = 4, //!< a morphological gradient
                        //!< \f[\texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )\f]
    MORPH_TOPHAT   = 5, //!< "top hat"
                        //!< \f[\texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )\f]
    MORPH_BLACKHAT = 6, //!< "black hat"
                        //!< \f[\texttt{dst} = \mathrm{blackhat} ( \texttt{src} , \texttt{element} )= \mathrm{close} ( \texttt{src} , \texttt{element} )- \texttt{src}\f]
    MORPH_HITMISS  = 7  //!< "hit or miss"
                        //!<   .- Only supported for CV_8UC1 binary images. A tutorial can be found in the documentation
};

//! shape of the structuring element
enum MorphShapes {
    MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
    MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
                       //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
    MORPH_ELLIPSE = 2 //!< an elliptic structuring element, that is, a filled ellipse inscribed
                      //!< into the rectangle Rect(0, 0, esize.width, 0.esize.height)
};

//! @} imgproc_filter

//! @addtogroup imgproc_transform
//! @{

//! interpolation algorithm
enum InterpolationFlags{
    /** nearest neighbor interpolation */
    INTER_NEAREST        = 0,
    /** bilinear interpolation */
    INTER_LINEAR         = 1,
    /** bicubic interpolation */
    INTER_CUBIC          = 2,
    /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    method. */
    INTER_AREA           = 3,
    /** Lanczos interpolation over 8x8 neighborhood */
    INTER_LANCZOS4       = 4,
    /** mask for interpolation codes */
    INTER_MAX            = 7,
    /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
    source image, they are set to zero */
    WARP_FILL_OUTLIERS   = 8,
    /** flag, inverse transformation

    For example, @ref cv::linearPolar or @ref cv::logPolar transforms:
    - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    */
    WARP_INVERSE_MAP     = 16
};

enum InterpolationMasks {
       INTER_BITS      = 5,
       INTER_BITS2     = INTER_BITS * 2,
       INTER_TAB_SIZE  = 1 << INTER_BITS,
       INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
     };

//! @} imgproc_transform

//! @addtogroup imgproc_misc
//! @{

//! Distance types for Distance Transform and M-estimators
//! @see cv::distanceTransform, cv::fitLine
enum DistanceTypes {
    DIST_USER    = -1,  //!< User defined distance
    DIST_L1      = 1,   //!< distance = |x1-x2| + |y1-y2|
    DIST_L2      = 2,   //!< the simple euclidean distance
    DIST_C       = 3,   //!< distance = max(|x1-x2|,|y1-y2|)
    DIST_L12     = 4,   //!< L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
    DIST_FAIR    = 5,   //!< distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
    DIST_WELSCH  = 6,   //!< distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
    DIST_HUBER   = 7    //!< distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
};

//! Mask size for distance transform
enum DistanceTransformMasks {
    DIST_MASK_3       = 3, //!< mask=3
    DIST_MASK_5       = 5, //!< mask=5
    DIST_MASK_PRECISE = 0  //!<
};

//! type of the threshold operation
//! ![threshold types](pics/threshold.png)
enum ThresholdTypes {
    THRESH_BINARY     = 0, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{maxval}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
    THRESH_BINARY_INV = 1, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{maxval}}{otherwise}\f]
    THRESH_TRUNC      = 2, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{threshold}}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
    THRESH_TOZERO     = 3, //!< \f[\texttt{dst} (x,y) =  \fork{\texttt{src}(x,y)}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{0}{otherwise}\f]
    THRESH_TOZERO_INV = 4, //!< \f[\texttt{dst} (x,y) =  \fork{0}{if \(\texttt{src}(x,y) > \texttt{thresh}\)}{\texttt{src}(x,y)}{otherwise}\f]
    THRESH_MASK       = 7,
    THRESH_OTSU       = 8, //!< flag, use Otsu algorithm to choose the optimal threshold value
    THRESH_TRIANGLE   = 16 //!< flag, use Triangle algorithm to choose the optimal threshold value
};

//! adaptive threshold algorithm
//! see cv::adaptiveThreshold
enum AdaptiveThresholdTypes {
    /** the threshold value \f$T(x,y)\f$ is a mean of the \f$\texttt{blockSize} \times
    \texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$ minus C */
    ADAPTIVE_THRESH_MEAN_C     = 0,
    /** the threshold value \f$T(x, y)\f$ is a weighted sum (cross-correlation with a Gaussian
    window) of the \f$\texttt{blockSize} \times \texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$
    minus C . The default sigma (standard deviation) is used for the specified blockSize . See
    cv::getGaussianKernel*/
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
};

//! cv::undistort mode
enum UndistortTypes {
       PROJ_SPHERICAL_ORTHO  = 0,
       PROJ_SPHERICAL_EQRECT = 1
     };

//! class of the pixel in GrabCut algorithm
enum GrabCutClasses {
    GC_BGD    = 0,  //!< an obvious background pixels
    GC_FGD    = 1,  //!< an obvious foreground (object) pixel
    GC_PR_BGD = 2,  //!< a possible background pixel
    GC_PR_FGD = 3   //!< a possible foreground pixel
};

//! GrabCut algorithm flags
enum GrabCutModes {
    /** The function initializes the state and the mask using the provided rectangle. After that it
    runs iterCount iterations of the algorithm. */
    GC_INIT_WITH_RECT  = 0,
    /** The function initializes the state using the provided mask. Note that GC_INIT_WITH_RECT
    and GC_INIT_WITH_MASK can be combined. Then, all the pixels outside of the ROI are
    automatically initialized with GC_BGD .*/
    GC_INIT_WITH_MASK  = 1,
    /** The value means that the algorithm should just resume. */
    GC_EVAL            = 2
};

//! distanceTransform algorithm flags
enum DistanceTransformLabelTypes {
    /** each connected component of zeros in src (as well as all the non-zero pixels closest to the
    connected component) will be assigned the same label */
    DIST_LABEL_CCOMP = 0,
    /** each zero pixel (and all the non-zero pixels closest to it) gets its own label. */
    DIST_LABEL_PIXEL = 1
};

//! floodfill algorithm flags
enum FloodFillFlags {
    /** If set, the difference between the current pixel and seed pixel is considered. Otherwise,
    the difference between neighbor pixels is considered (that is, the range is floating). */
    FLOODFILL_FIXED_RANGE = 1 << 16,
    /** If set, the function does not change the image ( newVal is ignored), and only fills the
    mask with the value specified in bits 8-16 of flags as described above. This option only make
    sense in function variants that have the mask parameter. */
    FLOODFILL_MASK_ONLY   = 1 << 17
};

//! @} imgproc_misc

//! @addtogroup imgproc_shape
//! @{

//! connected components algorithm output formats
enum ConnectedComponentsTypes {
    CC_STAT_LEFT   = 0, //!< The leftmost (x) coordinate which is the inclusive start of the bounding
                        //!< box in the horizontal direction.
    CC_STAT_TOP    = 1, //!< The topmost (y) coordinate which is the inclusive start of the bounding
                        //!< box in the vertical direction.
    CC_STAT_WIDTH  = 2, //!< The horizontal size of the bounding box
    CC_STAT_HEIGHT = 3, //!< The vertical size of the bounding box
    CC_STAT_AREA   = 4, //!< The total area (in pixels) of the connected component
    CC_STAT_MAX    = 5
};

//! connected components algorithm
enum ConnectedComponentsAlgorithmsTypes {
    CCL_WU      = 0,  //!< SAUF algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity
    CCL_DEFAULT = -1, //!< BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity
    CCL_GRANA   = 1   //!< BBDT algorithm for 8-way connectivity, SAUF algorithm for 4-way connectivity
};

//! mode of the contour retrieval algorithm
enum RetrievalModes {
    /** retrieves only the extreme outer contours. It sets `hierarchy[i][2]=hierarchy[i][3]=-1` for
    all the contours. */
    RETR_EXTERNAL  = 0,
    /** retrieves all of the contours without establishing any hierarchical relationships. */
    RETR_LIST      = 1,
    /** retrieves all of the contours and organizes them into a two-level hierarchy. At the top
    level, there are external boundaries of the components. At the second level, there are
    boundaries of the holes. If there is another contour inside a hole of a connected component, it
    is still put at the top level. */
    RETR_CCOMP     = 2,
    /** retrieves all of the contours and reconstructs a full hierarchy of nested contours.*/
    RETR_TREE      = 3,
    RETR_FLOODFILL = 4 //!<
};

//! the contour approximation algorithm
enum ContourApproximationModes {
    /** stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and
    (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is,
    max(abs(x1-x2),abs(y2-y1))==1. */
    CHAIN_APPROX_NONE      = 1,
    /** compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    For example, an up-right rectangular contour is encoded with 4 points. */
    CHAIN_APPROX_SIMPLE    = 2,
    /** applies one of the flavors of the Teh-Chin chain approximation algorithm @cite TehChin89 */
    CHAIN_APPROX_TC89_L1   = 3,
    /** applies one of the flavors of the Teh-Chin chain approximation algorithm @cite TehChin89 */
    CHAIN_APPROX_TC89_KCOS = 4
};

/** @brief Shape matching methods

\f$A\f$ denotes object1,\f$B\f$ denotes object2

\f$\begin{array}{l} m^A_i =  \mathrm{sign} (h^A_i)  \cdot \log{h^A_i} \\ m^B_i =  \mathrm{sign} (h^B_i)  \cdot \log{h^B_i} \end{array}\f$

and \f$h^A_i, h^B_i\f$ are the Hu moments of \f$A\f$ and \f$B\f$ , respectively.
*/
enum ShapeMatchModes {
    CONTOURS_MATCH_I1  =1, //!< \f[I_1(A,B) =  \sum _{i=1...7}  \left |  \frac{1}{m^A_i} -  \frac{1}{m^B_i} \right |\f]
    CONTOURS_MATCH_I2  =2, //!< \f[I_2(A,B) =  \sum _{i=1...7}  \left | m^A_i - m^B_i  \right |\f]
    CONTOURS_MATCH_I3  =3  //!< \f[I_3(A,B) =  \max _{i=1...7}  \frac{ \left| m^A_i - m^B_i \right| }{ \left| m^A_i \right| }\f]
};

//! @} imgproc_shape

//! Variants of a Hough transform
enum HoughModes {

    /** classical or standard Hough transform. Every line is represented by two floating-point
    numbers \f$(\rho, \theta)\f$ , where \f$\rho\f$ is a distance between (0,0) point and the line,
    and \f$\theta\f$ is the angle between x-axis and the normal to the line. Thus, the matrix must
    be (the created sequence will be) of CV_32FC2 type */
    HOUGH_STANDARD      = 0,
    /** probabilistic Hough transform (more efficient in case if the picture contains a few long
    linear segments). It returns line segments rather than the whole line. Each segment is
    represented by starting and ending points, and the matrix must be (the created sequence will
    be) of the CV_32SC4 type. */
    HOUGH_PROBABILISTIC = 1,
    /** multi-scale variant of the classical Hough transform. The lines are encoded the same way as
    HOUGH_STANDARD. */
    HOUGH_MULTI_SCALE   = 2,
    HOUGH_GRADIENT      = 3 //!< basically *21HT*, described in @cite Yuen90
};

//! Variants of Line Segment %Detector
//! @ingroup imgproc_feature
enum LineSegmentDetectorModes {
    LSD_REFINE_NONE = 0, //!< No refinement applied
    LSD_REFINE_STD  = 1, //!< Standard refinement is applied. E.g. breaking arches into smaller straighter line approximations.
    LSD_REFINE_ADV  = 2  //!< Advanced refinement. Number of false alarms is calculated, lines are
                         //!< refined through increase of precision, decrement in size, etc.
};

/** Histogram comparison methods
  @ingroup imgproc_hist
*/
enum HistCompMethods {
    /** Correlation
    \f[d(H_1,H_2) =  \frac{\sum_I (H_1(I) - \bar{H_1}) (H_2(I) - \bar{H_2})}{\sqrt{\sum_I(H_1(I) - \bar{H_1})^2 \sum_I(H_2(I) - \bar{H_2})^2}}\f]
    where
    \f[\bar{H_k} =  \frac{1}{N} \sum _J H_k(J)\f]
    and \f$N\f$ is a total number of histogram bins. */
    HISTCMP_CORREL        = 0,
    /** Chi-Square
    \f[d(H_1,H_2) =  \sum _I  \frac{\left(H_1(I)-H_2(I)\right)^2}{H_1(I)}\f] */
    HISTCMP_CHISQR        = 1,
    /** Intersection
    \f[d(H_1,H_2) =  \sum _I  \min (H_1(I), H_2(I))\f] */
    HISTCMP_INTERSECT     = 2,
    /** Bhattacharyya distance
    (In fact, OpenCV computes Hellinger distance, which is related to Bhattacharyya coefficient.)
    \f[d(H_1,H_2) =  \sqrt{1 - \frac{1}{\sqrt{\bar{H_1} \bar{H_2} N^2}} \sum_I \sqrt{H_1(I) \cdot H_2(I)}}\f] */
    HISTCMP_BHATTACHARYYA = 3,
    HISTCMP_HELLINGER     = HISTCMP_BHATTACHARYYA, //!< Synonym for HISTCMP_BHATTACHARYYA
    /** Alternative Chi-Square
    \f[d(H_1,H_2) =  2 * \sum _I  \frac{\left(H_1(I)-H_2(I)\right)^2}{H_1(I)+H_2(I)}\f]
    This alternative formula is regularly used for texture comparison. See e.g. @cite Puzicha1997 */
    HISTCMP_CHISQR_ALT    = 4,
    /** Kullback-Leibler divergence
    \f[d(H_1,H_2) = \sum _I H_1(I) \log \left(\frac{H_1(I)}{H_2(I)}\right)\f] */
    HISTCMP_KL_DIV        = 5
};

/** the color conversion code
@see @ref imgproc_color_conversions
@ingroup imgproc_misc
 */
enum ColorConversionCodes {
    COLOR_BGR2BGRA     = 0, //!< add alpha channel to RGB or BGR image
    COLOR_RGB2RGBA     = COLOR_BGR2BGRA,

    COLOR_BGRA2BGR     = 1, //!< remove alpha channel from RGB or BGR image
    COLOR_RGBA2RGB     = COLOR_BGRA2BGR,

    COLOR_BGR2RGBA     = 2, //!< convert between RGB and BGR color spaces (with or without alpha channel)
    COLOR_RGB2BGRA     = COLOR_BGR2RGBA,

    COLOR_RGBA2BGR     = 3,
    COLOR_BGRA2RGB     = COLOR_RGBA2BGR,

    COLOR_BGR2RGB      = 4,
    COLOR_RGB2BGR      = COLOR_BGR2RGB,

    COLOR_BGRA2RGBA    = 5,
    COLOR_RGBA2BGRA    = COLOR_BGRA2RGBA,

    COLOR_BGR2GRAY     = 6, //!< convert between RGB/BGR and grayscale, @ref color_convert_rgb_gray "color conversions"
    COLOR_RGB2GRAY     = 7,
    COLOR_GRAY2BGR     = 8,
    COLOR_GRAY2RGB     = COLOR_GRAY2BGR,
    COLOR_GRAY2BGRA    = 9,
    COLOR_GRAY2RGBA    = COLOR_GRAY2BGRA,
    COLOR_BGRA2GRAY    = 10,
    COLOR_RGBA2GRAY    = 11,

    COLOR_BGR2BGR565   = 12, //!< convert between RGB/BGR and BGR565 (16-bit images)
    COLOR_RGB2BGR565   = 13,
    COLOR_BGR5652BGR   = 14,
    COLOR_BGR5652RGB   = 15,
    COLOR_BGRA2BGR565  = 16,
    COLOR_RGBA2BGR565  = 17,
    COLOR_BGR5652BGRA  = 18,
    COLOR_BGR5652RGBA  = 19,

    COLOR_GRAY2BGR565  = 20, //!< convert between grayscale to BGR565 (16-bit images)
    COLOR_BGR5652GRAY  = 21,

    COLOR_BGR2BGR555   = 22,  //!< convert between RGB/BGR and BGR555 (16-bit images)
    COLOR_RGB2BGR555   = 23,
    COLOR_BGR5552BGR   = 24,
    COLOR_BGR5552RGB   = 25,
    COLOR_BGRA2BGR555  = 26,
    COLOR_RGBA2BGR555  = 27,
    COLOR_BGR5552BGRA  = 28,
    COLOR_BGR5552RGBA  = 29,

    COLOR_GRAY2BGR555  = 30, //!< convert between grayscale and BGR555 (16-bit images)
    COLOR_BGR5552GRAY  = 31,

    COLOR_BGR2XYZ      = 32, //!< convert RGB/BGR to CIE XYZ, @ref color_convert_rgb_xyz "color conversions"
    COLOR_RGB2XYZ      = 33,
    COLOR_XYZ2BGR      = 34,
    COLOR_XYZ2RGB      = 35,

    COLOR_BGR2YCrCb    = 36, //!< convert RGB/BGR to luma-chroma (aka YCC), @ref color_convert_rgb_ycrcb "color conversions"
    COLOR_RGB2YCrCb    = 37,
    COLOR_YCrCb2BGR    = 38,
    COLOR_YCrCb2RGB    = 39,

    COLOR_BGR2HSV      = 40, //!< convert RGB/BGR to HSV (hue saturation value), @ref color_convert_rgb_hsv "color conversions"
    COLOR_RGB2HSV      = 41,

    COLOR_BGR2Lab      = 44, //!< convert RGB/BGR to CIE Lab, @ref color_convert_rgb_lab "color conversions"
    COLOR_RGB2Lab      = 45,

    COLOR_BGR2Luv      = 50, //!< convert RGB/BGR to CIE Luv, @ref color_convert_rgb_luv "color conversions"
    COLOR_RGB2Luv      = 51,
    COLOR_BGR2HLS      = 52, //!< convert RGB/BGR to HLS (hue lightness saturation), @ref color_convert_rgb_hls "color conversions"
    COLOR_RGB2HLS      = 53,

    COLOR_HSV2BGR      = 54, //!< backward conversions to RGB/BGR
    COLOR_HSV2RGB      = 55,

    COLOR_Lab2BGR      = 56,
    COLOR_Lab2RGB      = 57,
    COLOR_Luv2BGR      = 58,
    COLOR_Luv2RGB      = 59,
    COLOR_HLS2BGR      = 60,
    COLOR_HLS2RGB      = 61,

    COLOR_BGR2HSV_FULL = 66, //!<
    COLOR_RGB2HSV_FULL = 67,
    COLOR_BGR2HLS_FULL = 68,
    COLOR_RGB2HLS_FULL = 69,

    COLOR_HSV2BGR_FULL = 70,
    COLOR_HSV2RGB_FULL = 71,
    COLOR_HLS2BGR_FULL = 72,
    COLOR_HLS2RGB_FULL = 73,

    COLOR_LBGR2Lab     = 74,
    COLOR_LRGB2Lab     = 75,
    COLOR_LBGR2Luv     = 76,
    COLOR_LRGB2Luv     = 77,

    COLOR_Lab2LBGR     = 78,
    COLOR_Lab2LRGB     = 79,
    COLOR_Luv2LBGR     = 80,
    COLOR_Luv2LRGB     = 81,

    COLOR_BGR2YUV      = 82, //!< convert between RGB/BGR and YUV
    COLOR_RGB2YUV      = 83,
    COLOR_YUV2BGR      = 84,
    COLOR_YUV2RGB      = 85,

    //! YUV 4:2:0 family to RGB
    COLOR_YUV2RGB_NV12  = 90,
    COLOR_YUV2BGR_NV12  = 91,
    COLOR_YUV2RGB_NV21  = 92,
    COLOR_YUV2BGR_NV21  = 93,
    COLOR_YUV420sp2RGB  = COLOR_YUV2RGB_NV21,
    COLOR_YUV420sp2BGR  = COLOR_YUV2BGR_NV21,

    COLOR_YUV2RGBA_NV12 = 94,
    COLOR_YUV2BGRA_NV12 = 95,
    COLOR_YUV2RGBA_NV21 = 96,
    COLOR_YUV2BGRA_NV21 = 97,
    COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21,
    COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21,

    COLOR_YUV2RGB_YV12  = 98,
    COLOR_YUV2BGR_YV12  = 99,
    COLOR_YUV2RGB_IYUV  = 100,
    COLOR_YUV2BGR_IYUV  = 101,
    COLOR_YUV2RGB_I420  = COLOR_YUV2RGB_IYUV,
    COLOR_YUV2BGR_I420  = COLOR_YUV2BGR_IYUV,
    COLOR_YUV420p2RGB   = COLOR_YUV2RGB_YV12,
    COLOR_YUV420p2BGR   = COLOR_YUV2BGR_YV12,

    COLOR_YUV2RGBA_YV12 = 102,
    COLOR_YUV2BGRA_YV12 = 103,
    COLOR_YUV2RGBA_IYUV = 104,
    COLOR_YUV2BGRA_IYUV = 105,
    COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV,
    COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV,
    COLOR_YUV420p2RGBA  = COLOR_YUV2RGBA_YV12,
    COLOR_YUV420p2BGRA  = COLOR_YUV2BGRA_YV12,

    COLOR_YUV2GRAY_420  = 106,
    COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420,
    COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420,
    COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420,
    COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420,
    COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420,
    COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420,
    COLOR_YUV420p2GRAY  = COLOR_YUV2GRAY_420,

    //! YUV 4:2:2 family to RGB
    COLOR_YUV2RGB_UYVY = 107,
    COLOR_YUV2BGR_UYVY = 108,
    //COLOR_YUV2RGB_VYUY = 109,
    //COLOR_YUV2BGR_VYUY = 110,
    COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY,
    COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY,
    COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY,
    COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY,

    COLOR_YUV2RGBA_UYVY = 111,
    COLOR_YUV2BGRA_UYVY = 112,
    //COLOR_YUV2RGBA_VYUY = 113,
    //COLOR_YUV2BGRA_VYUY = 114,
    COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY,
    COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY,
    COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY,
    COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY,

    COLOR_YUV2RGB_YUY2 = 115,
    COLOR_YUV2BGR_YUY2 = 116,
    COLOR_YUV2RGB_YVYU = 117,
    COLOR_YUV2BGR_YVYU = 118,
    COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2,
    COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2,
    COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2,
    COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2,

    COLOR_YUV2RGBA_YUY2 = 119,
    COLOR_YUV2BGRA_YUY2 = 120,
    COLOR_YUV2RGBA_YVYU = 121,
    COLOR_YUV2BGRA_YVYU = 122,
    COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2,
    COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2,
    COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2,
    COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2,

    COLOR_YUV2GRAY_UYVY = 123,
    COLOR_YUV2GRAY_YUY2 = 124,
    //CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY,
    COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY,
    COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY,
    COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2,
    COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2,
    COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2,

    //! alpha premultiplication
    COLOR_RGBA2mRGBA    = 125,
    COLOR_mRGBA2RGBA    = 126,

    //! RGB to YUV 4:2:0 family
    COLOR_RGB2YUV_I420  = 127,
    COLOR_BGR2YUV_I420  = 128,
    COLOR_RGB2YUV_IYUV  = COLOR_RGB2YUV_I420,
    COLOR_BGR2YUV_IYUV  = COLOR_BGR2YUV_I420,

    COLOR_RGBA2YUV_I420 = 129,
    COLOR_BGRA2YUV_I420 = 130,
    COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420,
    COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420,
    COLOR_RGB2YUV_YV12  = 131,
    COLOR_BGR2YUV_YV12  = 132,
    COLOR_RGBA2YUV_YV12 = 133,
    COLOR_BGRA2YUV_YV12 = 134,

    //! Demosaicing
    COLOR_BayerBG2BGR = 46,
    COLOR_BayerGB2BGR = 47,
    COLOR_BayerRG2BGR = 48,
    COLOR_BayerGR2BGR = 49,

    COLOR_BayerBG2RGB = COLOR_BayerRG2BGR,
    COLOR_BayerGB2RGB = COLOR_BayerGR2BGR,
    COLOR_BayerRG2RGB = COLOR_BayerBG2BGR,
    COLOR_BayerGR2RGB = COLOR_BayerGB2BGR,

    COLOR_BayerBG2GRAY = 86,
    COLOR_BayerGB2GRAY = 87,
    COLOR_BayerRG2GRAY = 88,
    COLOR_BayerGR2GRAY = 89,

    //! Demosaicing using Variable Number of Gradients
    COLOR_BayerBG2BGR_VNG = 62,
    COLOR_BayerGB2BGR_VNG = 63,
    COLOR_BayerRG2BGR_VNG = 64,
    COLOR_BayerGR2BGR_VNG = 65,

    COLOR_BayerBG2RGB_VNG = COLOR_BayerRG2BGR_VNG,
    COLOR_BayerGB2RGB_VNG = COLOR_BayerGR2BGR_VNG,
    COLOR_BayerRG2RGB_VNG = COLOR_BayerBG2BGR_VNG,
    COLOR_BayerGR2RGB_VNG = COLOR_BayerGB2BGR_VNG,

    //! Edge-Aware Demosaicing
    COLOR_BayerBG2BGR_EA  = 135,
    COLOR_BayerGB2BGR_EA  = 136,
    COLOR_BayerRG2BGR_EA  = 137,
    COLOR_BayerGR2BGR_EA  = 138,

    COLOR_BayerBG2RGB_EA  = COLOR_BayerRG2BGR_EA,
    COLOR_BayerGB2RGB_EA  = COLOR_BayerGR2BGR_EA,
    COLOR_BayerRG2RGB_EA  = COLOR_BayerBG2BGR_EA,
    COLOR_BayerGR2RGB_EA  = COLOR_BayerGB2BGR_EA,

    //! Demosaicing with alpha channel
    COLOR_BayerBG2BGRA = 139,
    COLOR_BayerGB2BGRA = 140,
    COLOR_BayerRG2BGRA = 141,
    COLOR_BayerGR2BGRA = 142,

    COLOR_BayerBG2RGBA = COLOR_BayerRG2BGRA,
    COLOR_BayerGB2RGBA = COLOR_BayerGR2BGRA,
    COLOR_BayerRG2RGBA = COLOR_BayerBG2BGRA,
    COLOR_BayerGR2RGBA = COLOR_BayerGB2BGRA,

    COLOR_COLORCVT_MAX  = 143
};

/** types of intersection between rectangles
@ingroup imgproc_shape
*/
enum RectanglesIntersectTypes {
    INTERSECT_NONE = 0, //!< No intersection
    INTERSECT_PARTIAL  = 1, //!< There is a partial intersection
    INTERSECT_FULL  = 2 //!< One of the rectangle is fully enclosed in the other
};

//! finds arbitrary template in the grayscale image using Generalized Hough Transform
class CV_EXPORTS GeneralizedHough : public Algorithm
{
public:
    //! set template to search
    virtual void setTemplate(InputArray templ, Point templCenter = Point(-1, -1)) = 0;
    virtual void setTemplate(InputArray edges, InputArray dx, InputArray dy, Point templCenter = Point(-1, -1)) = 0;

    //! find template on image
    virtual void detect(InputArray image, OutputArray positions, OutputArray votes = noArray()) = 0;
    virtual void detect(InputArray edges, InputArray dx, InputArray dy, OutputArray positions, OutputArray votes = noArray()) = 0;

    //! Canny low threshold.
    virtual void setCannyLowThresh(int cannyLowThresh) = 0;
    virtual int getCannyLowThresh() const = 0;

    //! Canny high threshold.
    virtual void setCannyHighThresh(int cannyHighThresh) = 0;
    virtual int getCannyHighThresh() const = 0;

    //! Minimum distance between the centers of the detected objects.
    virtual void setMinDist(double minDist) = 0;
    virtual double getMinDist() const = 0;

    //! Inverse ratio of the accumulator resolution to the image resolution.
    virtual void setDp(double dp) = 0;
    virtual double getDp() const = 0;

    //! Maximal size of inner buffers.
    virtual void setMaxBufferSize(int maxBufferSize) = 0;
    virtual int getMaxBufferSize() const = 0;
};

//! Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
//! Detects position only without translation and rotation
class CV_EXPORTS GeneralizedHoughBallard : public GeneralizedHough
{
public:
    //! R-Table levels.
    virtual void setLevels(int levels) = 0;
    virtual int getLevels() const = 0;

    //! The accumulator threshold for the template centers at the detection stage. The smaller it is, the more false positions may be detected.
    virtual void setVotesThreshold(int votesThreshold) = 0;
    virtual int getVotesThreshold() const = 0;
};

//! Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
//! Detects position, translation and rotation
class CV_EXPORTS GeneralizedHoughGuil : public GeneralizedHough
{
public:
    //! Angle difference in degrees between two points in feature.
    virtual void setXi(double xi) = 0;
    virtual double getXi() const = 0;

    //! Feature table levels.
    virtual void setLevels(int levels) = 0;
    virtual int getLevels() const = 0;

    //! Maximal difference between angles that treated as equal.
    virtual void setAngleEpsilon(double angleEpsilon) = 0;
    virtual double getAngleEpsilon() const = 0;

    //! Minimal rotation angle to detect in degrees.
    virtual void setMinAngle(double minAngle) = 0;
    virtual double getMinAngle() const = 0;

    //! Maximal rotation angle to detect in degrees.
    virtual void setMaxAngle(double maxAngle) = 0;
    virtual double getMaxAngle() const = 0;

    //! Angle step in degrees.
    virtual void setAngleStep(double angleStep) = 0;
    virtual double getAngleStep() const = 0;

    //! Angle votes threshold.
    virtual void setAngleThresh(int angleThresh) = 0;
    virtual int getAngleThresh() const = 0;

    //! Minimal scale to detect.
    virtual void setMinScale(double minScale) = 0;
    virtual double getMinScale() const = 0;

    //! Maximal scale to detect.
    virtual void setMaxScale(double maxScale) = 0;
    virtual double getMaxScale() const = 0;

    //! Scale step.
    virtual void setScaleStep(double scaleStep) = 0;
    virtual double getScaleStep() const = 0;

    //! Scale votes threshold.
    virtual void setScaleThresh(int scaleThresh) = 0;
    virtual int getScaleThresh() const = 0;

    //! Position votes threshold.
    virtual void setPosThresh(int posThresh) = 0;
    virtual int getPosThresh() const = 0;
};


class CV_EXPORTS_W CLAHE : public Algorithm
{
public:
    CV_WRAP virtual void apply(InputArray src, OutputArray dst) = 0;

    CV_WRAP virtual void setClipLimit(double clipLimit) = 0;
    CV_WRAP virtual double getClipLimit() const = 0;

    CV_WRAP virtual void setTilesGridSize(Size tileGridSize) = 0;
    CV_WRAP virtual Size getTilesGridSize() const = 0;

    CV_WRAP virtual void collectGarbage() = 0;
};


//! @addtogroup imgproc_subdiv2d
//! @{

class CV_EXPORTS_W Subdiv2D
{
public:
    /** Subdiv2D point location cases */
    enum { PTLOC_ERROR        = -2, //!< Point location error
           PTLOC_OUTSIDE_RECT = -1, //!< Point outside the subdivision bounding rect
           PTLOC_INSIDE       = 0, //!< Point inside some facet
           PTLOC_VERTEX       = 1, //!< Point coincides with one of the subdivision vertices
           PTLOC_ON_EDGE      = 2  //!< Point on some edge
         };

    /** Subdiv2D edge type navigation (see: getEdge()) */
    enum { NEXT_AROUND_ORG   = 0x00,
           NEXT_AROUND_DST   = 0x22,
           PREV_AROUND_ORG   = 0x11,
           PREV_AROUND_DST   = 0x33,
           NEXT_AROUND_LEFT  = 0x13,
           NEXT_AROUND_RIGHT = 0x31,
           PREV_AROUND_LEFT  = 0x20,
           PREV_AROUND_RIGHT = 0x02
         };

    /** creates an empty Subdiv2D object.
    To create a new empty Delaunay subdivision you need to use the initDelaunay() function.
     */
    CV_WRAP Subdiv2D();

    /** @overload

    @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.

    The function creates an empty Delaunay subdivision where 2D points can be added using the function
    insert() . All of the points to be added must be within the specified rectangle, otherwise a runtime
    error is raised.
     */
    CV_WRAP Subdiv2D(Rect rect);

    /** @brief Creates a new empty Delaunay subdivision

    @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.

     */
    CV_WRAP void initDelaunay(Rect rect);

    /** @brief Insert a single point into a Delaunay triangulation.

    @param pt Point to insert.

    The function inserts a single point into a subdivision and modifies the subdivision topology
    appropriately. If a point with the same coordinates exists already, no new point is added.
    @returns the ID of the point.

    @note If the point is outside of the triangulation specified rect a runtime error is raised.
     */
    CV_WRAP int insert(Point2f pt);

    /** @brief Insert multiple points into a Delaunay triangulation.

    @param ptvec Points to insert.

    The function inserts a vector of points into a subdivision and modifies the subdivision topology
    appropriately.
     */
    CV_WRAP void insert(const std::vector<Point2f>& ptvec);

    /** @brief Returns the location of a point within a Delaunay triangulation.

    @param pt Point to locate.
    @param edge Output edge that the point belongs to or is located to the right of it.
    @param vertex Optional output vertex the input point coincides with.

    The function locates the input point within the subdivision and gives one of the triangle edges
    or vertices.

    @returns an integer which specify one of the following five cases for point location:
    -  The point falls into some facet. The function returns PTLOC_INSIDE and edge will contain one of
       edges of the facet.
    -  The point falls onto the edge. The function returns PTLOC_ON_EDGE and edge will contain this edge.
    -  The point coincides with one of the subdivision vertices. The function returns PTLOC_VERTEX and
       vertex will contain a pointer to the vertex.
    -  The point is outside the subdivision reference rectangle. The function returns PTLOC_OUTSIDE_RECT
       and no pointers are filled.
    -  One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error
       processing mode is selected, CV_PTLOC_ERROR is returned.
     */
    CV_WRAP int locate(Point2f pt, CV_OUT int& edge, CV_OUT int& vertex);

    /** @brief Finds the subdivision vertex closest to the given point.

    @param pt Input point.
    @param nearestPt Output subdivision vertex point.

    The function is another function that locates the input point within the subdivision. It finds the
    subdivision vertex that is the closest to the input point. It is not necessarily one of vertices
    of the facet containing the input point, though the facet (located using locate() ) is used as a
    starting point.

    @returns vertex ID.
     */
    CV_WRAP int findNearest(Point2f pt, CV_OUT Point2f* nearestPt = 0);

    /** @brief Returns a list of all edges.

    @param edgeList Output vector.

    The function gives each edge as a 4 numbers vector, where each two are one of the edge
    vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3].
     */
    CV_WRAP void getEdgeList(CV_OUT std::vector<Vec4f>& edgeList) const;

    /** @brief Returns a list of the leading edge ID connected to each triangle.

    @param leadingEdgeList Output vector.

    The function gives one edge ID for each triangle.
     */
    CV_WRAP void getLeadingEdgeList(CV_OUT std::vector<int>& leadingEdgeList) const;

    /** @brief Returns a list of all triangles.

    @param triangleList Output vector.

    The function gives each triangle as a 6 numbers vector, where each two are one of the triangle
    vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
     */
    CV_WRAP void getTriangleList(CV_OUT std::vector<Vec6f>& triangleList) const;

    /** @brief Returns a list of all Voroni facets.

    @param idx Vector of vertices IDs to consider. For all vertices you can pass empty vector.
    @param facetList Output vector of the Voroni facets.
    @param facetCenters Output vector of the Voroni facets center points.

     */
    CV_WRAP void getVoronoiFacetList(const std::vector<int>& idx, CV_OUT std::vector<std::vector<Point2f> >& facetList,
                                     CV_OUT std::vector<Point2f>& facetCenters);

    /** @brief Returns vertex location from vertex ID.

    @param vertex vertex ID.
    @param firstEdge Optional. The first edge ID which is connected to the vertex.
    @returns vertex (x,y)

     */
    CV_WRAP Point2f getVertex(int vertex, CV_OUT int* firstEdge = 0) const;

    /** @brief Returns one of the edges related to the given edge.

    @param edge Subdivision edge ID.
    @param nextEdgeType Parameter specifying which of the related edges to return.
    The following values are possible:
    -   NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge)
    -   NEXT_AROUND_DST next around the edge vertex ( eDnext )
    -   PREV_AROUND_ORG previous around the edge origin (reversed eRnext )
    -   PREV_AROUND_DST previous around the edge destination (reversed eLnext )
    -   NEXT_AROUND_LEFT next around the left facet ( eLnext )
    -   NEXT_AROUND_RIGHT next around the right facet ( eRnext )
    -   PREV_AROUND_LEFT previous around the left facet (reversed eOnext )
    -   PREV_AROUND_RIGHT previous around the right facet (reversed eDnext )

    ![sample output](pics/quadedge.png)

    @returns edge ID related to the input edge.
     */
    CV_WRAP int getEdge( int edge, int nextEdgeType ) const;

    /** @brief Returns next edge around the edge origin.

    @param edge Subdivision edge ID.

    @returns an integer which is next edge ID around the edge origin: eOnext on the
    picture above if e is the input edge).
     */
    CV_WRAP int nextEdge(int edge) const;

    /** @brief Returns another edge of the same quad-edge.

    @param edge Subdivision edge ID.
    @param rotate Parameter specifying which of the edges of the same quad-edge as the input
    one to return. The following values are possible:
    -   0 - the input edge ( e on the picture below if e is the input edge)
    -   1 - the rotated edge ( eRot )
    -   2 - the reversed edge (reversed e (in green))
    -   3 - the reversed rotated edge (reversed eRot (in green))

    @returns one of the edges ID of the same quad-edge as the input edge.
     */
    CV_WRAP int rotateEdge(int edge, int rotate) const;
    CV_WRAP int symEdge(int edge) const;

    /** @brief Returns the edge origin.

    @param edge Subdivision edge ID.
    @param orgpt Output vertex location.

    @returns vertex ID.
     */
    CV_WRAP int edgeOrg(int edge, CV_OUT Point2f* orgpt = 0) const;

    /** @brief Returns the edge destination.

    @param edge Subdivision edge ID.
    @param dstpt Output vertex location.

    @returns vertex ID.
     */
    CV_WRAP int edgeDst(int edge, CV_OUT Point2f* dstpt = 0) const;

protected:
    int newEdge();
    void deleteEdge(int edge);
    int newPoint(Point2f pt, bool isvirtual, int firstEdge = 0);
    void deletePoint(int vtx);
    void setEdgePoints( int edge, int orgPt, int dstPt );
    void splice( int edgeA, int edgeB );
    int connectEdges( int edgeA, int edgeB );
    void swapEdges( int edge );
    int isRightOf(Point2f pt, int edge) const;
    void calcVoronoi();
    void clearVoronoi();
    void checkSubdiv() const;

    struct CV_EXPORTS Vertex
    {
        Vertex();
        Vertex(Point2f pt, bool _isvirtual, int _firstEdge=0);
        bool isvirtual() const;
        bool isfree() const;

        int firstEdge;
        int type;
        Point2f pt;
    };

    struct CV_EXPORTS QuadEdge
    {
        QuadEdge();
        QuadEdge(int edgeidx);
        bool isfree() const;

        int next[4];
        int pt[4];
    };

    //! All of the vertices
    std::vector<Vertex> vtx;
    //! All of the edges
    std::vector<QuadEdge> qedges;
    int freeQEdge;
    int freePoint;
    bool validGeometry;

    int recentEdge;
    //! Top left corner of the bounding rect
    Point2f topLeft;
    //! Bottom right corner of the bounding rect
    Point2f bottomRight;
};

//! @} imgproc_subdiv2d

//! @addtogroup imgproc_feature
//! @{

/** @example lsd_lines.cpp
An example using the LineSegmentDetector
*/

/** @brief Line segment detector class

following the algorithm described at @cite Rafael12 .
*/
class CV_EXPORTS_W LineSegmentDetector : public Algorithm
{
public:

    /** @brief Finds lines in the input image.

    This is the output of the default parameters of the algorithm on the above shown image.

    ![image](pics/building_lsd.png)

    @param _image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:
    `lsd_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
    @param _lines A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line. Where
    Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly
    oriented depending on the gradient.
    @param width Vector of widths of the regions, where the lines are found. E.g. Width of line.
    @param prec Vector of precisions with which the lines are found.
    @param nfa Vector containing number of false alarms in the line region, with precision of 10%. The
    bigger the value, logarithmically better the detection.
    - -1 corresponds to 10 mean false alarms
    - 0 corresponds to 1 mean false alarm
    - 1 corresponds to 0.1 mean false alarms
    This vector will be calculated only when the objects type is LSD_REFINE_ADV.
    */
    CV_WRAP virtual void detect(InputArray _image, OutputArray _lines,
                        OutputArray width = noArray(), OutputArray prec = noArray(),
                        OutputArray nfa = noArray()) = 0;

    /** @brief Draws the line segments on a given image.
    @param _image The image, where the lines will be drawn. Should be bigger or equal to the image,
    where the lines were found.
    @param lines A vector of the lines that needed to be drawn.
     */
    CV_WRAP virtual void drawSegments(InputOutputArray _image, InputArray lines) = 0;

    /** @brief Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.

    @param size The size of the image, where lines1 and lines2 were found.
    @param lines1 The first group of lines that needs to be drawn. It is visualized in blue color.
    @param lines2 The second group of lines. They visualized in red color.
    @param _image Optional image, where the lines will be drawn. The image should be color(3-channel)
    in order for lines1 and lines2 to be drawn in the above mentioned colors.
     */
    CV_WRAP virtual int compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image = noArray()) = 0;

    virtual ~LineSegmentDetector() { }
};

/** @brief Creates a smart pointer to a LineSegmentDetector object and initializes it.

The LineSegmentDetector algorithm is defined using the standard values. Only advanced users may want
to edit those, as to tailor it for their own application.

@param _refine The way found lines will be refined, see cv::LineSegmentDetectorModes
@param _scale The scale of the image that will be used to find the lines. Range (0..1].
@param _sigma_scale Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
@param _quant Bound to the quantization error on the gradient norm.
@param _ang_th Gradient angle tolerance in degrees.
@param _log_eps Detection threshold: -log10(NFA) \> log_eps. Used only when advancent refinement
is chosen.
@param _density_th Minimal density of aligned region points in the enclosing rectangle.
@param _n_bins Number of bins in pseudo-ordering of gradient modulus.
 */
CV_EXPORTS_W Ptr<LineSegmentDetector> createLineSegmentDetector(
    int _refine = LSD_REFINE_STD, double _scale = 0.8,
    double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
    double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);

//! @} imgproc_feature

//! @addtogroup imgproc_filter
//! @{

/** @brief Returns Gaussian filter coefficients.

The function computes and returns the \f$\texttt{ksize} \times 1\f$ matrix of Gaussian filter
coefficients:

\f[G_i= \alpha *e^{-(i-( \texttt{ksize} -1)/2)^2/(2* \texttt{sigma}^2)},\f]

where \f$i=0..\texttt{ksize}-1\f$ and \f$\alpha\f$ is the scale factor chosen so that \f$\sum_i G_i=1\f$.

Two of such generated kernels can be passed to sepFilter2D. Those functions automatically recognize
smoothing kernels (a symmetrical kernel with sum of weights equal to 1) and handle them accordingly.
You may also use the higher-level GaussianBlur.
@param ksize Aperture size. It should be odd ( \f$\texttt{ksize} \mod 2 = 1\f$ ) and positive.
@param sigma Gaussian standard deviation. If it is non-positive, it is computed from ksize as
`sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`.
@param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
@sa  sepFilter2D, getDerivKernels, getStructuringElement, GaussianBlur
 */
CV_EXPORTS_W Mat getGaussianKernel( int ksize, double sigma, int ktype = CV_64F );

/** @brief Returns filter coefficients for computing spatial image derivatives.

The function computes and returns the filter coefficients for spatial image derivatives. When
`ksize=CV_SCHARR`, the Scharr \f$3 \times 3\f$ kernels are generated (see cv::Scharr). Otherwise, Sobel
kernels are generated (see cv::Sobel). The filters are normally passed to sepFilter2D or to

@param kx Output matrix of row filter coefficients. It has the type ktype .
@param ky Output matrix of column filter coefficients. It has the type ktype .
@param dx Derivative order in respect of x.
@param dy Derivative order in respect of y.
@param ksize Aperture size. It can be CV_SCHARR, 1, 3, 5, or 7.
@param normalize Flag indicating whether to normalize (scale down) the filter coefficients or not.
Theoretically, the coefficients should have the denominator \f$=2^{ksize*2-dx-dy-2}\f$. If you are
going to filter floating-point images, you are likely to use the normalized kernels. But if you
compute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve
all the fractional bits, you may want to set normalize=false .
@param ktype Type of filter coefficients. It can be CV_32f or CV_64F .
 */
CV_EXPORTS_W void getDerivKernels( OutputArray kx, OutputArray ky,
                                   int dx, int dy, int ksize,
                                   bool normalize = false, int ktype = CV_32F );

/** @brief Returns Gabor filter coefficients.

For more details about gabor filter equations and parameters, see: [Gabor
Filter](http://en.wikipedia.org/wiki/Gabor_filter).

@param ksize Size of the filter returned.
@param sigma Standard deviation of the gaussian envelope.
@param theta Orientation of the normal to the parallel stripes of a Gabor function.
@param lambd Wavelength of the sinusoidal factor.
@param gamma Spatial aspect ratio.
@param psi Phase offset.
@param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
 */
CV_EXPORTS_W Mat getGaborKernel( Size ksize, double sigma, double theta, double lambd,
                                 double gamma, double psi = CV_PI*0.5, int ktype = CV_64F );

//! returns "magic" border value for erosion and dilation. It is automatically transformed to Scalar::all(-DBL_MAX) for dilation.
static inline Scalar morphologyDefaultBorderValue() { return Scalar::all(DBL_MAX); }

/** @brief Returns a structuring element of the specified size and shape for morphological operations.

The function constructs and returns the structuring element that can be further passed to cv::erode,
cv::dilate or cv::morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as
the structuring element.

@param shape Element shape that could be one of cv::MorphShapes
@param ksize Size of the structuring element.
@param anchor Anchor position within the element. The default value \f$(-1, -1)\f$ means that the
anchor is at the center. Note that only the shape of a cross-shaped element depends on the anchor
position. In other cases the anchor just regulates how much the result of the morphological
operation is shifted.
 */
CV_EXPORTS_W Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));

/** @brief Blurs an image using the median filter.

The function smoothes an image using the median filter with the \f$\texttt{ksize} \times
\texttt{ksize}\f$ aperture. Each channel of a multi-channel image is processed independently.
In-place operation is supported.

@note The median filter uses BORDER_REPLICATE internally to cope with border pixels, see cv::BorderTypes

@param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be
CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.
@param dst destination array of the same size and type as src.
@param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...
@sa  bilateralFilter, blur, boxFilter, GaussianBlur
 */
CV_EXPORTS_W void medianBlur( InputArray src, OutputArray dst, int ksize );

/** @brief Blurs an image using a Gaussian filter.

The function convolves the source image with the specified Gaussian kernel. In-place filtering is
supported.

@param src input image; the image can have any number of channels, which are processed
independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst output image of the same size and type as src.
@param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
positive and odd. Or, they can be zero's and then they are computed from sigma.
@param sigmaX Gaussian kernel standard deviation in X direction.
@param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be
equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,
respectively (see cv::getGaussianKernel for details); to fully control the result regardless of
possible future modifications of all this semantics, it is recommended to specify all of ksize,
sigmaX, and sigmaY.
@param borderType pixel extrapolation method, see cv::BorderTypes

@sa  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur
 */
CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,
                                double sigmaX, double sigmaY = 0,
                                int borderType = BORDER_DEFAULT );

/** @brief Applies the bilateral filter to an image.

The function applies bilateral filtering to the input image, as described in
http://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
bilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is
very slow compared to most filters.

_Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (\<
10), the filter will not have much effect, whereas if they are large (\> 150), they will have a very
strong effect, making the image look "cartoonish".

_Filter size_: Large filters (d \> 5) are very slow, so it is recommended to use d=5 for real-time
applications, and perhaps d=9 for offline applications that need heavy noise filtering.

This filter does not work inplace.
@param src Source 8-bit or floating-point, 1-channel or 3-channel image.
@param dst Destination image of the same size and type as src .
@param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
it is computed from sigmaSpace.
@param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
farther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting
in larger areas of semi-equal color.
@param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
farther pixels will influence each other as long as their colors are close enough (see sigmaColor
). When d\>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is
proportional to sigmaSpace.
@param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
 */
CV_EXPORTS_W void bilateralFilter( InputArray src, OutputArray dst, int d,
                                   double sigmaColor, double sigmaSpace,
                                   int borderType = BORDER_DEFAULT );

/** @brief Blurs an image using the box filter.

The function smooths an image using the kernel:

\f[\texttt{K} =  \alpha \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1 \end{bmatrix}\f]

where

\f[\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\f]

Unnormalized box filter is useful for computing various integral characteristics over each pixel
neighborhood, such as covariance matrices of image derivatives (used in dense optical flow
algorithms, and so on). If you need to compute pixel sums over variable-size windows, use cv::integral.

@param src input image.
@param dst output image of the same size and type as src.
@param ddepth the output image depth (-1 to use src.depth()).
@param ksize blurring kernel size.
@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
center.
@param normalize flag, specifying whether the kernel is normalized by its area or not.
@param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
@sa  blur, bilateralFilter, GaussianBlur, medianBlur, integral
 */
CV_EXPORTS_W void boxFilter( InputArray src, OutputArray dst, int ddepth,
                             Size ksize, Point anchor = Point(-1,-1),
                             bool normalize = true,
                             int borderType = BORDER_DEFAULT );

/** @brief Calculates the normalized sum of squares of the pixel values overlapping the filter.

For every pixel \f$ (x, y) \f$ in the source image, the function calculates the sum of squares of those neighboring
pixel values which overlap the filter placed over the pixel \f$ (x, y) \f$.

The unnormalized square box filter can be useful in computing local image statistics such as the the local
variance and standard deviation around the neighborhood of a pixel.

@param _src input image
@param _dst output image of the same size and type as _src
@param ddepth the output image depth (-1 to use src.depth())
@param ksize kernel size
@param anchor kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at the kernel
center.
@param normalize flag, specifying whether the kernel is to be normalized by it's area or not.
@param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
@sa boxFilter
*/
CV_EXPORTS_W void sqrBoxFilter( InputArray _src, OutputArray _dst, int ddepth,
                                Size ksize, Point anchor = Point(-1, -1),
                                bool normalize = true,
                                int borderType = BORDER_DEFAULT );

/** @brief Blurs an image using the normalized box filter.

The function smooths an image using the kernel:

\f[\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}\f]

The call `blur(src, dst, ksize, anchor, borderType)` is equivalent to `boxFilter(src, dst, src.type(),
anchor, true, borderType)`.

@param src input image; it can have any number of channels, which are processed independently, but
the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst output image of the same size and type as src.
@param ksize blurring kernel size.
@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel
center.
@param borderType border mode used to extrapolate pixels outside of the image, see cv::BorderTypes
@sa  boxFilter, bilateralFilter, GaussianBlur, medianBlur
 */
CV_EXPORTS_W void blur( InputArray src, OutputArray dst,
                        Size ksize, Point anchor = Point(-1,-1),
                        int borderType = BORDER_DEFAULT );

/** @brief Convolves an image with the kernel.

The function applies an arbitrary linear filter to an image. In-place operation is supported. When
the aperture is partially outside the image, the function interpolates outlier pixel values
according to the specified border mode.

The function does actually compute correlation, not the convolution:

\f[\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' < \texttt{kernel.cols},}{0\leq y' < \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\f]

That is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip
the kernel using cv::flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -
anchor.y - 1)`.

The function uses the DFT-based algorithm in case of sufficiently large kernels (~`11 x 11` or
larger) and the direct algorithm for small kernels.

@param src input image.
@param dst output image of the same size and the same number of channels as src.
@param ddepth desired depth of the destination image, see @ref filter_depths "combinations"
@param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point
matrix; if you want to apply different kernels to different channels, split the image into
separate color planes using split and process them individually.
@param anchor anchor of the kernel that indicates the relative position of a filtered point within
the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor
is at the kernel center.
@param delta optional value added to the filtered pixels before storing them in dst.
@param borderType pixel extrapolation method, see cv::BorderTypes
@sa  sepFilter2D, dft, matchTemplate
 */
CV_EXPORTS_W void filter2D( InputArray src, OutputArray dst, int ddepth,
                            InputArray kernel, Point anchor = Point(-1,-1),
                            double delta = 0, int borderType = BORDER_DEFAULT );

/** @brief Applies a separable linear filter to an image.

The function applies a separable linear filter to the image. That is, first, every row of src is
filtered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D
kernel kernelY. The final result shifted by delta is stored in dst .

@param src Source image.
@param dst Destination image of the same size and the same number of channels as src .
@param ddepth Destination image depth, see @ref filter_depths "combinations"
@param kernelX Coefficients for filtering each row.
@param kernelY Coefficients for filtering each column.
@param anchor Anchor position within the kernel. The default value \f$(-1,-1)\f$ means that the anchor
is at the kernel center.
@param delta Value added to the filtered results before storing them.
@param borderType Pixel extrapolation method, see cv::BorderTypes
@sa  filter2D, Sobel, GaussianBlur, boxFilter, blur
 */
CV_EXPORTS_W void sepFilter2D( InputArray src, OutputArray dst, int ddepth,
                               InputArray kernelX, InputArray kernelY,
                               Point anchor = Point(-1,-1),
                               double delta = 0, int borderType = BORDER_DEFAULT );

/** @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

In all cases except one, the \f$\texttt{ksize} \times \texttt{ksize}\f$ separable kernel is used to
calculate the derivative. When \f$\texttt{ksize = 1}\f$, the \f$3 \times 1\f$ or \f$1 \times 3\f$
kernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first
or the second x- or y- derivatives.

There is also the special value `ksize = CV_SCHARR (-1)` that corresponds to the \f$3\times3\f$ Scharr
filter that may give more accurate results than the \f$3\times3\f$ Sobel. The Scharr aperture is

\f[\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\f]

for the x-derivative, or transposed for the y-derivative.

The function calculates an image derivative by convolving the image with the appropriate kernel:

\f[\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\f]

The Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less
resistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)
or ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first
case corresponds to a kernel of:

\f[\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\f]

The second case corresponds to a kernel of:

\f[\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\f]

@param src input image.
@param dst output image of the same size and the same number of channels as src .
@param ddepth output image depth, see @ref filter_depths "combinations"; in the case of
    8-bit input images it will result in truncated derivatives.
@param dx order of the derivative x.
@param dy order of the derivative y.
@param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
@param scale optional scale factor for the computed derivative values; by default, no scaling is
applied (see cv::getDerivKernels for details).
@param delta optional delta value that is added to the results prior to storing them in dst.
@param borderType pixel extrapolation method, see cv::BorderTypes
@sa  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar
 */
CV_EXPORTS_W void Sobel( InputArray src, OutputArray dst, int ddepth,
                         int dx, int dy, int ksize = 3,
                         double scale = 1, double delta = 0,
                         int borderType = BORDER_DEFAULT );

/** @brief Calculates the first order image derivative in both x and y using a Sobel operator

Equivalent to calling:

@code
Sobel( src, dx, CV_16SC1, 1, 0, 3 );
Sobel( src, dy, CV_16SC1, 0, 1, 3 );
@endcode

@param src input image.
@param dx output image with first-order derivative in x.
@param dy output image with first-order derivative in y.
@param ksize size of Sobel kernel. It must be 3.
@param borderType pixel extrapolation method, see cv::BorderTypes

@sa Sobel
 */

CV_EXPORTS_W void spatialGradient( InputArray src, OutputArray dx,
                                   OutputArray dy, int ksize = 3,
                                   int borderType = BORDER_DEFAULT );

/** @brief Calculates the first x- or y- image derivative using Scharr operator.

The function computes the first x- or y- spatial image derivative using the Scharr operator. The
call

\f[\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\f]

is equivalent to

\f[\texttt{Sobel(src, dst, ddepth, dx, dy, CV\_SCHARR, scale, delta, borderType)} .\f]

@param src input image.
@param dst output image of the same size and the same number of channels as src.
@param ddepth output image depth, see @ref filter_depths "combinations"
@param dx order of the derivative x.
@param dy order of the derivative y.
@param scale optional scale factor for the computed derivative values; by default, no scaling is
applied (see getDerivKernels for details).
@param delta optional delta value that is added to the results prior to storing them in dst.
@param borderType pixel extrapolation method, see cv::BorderTypes
@sa  cartToPolar
 */
CV_EXPORTS_W void Scharr( InputArray src, OutputArray dst, int ddepth,
                          int dx, int dy, double scale = 1, double delta = 0,
                          int borderType = BORDER_DEFAULT );

/** @example laplace.cpp
  An example using Laplace transformations for edge detection
*/

/** @brief Calculates the Laplacian of an image.

The function calculates the Laplacian of the source image by adding up the second x and y
derivatives calculated using the Sobel operator:

\f[\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\f]

This is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image
with the following \f$3 \times 3\f$ aperture:

\f[\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\f]

@param src Source image.
@param dst Destination image of the same size and the same number of channels as src .
@param ddepth Desired depth of the destination image.
@param ksize Aperture size used to compute the second-derivative filters. See getDerivKernels for
details. The size must be positive and odd.
@param scale Optional scale factor for the computed Laplacian values. By default, no scaling is
applied. See getDerivKernels for details.
@param delta Optional delta value that is added to the results prior to storing them in dst .
@param borderType Pixel extrapolation method, see cv::BorderTypes
@sa  Sobel, Scharr
 */
CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
                             int ksize = 1, double scale = 1, double delta = 0,
                             int borderType = BORDER_DEFAULT );

//! @} imgproc_filter

//! @addtogroup imgproc_feature
//! @{

/** @example edge.cpp
  An example on using the canny edge detector
*/

/** @brief Finds edges in an image using the Canny algorithm @cite Canny86 .

The function finds edges in the input image image and marks them in the output map edges using the
Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The
largest value is used to find initial segments of strong edges. See
<http://en.wikipedia.org/wiki/Canny_edge_detector>

@param image 8-bit input image.
@param edges output edge map; single channels 8-bit image, which has the same size as image .
@param threshold1 first threshold for the hysteresis procedure.
@param threshold2 second threshold for the hysteresis procedure.
@param apertureSize aperture size for the Sobel operator.
@param L2gradient a flag, indicating whether a more accurate \f$L_2\f$ norm
\f$=\sqrt{(dI/dx)^2 + (dI/dy)^2}\f$ should be used to calculate the image gradient magnitude (
L2gradient=true ), or whether the default \f$L_1\f$ norm \f$=|dI/dx|+|dI/dy|\f$ is enough (
L2gradient=false ).
 */
CV_EXPORTS_W void Canny( InputArray image, OutputArray edges,
                         double threshold1, double threshold2,
                         int apertureSize = 3, bool L2gradient = false );

/** \overload

Finds edges in an image using the Canny algorithm with custom image gradient.

@param dx 16-bit x derivative of input image (CV_16SC1 or CV_16SC3).
@param dy 16-bit y derivative of input image (same type as dx).
@param edges,threshold1,threshold2,L2gradient See cv::Canny
 */
CV_EXPORTS_W void Canny( InputArray dx, InputArray dy,
                         OutputArray edges,
                         double threshold1, double threshold2,
                         bool L2gradient = false );

/** @brief Calculates the minimal eigenvalue of gradient matrices for corner detection.

The function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal
eigenvalue of the covariance matrix of derivatives, that is, \f$\min(\lambda_1, \lambda_2)\f$ in terms
of the formulae in the cornerEigenValsAndVecs description.

@param src Input single-channel 8-bit or floating-point image.
@param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as
src .
@param blockSize Neighborhood size (see the details on cornerEigenValsAndVecs ).
@param ksize Aperture parameter for the Sobel operator.
@param borderType Pixel extrapolation method. See cv::BorderTypes.
 */
CV_EXPORTS_W void cornerMinEigenVal( InputArray src, OutputArray dst,
                                     int blockSize, int ksize = 3,
                                     int borderType = BORDER_DEFAULT );

/** @brief Harris corner detector.

The function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and
cornerEigenValsAndVecs , for each pixel \f$(x, y)\f$ it calculates a \f$2\times2\f$ gradient covariance
matrix \f$M^{(x,y)}\f$ over a \f$\texttt{blockSize} \times \texttt{blockSize}\f$ neighborhood. Then, it
computes the following characteristic:

\f[\texttt{dst} (x,y) =  \mathrm{det} M^{(x,y)} - k  \cdot \left ( \mathrm{tr} M^{(x,y)} \right )^2\f]

Corners in the image can be found as the local maxima of this response map.

@param src Input single-channel 8-bit or floating-point image.
@param dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same
size as src .
@param blockSize Neighborhood size (see the details on cornerEigenValsAndVecs ).
@param ksize Aperture parameter for the Sobel operator.
@param k Harris detector free parameter. See the formula below.
@param borderType Pixel extrapolation method. See cv::BorderTypes.
 */
CV_EXPORTS_W void cornerHarris( InputArray src, OutputArray dst, int blockSize,
                                int ksize, double k,
                                int borderType = BORDER_DEFAULT );

/** @brief Calculates eigenvalues and eigenvectors of image blocks for corner detection.

For every pixel \f$p\f$ , the function cornerEigenValsAndVecs considers a blockSize \f$\times\f$ blockSize
neighborhood \f$S(p)\f$ . It calculates the covariation matrix of derivatives over the neighborhood as:

\f[M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &  \sum _{S(p)}dI/dx dI/dy  \\ \sum _{S(p)}dI/dx dI/dy &  \sum _{S(p)}(dI/dy)^2 \end{bmatrix}\f]

where the derivatives are computed using the Sobel operator.

After that, it finds eigenvectors and eigenvalues of \f$M\f$ and stores them in the destination image as
\f$(\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)\f$ where

-   \f$\lambda_1, \lambda_2\f$ are the non-sorted eigenvalues of \f$M\f$
-   \f$x_1, y_1\f$ are the eigenvectors corresponding to \f$\lambda_1\f$
-   \f$x_2, y_2\f$ are the eigenvectors corresponding to \f$\lambda_2\f$

The output of the function can be used for robust edge or corner detection.

@param src Input single-channel 8-bit or floating-point image.
@param dst Image to store the results. It has the same size as src and the type CV_32FC(6) .
@param blockSize Neighborhood size (see details below).
@param ksize Aperture parameter for the Sobel operator.
@param borderType Pixel extrapolation method. See cv::BorderTypes.

@sa  cornerMinEigenVal, cornerHarris, preCornerDetect
 */
CV_EXPORTS_W void cornerEigenValsAndVecs( InputArray src, OutputArray dst,
                                          int blockSize, int ksize,
                                          int borderType = BORDER_DEFAULT );

/** @brief Calculates a feature map for corner detection.

The function calculates the complex spatial derivative-based function of the source image

\f[\texttt{dst} = (D_x  \texttt{src} )^2  \cdot D_{yy}  \texttt{src} + (D_y  \texttt{src} )^2  \cdot D_{xx}  \texttt{src} - 2 D_x  \texttt{src} \cdot D_y  \texttt{src} \cdot D_{xy}  \texttt{src}\f]

where \f$D_x\f$,\f$D_y\f$ are the first image derivatives, \f$D_{xx}\f$,\f$D_{yy}\f$ are the second image
derivatives, and \f$D_{xy}\f$ is the mixed derivative.

The corners can be found as local maximums of the functions, as shown below:
@code
    Mat corners, dilated_corners;
    preCornerDetect(image, corners, 3);
    // dilation with 3x3 rectangular structuring element
    dilate(corners, dilated_corners, Mat(), 1);
    Mat corner_mask = corners == dilated_corners;
@endcode

@param src Source single-channel 8-bit of floating-point image.
@param dst Output image that has the type CV_32F and the same size as src .
@param ksize %Aperture size of the Sobel .
@param borderType Pixel extrapolation method. See cv::BorderTypes.
 */
CV_EXPORTS_W void preCornerDetect( InputArray src, OutputArray dst, int ksize,
                                   int borderType = BORDER_DEFAULT );

/** @brief Refines the corner locations.

The function iterates to find the sub-pixel accurate location of corners or radial saddle points, as
shown on the figure below.

![image](pics/cornersubpix.png)

Sub-pixel accurate corner locator is based on the observation that every vector from the center \f$q\f$
to a point \f$p\f$ located within a neighborhood of \f$q\f$ is orthogonal to the image gradient at \f$p\f$
subject to image and measurement noise. Consider the expression:

\f[\epsilon _i = {DI_{p_i}}^T  \cdot (q - p_i)\f]

where \f${DI_{p_i}}\f$ is an image gradient at one of the points \f$p_i\f$ in a neighborhood of \f$q\f$ . The
value of \f$q\f$ is to be found so that \f$\epsilon_i\f$ is minimized. A system of equations may be set up
with \f$\epsilon_i\f$ set to zero:

\f[\sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T) -  \sum _i(DI_{p_i}  \cdot {DI_{p_i}}^T  \cdot p_i)\f]

where the gradients are summed within a neighborhood ("search window") of \f$q\f$ . Calling the first
gradient term \f$G\f$ and the second gradient term \f$b\f$ gives:

\f[q = G^{-1}  \cdot b\f]

The algorithm sets the center of the neighborhood window at this new center \f$q\f$ and then iterates
until the center stays within a set threshold.

@param image Input image.
@param corners Initial coordinates of the input corners and refined coordinates provided for
output.
@param winSize Half of the side length of the search window. For example, if winSize=Size(5,5) ,
then a \f$5*2+1 \times 5*2+1 = 11 \times 11\f$ search window is used.
@param zeroZone Half of the size of the dead region in the middle of the search zone over which
the summation in the formula below is not done. It is used sometimes to avoid possible
singularities of the autocorrelation matrix. The value of (-1,-1) indicates that there is no such
a size.
@param criteria Criteria for termination of the iterative process of corner refinement. That is,
the process of corner position refinement stops either after criteria.maxCount iterations or when
the corner position moves by less than criteria.epsilon on some iteration.
 */
CV_EXPORTS_W void cornerSubPix( InputArray image, InputOutputArray corners,
                                Size winSize, Size zeroZone,
                                TermCriteria criteria );

/** @brief Determines strong corners on an image.

The function finds the most prominent corners in the image or in the specified image region, as
described in @cite Shi94

-   Function calculates the corner quality measure at every source image pixel using the
    cornerMinEigenVal or cornerHarris .
-   Function performs a non-maximum suppression (the local maximums in *3 x 3* neighborhood are
    retained).
-   The corners with the minimal eigenvalue less than
    \f$\texttt{qualityLevel} \cdot \max_{x,y} qualityMeasureMap(x,y)\f$ are rejected.
-   The remaining corners are sorted by the quality measure in the descending order.
-   Function throws away each corner for which there is a stronger corner at a distance less than
    maxDistance.

The function can be used to initialize a point-based tracker of an object.

@note If the function is called with different values A and B of the parameter qualityLevel , and
A \> B, the vector of returned corners with qualityLevel=A will be the prefix of the output vector
with qualityLevel=B .

@param image Input 8-bit or floating-point 32-bit, single-channel image.
@param corners Output vector of detected corners.
@param maxCorners Maximum number of corners to return. If there are more corners than are found,
the strongest of them is returned. `maxCorners <= 0` implies that no limit on the maximum is set
and all detected corners are returned.
@param qualityLevel Parameter characterizing the minimal accepted quality of image corners. The
parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
(see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the
quality measure less than the product are rejected. For example, if the best corner has the
quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure
less than 15 are rejected.
@param minDistance Minimum possible Euclidean distance between the returned corners.
@param mask Optional region of interest. If the image is not empty (it needs to have the type
CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
@param blockSize Size of an average block for computing a derivative covariation matrix over each
pixel neighborhood. See cornerEigenValsAndVecs .
@param useHarrisDetector Parameter indicating whether to use a Harris detector (see cornerHarris)
or cornerMinEigenVal.
@param k Free parameter of the Harris detector.

@sa  cornerMinEigenVal, cornerHarris, calcOpticalFlowPyrLK, estimateRigidTransform,
 */
CV_EXPORTS_W void goodFeaturesToTrack( InputArray image, OutputArray corners,
                                     int maxCorners, double qualityLevel, double minDistance,
                                     InputArray mask = noArray(), int blockSize = 3,
                                     bool useHarrisDetector = false, double k = 0.04 );

/** @example houghlines.cpp
An example using the Hough line detector
*/

/** @brief Finds lines in a binary image using the standard Hough transform.

The function implements the standard or standard multi-scale Hough transform algorithm for line
detection. See <http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm> for a good explanation of Hough
transform.

@param image 8-bit, single-channel binary source image. The image may be modified by the function.
@param lines Output vector of lines. Each line is represented by a two-element vector
\f$(\rho, \theta)\f$ . \f$\rho\f$ is the distance from the coordinate origin \f$(0,0)\f$ (top-left corner of
the image). \f$\theta\f$ is the line rotation angle in radians (
\f$0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\f$ ).
@param rho Distance resolution of the accumulator in pixels.
@param theta Angle resolution of the accumulator in radians.
@param threshold Accumulator threshold parameter. Only those lines are returned that get enough
votes ( \f$>\texttt{threshold}\f$ ).
@param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .
The coarse accumulator distance resolution is rho and the accurate accumulator resolution is
rho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these
parameters should be positive.
@param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.
@param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.
Must fall between 0 and max_theta.
@param max_theta For standard and multi-scale Hough transform, maximum angle to check for lines.
Must fall between min_theta and CV_PI.
 */
CV_EXPORTS_W void HoughLines( InputArray image, OutputArray lines,
                              double rho, double theta, int threshold,
                              double srn = 0, double stn = 0,
                              double min_theta = 0, double max_theta = CV_PI );

/** @brief Finds line segments in a binary image using the probabilistic Hough transform.

The function implements the probabilistic Hough transform algorithm for line detection, described
in @cite Matas00

See the line detection example below:

@code
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>

    using namespace cv;
    using namespace std;

    int main(int argc, char** argv)
    {
        Mat src, dst, color_dst;
        if( argc != 2 || !(src=imread(argv[1], 0)).data)
            return -1;

        Canny( src, dst, 50, 200, 3 );
        cvtColor( dst, color_dst, COLOR_GRAY2BGR );

    #if 0
        vector<Vec2f> lines;
        HoughLines( dst, lines, 1, CV_PI/180, 100 );

        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0];
            float theta = lines[i][1];
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            Point pt1(cvRound(x0 + 1000*(-b)),
                      cvRound(y0 + 1000*(a)));
            Point pt2(cvRound(x0 - 1000*(-b)),
                      cvRound(y0 - 1000*(a)));
            line( color_dst, pt1, pt2, Scalar(0,0,255), 3, 8 );
        }
    #else
        vector<Vec4i> lines;
        HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, 10 );
        for( size_t i = 0; i < lines.size(); i++ )
        {
            line( color_dst, Point(lines[i][0], lines[i][1]),
                Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
        }
    #endif
        namedWindow( "Source", 1 );
        imshow( "Source", src );

        namedWindow( "Detected Lines", 1 );
        imshow( "Detected Lines", color_dst );

        waitKey(0);
        return 0;
    }
@endcode
This is a sample picture the function parameters have been tuned for:

![image](pics/building.jpg)

And this is the output of the above program in case of the probabilistic Hough transform:

![image](pics/houghp.png)

@param image 8-bit, single-channel binary source image. The image may be modified by the function.
@param lines Output vector of lines. Each line is represented by a 4-element vector
\f$(x_1, y_1, x_2, y_2)\f$ , where \f$(x_1,y_1)\f$ and \f$(x_2, y_2)\f$ are the ending points of each detected
line segment.
@param rho Distance resolution of the accumulator in pixels.
@param theta Angle resolution of the accumulator in radians.
@param threshold Accumulator threshold parameter. Only those lines are returned that get enough
votes ( \f$>\texttt{threshold}\f$ ).
@param minLineLength Minimum line length. Line segments shorter than that are rejected.
@param maxLineGap Maximum allowed gap between points on the same line to link them.

@sa LineSegmentDetector
 */
CV_EXPORTS_W void HoughLinesP( InputArray image, OutputArray lines,
                               double rho, double theta, int threshold,
                               double minLineLength = 0, double maxLineGap = 0 );

/** @example houghcircles.cpp
An example using the Hough circle detector
*/

/** @brief Finds circles in a grayscale image using the Hough transform.

The function finds circles in a grayscale image using a modification of the Hough transform.

Example: :
@code
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>
    #include <math.h>

    using namespace cv;
    using namespace std;

    int main(int argc, char** argv)
    {
        Mat img, gray;
        if( argc != 2 || !(img=imread(argv[1], 1)).data)
            return -1;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        // smooth it, otherwise a lot of false circles may be detected
        GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
        vector<Vec3f> circles;
        HoughCircles(gray, circles, HOUGH_GRADIENT,
                     2, gray.rows/4, 200, 100 );
        for( size_t i = 0; i < circles.size(); i++ )
        {
             Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
             int radius = cvRound(circles[i][2]);
             // draw the circle center
             circle( img, center, 3, Scalar(0,255,0), -1, 8, 0 );
             // draw the circle outline
             circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }
        namedWindow( "circles", 1 );
        imshow( "circles", img );

        waitKey(0);
        return 0;
    }
@endcode

@note Usually the function detects the centers of circles well. However, it may fail to find correct
radii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if
you know it. Or, you may ignore the returned radius, use only the center, and find the correct
radius using an additional procedure.

@param image 8-bit, single-channel, grayscale input image.
@param circles Output vector of found circles. Each vector is encoded as a 3-element
floating-point vector \f$(x, y, radius)\f$ .
@param method Detection method, see cv::HoughModes. Currently, the only implemented method is HOUGH_GRADIENT
@param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if
dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has
half as big width and height.
@param minDist Minimum distance between the centers of the detected circles. If the parameter is
too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is
too large, some circles may be missed.
@param param1 First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher
threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
@param param2 Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the
accumulator threshold for the circle centers at the detection stage. The smaller it is, the more
false circles may be detected. Circles, corresponding to the larger accumulator values, will be
returned first.
@param minRadius Minimum circle radius.
@param maxRadius Maximum circle radius.

@sa fitEllipse, minEnclosingCircle
 */
CV_EXPORTS_W void HoughCircles( InputArray image, OutputArray circles,
                               int method, double dp, double minDist,
                               double param1 = 100, double param2 = 100,
                               int minRadius = 0, int maxRadius = 0 );

//! @} imgproc_feature

//! @addtogroup imgproc_filter
//! @{

/** @example morphology2.cpp
  An example using the morphological operations
*/

/** @brief Erodes an image by using a specific structuring element.

The function erodes the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the minimum is taken:

\f[\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In
case of multi-channel images, each channel is processed independently.

@param src input image; the number of channels can be arbitrary, but the depth should be one of
CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst output image of the same size and type as src.
@param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular
structuring element is used. Kernel can be created using getStructuringElement.
@param anchor position of the anchor within the element; default value (-1, -1) means that the
anchor is at the element center.
@param iterations number of times erosion is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  dilate, morphologyEx, getStructuringElement
 */
CV_EXPORTS_W void erode( InputArray src, OutputArray dst, InputArray kernel,
                         Point anchor = Point(-1,-1), int iterations = 1,
                         int borderType = BORDER_CONSTANT,
                         const Scalar& borderValue = morphologyDefaultBorderValue() );

/** @brief Dilates an image by using a specific structuring element.

The function dilates the source image using the specified structuring element that determines the
shape of a pixel neighborhood over which the maximum is taken:
\f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]

The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In
case of multi-channel images, each channel is processed independently.

@param src input image; the number of channels can be arbitrary, but the depth should be one of
CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst output image of the same size and type as src\`.
@param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular
structuring element is used. Kernel can be created using getStructuringElement
@param anchor position of the anchor within the element; default value (-1, -1) means that the
anchor is at the element center.
@param iterations number of times dilation is applied.
@param borderType pixel extrapolation method, see cv::BorderTypes
@param borderValue border value in case of a constant border
@sa  erode, morphologyEx, getStructuringElement
 */
CV_EXPORTS_W void dilate( InputArray src, OutputArray dst, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          int borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue() );

/** @brief Performs advanced morphological transformations.

The function morphologyEx can perform advanced morphological transformations using an erosion and dilation as
basic operations.

Any of the operations can be done in-place. In case of multi-channel images, each channel is
processed independently.

@param src Source image. The number of channels can be arbitrary. The depth should be one of
CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
@param dst Destination image of the same size and type as source image.
@param op Type of a morphological operation, see cv::MorphTypes
@param kernel Structuring element. It can be created using cv::getStructuringElement.
@param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
kernel center.
@param iterations Number of times erosion and dilation are applied.
@param borderType Pixel extrapolation method, see cv::BorderTypes
@param borderValue Border value in case of a constant border. The default value has a special
meaning.
@sa  dilate, erode, getStructuringElement
@note The number of iterations is the number of times erosion or dilatation operation will be applied.
For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
successively: erode -> erode -> dilate -> dilate (and not erode -> dilate -> erode -> dilate).
 */
CV_EXPORTS_W void morphologyEx( InputArray src, OutputArray dst,
                                int op, InputArray kernel,
                                Point anchor = Point(-1,-1), int iterations = 1,
                                int borderType = BORDER_CONSTANT,
                                const Scalar& borderValue = morphologyDefaultBorderValue() );

//! @} imgproc_filter

//! @addtogroup imgproc_transform
//! @{

/** @brief Resizes an image.

The function resize resizes the image src down to or up to the specified size. Note that the
initial dst type or size are not taken into account. Instead, the size and type are derived from
the `src`,`dsize`,`fx`, and `fy`. If you want to resize src so that it fits the pre-created dst,
you may call the function as follows:
@code
    // explicitly specify dsize=dst.size(); fx and fy will be computed from that.
    resize(src, dst, dst.size(), 0, 0, interpolation);
@endcode
If you want to decimate the image by factor of 2 in each direction, you can call the function this
way:
@code
    // specify fx and fy and let the function compute the destination image size.
    resize(src, dst, Size(), 0.5, 0.5, interpolation);
@endcode
To shrink an image, it will generally look best with cv::INTER_AREA interpolation, whereas to
enlarge an image, it will generally look best with cv::INTER_CUBIC (slow) or cv::INTER_LINEAR
(faster but still looks OK).

@param src input image.
@param dst output image; it has the size dsize (when it is non-zero) or the size computed from
src.size(), fx, and fy; the type of dst is the same as of src.
@param dsize output image size; if it equals zero, it is computed as:
 \f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]
 Either dsize or both fx and fy must be non-zero.
@param fx scale factor along the horizontal axis; when it equals 0, it is computed as
\f[\texttt{(double)dsize.width/src.cols}\f]
@param fy scale factor along the vertical axis; when it equals 0, it is computed as
\f[\texttt{(double)dsize.height/src.rows}\f]
@param interpolation interpolation method, see cv::InterpolationFlags

@sa  warpAffine, warpPerspective, remap
 */
CV_EXPORTS_W void resize( InputArray src, OutputArray dst,
                          Size dsize, double fx = 0, double fy = 0,
                          int interpolation = INTER_LINEAR );

/** @brief Applies an affine transformation to an image.

The function warpAffine transforms the source image using the specified matrix:

\f[\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\f]

when the flag WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted
with cv::invertAffineTransform and then put in the formula above instead of M. The function cannot
operate in-place.

@param src input image.
@param dst output image that has the size dsize and the same type as src .
@param M \f$2\times 3\f$ transformation matrix.
@param dsize size of the output image.
@param flags combination of interpolation methods (see cv::InterpolationFlags) and the optional
flag WARP_INVERSE_MAP that means that M is the inverse transformation (
\f$\texttt{dst}\rightarrow\texttt{src}\f$ ).
@param borderMode pixel extrapolation method (see cv::BorderTypes); when
borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to
the "outliers" in the source image are not modified by the function.
@param borderValue value used in case of a constant border; by default, it is 0.

@sa  warpPerspective, resize, remap, getRectSubPix, transform
 */
CV_EXPORTS_W void warpAffine( InputArray src, OutputArray dst,
                              InputArray M, Size dsize,
                              int flags = INTER_LINEAR,
                              int borderMode = BORDER_CONSTANT,
                              const Scalar& borderValue = Scalar());

/** @brief Applies a perspective transformation to an image.

The function warpPerspective transforms the source image using the specified matrix:

\f[\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,
     \frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\f]

when the flag WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert
and then put in the formula above instead of M. The function cannot operate in-place.

@param src input image.
@param dst output image that has the size dsize and the same type as src .
@param M \f$3\times 3\f$ transformation matrix.
@param dsize size of the output image.
@param flags combination of interpolation methods (INTER_LINEAR or INTER_NEAREST) and the
optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation (
\f$\texttt{dst}\rightarrow\texttt{src}\f$ ).
@param borderMode pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).
@param borderValue value used in case of a constant border; by default, it equals 0.

@sa  warpAffine, resize, remap, getRectSubPix, perspectiveTransform
 */
CV_EXPORTS_W void warpPerspective( InputArray src, OutputArray dst,
                                   InputArray M, Size dsize,
                                   int flags = INTER_LINEAR,
                                   int borderMode = BORDER_CONSTANT,
                                   const Scalar& borderValue = Scalar());

/** @brief Applies a generic geometrical transformation to an image.

The function remap transforms the source image using the specified map:

\f[\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\f]

where values of pixels with non-integer coordinates are computed using one of available
interpolation methods. \f$map_x\f$ and \f$map_y\f$ can be encoded as separate floating-point maps
in \f$map_1\f$ and \f$map_2\f$ respectively, or interleaved floating-point maps of \f$(x,y)\f$ in
\f$map_1\f$, or fixed-point maps created by using convertMaps. The reason you might want to
convert from floating to fixed-point representations of a map is that they can yield much faster
(\~2x) remapping operations. In the converted case, \f$map_1\f$ contains pairs (cvFloor(x),
cvFloor(y)) and \f$map_2\f$ contains indices in a table of interpolation coefficients.

This function cannot operate in-place.

@param src Source image.
@param dst Destination image. It has the same size as map1 and the same type as src .
@param map1 The first map of either (x,y) points or just x values having the type CV_16SC2 ,
CV_32FC1, or CV_32FC2. See convertMaps for details on converting a floating point
representation to fixed-point for speed.
@param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map
if map1 is (x,y) points), respectively.
@param interpolation Interpolation method (see cv::InterpolationFlags). The method INTER_AREA is
not supported by this function.
@param borderMode Pixel extrapolation method (see cv::BorderTypes). When
borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image that
corresponds to the "outliers" in the source image are not modified by the function.
@param borderValue Value used in case of a constant border. By default, it is 0.
@note
Due to current implementaion limitations the size of an input and output images should be less than 32767x32767.
 */
CV_EXPORTS_W void remap( InputArray src, OutputArray dst,
                         InputArray map1, InputArray map2,
                         int interpolation, int borderMode = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar());

/** @brief Converts image transformation maps from one representation to another.

The function converts a pair of maps for remap from one representation to another. The following
options ( (map1.type(), map2.type()) \f$\rightarrow\f$ (dstmap1.type(), dstmap2.type()) ) are
supported:

- \f$\texttt{(CV_32FC1, CV_32FC1)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}\f$. This is the
most frequently used conversion operation, in which the original floating-point maps (see remap )
are converted to a more compact and much faster fixed-point representation. The first output array
contains the rounded coordinates and the second array (created only when nninterpolation=false )
contains indices in the interpolation tables.

- \f$\texttt{(CV_32FC2)} \rightarrow \texttt{(CV_16SC2, CV_16UC1)}\f$. The same as above but
the original maps are stored in one 2-channel matrix.

- Reverse conversion. Obviously, the reconstructed floating-point maps will not be exactly the same
as the originals.

@param map1 The first input map of type CV_16SC2, CV_32FC1, or CV_32FC2 .
@param map2 The second input map of type CV_16UC1, CV_32FC1, or none (empty matrix),
respectively.
@param dstmap1 The first output map that has the type dstmap1type and the same size as src .
@param dstmap2 The second output map.
@param dstmap1type Type of the first output map that should be CV_16SC2, CV_32FC1, or
CV_32FC2 .
@param nninterpolation Flag indicating whether the fixed-point maps are used for the
nearest-neighbor or for a more complex interpolation.

@sa  remap, undistort, initUndistortRectifyMap
 */
CV_EXPORTS_W void convertMaps( InputArray map1, InputArray map2,
                               OutputArray dstmap1, OutputArray dstmap2,
                               int dstmap1type, bool nninterpolation = false );

/** @brief Calculates an affine matrix of 2D rotation.

The function calculates the following matrix:

\f[\begin{bmatrix} \alpha &  \beta & (1- \alpha )  \cdot \texttt{center.x} -  \beta \cdot \texttt{center.y} \\ - \beta &  \alpha &  \beta \cdot \texttt{center.x} + (1- \alpha )  \cdot \texttt{center.y} \end{bmatrix}\f]

where

\f[\begin{array}{l} \alpha =  \texttt{scale} \cdot \cos \texttt{angle} , \\ \beta =  \texttt{scale} \cdot \sin \texttt{angle} \end{array}\f]

The transformation maps the rotation center to itself. If this is not the target, adjust the shift.

@param center Center of the rotation in the source image.
@param angle Rotation angle in degrees. Positive values mean counter-clockwise rotation (the
coordinate origin is assumed to be the top-left corner).
@param scale Isotropic scale factor.

@sa  getAffineTransform, warpAffine, transform
 */
CV_EXPORTS_W Mat getRotationMatrix2D( Point2f center, double angle, double scale );

//! returns 3x3 perspective transformation for the corresponding 4 point pairs.
CV_EXPORTS Mat getPerspectiveTransform( const Point2f src[], const Point2f dst[] );

/** @brief Calculates an affine transform from three pairs of the corresponding points.

The function calculates the \f$2 \times 3\f$ matrix of an affine transform so that:

\f[\begin{bmatrix} x'_i \\ y'_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}\f]

where

\f[dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2\f]

@param src Coordinates of triangle vertices in the source image.
@param dst Coordinates of the corresponding triangle vertices in the destination image.

@sa  warpAffine, transform
 */
CV_EXPORTS Mat getAffineTransform( const Point2f src[], const Point2f dst[] );

/** @brief Inverts an affine transformation.

The function computes an inverse affine transformation represented by \f$2 \times 3\f$ matrix M:

\f[\begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}\f]

The result is also a \f$2 \times 3\f$ matrix of the same type as M.

@param M Original affine transformation.
@param iM Output reverse affine transformation.
 */
CV_EXPORTS_W void invertAffineTransform( InputArray M, OutputArray iM );

/** @brief Calculates a perspective transform from four pairs of the corresponding points.

The function calculates the \f$3 \times 3\f$ matrix of a perspective transform so that:

\f[\begin{bmatrix} t_i x'_i \\ t_i y'_i \\ t_i \end{bmatrix} = \texttt{map_matrix} \cdot \begin{bmatrix} x_i \\ y_i \\ 1 \end{bmatrix}\f]

where

\f[dst(i)=(x'_i,y'_i), src(i)=(x_i, y_i), i=0,1,2,3\f]

@param src Coordinates of quadrangle vertices in the source image.
@param dst Coordinates of the corresponding quadrangle vertices in the destination image.

@sa  findHomography, warpPerspective, perspectiveTransform
 */
CV_EXPORTS_W Mat getPerspectiveTransform( InputArray src, InputArray dst );

CV_EXPORTS_W Mat getAffineTransform( InputArray src, InputArray dst );

/** @brief Retrieves a pixel rectangle from an image with sub-pixel accuracy.

The function getRectSubPix extracts pixels from src:

\f[dst(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)\f]

where the values of the pixels at non-integer coordinates are retrieved using bilinear
interpolation. Every channel of multi-channel images is processed independently. While the center of
the rectangle must be inside the image, parts of the rectangle may be outside. In this case, the
replication border mode (see cv::BorderTypes) is used to extrapolate the pixel values outside of
the image.

@param image Source image.
@param patchSize Size of the extracted patch.
@param center Floating point coordinates of the center of the extracted rectangle within the
source image. The center must be inside the image.
@param patch Extracted patch that has the size patchSize and the same number of channels as src .
@param patchType Depth of the extracted pixels. By default, they have the same depth as src .

@sa  warpAffine, warpPerspective
 */
CV_EXPORTS_W void getRectSubPix( InputArray image, Size patchSize,
                                 Point2f center, OutputArray patch, int patchType = -1 );

/** @example polar_transforms.cpp
An example using the cv::linearPolar and cv::logPolar operations
*/

/** @brief Remaps an image to semilog-polar coordinates space.

Transform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image"):
\f[\begin{array}{l}
  dst( \rho , \phi ) = src(x,y) \\
  dst.size() \leftarrow src.size()
\end{array}\f]

where
\f[\begin{array}{l}
  I = (dx,dy) = (x - center.x,y - center.y) \\
  \rho = M \cdot log_e(\texttt{magnitude} (I)) ,\\
  \phi = Ky \cdot \texttt{angle} (I)_{0..360 deg} \\
\end{array}\f]

and
\f[\begin{array}{l}
  M = src.cols / log_e(maxRadius) \\
  Ky = src.rows / 360 \\
\end{array}\f]

The function emulates the human "foveal" vision and can be used for fast scale and
rotation-invariant template matching, for object tracking and so forth.
@param src Source image
@param dst Destination image. It will have same size and type as src.
@param center The transformation center; where the output precision is maximal
@param M Magnitude scale parameter. It determines the radius of the bounding circle to transform too.
@param flags A combination of interpolation methods, see cv::InterpolationFlags

@note
-   The function can not operate in-place.
-   To calculate magnitude and angle in degrees @ref cv::cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.
*/
CV_EXPORTS_W void logPolar( InputArray src, OutputArray dst,
                            Point2f center, double M, int flags );

/** @brief Remaps an image to polar coordinates space.

@anchor polar_remaps_reference_image
![Polar remaps reference](pics/polar_remap_doc.png)

Transform the source image using the following transformation:
\f[\begin{array}{l}
  dst( \rho , \phi ) = src(x,y) \\
  dst.size() \leftarrow src.size()
\end{array}\f]

where
\f[\begin{array}{l}
  I = (dx,dy) = (x - center.x,y - center.y) \\
  \rho = Kx \cdot \texttt{magnitude} (I) ,\\
  \phi = Ky \cdot \texttt{angle} (I)_{0..360 deg}
\end{array}\f]

and
\f[\begin{array}{l}
  Kx = src.cols / maxRadius \\
  Ky = src.rows / 360
\end{array}\f]


@param src Source image
@param dst Destination image. It will have same size and type as src.
@param center The transformation center;
@param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.
@param flags A combination of interpolation methods, see cv::InterpolationFlags

@note
-   The function can not operate in-place.
-   To calculate magnitude and angle in degrees @ref cv::cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.

*/
CV_EXPORTS_W void linearPolar( InputArray src, OutputArray dst,
                               Point2f center, double maxRadius, int flags );

//! @} imgproc_transform

//! @addtogroup imgproc_misc
//! @{

/** @overload */
CV_EXPORTS_W void integral( InputArray src, OutputArray sum, int sdepth = -1 );

/** @overload */
CV_EXPORTS_AS(integral2) void integral( InputArray src, OutputArray sum,
                                        OutputArray sqsum, int sdepth = -1, int sqdepth = -1 );

/** @brief Calculates the integral of an image.

The function calculates one or more integral images for the source image as follows:

\f[\texttt{sum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)\f]

\f[\texttt{sqsum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)^2\f]

\f[\texttt{tilted} (X,Y) =  \sum _{y<Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)\f]

Using these integral images, you can calculate sum, mean, and standard deviation over a specific
up-right or rotated rectangular region of the image in a constant time, for example:

\f[\sum _{x_1 \leq x < x_2,  \, y_1  \leq y < y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)\f]

It makes possible to do a fast blurring or fast block correlation with a variable window size, for
example. In case of multi-channel images, sums for each channel are accumulated independently.

As a practical example, the next figure shows the calculation of the integral of a straight
rectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the
original image are shown, as well as the relative pixels in the integral images sum and tilted .

![integral calculation example](pics/integral.png)

@param src input image as \f$W \times H\f$, 8-bit or floating-point (32f or 64f).
@param sum integral image as \f$(W+1)\times (H+1)\f$ , 32-bit integer or floating-point (32f or 64f).
@param sqsum integral image for squared pixel values; it is \f$(W+1)\times (H+1)\f$, double-precision
floating-point (64f) array.
@param tilted integral for the image rotated by 45 degrees; it is \f$(W+1)\times (H+1)\f$ array with
the same data type as sum.
@param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or
CV_64F.
@param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.
 */
CV_EXPORTS_AS(integral3) void integral( InputArray src, OutputArray sum,
                                        OutputArray sqsum, OutputArray tilted,
                                        int sdepth = -1, int sqdepth = -1 );

//! @} imgproc_misc

//! @addtogroup imgproc_motion
//! @{

/** @brief Adds an image to the accumulator.

The function adds src or some of its elements to dst :

\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

The function supports multi-channel images. Each channel is processed independently.

The functions accumulate\* can be used, for example, to collect statistics of a scene background
viewed by a still camera and for the further foreground-background segmentation.

@param src Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.
@param dst %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.
@param mask Optional operation mask.

@sa  accumulateSquare, accumulateProduct, accumulateWeighted
 */
CV_EXPORTS_W void accumulate( InputArray src, InputOutputArray dst,
                              InputArray mask = noArray() );

/** @brief Adds the square of a source image to the accumulator.

The function adds the input image src or its selected region, raised to a power of 2, to the
accumulator dst :

\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

The function supports multi-channel images. Each channel is processed independently.

@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
floating-point.
@param mask Optional operation mask.

@sa  accumulateSquare, accumulateProduct, accumulateWeighted
 */
CV_EXPORTS_W void accumulateSquare( InputArray src, InputOutputArray dst,
                                    InputArray mask = noArray() );

/** @brief Adds the per-element product of two input images to the accumulator.

The function adds the product of two images or their selected regions to the accumulator dst :

\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

The function supports multi-channel images. Each channel is processed independently.

@param src1 First input image, 1- or 3-channel, 8-bit or 32-bit floating point.
@param src2 Second input image of the same type and the same size as src1 .
@param dst %Accumulator with the same number of channels as input images, 32-bit or 64-bit
floating-point.
@param mask Optional operation mask.

@sa  accumulate, accumulateSquare, accumulateWeighted
 */
CV_EXPORTS_W void accumulateProduct( InputArray src1, InputArray src2,
                                     InputOutputArray dst, InputArray mask=noArray() );

/** @brief Updates a running average.

The function calculates the weighted sum of the input image src and the accumulator dst so that dst
becomes a running average of a frame sequence:

\f[\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]

That is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images).
The function supports multi-channel images. Each channel is processed independently.

@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.
@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit
floating-point.
@param alpha Weight of the input image.
@param mask Optional operation mask.

@sa  accumulate, accumulateSquare, accumulateProduct
 */
CV_EXPORTS_W void accumulateWeighted( InputArray src, InputOutputArray dst,
                                      double alpha, InputArray mask = noArray() );

/** @brief The function is used to detect translational shifts that occur between two images.

The operation takes advantage of the Fourier shift theorem for detecting the translational shift in
the frequency domain. It can be used for fast image registration as well as motion estimation. For
more information please see <http://en.wikipedia.org/wiki/Phase_correlation>

Calculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed
with getOptimalDFTSize.

The function performs the following equations:
- First it applies a Hanning window (see <http://en.wikipedia.org/wiki/Hann_function>) to each
image to remove possible edge effects. This window is cached until the array size changes to speed
up processing time.
- Next it computes the forward DFTs of each source array:
\f[\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}\f]
where \f$\mathcal{F}\f$ is the forward DFT.
- It then computes the cross-power spectrum of each frequency domain array:
\f[R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}\f]
- Next the cross-correlation is converted back into the time domain via the inverse DFT:
\f[r = \mathcal{F}^{-1}\{R\}\f]
- Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to
achieve sub-pixel accuracy.
\f[(\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}\f]
- If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5
centroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single
peak) and will be smaller when there are multiple peaks.

@param src1 Source floating point array (CV_32FC1 or CV_64FC1)
@param src2 Source floating point array (CV_32FC1 or CV_64FC1)
@param window Floating point array with windowing coefficients to reduce edge effects (optional).
@param response Signal power within the 5x5 centroid around the peak, between 0 and 1 (optional).
@returns detected phase shift (sub-pixel) between the two arrays.

@sa dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow
 */
CV_EXPORTS_W Point2d phaseCorrelate(InputArray src1, InputArray src2,
                                    InputArray window = noArray(), CV_OUT double* response = 0);

/** @brief This function computes a Hanning window coefficients in two dimensions.

See (http://en.wikipedia.org/wiki/Hann_function) and (http://en.wikipedia.org/wiki/Window_function)
for more information.

An example is shown below:
@code
    // create hanning window of size 100x100 and type CV_32F
    Mat hann;
    createHanningWindow(hann, Size(100, 100), CV_32F);
@endcode
@param dst Destination array to place Hann coefficients in
@param winSize The window size specifications
@param type Created array type
 */
CV_EXPORTS_W void createHanningWindow(OutputArray dst, Size winSize, int type);

//! @} imgproc_motion

//! @addtogroup imgproc_misc
//! @{

/** @brief Applies a fixed-level threshold to each array element.

The function applies fixed-level thresholding to a multiple-channel array. The function is typically
used to get a bi-level (binary) image out of a grayscale image ( cv::compare could be also used for
this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
values. There are several types of thresholding supported by the function. They are determined by
type parameter.

Also, the special values cv::THRESH_OTSU or cv::THRESH_TRIANGLE may be combined with one of the
above values. In these cases, the function determines the optimal threshold value using the Otsu's
or Triangle algorithm and uses it instead of the specified thresh . The function returns the
computed threshold value. Currently, the Otsu's and Triangle methods are implemented only for 8-bit
images.

@note Input image should be single channel only in case of CV_THRESH_OTSU or CV_THRESH_TRIANGLE flags

@param src input array (multiple-channel, 8-bit or 32-bit floating point).
@param dst output array of the same size  and type and the same number of channels as src.
@param thresh threshold value.
@param maxval maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding
types.
@param type thresholding type (see the cv::ThresholdTypes).

@sa  adaptiveThreshold, findContours, compare, min, max
 */
CV_EXPORTS_W double threshold( InputArray src, OutputArray dst,
                               double thresh, double maxval, int type );


/** @brief Applies an adaptive threshold to an array.

The function transforms a grayscale image to a binary image according to the formulae:
-   **THRESH_BINARY**
    \f[dst(x,y) =  \fork{\texttt{maxValue}}{if \(src(x,y) > T(x,y)\)}{0}{otherwise}\f]
-   **THRESH_BINARY_INV**
    \f[dst(x,y) =  \fork{0}{if \(src(x,y) > T(x,y)\)}{\texttt{maxValue}}{otherwise}\f]
where \f$T(x,y)\f$ is a threshold calculated individually for each pixel (see adaptiveMethod parameter).

The function can process the image in-place.

@param src Source 8-bit single-channel image.
@param dst Destination image of the same size and the same type as src.
@param maxValue Non-zero value assigned to the pixels for which the condition is satisfied
@param adaptiveMethod Adaptive thresholding algorithm to use, see cv::AdaptiveThresholdTypes
@param thresholdType Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV,
see cv::ThresholdTypes.
@param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the
pixel: 3, 5, 7, and so on.
@param C Constant subtracted from the mean or weighted mean (see the details below). Normally, it
is positive but may be zero or negative as well.

@sa  threshold, blur, GaussianBlur
 */
CV_EXPORTS_W void adaptiveThreshold( InputArray src, OutputArray dst,
                                     double maxValue, int adaptiveMethod,
                                     int thresholdType, int blockSize, double C );

//! @} imgproc_misc

//! @addtogroup imgproc_filter
//! @{

/** @example Pyramids.cpp
An example using pyrDown and pyrUp functions
 */
/** @brief Blurs an image and downsamples it.

By default, size of the output image is computed as `Size((src.cols+1)/2, (src.rows+1)/2)`, but in
any case, the following conditions should be satisfied:

\f[\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}\f]

The function performs the downsampling step of the Gaussian pyramid construction. First, it
convolves the source image with the kernel:

\f[\frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}\f]

Then, it downsamples the image by rejecting even rows and columns.

@param src input image.
@param dst output image; it has the specified size and the same type as src.
@param dstsize size of the output image.
@param borderType Pixel extrapolation method, see cv::BorderTypes (BORDER_CONSTANT isn't supported)
 */
CV_EXPORTS_W void pyrDown( InputArray src, OutputArray dst,
                           const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );

/** @brief Upsamples an image and then blurs it.

By default, size of the output image is computed as `Size(src.cols\*2, (src.rows\*2)`, but in any
case, the following conditions should be satisfied:

\f[\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}\f]

The function performs the upsampling step of the Gaussian pyramid construction, though it can
actually be used to construct the Laplacian pyramid. First, it upsamples the source image by
injecting even zero rows and columns and then convolves the result with the same kernel as in
pyrDown multiplied by 4.

@param src input image.
@param dst output image. It has the specified size and the same type as src .
@param dstsize size of the output image.
@param borderType Pixel extrapolation method, see cv::BorderTypes (only BORDER_DEFAULT is supported)
 */
CV_EXPORTS_W void pyrUp( InputArray src, OutputArray dst,
                         const Size& dstsize = Size(), int borderType = BORDER_DEFAULT );

/** @brief Constructs the Gaussian pyramid for an image.

The function constructs a vector of images and builds the Gaussian pyramid by recursively applying
pyrDown to the previously built pyramid layers, starting from `dst[0]==src`.

@param src Source image. Check pyrDown for the list of supported types.
@param dst Destination vector of maxlevel+1 images of the same type as src. dst[0] will be the
same as src. dst[1] is the next pyramid layer, a smoothed and down-sized src, and so on.
@param maxlevel 0-based index of the last (the smallest) pyramid layer. It must be non-negative.
@param borderType Pixel extrapolation method, see cv::BorderTypes (BORDER_CONSTANT isn't supported)
 */
CV_EXPORTS void buildPyramid( InputArray src, OutputArrayOfArrays dst,
                              int maxlevel, int borderType = BORDER_DEFAULT );

//! @} imgproc_filter

//! @addtogroup imgproc_transform
//! @{

/** @brief Transforms an image to compensate for lens distortion.

The function transforms an image to compensate radial and tangential lens distortion.

The function is simply a combination of cv::initUndistortRectifyMap (with unity R ) and cv::remap
(with bilinear interpolation). See the former function for details of the transformation being
performed.

Those pixels in the destination image, for which there is no correspondent pixels in the source
image, are filled with zeros (black color).

A particular subset of the source image that will be visible in the corrected image can be regulated
by newCameraMatrix. You can use cv::getOptimalNewCameraMatrix to compute the appropriate
newCameraMatrix depending on your requirements.

The camera matrix and the distortion parameters can be determined using cv::calibrateCamera. If
the resolution of images is different from the resolution used at the calibration stage, \f$f_x,
f_y, c_x\f$ and \f$c_y\f$ need to be scaled accordingly, while the distortion coefficients remain
the same.

@param src Input (distorted) image.
@param dst Output (corrected) image that has the same size and type as src .
@param cameraMatrix Input camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param newCameraMatrix Camera matrix of the distorted image. By default, it is the same as
cameraMatrix but you may additionally scale and shift the result by using a different matrix.
 */
CV_EXPORTS_W void undistort( InputArray src, OutputArray dst,
                             InputArray cameraMatrix,
                             InputArray distCoeffs,
                             InputArray newCameraMatrix = noArray() );

/** @brief Computes the undistortion and rectification transformation map.

The function computes the joint undistortion and rectification transformation and represents the
result in the form of maps for remap. The undistorted image looks like original, as if it is
captured with a camera using the camera matrix =newCameraMatrix and zero distortion. In case of a
monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by
cv::getOptimalNewCameraMatrix for a better control over scaling. In case of a stereo camera,
newCameraMatrix is normally set to P1 or P2 computed by cv::stereoRectify .

Also, this new camera is oriented differently in the coordinate space, according to R. That, for
example, helps to align two heads of a stereo camera so that the epipolar lines on both images
become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).

The function actually builds the maps for the inverse mapping algorithm that is used by remap. That
is, for each pixel \f$(u, v)\f$ in the destination (corrected and rectified) image, the function
computes the corresponding coordinates in the source image (that is, in the original image from
camera). The following process is applied:
\f[
\begin{array}{l}
x  \leftarrow (u - {c'}_x)/{f'}_x  \\
y  \leftarrow (v - {c'}_y)/{f'}_y  \\
{[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\
x'  \leftarrow X/W  \\
y'  \leftarrow Y/W  \\
r^2  \leftarrow x'^2 + y'^2 \\
x''  \leftarrow x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
+ 2p_1 x' y' + p_2(r^2 + 2 x'^2)  + s_1 r^2 + s_2 r^4\\
y''  \leftarrow y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}
+ p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\
s\vecthree{x'''}{y'''}{1} =
\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}((\tau_x, \tau_y)}
{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}
{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\
map_x(u,v)  \leftarrow x''' f_x + c_x  \\
map_y(u,v)  \leftarrow y''' f_y + c_y
\end{array}
\f]
where \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
are the distortion coefficients.

In case of a stereo camera, this function is called twice: once for each camera head, after
stereoRectify, which in its turn is called after cv::stereoCalibrate. But if the stereo camera
was not calibrated, it is still possible to compute the rectification transformations directly from
the fundamental matrix using cv::stereoRectifyUncalibrated. For each camera, the function computes
homography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D
space. R can be computed from H as
\f[\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}\f]
where cameraMatrix can be chosen arbitrarily.

@param cameraMatrix Input camera matrix \f$A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param R Optional rectification transformation in the object space (3x3 matrix). R1 or R2 ,
computed by stereoRectify can be passed here. If the matrix is empty, the identity transformation
is assumed. In cvInitUndistortMap R assumed to be an identity matrix.
@param newCameraMatrix New camera matrix \f$A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}\f$.
@param size Undistorted image size.
@param m1type Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see cv::convertMaps
@param map1 The first output map.
@param map2 The second output map.
 */
CV_EXPORTS_W void initUndistortRectifyMap( InputArray cameraMatrix, InputArray distCoeffs,
                           InputArray R, InputArray newCameraMatrix,
                           Size size, int m1type, OutputArray map1, OutputArray map2 );

//! initializes maps for cv::remap() for wide-angle
CV_EXPORTS_W float initWideAngleProjMap( InputArray cameraMatrix, InputArray distCoeffs,
                                         Size imageSize, int destImageWidth,
                                         int m1type, OutputArray map1, OutputArray map2,
                                         int projType = PROJ_SPHERICAL_EQRECT, double alpha = 0);

/** @brief Returns the default new camera matrix.

The function returns the camera matrix that is either an exact copy of the input cameraMatrix (when
centerPrinicipalPoint=false ), or the modified one (when centerPrincipalPoint=true).

In the latter case, the new camera matrix will be:

\f[\begin{bmatrix} f_x && 0 && ( \texttt{imgSize.width} -1)*0.5  \\ 0 && f_y && ( \texttt{imgSize.height} -1)*0.5  \\ 0 && 0 && 1 \end{bmatrix} ,\f]

where \f$f_x\f$ and \f$f_y\f$ are \f$(0,0)\f$ and \f$(1,1)\f$ elements of cameraMatrix, respectively.

By default, the undistortion functions in OpenCV (see initUndistortRectifyMap, undistort) do not
move the principal point. However, when you work with stereo, it is important to move the principal
points in both views to the same y-coordinate (which is required by most of stereo correspondence
algorithms), and may be to the same x-coordinate too. So, you can form the new camera matrix for
each view where the principal points are located at the center.

@param cameraMatrix Input camera matrix.
@param imgsize Camera view image size in pixels.
@param centerPrincipalPoint Location of the principal point in the new camera matrix. The
parameter indicates whether this location should be at the image center or not.
 */
CV_EXPORTS_W Mat getDefaultNewCameraMatrix( InputArray cameraMatrix, Size imgsize = Size(),
                                            bool centerPrincipalPoint = false );

/** @brief Computes the ideal point coordinates from the observed point coordinates.

The function is similar to cv::undistort and cv::initUndistortRectifyMap but it operates on a
sparse set of points instead of a raster image. Also the function performs a reverse transformation
to projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a
planar object, it does, up to a translation vector, if the proper R is specified.

For each observed point coordinate \f$(u, v)\f$ the function computes:
\f[
\begin{array}{l}
x^{"}  \leftarrow (u - c_x)/f_x  \\
y^{"}  \leftarrow (v - c_y)/f_y  \\
(x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\
{[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\
x  \leftarrow X/W  \\
y  \leftarrow Y/W  \\
\text{only performed if P is specified:} \\
u'  \leftarrow x {f'}_x + {c'}_x  \\
v'  \leftarrow y {f'}_y + {c'}_y
\end{array}
\f]

where *undistort* is an approximate iterative algorithm that estimates the normalized original
point coordinates out of the normalized distorted point coordinates ("normalized" means that the
coordinates do not depend on the camera matrix).

The function can be used for both a stereo camera head or a monocular camera (when R is empty).

@param src Observed point coordinates, 1xN or Nx1 2-channel (CV_32FC2 or CV_64FC2).
@param dst Output ideal point coordinates after undistortion and reverse perspective
transformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.
@param cameraMatrix Camera matrix \f$\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .
@param distCoeffs Input vector of distortion coefficients
\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$
of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
@param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by
cv::stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.
@param P New camera matrix (3x3) or new projection matrix (3x4) \f$\begin{bmatrix} {f'}_x & 0 & {c'}_x & t_x \\ 0 & {f'}_y & {c'}_y & t_y \\ 0 & 0 & 1 & t_z \end{bmatrix}\f$. P1 or P2 computed by
cv::stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.
 */
CV_EXPORTS_W void undistortPoints( InputArray src, OutputArray dst,
                                   InputArray cameraMatrix, InputArray distCoeffs,
                                   InputArray R = noArray(), InputArray P = noArray());

//! @} imgproc_transform

//! @addtogroup imgproc_hist
//! @{

/** @example demhist.cpp
An example for creating histograms of an image
*/

/** @brief Calculates a histogram of a set of arrays.

The function cv::calcHist calculates the histogram of one or more arrays. The elements of a tuple used
to increment a histogram bin are taken from the corresponding input arrays at the same location. The
sample below shows how to compute a 2D Hue-Saturation histogram for a color image. :
@code
    #include <opencv2/imgproc.hpp>
    #include <opencv2/highgui.hpp>

    using namespace cv;

    int main( int argc, char** argv )
    {
        Mat src, hsv;
        if( argc != 2 || !(src=imread(argv[1], 1)).data )
            return -1;

        cvtColor(src, hsv, COLOR_BGR2HSV);

        // Quantize the hue to 30 levels
        // and the saturation to 32 levels
        int hbins = 30, sbins = 32;
        int histSize[] = {hbins, sbins};
        // hue varies from 0 to 179, see cvtColor
        float hranges[] = { 0, 180 };
        // saturation varies from 0 (black-gray-white) to
        // 255 (pure spectrum color)
        float sranges[] = { 0, 256 };
        const float* ranges[] = { hranges, sranges };
        MatND hist;
        // we compute the histogram from the 0-th and 1-st channels
        int channels[] = {0, 1};

        calcHist( &hsv, 1, channels, Mat(), // do not use mask
                 hist, 2, histSize, ranges,
                 true, // the histogram is uniform
                 false );
        double maxVal=0;
        minMaxLoc(hist, 0, &maxVal, 0, 0);

        int scale = 10;
        Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

        for( int h = 0; h < hbins; h++ )
            for( int s = 0; s < sbins; s++ )
            {
                float binVal = hist.at<float>(h, s);
                int intensity = cvRound(binVal*255/maxVal);
                rectangle( histImg, Point(h*scale, s*scale),
                            Point( (h+1)*scale - 1, (s+1)*scale - 1),
                            Scalar::all(intensity),
                            CV_FILLED );
            }

        namedWindow( "Source", 1 );
        imshow( "Source", src );

        namedWindow( "H-S Histogram", 1 );
        imshow( "H-S Histogram", histImg );
        waitKey();
    }
@endcode

@param images Source arrays. They all should have the same depth, CV_8U, CV_16U or CV_32F , and the same
size. Each of them can have an arbitrary number of channels.
@param nimages Number of source images.
@param channels List of the dims channels used to compute the histogram. The first array channels
are numerated from 0 to images[0].channels()-1 , the second array channels are counted from
images[0].channels() to images[0].channels() + images[1].channels()-1, and so on.
@param mask Optional mask. If the matrix is not empty, it must be an 8-bit array of the same size
as images[i] . The non-zero mask elements mark the array elements counted in the histogram.
@param hist Output histogram, which is a dense or sparse dims -dimensional array.
@param dims Histogram dimensionality that must be positive and not greater than CV_MAX_DIMS
(equal to 32 in the current OpenCV version).
@param histSize Array of histogram sizes in each dimension.
@param ranges Array of the dims arrays of the histogram bin boundaries in each dimension. When the
histogram is uniform ( uniform =true), then for each dimension i it is enough to specify the lower
(inclusive) boundary \f$L_0\f$ of the 0-th histogram bin and the upper (exclusive) boundary
\f$U_{\texttt{histSize}[i]-1}\f$ for the last histogram bin histSize[i]-1 . That is, in case of a
uniform histogram each of ranges[i] is an array of 2 elements. When the histogram is not uniform (
uniform=false ), then each of ranges[i] contains histSize[i]+1 elements:
\f$L_0, U_0=L_1, U_1=L_2, ..., U_{\texttt{histSize[i]}-2}=L_{\texttt{histSize[i]}-1}, U_{\texttt{histSize[i]}-1}\f$
. The array elements, that are not between \f$L_0\f$ and \f$U_{\texttt{histSize[i]}-1}\f$ , are not
counted in the histogram.
@param uniform Flag indicating whether the histogram is uniform or not (see above).
@param accumulate Accumulation flag. If it is set, the histogram is not cleared in the beginning
when it is allocated. This feature enables you to compute a single histogram from several sets of
arrays, or to update the histogram in time.
*/
CV_EXPORTS void calcHist( const Mat* images, int nimages,
                          const int* channels, InputArray mask,
                          OutputArray hist, int dims, const int* histSize,
                          const float** ranges, bool uniform = true, bool accumulate = false );

/** @overload

this variant uses cv::SparseMat for output
*/
CV_EXPORTS void calcHist( const Mat* images, int nimages,
                          const int* channels, InputArray mask,
                          SparseMat& hist, int dims,
                          const int* histSize, const float** ranges,
                          bool uniform = true, bool accumulate = false );

/** @overload */
CV_EXPORTS_W void calcHist( InputArrayOfArrays images,
                            const std::vector<int>& channels,
                            InputArray mask, OutputArray hist,
                            const std::vector<int>& histSize,
                            const std::vector<float>& ranges,
                            bool accumulate = false );

/** @brief Calculates the back projection of a histogram.

The function cv::calcBackProject calculates the back project of the histogram. That is, similarly to
cv::calcHist , at each location (x, y) the function collects the values from the selected channels
in the input images and finds the corresponding histogram bin. But instead of incrementing it, the
function reads the bin value, scales it by scale , and stores in backProject(x,y) . In terms of
statistics, the function computes probability of each element value in respect with the empirical
probability distribution represented by the histogram. See how, for example, you can find and track
a bright-colored object in a scene:

- Before tracking, show the object to the camera so that it covers almost the whole frame.
Calculate a hue histogram. The histogram may have strong maximums, corresponding to the dominant
colors in the object.

- When tracking, calculate a back projection of a hue plane of each input video frame using that
pre-computed histogram. Threshold the back projection to suppress weak colors. It may also make
sense to suppress pixels with non-sufficient color saturation and too dark or too bright pixels.

- Find connected components in the resulting picture and choose, for example, the largest
component.

This is an approximate algorithm of the CamShift color object tracker.

@param images Source arrays. They all should have the same depth, CV_8U, CV_16U or CV_32F , and the same
size. Each of them can have an arbitrary number of channels.
@param nimages Number of source images.
@param channels The list of channels used to compute the back projection. The number of channels
must match the histogram dimensionality. The first array channels are numerated from 0 to
images[0].channels()-1 , the second array channels are counted from images[0].channels() to
images[0].channels() + images[1].channels()-1, and so on.
@param hist Input histogram that can be dense or sparse.
@param backProject Destination back projection array that is a single-channel array of the same
size and depth as images[0] .
@param ranges Array of arrays of the histogram bin boundaries in each dimension. See cv::calcHist .
@param scale Optional scale factor for the output back projection.
@param uniform Flag indicating whether the histogram is uniform or not (see above).

@sa cv::calcHist, cv::compareHist
 */
CV_EXPORTS void calcBackProject( const Mat* images, int nimages,
                                 const int* channels, InputArray hist,
                                 OutputArray backProject, const float** ranges,
                                 double scale = 1, bool uniform = true );

/** @overload */
CV_EXPORTS void calcBackProject( const Mat* images, int nimages,
                                 const int* channels, const SparseMat& hist,
                                 OutputArray backProject, const float** ranges,
                                 double scale = 1, bool uniform = true );

/** @overload */
CV_EXPORTS_W void calcBackProject( InputArrayOfArrays images, const std::vector<int>& channels,
                                   InputArray hist, OutputArray dst,
                                   const std::vector<float>& ranges,
                                   double scale );

/** @brief Compares two histograms.

The function cv::compareHist compares two dense or two sparse histograms using the specified method.

The function returns \f$d(H_1, H_2)\f$ .

While the function works well with 1-, 2-, 3-dimensional dense histograms, it may not be suitable
for high-dimensional sparse histograms. In such histograms, because of aliasing and sampling
problems, the coordinates of non-zero histogram bins can slightly shift. To compare such histograms
or more general sparse configurations of weighted points, consider using the cv::EMD function.

@param H1 First compared histogram.
@param H2 Second compared histogram of the same size as H1 .
@param method Comparison method, see cv::HistCompMethods
 */
CV_EXPORTS_W double compareHist( InputArray H1, InputArray H2, int method );

/** @overload */
CV_EXPORTS double compareHist( const SparseMat& H1, const SparseMat& H2, int method );

/** @brief Equalizes the histogram of a grayscale image.

The function equalizes the histogram of the input image using the following algorithm:

- Calculate the histogram \f$H\f$ for src .
- Normalize the histogram so that the sum of histogram bins is 255.
- Compute the integral of the histogram:
\f[H'_i =  \sum _{0  \le j < i} H(j)\f]
- Transform the image using \f$H'\f$ as a look-up table: \f$\texttt{dst}(x,y) = H'(\texttt{src}(x,y))\f$

The algorithm normalizes the brightness and increases the contrast of the image.

@param src Source 8-bit single channel image.
@param dst Destination image of the same size and type as src .
 */
CV_EXPORTS_W void equalizeHist( InputArray src, OutputArray dst );

/** @brief Computes the "minimal work" distance between two weighted point configurations.

The function computes the earth mover distance and/or a lower boundary of the distance between the
two weighted point configurations. One of the applications described in @cite RubnerSept98,
@cite Rubner2000 is multi-dimensional histogram comparison for image retrieval. EMD is a transportation
problem that is solved using some modification of a simplex algorithm, thus the complexity is
exponential in the worst case, though, on average it is much faster. In the case of a real metric
the lower boundary can be calculated even faster (using linear-time algorithm) and it can be used
to determine roughly whether the two signatures are far enough so that they cannot relate to the
same object.

@param signature1 First signature, a \f$\texttt{size1}\times \texttt{dims}+1\f$ floating-point matrix.
Each row stores the point weight followed by the point coordinates. The matrix is allowed to have
a single column (weights only) if the user-defined cost matrix is used. The weights must be
non-negative and have at least one non-zero value.
@param signature2 Second signature of the same format as signature1 , though the number of rows
may be different. The total weights may be different. In this case an extra "dummy" point is added
to either signature1 or signature2. The weights must be non-negative and have at least one non-zero
value.
@param distType Used metric. See cv::DistanceTypes.
@param cost User-defined \f$\texttt{size1}\times \texttt{size2}\f$ cost matrix. Also, if a cost matrix
is used, lower boundary lowerBound cannot be calculated because it needs a metric function.
@param lowerBound Optional input/output parameter: lower boundary of a distance between the two
signatures that is a distance between mass centers. The lower boundary may not be calculated if
the user-defined cost matrix is used, the total weights of point configurations are not equal, or
if the signatures consist of weights only (the signature matrices have a single column). You
**must** initialize \*lowerBound . If the calculated distance between mass centers is greater or
equal to \*lowerBound (it means that the signatures are far enough), the function does not
calculate EMD. In any case \*lowerBound is set to the calculated distance between mass centers on
return. Thus, if you want to calculate both distance between mass centers and EMD, \*lowerBound
should be set to 0.
@param flow Resultant \f$\texttt{size1} \times \texttt{size2}\f$ flow matrix: \f$\texttt{flow}_{i,j}\f$ is
a flow from \f$i\f$ -th point of signature1 to \f$j\f$ -th point of signature2 .
 */
CV_EXPORTS float EMD( InputArray signature1, InputArray signature2,
                      int distType, InputArray cost=noArray(),
                      float* lowerBound = 0, OutputArray flow = noArray() );

CV_EXPORTS_AS(EMD) float wrapperEMD( InputArray signature1, InputArray signature2,
                      int distType, InputArray cost=noArray(),
                      CV_IN_OUT Ptr<float> lowerBound = Ptr<float>(), OutputArray flow = noArray() );

//! @} imgproc_hist

/** @example watershed.cpp
An example using the watershed algorithm
 */

/** @brief Performs a marker-based image segmentation using the watershed algorithm.

The function implements one of the variants of watershed, non-parametric marker-based segmentation
algorithm, described in @cite Meyer92 .

Before passing the image to the function, you have to roughly outline the desired regions in the
image markers with positive (\>0) indices. So, every region is represented as one or more connected
components with the pixel values 1, 2, 3, and so on. Such markers can be retrieved from a binary
mask using findContours and drawContours (see the watershed.cpp demo). The markers are "seeds" of
the future image regions. All the other pixels in markers , whose relation to the outlined regions
is not known and should be defined by the algorithm, should be set to 0's. In the function output,
each pixel in markers is set to a value of the "seed" components or to -1 at boundaries between the
regions.

@note Any two neighbor connected components are not necessarily separated by a watershed boundary
(-1's pixels); for example, they can touch each other in the initial marker image passed to the
function.

@param image Input 8-bit 3-channel image.
@param markers Input/output 32-bit single-channel image (map) of markers. It should have the same
size as image .

@sa findContours

@ingroup imgproc_misc
 */
CV_EXPORTS_W void watershed( InputArray image, InputOutputArray markers );

//! @addtogroup imgproc_filter
//! @{

/** @brief Performs initial step of meanshift segmentation of an image.

The function implements the filtering stage of meanshift segmentation, that is, the output of the
function is the filtered "posterized" image with color gradients and fine-grain texture flattened.
At every pixel (X,Y) of the input image (or down-sized input image, see below) the function executes
meanshift iterations, that is, the pixel (X,Y) neighborhood in the joint space-color hyperspace is
considered:

\f[(x,y): X- \texttt{sp} \le x  \le X+ \texttt{sp} , Y- \texttt{sp} \le y  \le Y+ \texttt{sp} , ||(R,G,B)-(r,g,b)||   \le \texttt{sr}\f]

where (R,G,B) and (r,g,b) are the vectors of color components at (X,Y) and (x,y), respectively
(though, the algorithm does not depend on the color space used, so any 3-component color space can
be used instead). Over the neighborhood the average spatial value (X',Y') and average color vector
(R',G',B') are found and they act as the neighborhood center on the next iteration:

\f[(X,Y)~(X',Y'), (R,G,B)~(R',G',B').\f]

After the iterations over, the color components of the initial pixel (that is, the pixel from where
the iterations started) are set to the final value (average color at the last iteration):

\f[I(X,Y) <- (R*,G*,B*)\f]

When maxLevel \> 0, the gaussian pyramid of maxLevel+1 levels is built, and the above procedure is
run on the smallest layer first. After that, the results are propagated to the larger layer and the
iterations are run again only on those pixels where the layer colors differ by more than sr from the
lower-resolution layer of the pyramid. That makes boundaries of color regions sharper. Note that the
results will be actually different from the ones obtained by running the meanshift procedure on the
whole original image (i.e. when maxLevel==0).

@param src The source 8-bit, 3-channel image.
@param dst The destination image of the same format and the same size as the source.
@param sp The spatial window radius.
@param sr The color window radius.
@param maxLevel Maximum level of the pyramid for the segmentation.
@param termcrit Termination criteria: when to stop meanshift iterations.
 */
CV_EXPORTS_W void pyrMeanShiftFiltering( InputArray src, OutputArray dst,
                                         double sp, double sr, int maxLevel = 1,
                                         TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1) );

//! @}

//! @addtogroup imgproc_misc
//! @{

/** @example grabcut.cpp
An example using the GrabCut algorithm
 */

/** @brief Runs the GrabCut algorithm.

The function implements the [GrabCut image segmentation algorithm](http://en.wikipedia.org/wiki/GrabCut).

@param img Input 8-bit 3-channel image.
@param mask Input/output 8-bit single-channel mask. The mask is initialized by the function when
mode is set to GC_INIT_WITH_RECT. Its elements may have one of the cv::GrabCutClasses.
@param rect ROI containing a segmented object. The pixels outside of the ROI are marked as
"obvious background". The parameter is only used when mode==GC_INIT_WITH_RECT .
@param bgdModel Temporary array for the background model. Do not modify it while you are
processing the same image.
@param fgdModel Temporary arrays for the foreground model. Do not modify it while you are
processing the same image.
@param iterCount Number of iterations the algorithm should make before returning the result. Note
that the result can be refined with further calls with mode==GC_INIT_WITH_MASK or
mode==GC_EVAL .
@param mode Operation mode that could be one of the cv::GrabCutModes
 */
CV_EXPORTS_W void grabCut( InputArray img, InputOutputArray mask, Rect rect,
                           InputOutputArray bgdModel, InputOutputArray fgdModel,
                           int iterCount, int mode = GC_EVAL );

/** @example distrans.cpp
An example on using the distance transform\
*/


/** @brief Calculates the distance to the closest zero pixel for each pixel of the source image.

The function cv::distanceTransform calculates the approximate or precise distance from every binary
image pixel to the nearest zero pixel. For zero image pixels, the distance will obviously be zero.

When maskSize == DIST_MASK_PRECISE and distanceType == DIST_L2 , the function runs the
algorithm described in @cite Felzenszwalb04 . This algorithm is parallelized with the TBB library.

In other cases, the algorithm @cite Borgefors86 is used. This means that for a pixel the function
finds the shortest path to the nearest zero pixel consisting of basic shifts: horizontal, vertical,
diagonal, or knight's move (the latest is available for a \f$5\times 5\f$ mask). The overall
distance is calculated as a sum of these basic distances. Since the distance function should be
symmetric, all of the horizontal and vertical shifts must have the same cost (denoted as a ), all
the diagonal shifts must have the same cost (denoted as `b`), and all knight's moves must have the
same cost (denoted as `c`). For the cv::DIST_C and cv::DIST_L1 types, the distance is calculated
precisely, whereas for cv::DIST_L2 (Euclidean distance) the distance can be calculated only with a
relative error (a \f$5\times 5\f$ mask gives more accurate results). For `a`,`b`, and `c`, OpenCV
uses the values suggested in the original paper:
- DIST_L1: `a = 1, b = 2`
- DIST_L2:
    - `3 x 3`: `a=0.955, b=1.3693`
    - `5 x 5`: `a=1, b=1.4, c=2.1969`
- DIST_C: `a = 1, b = 1`

Typically, for a fast, coarse distance estimation DIST_L2, a \f$3\times 3\f$ mask is used. For a
more accurate distance estimation DIST_L2, a \f$5\times 5\f$ mask or the precise algorithm is used.
Note that both the precise and the approximate algorithms are linear on the number of pixels.

This variant of the function does not only compute the minimum distance for each pixel \f$(x, y)\f$
but also identifies the nearest connected component consisting of zero pixels
(labelType==DIST_LABEL_CCOMP) or the nearest zero pixel (labelType==DIST_LABEL_PIXEL). Index of the
component/pixel is stored in `labels(x, y)`. When labelType==DIST_LABEL_CCOMP, the function
automatically finds connected components of zero pixels in the input image and marks them with
distinct labels. When labelType==DIST_LABEL_CCOMP, the function scans through the input image and
marks all the zero pixels with distinct labels.

In this mode, the complexity is still linear. That is, the function provides a very fast way to
compute the Voronoi diagram for a binary image. Currently, the second variant can use only the
approximate distance transform algorithm, i.e. maskSize=DIST_MASK_PRECISE is not supported
yet.

@param src 8-bit, single-channel (binary) source image.
@param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
single-channel image of the same size as src.
@param labels Output 2D array of labels (the discrete Voronoi diagram). It has the type
CV_32SC1 and the same size as src.
@param distanceType Type of distance, see cv::DistanceTypes
@param maskSize Size of the distance transform mask, see cv::DistanceTransformMasks.
DIST_MASK_PRECISE is not supported by this variant. In case of the DIST_L1 or DIST_C distance type,
the parameter is forced to 3 because a \f$3\times 3\f$ mask gives the same result as \f$5\times
5\f$ or any larger aperture.
@param labelType Type of the label array to build, see cv::DistanceTransformLabelTypes.
 */
CV_EXPORTS_AS(distanceTransformWithLabels) void distanceTransform( InputArray src, OutputArray dst,
                                     OutputArray labels, int distanceType, int maskSize,
                                     int labelType = DIST_LABEL_CCOMP );

/** @overload
@param src 8-bit, single-channel (binary) source image.
@param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,
single-channel image of the same size as src .
@param distanceType Type of distance, see cv::DistanceTypes
@param maskSize Size of the distance transform mask, see cv::DistanceTransformMasks. In case of the
DIST_L1 or DIST_C distance type, the parameter is forced to 3 because a \f$3\times 3\f$ mask gives
the same result as \f$5\times 5\f$ or any larger aperture.
@param dstType Type of output image. It can be CV_8U or CV_32F. Type CV_8U can be used only for
the first variant of the function and distanceType == DIST_L1.
*/
CV_EXPORTS_W void distanceTransform( InputArray src, OutputArray dst,
                                     int distanceType, int maskSize, int dstType=CV_32F);

/** @example ffilldemo.cpp
  An example using the FloodFill technique
*/

/** @overload

variant without `mask` parameter
*/
CV_EXPORTS int floodFill( InputOutputArray image,
                          Point seedPoint, Scalar newVal, CV_OUT Rect* rect = 0,
                          Scalar loDiff = Scalar(), Scalar upDiff = Scalar(),
                          int flags = 4 );

/** @brief Fills a connected component with the given color.

The function cv::floodFill fills a connected component starting from the seed point with the specified
color. The connectivity is determined by the color/brightness closeness of the neighbor pixels. The
pixel at \f$(x,y)\f$ is considered to belong to the repainted domain if:

- in case of a grayscale image and floating range
\f[\texttt{src} (x',y')- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} (x',y')+ \texttt{upDiff}\f]


- in case of a grayscale image and fixed range
\f[\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)- \texttt{loDiff} \leq \texttt{src} (x,y)  \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)+ \texttt{upDiff}\f]


- in case of a color image and floating range
\f[\texttt{src} (x',y')_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} (x',y')_r+ \texttt{upDiff} _r,\f]
\f[\texttt{src} (x',y')_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} (x',y')_g+ \texttt{upDiff} _g\f]
and
\f[\texttt{src} (x',y')_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} (x',y')_b+ \texttt{upDiff} _b\f]


- in case of a color image and fixed range
\f[\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r- \texttt{loDiff} _r \leq \texttt{src} (x,y)_r \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_r+ \texttt{upDiff} _r,\f]
\f[\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g- \texttt{loDiff} _g \leq \texttt{src} (x,y)_g \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_g+ \texttt{upDiff} _g\f]
and
\f[\texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b- \texttt{loDiff} _b \leq \texttt{src} (x,y)_b \leq \texttt{src} ( \texttt{seedPoint} .x, \texttt{seedPoint} .y)_b+ \texttt{upDiff} _b\f]


where \f$src(x',y')\f$ is the value of one of pixel neighbors that is already known to belong to the
component. That is, to be added to the connected component, a color/brightness of the pixel should
be close enough to:
- Color/brightness of one of its neighbors that already belong to the connected component in case
of a floating range.
- Color/brightness of the seed point in case of a fixed range.

Use these functions to either mark a connected component with the specified color in-place, or build
a mask and then extract the contour, or copy the region to another image, and so on.

@param image Input/output 1- or 3-channel, 8-bit, or floating-point image. It is modified by the
function unless the FLOODFILL_MASK_ONLY flag is set in the second variant of the function. See
the details below.
@param mask Operation mask that should be a single-channel 8-bit image, 2 pixels wider and 2 pixels
taller than image. Since this is both an input and output parameter, you must take responsibility
of initializing it. Flood-filling cannot go across non-zero pixels in the input mask. For example,
an edge detector output can be used as a mask to stop filling at edges. On output, pixels in the
mask corresponding to filled pixels in the image are set to 1 or to the a value specified in flags
as described below. It is therefore possible to use the same mask in multiple calls to the function
to make sure the filled areas do not overlap.
@param seedPoint Starting point.
@param newVal New value of the repainted domain pixels.
@param loDiff Maximal lower brightness/color difference between the currently observed pixel and
one of its neighbors belonging to the component, or a seed pixel being added to the component.
@param upDiff Maximal upper brightness/color difference between the currently observed pixel and
one of its neighbors belonging to the component, or a seed pixel being added to the component.
@param rect Optional output parameter set by the function to the minimum bounding rectangle of the
repainted domain.
@param flags Operation flags. The first 8 bits contain a connectivity value. The default value of
4 means that only the four nearest neighbor pixels (those that share an edge) are considered. A
connectivity value of 8 means that the eight nearest neighbor pixels (those that share a corner)
will be considered. The next 8 bits (8-16) contain a value between 1 and 255 with which to fill
the mask (the default value is 1). For example, 4 | ( 255 \<\< 8 ) will consider 4 nearest
neighbours and fill the mask with a value of 255. The following additional options occupy higher
bits and therefore may be further combined with the connectivity and mask fill values using
bit-wise or (|), see cv::FloodFillFlags.

@note Since the mask is larger than the filled image, a pixel \f$(x, y)\f$ in image corresponds to the
pixel \f$(x+1, y+1)\f$ in the mask .

@sa findContours
 */
CV_EXPORTS_W int floodFill( InputOutputArray image, InputOutputArray mask,
                            Point seedPoint, Scalar newVal, CV_OUT Rect* rect=0,
                            Scalar loDiff = Scalar(), Scalar upDiff = Scalar(),
                            int flags = 4 );

/** @brief Converts an image from one color space to another.

The function converts an input image from one color space to another. In case of a transformation
to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note
that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the
bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue
component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and
sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.

The conventional ranges for R, G, and B channel values are:
-   0 to 255 for CV_8U images
-   0 to 65535 for CV_16U images
-   0 to 1 for CV_32F images

In case of linear transformations, the range does not matter. But in case of a non-linear
transformation, an input RGB image should be normalized to the proper value range to get the correct
results, for example, for RGB \f$\rightarrow\f$ L\*u\*v\* transformation. For example, if you have a
32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will
have the 0..255 value range instead of 0..1 assumed by the function. So, before calling cvtColor ,
you need first to scale the image down:
@code
    img *= 1./255;
    cvtColor(img, img, COLOR_BGR2Luv);
@endcode
If you use cvtColor with 8-bit images, the conversion will have some information lost. For many
applications, this will not be noticeable but it is recommended to use 32-bit images in applications
that need the full range of colors or that convert an image before an operation and then convert
back.

If conversion adds the alpha channel, its value will set to the maximum of corresponding channel
range: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F.

@param src input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision
floating-point.
@param dst output image of the same size and depth as src.
@param code color space conversion code (see cv::ColorConversionCodes).
@param dstCn number of channels in the destination image; if the parameter is 0, the number of the
channels is derived automatically from src and code.

@see @ref imgproc_color_conversions
 */
CV_EXPORTS_W void cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 );

//! @} imgproc_misc

// main function for all demosaicing processes
CV_EXPORTS_W void demosaicing(InputArray _src, OutputArray _dst, int code, int dcn = 0);

//! @addtogroup imgproc_shape
//! @{

/** @brief Calculates all of the moments up to the third order of a polygon or rasterized shape.

The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The
results are returned in the structure cv::Moments.

@param array Raster image (single-channel, 8-bit or floating-point 2D array) or an array (
\f$1 \times N\f$ or \f$N \times 1\f$ ) of 2D points (Point or Point2f ).
@param binaryImage If it is true, all non-zero image pixels are treated as 1's. The parameter is
used for images only.
@returns moments.

@note Only applicable to contour moments calculations from Python bindings: Note that the numpy
type for the input array should be either np.int32 or np.float32.

@sa  contourArea, arcLength
 */
CV_EXPORTS_W Moments moments( InputArray array, bool binaryImage = false );

/** @brief Calculates seven Hu invariants.

The function calculates seven Hu invariants (introduced in @cite Hu62; see also
<http://en.wikipedia.org/wiki/Image_moment>) defined as:

\f[\begin{array}{l} hu[0]= \eta _{20}+ \eta _{02} \\ hu[1]=( \eta _{20}- \eta _{02})^{2}+4 \eta _{11}^{2} \\ hu[2]=( \eta _{30}-3 \eta _{12})^{2}+ (3 \eta _{21}- \eta _{03})^{2} \\ hu[3]=( \eta _{30}+ \eta _{12})^{2}+ ( \eta _{21}+ \eta _{03})^{2} \\ hu[4]=( \eta _{30}-3 \eta _{12})( \eta _{30}+ \eta _{12})[( \eta _{30}+ \eta _{12})^{2}-3( \eta _{21}+ \eta _{03})^{2}]+(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ hu[5]=( \eta _{20}- \eta _{02})[( \eta _{30}+ \eta _{12})^{2}- ( \eta _{21}+ \eta _{03})^{2}]+4 \eta _{11}( \eta _{30}+ \eta _{12})( \eta _{21}+ \eta _{03}) \\ hu[6]=(3 \eta _{21}- \eta _{03})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}]-( \eta _{30}-3 \eta _{12})( \eta _{21}+ \eta _{03})[3( \eta _{30}+ \eta _{12})^{2}-( \eta _{21}+ \eta _{03})^{2}] \\ \end{array}\f]

where \f$\eta_{ji}\f$ stands for \f$\texttt{Moments::nu}_{ji}\f$ .

These values are proved to be invariants to the image scale, rotation, and reflection except the
seventh one, whose sign is changed by reflection. This invariance is proved with the assumption of
infinite image resolution. In case of raster images, the computed Hu invariants for the original and
transformed images are a bit different.

@param moments Input moments computed with moments .
@param hu Output Hu invariants.

@sa matchShapes
 */
CV_EXPORTS void HuMoments( const Moments& moments, double hu[7] );

/** @overload */
CV_EXPORTS_W void HuMoments( const Moments& m, OutputArray hu );

//! @} imgproc_shape

//! @addtogroup imgproc_object
//! @{

//! type of the template matching operation
enum TemplateMatchModes {
    TM_SQDIFF        = 0, //!< \f[R(x,y)= \sum _{x',y'} (T(x',y')-I(x+x',y+y'))^2\f]
    TM_SQDIFF_NORMED = 1, //!< \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y')-I(x+x',y+y'))^2}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]
    TM_CCORR         = 2, //!< \f[R(x,y)= \sum _{x',y'} (T(x',y')  \cdot I(x+x',y+y'))\f]
    TM_CCORR_NORMED  = 3, //!< \f[R(x,y)= \frac{\sum_{x',y'} (T(x',y') \cdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}\f]
    TM_CCOEFF        = 4, //!< \f[R(x,y)= \sum _{x',y'} (T'(x',y')  \cdot I'(x+x',y+y'))\f]
                          //!< where
                          //!< \f[\begin{array}{l} T'(x',y')=T(x',y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} T(x'',y'') \\ I'(x+x',y+y')=I(x+x',y+y') - 1/(w  \cdot h)  \cdot \sum _{x'',y''} I(x+x'',y+y'') \end{array}\f]
    TM_CCOEFF_NORMED = 5  //!< \f[R(x,y)= \frac{ \sum_{x',y'} (T'(x',y') \cdot I'(x+x',y+y')) }{ \sqrt{\sum_{x',y'}T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x',y+y')^2} }\f]
};

/** @example MatchTemplate_Demo.cpp
An example using Template Matching algorithm
 */
/** @brief Compares a template against overlapped image regions.

The function slides through image , compares the overlapped patches of size \f$w \times h\f$ against
templ using the specified method and stores the comparison results in result . Here are the formulae
for the available comparison methods ( \f$I\f$ denotes image, \f$T\f$ template, \f$R\f$ result ). The summation
is done over template and/or the image patch: \f$x' = 0...w-1, y' = 0...h-1\f$

After the function finishes the comparison, the best matches can be found as global minimums (when
TM_SQDIFF was used) or maximums (when TM_CCORR or TM_CCOEFF was used) using the
minMaxLoc function. In case of a color image, template summation in the numerator and each sum in
the denominator is done over all of the channels and separate mean values are used for each channel.
That is, the function can take a color template and a color image. The result will still be a
single-channel image, which is easier to analyze.

@param image Image where the search is running. It must be 8-bit or 32-bit floating-point.
@param templ Searched template. It must be not greater than the source image and have the same
data type.
@param result Map of comparison results. It must be single-channel 32-bit floating-point. If image
is \f$W \times H\f$ and templ is \f$w \times h\f$ , then result is \f$(W-w+1) \times (H-h+1)\f$ .
@param method Parameter specifying the comparison method, see cv::TemplateMatchModes
@param mask Mask of searched template. It must have the same datatype and size with templ. It is
not set by default. Currently, only the TM_SQDIFF and TM_CCORR_NORMED methods are supported.
 */
CV_EXPORTS_W void matchTemplate( InputArray image, InputArray templ,
                                 OutputArray result, int method, InputArray mask = noArray() );

//! @}

//! @addtogroup imgproc_shape
//! @{

/** @brief computes the connected components labeled image of boolean image

image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
represents the background label. ltype specifies the output label image type, an important
consideration based on the total number of labels or alternatively the total number of pixels in
the source image. ccltype specifies the connected components labeling algorithm to use, currently
Grana (BBDT) and Wu's (SAUF) algorithms are supported, see the cv::ConnectedComponentsAlgorithmsTypes
for details. Note that SAUF algorithm forces a row major ordering of labels while BBDT does not.
This function uses parallel version of both Grana and Wu's algorithms if at least one allowed
parallel framework is enabled and if the rows of the image are at least twice the number returned by getNumberOfCPUs.

@param image the 8-bit single-channel image to be labeled
@param labels destination labeled image
@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
@param ltype output image label type. Currently CV_32S and CV_16U are supported.
@param ccltype connected components algorithm type (see the cv::ConnectedComponentsAlgorithmsTypes).
*/
CV_EXPORTS_AS(connectedComponentsWithAlgorithm) int connectedComponents(InputArray image, OutputArray labels,
                                                                        int connectivity, int ltype, int ccltype);


/** @overload

@param image the 8-bit single-channel image to be labeled
@param labels destination labeled image
@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
@param ltype output image label type. Currently CV_32S and CV_16U are supported.
*/
CV_EXPORTS_W int connectedComponents(InputArray image, OutputArray labels,
                                     int connectivity = 8, int ltype = CV_32S);


/** @brief computes the connected components labeled image of boolean image and also produces a statistics output for each label

image with 4 or 8 way connectivity - returns N, the total number of labels [0, N-1] where 0
represents the background label. ltype specifies the output label image type, an important
consideration based on the total number of labels or alternatively the total number of pixels in
the source image. ccltype specifies the connected components labeling algorithm to use, currently
Grana's (BBDT) and Wu's (SAUF) algorithms are supported, see the cv::ConnectedComponentsAlgorithmsTypes
for details. Note that SAUF algorithm forces a row major ordering of labels while BBDT does not.
This function uses parallel version of both Grana and Wu's algorithms (statistics included) if at least one allowed
parallel framework is enabled and if the rows of the image are at least twice the number returned by getNumberOfCPUs.

@param image the 8-bit single-channel image to be labeled
@param labels destination labeled image
@param stats statistics output for each label, including the background label, see below for
available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
cv::ConnectedComponentsTypes. The data type is CV_32S.
@param centroids centroid output for each label, including the background label. Centroids are
accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
@param ltype output image label type. Currently CV_32S and CV_16U are supported.
@param ccltype connected components algorithm type (see the cv::ConnectedComponentsAlgorithmsTypes).
*/
CV_EXPORTS_AS(connectedComponentsWithStatsWithAlgorithm) int connectedComponentsWithStats(InputArray image, OutputArray labels,
                                                                                          OutputArray stats, OutputArray centroids,
                                                                                          int connectivity, int ltype, int ccltype);

/** @overload
@param image the 8-bit single-channel image to be labeled
@param labels destination labeled image
@param stats statistics output for each label, including the background label, see below for
available statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of
cv::ConnectedComponentsTypes. The data type is CV_32S.
@param centroids centroid output for each label, including the background label. Centroids are
accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.
@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively
@param ltype output image label type. Currently CV_32S and CV_16U are supported.
*/
CV_EXPORTS_W int connectedComponentsWithStats(InputArray image, OutputArray labels,
                                              OutputArray stats, OutputArray centroids,
                                              int connectivity = 8, int ltype = CV_32S);


/** @brief Finds contours in a binary image.

The function retrieves contours from the binary image using the algorithm @cite Suzuki85 . The contours
are a useful tool for shape analysis and object detection and recognition. See squares.cpp in the
OpenCV sample directory.
@note Since opencv 3.2 source image is not modified by this function.

@param image Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero
pixels remain 0's, so the image is treated as binary . You can use cv::compare, cv::inRange, cv::threshold ,
cv::adaptiveThreshold, cv::Canny, and others to create a binary image out of a grayscale or color one.
If mode equals to cv::RETR_CCOMP or cv::RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).
@param contours Detected contours. Each contour is stored as a vector of points (e.g.
std::vector<std::vector<cv::Point> >).
@param hierarchy Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology. It has
as many elements as the number of contours. For each i-th contour contours[i], the elements
hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices
in contours of the next and previous contours at the same hierarchical level, the first child
contour and the parent contour, respectively. If for the contour i there are no next, previous,
parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
@param mode Contour retrieval mode, see cv::RetrievalModes
@param method Contour approximation method, see cv::ContourApproximationModes
@param offset Optional offset by which every contour point is shifted. This is useful if the
contours are extracted from the image ROI and then they should be analyzed in the whole image
context.
 */
CV_EXPORTS_W void findContours( InputOutputArray image, OutputArrayOfArrays contours,
                              OutputArray hierarchy, int mode,
                              int method, Point offset = Point());

/** @overload */
CV_EXPORTS void findContours( InputOutputArray image, OutputArrayOfArrays contours,
                              int mode, int method, Point offset = Point());

/** @brief Approximates a polygonal curve(s) with the specified precision.

The function cv::approxPolyDP approximates a curve or a polygon with another curve/polygon with less
vertices so that the distance between them is less or equal to the specified precision. It uses the
Douglas-Peucker algorithm <http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm>

@param curve Input vector of a 2D point stored in std::vector or Mat
@param approxCurve Result of the approximation. The type should match the type of the input curve.
@param epsilon Parameter specifying the approximation accuracy. This is the maximum distance
between the original curve and its approximation.
@param closed If true, the approximated curve is closed (its first and last vertices are
connected). Otherwise, it is not closed.
 */
CV_EXPORTS_W void approxPolyDP( InputArray curve,
                                OutputArray approxCurve,
                                double epsilon, bool closed );

/** @brief Calculates a contour perimeter or a curve length.

The function computes a curve length or a closed contour perimeter.

@param curve Input vector of 2D points, stored in std::vector or Mat.
@param closed Flag indicating whether the curve is closed or not.
 */
CV_EXPORTS_W double arcLength( InputArray curve, bool closed );

/** @brief Calculates the up-right bounding rectangle of a point set.

The function calculates and returns the minimal up-right bounding rectangle for the specified point set.

@param points Input 2D point set, stored in std::vector or Mat.
 */
CV_EXPORTS_W Rect boundingRect( InputArray points );

/** @brief Calculates a contour area.

The function computes a contour area. Similarly to moments , the area is computed using the Green
formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using
drawContours or fillPoly , can be different. Also, the function will most certainly give a wrong
results for contours with self-intersections.

Example:
@code
    vector<Point> contour;
    contour.push_back(Point2f(0, 0));
    contour.push_back(Point2f(10, 0));
    contour.push_back(Point2f(10, 10));
    contour.push_back(Point2f(5, 4));

    double area0 = contourArea(contour);
    vector<Point> approx;
    approxPolyDP(contour, approx, 5, true);
    double area1 = contourArea(approx);

    cout << "area0 =" << area0 << endl <<
            "area1 =" << area1 << endl <<
            "approx poly vertices" << approx.size() << endl;
@endcode
@param contour Input vector of 2D points (contour vertices), stored in std::vector or Mat.
@param oriented Oriented area flag. If it is true, the function returns a signed area value,
depending on the contour orientation (clockwise or counter-clockwise). Using this feature you can
determine orientation of a contour by taking the sign of an area. By default, the parameter is
false, which means that the absolute value is returned.
 */
CV_EXPORTS_W double contourArea( InputArray contour, bool oriented = false );

/** @brief Finds a rotated rectangle of the minimum area enclosing the input 2D point set.

The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a
specified point set. See the OpenCV sample minarea.cpp . Developer should keep in mind that the
returned rotatedRect can contain negative indices when data is close to the containing Mat element
boundary.

@param points Input vector of 2D points, stored in std::vector\<\> or Mat
 */
CV_EXPORTS_W RotatedRect minAreaRect( InputArray points );

/** @brief Finds the four vertices of a rotated rect. Useful to draw the rotated rectangle.

The function finds the four vertices of a rotated rectangle. This function is useful to draw the
rectangle. In C++, instead of using this function, you can directly use box.points() method. Please
visit the [tutorial on bounding
rectangle](http://docs.opencv.org/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html#bounding-rects-circles)
for more information.

@param box The input rotated rectangle. It may be the output of
@param points The output array of four vertices of rectangles.
 */
CV_EXPORTS_W void boxPoints(RotatedRect box, OutputArray points);

/** @brief Finds a circle of the minimum area enclosing a 2D point set.

The function finds the minimal enclosing circle of a 2D point set using an iterative algorithm. See
the OpenCV sample minarea.cpp .

@param points Input vector of 2D points, stored in std::vector\<\> or Mat
@param center Output center of the circle.
@param radius Output radius of the circle.
 */
CV_EXPORTS_W void minEnclosingCircle( InputArray points,
                                      CV_OUT Point2f& center, CV_OUT float& radius );

/** @example minarea.cpp
  */

/** @brief Finds a triangle of minimum area enclosing a 2D point set and returns its area.

The function finds a triangle of minimum area enclosing the given set of 2D points and returns its
area. The output for a given 2D point set is shown in the image below. 2D points are depicted in
*red* and the enclosing triangle in *yellow*.

![Sample output of the minimum enclosing triangle function](pics/minenclosingtriangle.png)

The implementation of the algorithm is based on O'Rourke's @cite ORourke86 and Klee and Laskowski's
@cite KleeLaskowski85 papers. O'Rourke provides a \f$\theta(n)\f$ algorithm for finding the minimal
enclosing triangle of a 2D convex polygon with n vertices. Since the minEnclosingTriangle function
takes a 2D point set as input an additional preprocessing step of computing the convex hull of the
2D point set is required. The complexity of the convexHull function is \f$O(n log(n))\f$ which is higher
than \f$\theta(n)\f$. Thus the overall complexity of the function is \f$O(n log(n))\f$.

@param points Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector\<\> or Mat
@param triangle Output vector of three 2D points defining the vertices of the triangle. The depth
of the OutputArray must be CV_32F.
 */
CV_EXPORTS_W double minEnclosingTriangle( InputArray points, CV_OUT OutputArray triangle );

/** @brief Compares two shapes.

The function compares two shapes. All three implemented methods use the Hu invariants (see cv::HuMoments)

@param contour1 First contour or grayscale image.
@param contour2 Second contour or grayscale image.
@param method Comparison method, see cv::ShapeMatchModes
@param parameter Method-specific parameter (not supported now).
 */
CV_EXPORTS_W double matchShapes( InputArray contour1, InputArray contour2,
                                 int method, double parameter );

/** @example convexhull.cpp
An example using the convexHull functionality
*/

/** @brief Finds the convex hull of a point set.

The function cv::convexHull finds the convex hull of a 2D point set using the Sklansky's algorithm @cite Sklansky82
that has *O(N logN)* complexity in the current implementation. See the OpenCV sample convexhull.cpp
that demonstrates the usage of different function variants.

@param points Input 2D point set, stored in std::vector or Mat.
@param hull Output convex hull. It is either an integer vector of indices or vector of points. In
the first case, the hull elements are 0-based indices of the convex hull points in the original
array (since the set of convex hull points is a subset of the original point set). In the second
case, hull elements are the convex hull points themselves.
@param clockwise Orientation flag. If it is true, the output convex hull is oriented clockwise.
Otherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing
to the right, and its Y axis pointing upwards.
@param returnPoints Operation flag. In case of a matrix, when the flag is true, the function
returns convex hull points. Otherwise, it returns indices of the convex hull points. When the
output array is std::vector, the flag is ignored, and the output depends on the type of the
vector: std::vector\<int\> implies returnPoints=false, std::vector\<Point\> implies
returnPoints=true.
 */
CV_EXPORTS_W void convexHull( InputArray points, OutputArray hull,
                              bool clockwise = false, bool returnPoints = true );

/** @brief Finds the convexity defects of a contour.

The figure below displays convexity defects of a hand contour:

![image](pics/defects.png)

@param contour Input contour.
@param convexhull Convex hull obtained using convexHull that should contain indices of the contour
points that make the hull.
@param convexityDefects The output vector of convexity defects. In C++ and the new Python/Java
interface each convexity defect is represented as 4-element integer vector (a.k.a. cv::Vec4i):
(start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices
in the original contour of the convexity defect beginning, end and the farthest point, and
fixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the
farthest contour point and the hull. That is, to get the floating-point value of the depth will be
fixpt_depth/256.0.
 */
CV_EXPORTS_W void convexityDefects( InputArray contour, InputArray convexhull, OutputArray convexityDefects );

/** @brief Tests a contour convexity.

The function tests whether the input contour is convex or not. The contour must be simple, that is,
without self-intersections. Otherwise, the function output is undefined.

@param contour Input vector of 2D points, stored in std::vector\<\> or Mat
 */
CV_EXPORTS_W bool isContourConvex( InputArray contour );

//! finds intersection of two convex polygons
CV_EXPORTS_W float intersectConvexConvex( InputArray _p1, InputArray _p2,
                                          OutputArray _p12, bool handleNested = true );

/** @example fitellipse.cpp
  An example using the fitEllipse technique
*/

/** @brief Fits an ellipse around a set of 2D points.

The function calculates the ellipse that fits (in a least-squares sense) a set of 2D points best of
all. It returns the rotated rectangle in which the ellipse is inscribed. The first algorithm described by @cite Fitzgibbon95
is used. Developer should keep in mind that it is possible that the returned
ellipse/rotatedRect data contains negative indices, due to the data points being close to the
border of the containing Mat element.

@param points Input 2D point set, stored in std::vector\<\> or Mat
 */
CV_EXPORTS_W RotatedRect fitEllipse( InputArray points );

/** @brief Fits a line to a 2D or 3D point set.

The function fitLine fits a line to a 2D or 3D point set by minimizing \f$\sum_i \rho(r_i)\f$ where
\f$r_i\f$ is a distance between the \f$i^{th}\f$ point, the line and \f$\rho(r)\f$ is a distance function, one
of the following:
-  DIST_L2
\f[\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}\f]
- DIST_L1
\f[\rho (r) = r\f]
- DIST_L12
\f[\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)\f]
- DIST_FAIR
\f[\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998\f]
- DIST_WELSCH
\f[\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846\f]
- DIST_HUBER
\f[\rho (r) =  \fork{r^2/2}{if \(r < C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345\f]

The algorithm is based on the M-estimator ( <http://en.wikipedia.org/wiki/M-estimator> ) technique
that iteratively fits the line using the weighted least-squares algorithm. After each iteration the
weights \f$w_i\f$ are adjusted to be inversely proportional to \f$\rho(r_i)\f$ .

@param points Input vector of 2D or 3D points, stored in std::vector\<\> or Mat.
@param line Output line parameters. In case of 2D fitting, it should be a vector of 4 elements
(like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and
(x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like
Vec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line
and (x0, y0, z0) is a point on the line.
@param distType Distance used by the M-estimator, see cv::DistanceTypes
@param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value
is chosen.
@param reps Sufficient accuracy for the radius (distance between the coordinate origin and the line).
@param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.
 */
CV_EXPORTS_W void fitLine( InputArray points, OutputArray line, int distType,
                           double param, double reps, double aeps );

/** @brief Performs a point-in-contour test.

The function determines whether the point is inside a contour, outside, or lies on an edge (or
coincides with a vertex). It returns positive (inside), negative (outside), or zero (on an edge)
value, correspondingly. When measureDist=false , the return value is +1, -1, and 0, respectively.
Otherwise, the return value is a signed distance between the point and the nearest contour edge.

See below a sample output of the function where each image pixel is tested against the contour:

![sample output](pics/pointpolygon.png)

@param contour Input contour.
@param pt Point tested against the contour.
@param measureDist If true, the function estimates the signed distance from the point to the
nearest contour edge. Otherwise, the function only checks if the point is inside a contour or not.
 */
CV_EXPORTS_W double pointPolygonTest( InputArray contour, Point2f pt, bool measureDist );

/** @brief Finds out if there is any intersection between two rotated rectangles.

If there is then the vertices of the intersecting region are returned as well.

Below are some examples of intersection configurations. The hatched pattern indicates the
intersecting region and the red vertices are returned by the function.

![intersection examples](pics/intersection.png)

@param rect1 First rectangle
@param rect2 Second rectangle
@param intersectingRegion The output array of the verticies of the intersecting region. It returns
at most 8 vertices. Stored as std::vector\<cv::Point2f\> or cv::Mat as Mx1 of type CV_32FC2.
@returns One of cv::RectanglesIntersectTypes
 */
CV_EXPORTS_W int rotatedRectangleIntersection( const RotatedRect& rect1, const RotatedRect& rect2, OutputArray intersectingRegion  );

//! @} imgproc_shape

CV_EXPORTS_W Ptr<CLAHE> createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8));

//! Ballard, D.H. (1981). Generalizing the Hough transform to detect arbitrary shapes. Pattern Recognition 13 (2): 111-122.
//! Detects position only without translation and rotation
CV_EXPORTS Ptr<GeneralizedHoughBallard> createGeneralizedHoughBallard();

//! Guil, N., González-Linares, J.M. and Zapata, E.L. (1999). Bidimensional shape detection using an invariant approach. Pattern Recognition 32 (6): 1025-1038.
//! Detects position, translation and rotation
CV_EXPORTS Ptr<GeneralizedHoughGuil> createGeneralizedHoughGuil();

//! Performs linear blending of two images:
//! \f[ \texttt{dst}(i,j) = \texttt{weights1}(i,j)*\texttt{src1}(i,j) + \texttt{weights2}(i,j)*\texttt{src2}(i,j) \f]
//! @param src1 It has a type of CV_8UC(n) or CV_32FC(n), where n is a positive integer.
//! @param src2 It has the same type and size as src1.
//! @param weights1 It has a type of CV_32FC1 and the same size with src1.
//! @param weights2 It has a type of CV_32FC1 and the same size with src1.
//! @param dst It is created if it does not have the same size and type with src1.
CV_EXPORTS void blendLinear(InputArray src1, InputArray src2, InputArray weights1, InputArray weights2, OutputArray dst);

//! @addtogroup imgproc_colormap
//! @{

//! GNU Octave/MATLAB equivalent colormaps
enum ColormapTypes
{
    COLORMAP_AUTUMN = 0, //!< ![autumn](pics/colormaps/colorscale_autumn.jpg)
    COLORMAP_BONE = 1, //!< ![bone](pics/colormaps/colorscale_bone.jpg)
    COLORMAP_JET = 2, //!< ![jet](pics/colormaps/colorscale_jet.jpg)
    COLORMAP_WINTER = 3, //!< ![winter](pics/colormaps/colorscale_winter.jpg)
    COLORMAP_RAINBOW = 4, //!< ![rainbow](pics/colormaps/colorscale_rainbow.jpg)
    COLORMAP_OCEAN = 5, //!< ![ocean](pics/colormaps/colorscale_ocean.jpg)
    COLORMAP_SUMMER = 6, //!< ![summer](pics/colormaps/colorscale_summer.jpg)
    COLORMAP_SPRING = 7, //!< ![spring](pics/colormaps/colorscale_spring.jpg)
    COLORMAP_COOL = 8, //!< ![cool](pics/colormaps/colorscale_cool.jpg)
    COLORMAP_HSV = 9, //!< ![HSV](pics/colormaps/colorscale_hsv.jpg)
    COLORMAP_PINK = 10, //!< ![pink](pics/colormaps/colorscale_pink.jpg)
    COLORMAP_HOT = 11, //!< ![hot](pics/colormaps/colorscale_hot.jpg)
    COLORMAP_PARULA = 12 //!< ![parula](pics/colormaps/colorscale_parula.jpg)
};

/** @example falsecolor.cpp
An example using applyColorMap function
*/
/** @brief Applies a GNU Octave/MATLAB equivalent colormap on a given image.

@param src The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.
@param dst The result is the colormapped source image. Note: Mat::create is called on dst.
@param colormap The colormap to apply, see cv::ColormapTypes
*/
CV_EXPORTS_W void applyColorMap(InputArray src, OutputArray dst, int colormap);

/** @brief Applies a user colormap on a given image.

@param src The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.
@param dst The result is the colormapped source image. Note: Mat::create is called on dst.
@param userColor The colormap to apply of type CV_8UC1 or CV_8UC3 and size 256
*/
CV_EXPORTS_W void applyColorMap(InputArray src, OutputArray dst, InputArray userColor);

//! @} imgproc_colormap

//! @addtogroup imgproc_draw
//! @{

/** @brief Draws a line segment connecting two points.

The function line draws the line segment between pt1 and pt2 points in the image. The line is
clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
lines are drawn using Gaussian filtering.

@param img Image.
@param pt1 First point of the line segment.
@param pt2 Second point of the line segment.
@param color Line color.
@param thickness Line thickness.
@param lineType Type of the line, see cv::LineTypes.
@param shift Number of fractional bits in the point coordinates.
 */
CV_EXPORTS_W void line(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,
                     int thickness = 1, int lineType = LINE_8, int shift = 0);

/** @brief Draws a arrow segment pointing from the first point to the second one.

The function arrowedLine draws an arrow between pt1 and pt2 points in the image. See also cv::line.

@param img Image.
@param pt1 The point the arrow starts from.
@param pt2 The point the arrow points to.
@param color Line color.
@param thickness Line thickness.
@param line_type Type of the line, see cv::LineTypes
@param shift Number of fractional bits in the point coordinates.
@param tipLength The length of the arrow tip in relation to the arrow length
 */
CV_EXPORTS_W void arrowedLine(InputOutputArray img, Point pt1, Point pt2, const Scalar& color,
                     int thickness=1, int line_type=8, int shift=0, double tipLength=0.1);

/** @brief Draws a simple, thick, or filled up-right rectangle.

The function rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
are pt1 and pt2.

@param img Image.
@param pt1 Vertex of the rectangle.
@param pt2 Vertex of the rectangle opposite to pt1 .
@param color Rectangle color or brightness (grayscale image).
@param thickness Thickness of lines that make up the rectangle. Negative values, like CV_FILLED ,
mean that the function has to draw a filled rectangle.
@param lineType Type of the line. See the line description.
@param shift Number of fractional bits in the point coordinates.
 */
CV_EXPORTS_W void rectangle(InputOutputArray img, Point pt1, Point pt2,
                          const Scalar& color, int thickness = 1,
                          int lineType = LINE_8, int shift = 0);

/** @overload

use `rec` parameter as alternative specification of the drawn rectangle: `r.tl() and
r.br()-Point(1,1)` are opposite corners
*/
CV_EXPORTS void rectangle(CV_IN_OUT Mat& img, Rect rec,
                          const Scalar& color, int thickness = 1,
                          int lineType = LINE_8, int shift = 0);

/** @example Drawing_2.cpp
An example using drawing functions
 */
/** @brief Draws a circle.

The function circle draws a simple or filled circle with a given center and radius.
@param img Image where the circle is drawn.
@param center Center of the circle.
@param radius Radius of the circle.
@param color Circle color.
@param thickness Thickness of the circle outline, if positive. Negative thickness means that a
filled circle is to be drawn.
@param lineType Type of the circle boundary. See the line description.
@param shift Number of fractional bits in the coordinates of the center and in the radius value.
 */
CV_EXPORTS_W void circle(InputOutputArray img, Point center, int radius,
                       const Scalar& color, int thickness = 1,
                       int lineType = LINE_8, int shift = 0);

/** @brief Draws a simple or thick elliptic arc or fills an ellipse sector.

The function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic
arc, or a filled ellipse sector. The drawing code uses general parametric form.
A piecewise-linear curve is used to approximate the elliptic arc
boundary. If you need more control of the ellipse rendering, you can retrieve the curve using
cv::ellipse2Poly and then render it with polylines or fill it with cv::fillPoly. If you use the first
variant of the function and want to draw the whole ellipse, not an arc, pass `startAngle=0` and
`endAngle=360`. If `startAngle` is greater than `endAngle`, they are swapped. The figure below explains
the meaning of the parameters to draw the blue arc.

![Parameters of Elliptic Arc](pics/ellipse.svg)

@param img Image.
@param center Center of the ellipse.
@param axes Half of the size of the ellipse main axes.
@param angle Ellipse rotation angle in degrees.
@param startAngle Starting angle of the elliptic arc in degrees.
@param endAngle Ending angle of the elliptic arc in degrees.
@param color Ellipse color.
@param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
a filled ellipse sector is to be drawn.
@param lineType Type of the ellipse boundary. See the line description.
@param shift Number of fractional bits in the coordinates of the center and values of axes.
 */
CV_EXPORTS_W void ellipse(InputOutputArray img, Point center, Size axes,
                        double angle, double startAngle, double endAngle,
                        const Scalar& color, int thickness = 1,
                        int lineType = LINE_8, int shift = 0);

/** @overload
@param img Image.
@param box Alternative ellipse representation via RotatedRect. This means that the function draws
an ellipse inscribed in the rotated rectangle.
@param color Ellipse color.
@param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that
a filled ellipse sector is to be drawn.
@param lineType Type of the ellipse boundary. See the line description.
*/
CV_EXPORTS_W void ellipse(InputOutputArray img, const RotatedRect& box, const Scalar& color,
                        int thickness = 1, int lineType = LINE_8);

/* ----------------------------------------------------------------------------------------- */
/* ADDING A SET OF PREDEFINED MARKERS WHICH COULD BE USED TO HIGHLIGHT POSITIONS IN AN IMAGE */
/* ----------------------------------------------------------------------------------------- */

//! Possible set of marker types used for the cv::drawMarker function
enum MarkerTypes
{
    MARKER_CROSS = 0,           //!< A crosshair marker shape
    MARKER_TILTED_CROSS = 1,    //!< A 45 degree tilted crosshair marker shape
    MARKER_STAR = 2,            //!< A star marker shape, combination of cross and tilted cross
    MARKER_DIAMOND = 3,         //!< A diamond marker shape
    MARKER_SQUARE = 4,          //!< A square marker shape
    MARKER_TRIANGLE_UP = 5,     //!< An upwards pointing triangle marker shape
    MARKER_TRIANGLE_DOWN = 6    //!< A downwards pointing triangle marker shape
};

/** @brief Draws a marker on a predefined position in an image.

The function drawMarker draws a marker on a given position in the image. For the moment several
marker types are supported, see cv::MarkerTypes for more information.

@param img Image.
@param position The point where the crosshair is positioned.
@param color Line color.
@param markerType The specific type of marker you want to use, see cv::MarkerTypes
@param thickness Line thickness.
@param line_type Type of the line, see cv::LineTypes
@param markerSize The length of the marker axis [default = 20 pixels]
 */
CV_EXPORTS_W void drawMarker(CV_IN_OUT Mat& img, Point position, const Scalar& color,
                             int markerType = MARKER_CROSS, int markerSize=20, int thickness=1,
                             int line_type=8);

/* ----------------------------------------------------------------------------------------- */
/* END OF MARKER SECTION */
/* ----------------------------------------------------------------------------------------- */

/** @overload */
CV_EXPORTS void fillConvexPoly(Mat& img, const Point* pts, int npts,
                               const Scalar& color, int lineType = LINE_8,
                               int shift = 0);

/** @brief Fills a convex polygon.

The function fillConvexPoly draws a filled convex polygon. This function is much faster than the
function cv::fillPoly . It can fill not only convex polygons but any monotonic polygon without
self-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)
twice at the most (though, its top-most and/or the bottom edge could be horizontal).

@param img Image.
@param points Polygon vertices.
@param color Polygon color.
@param lineType Type of the polygon boundaries. See the line description.
@param shift Number of fractional bits in the vertex coordinates.
 */
CV_EXPORTS_W void fillConvexPoly(InputOutputArray img, InputArray points,
                                 const Scalar& color, int lineType = LINE_8,
                                 int shift = 0);

/** @overload */
CV_EXPORTS void fillPoly(Mat& img, const Point** pts,
                         const int* npts, int ncontours,
                         const Scalar& color, int lineType = LINE_8, int shift = 0,
                         Point offset = Point() );

/** @example Drawing_1.cpp
An example using drawing functions
 */
/** @brief Fills the area bounded by one or more polygons.

The function fillPoly fills an area bounded by several polygonal contours. The function can fill
complex areas, for example, areas with holes, contours with self-intersections (some of their
parts), and so forth.

@param img Image.
@param pts Array of polygons where each polygon is represented as an array of points.
@param color Polygon color.
@param lineType Type of the polygon boundaries. See the line description.
@param shift Number of fractional bits in the vertex coordinates.
@param offset Optional offset of all points of the contours.
 */
CV_EXPORTS_W void fillPoly(InputOutputArray img, InputArrayOfArrays pts,
                           const Scalar& color, int lineType = LINE_8, int shift = 0,
                           Point offset = Point() );

/** @overload */
CV_EXPORTS void polylines(Mat& img, const Point* const* pts, const int* npts,
                          int ncontours, bool isClosed, const Scalar& color,
                          int thickness = 1, int lineType = LINE_8, int shift = 0 );

/** @brief Draws several polygonal curves.

@param img Image.
@param pts Array of polygonal curves.
@param isClosed Flag indicating whether the drawn polylines are closed or not. If they are closed,
the function draws a line from the last vertex of each curve to its first vertex.
@param color Polyline color.
@param thickness Thickness of the polyline edges.
@param lineType Type of the line segments. See the line description.
@param shift Number of fractional bits in the vertex coordinates.

The function polylines draws one or more polygonal curves.
 */
CV_EXPORTS_W void polylines(InputOutputArray img, InputArrayOfArrays pts,
                            bool isClosed, const Scalar& color,
                            int thickness = 1, int lineType = LINE_8, int shift = 0 );

/** @example contours2.cpp
  An example using the drawContour functionality
*/

/** @example segment_objects.cpp
An example using drawContours to clean up a background segmentation result
 */

/** @brief Draws contours outlines or filled contours.

The function draws contour outlines in the image if \f$\texttt{thickness} \ge 0\f$ or fills the area
bounded by the contours if \f$\texttt{thickness}<0\f$ . The example below shows how to retrieve
connected components from the binary image and label them: :
@code
    #include "opencv2/imgproc.hpp"
    #include "opencv2/highgui.hpp"

    using namespace cv;
    using namespace std;

    int main( int argc, char** argv )
    {
        Mat src;
        // the first command-line parameter must be a filename of the binary
        // (black-n-white) image
        if( argc != 2 || !(src=imread(argv[1], 0)).data)
            return -1;

        Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);

        src = src > 1;
        namedWindow( "Source", 1 );
        imshow( "Source", src );

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        findContours( src, contours, hierarchy,
            RETR_CCOMP, CHAIN_APPROX_SIMPLE );

        // iterate through all the top-level contours,
        // draw each connected component with its own random color
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0] )
        {
            Scalar color( rand()&255, rand()&255, rand()&255 );
            drawContours( dst, contours, idx, color, FILLED, 8, hierarchy );
        }

        namedWindow( "Components", 1 );
        imshow( "Components", dst );
        waitKey(0);
    }
@endcode

@param image Destination image.
@param contours All the input contours. Each contour is stored as a point vector.
@param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.
@param color Color of the contours.
@param thickness Thickness of lines the contours are drawn with. If it is negative (for example,
thickness=CV_FILLED ), the contour interiors are drawn.
@param lineType Line connectivity. See cv::LineTypes.
@param hierarchy Optional information about hierarchy. It is only needed if you want to draw only
some of the contours (see maxLevel ).
@param maxLevel Maximal level for drawn contours. If it is 0, only the specified contour is drawn.
If it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function
draws the contours, all the nested contours, all the nested-to-nested contours, and so on. This
parameter is only taken into account when there is hierarchy available.
@param offset Optional contour shift parameter. Shift all the drawn contours by the specified
\f$\texttt{offset}=(dx,dy)\f$ .
 */
CV_EXPORTS_W void drawContours( InputOutputArray image, InputArrayOfArrays contours,
                              int contourIdx, const Scalar& color,
                              int thickness = 1, int lineType = LINE_8,
                              InputArray hierarchy = noArray(),
                              int maxLevel = INT_MAX, Point offset = Point() );

/** @brief Clips the line against the image rectangle.

The function cv::clipLine calculates a part of the line segment that is entirely within the specified
rectangle. it returns false if the line segment is completely outside the rectangle. Otherwise,
it returns true .
@param imgSize Image size. The image rectangle is Rect(0, 0, imgSize.width, imgSize.height) .
@param pt1 First line point.
@param pt2 Second line point.
 */
CV_EXPORTS bool clipLine(Size imgSize, CV_IN_OUT Point& pt1, CV_IN_OUT Point& pt2);

/** @overload
@param imgSize Image size. The image rectangle is Rect(0, 0, imgSize.width, imgSize.height) .
@param pt1 First line point.
@param pt2 Second line point.
*/
CV_EXPORTS bool clipLine(Size2l imgSize, CV_IN_OUT Point2l& pt1, CV_IN_OUT Point2l& pt2);

/** @overload
@param imgRect Image rectangle.
@param pt1 First line point.
@param pt2 Second line point.
*/
CV_EXPORTS_W bool clipLine(Rect imgRect, CV_OUT CV_IN_OUT Point& pt1, CV_OUT CV_IN_OUT Point& pt2);

/** @brief Approximates an elliptic arc with a polyline.

The function ellipse2Poly computes the vertices of a polyline that approximates the specified
elliptic arc. It is used by cv::ellipse. If `arcStart` is greater than `arcEnd`, they are swapped.

@param center Center of the arc.
@param axes Half of the size of the ellipse main axes. See the ellipse for details.
@param angle Rotation angle of the ellipse in degrees. See the ellipse for details.
@param arcStart Starting angle of the elliptic arc in degrees.
@param arcEnd Ending angle of the elliptic arc in degrees.
@param delta Angle between the subsequent polyline vertices. It defines the approximation
accuracy.
@param pts Output vector of polyline vertices.
 */
CV_EXPORTS_W void ellipse2Poly( Point center, Size axes, int angle,
                                int arcStart, int arcEnd, int delta,
                                CV_OUT std::vector<Point>& pts );

/** @overload
@param center Center of the arc.
@param axes Half of the size of the ellipse main axes. See the ellipse for details.
@param angle Rotation angle of the ellipse in degrees. See the ellipse for details.
@param arcStart Starting angle of the elliptic arc in degrees.
@param arcEnd Ending angle of the elliptic arc in degrees.
@param delta Angle between the subsequent polyline vertices. It defines the approximation
accuracy.
@param pts Output vector of polyline vertices.
*/
CV_EXPORTS void ellipse2Poly(Point2d center, Size2d axes, int angle,
                             int arcStart, int arcEnd, int delta,
                             CV_OUT std::vector<Point2d>& pts);

/** @brief Draws a text string.

The function putText renders the specified text string in the image. Symbols that cannot be rendered
using the specified font are replaced by question marks. See getTextSize for a text rendering code
example.

@param img Image.
@param text Text string to be drawn.
@param org Bottom-left corner of the text string in the image.
@param fontFace Font type, see cv::HersheyFonts.
@param fontScale Font scale factor that is multiplied by the font-specific base size.
@param color Text color.
@param thickness Thickness of the lines used to draw a text.
@param lineType Line type. See the line for details.
@param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,
it is at the top-left corner.
 */
CV_EXPORTS_W void putText( InputOutputArray img, const String& text, Point org,
                         int fontFace, double fontScale, Scalar color,
                         int thickness = 1, int lineType = LINE_8,
                         bool bottomLeftOrigin = false );

/** @brief Calculates the width and height of a text string.

The function getTextSize calculates and returns the size of a box that contains the specified text.
That is, the following code renders some text, the tight box surrounding it, and the baseline: :
@code
    String text = "Funny text inside the box";
    int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    double fontScale = 2;
    int thickness = 3;

    Mat img(600, 800, CV_8UC3, Scalar::all(0));

    int baseline=0;
    Size textSize = getTextSize(text, fontFace,
                                fontScale, thickness, &baseline);
    baseline += thickness;

    // center the text
    Point textOrg((img.cols - textSize.width)/2,
                  (img.rows + textSize.height)/2);

    // draw the box
    rectangle(img, textOrg + Point(0, baseline),
              textOrg + Point(textSize.width, -textSize.height),
              Scalar(0,0,255));
    // ... and the baseline first
    line(img, textOrg + Point(0, thickness),
         textOrg + Point(textSize.width, thickness),
         Scalar(0, 0, 255));

    // then put the text itself
    putText(img, text, textOrg, fontFace, fontScale,
            Scalar::all(255), thickness, 8);
@endcode

@param text Input text string.
@param fontFace Font to use, see cv::HersheyFonts.
@param fontScale Font scale factor that is multiplied by the font-specific base size.
@param thickness Thickness of lines used to render the text. See putText for details.
@param[out] baseLine y-coordinate of the baseline relative to the bottom-most text
point.
@return The size of a box that contains the specified text.

@see cv::putText
 */
CV_EXPORTS_W Size getTextSize(const String& text, int fontFace,
                            double fontScale, int thickness,
                            CV_OUT int* baseLine);

/** @brief Line iterator

The class is used to iterate over all the pixels on the raster line
segment connecting two specified points.

The class LineIterator is used to get each pixel of a raster line. It
can be treated as versatile implementation of the Bresenham algorithm
where you can stop at each pixel and do some extra processing, for
example, grab pixel values along the line or draw a line with an effect
(for example, with XOR operation).

The number of pixels along the line is stored in LineIterator::count.
The method LineIterator::pos returns the current position in the image:

@code{.cpp}
// grabs pixels along the line (pt1, pt2)
// from 8-bit 3-channel image to the buffer
LineIterator it(img, pt1, pt2, 8);
LineIterator it2 = it;
vector<Vec3b> buf(it.count);

for(int i = 0; i < it.count; i++, ++it)
    buf[i] = *(const Vec3b)*it;

// alternative way of iterating through the line
for(int i = 0; i < it2.count; i++, ++it2)
{
    Vec3b val = img.at<Vec3b>(it2.pos());
    CV_Assert(buf[i] == val);
}
@endcode
*/
class CV_EXPORTS LineIterator
{
public:
    /** @brief intializes the iterator

    creates iterators for the line connecting pt1 and pt2
    the line will be clipped on the image boundaries
    the line is 8-connected or 4-connected
    If leftToRight=true, then the iteration is always done
    from the left-most point to the right most,
    not to depend on the ordering of pt1 and pt2 parameters
    */
    LineIterator( const Mat& img, Point pt1, Point pt2,
                  int connectivity = 8, bool leftToRight = false );
    /** @brief returns pointer to the current pixel
    */
    uchar* operator *();
    /** @brief prefix increment operator (++it). shifts iterator to the next pixel
    */
    LineIterator& operator ++();
    /** @brief postfix increment operator (it++). shifts iterator to the next pixel
    */
    LineIterator operator ++(int);
    /** @brief returns coordinates of the current pixel
    */
    Point pos() const;

    uchar* ptr;
    const uchar* ptr0;
    int step, elemSize;
    int err, count;
    int minusDelta, plusDelta;
    int minusStep, plusStep;
};

//! @cond IGNORED

// === LineIterator implementation ===

inline
uchar* LineIterator::operator *()
{
    return ptr;
}

inline
LineIterator& LineIterator::operator ++()
{
    int mask = err < 0 ? -1 : 0;
    err += minusDelta + (plusDelta & mask);
    ptr += minusStep + (plusStep & mask);
    return *this;
}

inline
LineIterator LineIterator::operator ++(int)
{
    LineIterator it = *this;
    ++(*this);
    return it;
}

inline
Point LineIterator::pos() const
{
    Point p;
    p.y = (int)((ptr - ptr0)/step);
    p.x = (int)(((ptr - ptr0) - p.y*step)/elemSize);
    return p;
}

//! @endcond

//! @} imgproc_draw

//! @} imgproc

} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/imgproc/imgproc_c.h"
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
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

#ifndef OPENCV_ML_HPP
#define OPENCV_ML_HPP

#ifdef __cplusplus
#  include "opencv2/core.hpp"
#endif

#ifdef __cplusplus

#include <float.h>
#include <map>
#include <iostream>

/**
  @defgroup ml Machine Learning

  The Machine Learning Library (MLL) is a set of classes and functions for statistical
  classification, regression, and clustering of data.

  Most of the classification and regression algorithms are implemented as C++ classes. As the
  algorithms have different sets of features (like an ability to handle missing measurements or
  categorical input variables), there is a little common ground between the classes. This common
  ground is defined by the class cv::ml::StatModel that all the other ML classes are derived from.

  See detailed overview here: @ref ml_intro.
 */

namespace cv
{

namespace ml
{

//! @addtogroup ml
//! @{

/** @brief Variable types */
enum VariableTypes
{
    VAR_NUMERICAL    =0, //!< same as VAR_ORDERED
    VAR_ORDERED      =0, //!< ordered variables
    VAR_CATEGORICAL  =1  //!< categorical variables
};

/** @brief %Error types */
enum ErrorTypes
{
    TEST_ERROR = 0,
    TRAIN_ERROR = 1
};

/** @brief Sample types */
enum SampleTypes
{
    ROW_SAMPLE = 0, //!< each training sample is a row of samples
    COL_SAMPLE = 1  //!< each training sample occupies a column of samples
};

/** @brief The structure represents the logarithmic grid range of statmodel parameters.

It is used for optimizing statmodel accuracy by varying model parameters, the accuracy estimate
being computed by cross-validation.
 */
class CV_EXPORTS_W ParamGrid
{
public:
    /** @brief Default constructor */
    ParamGrid();
    /** @brief Constructor with parameters */
    ParamGrid(double _minVal, double _maxVal, double _logStep);

    CV_PROP_RW double minVal; //!< Minimum value of the statmodel parameter. Default value is 0.
    CV_PROP_RW double maxVal; //!< Maximum value of the statmodel parameter. Default value is 0.
    /** @brief Logarithmic step for iterating the statmodel parameter.

    The grid determines the following iteration sequence of the statmodel parameter values:
    \f[(minVal, minVal*step, minVal*{step}^2, \dots,  minVal*{logStep}^n),\f]
    where \f$n\f$ is the maximal index satisfying
    \f[\texttt{minVal} * \texttt{logStep} ^n <  \texttt{maxVal}\f]
    The grid is logarithmic, so logStep must always be greater then 1. Default value is 1.
    */
    CV_PROP_RW double logStep;

    /** @brief Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method

    @param minVal minimum value of the parameter grid
    @param maxVal maximum value of the parameter grid
    @param logstep Logarithmic step for iterating the statmodel parameter
    */
    CV_WRAP static Ptr<ParamGrid> create(double minVal=0., double maxVal=0., double logstep=1.);
};

/** @brief Class encapsulating training data.

Please note that the class only specifies the interface of training data, but not implementation.
All the statistical model classes in _ml_ module accepts Ptr\<TrainData\> as parameter. In other
words, you can create your own class derived from TrainData and pass smart pointer to the instance
of this class into StatModel::train.

@sa @ref ml_intro_data
 */
class CV_EXPORTS_W TrainData
{
public:
    static inline float missingValue() { return FLT_MAX; }
    virtual ~TrainData();

    CV_WRAP virtual int getLayout() const = 0;
    CV_WRAP virtual int getNTrainSamples() const = 0;
    CV_WRAP virtual int getNTestSamples() const = 0;
    CV_WRAP virtual int getNSamples() const = 0;
    CV_WRAP virtual int getNVars() const = 0;
    CV_WRAP virtual int getNAllVars() const = 0;

    CV_WRAP virtual void getSample(InputArray varIdx, int sidx, float* buf) const = 0;
    CV_WRAP virtual Mat getSamples() const = 0;
    CV_WRAP virtual Mat getMissing() const = 0;

    /** @brief Returns matrix of train samples

    @param layout The requested layout. If it's different from the initial one, the matrix is
        transposed. See ml::SampleTypes.
    @param compressSamples if true, the function returns only the training samples (specified by
        sampleIdx)
    @param compressVars if true, the function returns the shorter training samples, containing only
        the active variables.

    In current implementation the function tries to avoid physical data copying and returns the
    matrix stored inside TrainData (unless the transposition or compression is needed).
     */
    CV_WRAP virtual Mat getTrainSamples(int layout=ROW_SAMPLE,
                                bool compressSamples=true,
                                bool compressVars=true) const = 0;

    /** @brief Returns the vector of responses

    The function returns ordered or the original categorical responses. Usually it's used in
    regression algorithms.
     */
    CV_WRAP virtual Mat getTrainResponses() const = 0;

    /** @brief Returns the vector of normalized categorical responses

    The function returns vector of responses. Each response is integer from `0` to `<number of
    classes>-1`. The actual label value can be retrieved then from the class label vector, see
    TrainData::getClassLabels.
     */
    CV_WRAP virtual Mat getTrainNormCatResponses() const = 0;
    CV_WRAP virtual Mat getTestResponses() const = 0;
    CV_WRAP virtual Mat getTestNormCatResponses() const = 0;
    CV_WRAP virtual Mat getResponses() const = 0;
    CV_WRAP virtual Mat getNormCatResponses() const = 0;
    CV_WRAP virtual Mat getSampleWeights() const = 0;
    CV_WRAP virtual Mat getTrainSampleWeights() const = 0;
    CV_WRAP virtual Mat getTestSampleWeights() const = 0;
    CV_WRAP virtual Mat getVarIdx() const = 0;
    CV_WRAP virtual Mat getVarType() const = 0;
    CV_WRAP Mat getVarSymbolFlags() const;
    CV_WRAP virtual int getResponseType() const = 0;
    CV_WRAP virtual Mat getTrainSampleIdx() const = 0;
    CV_WRAP virtual Mat getTestSampleIdx() const = 0;
    CV_WRAP virtual void getValues(int vi, InputArray sidx, float* values) const = 0;
    virtual void getNormCatValues(int vi, InputArray sidx, int* values) const = 0;
    CV_WRAP virtual Mat getDefaultSubstValues() const = 0;

    CV_WRAP virtual int getCatCount(int vi) const = 0;

    /** @brief Returns the vector of class labels

    The function returns vector of unique labels occurred in the responses.
     */
    CV_WRAP virtual Mat getClassLabels() const = 0;

    CV_WRAP virtual Mat getCatOfs() const = 0;
    CV_WRAP virtual Mat getCatMap() const = 0;

    /** @brief Splits the training data into the training and test parts
    @sa TrainData::setTrainTestSplitRatio
     */
    CV_WRAP virtual void setTrainTestSplit(int count, bool shuffle=true) = 0;

    /** @brief Splits the training data into the training and test parts

    The function selects a subset of specified relative size and then returns it as the training
    set. If the function is not called, all the data is used for training. Please, note that for
    each of TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test
    subset can be retrieved and processed as well.
    @sa TrainData::setTrainTestSplit
     */
    CV_WRAP virtual void setTrainTestSplitRatio(double ratio, bool shuffle=true) = 0;
    CV_WRAP virtual void shuffleTrainTest() = 0;

    /** @brief Returns matrix of test samples */
    CV_WRAP Mat getTestSamples() const;

    /** @brief Returns vector of symbolic names captured in loadFromCSV() */
    CV_WRAP void getNames(std::vector<String>& names) const;

    CV_WRAP static Mat getSubVector(const Mat& vec, const Mat& idx);

    /** @brief Reads the dataset from a .csv file and returns the ready-to-use training data.

    @param filename The input file name
    @param headerLineCount The number of lines in the beginning to skip; besides the header, the
        function also skips empty lines and lines staring with `#`
    @param responseStartIdx Index of the first output variable. If -1, the function considers the
        last variable as the response
    @param responseEndIdx Index of the last output variable + 1. If -1, then there is single
        response variable at responseStartIdx.
    @param varTypeSpec The optional text string that specifies the variables' types. It has the
        format `ord[n1-n2,n3,n4-n5,...]cat[n6,n7-n8,...]`. That is, variables from `n1 to n2`
        (inclusive range), `n3`, `n4 to n5` ... are considered ordered and `n6`, `n7 to n8` ... are
        considered as categorical. The range `[n1..n2] + [n3] + [n4..n5] + ... + [n6] + [n7..n8]`
        should cover all the variables. If varTypeSpec is not specified, then algorithm uses the
        following rules:
        - all input variables are considered ordered by default. If some column contains has non-
          numerical values, e.g. 'apple', 'pear', 'apple', 'apple', 'mango', the corresponding
          variable is considered categorical.
        - if there are several output variables, they are all considered as ordered. Error is
          reported when non-numerical values are used.
        - if there is a single output variable, then if its values are non-numerical or are all
          integers, then it's considered categorical. Otherwise, it's considered ordered.
    @param delimiter The character used to separate values in each line.
    @param missch The character used to specify missing measurements. It should not be a digit.
        Although it's a non-numerical value, it surely does not affect the decision of whether the
        variable ordered or categorical.
    @note If the dataset only contains input variables and no responses, use responseStartIdx = -2
        and responseEndIdx = 0. The output variables vector will just contain zeros.
     */
    static Ptr<TrainData> loadFromCSV(const String& filename,
                                      int headerLineCount,
                                      int responseStartIdx=-1,
                                      int responseEndIdx=-1,
                                      const String& varTypeSpec=String(),
                                      char delimiter=',',
                                      char missch='?');

    /** @brief Creates training data from in-memory arrays.

    @param samples matrix of samples. It should have CV_32F type.
    @param layout see ml::SampleTypes.
    @param responses matrix of responses. If the responses are scalar, they should be stored as a
        single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
        former case the responses are considered as ordered by default; in the latter case - as
        categorical)
    @param varIdx vector specifying which variables to use for training. It can be an integer vector
        (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
        active variables.
    @param sampleIdx vector specifying which samples to use for training. It can be an integer
        vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
        of training samples.
    @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
    @param varType optional vector of type CV_8U and size `<number_of_variables_in_samples> +
        <number_of_variables_in_responses>`, containing types of each input and output variable. See
        ml::VariableTypes.
     */
    CV_WRAP static Ptr<TrainData> create(InputArray samples, int layout, InputArray responses,
                                 InputArray varIdx=noArray(), InputArray sampleIdx=noArray(),
                                 InputArray sampleWeights=noArray(), InputArray varType=noArray());
};

/** @brief Base class for statistical models in OpenCV ML.
 */
class CV_EXPORTS_W StatModel : public Algorithm
{
public:
    /** Predict options */
    enum Flags {
        UPDATE_MODEL = 1,
        RAW_OUTPUT=1, //!< makes the method return the raw results (the sum), not the class label
        COMPRESSED_INPUT=2,
        PREPROCESSED_INPUT=4
    };

    /** @brief Returns the number of variables in training samples */
    CV_WRAP virtual int getVarCount() const = 0;

    CV_WRAP virtual bool empty() const;

    /** @brief Returns true if the model is trained */
    CV_WRAP virtual bool isTrained() const = 0;
    /** @brief Returns true if the model is classifier */
    CV_WRAP virtual bool isClassifier() const = 0;

    /** @brief Trains the statistical model

    @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
        created with TrainData::create.
    @param flags optional flags, depending on the model. Some of the models can be updated with the
        new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
     */
    CV_WRAP virtual bool train( const Ptr<TrainData>& trainData, int flags=0 );

    /** @brief Trains the statistical model

    @param samples training samples
    @param layout See ml::SampleTypes.
    @param responses vector of responses associated with the training samples.
    */
    CV_WRAP virtual bool train( InputArray samples, int layout, InputArray responses );

    /** @brief Computes error on the training or test dataset

    @param data the training data
    @param test if true, the error is computed over the test subset of the data, otherwise it's
        computed over the training subset of the data. Please note that if you loaded a completely
        different dataset to evaluate already trained classifier, you will probably want not to set
        the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so
        that the error is computed for the whole new set. Yes, this sounds a bit confusing.
    @param resp the optional output responses.

    The method uses StatModel::predict to compute the error. For regression models the error is
    computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
     */
    CV_WRAP virtual float calcError( const Ptr<TrainData>& data, bool test, OutputArray resp ) const;

    /** @brief Predicts response(s) for the provided sample(s)

    @param samples The input samples, floating-point matrix
    @param results The optional output matrix of results.
    @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
     */
    CV_WRAP virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;

    /** @brief Create and train model with default parameters

    The class must implement static `create()` method with no parameters or with all default parameter values
    */
    template<typename _Tp> static Ptr<_Tp> train(const Ptr<TrainData>& data, int flags=0)
    {
        Ptr<_Tp> model = _Tp::create();
        return !model.empty() && model->train(data, flags) ? model : Ptr<_Tp>();
    }
};

/****************************************************************************************\
*                                 Normal Bayes Classifier                                *
\****************************************************************************************/

/** @brief Bayes classifier for normally distributed data.

@sa @ref ml_intro_bayes
 */
class CV_EXPORTS_W NormalBayesClassifier : public StatModel
{
public:
    /** @brief Predicts the response for sample(s).

    The method estimates the most probable classes for input vectors. Input vectors (one or more)
    are stored as rows of the matrix inputs. In case of multiple input vectors, there should be one
    output vector outputs. The predicted class for a single input vector is returned by the method.
    The vector outputProbs contains the output probabilities corresponding to each element of
    result.
     */
    CV_WRAP virtual float predictProb( InputArray inputs, OutputArray outputs,
                               OutputArray outputProbs, int flags=0 ) const = 0;

    /** Creates empty model
    Use StatModel::train to train the model after creation. */
    CV_WRAP static Ptr<NormalBayesClassifier> create();

    /** @brief Loads and creates a serialized NormalBayesClassifier from a file
     *
     * Use NormalBayesClassifier::save to serialize and store an NormalBayesClassifier to disk.
     * Load the NormalBayesClassifier from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized NormalBayesClassifier
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<NormalBayesClassifier> load(const String& filepath , const String& nodeName = String());
};

/****************************************************************************************\
*                          K-Nearest Neighbour Classifier                                *
\****************************************************************************************/

/** @brief The class implements K-Nearest Neighbors model

@sa @ref ml_intro_knn
 */
class CV_EXPORTS_W KNearest : public StatModel
{
public:

    /** Default number of neighbors to use in predict method. */
    /** @see setDefaultK */
    CV_WRAP virtual int getDefaultK() const = 0;
    /** @copybrief getDefaultK @see getDefaultK */
    CV_WRAP virtual void setDefaultK(int val) = 0;

    /** Whether classification or regression model should be trained. */
    /** @see setIsClassifier */
    CV_WRAP virtual bool getIsClassifier() const = 0;
    /** @copybrief getIsClassifier @see getIsClassifier */
    CV_WRAP virtual void setIsClassifier(bool val) = 0;

    /** Parameter for KDTree implementation. */
    /** @see setEmax */
    CV_WRAP virtual int getEmax() const = 0;
    /** @copybrief getEmax @see getEmax */
    CV_WRAP virtual void setEmax(int val) = 0;

    /** %Algorithm type, one of KNearest::Types. */
    /** @see setAlgorithmType */
    CV_WRAP virtual int getAlgorithmType() const = 0;
    /** @copybrief getAlgorithmType @see getAlgorithmType */
    CV_WRAP virtual void setAlgorithmType(int val) = 0;

    /** @brief Finds the neighbors and predicts responses for input vectors.

    @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
        `<number_of_samples> * k` size.
    @param k Number of used nearest neighbors. Should be greater than 1.
    @param results Vector with results of prediction (regression or classification) for each input
        sample. It is a single-precision floating-point vector with `<number_of_samples>` elements.
    @param neighborResponses Optional output values for corresponding neighbors. It is a single-
        precision floating-point matrix of `<number_of_samples> * k` size.
    @param dist Optional output distances from the input vectors to the corresponding neighbors. It
        is a single-precision floating-point matrix of `<number_of_samples> * k` size.

    For each input vector (a row of the matrix samples), the method finds the k nearest neighbors.
    In case of regression, the predicted result is a mean value of the particular vector's neighbor
    responses. In case of classification, the class is determined by voting.

    For each input vector, the neighbors are sorted by their distances to the vector.

    In case of C++ interface you can use output pointers to empty matrices and the function will
    allocate memory itself.

    If only a single input vector is passed, all output matrices are optional and the predicted
    value is returned by the method.

    The function is parallelized with the TBB library.
     */
    CV_WRAP virtual float findNearest( InputArray samples, int k,
                               OutputArray results,
                               OutputArray neighborResponses=noArray(),
                               OutputArray dist=noArray() ) const = 0;

    /** @brief Implementations of KNearest algorithm
       */
    enum Types
    {
        BRUTE_FORCE=1,
        KDTREE=2
    };

    /** @brief Creates the empty model

    The static method creates empty %KNearest classifier. It should be then trained using StatModel::train method.
     */
    CV_WRAP static Ptr<KNearest> create();
};

/****************************************************************************************\
*                                   Support Vector Machines                              *
\****************************************************************************************/

/** @brief Support Vector Machines.

@sa @ref ml_intro_svm
 */
class CV_EXPORTS_W SVM : public StatModel
{
public:

    class CV_EXPORTS Kernel : public Algorithm
    {
    public:
        virtual int getType() const = 0;
        virtual void calc( int vcount, int n, const float* vecs, const float* another, float* results ) = 0;
    };

    /** Type of a %SVM formulation.
    See SVM::Types. Default value is SVM::C_SVC. */
    /** @see setType */
    CV_WRAP virtual int getType() const = 0;
    /** @copybrief getType @see getType */
    CV_WRAP virtual void setType(int val) = 0;

    /** Parameter \f$\gamma\f$ of a kernel function.
    For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1. */
    /** @see setGamma */
    CV_WRAP virtual double getGamma() const = 0;
    /** @copybrief getGamma @see getGamma */
    CV_WRAP virtual void setGamma(double val) = 0;

    /** Parameter _coef0_ of a kernel function.
    For SVM::POLY or SVM::SIGMOID. Default value is 0.*/
    /** @see setCoef0 */
    CV_WRAP virtual double getCoef0() const = 0;
    /** @copybrief getCoef0 @see getCoef0 */
    CV_WRAP virtual void setCoef0(double val) = 0;

    /** Parameter _degree_ of a kernel function.
    For SVM::POLY. Default value is 0. */
    /** @see setDegree */
    CV_WRAP virtual double getDegree() const = 0;
    /** @copybrief getDegree @see getDegree */
    CV_WRAP virtual void setDegree(double val) = 0;

    /** Parameter _C_ of a %SVM optimization problem.
    For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0. */
    /** @see setC */
    CV_WRAP virtual double getC() const = 0;
    /** @copybrief getC @see getC */
    CV_WRAP virtual void setC(double val) = 0;

    /** Parameter \f$\nu\f$ of a %SVM optimization problem.
    For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0. */
    /** @see setNu */
    CV_WRAP virtual double getNu() const = 0;
    /** @copybrief getNu @see getNu */
    CV_WRAP virtual void setNu(double val) = 0;

    /** Parameter \f$\epsilon\f$ of a %SVM optimization problem.
    For SVM::EPS_SVR. Default value is 0. */
    /** @see setP */
    CV_WRAP virtual double getP() const = 0;
    /** @copybrief getP @see getP */
    CV_WRAP virtual void setP(double val) = 0;

    /** Optional weights in the SVM::C_SVC problem, assigned to particular classes.
    They are multiplied by _C_ so the parameter _C_ of class _i_ becomes `classWeights(i) * C`. Thus
    these weights affect the misclassification penalty for different classes. The larger weight,
    the larger penalty on misclassification of data from the corresponding class. Default value is
    empty Mat. */
    /** @see setClassWeights */
    CV_WRAP virtual cv::Mat getClassWeights() const = 0;
    /** @copybrief getClassWeights @see getClassWeights */
    CV_WRAP virtual void setClassWeights(const cv::Mat &val) = 0;

    /** Termination criteria of the iterative %SVM training procedure which solves a partial
    case of constrained quadratic optimization problem.
    You can specify tolerance and/or the maximum number of iterations. Default value is
    `TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON )`; */
    /** @see setTermCriteria */
    CV_WRAP virtual cv::TermCriteria getTermCriteria() const = 0;
    /** @copybrief getTermCriteria @see getTermCriteria */
    CV_WRAP virtual void setTermCriteria(const cv::TermCriteria &val) = 0;

    /** Type of a %SVM kernel.
    See SVM::KernelTypes. Default value is SVM::RBF. */
    CV_WRAP virtual int getKernelType() const = 0;

    /** Initialize with one of predefined kernels.
    See SVM::KernelTypes. */
    CV_WRAP virtual void setKernel(int kernelType) = 0;

    /** Initialize with custom kernel.
    See SVM::Kernel class for implementation details */
    virtual void setCustomKernel(const Ptr<Kernel> &_kernel) = 0;

    //! %SVM type
    enum Types {
        /** C-Support Vector Classification. n-class classification (n \f$\geq\f$ 2), allows
        imperfect separation of classes with penalty multiplier C for outliers. */
        C_SVC=100,
        /** \f$\nu\f$-Support Vector Classification. n-class classification with possible
        imperfect separation. Parameter \f$\nu\f$ (in the range 0..1, the larger the value, the smoother
        the decision boundary) is used instead of C. */
        NU_SVC=101,
        /** Distribution Estimation (One-class %SVM). All the training data are from
        the same class, %SVM builds a boundary that separates the class from the rest of the feature
        space. */
        ONE_CLASS=102,
        /** \f$\epsilon\f$-Support Vector Regression. The distance between feature vectors
        from the training set and the fitting hyper-plane must be less than p. For outliers the
        penalty multiplier C is used. */
        EPS_SVR=103,
        /** \f$\nu\f$-Support Vector Regression. \f$\nu\f$ is used instead of p.
        See @cite LibSVM for details. */
        NU_SVR=104
    };

    /** @brief %SVM kernel type

    A comparison of different kernels on the following 2D test case with four classes. Four
    SVM::C_SVC SVMs have been trained (one against rest) with auto_train. Evaluation on three
    different kernels (SVM::CHI2, SVM::INTER, SVM::RBF). The color depicts the class with max score.
    Bright means max-score \> 0, dark means max-score \< 0.
    ![image](pics/SVM_Comparison.png)
    */
    enum KernelTypes {
        /** Returned by SVM::getKernelType in case when custom kernel has been set */
        CUSTOM=-1,
        /** Linear kernel. No mapping is done, linear discrimination (or regression) is
        done in the original feature space. It is the fastest option. \f$K(x_i, x_j) = x_i^T x_j\f$. */
        LINEAR=0,
        /** Polynomial kernel:
        \f$K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0\f$. */
        POLY=1,
        /** Radial basis function (RBF), a good choice in most cases.
        \f$K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0\f$. */
        RBF=2,
        /** Sigmoid kernel: \f$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0)\f$. */
        SIGMOID=3,
        /** Exponential Chi2 kernel, similar to the RBF kernel:
        \f$K(x_i, x_j) = e^{-\gamma \chi^2(x_i,x_j)}, \chi^2(x_i,x_j) = (x_i-x_j)^2/(x_i+x_j), \gamma > 0\f$. */
        CHI2=4,
        /** Histogram intersection kernel. A fast kernel. \f$K(x_i, x_j) = min(x_i,x_j)\f$. */
        INTER=5
    };

    //! %SVM params type
    enum ParamTypes {
        C=0,
        GAMMA=1,
        P=2,
        NU=3,
        COEF=4,
        DEGREE=5
    };

    /** @brief Trains an %SVM with optimal parameters.

    @param data the training data that can be constructed using TrainData::create or
        TrainData::loadFromCSV.
    @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
        subset is used to test the model, the others form the train set. So, the %SVM algorithm is
        executed kFold times.
    @param Cgrid grid for C
    @param gammaGrid grid for gamma
    @param pGrid grid for p
    @param nuGrid grid for nu
    @param coeffGrid grid for coeff
    @param degreeGrid grid for degree
    @param balanced If true and the problem is 2-class classification then the method creates more
        balanced cross-validation subsets that is proportions between classes in subsets are close
        to such proportion in the whole train dataset.

    The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
    nu, coef0, degree. Parameters are considered optimal when the cross-validation
    estimate of the test set error is minimal.

    If there is no need to optimize a parameter, the corresponding grid step should be set to any
    value less than or equal to 1. For example, to avoid optimization in gamma, set `gammaGrid.step
    = 0`, `gammaGrid.minVal`, `gamma_grid.maxVal` as arbitrary numbers. In this case, the value
    `Gamma` is taken for gamma.

    And, finally, if the optimization in a parameter is required but the corresponding grid is
    unknown, you may call the function SVM::getDefaultGrid. To generate a grid, for example, for
    gamma, call `SVM::getDefaultGrid(SVM::GAMMA)`.

    This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
    regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
    the usual %SVM with parameters specified in params is executed.
     */
    virtual bool trainAuto( const Ptr<TrainData>& data, int kFold = 10,
                    ParamGrid Cgrid = getDefaultGrid(C),
                    ParamGrid gammaGrid  = getDefaultGrid(GAMMA),
                    ParamGrid pGrid      = getDefaultGrid(P),
                    ParamGrid nuGrid     = getDefaultGrid(NU),
                    ParamGrid coeffGrid  = getDefaultGrid(COEF),
                    ParamGrid degreeGrid = getDefaultGrid(DEGREE),
                    bool balanced=false) = 0;

    /** @brief Trains an %SVM with optimal parameters

    @param samples training samples
    @param layout See ml::SampleTypes.
    @param responses vector of responses associated with the training samples.
    @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
        subset is used to test the model, the others form the train set. So, the %SVM algorithm is
    @param Cgrid grid for C
    @param gammaGrid grid for gamma
    @param pGrid grid for p
    @param nuGrid grid for nu
    @param coeffGrid grid for coeff
    @param degreeGrid grid for degree
    @param balanced If true and the problem is 2-class classification then the method creates more
        balanced cross-validation subsets that is proportions between classes in subsets are close
        to such proportion in the whole train dataset.

    The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
    nu, coef0, degree. Parameters are considered optimal when the cross-validation
    estimate of the test set error is minimal.

    This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
    offers rudimentary parameter options.

    This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
    regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
    the usual %SVM with parameters specified in params is executed.
    */
    CV_WRAP bool trainAuto(InputArray samples,
            int layout,
            InputArray responses,
            int kFold = 10,
            Ptr<ParamGrid> Cgrid = SVM::getDefaultGridPtr(SVM::C),
            Ptr<ParamGrid> gammaGrid  = SVM::getDefaultGridPtr(SVM::GAMMA),
            Ptr<ParamGrid> pGrid      = SVM::getDefaultGridPtr(SVM::P),
            Ptr<ParamGrid> nuGrid     = SVM::getDefaultGridPtr(SVM::NU),
            Ptr<ParamGrid> coeffGrid  = SVM::getDefaultGridPtr(SVM::COEF),
            Ptr<ParamGrid> degreeGrid = SVM::getDefaultGridPtr(SVM::DEGREE),
            bool balanced=false);

    /** @brief Retrieves all the support vectors

    The method returns all the support vectors as a floating-point matrix, where support vectors are
    stored as matrix rows.
     */
    CV_WRAP virtual Mat getSupportVectors() const = 0;

    /** @brief Retrieves all the uncompressed support vectors of a linear %SVM

    The method returns all the uncompressed support vectors of a linear %SVM that the compressed
    support vector, used for prediction, was derived from. They are returned in a floating-point
    matrix, where the support vectors are stored as matrix rows.
     */
    CV_WRAP Mat getUncompressedSupportVectors() const;

    /** @brief Retrieves the decision function

    @param i the index of the decision function. If the problem solved is regression, 1-class or
        2-class classification, then there will be just one decision function and the index should
        always be 0. Otherwise, in the case of N-class classification, there will be \f$N(N-1)/2\f$
        decision functions.
    @param alpha the optional output vector for weights, corresponding to different support vectors.
        In the case of linear %SVM all the alpha's will be 1's.
    @param svidx the optional output vector of indices of support vectors within the matrix of
        support vectors (which can be retrieved by SVM::getSupportVectors). In the case of linear
        %SVM each decision function consists of a single "compressed" support vector.

    The method returns rho parameter of the decision function, a scalar subtracted from the weighted
    sum of kernel responses.
     */
    CV_WRAP virtual double getDecisionFunction(int i, OutputArray alpha, OutputArray svidx) const = 0;

    /** @brief Generates a grid for %SVM parameters.

    @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is
    generated for the parameter with this ID.

    The function generates a grid for the specified parameter of the %SVM algorithm. The grid may be
    passed to the function SVM::trainAuto.
     */
    static ParamGrid getDefaultGrid( int param_id );

    /** @brief Generates a grid for %SVM parameters.

    @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is
    generated for the parameter with this ID.

    The function generates a grid pointer for the specified parameter of the %SVM algorithm.
    The grid may be passed to the function SVM::trainAuto.
     */
    CV_WRAP static Ptr<ParamGrid> getDefaultGridPtr( int param_id );

    /** Creates empty model.
    Use StatModel::train to train the model. Since %SVM has several parameters, you may want to
    find the best parameters for your problem, it can be done with SVM::trainAuto. */
    CV_WRAP static Ptr<SVM> create();

    /** @brief Loads and creates a serialized svm from a file
     *
     * Use SVM::save to serialize and store an SVM to disk.
     * Load the SVM from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized svm
     */
    CV_WRAP static Ptr<SVM> load(const String& filepath);
};

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/

/** @brief The class implements the Expectation Maximization algorithm.

@sa @ref ml_intro_em
 */
class CV_EXPORTS_W EM : public StatModel
{
public:
    //! Type of covariation matrices
    enum Types {
        /** A scaled identity matrix \f$\mu_k * I\f$. There is the only
        parameter \f$\mu_k\f$ to be estimated for each matrix. The option may be used in special cases,
        when the constraint is relevant, or as a first step in the optimization (for example in case
        when the data is preprocessed with PCA). The results of such preliminary estimation may be
        passed again to the optimization procedure, this time with
        covMatType=EM::COV_MAT_DIAGONAL. */
        COV_MAT_SPHERICAL=0,
        /** A diagonal matrix with positive diagonal elements. The number of
        free parameters is d for each matrix. This is most commonly used option yielding good
        estimation results. */
        COV_MAT_DIAGONAL=1,
        /** A symmetric positively defined matrix. The number of free
        parameters in each matrix is about \f$d^2/2\f$. It is not recommended to use this option, unless
        there is pretty accurate initial estimation of the parameters and/or a huge number of
        training samples. */
        COV_MAT_GENERIC=2,
        COV_MAT_DEFAULT=COV_MAT_DIAGONAL
    };

    //! Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};

    //! The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

    /** The number of mixture components in the Gaussian mixture model.
    Default value of the parameter is EM::DEFAULT_NCLUSTERS=5. Some of %EM implementation could
    determine the optimal number of mixtures within a specified value range, but that is not the
    case in ML yet. */
    /** @see setClustersNumber */
    CV_WRAP virtual int getClustersNumber() const = 0;
    /** @copybrief getClustersNumber @see getClustersNumber */
    CV_WRAP virtual void setClustersNumber(int val) = 0;

    /** Constraint on covariance matrices which defines type of matrices.
    See EM::Types. */
    /** @see setCovarianceMatrixType */
    CV_WRAP virtual int getCovarianceMatrixType() const = 0;
    /** @copybrief getCovarianceMatrixType @see getCovarianceMatrixType */
    CV_WRAP virtual void setCovarianceMatrixType(int val) = 0;

    /** The termination criteria of the %EM algorithm.
    The %EM algorithm can be terminated by the number of iterations termCrit.maxCount (number of
    M-steps) or when relative change of likelihood logarithm is less than termCrit.epsilon. Default
    maximum number of iterations is EM::DEFAULT_MAX_ITERS=100. */
    /** @see setTermCriteria */
    CV_WRAP virtual TermCriteria getTermCriteria() const = 0;
    /** @copybrief getTermCriteria @see getTermCriteria */
    CV_WRAP virtual void setTermCriteria(const TermCriteria &val) = 0;

    /** @brief Returns weights of the mixtures

    Returns vector with the number of elements equal to the number of mixtures.
     */
    CV_WRAP virtual Mat getWeights() const = 0;
    /** @brief Returns the cluster centers (means of the Gaussian mixture)

    Returns matrix with the number of rows equal to the number of mixtures and number of columns
    equal to the space dimensionality.
     */
    CV_WRAP virtual Mat getMeans() const = 0;
    /** @brief Returns covariation matrices

    Returns vector of covariation matrices. Number of matrices is the number of gaussian mixtures,
    each matrix is a square floating-point matrix NxN, where N is the space dimensionality.
     */
    CV_WRAP virtual void getCovs(CV_OUT std::vector<Mat>& covs) const = 0;

    /** @brief Returns posterior probabilities for the provided samples

    @param samples The input samples, floating-point matrix
    @param results The optional output \f$ nSamples \times nClusters\f$ matrix of results. It contains
    posterior probabilities for each sample from the input
    @param flags This parameter will be ignored
     */
    CV_WRAP virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;

    /** @brief Returns a likelihood logarithm value and an index of the most probable mixture component
    for the given sample.

    @param sample A sample for classification. It should be a one-channel matrix of
        \f$1 \times dims\f$ or \f$dims \times 1\f$ size.
    @param probs Optional output matrix that contains posterior probabilities of each component
        given the sample. It has \f$1 \times nclusters\f$ size and CV_64FC1 type.

    The method returns a two-element double vector. Zero element is a likelihood logarithm value for
    the sample. First element is an index of the most probable mixture component for the given
    sample.
     */
    CV_WRAP virtual Vec2d predict2(InputArray sample, OutputArray probs) const = 0;

    /** @brief Estimate the Gaussian mixture parameters from a samples set.

    This variation starts with Expectation step. Initial values of the model parameters will be
    estimated by the k-means algorithm.

    Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
    responses (class labels or function values) as input. Instead, it computes the *Maximum
    Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
    parameters inside the structure: \f$p_{i,k}\f$ in probs, \f$a_k\f$ in means , \f$S_k\f$ in
    covs[k], \f$\pi_k\f$ in weights , and optionally computes the output "class label" for each
    sample: \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most
    probable mixture component for each sample).

    The trained model can be used further for prediction, just like any other classifier. The
    trained model is similar to the NormalBayesClassifier.

    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        it will be converted to the inner matrix of such type for the further computing.
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
        \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable
        mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and
        CV_64FC1 type.
     */
    CV_WRAP virtual bool trainEM(InputArray samples,
                         OutputArray logLikelihoods=noArray(),
                         OutputArray labels=noArray(),
                         OutputArray probs=noArray()) = 0;

    /** @brief Estimate the Gaussian mixture parameters from a samples set.

    This variation starts with Expectation step. You need to provide initial means \f$a_k\f$ of
    mixture components. Optionally you can pass initial weights \f$\pi_k\f$ and covariance matrices
    \f$S_k\f$ of mixture components.

    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        it will be converted to the inner matrix of such type for the further computing.
    @param means0 Initial means \f$a_k\f$ of mixture components. It is a one-channel matrix of
        \f$nclusters \times dims\f$ size. If the matrix does not have CV_64F type it will be
        converted to the inner matrix of such type for the further computing.
    @param covs0 The vector of initial covariance matrices \f$S_k\f$ of mixture components. Each of
        covariance matrices is a one-channel matrix of \f$dims \times dims\f$ size. If the matrices
        do not have CV_64F type they will be converted to the inner matrices of such type for the
        further computing.
    @param weights0 Initial weights \f$\pi_k\f$ of mixture components. It should be a one-channel
        floating-point matrix with \f$1 \times nclusters\f$ or \f$nclusters \times 1\f$ size.
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
        \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable
        mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and
        CV_64FC1 type.
    */
    CV_WRAP virtual bool trainE(InputArray samples, InputArray means0,
                        InputArray covs0=noArray(),
                        InputArray weights0=noArray(),
                        OutputArray logLikelihoods=noArray(),
                        OutputArray labels=noArray(),
                        OutputArray probs=noArray()) = 0;

    /** @brief Estimate the Gaussian mixture parameters from a samples set.

    This variation starts with Maximization step. You need to provide initial probabilities
    \f$p_{i,k}\f$ to use this option.

    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        it will be converted to the inner matrix of such type for the further computing.
    @param probs0
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
        \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable
        mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and
        CV_64FC1 type.
    */
    CV_WRAP virtual bool trainM(InputArray samples, InputArray probs0,
                        OutputArray logLikelihoods=noArray(),
                        OutputArray labels=noArray(),
                        OutputArray probs=noArray()) = 0;

    /** Creates empty %EM model.
    The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
    can use one of the EM::train\* methods or load it from file using Algorithm::load\<EM\>(filename).
     */
    CV_WRAP static Ptr<EM> create();

    /** @brief Loads and creates a serialized EM from a file
     *
     * Use EM::save to serialize and store an EM to disk.
     * Load the EM from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized EM
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<EM> load(const String& filepath , const String& nodeName = String());
};

/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/

/** @brief The class represents a single decision tree or a collection of decision trees.

The current public interface of the class allows user to train only a single decision tree, however
the class is capable of storing multiple decision trees and using them for prediction (by summing
responses or using a voting schemes), and the derived from DTrees classes (such as RTrees and Boost)
use this capability to implement decision tree ensembles.

@sa @ref ml_intro_trees
*/
class CV_EXPORTS_W DTrees : public StatModel
{
public:
    /** Predict options */
    enum Flags { PREDICT_AUTO=0, PREDICT_SUM=(1<<8), PREDICT_MAX_VOTE=(2<<8), PREDICT_MASK=(3<<8) };

    /** Cluster possible values of a categorical variable into K\<=maxCategories clusters to
    find a suboptimal split.
    If a discrete variable, on which the training procedure tries to make a split, takes more than
    maxCategories values, the precise best subset estimation may take a very long time because the
    algorithm is exponential. Instead, many decision trees engines (including our implementation)
    try to find sub-optimal split in this case by clustering all the samples into maxCategories
    clusters that is some categories are merged together. The clustering is applied only in n \>
    2-class classification problems for categorical variables with N \> max_categories possible
    values. In case of regression and 2-class classification the optimal split can be found
    efficiently without employing clustering, thus the parameter is not used in these cases.
    Default value is 10.*/
    /** @see setMaxCategories */
    CV_WRAP virtual int getMaxCategories() const = 0;
    /** @copybrief getMaxCategories @see getMaxCategories */
    CV_WRAP virtual void setMaxCategories(int val) = 0;

    /** The maximum possible depth of the tree.
    That is the training algorithms attempts to split a node while its depth is less than maxDepth.
    The root node has zero depth. The actual depth may be smaller if the other termination criteria
    are met (see the outline of the training procedure @ref ml_intro_trees "here"), and/or if the
    tree is pruned. Default value is INT_MAX.*/
    /** @see setMaxDepth */
    CV_WRAP virtual int getMaxDepth() const = 0;
    /** @copybrief getMaxDepth @see getMaxDepth */
    CV_WRAP virtual void setMaxDepth(int val) = 0;

    /** If the number of samples in a node is less than this parameter then the node will not be split.

    Default value is 10.*/
    /** @see setMinSampleCount */
    CV_WRAP virtual int getMinSampleCount() const = 0;
    /** @copybrief getMinSampleCount @see getMinSampleCount */
    CV_WRAP virtual void setMinSampleCount(int val) = 0;

    /** If CVFolds \> 1 then algorithms prunes the built decision tree using K-fold
    cross-validation procedure where K is equal to CVFolds.
    Default value is 10.*/
    /** @see setCVFolds */
    CV_WRAP virtual int getCVFolds() const = 0;
    /** @copybrief getCVFolds @see getCVFolds */
    CV_WRAP virtual void setCVFolds(int val) = 0;

    /** If true then surrogate splits will be built.
    These splits allow to work with missing data and compute variable importance correctly.
    Default value is false.
    @note currently it's not implemented.*/
    /** @see setUseSurrogates */
    CV_WRAP virtual bool getUseSurrogates() const = 0;
    /** @copybrief getUseSurrogates @see getUseSurrogates */
    CV_WRAP virtual void setUseSurrogates(bool val) = 0;

    /** If true then a pruning will be harsher.
    This will make a tree more compact and more resistant to the training data noise but a bit less
    accurate. Default value is true.*/
    /** @see setUse1SERule */
    CV_WRAP virtual bool getUse1SERule() const = 0;
    /** @copybrief getUse1SERule @see getUse1SERule */
    CV_WRAP virtual void setUse1SERule(bool val) = 0;

    /** If true then pruned branches are physically removed from the tree.
    Otherwise they are retained and it is possible to get results from the original unpruned (or
    pruned less aggressively) tree. Default value is true.*/
    /** @see setTruncatePrunedTree */
    CV_WRAP virtual bool getTruncatePrunedTree() const = 0;
    /** @copybrief getTruncatePrunedTree @see getTruncatePrunedTree */
    CV_WRAP virtual void setTruncatePrunedTree(bool val) = 0;

    /** Termination criteria for regression trees.
    If all absolute differences between an estimated value in a node and values of train samples
    in this node are less than this parameter then the node will not be split further. Default
    value is 0.01f*/
    /** @see setRegressionAccuracy */
    CV_WRAP virtual float getRegressionAccuracy() const = 0;
    /** @copybrief getRegressionAccuracy @see getRegressionAccuracy */
    CV_WRAP virtual void setRegressionAccuracy(float val) = 0;

    /** @brief The array of a priori class probabilities, sorted by the class label value.

    The parameter can be used to tune the decision tree preferences toward a certain class. For
    example, if you want to detect some rare anomaly occurrence, the training base will likely
    contain much more normal cases than anomalies, so a very good classification performance
    will be achieved just by considering every case as normal. To avoid this, the priors can be
    specified, where the anomaly probability is artificially increased (up to 0.5 or even
    greater), so the weight of the misclassified anomalies becomes much bigger, and the tree is
    adjusted properly.

    You can also think about this parameter as weights of prediction categories which determine
    relative weights that you give to misclassification. That is, if the weight of the first
    category is 1 and the weight of the second category is 10, then each mistake in predicting
    the second category is equivalent to making 10 mistakes in predicting the first category.
    Default value is empty Mat.*/
    /** @see setPriors */
    CV_WRAP virtual cv::Mat getPriors() const = 0;
    /** @copybrief getPriors @see getPriors */
    CV_WRAP virtual void setPriors(const cv::Mat &val) = 0;

    /** @brief The class represents a decision tree node.
     */
    class CV_EXPORTS Node
    {
    public:
        Node();
        double value; //!< Value at the node: a class label in case of classification or estimated
                      //!< function value in case of regression.
        int classIdx; //!< Class index normalized to 0..class_count-1 range and assigned to the
                      //!< node. It is used internally in classification trees and tree ensembles.
        int parent; //!< Index of the parent node
        int left; //!< Index of the left child node
        int right; //!< Index of right child node
        int defaultDir; //!< Default direction where to go (-1: left or +1: right). It helps in the
                        //!< case of missing values.
        int split; //!< Index of the first split
    };

    /** @brief The class represents split in a decision tree.
     */
    class CV_EXPORTS Split
    {
    public:
        Split();
        int varIdx; //!< Index of variable on which the split is created.
        bool inversed; //!< If true, then the inverse split rule is used (i.e. left and right
                       //!< branches are exchanged in the rule expressions below).
        float quality; //!< The split quality, a positive number. It is used to choose the best split.
        int next; //!< Index of the next split in the list of splits for the node
        float c; /**< The threshold value in case of split on an ordered variable.
                      The rule is:
                      @code{.none}
                      if var_value < c
                        then next_node <- left
                        else next_node <- right
                      @endcode */
        int subsetOfs; /**< Offset of the bitset used by the split on a categorical variable.
                            The rule is:
                            @code{.none}
                            if bitset[var_value] == 1
                                then next_node <- left
                                else next_node <- right
                            @endcode */
    };

    /** @brief Returns indices of root nodes
    */
    virtual const std::vector<int>& getRoots() const = 0;
    /** @brief Returns all the nodes

    all the node indices are indices in the returned vector
     */
    virtual const std::vector<Node>& getNodes() const = 0;
    /** @brief Returns all the splits

    all the split indices are indices in the returned vector
     */
    virtual const std::vector<Split>& getSplits() const = 0;
    /** @brief Returns all the bitsets for categorical splits

    Split::subsetOfs is an offset in the returned vector
     */
    virtual const std::vector<int>& getSubsets() const = 0;

    /** @brief Creates the empty model

    The static method creates empty decision tree with the specified parameters. It should be then
    trained using train method (see StatModel::train). Alternatively, you can load the model from
    file using Algorithm::load\<DTrees\>(filename).
     */
    CV_WRAP static Ptr<DTrees> create();

    /** @brief Loads and creates a serialized DTrees from a file
     *
     * Use DTree::save to serialize and store an DTree to disk.
     * Load the DTree from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized DTree
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<DTrees> load(const String& filepath , const String& nodeName = String());
};

/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

/** @brief The class implements the random forest predictor.

@sa @ref ml_intro_rtrees
 */
class CV_EXPORTS_W RTrees : public DTrees
{
public:

    /** If true then variable importance will be calculated and then it can be retrieved by RTrees::getVarImportance.
    Default value is false.*/
    /** @see setCalculateVarImportance */
    CV_WRAP virtual bool getCalculateVarImportance() const = 0;
    /** @copybrief getCalculateVarImportance @see getCalculateVarImportance */
    CV_WRAP virtual void setCalculateVarImportance(bool val) = 0;

    /** The size of the randomly selected subset of features at each tree node and that are used
    to find the best split(s).
    If you set it to 0 then the size will be set to the square root of the total number of
    features. Default value is 0.*/
    /** @see setActiveVarCount */
    CV_WRAP virtual int getActiveVarCount() const = 0;
    /** @copybrief getActiveVarCount @see getActiveVarCount */
    CV_WRAP virtual void setActiveVarCount(int val) = 0;

    /** The termination criteria that specifies when the training algorithm stops.
    Either when the specified number of trees is trained and added to the ensemble or when
    sufficient accuracy (measured as OOB error) is achieved. Typically the more trees you have the
    better the accuracy. However, the improvement in accuracy generally diminishes and asymptotes
    pass a certain number of trees. Also to keep in mind, the number of tree increases the
    prediction time linearly. Default value is TermCriteria(TermCriteria::MAX_ITERS +
    TermCriteria::EPS, 50, 0.1)*/
    /** @see setTermCriteria */
    CV_WRAP virtual TermCriteria getTermCriteria() const = 0;
    /** @copybrief getTermCriteria @see getTermCriteria */
    CV_WRAP virtual void setTermCriteria(const TermCriteria &val) = 0;

    /** Returns the variable importance array.
    The method returns the variable importance vector, computed at the training stage when
    CalculateVarImportance is set to true. If this flag was set to false, the empty matrix is
    returned.
     */
    CV_WRAP virtual Mat getVarImportance() const = 0;

    /** Returns the result of each individual tree in the forest.
    In case the model is a regression problem, the method will return each of the trees'
    results for each of the sample cases. If the model is a classifier, it will return
    a Mat with samples + 1 rows, where the first row gives the class number and the
    following rows return the votes each class had for each sample.
        @param samples Array containg the samples for which votes will be calculated.
        @param results Array where the result of the calculation will be written.
        @param flags Flags for defining the type of RTrees.
    */
    CV_WRAP void getVotes(InputArray samples, OutputArray results, int flags) const;

    /** Creates the empty model.
    Use StatModel::train to train the model, StatModel::train to create and train the model,
    Algorithm::load to load the pre-trained model.
     */
    CV_WRAP static Ptr<RTrees> create();

    /** @brief Loads and creates a serialized RTree from a file
     *
     * Use RTree::save to serialize and store an RTree to disk.
     * Load the RTree from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized RTree
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<RTrees> load(const String& filepath , const String& nodeName = String());
};

/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/

/** @brief Boosted tree classifier derived from DTrees

@sa @ref ml_intro_boost
 */
class CV_EXPORTS_W Boost : public DTrees
{
public:
    /** Type of the boosting algorithm.
    See Boost::Types. Default value is Boost::REAL. */
    /** @see setBoostType */
    CV_WRAP virtual int getBoostType() const = 0;
    /** @copybrief getBoostType @see getBoostType */
    CV_WRAP virtual void setBoostType(int val) = 0;

    /** The number of weak classifiers.
    Default value is 100. */
    /** @see setWeakCount */
    CV_WRAP virtual int getWeakCount() const = 0;
    /** @copybrief getWeakCount @see getWeakCount */
    CV_WRAP virtual void setWeakCount(int val) = 0;

    /** A threshold between 0 and 1 used to save computational time.
    Samples with summary weight \f$\leq 1 - weight_trim_rate\f$ do not participate in the *next*
    iteration of training. Set this parameter to 0 to turn off this functionality. Default value is 0.95.*/
    /** @see setWeightTrimRate */
    CV_WRAP virtual double getWeightTrimRate() const = 0;
    /** @copybrief getWeightTrimRate @see getWeightTrimRate */
    CV_WRAP virtual void setWeightTrimRate(double val) = 0;

    /** Boosting type.
    Gentle AdaBoost and Real AdaBoost are often the preferable choices. */
    enum Types {
        DISCRETE=0, //!< Discrete AdaBoost.
        REAL=1, //!< Real AdaBoost. It is a technique that utilizes confidence-rated predictions
                //!< and works well with categorical data.
        LOGIT=2, //!< LogitBoost. It can produce good regression fits.
        GENTLE=3 //!< Gentle AdaBoost. It puts less weight on outlier data points and for that
                 //!<reason is often good with regression data.
    };

    /** Creates the empty model.
    Use StatModel::train to train the model, Algorithm::load\<Boost\>(filename) to load the pre-trained model. */
    CV_WRAP static Ptr<Boost> create();

    /** @brief Loads and creates a serialized Boost from a file
     *
     * Use Boost::save to serialize and store an RTree to disk.
     * Load the Boost from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized Boost
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<Boost> load(const String& filepath , const String& nodeName = String());
};

/****************************************************************************************\
*                                   Gradient Boosted Trees                               *
\****************************************************************************************/

/*class CV_EXPORTS_W GBTrees : public DTrees
{
public:
    struct CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
        CV_PROP_RW int weakCount;
        CV_PROP_RW int lossFunctionType;
        CV_PROP_RW float subsamplePortion;
        CV_PROP_RW float shrinkage;

        Params();
        Params( int lossFunctionType, int weakCount, float shrinkage,
                float subsamplePortion, int maxDepth, bool useSurrogates );
    };

    enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS};

    virtual void setK(int k) = 0;

    virtual float predictSerial( InputArray samples,
                                 OutputArray weakResponses, int flags) const = 0;

    static Ptr<GBTrees> create(const Params& p);
};*/

/****************************************************************************************\
*                              Artificial Neural Networks (ANN)                          *
\****************************************************************************************/

/////////////////////////////////// Multi-Layer Perceptrons //////////////////////////////

/** @brief Artificial Neural Networks - Multi-Layer Perceptrons.

Unlike many other models in ML that are constructed and trained at once, in the MLP model these
steps are separated. First, a network with the specified topology is created using the non-default
constructor or the method ANN_MLP::create. All the weights are set to zeros. Then, the network is
trained using a set of input and output vectors. The training procedure can be repeated more than
once, that is, the weights can be adjusted based on the new training data.

Additional flags for StatModel::train are available: ANN_MLP::TrainFlags.

@sa @ref ml_intro_ann
 */
class CV_EXPORTS_W ANN_MLP : public StatModel
{
public:
    /** Available training methods */
    enum TrainingMethods {
        BACKPROP=0, //!< The back-propagation algorithm.
        RPROP=1 //!< The RPROP algorithm. See @cite RPROP93 for details.
    };

    /** Sets training method and common parameters.
    @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
    @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP
    @param param2 passed to setRpropDWMin for ANN_MLP::RPROP and to setBackpropMomentumScale for ANN_MLP::BACKPROP.
    */
    CV_WRAP virtual void setTrainMethod(int method, double param1 = 0, double param2 = 0) = 0;

    /** Returns current training method */
    CV_WRAP virtual int getTrainMethod() const = 0;

    /** Initialize the activation function for each neuron.
    Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
    @param type The type of activation function. See ANN_MLP::ActivationFunctions.
    @param param1 The first parameter of the activation function, \f$\alpha\f$. Default value is 0.
    @param param2 The second parameter of the activation function, \f$\beta\f$. Default value is 0.
    */
    CV_WRAP virtual void setActivationFunction(int type, double param1 = 0, double param2 = 0) = 0;

    /**  Integer vector specifying the number of neurons in each layer including the input and output layers.
    The very first element specifies the number of elements in the input layer.
    The last element - number of elements in the output layer. Default value is empty Mat.
    @sa getLayerSizes */
    CV_WRAP virtual void setLayerSizes(InputArray _layer_sizes) = 0;

    /**  Integer vector specifying the number of neurons in each layer including the input and output layers.
    The very first element specifies the number of elements in the input layer.
    The last element - number of elements in the output layer.
    @sa setLayerSizes */
    CV_WRAP virtual cv::Mat getLayerSizes() const = 0;

    /** Termination criteria of the training algorithm.
    You can specify the maximum number of iterations (maxCount) and/or how much the error could
    change between the iterations to make the algorithm continue (epsilon). Default value is
    TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01).*/
    /** @see setTermCriteria */
    CV_WRAP virtual TermCriteria getTermCriteria() const = 0;
    /** @copybrief getTermCriteria @see getTermCriteria */
    CV_WRAP virtual void setTermCriteria(TermCriteria val) = 0;

    /** BPROP: Strength of the weight gradient term.
    The recommended value is about 0.1. Default value is 0.1.*/
    /** @see setBackpropWeightScale */
    CV_WRAP virtual double getBackpropWeightScale() const = 0;
    /** @copybrief getBackpropWeightScale @see getBackpropWeightScale */
    CV_WRAP virtual void setBackpropWeightScale(double val) = 0;

    /** BPROP: Strength of the momentum term (the difference between weights on the 2 previous iterations).
    This parameter provides some inertia to smooth the random fluctuations of the weights. It can
    vary from 0 (the feature is disabled) to 1 and beyond. The value 0.1 or so is good enough.
    Default value is 0.1.*/
    /** @see setBackpropMomentumScale */
    CV_WRAP virtual double getBackpropMomentumScale() const = 0;
    /** @copybrief getBackpropMomentumScale @see getBackpropMomentumScale */
    CV_WRAP virtual void setBackpropMomentumScale(double val) = 0;

    /** RPROP: Initial value \f$\Delta_0\f$ of update-values \f$\Delta_{ij}\f$.
    Default value is 0.1.*/
    /** @see setRpropDW0 */
    CV_WRAP virtual double getRpropDW0() const = 0;
    /** @copybrief getRpropDW0 @see getRpropDW0 */
    CV_WRAP virtual void setRpropDW0(double val) = 0;

    /** RPROP: Increase factor \f$\eta^+\f$.
    It must be \>1. Default value is 1.2.*/
    /** @see setRpropDWPlus */
    CV_WRAP virtual double getRpropDWPlus() const = 0;
    /** @copybrief getRpropDWPlus @see getRpropDWPlus */
    CV_WRAP virtual void setRpropDWPlus(double val) = 0;

    /** RPROP: Decrease factor \f$\eta^-\f$.
    It must be \<1. Default value is 0.5.*/
    /** @see setRpropDWMinus */
    CV_WRAP virtual double getRpropDWMinus() const = 0;
    /** @copybrief getRpropDWMinus @see getRpropDWMinus */
    CV_WRAP virtual void setRpropDWMinus(double val) = 0;

    /** RPROP: Update-values lower limit \f$\Delta_{min}\f$.
    It must be positive. Default value is FLT_EPSILON.*/
    /** @see setRpropDWMin */
    CV_WRAP virtual double getRpropDWMin() const = 0;
    /** @copybrief getRpropDWMin @see getRpropDWMin */
    CV_WRAP virtual void setRpropDWMin(double val) = 0;

    /** RPROP: Update-values upper limit \f$\Delta_{max}\f$.
    It must be \>1. Default value is 50.*/
    /** @see setRpropDWMax */
    CV_WRAP virtual double getRpropDWMax() const = 0;
    /** @copybrief getRpropDWMax @see getRpropDWMax */
    CV_WRAP virtual void setRpropDWMax(double val) = 0;

    /** possible activation functions */
    enum ActivationFunctions {
        /** Identity function: \f$f(x)=x\f$ */
        IDENTITY = 0,
        /** Symmetrical sigmoid: \f$f(x)=\beta*(1-e^{-\alpha x})/(1+e^{-\alpha x}\f$
        @note
        If you are using the default sigmoid activation function with the default parameter values
        fparam1=0 and fparam2=0 then the function used is y = 1.7159\*tanh(2/3 \* x), so the output
        will range from [-1.7159, 1.7159], instead of [0,1].*/
        SIGMOID_SYM = 1,
        /** Gaussian function: \f$f(x)=\beta e^{-\alpha x*x}\f$ */
        GAUSSIAN = 2
    };

    /** Train options */
    enum TrainFlags {
        /** Update the network weights, rather than compute them from scratch. In the latter case
        the weights are initialized using the Nguyen-Widrow algorithm. */
        UPDATE_WEIGHTS = 1,
        /** Do not normalize the input vectors. If this flag is not set, the training algorithm
        normalizes each input feature independently, shifting its mean value to 0 and making the
        standard deviation equal to 1. If the network is assumed to be updated frequently, the new
        training data could be much different from original one. In this case, you should take care
        of proper normalization. */
        NO_INPUT_SCALE = 2,
        /** Do not normalize the output vectors. If the flag is not set, the training algorithm
        normalizes each output feature independently, by transforming it to the certain range
        depending on the used activation function. */
        NO_OUTPUT_SCALE = 4
    };

    CV_WRAP virtual Mat getWeights(int layerIdx) const = 0;

    /** @brief Creates empty model

    Use StatModel::train to train the model, Algorithm::load\<ANN_MLP\>(filename) to load the pre-trained model.
    Note that the train method has optional flags: ANN_MLP::TrainFlags.
     */
    CV_WRAP static Ptr<ANN_MLP> create();

    /** @brief Loads and creates a serialized ANN from a file
     *
     * Use ANN::save to serialize and store an ANN to disk.
     * Load the ANN from this file again, by calling this function with the path to the file.
     *
     * @param filepath path to serialized ANN
     */
    CV_WRAP static Ptr<ANN_MLP> load(const String& filepath);

};

/****************************************************************************************\
*                           Logistic Regression                                          *
\****************************************************************************************/

/** @brief Implements Logistic Regression classifier.

@sa @ref ml_intro_lr
 */
class CV_EXPORTS_W LogisticRegression : public StatModel
{
public:

    /** Learning rate. */
    /** @see setLearningRate */
    CV_WRAP virtual double getLearningRate() const = 0;
    /** @copybrief getLearningRate @see getLearningRate */
    CV_WRAP virtual void setLearningRate(double val) = 0;

    /** Number of iterations. */
    /** @see setIterations */
    CV_WRAP virtual int getIterations() const = 0;
    /** @copybrief getIterations @see getIterations */
    CV_WRAP virtual void setIterations(int val) = 0;

    /** Kind of regularization to be applied. See LogisticRegression::RegKinds. */
    /** @see setRegularization */
    CV_WRAP virtual int getRegularization() const = 0;
    /** @copybrief getRegularization @see getRegularization */
    CV_WRAP virtual void setRegularization(int val) = 0;

    /** Kind of training method used. See LogisticRegression::Methods. */
    /** @see setTrainMethod */
    CV_WRAP virtual int getTrainMethod() const = 0;
    /** @copybrief getTrainMethod @see getTrainMethod */
    CV_WRAP virtual void setTrainMethod(int val) = 0;

    /** Specifies the number of training samples taken in each step of Mini-Batch Gradient
    Descent. Will only be used if using LogisticRegression::MINI_BATCH training algorithm. It
    has to take values less than the total number of training samples. */
    /** @see setMiniBatchSize */
    CV_WRAP virtual int getMiniBatchSize() const = 0;
    /** @copybrief getMiniBatchSize @see getMiniBatchSize */
    CV_WRAP virtual void setMiniBatchSize(int val) = 0;

    /** Termination criteria of the algorithm. */
    /** @see setTermCriteria */
    CV_WRAP virtual TermCriteria getTermCriteria() const = 0;
    /** @copybrief getTermCriteria @see getTermCriteria */
    CV_WRAP virtual void setTermCriteria(TermCriteria val) = 0;

    //! Regularization kinds
    enum RegKinds {
        REG_DISABLE = -1, //!< Regularization disabled
        REG_L1 = 0, //!< %L1 norm
        REG_L2 = 1 //!< %L2 norm
    };

    //! Training methods
    enum Methods {
        BATCH = 0,
        MINI_BATCH = 1 //!< Set MiniBatchSize to a positive integer when using this method.
    };

    /** @brief Predicts responses for input samples and returns a float type.

    @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
        contains variables (features) of one object being classified. Should have data type CV_32F.
    @param results Predicted labels as a column matrix of type CV_32S.
    @param flags Not used.
     */
    CV_WRAP virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;

    /** @brief This function returns the trained paramters arranged across rows.

    For a two class classifcation problem, it returns a row matrix. It returns learnt paramters of
    the Logistic Regression as a matrix of type CV_32F.
     */
    CV_WRAP virtual Mat get_learnt_thetas() const = 0;

    /** @brief Creates empty model.

    Creates Logistic Regression model with parameters given.
     */
    CV_WRAP static Ptr<LogisticRegression> create();

    /** @brief Loads and creates a serialized LogisticRegression from a file
     *
     * Use LogisticRegression::save to serialize and store an LogisticRegression to disk.
     * Load the LogisticRegression from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized LogisticRegression
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<LogisticRegression> load(const String& filepath , const String& nodeName = String());
};


/****************************************************************************************\
*                        Stochastic Gradient Descent SVM Classifier                      *
\****************************************************************************************/

/*!
@brief Stochastic Gradient Descent SVM classifier

SVMSGD provides a fast and easy-to-use implementation of the SVM classifier using the Stochastic Gradient Descent approach,
as presented in @cite bottou2010large.

The classifier has following parameters:
- model type,
- margin type,
- margin regularization (\f$\lambda\f$),
- initial step size (\f$\gamma_0\f$),
- step decreasing power (\f$c\f$),
- and termination criteria.

The model type may have one of the following values: \ref SGD and \ref ASGD.

- \ref SGD is the classic version of SVMSGD classifier: every next step is calculated by the formula
  \f[w_{t+1} = w_t - \gamma(t) \frac{dQ_i}{dw} |_{w = w_t}\f]
  where
  - \f$w_t\f$ is the weights vector for decision function at step \f$t\f$,
  - \f$\gamma(t)\f$ is the step size of model parameters at the iteration \f$t\f$, it is decreased on each step by the formula
    \f$\gamma(t) = \gamma_0  (1 + \lambda  \gamma_0 t) ^ {-c}\f$
  - \f$Q_i\f$ is the target functional from SVM task for sample with number \f$i\f$, this sample is chosen stochastically on each step of the algorithm.

- \ref ASGD is Average Stochastic Gradient Descent SVM Classifier. ASGD classifier averages weights vector on each step of algorithm by the formula
\f$\widehat{w}_{t+1} = \frac{t}{1+t}\widehat{w}_{t} + \frac{1}{1+t}w_{t+1}\f$

The recommended model type is ASGD (following @cite bottou2010large).

The margin type may have one of the following values: \ref SOFT_MARGIN or \ref HARD_MARGIN.

- You should use \ref HARD_MARGIN type, if you have linearly separable sets.
- You should use \ref SOFT_MARGIN type, if you have non-linearly separable sets or sets with outliers.
- In the general case (if you know nothing about linear separability of your sets), use SOFT_MARGIN.

The other parameters may be described as follows:
- Margin regularization parameter is responsible for weights decreasing at each step and for the strength of restrictions on outliers
  (the less the parameter, the less probability that an outlier will be ignored).
  Recommended value for SGD model is 0.0001, for ASGD model is 0.00001.

- Initial step size parameter is the initial value for the step size \f$\gamma(t)\f$.
  You will have to find the best initial step for your problem.

- Step decreasing power is the power parameter for \f$\gamma(t)\f$ decreasing by the formula, mentioned above.
  Recommended value for SGD model is 1, for ASGD model is 0.75.

- Termination criteria can be TermCriteria::COUNT, TermCriteria::EPS or TermCriteria::COUNT + TermCriteria::EPS.
  You will have to find the best termination criteria for your problem.

Note that the parameters margin regularization, initial step size, and step decreasing power should be positive.

To use SVMSGD algorithm do as follows:

- first, create the SVMSGD object. The algoorithm will set optimal parameters by default, but you can set your own parameters via functions setSvmsgdType(),
  setMarginType(), setMarginRegularization(), setInitialStepSize(), and setStepDecreasingPower().

- then the SVM model can be trained using the train features and the correspondent labels by the method train().

- after that, the label of a new feature vector can be predicted using the method predict().

@code
// Create empty object
cv::Ptr<SVMSGD> svmsgd = SVMSGD::create();

// Train the Stochastic Gradient Descent SVM
svmsgd->train(trainData);

// Predict labels for the new samples
svmsgd->predict(samples, responses);
@endcode

*/

class CV_EXPORTS_W SVMSGD : public cv::ml::StatModel
{
public:

    /** SVMSGD type.
    ASGD is often the preferable choice. */
    enum SvmsgdType
    {
        SGD, //!< Stochastic Gradient Descent
        ASGD //!< Average Stochastic Gradient Descent
    };

    /** Margin type.*/
    enum MarginType
    {
        SOFT_MARGIN, //!< General case, suits to the case of non-linearly separable sets, allows outliers.
        HARD_MARGIN  //!< More accurate for the case of linearly separable sets.
    };

    /**
     * @return the weights of the trained model (decision function f(x) = weights * x + shift).
    */
    CV_WRAP virtual Mat getWeights() = 0;

    /**
     * @return the shift of the trained model (decision function f(x) = weights * x + shift).
    */
    CV_WRAP virtual float getShift() = 0;

    /** @brief Creates empty model.
     * Use StatModel::train to train the model. Since %SVMSGD has several parameters, you may want to
     * find the best parameters for your problem or use setOptimalParameters() to set some default parameters.
    */
    CV_WRAP static Ptr<SVMSGD> create();

    /** @brief Loads and creates a serialized SVMSGD from a file
     *
     * Use SVMSGD::save to serialize and store an SVMSGD to disk.
     * Load the SVMSGD from this file again, by calling this function with the path to the file.
     * Optionally specify the node for the file containing the classifier
     *
     * @param filepath path to serialized SVMSGD
     * @param nodeName name of node containing the classifier
     */
    CV_WRAP static Ptr<SVMSGD> load(const String& filepath , const String& nodeName = String());

    /** @brief Function sets optimal parameters values for chosen SVM SGD model.
     * @param svmsgdType is the type of SVMSGD classifier.
     * @param marginType is the type of margin constraint.
    */
    CV_WRAP virtual void setOptimalParameters(int svmsgdType = SVMSGD::ASGD, int marginType = SVMSGD::SOFT_MARGIN) = 0;

    /** @brief %Algorithm type, one of SVMSGD::SvmsgdType. */
    /** @see setSvmsgdType */
    CV_WRAP virtual int getSvmsgdType() const = 0;
    /** @copybrief getSvmsgdType @see getSvmsgdType */
    CV_WRAP virtual void setSvmsgdType(int svmsgdType) = 0;

    /** @brief %Margin type, one of SVMSGD::MarginType. */
    /** @see setMarginType */
    CV_WRAP virtual int getMarginType() const = 0;
    /** @copybrief getMarginType @see getMarginType */
    CV_WRAP virtual void setMarginType(int marginType) = 0;

    /** @brief Parameter marginRegularization of a %SVMSGD optimization problem. */
    /** @see setMarginRegularization */
    CV_WRAP virtual float getMarginRegularization() const = 0;
    /** @copybrief getMarginRegularization @see getMarginRegularization */
    CV_WRAP virtual void setMarginRegularization(float marginRegularization) = 0;

    /** @brief Parameter initialStepSize of a %SVMSGD optimization problem. */
    /** @see setInitialStepSize */
    CV_WRAP virtual float getInitialStepSize() const = 0;
    /** @copybrief getInitialStepSize @see getInitialStepSize */
    CV_WRAP virtual void setInitialStepSize(float InitialStepSize) = 0;

    /** @brief Parameter stepDecreasingPower of a %SVMSGD optimization problem. */
    /** @see setStepDecreasingPower */
    CV_WRAP virtual float getStepDecreasingPower() const = 0;
    /** @copybrief getStepDecreasingPower @see getStepDecreasingPower */
    CV_WRAP virtual void setStepDecreasingPower(float stepDecreasingPower) = 0;

    /** @brief Termination criteria of the training algorithm.
    You can specify the maximum number of iterations (maxCount) and/or how much the error could
    change between the iterations to make the algorithm continue (epsilon).*/
    /** @see setTermCriteria */
    CV_WRAP virtual TermCriteria getTermCriteria() const = 0;
    /** @copybrief getTermCriteria @see getTermCriteria */
    CV_WRAP virtual void setTermCriteria(const cv::TermCriteria &val) = 0;
};


/****************************************************************************************\
*                           Auxilary functions declarations                              *
\****************************************************************************************/

/** @brief Generates _sample_ from multivariate normal distribution

@param mean an average row vector
@param cov symmetric covariation matrix
@param nsamples returned samples count
@param samples returned samples array
*/
CV_EXPORTS void randMVNormal( InputArray mean, InputArray cov, int nsamples, OutputArray samples);

/** @brief Creates test set */
CV_EXPORTS void createConcentricSpheresTestSet( int nsamples, int nfeatures, int nclasses,
                                                OutputArray samples, OutputArray responses);

//! @} ml

}
}

#endif // __cplusplus
#endif // OPENCV_ML_HPP

/* End of file. */
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

#ifndef OPENCV_OBJDETECT_HPP
#define OPENCV_OBJDETECT_HPP

#include "opencv2/core.hpp"

/**
@defgroup objdetect Object Detection

Haar Feature-based Cascade Classifier for Object Detection
----------------------------------------------------------

The object detector described below has been initially proposed by Paul Viola @cite Viola01 and
improved by Rainer Lienhart @cite Lienhart02 .

First, a classifier (namely a *cascade of boosted classifiers working with haar-like features*) is
trained with a few hundred sample views of a particular object (i.e., a face or a car), called
positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary
images of the same size.

After a classifier is trained, it can be applied to a region of interest (of the same size as used
during the training) in an input image. The classifier outputs a "1" if the region is likely to show
the object (i.e., face/car), and "0" otherwise. To search for the object in the whole image one can
move the search window across the image and check every location using the classifier. The
classifier is designed so that it can be easily "resized" in order to be able to find the objects of
interest at different sizes, which is more efficient than resizing the image itself. So, to find an
object of an unknown size in the image the scan procedure should be done several times at different
scales.

The word "cascade" in the classifier name means that the resultant classifier consists of several
simpler classifiers (*stages*) that are applied subsequently to a region of interest until at some
stage the candidate is rejected or all the stages are passed. The word "boosted" means that the
classifiers at every stage of the cascade are complex themselves and they are built out of basic
classifiers using one of four different boosting techniques (weighted voting). Currently Discrete
Adaboost, Real Adaboost, Gentle Adaboost and Logitboost are supported. The basic classifiers are
decision-tree classifiers with at least 2 leaves. Haar-like features are the input to the basic
classifiers, and are calculated as described below. The current algorithm uses the following
Haar-like features:

![image](pics/haarfeatures.png)

The feature used in a particular classifier is specified by its shape (1a, 2b etc.), position within
the region of interest and the scale (this scale is not the same as the scale used at the detection
stage, though these two scales are multiplied). For example, in the case of the third line feature
(2c) the response is calculated as the difference between the sum of image pixels under the
rectangle covering the whole feature (including the two white stripes and the black stripe in the
middle) and the sum of the image pixels under the black stripe multiplied by 3 in order to
compensate for the differences in the size of areas. The sums of pixel values over a rectangular
regions are calculated rapidly using integral images (see below and the integral description).

To see the object detector at work, have a look at the facedetect demo:
<https://github.com/opencv/opencv/tree/master/samples/cpp/dbt_face_detection.cpp>

The following reference is for the detection part only. There is a separate application called
opencv_traincascade that can train a cascade of boosted classifiers from a set of samples.

@note In the new C++ interface it is also possible to use LBP (local binary pattern) features in
addition to Haar-like features. .. [Viola01] Paul Viola and Michael J. Jones. Rapid Object Detection
using a Boosted Cascade of Simple Features. IEEE CVPR, 2001. The paper is available online at
<http://research.microsoft.com/en-us/um/people/viola/Pubs/Detect/violaJones_CVPR2001.pdf>

@{
    @defgroup objdetect_c C API
@}
 */

typedef struct CvHaarClassifierCascade CvHaarClassifierCascade;

namespace cv
{

//! @addtogroup objdetect
//! @{

///////////////////////////// Object Detection ////////////////////////////

//! class for grouping object candidates, detected by Cascade Classifier, HOG etc.
//! instance of the class is to be passed to cv::partition (see cxoperations.hpp)
class CV_EXPORTS SimilarRects
{
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator()(const Rect& r1, const Rect& r2) const
    {
        double delta = eps * ((std::min)(r1.width, r2.width) + (std::min)(r1.height, r2.height)) * 0.5;
        return std::abs(r1.x - r2.x) <= delta &&
            std::abs(r1.y - r2.y) <= delta &&
            std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
            std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
    }
    double eps;
};

/** @brief Groups the object candidate rectangles.

@param rectList Input/output vector of rectangles. Output vector includes retained and grouped
rectangles. (The Python list is not modified in place.)
@param groupThreshold Minimum possible number of rectangles minus 1. The threshold is used in a
group of rectangles to retain it.
@param eps Relative difference between sides of the rectangles to merge them into a group.

The function is a wrapper for the generic function partition . It clusters all the input rectangles
using the rectangle equivalence criteria that combines rectangles with similar sizes and similar
locations. The similarity is defined by eps. When eps=0 , no clustering is done at all. If
\f$\texttt{eps}\rightarrow +\inf\f$ , all the rectangles are put in one cluster. Then, the small
clusters containing less than or equal to groupThreshold rectangles are rejected. In each other
cluster, the average rectangle is computed and put into the output rectangle list.
 */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS_W void groupRectangles(CV_IN_OUT std::vector<Rect>& rectList, CV_OUT std::vector<int>& weights,
                                  int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, int groupThreshold,
                                  double eps, std::vector<int>* weights, std::vector<double>* levelWeights );
/** @overload */
CV_EXPORTS   void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                                  std::vector<double>& levelWeights, int groupThreshold, double eps = 0.2);
/** @overload */
CV_EXPORTS   void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                                            std::vector<double>& foundScales,
                                            double detectThreshold = 0.0, Size winDetSize = Size(64, 128));

template<> CV_EXPORTS void DefaultDeleter<CvHaarClassifierCascade>::operator ()(CvHaarClassifierCascade* obj) const;

enum { CASCADE_DO_CANNY_PRUNING    = 1,
       CASCADE_SCALE_IMAGE         = 2,
       CASCADE_FIND_BIGGEST_OBJECT = 4,
       CASCADE_DO_ROUGH_SEARCH     = 8
     };

class CV_EXPORTS_W BaseCascadeClassifier : public Algorithm
{
public:
    virtual ~BaseCascadeClassifier();
    virtual bool empty() const = 0;
    virtual bool load( const String& filename ) = 0;
    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                           CV_OUT std::vector<Rect>& objects,
                           CV_OUT std::vector<int>& numDetections,
                           double scaleFactor,
                           int minNeighbors, int flags,
                           Size minSize, Size maxSize ) = 0;

    virtual void detectMultiScale( InputArray image,
                                   CV_OUT std::vector<Rect>& objects,
                                   CV_OUT std::vector<int>& rejectLevels,
                                   CV_OUT std::vector<double>& levelWeights,
                                   double scaleFactor,
                                   int minNeighbors, int flags,
                                   Size minSize, Size maxSize,
                                   bool outputRejectLevels ) = 0;

    virtual bool isOldFormatCascade() const = 0;
    virtual Size getOriginalWindowSize() const = 0;
    virtual int getFeatureType() const = 0;
    virtual void* getOldCascade() = 0;

    class CV_EXPORTS MaskGenerator
    {
    public:
        virtual ~MaskGenerator() {}
        virtual Mat generateMask(const Mat& src)=0;
        virtual void initializeMask(const Mat& /*src*/) { }
    };
    virtual void setMaskGenerator(const Ptr<MaskGenerator>& maskGenerator) = 0;
    virtual Ptr<MaskGenerator> getMaskGenerator() = 0;
};

/** @example facedetect.cpp
*/
/** @brief Cascade classifier class for object detection.
 */
class CV_EXPORTS_W CascadeClassifier
{
public:
    CV_WRAP CascadeClassifier();
    /** @brief Loads a classifier from a file.

    @param filename Name of the file from which the classifier is loaded.
     */
    CV_WRAP CascadeClassifier(const String& filename);
    ~CascadeClassifier();
    /** @brief Checks whether the classifier has been loaded.
    */
    CV_WRAP bool empty() const;
    /** @brief Loads a classifier from a file.

    @param filename Name of the file from which the classifier is loaded. The file may contain an old
    HAAR classifier trained by the haartraining application or a new cascade classifier trained by the
    traincascade application.
     */
    CV_WRAP bool load( const String& filename );
    /** @brief Reads a classifier from a FileStorage node.

    @note The file may contain a new cascade classifier (trained traincascade application) only.
     */
    CV_WRAP bool read( const FileNode& node );

    /** @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
    of rectangles.

    @param image Matrix of the type CV_8U containing an image where objects are detected.
    @param objects Vector of rectangles where each rectangle contains the detected object, the
    rectangles may be partially outside the original image.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param flags Parameter with the same meaning for an old cascade as in the function
    cvHaarDetectObjects. It is not used for a new cascade.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.
    @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.

    The function is parallelized with the TBB library.

    @note
       -   (Python) A face detection example using cascade classifiers can be found at
            opencv_source_code/samples/python/facedetect.py
    */
    CV_WRAP void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size() );

    /** @overload
    @param image Matrix of the type CV_8U containing an image where objects are detected.
    @param objects Vector of rectangles where each rectangle contains the detected object, the
    rectangles may be partially outside the original image.
    @param numDetections Vector of detection numbers for the corresponding objects. An object's number
    of detections is the number of neighboring positively classified rectangles that were joined
    together to form the object.
    @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    to retain it.
    @param flags Parameter with the same meaning for an old cascade as in the function
    cvHaarDetectObjects. It is not used for a new cascade.
    @param minSize Minimum possible object size. Objects smaller than that are ignored.
    @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
    */
    CV_WRAP_AS(detectMultiScale2) void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& numDetections,
                          double scaleFactor=1.1,
                          int minNeighbors=3, int flags=0,
                          Size minSize=Size(),
                          Size maxSize=Size() );

    /** @overload
    This function allows you to retrieve the final stage decision certainty of classification.
    For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
    For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
    This value can then be used to separate strong from weaker classifications.

    A code sample on how to use it efficiently can be found below:
    @code
    Mat img;
    vector<double> weights;
    vector<int> levels;
    vector<Rect> detections;
    CascadeClassifier model("/path/to/your/model.xml");
    model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
    cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
    @endcode
    */
    CV_WRAP_AS(detectMultiScale3) void detectMultiScale( InputArray image,
                                  CV_OUT std::vector<Rect>& objects,
                                  CV_OUT std::vector<int>& rejectLevels,
                                  CV_OUT std::vector<double>& levelWeights,
                                  double scaleFactor = 1.1,
                                  int minNeighbors = 3, int flags = 0,
                                  Size minSize = Size(),
                                  Size maxSize = Size(),
                                  bool outputRejectLevels = false );

    CV_WRAP bool isOldFormatCascade() const;
    CV_WRAP Size getOriginalWindowSize() const;
    CV_WRAP int getFeatureType() const;
    void* getOldCascade();

    CV_WRAP static bool convert(const String& oldcascade, const String& newcascade);

    void setMaskGenerator(const Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator);
    Ptr<BaseCascadeClassifier::MaskGenerator> getMaskGenerator();

    Ptr<BaseCascadeClassifier> cc;
};

CV_EXPORTS Ptr<BaseCascadeClassifier::MaskGenerator> createFaceDetectionMaskGenerator();

//////////////// HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector //////////////

//! struct for detection region of interest (ROI)
struct DetectionROI
{
   //! scale(size) of the bounding box
   double scale;
   //! set of requrested locations to be evaluated
   std::vector<cv::Point> locations;
   //! vector that will contain confidence values for each location
   std::vector<double> confidences;
};

/**@example peopledetect.cpp
 */
struct CV_EXPORTS_W HOGDescriptor
{
public:
    enum { L2Hys = 0
         };
    enum { DEFAULT_NLEVELS = 64
         };

    CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
        cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
        histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
        free_coef(-1.f), nlevels(HOGDescriptor::DEFAULT_NLEVELS), signedGradient(false)
    {}

    CV_WRAP HOGDescriptor(Size _winSize, Size _blockSize, Size _blockStride,
                  Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
                  int _histogramNormType=HOGDescriptor::L2Hys,
                  double _L2HysThreshold=0.2, bool _gammaCorrection=false,
                  int _nlevels=HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient=false)
    : winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
    nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
    histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
    gammaCorrection(_gammaCorrection), free_coef(-1.f), nlevels(_nlevels), signedGradient(_signedGradient)
    {}

    CV_WRAP HOGDescriptor(const String& filename)
    {
        load(filename);
    }

    HOGDescriptor(const HOGDescriptor& d)
    {
        d.copyTo(*this);
    }

    virtual ~HOGDescriptor() {}

    CV_WRAP size_t getDescriptorSize() const;
    CV_WRAP bool checkDetectorSize() const;
    CV_WRAP double getWinSigma() const;

    CV_WRAP virtual void setSVMDetector(InputArray _svmdetector);

    virtual bool read(FileNode& fn);
    virtual void write(FileStorage& fs, const String& objname) const;

    CV_WRAP virtual bool load(const String& filename, const String& objname = String());
    CV_WRAP virtual void save(const String& filename, const String& objname = String()) const;
    virtual void copyTo(HOGDescriptor& c) const;

    CV_WRAP virtual void compute(InputArray img,
                         CV_OUT std::vector<float>& descriptors,
                         Size winStride = Size(), Size padding = Size(),
                         const std::vector<Point>& locations = std::vector<Point>()) const;

    //! with found weights output
    CV_WRAP virtual void detect(const Mat& img, CV_OUT std::vector<Point>& foundLocations,
                        CV_OUT std::vector<double>& weights,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations = std::vector<Point>()) const;
    //! without found weights output
    virtual void detect(const Mat& img, CV_OUT std::vector<Point>& foundLocations,
                        double hitThreshold = 0, Size winStride = Size(),
                        Size padding = Size(),
                        const std::vector<Point>& searchLocations=std::vector<Point>()) const;

    //! with result weights output
    CV_WRAP virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  CV_OUT std::vector<double>& foundWeights, double hitThreshold = 0,
                                  Size winStride = Size(), Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0,bool useMeanshiftGrouping = false) const;
    //! without found weights output
    virtual void detectMultiScale(InputArray img, CV_OUT std::vector<Rect>& foundLocations,
                                  double hitThreshold = 0, Size winStride = Size(),
                                  Size padding = Size(), double scale = 1.05,
                                  double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

    CV_WRAP virtual void computeGradient(const Mat& img, CV_OUT Mat& grad, CV_OUT Mat& angleOfs,
                                 Size paddingTL = Size(), Size paddingBR = Size()) const;

    CV_WRAP static std::vector<float> getDefaultPeopleDetector();
    CV_WRAP static std::vector<float> getDaimlerPeopleDetector();

    CV_PROP Size winSize;
    CV_PROP Size blockSize;
    CV_PROP Size blockStride;
    CV_PROP Size cellSize;
    CV_PROP int nbins;
    CV_PROP int derivAperture;
    CV_PROP double winSigma;
    CV_PROP int histogramNormType;
    CV_PROP double L2HysThreshold;
    CV_PROP bool gammaCorrection;
    CV_PROP std::vector<float> svmDetector;
    UMat oclSvmDetector;
    float free_coef;
    CV_PROP int nlevels;
    CV_PROP bool signedGradient;


    //! evaluate specified ROI and return confidence value for each location
    virtual void detectROI(const cv::Mat& img, const std::vector<cv::Point> &locations,
                                   CV_OUT std::vector<cv::Point>& foundLocations, CV_OUT std::vector<double>& confidences,
                                   double hitThreshold = 0, cv::Size winStride = Size(),
                                   cv::Size padding = Size()) const;

    //! evaluate specified ROI and return confidence value for each location in multiple scales
    virtual void detectMultiScaleROI(const cv::Mat& img,
                                                       CV_OUT std::vector<cv::Rect>& foundLocations,
                                                       std::vector<DetectionROI>& locations,
                                                       double hitThreshold = 0,
                                                       int groupThreshold = 0) const;

    //! read/parse Dalal's alt model file
    void readALTModel(String modelfile);
    void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<double>& weights, int groupThreshold, double eps) const;
};

//! @} objdetect

}

#include "opencv2/objdetect/detection_based_tracker.hpp"

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/objdetect/objdetect_c.h"
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
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_ALL_HPP
#define OPENCV_ALL_HPP

// File that defines what modules where included during the build of OpenCV
// These are purely the defines of the correct HAVE_OPENCV_modulename values
#include "opencv2/opencv_modules.hpp"

// Then the list of defines is checked to include the correct headers
// Core library is always included --> without no OpenCV functionality available
#include "opencv2/core.hpp"

// Then the optional modules are checked
#ifdef HAVE_OPENCV_CALIB3D
#include "opencv2/calib3d.hpp"
#endif
#ifdef HAVE_OPENCV_FEATURES2D
#include "opencv2/features2d.hpp"
#endif
#ifdef HAVE_OPENCV_FLANN
#include "opencv2/flann.hpp"
#endif
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#ifdef HAVE_OPENCV_IMGCODECS
#include "opencv2/imgcodecs.hpp"
#endif
#ifdef HAVE_OPENCV_IMGPROC
#include "opencv2/imgproc.hpp"
#endif
#ifdef HAVE_OPENCV_ML
#include "opencv2/ml.hpp"
#endif
#ifdef HAVE_OPENCV_OBJDETECT
#include "opencv2/objdetect.hpp"
#endif
#ifdef HAVE_OPENCV_PHOTO
#include "opencv2/photo.hpp"
#endif
#ifdef HAVE_OPENCV_SHAPE
#include "opencv2/shape.hpp"
#endif
#ifdef HAVE_OPENCV_STITCHING
#include "opencv2/stitching.hpp"
#endif
#ifdef HAVE_OPENCV_SUPERRES
#include "opencv2/superres.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEO
#include "opencv2/video.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOIO
#include "opencv2/videoio.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOSTAB
#include "opencv2/videostab.hpp"
#endif
#ifdef HAVE_OPENCV_VIZ
#include "opencv2/viz.hpp"
#endif

// Finally CUDA specific entries are checked and added
#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/cudaarithm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDABGSEGM
#include "opencv2/cudabgsegm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDACODEC
#include "opencv2/cudacodec.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFEATURES2D
#include "opencv2/cudafeatures2d.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFILTERS
#include "opencv2/cudafilters.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAIMGPROC
#include "opencv2/cudaimgproc.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAOBJDETECT
#include "opencv2/cudaobjdetect.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAOPTFLOW
#include "opencv2/cudaoptflow.hpp"
#endif
#ifdef HAVE_OPENCV_CUDASTEREO
#include "opencv2/cudastereo.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif

#endif
/*
 *      ** File generated automatically, do not modify **
 *
 * This file defines the list of modules available in current build configuration
 *
 *
*/

// This definition means that OpenCV is built with enabled non-free code.
// For example, patented algorithms for non-profit/non-commercial use only.
/* #undef OPENCV_ENABLE_NONFREE */

#define HAVE_OPENCV_CALIB3D
#define HAVE_OPENCV_CORE
#define HAVE_OPENCV_DNN
#define HAVE_OPENCV_FEATURES2D
#define HAVE_OPENCV_FLANN
#define HAVE_OPENCV_HIGHGUI
#define HAVE_OPENCV_IMGCODECS
#define HAVE_OPENCV_IMGPROC
#define HAVE_OPENCV_ML
#define HAVE_OPENCV_OBJDETECT
#define HAVE_OPENCV_PHOTO
#define HAVE_OPENCV_SHAPE
#define HAVE_OPENCV_STITCHING
#define HAVE_OPENCV_SUPERRES
#define HAVE_OPENCV_VIDEO
#define HAVE_OPENCV_VIDEOIO
#define HAVE_OPENCV_VIDEOSTAB
#define HAVE_OPENCV_WORLD


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
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_PHOTO_HPP
#define OPENCV_PHOTO_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/**
@defgroup photo Computational Photography
@{
    @defgroup photo_denoise Denoising
    @defgroup photo_hdr HDR imaging

This section describes high dynamic range imaging algorithms namely tonemapping, exposure alignment,
camera calibration with multiple exposures and exposure fusion.

    @defgroup photo_clone Seamless Cloning
    @defgroup photo_render Non-Photorealistic Rendering
    @defgroup photo_c C API
@}
  */

namespace cv
{

//! @addtogroup photo
//! @{

//! the inpainting algorithm
enum
{
    INPAINT_NS    = 0, // Navier-Stokes algorithm
    INPAINT_TELEA = 1 // A. Telea algorithm
};

enum
{
    NORMAL_CLONE = 1,
    MIXED_CLONE  = 2,
    MONOCHROME_TRANSFER = 3
};

enum
{
    RECURS_FILTER = 1,
    NORMCONV_FILTER = 2
};

/** @brief Restores the selected region in an image using the region neighborhood.

@param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.
@param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that
needs to be inpainted.
@param dst Output image with the same size and type as src .
@param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered
by the algorithm.
@param flags Inpainting method that could be one of the following:
-   **INPAINT_NS** Navier-Stokes based method [Navier01]
-   **INPAINT_TELEA** Method by Alexandru Telea @cite Telea04 .

The function reconstructs the selected image area from the pixel near the area boundary. The
function may be used to remove dust and scratches from a scanned photo, or to remove undesirable
objects from still images or video. See <http://en.wikipedia.org/wiki/Inpainting> for more details.

@note
   -   An example using the inpainting technique can be found at
        opencv_source_code/samples/cpp/inpaint.cpp
    -   (Python) An example using the inpainting technique can be found at
        opencv_source_code/samples/python/inpaint.py
 */
CV_EXPORTS_W void inpaint( InputArray src, InputArray inpaintMask,
        OutputArray dst, double inpaintRadius, int flags );

//! @addtogroup photo_denoise
//! @{

/** @brief Perform image denoising using Non-local Means Denoising algorithm
<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational
optimizations. Noise expected to be a gaussian white noise

@param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
@param dst Output image with the same size and type as src .
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength. Big h value perfectly removes noise but also
removes image details, smaller h value preserves details but also preserves some noise

This function expected to be applied to grayscale images. For colored images look at
fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
image to CIELAB colorspace and then separately denoise L and AB components with different h
parameter.
 */
CV_EXPORTS_W void fastNlMeansDenoising( InputArray src, OutputArray dst, float h = 3,
        int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Perform image denoising using Non-local Means Denoising algorithm
<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational
optimizations. Noise expected to be a gaussian white noise

@param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
2-channel, 3-channel or 4-channel image.
@param dst Output image with the same size and type as src .
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Array of parameters regulating filter strength, either one
parameter applied to all channels or one per channel in dst. Big h value
perfectly removes noise but also removes image details, smaller h
value preserves details but also preserves some noise
@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1

This function expected to be applied to grayscale images. For colored images look at
fastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored
image in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting
image to CIELAB colorspace and then separately denoise L and AB components with different h
parameter.
 */
CV_EXPORTS_W void fastNlMeansDenoising( InputArray src, OutputArray dst,
                                        const std::vector<float>& h,
                                        int templateWindowSize = 7, int searchWindowSize = 21,
                                        int normType = NORM_L2);

/** @brief Modification of fastNlMeansDenoising function for colored images

@param src Input 8-bit 3-channel image.
@param dst Output image with the same size and type as src .
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
removes noise but also removes image details, smaller h value preserves details but also preserves
some noise
@param hColor The same as h but for color components. For most images value equals 10
will be enough to remove colored noise and do not distort colors

The function converts image to CIELAB colorspace and then separately denoise L and AB components
with given h parameters using fastNlMeansDenoising function.
 */
CV_EXPORTS_W void fastNlMeansDenoisingColored( InputArray src, OutputArray dst,
        float h = 3, float hColor = 3,
        int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Modification of fastNlMeansDenoising function for images sequence where consequtive images have been
captured in small period of time. For example video. This version of the function is for grayscale
images or for manual manipulation with colorspaces. For more details see
<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394>

@param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or
4-channel images sequence. All images should have the same type and
size.
@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
@param temporalWindowSize Number of surrounding images to use for target image denoising. Should
be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
srcImgs[imgToDenoiseIndex] image.
@param dst Output image with the same size and type as srcImgs images.
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength. Bigger h value
perfectly removes noise but also removes image details, smaller h
value preserves details but also preserves some noise
 */
CV_EXPORTS_W void fastNlMeansDenoisingMulti( InputArrayOfArrays srcImgs, OutputArray dst,
        int imgToDenoiseIndex, int temporalWindowSize,
        float h = 3, int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Modification of fastNlMeansDenoising function for images sequence where consequtive images have been
captured in small period of time. For example video. This version of the function is for grayscale
images or for manual manipulation with colorspaces. For more details see
<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394>

@param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,
2-channel, 3-channel or 4-channel images sequence. All images should
have the same type and size.
@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
@param temporalWindowSize Number of surrounding images to use for target image denoising. Should
be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
srcImgs[imgToDenoiseIndex] image.
@param dst Output image with the same size and type as srcImgs images.
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Array of parameters regulating filter strength, either one
parameter applied to all channels or one per channel in dst. Big h value
perfectly removes noise but also removes image details, smaller h
value preserves details but also preserves some noise
@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1
 */
CV_EXPORTS_W void fastNlMeansDenoisingMulti( InputArrayOfArrays srcImgs, OutputArray dst,
                                             int imgToDenoiseIndex, int temporalWindowSize,
                                             const std::vector<float>& h,
                                             int templateWindowSize = 7, int searchWindowSize = 21,
                                             int normType = NORM_L2);

/** @brief Modification of fastNlMeansDenoisingMulti function for colored images sequences

@param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and
size.
@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence
@param temporalWindowSize Number of surrounding images to use for target image denoising. Should
be odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to
imgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise
srcImgs[imgToDenoiseIndex] image.
@param dst Output image with the same size and type as srcImgs images.
@param templateWindowSize Size in pixels of the template patch that is used to compute weights.
Should be odd. Recommended value 7 pixels
@param searchWindowSize Size in pixels of the window that is used to compute weighted average for
given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater
denoising time. Recommended value 21 pixels
@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly
removes noise but also removes image details, smaller h value preserves details but also preserves
some noise.
@param hColor The same as h but for color components.

The function converts images to CIELAB colorspace and then separately denoise L and AB components
with given h parameters using fastNlMeansDenoisingMulti function.
 */
CV_EXPORTS_W void fastNlMeansDenoisingColoredMulti( InputArrayOfArrays srcImgs, OutputArray dst,
        int imgToDenoiseIndex, int temporalWindowSize,
        float h = 3, float hColor = 3,
        int templateWindowSize = 7, int searchWindowSize = 21);

/** @brief Primal-dual algorithm is an algorithm for solving special types of variational problems (that is,
finding a function to minimize some functional). As the image denoising, in particular, may be seen
as the variational problem, primal-dual algorithm then can be used to perform denoising and this is
exactly what is implemented.

It should be noted, that this implementation was taken from the July 2013 blog entry
@cite MA13 , which also contained (slightly more general) ready-to-use source code on Python.
Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky at the end
of July 2013 and finally it was slightly adapted by later authors.

Although the thorough discussion and justification of the algorithm involved may be found in
@cite ChambolleEtAl, it might make sense to skim over it here, following @cite MA13 . To begin
with, we consider the 1-byte gray-level images as the functions from the rectangular domain of
pixels (it may be seen as set
\f$\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}\f$ for some
\f$m,\;n\in\mathbb{N}\f$) into \f$\{0,1,\dots,255\}\f$. We shall denote the noised images as \f$f_i\f$ and with
this view, given some image \f$x\f$ of the same size, we may measure how bad it is by the formula

\f[\left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|\f]

\f$\|\|\cdot\|\|\f$ here denotes \f$L_2\f$-norm and as you see, the first addend states that we want our
image to be smooth (ideally, having zero gradient, thus being constant) and the second states that
we want our result to be close to the observations we've got. If we treat \f$x\f$ as a function, this is
exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes into play.

@param observations This array should contain one or more noised versions of the image that is to
be restored.
@param result Here the denoised image will be stored. There is no need to do pre-allocation of
storage space, as it will be automatically allocated, if necessary.
@param lambda Corresponds to \f$\lambda\f$ in the formulas above. As it is enlarged, the smooth
(blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly
speaking, as it becomes smaller, the result will be more blur but more sever outliers will be
removed.
@param niters Number of iterations that the algorithm will run. Of course, as more iterations as
better, but it is hard to quantitatively refine this statement, so just use the default and
increase it if the results are poor.
 */
CV_EXPORTS_W void denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda=1.0, int niters=30);

//! @} photo_denoise

//! @addtogroup photo_hdr
//! @{

enum { LDR_SIZE = 256 };

/** @brief Base class for tonemapping algorithms - tools that are used to map HDR image to 8-bit range.
 */
class CV_EXPORTS_W Tonemap : public Algorithm
{
public:
    /** @brief Tonemaps image

    @param src source image - 32-bit 3-channel Mat
    @param dst destination image - 32-bit 3-channel Mat with values in [0, 1] range
     */
    CV_WRAP virtual void process(InputArray src, OutputArray dst) = 0;

    CV_WRAP virtual float getGamma() const = 0;
    CV_WRAP virtual void setGamma(float gamma) = 0;
};

/** @brief Creates simple linear mapper with gamma correction

@param gamma positive value for gamma correction. Gamma value of 1.0 implies no correction, gamma
equal to 2.2f is suitable for most displays.
Generally gamma \> 1 brightens the image and gamma \< 1 darkens it.
 */
CV_EXPORTS_W Ptr<Tonemap> createTonemap(float gamma = 1.0f);

/** @brief Adaptive logarithmic mapping is a fast global tonemapping algorithm that scales the image in
logarithmic domain.

Since it's a global operator the same function is applied to all the pixels, it is controlled by the
bias parameter.

Optional saturation enhancement is possible as described in @cite FL02 .

For more information see @cite DM03 .
 */
class CV_EXPORTS_W TonemapDrago : public Tonemap
{
public:

    CV_WRAP virtual float getSaturation() const = 0;
    CV_WRAP virtual void setSaturation(float saturation) = 0;

    CV_WRAP virtual float getBias() const = 0;
    CV_WRAP virtual void setBias(float bias) = 0;
};

/** @brief Creates TonemapDrago object

@param gamma gamma value for gamma correction. See createTonemap
@param saturation positive saturation enhancement value. 1.0 preserves saturation, values greater
than 1 increase saturation and values less than 1 decrease it.
@param bias value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give best
results, default value is 0.85.
 */
CV_EXPORTS_W Ptr<TonemapDrago> createTonemapDrago(float gamma = 1.0f, float saturation = 1.0f, float bias = 0.85f);

/** @brief This algorithm decomposes image into two layers: base layer and detail layer using bilateral filter
and compresses contrast of the base layer thus preserving all the details.

This implementation uses regular bilateral filter from opencv.

Saturation enhancement is possible as in ocvTonemapDrago.

For more information see @cite DD02 .
 */
class CV_EXPORTS_W TonemapDurand : public Tonemap
{
public:

    CV_WRAP virtual float getSaturation() const = 0;
    CV_WRAP virtual void setSaturation(float saturation) = 0;

    CV_WRAP virtual float getContrast() const = 0;
    CV_WRAP virtual void setContrast(float contrast) = 0;

    CV_WRAP virtual float getSigmaSpace() const = 0;
    CV_WRAP virtual void setSigmaSpace(float sigma_space) = 0;

    CV_WRAP virtual float getSigmaColor() const = 0;
    CV_WRAP virtual void setSigmaColor(float sigma_color) = 0;
};

/** @brief Creates TonemapDurand object

@param gamma gamma value for gamma correction. See createTonemap
@param contrast resulting contrast on logarithmic scale, i. e. log(max / min), where max and min
are maximum and minimum luminance values of the resulting image.
@param saturation saturation enhancement value. See createTonemapDrago
@param sigma_space bilateral filter sigma in color space
@param sigma_color bilateral filter sigma in coordinate space
 */
CV_EXPORTS_W Ptr<TonemapDurand>
createTonemapDurand(float gamma = 1.0f, float contrast = 4.0f, float saturation = 1.0f, float sigma_space = 2.0f, float sigma_color = 2.0f);

/** @brief This is a global tonemapping operator that models human visual system.

Mapping function is controlled by adaptation parameter, that is computed using light adaptation and
color adaptation.

For more information see @cite RD05 .
 */
class CV_EXPORTS_W TonemapReinhard : public Tonemap
{
public:
    CV_WRAP virtual float getIntensity() const = 0;
    CV_WRAP virtual void setIntensity(float intensity) = 0;

    CV_WRAP virtual float getLightAdaptation() const = 0;
    CV_WRAP virtual void setLightAdaptation(float light_adapt) = 0;

    CV_WRAP virtual float getColorAdaptation() const = 0;
    CV_WRAP virtual void setColorAdaptation(float color_adapt) = 0;
};

/** @brief Creates TonemapReinhard object

@param gamma gamma value for gamma correction. See createTonemap
@param intensity result intensity in [-8, 8] range. Greater intensity produces brighter results.
@param light_adapt light adaptation in [0, 1] range. If 1 adaptation is based only on pixel
value, if 0 it's global, otherwise it's a weighted mean of this two cases.
@param color_adapt chromatic adaptation in [0, 1] range. If 1 channels are treated independently,
if 0 adaptation level is the same for each channel.
 */
CV_EXPORTS_W Ptr<TonemapReinhard>
createTonemapReinhard(float gamma = 1.0f, float intensity = 0.0f, float light_adapt = 1.0f, float color_adapt = 0.0f);

/** @brief This algorithm transforms image to contrast using gradients on all levels of gaussian pyramid,
transforms contrast values to HVS response and scales the response. After this the image is
reconstructed from new contrast values.

For more information see @cite MM06 .
 */
class CV_EXPORTS_W TonemapMantiuk : public Tonemap
{
public:
    CV_WRAP virtual float getScale() const = 0;
    CV_WRAP virtual void setScale(float scale) = 0;

    CV_WRAP virtual float getSaturation() const = 0;
    CV_WRAP virtual void setSaturation(float saturation) = 0;
};

/** @brief Creates TonemapMantiuk object

@param gamma gamma value for gamma correction. See createTonemap
@param scale contrast scale factor. HVS response is multiplied by this parameter, thus compressing
dynamic range. Values from 0.6 to 0.9 produce best results.
@param saturation saturation enhancement value. See createTonemapDrago
 */
CV_EXPORTS_W Ptr<TonemapMantiuk>
createTonemapMantiuk(float gamma = 1.0f, float scale = 0.7f, float saturation = 1.0f);

/** @brief The base class for algorithms that align images of the same scene with different exposures
 */
class CV_EXPORTS_W AlignExposures : public Algorithm
{
public:
    /** @brief Aligns images

    @param src vector of input images
    @param dst vector of aligned images
    @param times vector of exposure time values for each image
    @param response 256x1 matrix with inverse camera response function for each pixel value, it should
    have the same number of channels as images.
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, std::vector<Mat>& dst,
                                 InputArray times, InputArray response) = 0;
};

/** @brief This algorithm converts images to median threshold bitmaps (1 for pixels brighter than median
luminance and 0 otherwise) and than aligns the resulting bitmaps using bit operations.

It is invariant to exposure, so exposure values and camera response are not necessary.

In this implementation new image regions are filled with zeros.

For more information see @cite GW03 .
 */
class CV_EXPORTS_W AlignMTB : public AlignExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, std::vector<Mat>& dst,
                                 InputArray times, InputArray response) = 0;

    /** @brief Short version of process, that doesn't take extra arguments.

    @param src vector of input images
    @param dst vector of aligned images
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, std::vector<Mat>& dst) = 0;

    /** @brief Calculates shift between two images, i. e. how to shift the second image to correspond it with the
    first.

    @param img0 first image
    @param img1 second image
     */
    CV_WRAP virtual Point calculateShift(InputArray img0, InputArray img1) = 0;
    /** @brief Helper function, that shift Mat filling new regions with zeros.

    @param src input image
    @param dst result image
    @param shift shift value
     */
    CV_WRAP virtual void shiftMat(InputArray src, OutputArray dst, const Point shift) = 0;
    /** @brief Computes median threshold and exclude bitmaps of given image.

    @param img input image
    @param tb median threshold bitmap
    @param eb exclude bitmap
     */
    CV_WRAP virtual void computeBitmaps(InputArray img, OutputArray tb, OutputArray eb) = 0;

    CV_WRAP virtual int getMaxBits() const = 0;
    CV_WRAP virtual void setMaxBits(int max_bits) = 0;

    CV_WRAP virtual int getExcludeRange() const = 0;
    CV_WRAP virtual void setExcludeRange(int exclude_range) = 0;

    CV_WRAP virtual bool getCut() const = 0;
    CV_WRAP virtual void setCut(bool value) = 0;
};

/** @brief Creates AlignMTB object

@param max_bits logarithm to the base 2 of maximal shift in each dimension. Values of 5 and 6 are
usually good enough (31 and 63 pixels shift respectively).
@param exclude_range range for exclusion bitmap that is constructed to suppress noise around the
median value.
@param cut if true cuts images, otherwise fills the new regions with zeros.
 */
CV_EXPORTS_W Ptr<AlignMTB> createAlignMTB(int max_bits = 6, int exclude_range = 4, bool cut = true);

/** @brief The base class for camera response calibration algorithms.
 */
class CV_EXPORTS_W CalibrateCRF : public Algorithm
{
public:
    /** @brief Recovers inverse camera response.

    @param src vector of input images
    @param dst 256x1 matrix with inverse camera response function
    @param times vector of exposure time values for each image
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst, InputArray times) = 0;
};

/** @brief Inverse camera response function is extracted for each brightness value by minimizing an objective
function as linear system. Objective function is constructed using pixel values on the same position
in all images, extra term is added to make the result smoother.

For more information see @cite DM97 .
 */
class CV_EXPORTS_W CalibrateDebevec : public CalibrateCRF
{
public:
    CV_WRAP virtual float getLambda() const = 0;
    CV_WRAP virtual void setLambda(float lambda) = 0;

    CV_WRAP virtual int getSamples() const = 0;
    CV_WRAP virtual void setSamples(int samples) = 0;

    CV_WRAP virtual bool getRandom() const = 0;
    CV_WRAP virtual void setRandom(bool random) = 0;
};

/** @brief Creates CalibrateDebevec object

@param samples number of pixel locations to use
@param lambda smoothness term weight. Greater values produce smoother results, but can alter the
response.
@param random if true sample pixel locations are chosen at random, otherwise they form a
rectangular grid.
 */
CV_EXPORTS_W Ptr<CalibrateDebevec> createCalibrateDebevec(int samples = 70, float lambda = 10.0f, bool random = false);

/** @brief Inverse camera response function is extracted for each brightness value by minimizing an objective
function as linear system. This algorithm uses all image pixels.

For more information see @cite RB99 .
 */
class CV_EXPORTS_W CalibrateRobertson : public CalibrateCRF
{
public:
    CV_WRAP virtual int getMaxIter() const = 0;
    CV_WRAP virtual void setMaxIter(int max_iter) = 0;

    CV_WRAP virtual float getThreshold() const = 0;
    CV_WRAP virtual void setThreshold(float threshold) = 0;

    CV_WRAP virtual Mat getRadiance() const = 0;
};

/** @brief Creates CalibrateRobertson object

@param max_iter maximal number of Gauss-Seidel solver iterations.
@param threshold target difference between results of two successive steps of the minimization.
 */
CV_EXPORTS_W Ptr<CalibrateRobertson> createCalibrateRobertson(int max_iter = 30, float threshold = 0.01f);

/** @brief The base class algorithms that can merge exposure sequence to a single image.
 */
class CV_EXPORTS_W MergeExposures : public Algorithm
{
public:
    /** @brief Merges images.

    @param src vector of input images
    @param dst result image
    @param times vector of exposure time values for each image
    @param response 256x1 matrix with inverse camera response function for each pixel value, it should
    have the same number of channels as images.
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) = 0;
};

/** @brief The resulting HDR image is calculated as weighted average of the exposures considering exposure
values and camera response.

For more information see @cite DM97 .
 */
class CV_EXPORTS_W MergeDebevec : public MergeExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) = 0;
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst, InputArray times) = 0;
};

/** @brief Creates MergeDebevec object
 */
CV_EXPORTS_W Ptr<MergeDebevec> createMergeDebevec();

/** @brief Pixels are weighted using contrast, saturation and well-exposedness measures, than images are
combined using laplacian pyramids.

The resulting image weight is constructed as weighted average of contrast, saturation and
well-exposedness measures.

The resulting image doesn't require tonemapping and can be converted to 8-bit image by multiplying
by 255, but it's recommended to apply gamma correction and/or linear tonemapping.

For more information see @cite MK07 .
 */
class CV_EXPORTS_W MergeMertens : public MergeExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) = 0;
    /** @brief Short version of process, that doesn't take extra arguments.

    @param src vector of input images
    @param dst result image
     */
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst) = 0;

    CV_WRAP virtual float getContrastWeight() const = 0;
    CV_WRAP virtual void setContrastWeight(float contrast_weiht) = 0;

    CV_WRAP virtual float getSaturationWeight() const = 0;
    CV_WRAP virtual void setSaturationWeight(float saturation_weight) = 0;

    CV_WRAP virtual float getExposureWeight() const = 0;
    CV_WRAP virtual void setExposureWeight(float exposure_weight) = 0;
};

/** @brief Creates MergeMertens object

@param contrast_weight contrast measure weight. See MergeMertens.
@param saturation_weight saturation measure weight
@param exposure_weight well-exposedness measure weight
 */
CV_EXPORTS_W Ptr<MergeMertens>
createMergeMertens(float contrast_weight = 1.0f, float saturation_weight = 1.0f, float exposure_weight = 0.0f);

/** @brief The resulting HDR image is calculated as weighted average of the exposures considering exposure
values and camera response.

For more information see @cite RB99 .
 */
class CV_EXPORTS_W MergeRobertson : public MergeExposures
{
public:
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst,
                                 InputArray times, InputArray response) = 0;
    CV_WRAP virtual void process(InputArrayOfArrays src, OutputArray dst, InputArray times) = 0;
};

/** @brief Creates MergeRobertson object
 */
CV_EXPORTS_W Ptr<MergeRobertson> createMergeRobertson();

//! @} photo_hdr

/** @brief Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylized
black-and-white photograph rendering, and in many single channel image processing applications
@cite CL12 .

@param src Input 8-bit 3-channel image.
@param grayscale Output 8-bit 1-channel image.
@param color_boost Output 8-bit 3-channel image.

This function is to be applied on color images.
 */
CV_EXPORTS_W void decolor( InputArray src, OutputArray grayscale, OutputArray color_boost);

//! @addtogroup photo_clone
//! @{

/** @example cloning_demo.cpp
An example using seamlessClone function
*/
/** @brief Image editing tasks concern either global changes (color/intensity corrections, filters,
deformations) or local changes concerned to a selection. Here we are interested in achieving local
changes, ones that are restricted to a region manually selected (ROI), in a seamless and effortless
manner. The extent of the changes ranges from slight distortions to complete replacement by novel
content @cite PM03 .

@param src Input 8-bit 3-channel image.
@param dst Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param p Point in dst image where object is placed.
@param blend Output image with the same size and type as dst.
@param flags Cloning method that could be one of the following:
-   **NORMAL_CLONE** The power of the method is fully expressed when inserting objects with
complex outlines into a new background
-   **MIXED_CLONE** The classic method, color-based selection and alpha masking might be time
consuming and often leaves an undesirable halo. Seamless cloning, even averaged with the
original image, is not effective. Mixed seamless cloning based on a loose selection proves
effective.
-   **MONOCHROME_TRANSFER** Monochrome transfer allows the user to easily replace certain features of
one object by alternative features.
 */
CV_EXPORTS_W void seamlessClone( InputArray src, InputArray dst, InputArray mask, Point p,
        OutputArray blend, int flags);

/** @brief Given an original color image, two differently colored versions of this image can be mixed
seamlessly.

@param src Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param dst Output image with the same size and type as src .
@param red_mul R-channel multiply factor.
@param green_mul G-channel multiply factor.
@param blue_mul B-channel multiply factor.

Multiplication factor is between .5 to 2.5.
 */
CV_EXPORTS_W void colorChange(InputArray src, InputArray mask, OutputArray dst, float red_mul = 1.0f,
        float green_mul = 1.0f, float blue_mul = 1.0f);

/** @brief Applying an appropriate non-linear transformation to the gradient field inside the selection and
then integrating back with a Poisson solver, modifies locally the apparent illumination of an image.

@param src Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param dst Output image with the same size and type as src.
@param alpha Value ranges between 0-2.
@param beta Value ranges between 0-2.

This is useful to highlight under-exposed foreground objects or to reduce specular reflections.
 */
CV_EXPORTS_W void illuminationChange(InputArray src, InputArray mask, OutputArray dst,
        float alpha = 0.2f, float beta = 0.4f);

/** @brief By retaining only the gradients at edge locations, before integrating with the Poisson solver, one
washes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge
Detector is used.

@param src Input 8-bit 3-channel image.
@param mask Input 8-bit 1 or 3-channel image.
@param dst Output image with the same size and type as src.
@param low_threshold Range from 0 to 100.
@param high_threshold Value \> 100.
@param kernel_size The size of the Sobel kernel to be used.

**NOTE:**

The algorithm assumes that the color of the source image is close to that of the destination. This
assumption means that when the colors don't match, the source image color gets tinted toward the
color of the destination image.
 */
CV_EXPORTS_W void textureFlattening(InputArray src, InputArray mask, OutputArray dst,
        float low_threshold = 30, float high_threshold = 45,
        int kernel_size = 3);

//! @} photo_clone

//! @addtogroup photo_render
//! @{

/** @brief Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing
filters are used in many different applications @cite EM11 .

@param src Input 8-bit 3-channel image.
@param dst Output 8-bit 3-channel image.
@param flags Edge preserving filters:
-   **RECURS_FILTER** = 1
-   **NORMCONV_FILTER** = 2
@param sigma_s Range between 0 to 200.
@param sigma_r Range between 0 to 1.
 */
CV_EXPORTS_W void edgePreservingFilter(InputArray src, OutputArray dst, int flags = 1,
        float sigma_s = 60, float sigma_r = 0.4f);

/** @brief This filter enhances the details of a particular image.

@param src Input 8-bit 3-channel image.
@param dst Output image with the same size and type as src.
@param sigma_s Range between 0 to 200.
@param sigma_r Range between 0 to 1.
 */
CV_EXPORTS_W void detailEnhance(InputArray src, OutputArray dst, float sigma_s = 10,
        float sigma_r = 0.15f);

/** @example npr_demo.cpp
An example using non-photorealistic line drawing functions
*/
/** @brief Pencil-like non-photorealistic line drawing

@param src Input 8-bit 3-channel image.
@param dst1 Output 8-bit 1-channel image.
@param dst2 Output image with the same size and type as src.
@param sigma_s Range between 0 to 200.
@param sigma_r Range between 0 to 1.
@param shade_factor Range between 0 to 0.1.
 */
CV_EXPORTS_W void pencilSketch(InputArray src, OutputArray dst1, OutputArray dst2,
        float sigma_s = 60, float sigma_r = 0.07f, float shade_factor = 0.02f);

/** @brief Stylization aims to produce digital imagery with a wide variety of effects not focused on
photorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low
contrast while preserving, or enhancing, high-contrast features.

@param src Input 8-bit 3-channel image.
@param dst Output image with the same size and type as src.
@param sigma_s Range between 0 to 200.
@param sigma_r Range between 0 to 1.
 */
CV_EXPORTS_W void stylization(InputArray src, OutputArray dst, float sigma_s = 60,
        float sigma_r = 0.45f);

//! @} photo_render

//! @} photo

} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/photo/photo_c.h"
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
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_SHAPE_HPP
#define OPENCV_SHAPE_HPP

#include "opencv2/shape/emdL1.hpp"
#include "opencv2/shape/shape_transformer.hpp"
#include "opencv2/shape/hist_cost.hpp"
#include "opencv2/shape/shape_distance.hpp"

/**
  @defgroup shape Shape Distance and Matching
 */

#endif

/* End of file. */
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

#ifndef OPENCV_STITCHING_STITCHER_HPP
#define OPENCV_STITCHING_STITCHER_HPP

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"


#if defined(Status)
#  warning Detected X11 'Status' macro definition, it can cause build conflicts. Please, include this header before any X11 headers.
#endif


/**
@defgroup stitching Images stitching

This figure illustrates the stitching module pipeline implemented in the Stitcher class. Using that
class it's possible to configure/remove some steps, i.e. adjust the stitching pipeline according to
the particular needs. All building blocks from the pipeline are available in the detail namespace,
one can combine and use them separately.

The implemented stitching pipeline is very similar to the one proposed in @cite BL07 .

![stitching pipeline](StitchingPipeline.jpg)

Camera models
-------------

There are currently 2 camera models implemented in stitching pipeline.

- _Homography model_ expecting perspective transformations between images
  implemented in @ref cv::detail::BestOf2NearestMatcher cv::detail::HomographyBasedEstimator
  cv::detail::BundleAdjusterReproj cv::detail::BundleAdjusterRay
- _Affine model_ expecting affine transformation with 6 DOF or 4 DOF implemented in
  @ref cv::detail::AffineBestOf2NearestMatcher cv::detail::AffineBasedEstimator
  cv::detail::BundleAdjusterAffine cv::detail::BundleAdjusterAffinePartial cv::AffineWarper

Homography model is useful for creating photo panoramas captured by camera,
while affine-based model can be used to stitch scans and object captured by
specialized devices. Use @ref cv::Stitcher::create to get preconfigured pipeline for one
of those models.

@note
Certain detailed settings of @ref cv::Stitcher might not make sense. Especially
you should not mix classes implementing affine model and classes implementing
Homography model, as they work with different transformations.

@{
    @defgroup stitching_match Features Finding and Images Matching
    @defgroup stitching_rotation Rotation Estimation
    @defgroup stitching_autocalib Autocalibration
    @defgroup stitching_warp Images Warping
    @defgroup stitching_seam Seam Estimation
    @defgroup stitching_exposure Exposure Compensation
    @defgroup stitching_blend Image Blenders
@}
  */

namespace cv {

//! @addtogroup stitching
//! @{

/** @brief High level image stitcher.

It's possible to use this class without being aware of the entire stitching pipeline. However, to
be able to achieve higher stitching stability and quality of the final images at least being
familiar with the theory is recommended.

@note
   -   A basic example on image stitching can be found at
        opencv_source_code/samples/cpp/stitching.cpp
    -   A detailed example on image stitching can be found at
        opencv_source_code/samples/cpp/stitching_detailed.cpp
 */
class CV_EXPORTS_W Stitcher
{
public:
    enum { ORIG_RESOL = -1 };
    enum Status
    {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };
    enum Mode
    {
        /** Mode for creating photo panoramas. Expects images under perspective
        transformation and projects resulting pano to sphere.

        @sa detail::BestOf2NearestMatcher SphericalWarper
        */
        PANORAMA = 0,
        /** Mode for composing scans. Expects images under affine transformation does
        not compensate exposure by default.

        @sa detail::AffineBestOf2NearestMatcher AffineWarper
        */
        SCANS = 1,

    };

   // Stitcher() {}
    /** @brief Creates a stitcher with the default parameters.

    @param try_use_gpu Flag indicating whether GPU should be used whenever it's possible.
    @return Stitcher class instance.
     */
    static Stitcher createDefault(bool try_use_gpu = false);
    /** @brief Creates a Stitcher configured in one of the stitching modes.

    @param mode Scenario for stitcher operation. This is usually determined by source of images
    to stitch and their transformation. Default parameters will be chosen for operation in given
    scenario.
    @param try_use_gpu Flag indicating whether GPU should be used whenever it's possible.
    @return Stitcher class instance.
     */
    static Ptr<Stitcher> create(Mode mode = PANORAMA, bool try_use_gpu = false);

    CV_WRAP double registrationResol() const { return registr_resol_; }
    CV_WRAP void setRegistrationResol(double resol_mpx) { registr_resol_ = resol_mpx; }

    CV_WRAP double seamEstimationResol() const { return seam_est_resol_; }
    CV_WRAP void setSeamEstimationResol(double resol_mpx) { seam_est_resol_ = resol_mpx; }

    CV_WRAP double compositingResol() const { return compose_resol_; }
    CV_WRAP void setCompositingResol(double resol_mpx) { compose_resol_ = resol_mpx; }

    CV_WRAP double panoConfidenceThresh() const { return conf_thresh_; }
    CV_WRAP void setPanoConfidenceThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    CV_WRAP bool waveCorrection() const { return do_wave_correct_; }
    CV_WRAP void setWaveCorrection(bool flag) { do_wave_correct_ = flag; }

    detail::WaveCorrectKind waveCorrectKind() const { return wave_correct_kind_; }
    void setWaveCorrectKind(detail::WaveCorrectKind kind) { wave_correct_kind_ = kind; }

    Ptr<detail::FeaturesFinder> featuresFinder() { return features_finder_; }
    const Ptr<detail::FeaturesFinder> featuresFinder() const { return features_finder_; }
    void setFeaturesFinder(Ptr<detail::FeaturesFinder> features_finder)
        { features_finder_ = features_finder; }

    Ptr<detail::FeaturesMatcher> featuresMatcher() { return features_matcher_; }
    const Ptr<detail::FeaturesMatcher> featuresMatcher() const { return features_matcher_; }
    void setFeaturesMatcher(Ptr<detail::FeaturesMatcher> features_matcher)
        { features_matcher_ = features_matcher; }

    const cv::UMat& matchingMask() const { return matching_mask_; }
    void setMatchingMask(const cv::UMat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
        matching_mask_ = mask.clone();
    }

    Ptr<detail::BundleAdjusterBase> bundleAdjuster() { return bundle_adjuster_; }
    const Ptr<detail::BundleAdjusterBase> bundleAdjuster() const { return bundle_adjuster_; }
    void setBundleAdjuster(Ptr<detail::BundleAdjusterBase> bundle_adjuster)
        { bundle_adjuster_ = bundle_adjuster; }

    /* TODO OpenCV ABI 4.x
    Ptr<detail::Estimator> estimator() { return estimator_; }
    const Ptr<detail::Estimator> estimator() const { return estimator_; }
    void setEstimator(Ptr<detail::Estimator> estimator)
        { estimator_ = estimator; }
    */

    Ptr<WarperCreator> warper() { return warper_; }
    const Ptr<WarperCreator> warper() const { return warper_; }
    void setWarper(Ptr<WarperCreator> creator) { warper_ = creator; }

    Ptr<detail::ExposureCompensator> exposureCompensator() { return exposure_comp_; }
    const Ptr<detail::ExposureCompensator> exposureCompensator() const { return exposure_comp_; }
    void setExposureCompensator(Ptr<detail::ExposureCompensator> exposure_comp)
        { exposure_comp_ = exposure_comp; }

    Ptr<detail::SeamFinder> seamFinder() { return seam_finder_; }
    const Ptr<detail::SeamFinder> seamFinder() const { return seam_finder_; }
    void setSeamFinder(Ptr<detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }

    Ptr<detail::Blender> blender() { return blender_; }
    const Ptr<detail::Blender> blender() const { return blender_; }
    void setBlender(Ptr<detail::Blender> b) { blender_ = b; }

    /** @overload */
    CV_WRAP Status estimateTransform(InputArrayOfArrays images);
    /** @brief These functions try to match the given images and to estimate rotations of each camera.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param rois Region of interest rectangles.
    @return Status code.
     */
    Status estimateTransform(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois);

    /** @overload */
    CV_WRAP Status composePanorama(OutputArray pano);
    /** @brief These functions try to compose the given images (or images stored internally from the other function
    calls) into the final pano under the assumption that the image transformations were estimated
    before.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param pano Final pano.
    @return Status code.
     */
    Status composePanorama(InputArrayOfArrays images, OutputArray pano);

    /** @overload */
    CV_WRAP Status stitch(InputArrayOfArrays images, OutputArray pano);
    /** @brief These functions try to stitch the given images.

    @param images Input images.
    @param rois Region of interest rectangles.
    @param pano Final pano.
    @return Status code.
     */
    Status stitch(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois, OutputArray pano);

    std::vector<int> component() const { return indices_; }
    std::vector<detail::CameraParams> cameras() const { return cameras_; }
    CV_WRAP double workScale() const { return work_scale_; }

private:
    //Stitcher() {}

    Status matchImages();
    Status estimateCameraParams();

    double registr_resol_;
    double seam_est_resol_;
    double compose_resol_;
    double conf_thresh_;
    Ptr<detail::FeaturesFinder> features_finder_;
    Ptr<detail::FeaturesMatcher> features_matcher_;
    cv::UMat matching_mask_;
    Ptr<detail::BundleAdjusterBase> bundle_adjuster_;
    /* TODO OpenCV ABI 4.x
    Ptr<detail::Estimator> estimator_;
    */
    bool do_wave_correct_;
    detail::WaveCorrectKind wave_correct_kind_;
    Ptr<WarperCreator> warper_;
    Ptr<detail::ExposureCompensator> exposure_comp_;
    Ptr<detail::SeamFinder> seam_finder_;
    Ptr<detail::Blender> blender_;

    std::vector<cv::UMat> imgs_;
    std::vector<std::vector<cv::Rect> > rois_;
    std::vector<cv::Size> full_img_sizes_;
    std::vector<detail::ImageFeatures> features_;
    std::vector<detail::MatchesInfo> pairwise_matches_;
    std::vector<cv::UMat> seam_est_imgs_;
    std::vector<int> indices_;
    std::vector<detail::CameraParams> cameras_;
    double work_scale_;
    double seam_scale_;
    double seam_work_aspect_;
    double warped_image_scale_;
};

CV_EXPORTS_W Ptr<Stitcher> createStitcher(bool try_use_gpu = false);

//! @} stitching

} // namespace cv

#endif // OPENCV_STITCHING_STITCHER_HPP
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

#ifndef OPENCV_SUPERRES_HPP
#define OPENCV_SUPERRES_HPP

#include "opencv2/core.hpp"
#include "opencv2/superres/optical_flow.hpp"

/**
  @defgroup superres Super Resolution

The Super Resolution module contains a set of functions and classes that can be used to solve the
problem of resolution enhancement. There are a few methods implemented, most of them are descibed in
the papers @cite Farsiu03 and @cite Mitzel09 .

 */

namespace cv
{
    namespace superres
    {

//! @addtogroup superres
//! @{

        class CV_EXPORTS FrameSource
        {
        public:
            virtual ~FrameSource();

            virtual void nextFrame(OutputArray frame) = 0;
            virtual void reset() = 0;
        };

        CV_EXPORTS Ptr<FrameSource> createFrameSource_Empty();

        CV_EXPORTS Ptr<FrameSource> createFrameSource_Video(const String& fileName);
        CV_EXPORTS Ptr<FrameSource> createFrameSource_Video_CUDA(const String& fileName);

        CV_EXPORTS Ptr<FrameSource> createFrameSource_Camera(int deviceId = 0);

        /** @brief Base class for Super Resolution algorithms.

        The class is only used to define the common interface for the whole family of Super Resolution
        algorithms.
         */
        class CV_EXPORTS SuperResolution : public cv::Algorithm, public FrameSource
        {
        public:
            /** @brief Set input frame source for Super Resolution algorithm.

            @param frameSource Input frame source
             */
            void setInput(const Ptr<FrameSource>& frameSource);

            /** @brief Process next frame from input and return output result.

            @param frame Output result
             */
            void nextFrame(OutputArray frame);
            void reset();

            /** @brief Clear all inner buffers.
            */
            virtual void collectGarbage();

            //! @brief Scale factor
            /** @see setScale */
            virtual int getScale() const = 0;
            /** @copybrief getScale @see getScale */
            virtual void setScale(int val) = 0;

            //! @brief Iterations count
            /** @see setIterations */
            virtual int getIterations() const = 0;
            /** @copybrief getIterations @see getIterations */
            virtual void setIterations(int val) = 0;

            //! @brief Asymptotic value of steepest descent method
            /** @see setTau */
            virtual double getTau() const = 0;
            /** @copybrief getTau @see getTau */
            virtual void setTau(double val) = 0;

            //! @brief Weight parameter to balance data term and smoothness term
            /** @see setLabmda */
            virtual double getLabmda() const = 0;
            /** @copybrief getLabmda @see getLabmda */
            virtual void setLabmda(double val) = 0;

            //! @brief Parameter of spacial distribution in Bilateral-TV
            /** @see setAlpha */
            virtual double getAlpha() const = 0;
            /** @copybrief getAlpha @see getAlpha */
            virtual void setAlpha(double val) = 0;

            //! @brief Kernel size of Bilateral-TV filter
            /** @see setKernelSize */
            virtual int getKernelSize() const = 0;
            /** @copybrief getKernelSize @see getKernelSize */
            virtual void setKernelSize(int val) = 0;

            //! @brief Gaussian blur kernel size
            /** @see setBlurKernelSize */
            virtual int getBlurKernelSize() const = 0;
            /** @copybrief getBlurKernelSize @see getBlurKernelSize */
            virtual void setBlurKernelSize(int val) = 0;

            //! @brief Gaussian blur sigma
            /** @see setBlurSigma */
            virtual double getBlurSigma() const = 0;
            /** @copybrief getBlurSigma @see getBlurSigma */
            virtual void setBlurSigma(double val) = 0;

            //! @brief Radius of the temporal search area
            /** @see setTemporalAreaRadius */
            virtual int getTemporalAreaRadius() const = 0;
            /** @copybrief getTemporalAreaRadius @see getTemporalAreaRadius */
            virtual void setTemporalAreaRadius(int val) = 0;

            //! @brief Dense optical flow algorithm
            /** @see setOpticalFlow */
            virtual Ptr<cv::superres::DenseOpticalFlowExt> getOpticalFlow() const = 0;
            /** @copybrief getOpticalFlow @see getOpticalFlow */
            virtual void setOpticalFlow(const Ptr<cv::superres::DenseOpticalFlowExt> &val) = 0;

        protected:
            SuperResolution();

            virtual void initImpl(Ptr<FrameSource>& frameSource) = 0;
            virtual void processImpl(Ptr<FrameSource>& frameSource, OutputArray output) = 0;

            bool isUmat_;

        private:
            Ptr<FrameSource> frameSource_;
            bool firstCall_;
        };

        /** @brief Create Bilateral TV-L1 Super Resolution.

        This class implements Super Resolution algorithm described in the papers @cite Farsiu03 and
        @cite Mitzel09 .

        Here are important members of the class that control the algorithm, which you can set after
        constructing the class instance:

        -   **int scale** Scale factor.
        -   **int iterations** Iteration count.
        -   **double tau** Asymptotic value of steepest descent method.
        -   **double lambda** Weight parameter to balance data term and smoothness term.
        -   **double alpha** Parameter of spacial distribution in Bilateral-TV.
        -   **int btvKernelSize** Kernel size of Bilateral-TV filter.
        -   **int blurKernelSize** Gaussian blur kernel size.
        -   **double blurSigma** Gaussian blur sigma.
        -   **int temporalAreaRadius** Radius of the temporal search area.
        -   **Ptr\<DenseOpticalFlowExt\> opticalFlow** Dense optical flow algorithm.
         */
        CV_EXPORTS Ptr<SuperResolution> createSuperResolution_BTVL1();
        CV_EXPORTS Ptr<SuperResolution> createSuperResolution_BTVL1_CUDA();

//! @} superres

    }
}

#endif // OPENCV_SUPERRES_HPP
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

#ifndef OPENCV_VIDEO_HPP
#define OPENCV_VIDEO_HPP

/**
  @defgroup video Video Analysis
  @{
    @defgroup video_motion Motion Analysis
    @defgroup video_track Object Tracking
    @defgroup video_c C API
  @}
*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/video/tracking_c.h"
#endif

#endif //OPENCV_VIDEO_HPP
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

#ifndef OPENCV_VIDEOIO_HPP
#define OPENCV_VIDEOIO_HPP

#include "opencv2/core.hpp"

/**
  @defgroup videoio Video I/O

  @brief Read and write video or images sequence with OpenCV

  ### See also:
  - @ref videoio_overview
  - Tutorials: @ref tutorial_table_of_content_videoio
  @{
    @defgroup videoio_flags_base Flags for video I/O
    @defgroup videoio_flags_others Additional flags for video I/O API backends
    @defgroup videoio_c C API for video I/O
    @defgroup videoio_ios iOS glue for video I/O
    @defgroup videoio_winrt WinRT glue for video I/O
  @}
*/

////////////////////////////////// video io /////////////////////////////////

typedef struct CvCapture CvCapture;
typedef struct CvVideoWriter CvVideoWriter;

namespace cv
{

//! @addtogroup videoio
//! @{

//! @addtogroup videoio_flags_base
//! @{


/** @brief %VideoCapture API backends identifier.

Select preferred API for a capture object.
To be used in the VideoCapture::VideoCapture() constructor or VideoCapture::open()

@note Backends are available only if they have been built with your OpenCV binaries.
See @ref videoio_overview for more information.
*/
enum VideoCaptureAPIs {
       CAP_ANY          = 0,            //!< Auto detect == 0
       CAP_VFW          = 200,          //!< Video For Windows (platform native)
       CAP_V4L          = 200,          //!< V4L/V4L2 capturing support via libv4l
       CAP_V4L2         = CAP_V4L,      //!< Same as CAP_V4L
       CAP_FIREWIRE     = 300,          //!< IEEE 1394 drivers
       CAP_FIREWARE     = CAP_FIREWIRE, //!< Same as CAP_FIREWIRE
       CAP_IEEE1394     = CAP_FIREWIRE, //!< Same as CAP_FIREWIRE
       CAP_DC1394       = CAP_FIREWIRE, //!< Same as CAP_FIREWIRE
       CAP_CMU1394      = CAP_FIREWIRE, //!< Same as CAP_FIREWIRE
       CAP_QT           = 500,          //!< QuickTime
       CAP_UNICAP       = 600,          //!< Unicap drivers
       CAP_DSHOW        = 700,          //!< DirectShow (via videoInput)
       CAP_PVAPI        = 800,          //!< PvAPI, Prosilica GigE SDK
       CAP_OPENNI       = 900,          //!< OpenNI (for Kinect)
       CAP_OPENNI_ASUS  = 910,          //!< OpenNI (for Asus Xtion)
       CAP_ANDROID      = 1000,         //!< Android - not used
       CAP_XIAPI        = 1100,         //!< XIMEA Camera API
       CAP_AVFOUNDATION = 1200,         //!< AVFoundation framework for iOS (OS X Lion will have the same API)
       CAP_GIGANETIX    = 1300,         //!< Smartek Giganetix GigEVisionSDK
       CAP_MSMF         = 1400,         //!< Microsoft Media Foundation (via videoInput)
       CAP_WINRT        = 1410,         //!< Microsoft Windows Runtime using Media Foundation
       CAP_INTELPERC    = 1500,         //!< Intel Perceptual Computing SDK
       CAP_OPENNI2      = 1600,         //!< OpenNI2 (for Kinect)
       CAP_OPENNI2_ASUS = 1610,         //!< OpenNI2 (for Asus Xtion and Occipital Structure sensors)
       CAP_GPHOTO2      = 1700,         //!< gPhoto2 connection
       CAP_GSTREAMER    = 1800,         //!< GStreamer
       CAP_FFMPEG       = 1900,         //!< Open and record video file or stream using the FFMPEG library
       CAP_IMAGES       = 2000,         //!< OpenCV Image Sequence (e.g. img_%02d.jpg)
       CAP_ARAVIS       = 2100,         //!< Aravis SDK
       CAP_OPENCV_MJPEG = 2200,         //!< Built-in OpenCV MotionJPEG codec
       CAP_INTEL_MFX    = 2300          //!< Intel MediaSDK
     };

/** @brief %VideoCapture generic properties identifier.

 Reading / writing properties involves many layers. Some unexpected result might happens along this chain.
 Effective behaviour depends from device hardware, driver and API Backend.
 @sa videoio_flags_others, VideoCapture::get(), VideoCapture::set()
*/
enum VideoCaptureProperties {
       CAP_PROP_POS_MSEC       =0, //!< Current position of the video file in milliseconds.
       CAP_PROP_POS_FRAMES     =1, //!< 0-based index of the frame to be decoded/captured next.
       CAP_PROP_POS_AVI_RATIO  =2, //!< Relative position of the video file: 0=start of the film, 1=end of the film.
       CAP_PROP_FRAME_WIDTH    =3, //!< Width of the frames in the video stream.
       CAP_PROP_FRAME_HEIGHT   =4, //!< Height of the frames in the video stream.
       CAP_PROP_FPS            =5, //!< Frame rate.
       CAP_PROP_FOURCC         =6, //!< 4-character code of codec. see VideoWriter::fourcc .
       CAP_PROP_FRAME_COUNT    =7, //!< Number of frames in the video file.
       CAP_PROP_FORMAT         =8, //!< Format of the %Mat objects returned by VideoCapture::retrieve().
       CAP_PROP_MODE           =9, //!< Backend-specific value indicating the current capture mode.
       CAP_PROP_BRIGHTNESS    =10, //!< Brightness of the image (only for those cameras that support).
       CAP_PROP_CONTRAST      =11, //!< Contrast of the image (only for cameras).
       CAP_PROP_SATURATION    =12, //!< Saturation of the image (only for cameras).
       CAP_PROP_HUE           =13, //!< Hue of the image (only for cameras).
       CAP_PROP_GAIN          =14, //!< Gain of the image (only for those cameras that support).
       CAP_PROP_EXPOSURE      =15, //!< Exposure (only for those cameras that support).
       CAP_PROP_CONVERT_RGB   =16, //!< Boolean flags indicating whether images should be converted to RGB.
       CAP_PROP_WHITE_BALANCE_BLUE_U =17, //!< Currently unsupported.
       CAP_PROP_RECTIFICATION =18, //!< Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently).
       CAP_PROP_MONOCHROME    =19,
       CAP_PROP_SHARPNESS     =20,
       CAP_PROP_AUTO_EXPOSURE =21, //!< DC1394: exposure control done by camera, user can adjust reference level using this feature.
       CAP_PROP_GAMMA         =22,
       CAP_PROP_TEMPERATURE   =23,
       CAP_PROP_TRIGGER       =24,
       CAP_PROP_TRIGGER_DELAY =25,
       CAP_PROP_WHITE_BALANCE_RED_V =26,
       CAP_PROP_ZOOM          =27,
       CAP_PROP_FOCUS         =28,
       CAP_PROP_GUID          =29,
       CAP_PROP_ISO_SPEED     =30,
       CAP_PROP_BACKLIGHT     =32,
       CAP_PROP_PAN           =33,
       CAP_PROP_TILT          =34,
       CAP_PROP_ROLL          =35,
       CAP_PROP_IRIS          =36,
       CAP_PROP_SETTINGS      =37, //!< Pop up video/camera filter dialog (note: only supported by DSHOW backend currently. The property value is ignored)
       CAP_PROP_BUFFERSIZE    =38,
       CAP_PROP_AUTOFOCUS     =39
     };


/** @brief Generic camera output modes identifier.
@note Currently, these are supported through the libv4l backend only.
*/
enum VideoCaptureModes {
       CAP_MODE_BGR  = 0, //!< BGR24 (default)
       CAP_MODE_RGB  = 1, //!< RGB24
       CAP_MODE_GRAY = 2, //!< Y8
       CAP_MODE_YUYV = 3  //!< YUYV
     };

/** @brief %VideoWriter generic properties identifier.
 @sa VideoWriter::get(), VideoWriter::set()
*/
enum VideoWriterProperties {
  VIDEOWRITER_PROP_QUALITY = 1,    //!< Current quality (0..100%) of the encoded videostream. Can be adjusted dynamically in some codecs.
  VIDEOWRITER_PROP_FRAMEBYTES = 2, //!< (Read-only): Size of just encoded video frame. Note that the encoding order may be different from representation order.
  VIDEOWRITER_PROP_NSTRIPES = 3    //!< Number of stripes for parallel encoding. -1 for auto detection.
};

//! @} videoio_flags_base

//! @addtogroup videoio_flags_others
//! @{

/** @name IEEE 1394 drivers
    @{
*/

/** @brief Modes of the IEEE 1394 controlling registers
(can be: auto, manual, auto single push, absolute Latter allowed with any other mode)
every feature can have only one mode turned on at a time
*/
enum { CAP_PROP_DC1394_OFF                = -4, //!< turn the feature off (not controlled manually nor automatically).
       CAP_PROP_DC1394_MODE_MANUAL        = -3, //!< set automatically when a value of the feature is set by the user.
       CAP_PROP_DC1394_MODE_AUTO          = -2,
       CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO = -1,
       CAP_PROP_DC1394_MAX                = 31
     };

//! @} IEEE 1394 drivers

/** @name OpenNI (for Kinect)
    @{
*/

//! OpenNI map generators
enum { CAP_OPENNI_DEPTH_GENERATOR = 1 << 31,
       CAP_OPENNI_IMAGE_GENERATOR = 1 << 30,
       CAP_OPENNI_IR_GENERATOR    = 1 << 29,
       CAP_OPENNI_GENERATORS_MASK = CAP_OPENNI_DEPTH_GENERATOR + CAP_OPENNI_IMAGE_GENERATOR + CAP_OPENNI_IR_GENERATOR
     };

//! Properties of cameras available through OpenNI backend
enum { CAP_PROP_OPENNI_OUTPUT_MODE       = 100,
       CAP_PROP_OPENNI_FRAME_MAX_DEPTH   = 101, //!< In mm
       CAP_PROP_OPENNI_BASELINE          = 102, //!< In mm
       CAP_PROP_OPENNI_FOCAL_LENGTH      = 103, //!< In pixels
       CAP_PROP_OPENNI_REGISTRATION      = 104, //!< Flag that synchronizes the remapping depth map to image map
                                                //!< by changing depth generator's view point (if the flag is "on") or
                                                //!< sets this view point to its normal one (if the flag is "off").
       CAP_PROP_OPENNI_REGISTRATION_ON   = CAP_PROP_OPENNI_REGISTRATION,
       CAP_PROP_OPENNI_APPROX_FRAME_SYNC = 105,
       CAP_PROP_OPENNI_MAX_BUFFER_SIZE   = 106,
       CAP_PROP_OPENNI_CIRCLE_BUFFER     = 107,
       CAP_PROP_OPENNI_MAX_TIME_DURATION = 108,
       CAP_PROP_OPENNI_GENERATOR_PRESENT = 109,
       CAP_PROP_OPENNI2_SYNC             = 110,
       CAP_PROP_OPENNI2_MIRROR           = 111
     };

//! OpenNI shortcuts
enum { CAP_OPENNI_IMAGE_GENERATOR_PRESENT         = CAP_OPENNI_IMAGE_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT,
       CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE     = CAP_OPENNI_IMAGE_GENERATOR + CAP_PROP_OPENNI_OUTPUT_MODE,
       CAP_OPENNI_DEPTH_GENERATOR_PRESENT         = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT,
       CAP_OPENNI_DEPTH_GENERATOR_BASELINE        = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_BASELINE,
       CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH    = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_FOCAL_LENGTH,
       CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION    = CAP_OPENNI_DEPTH_GENERATOR + CAP_PROP_OPENNI_REGISTRATION,
       CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON = CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION,
       CAP_OPENNI_IR_GENERATOR_PRESENT            = CAP_OPENNI_IR_GENERATOR + CAP_PROP_OPENNI_GENERATOR_PRESENT,
     };

//! OpenNI data given from depth generator
enum { CAP_OPENNI_DEPTH_MAP         = 0, //!< Depth values in mm (CV_16UC1)
       CAP_OPENNI_POINT_CLOUD_MAP   = 1, //!< XYZ in meters (CV_32FC3)
       CAP_OPENNI_DISPARITY_MAP     = 2, //!< Disparity in pixels (CV_8UC1)
       CAP_OPENNI_DISPARITY_MAP_32F = 3, //!< Disparity in pixels (CV_32FC1)
       CAP_OPENNI_VALID_DEPTH_MASK  = 4, //!< CV_8UC1

       CAP_OPENNI_BGR_IMAGE         = 5, //!< Data given from RGB image generator
       CAP_OPENNI_GRAY_IMAGE        = 6, //!< Data given from RGB image generator

       CAP_OPENNI_IR_IMAGE          = 7  //!< Data given from IR image generator
     };

//! Supported output modes of OpenNI image generator
enum { CAP_OPENNI_VGA_30HZ  = 0,
       CAP_OPENNI_SXGA_15HZ = 1,
       CAP_OPENNI_SXGA_30HZ = 2,
       CAP_OPENNI_QVGA_30HZ = 3,
       CAP_OPENNI_QVGA_60HZ = 4
     };

//! @} OpenNI

/** @name GStreamer
    @{
*/

enum { CAP_PROP_GSTREAMER_QUEUE_LENGTH = 200 //!< Default is 1
     };

//! @} GStreamer

/** @name PvAPI, Prosilica GigE SDK
    @{
*/

//! PVAPI
enum { CAP_PROP_PVAPI_MULTICASTIP           = 300, //!< IP for enable multicast master mode. 0 for disable multicast.
       CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE = 301, //!< FrameStartTriggerMode: Determines how a frame is initiated.
       CAP_PROP_PVAPI_DECIMATIONHORIZONTAL  = 302, //!< Horizontal sub-sampling of the image.
       CAP_PROP_PVAPI_DECIMATIONVERTICAL    = 303, //!< Vertical sub-sampling of the image.
       CAP_PROP_PVAPI_BINNINGX              = 304, //!< Horizontal binning factor.
       CAP_PROP_PVAPI_BINNINGY              = 305, //!< Vertical binning factor.
       CAP_PROP_PVAPI_PIXELFORMAT           = 306  //!< Pixel format.
     };

//! PVAPI: FrameStartTriggerMode
enum { CAP_PVAPI_FSTRIGMODE_FREERUN     = 0,    //!< Freerun
       CAP_PVAPI_FSTRIGMODE_SYNCIN1     = 1,    //!< SyncIn1
       CAP_PVAPI_FSTRIGMODE_SYNCIN2     = 2,    //!< SyncIn2
       CAP_PVAPI_FSTRIGMODE_FIXEDRATE   = 3,    //!< FixedRate
       CAP_PVAPI_FSTRIGMODE_SOFTWARE    = 4     //!< Software
     };

//! PVAPI: DecimationHorizontal, DecimationVertical
enum { CAP_PVAPI_DECIMATION_OFF       = 1,    //!< Off
       CAP_PVAPI_DECIMATION_2OUTOF4   = 2,    //!< 2 out of 4 decimation
       CAP_PVAPI_DECIMATION_2OUTOF8   = 4,    //!< 2 out of 8 decimation
       CAP_PVAPI_DECIMATION_2OUTOF16  = 8     //!< 2 out of 16 decimation
     };

//! PVAPI: PixelFormat
enum { CAP_PVAPI_PIXELFORMAT_MONO8    = 1,    //!< Mono8
       CAP_PVAPI_PIXELFORMAT_MONO16   = 2,    //!< Mono16
       CAP_PVAPI_PIXELFORMAT_BAYER8   = 3,    //!< Bayer8
       CAP_PVAPI_PIXELFORMAT_BAYER16  = 4,    //!< Bayer16
       CAP_PVAPI_PIXELFORMAT_RGB24    = 5,    //!< Rgb24
       CAP_PVAPI_PIXELFORMAT_BGR24    = 6,    //!< Bgr24
       CAP_PVAPI_PIXELFORMAT_RGBA32   = 7,    //!< Rgba32
       CAP_PVAPI_PIXELFORMAT_BGRA32   = 8,    //!< Bgra32
     };

//! @} PvAPI

/** @name XIMEA Camera API
    @{
*/

//! Properties of cameras available through XIMEA SDK backend
enum { CAP_PROP_XI_DOWNSAMPLING                                 = 400, //!< Change image resolution by binning or skipping.
       CAP_PROP_XI_DATA_FORMAT                                  = 401, //!< Output data format.
       CAP_PROP_XI_OFFSET_X                                     = 402, //!< Horizontal offset from the origin to the area of interest (in pixels).
       CAP_PROP_XI_OFFSET_Y                                     = 403, //!< Vertical offset from the origin to the area of interest (in pixels).
       CAP_PROP_XI_TRG_SOURCE                                   = 404, //!< Defines source of trigger.
       CAP_PROP_XI_TRG_SOFTWARE                                 = 405, //!< Generates an internal trigger. PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
       CAP_PROP_XI_GPI_SELECTOR                                 = 406, //!< Selects general purpose input.
       CAP_PROP_XI_GPI_MODE                                     = 407, //!< Set general purpose input mode.
       CAP_PROP_XI_GPI_LEVEL                                    = 408, //!< Get general purpose level.
       CAP_PROP_XI_GPO_SELECTOR                                 = 409, //!< Selects general purpose output.
       CAP_PROP_XI_GPO_MODE                                     = 410, //!< Set general purpose output mode.
       CAP_PROP_XI_LED_SELECTOR                                 = 411, //!< Selects camera signalling LED.
       CAP_PROP_XI_LED_MODE                                     = 412, //!< Define camera signalling LED functionality.
       CAP_PROP_XI_MANUAL_WB                                    = 413, //!< Calculates White Balance(must be called during acquisition).
       CAP_PROP_XI_AUTO_WB                                      = 414, //!< Automatic white balance.
       CAP_PROP_XI_AEAG                                         = 415, //!< Automatic exposure/gain.
       CAP_PROP_XI_EXP_PRIORITY                                 = 416, //!< Exposure priority (0.5 - exposure 50%, gain 50%).
       CAP_PROP_XI_AE_MAX_LIMIT                                 = 417, //!< Maximum limit of exposure in AEAG procedure.
       CAP_PROP_XI_AG_MAX_LIMIT                                 = 418, //!< Maximum limit of gain in AEAG procedure.
       CAP_PROP_XI_AEAG_LEVEL                                   = 419, //!< Average intensity of output signal AEAG should achieve(in %).
       CAP_PROP_XI_TIMEOUT                                      = 420, //!< Image capture timeout in milliseconds.
       CAP_PROP_XI_EXPOSURE                                     = 421, //!< Exposure time in microseconds.
       CAP_PROP_XI_EXPOSURE_BURST_COUNT                         = 422, //!< Sets the number of times of exposure in one frame.
       CAP_PROP_XI_GAIN_SELECTOR                                = 423, //!< Gain selector for parameter Gain allows to select different type of gains.
       CAP_PROP_XI_GAIN                                         = 424, //!< Gain in dB.
       CAP_PROP_XI_DOWNSAMPLING_TYPE                            = 426, //!< Change image downsampling type.
       CAP_PROP_XI_BINNING_SELECTOR                             = 427, //!< Binning engine selector.
       CAP_PROP_XI_BINNING_VERTICAL                             = 428, //!< Vertical Binning - number of vertical photo-sensitive cells to combine together.
       CAP_PROP_XI_BINNING_HORIZONTAL                           = 429, //!< Horizontal Binning - number of horizontal photo-sensitive cells to combine together.
       CAP_PROP_XI_BINNING_PATTERN                              = 430, //!< Binning pattern type.
       CAP_PROP_XI_DECIMATION_SELECTOR                          = 431, //!< Decimation engine selector.
       CAP_PROP_XI_DECIMATION_VERTICAL                          = 432, //!< Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.
       CAP_PROP_XI_DECIMATION_HORIZONTAL                        = 433, //!< Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.
       CAP_PROP_XI_DECIMATION_PATTERN                           = 434, //!< Decimation pattern type.
       CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR              = 587, //!< Selects which test pattern generator is controlled by the TestPattern feature.
       CAP_PROP_XI_TEST_PATTERN                                 = 588, //!< Selects which test pattern type is generated by the selected generator.
       CAP_PROP_XI_IMAGE_DATA_FORMAT                            = 435, //!< Output data format.
       CAP_PROP_XI_SHUTTER_TYPE                                 = 436, //!< Change sensor shutter type(CMOS sensor).
       CAP_PROP_XI_SENSOR_TAPS                                  = 437, //!< Number of taps.
       CAP_PROP_XI_AEAG_ROI_OFFSET_X                            = 439, //!< Automatic exposure/gain ROI offset X.
       CAP_PROP_XI_AEAG_ROI_OFFSET_Y                            = 440, //!< Automatic exposure/gain ROI offset Y.
       CAP_PROP_XI_AEAG_ROI_WIDTH                               = 441, //!< Automatic exposure/gain ROI Width.
       CAP_PROP_XI_AEAG_ROI_HEIGHT                              = 442, //!< Automatic exposure/gain ROI Height.
       CAP_PROP_XI_BPC                                          = 445, //!< Correction of bad pixels.
       CAP_PROP_XI_WB_KR                                        = 448, //!< White balance red coefficient.
       CAP_PROP_XI_WB_KG                                        = 449, //!< White balance green coefficient.
       CAP_PROP_XI_WB_KB                                        = 450, //!< White balance blue coefficient.
       CAP_PROP_XI_WIDTH                                        = 451, //!< Width of the Image provided by the device (in pixels).
       CAP_PROP_XI_HEIGHT                                       = 452, //!< Height of the Image provided by the device (in pixels).
       CAP_PROP_XI_REGION_SELECTOR                              = 589, //!< Selects Region in Multiple ROI which parameters are set by width, height, ... ,region mode.
       CAP_PROP_XI_REGION_MODE                                  = 595, //!< Activates/deactivates Region selected by Region Selector.
       CAP_PROP_XI_LIMIT_BANDWIDTH                              = 459, //!< Set/get bandwidth(datarate)(in Megabits).
       CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH                        = 460, //!< Sensor output data bit depth.
       CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH                        = 461, //!< Device output data bit depth.
       CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH                         = 462, //!< bitdepth of data returned by function xiGetImage.
       CAP_PROP_XI_OUTPUT_DATA_PACKING                          = 463, //!< Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.
       CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE                     = 464, //!< Data packing type. Some cameras supports only specific packing type.
       CAP_PROP_XI_IS_COOLED                                    = 465, //!< Returns 1 for cameras that support cooling.
       CAP_PROP_XI_COOLING                                      = 466, //!< Start camera cooling.
       CAP_PROP_XI_TARGET_TEMP                                  = 467, //!< Set sensor target temperature for cooling.
       CAP_PROP_XI_CHIP_TEMP                                    = 468, //!< Camera sensor temperature.
       CAP_PROP_XI_HOUS_TEMP                                    = 469, //!< Camera housing temperature.
       CAP_PROP_XI_HOUS_BACK_SIDE_TEMP                          = 590, //!< Camera housing back side temperature.
       CAP_PROP_XI_SENSOR_BOARD_TEMP                            = 596, //!< Camera sensor board temperature.
       CAP_PROP_XI_CMS                                          = 470, //!< Mode of color management system.
       CAP_PROP_XI_APPLY_CMS                                    = 471, //!< Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).
       CAP_PROP_XI_IMAGE_IS_COLOR                               = 474, //!< Returns 1 for color cameras.
       CAP_PROP_XI_COLOR_FILTER_ARRAY                           = 475, //!< Returns color filter array type of RAW data.
       CAP_PROP_XI_GAMMAY                                       = 476, //!< Luminosity gamma.
       CAP_PROP_XI_GAMMAC                                       = 477, //!< Chromaticity gamma.
       CAP_PROP_XI_SHARPNESS                                    = 478, //!< Sharpness Strength.
       CAP_PROP_XI_CC_MATRIX_00                                 = 479, //!< Color Correction Matrix element [0][0].
       CAP_PROP_XI_CC_MATRIX_01                                 = 480, //!< Color Correction Matrix element [0][1].
       CAP_PROP_XI_CC_MATRIX_02                                 = 481, //!< Color Correction Matrix element [0][2].
       CAP_PROP_XI_CC_MATRIX_03                                 = 482, //!< Color Correction Matrix element [0][3].
       CAP_PROP_XI_CC_MATRIX_10                                 = 483, //!< Color Correction Matrix element [1][0].
       CAP_PROP_XI_CC_MATRIX_11                                 = 484, //!< Color Correction Matrix element [1][1].
       CAP_PROP_XI_CC_MATRIX_12                                 = 485, //!< Color Correction Matrix element [1][2].
       CAP_PROP_XI_CC_MATRIX_13                                 = 486, //!< Color Correction Matrix element [1][3].
       CAP_PROP_XI_CC_MATRIX_20                                 = 487, //!< Color Correction Matrix element [2][0].
       CAP_PROP_XI_CC_MATRIX_21                                 = 488, //!< Color Correction Matrix element [2][1].
       CAP_PROP_XI_CC_MATRIX_22                                 = 489, //!< Color Correction Matrix element [2][2].
       CAP_PROP_XI_CC_MATRIX_23                                 = 490, //!< Color Correction Matrix element [2][3].
       CAP_PROP_XI_CC_MATRIX_30                                 = 491, //!< Color Correction Matrix element [3][0].
       CAP_PROP_XI_CC_MATRIX_31                                 = 492, //!< Color Correction Matrix element [3][1].
       CAP_PROP_XI_CC_MATRIX_32                                 = 493, //!< Color Correction Matrix element [3][2].
       CAP_PROP_XI_CC_MATRIX_33                                 = 494, //!< Color Correction Matrix element [3][3].
       CAP_PROP_XI_DEFAULT_CC_MATRIX                            = 495, //!< Set default Color Correction Matrix.
       CAP_PROP_XI_TRG_SELECTOR                                 = 498, //!< Selects the type of trigger.
       CAP_PROP_XI_ACQ_FRAME_BURST_COUNT                        = 499, //!< Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStart.
       CAP_PROP_XI_DEBOUNCE_EN                                  = 507, //!< Enable/Disable debounce to selected GPI.
       CAP_PROP_XI_DEBOUNCE_T0                                  = 508, //!< Debounce time (x * 10us).
       CAP_PROP_XI_DEBOUNCE_T1                                  = 509, //!< Debounce time (x * 10us).
       CAP_PROP_XI_DEBOUNCE_POL                                 = 510, //!< Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge).
       CAP_PROP_XI_LENS_MODE                                    = 511, //!< Status of lens control interface. This shall be set to XI_ON before any Lens operations.
       CAP_PROP_XI_LENS_APERTURE_VALUE                          = 512, //!< Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11.
       CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE                    = 513, //!< Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.
       CAP_PROP_XI_LENS_FOCUS_MOVE                              = 514, //!< Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.
       CAP_PROP_XI_LENS_FOCUS_DISTANCE                          = 515, //!< Lens focus distance in cm.
       CAP_PROP_XI_LENS_FOCAL_LENGTH                            = 516, //!< Lens focal distance in mm.
       CAP_PROP_XI_LENS_FEATURE_SELECTOR                        = 517, //!< Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.
       CAP_PROP_XI_LENS_FEATURE                                 = 518, //!< Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.
       CAP_PROP_XI_DEVICE_MODEL_ID                              = 521, //!< Returns device model id.
       CAP_PROP_XI_DEVICE_SN                                    = 522, //!< Returns device serial number.
       CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA                = 529, //!< The alpha channel of RGB32 output image format.
       CAP_PROP_XI_IMAGE_PAYLOAD_SIZE                           = 530, //!< Buffer size in bytes sufficient for output image returned by xiGetImage.
       CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT                       = 531, //!< Current format of pixels on transport layer.
       CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ                         = 532, //!< Sensor clock frequency in Hz.
       CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX                      = 533, //!< Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.
       CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT                  = 534, //!< Number of output channels from sensor used for data transfer.
       CAP_PROP_XI_FRAMERATE                                    = 535, //!< Define framerate in Hz.
       CAP_PROP_XI_COUNTER_SELECTOR                             = 536, //!< Select counter.
       CAP_PROP_XI_COUNTER_VALUE                                = 537, //!< Counter status.
       CAP_PROP_XI_ACQ_TIMING_MODE                              = 538, //!< Type of sensor frames timing.
       CAP_PROP_XI_AVAILABLE_BANDWIDTH                          = 539, //!< Calculate and returns available interface bandwidth(int Megabits).
       CAP_PROP_XI_BUFFER_POLICY                                = 540, //!< Data move policy.
       CAP_PROP_XI_LUT_EN                                       = 541, //!< Activates LUT.
       CAP_PROP_XI_LUT_INDEX                                    = 542, //!< Control the index (offset) of the coefficient to access in the LUT.
       CAP_PROP_XI_LUT_VALUE                                    = 543, //!< Value at entry LUTIndex of the LUT.
       CAP_PROP_XI_TRG_DELAY                                    = 544, //!< Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.
       CAP_PROP_XI_TS_RST_MODE                                  = 545, //!< Defines how time stamp reset engine will be armed.
       CAP_PROP_XI_TS_RST_SOURCE                                = 546, //!< Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming).
       CAP_PROP_XI_IS_DEVICE_EXIST                              = 547, //!< Returns 1 if camera connected and works properly.
       CAP_PROP_XI_ACQ_BUFFER_SIZE                              = 548, //!< Acquisition buffer size in buffer_size_unit. Default bytes.
       CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT                         = 549, //!< Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytes.
       CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE                    = 550, //!< Acquisition transport buffer size in bytes.
       CAP_PROP_XI_BUFFERS_QUEUE_SIZE                           = 551, //!< Queue of field/frame buffers.
       CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT                  = 552, //!< Number of buffers to commit to low level.
       CAP_PROP_XI_RECENT_FRAME                                 = 553, //!< GetImage returns most recent frame.
       CAP_PROP_XI_DEVICE_RESET                                 = 554, //!< Resets the camera to default state.
       CAP_PROP_XI_COLUMN_FPN_CORRECTION                        = 555, //!< Correction of column FPN.
       CAP_PROP_XI_ROW_FPN_CORRECTION                           = 591, //!< Correction of row FPN.
       CAP_PROP_XI_SENSOR_MODE                                  = 558, //!< Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.
       CAP_PROP_XI_HDR                                          = 559, //!< Enable High Dynamic Range feature.
       CAP_PROP_XI_HDR_KNEEPOINT_COUNT                          = 560, //!< The number of kneepoints in the PWLR.
       CAP_PROP_XI_HDR_T1                                       = 561, //!< Position of first kneepoint(in % of XI_PRM_EXPOSURE).
       CAP_PROP_XI_HDR_T2                                       = 562, //!< Position of second kneepoint (in % of XI_PRM_EXPOSURE).
       CAP_PROP_XI_KNEEPOINT1                                   = 563, //!< Value of first kneepoint (% of sensor saturation).
       CAP_PROP_XI_KNEEPOINT2                                   = 564, //!< Value of second kneepoint (% of sensor saturation).
       CAP_PROP_XI_IMAGE_BLACK_LEVEL                            = 565, //!< Last image black level counts. Can be used for Offline processing to recall it.
       CAP_PROP_XI_HW_REVISION                                  = 571, //!< Returns hardware revision number.
       CAP_PROP_XI_DEBUG_LEVEL                                  = 572, //!< Set debug level.
       CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION                   = 573, //!< Automatic bandwidth calculation.
       CAP_PROP_XI_FFS_FILE_ID                                  = 594, //!< File number.
       CAP_PROP_XI_FFS_FILE_SIZE                                = 580, //!< Size of file.
       CAP_PROP_XI_FREE_FFS_SIZE                                = 581, //!< Size of free camera FFS.
       CAP_PROP_XI_USED_FFS_SIZE                                = 582, //!< Size of used camera FFS.
       CAP_PROP_XI_FFS_ACCESS_KEY                               = 583, //!< Setting of key enables file operations on some cameras.
       CAP_PROP_XI_SENSOR_FEATURE_SELECTOR                      = 585, //!< Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.
       CAP_PROP_XI_SENSOR_FEATURE_VALUE                         = 586, //!< Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.
     };

//! @} XIMEA

/** @name AVFoundation framework for iOS
    OS X Lion will have the same API
    @{
*/

//! Properties of cameras available through AVFOUNDATION backend
enum { CAP_PROP_IOS_DEVICE_FOCUS        = 9001,
       CAP_PROP_IOS_DEVICE_EXPOSURE     = 9002,
       CAP_PROP_IOS_DEVICE_FLASH        = 9003,
       CAP_PROP_IOS_DEVICE_WHITEBALANCE = 9004,
       CAP_PROP_IOS_DEVICE_TORCH        = 9005
     };

/** @name Smartek Giganetix GigEVisionSDK
    @{
*/

//! Properties of cameras available through Smartek Giganetix Ethernet Vision backend
/* --- Vladimir Litvinenko (litvinenko.vladimir@gmail.com) --- */
enum { CAP_PROP_GIGA_FRAME_OFFSET_X   = 10001,
       CAP_PROP_GIGA_FRAME_OFFSET_Y   = 10002,
       CAP_PROP_GIGA_FRAME_WIDTH_MAX  = 10003,
       CAP_PROP_GIGA_FRAME_HEIGH_MAX  = 10004,
       CAP_PROP_GIGA_FRAME_SENS_WIDTH = 10005,
       CAP_PROP_GIGA_FRAME_SENS_HEIGH = 10006
     };

//! @} Smartek

/** @name Intel Perceptual Computing SDK
    @{
*/
enum { CAP_PROP_INTELPERC_PROFILE_COUNT               = 11001,
       CAP_PROP_INTELPERC_PROFILE_IDX                 = 11002,
       CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE  = 11003,
       CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE      = 11004,
       CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD  = 11005,
       CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ     = 11006,
       CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT     = 11007
     };

//! Intel Perceptual Streams
enum { CAP_INTELPERC_DEPTH_GENERATOR = 1 << 29,
       CAP_INTELPERC_IMAGE_GENERATOR = 1 << 28,
       CAP_INTELPERC_GENERATORS_MASK = CAP_INTELPERC_DEPTH_GENERATOR + CAP_INTELPERC_IMAGE_GENERATOR
     };

enum { CAP_INTELPERC_DEPTH_MAP              = 0, //!< Each pixel is a 16-bit integer. The value indicates the distance from an object to the camera's XY plane or the Cartesian depth.
       CAP_INTELPERC_UVDEPTH_MAP            = 1, //!< Each pixel contains two 32-bit floating point values in the range of 0-1, representing the mapping of depth coordinates to the color coordinates.
       CAP_INTELPERC_IR_MAP                 = 2, //!< Each pixel is a 16-bit integer. The value indicates the intensity of the reflected laser beam.
       CAP_INTELPERC_IMAGE                  = 3
     };

//! @} Intel Perceptual

/** @name gPhoto2 connection
    @{
*/

/** @brief gPhoto2 properties

If `propertyId` is less than 0 then work on widget with that __additive inversed__ camera setting ID
Get IDs by using CAP_PROP_GPHOTO2_WIDGET_ENUMERATE.
@see CvCaptureCAM_GPHOTO2 for more info
*/
enum { CAP_PROP_GPHOTO2_PREVIEW           = 17001, //!< Capture only preview from liveview mode.
       CAP_PROP_GPHOTO2_WIDGET_ENUMERATE  = 17002, //!< Readonly, returns (const char *).
       CAP_PROP_GPHOTO2_RELOAD_CONFIG     = 17003, //!< Trigger, only by set. Reload camera settings.
       CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE  = 17004, //!< Reload all settings on set.
       CAP_PROP_GPHOTO2_COLLECT_MSGS      = 17005, //!< Collect messages with details.
       CAP_PROP_GPHOTO2_FLUSH_MSGS        = 17006, //!< Readonly, returns (const char *).
       CAP_PROP_SPEED                     = 17007, //!< Exposure speed. Can be readonly, depends on camera program.
       CAP_PROP_APERTURE                  = 17008, //!< Aperture. Can be readonly, depends on camera program.
       CAP_PROP_EXPOSUREPROGRAM           = 17009, //!< Camera exposure program.
       CAP_PROP_VIEWFINDER                = 17010  //!< Enter liveview mode.
     };

//! @} gPhoto2


/** @name Images backend
    @{
*/

/** @brief Images backend properties

*/
enum { CAP_PROP_IMAGES_BASE = 18000,
       CAP_PROP_IMAGES_LAST = 19000 // excluding
     };

//! @} Images

//! @} videoio_flags_others


class IVideoCapture;

/** @brief Class for video capturing from video files, image sequences or cameras.

The class provides C++ API for capturing video from cameras or for reading video files and image sequences.

Here is how the class can be used:
@include samples/cpp/videocapture_basic.cpp

@note In @ref videoio_c "C API" the black-box structure `CvCapture` is used instead of %VideoCapture.
@note
-   (C++) A basic sample on using the %VideoCapture interface can be found at
    `OPENCV_SOURCE_CODE/samples/cpp/videocapture_starter.cpp`
-   (Python) A basic sample on using the %VideoCapture interface can be found at
    `OPENCV_SOURCE_CODE/samples/python/video.py`
-   (Python) A multi threaded video processing sample can be found at
    `OPENCV_SOURCE_CODE/samples/python/video_threaded.py`
-   (Python) %VideoCapture sample showcasing some features of the Video4Linux2 backend
    `OPENCV_SOURCE_CODE/samples/python/video_v4l2.py`
 */
class CV_EXPORTS_W VideoCapture
{
public:
    /** @brief Default constructor
    @note In @ref videoio_c "C API", when you finished working with video, release CvCapture structure with
    cvReleaseCapture(), or use Ptr\<CvCapture\> that calls cvReleaseCapture() automatically in the
    destructor.
     */
    CV_WRAP VideoCapture();

    /** @overload
    @brief  Open video file or a capturing device or a IP video stream for video capturing

    Same as VideoCapture(const String& filename, int apiPreference) but using default Capture API backends
    */
    CV_WRAP VideoCapture(const String& filename);

    /** @overload
    @brief  Open video file or a capturing device or a IP video stream for video capturing with API Preference

    @param filename it can be:
    - name of video file (eg. `video.avi`)
    - or image sequence (eg. `img_%02d.jpg`, which will read samples like `img_00.jpg, img_01.jpg, img_02.jpg, ...`)
    - or URL of video stream (eg. `protocol://host:port/script_name?script_params|auth`).
      Note that each video stream or IP camera feed has its own URL scheme. Please refer to the
      documentation of source stream to know the right URL.
    @param apiPreference preferred Capture API backends to use. Can be used to enforce a specific reader
    implementation if multiple are available: e.g. cv::CAP_FFMPEG or cv::CAP_IMAGES or cv::CAP_DSHOW.
    @sa The list of supported API backends cv::VideoCaptureAPIs
    */
    CV_WRAP VideoCapture(const String& filename, int apiPreference);

    /** @overload
    @brief  Open a camera for video capturing

    @param index camera_id + domain_offset (CAP_*) id of the video capturing device to open. To open default camera using default backend just pass 0.
    Use a `domain_offset` to enforce a specific reader implementation if multiple are available like cv::CAP_FFMPEG or cv::CAP_IMAGES or cv::CAP_DSHOW.
    e.g. to open Camera 1 using the MS Media Foundation API use `index = 1 + cv::CAP_MSMF`

    @sa The list of supported API backends cv::VideoCaptureAPIs
    */
    CV_WRAP VideoCapture(int index);

    /** @brief Default destructor

    The method first calls VideoCapture::release to close the already opened file or camera.
    */
    virtual ~VideoCapture();

    /** @brief  Open video file or a capturing device or a IP video stream for video capturing

    @overload

    Parameters are same as the constructor VideoCapture(const String& filename)
    @return `true` if the file has been successfully opened

    The method first calls VideoCapture::release to close the already opened file or camera.
     */
    CV_WRAP virtual bool open(const String& filename);

    /** @brief  Open a camera for video capturing

    @overload

    Parameters are same as the constructor VideoCapture(int index)
    @return `true` if the camera has been successfully opened.

    The method first calls VideoCapture::release to close the already opened file or camera.
    */
    CV_WRAP virtual bool open(int index);

   /** @brief  Open a camera for video capturing

    @overload

    Parameters are similar as the constructor VideoCapture(int index),except it takes an additional argument apiPreference.
    Definitely, is same as open(int index) where `index=cameraNum + apiPreference`
    @return `true` if the camera has been successfully opened.
    */
    CV_WRAP bool open(int cameraNum, int apiPreference);

    /** @brief Returns true if video capturing has been initialized already.

    If the previous call to VideoCapture constructor or VideoCapture::open() succeeded, the method returns
    true.
     */
    CV_WRAP virtual bool isOpened() const;

    /** @brief Closes video file or capturing device.

    The method is automatically called by subsequent VideoCapture::open and by VideoCapture
    destructor.

    The C function also deallocates memory and clears \*capture pointer.
     */
    CV_WRAP virtual void release();

    /** @brief Grabs the next frame from video file or capturing device.

    @return `true` (non-zero) in the case of success.

    The method/function grabs the next frame from video file or camera and returns true (non-zero) in
    the case of success.

    The primary use of the function is in multi-camera environments, especially when the cameras do not
    have hardware synchronization. That is, you call VideoCapture::grab() for each camera and after that
    call the slower method VideoCapture::retrieve() to decode and get frame from each camera. This way
    the overhead on demosaicing or motion jpeg decompression etc. is eliminated and the retrieved frames
    from different cameras will be closer in time.

    Also, when a connected camera is multi-head (for example, a stereo camera or a Kinect device), the
    correct way of retrieving data from it is to call VideoCapture::grab() first and then call
    VideoCapture::retrieve() one or more times with different values of the channel parameter.

    @ref tutorial_kinect_openni
     */
    CV_WRAP virtual bool grab();

    /** @brief Decodes and returns the grabbed video frame.

    @param [out] image the video frame is returned here. If no frames has been grabbed the image will be empty.
    @param flag it could be a frame index or a driver specific flag
    @return `false` if no frames has been grabbed

    The method decodes and returns the just grabbed frame. If no frames has been grabbed
    (camera has been disconnected, or there are no more frames in video file), the method returns false
    and the function returns an empty image (with %cv::Mat, test it with Mat::empty()).

    @sa read()

    @note In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the video
    capturing structure. It is not allowed to modify or release the image! You can copy the frame using
    :ocvcvCloneImage and then do whatever you want with the copy.
     */
    CV_WRAP virtual bool retrieve(OutputArray image, int flag = 0);

    /** @brief Stream operator to read the next video frame.
    @sa read()
    */
    virtual VideoCapture& operator >> (CV_OUT Mat& image);

    /** @overload
    @sa read()
    */
    virtual VideoCapture& operator >> (CV_OUT UMat& image);

    /** @brief Grabs, decodes and returns the next video frame.

    @param [out] image the video frame is returned here. If no frames has been grabbed the image will be empty.
    @return `false` if no frames has been grabbed

    The method/function combines VideoCapture::grab() and VideoCapture::retrieve() in one call. This is the
    most convenient method for reading video files or capturing data from decode and returns the just
    grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more
    frames in video file), the method returns false and the function returns empty image (with %cv::Mat, test it with Mat::empty()).

    @note In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the video
    capturing structure. It is not allowed to modify or release the image! You can copy the frame using
    :ocvcvCloneImage and then do whatever you want with the copy.
     */
    CV_WRAP virtual bool read(OutputArray image);

    /** @brief Sets a property in the VideoCapture.

    @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...)
    or one from @ref videoio_flags_others
    @param value Value of the property.
    @return `true` if the property is supported by backend used by the VideoCapture instance.
    @note Even if it returns `true` this doesn't ensure that the property
    value has been accepted by the capture device. See note in VideoCapture::get()
     */
    CV_WRAP virtual bool set(int propId, double value);

    /** @brief Returns the specified VideoCapture property

    @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...)
    or one from @ref videoio_flags_others
    @return Value for the specified property. Value 0 is returned when querying a property that is
    not supported by the backend used by the VideoCapture instance.

    @note Reading / writing properties involves many layers. Some unexpected result might happens
    along this chain.
    @code {.txt}
    `VideoCapture -> API Backend -> Operating System -> Device Driver -> Device Hardware`
    @endcode
    The returned value might be different from what really used by the device or it could be encoded
    using device dependant rules (eg. steps or percentage). Effective behaviour depends from device
    driver and API Backend

    */
    CV_WRAP virtual double get(int propId) const;

    /** @brief Open video file or a capturing device or a IP video stream for video capturing with API Preference

    @overload

    Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference)
    @return `true` if the file has been successfully opened

    The method first calls VideoCapture::release to close the already opened file or camera.
    */
    CV_WRAP virtual bool open(const String& filename, int apiPreference);

protected:
    Ptr<CvCapture> cap;
    Ptr<IVideoCapture> icap;
};

class IVideoWriter;

/** @example videowriter_basic.cpp
An example using VideoCapture and VideoWriter class
 */
/** @brief Video writer class.

The class provides C++ API for writing video files or image sequences.
 */
class CV_EXPORTS_W VideoWriter
{
public:
    /** @brief Default constructors

    The constructors/functions initialize video writers.
    -   On Linux FFMPEG is used to write videos;
    -   On Windows FFMPEG or VFW is used;
    -   On MacOSX QTKit is used.
     */
    CV_WRAP VideoWriter();

    /** @overload
    @param filename Name of the output video file.
    @param fourcc 4-character code of codec used to compress the frames. For example,
    VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a
    motion-jpeg codec etc. List of codes can be obtained at [Video Codecs by
    FOURCC](http://www.fourcc.org/codecs.php) page. FFMPEG backend with MP4 container natively uses
    other values as fourcc code: see [ObjectType](http://www.mp4ra.org/codecs.html),
    so you may receive a warning message from OpenCV about fourcc code conversion.
    @param fps Framerate of the created video stream.
    @param frameSize Size of the video frames.
    @param isColor If it is not zero, the encoder will expect and encode color frames, otherwise it
    will work with grayscale frames (the flag is currently supported on Windows only).

    @b Tips:
    - With some backends `fourcc=-1` pops up the codec selection dialog from the system.
    - To save image sequence use a proper filename (eg. `img_%02d.jpg`) and `fourcc=0`
      OR `fps=0`. Use uncompressed image format (eg. `img_%02d.BMP`) to save raw frames.
    - Most codecs are lossy. If you want lossless video file you need to use a lossless codecs
      (eg. FFMPEG FFV1, Huffman HFYU, Lagarith LAGS, etc...)
    - If FFMPEG is enabled, using `codec=0; fps=0;` you can create an uncompressed (raw) video file.
    */
    CV_WRAP VideoWriter(const String& filename, int fourcc, double fps,
                Size frameSize, bool isColor = true);

    /** @overload
    The `apiPreference` parameter allows to specify API backends to use. Can be used to enforce a specific reader implementation
    if multiple are available: e.g. cv::CAP_FFMPEG or cv::CAP_GSTREAMER.
     */
    CV_WRAP VideoWriter(const String& filename, int apiPreference, int fourcc, double fps,
                Size frameSize, bool isColor = true);

    /** @brief Default destructor

    The method first calls VideoWriter::release to close the already opened file.
    */
    virtual ~VideoWriter();

    /** @brief Initializes or reinitializes video writer.

    The method opens video writer. Parameters are the same as in the constructor
    VideoWriter::VideoWriter.
    @return `true` if video writer has been successfully initialized

    The method first calls VideoWriter::release to close the already opened file.
     */
    CV_WRAP virtual bool open(const String& filename, int fourcc, double fps,
                      Size frameSize, bool isColor = true);

    /** @overload
     */
    CV_WRAP bool open(const String& filename, int apiPreference, int fourcc, double fps,
                      Size frameSize, bool isColor = true);

    /** @brief Returns true if video writer has been successfully initialized.
    */
    CV_WRAP virtual bool isOpened() const;

    /** @brief Closes the video writer.

    The method is automatically called by subsequent VideoWriter::open and by the VideoWriter
    destructor.
     */
    CV_WRAP virtual void release();

    /** @brief Stream operator to write the next video frame.
    @sa write
    */
    virtual VideoWriter& operator << (const Mat& image);

    /** @brief Writes the next video frame

    @param image The written frame

    The function/method writes the specified image to video file. It must have the same size as has
    been specified when opening the video writer.
     */
    CV_WRAP virtual void write(const Mat& image);

    /** @brief Sets a property in the VideoWriter.

     @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY)
     or one of @ref videoio_flags_others

     @param value Value of the property.
     @return  `true` if the property is supported by the backend used by the VideoWriter instance.
     */
    CV_WRAP virtual bool set(int propId, double value);

    /** @brief Returns the specified VideoWriter property

     @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY)
     or one of @ref videoio_flags_others

     @return Value for the specified property. Value 0 is returned when querying a property that is
     not supported by the backend used by the VideoWriter instance.
     */
    CV_WRAP virtual double get(int propId) const;

    /** @brief Concatenates 4 chars to a fourcc code

    @return a fourcc code

    This static method constructs the fourcc code of the codec to be used in the constructor
    VideoWriter::VideoWriter or VideoWriter::open.
     */
    CV_WRAP static int fourcc(char c1, char c2, char c3, char c4);

protected:
    Ptr<CvVideoWriter> writer;
    Ptr<IVideoWriter> iwriter;

    static Ptr<IVideoWriter> create(const String& filename, int fourcc, double fps,
                                    Size frameSize, bool isColor = true);
};

template<> CV_EXPORTS void DefaultDeleter<CvCapture>::operator ()(CvCapture* obj) const;
template<> CV_EXPORTS void DefaultDeleter<CvVideoWriter>::operator ()(CvVideoWriter* obj) const;

//! @} videoio

} // cv

#endif //OPENCV_VIDEOIO_HPP
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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_VIDEOSTAB_HPP
#define OPENCV_VIDEOSTAB_HPP

/**
  @defgroup videostab Video Stabilization

The video stabilization module contains a set of functions and classes that can be used to solve the
problem of video stabilization. There are a few methods implemented, most of them are described in
the papers @cite OF06 and @cite G11 . However, there are some extensions and deviations from the original
paper methods.

### References

 1. "Full-Frame Video Stabilization with Motion Inpainting"
     Yasuyuki Matsushita, Eyal Ofek, Weina Ge, Xiaoou Tang, Senior Member, and Heung-Yeung Shum
 2. "Auto-Directed Video Stabilization with Robust L1 Optimal Camera Paths"
     Matthias Grundmann, Vivek Kwatra, Irfan Essa

     @{
         @defgroup videostab_motion Global Motion Estimation

The video stabilization module contains a set of functions and classes for global motion estimation
between point clouds or between images. In the last case features are extracted and matched
internally. For the sake of convenience the motion estimation functions are wrapped into classes.
Both the functions and the classes are available.

         @defgroup videostab_marching Fast Marching Method

The Fast Marching Method @cite Telea04 is used in of the video stabilization routines to do motion and
color inpainting. The method is implemented is a flexible way and it's made public for other users.

     @}

*/

#include "opencv2/videostab/stabilizer.hpp"
#include "opencv2/videostab/ring_buffer.hpp"

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
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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

#ifndef OPENCV_WORLD_HPP
#define OPENCV_WORLD_HPP

#include "opencv2/core.hpp"

#ifdef __cplusplus
namespace cv
{

CV_EXPORTS_W bool initAll();

}

#endif

#endif
