// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// #include "CompilerDirectives.h"
#include "Point.h"

#include <algorithm>
#include <immintrin.h>
#include <xmmintrin.h>

#ifndef __AABB_H__
#define __AABB_H__

/*! \file AABB.h
    \brief Basic AABB routines
*/


#define DEVICE __attribute__((always_inline))

#if __SSE__ && SINGLEPREC
inline __m128 sse_load_vec3_float(const Point& value)
    {
    float in[4];
    in[0] = value.x[0];
    in[1] = value.x[1];
    in[2] = value.x[2];
    in[3] = 0.0f;
    return _mm_loadu_ps(in);
    }

inline Point sse_unload_vec3_float(const __m128& v)
    {
    float out[4];
    _mm_storeu_ps(out, v);
    return Point(out[0], out[1], out[2]);
    }
#endif

#if __AVX__ && !SINGLEPREC
inline __m256d sse_load_vec3_double(const Point& value)
    {
    double in[4];
    in[0] = value.x[0];
    in[1] = value.x[1];
    in[2] = value.x[2];
    in[3] = 0.0;
    return _mm256_loadu_pd(in);
    }

inline Point sse_unload_vec3_double(const __m256d& v)
    {
    double out[4];
    _mm256_storeu_pd(out, v);
    return Point(out[0], out[1], out[2]);
    }
#endif

namespace hpmc
{

namespace detail
{

/*! \addtogroup overlap
    @{
*/

//! Axis aligned bounding box
/*! An AABB represents a bounding volume defined by an axis-aligned bounding box. It is stored as plain old data
    with a lower and upper bound. This is to make the most common operation of AABB overlap testing fast.

    Do not access data members directly. AABB uses SSE and AVX optimizations and the internal data format changes.
    It also changes between the CPU and GPU. Instead, use the accessor methods getLower(), getUpper() and getPosition().

    Operations are provided as free functions to perform the following operations:

    - merge()
    - overlap()
    - contains()
*/
struct AABB
    {
    #if __AVX__ && !SINGLEPREC && 0
    __m256d lower_v;    //!< Lower left corner (AVX data type)
    __m256d upper_v;   //!< Upper left corner (AVX data type)

    __m128 lower_v;     //! Lower left corner (SSE data type)
    __m128 upper_v;     //! Upper left corner (SSE data type)

    #else
    Point lower;  //!< Lower left corner
    Point upper;  //!< Upper right corner

    #endif

    unsigned int tag;  //! Optional tag id, useful for particle ids

    //! Default construct a 0 AABB
    DEVICE AABB() : tag(0)
        {

        #if __AVX__ && !SINGLEPREC && 0
        double in = 0.0f;
        lower_v = _mm256_broadcast_sd(&in);
        upper_v = _mm256_broadcast_sd(&in);

        #elif __SSE__ && SINGLEPREC
        float in = 0.0f;
        lower_v = _mm_load_ps1(&in);
        upper_v = _mm_load_ps1(&in);

        #else

        lower = Point(0,0,0);
        upper = Point(0,0,0);
        #endif
        }

    //! Construct an AABB from the given lower and upper corners
    /*! @param _lower Lower left corner of the AABB
        @param _upper Upper right corner of the AABB
    */
    DEVICE AABB(const Point& _lower, const Point& _upper) : tag(0)
        {
        #if __AVX__ && !SINGLEPREC && 0
        lower_v = sse_load_vec3_double(_lower);
        upper_v = sse_load_vec3_double(_upper);

        #elif __SSE__ && SINGLEPREC
        lower_v = sse_load_vec3_float(_lower);
        upper_v = sse_load_vec3_float(_upper);

        #else
        lower = _lower;
        upper = _upper;

        #endif
        }

    //! Construct an AABB from a sphere
    /*! @param _position Position of the sphere
        @param radius Radius of the sphere
    */
    DEVICE AABB(const Point& _position, double radius) : tag(0)
        {
        Point new_lower, new_upper;
        new_lower.x[0] = _position.x[0] - radius;
        new_lower.x[1] = _position.x[1] - radius;
        new_lower.x[2] = _position.x[2] - radius;
        new_upper.x[0] = _position.x[0] + radius;
        new_upper.x[1] = _position.x[1] + radius;
        new_upper.x[2] = _position.x[2] + radius;

        #if __AVX__ && !SINGLEPREC && 0
        lower_v = sse_load_vec3_double(new_lower);
        upper_v = sse_load_vec3_double(new_upper);

        #elif __SSE__ && SINGLEPREC
        lower_v = sse_load_vec3_float(new_lower);
        upper_v = sse_load_vec3_float(new_upper);

        #else
        lower = new_lower;
        upper = new_upper;

        #endif
        }

    //! Construct an AABB from a point with a particle tag
    /*! @param _position Position of the point
        @param _tag Global particle tag id
    */
    DEVICE AABB(const Point& _position, unsigned int _tag) : tag(_tag)
        {
        #if __AVX__ && !SINGLEPREC && 0
        lower_v = sse_load_vec3_double(_position);
        upper_v = sse_load_vec3_double(_position);

        #elif __SSE__ && SINGLEPREC
        lower_v = sse_load_vec3_float(_position);
        upper_v = sse_load_vec3_float(_position);

        #else
        lower = _position;
        upper = _position;

        #endif
        }

    //! Get the AABB's position
    DEVICE Point getPosition() const
        {
        #if __AVX__ && !SINGLEPREC && 0
        double half = 0.5;
        __m256d half_v = _mm256_broadcast_sd(&half);
        __m256d pos_v = _mm256_mul_pd(half_v, _mm256_add_pd(lower_v, upper_v));
        return sse_unload_vec3_double(pos_v);

        #elif __SSE__ && SINGLEPREC
        float half = 0.5f;
        __m128 half_v = _mm_load_ps1(&half);
        __m128 pos_v = _mm_mul_ps(half_v, _mm_add_ps(lower_v, upper_v));
        return sse_unload_vec3_float(pos_v);

        #else
        return (lower + upper) / 2;

        #endif
        }

    //! Get the AABB's lower point
    DEVICE Point getLower() const
        {
        #if __AVX__ && !SINGLEPREC && 0
        return sse_unload_vec3_double(lower_v);

        #elif __SSE__ && SINGLEPREC
        return sse_unload_vec3_float(lower_v);

        #else
        return lower;

        #endif
        }

    //! Get the AABB's upper point
    DEVICE Point getUpper() const
        {
        #if __AVX__ && !SINGLEPREC && 0
        return sse_unload_vec3_double(upper_v);

        #elif __SSE__ && SINGLEPREC
        return sse_unload_vec3_float(upper_v);

        #else
        return upper;

        #endif
        }

    //! Translate the AABB by the given vector
    DEVICE void translate(const Point& v)
        {
        #if __AVX__ && !SINGLEPREC && 0
        __m256d v_v = sse_load_vec3_double(v);
        lower_v = _mm256_add_pd(lower_v, v_v);
        upper_v = _mm256_add_pd(upper_v, v_v);

        #elif __SSE__ && SINGLEPREC
        __m128 v_v = sse_load_vec3_float(v);
        lower_v = _mm_add_ps(lower_v, v_v);
        upper_v = _mm_add_ps(upper_v, v_v);

        #else
        upper += v;
        lower += v;

        #endif
        }
    } __attribute__((aligned(32)));

//! Check if two AABBs overlap
/*! @param a First AABB
    @param b Second AABB
    \returns true when the two AABBs overlap, false otherwise
*/
DEVICE inline bool overlap(const AABB& a, const AABB& b)
    {
    #if __AVX__ && !SINGLEPREC && 0
    int r0 = _mm256_movemask_pd(_mm256_cmp_pd(b.upper_v,a.lower_v,0x11));  // 0x11=lt
    int r1 = _mm256_movemask_pd(_mm256_cmp_pd(b.lower_v,a.upper_v,0x1e));  // 0x1e=gt
    return !(r0 || r1);

    #elif __SSE__ && SINGLEPREC
    int r0 = _mm_movemask_ps(_mm_cmplt_ps(b.upper_v,a.lower_v));
    int r1 = _mm_movemask_ps(_mm_cmpgt_ps(b.lower_v,a.upper_v));
    return !(r0 || r1);

    #else
    return !(   b.upper.x[0] < a.lower.x[0]
             || b.lower.x[0] > a.upper.x[0]
             || b.upper.x[1] < a.lower.x[1]
             || b.lower.x[1] > a.upper.x[1]
             || b.upper.x[2] < a.lower.x[2]
             || b.lower.x[2] > a.upper.x[2]
            );

    #endif
    }

//! Check if one AABB contains another
/*! @param a First AABB
    @param b Second AABB
    \returns true when b is fully contained within a
*/
DEVICE inline bool contains(const AABB& a, const AABB& b)
    {
    #if __AVX__ && !SINGLEPREC && 0
    int r0 = _mm256_movemask_pd(_mm256_cmp_pd(b.lower_v,a.lower_v,0x1d));  // 0x1d=ge
    int r1 = _mm256_movemask_pd(_mm256_cmp_pd(b.upper_v,a.upper_v,0x12));  // 0x12=le
    return ((r0 & r1) == 0xF);

    #elif __SSE__ && SINGLEPREC
    int r0 = _mm_movemask_ps(_mm_cmpge_ps(b.lower_v,a.lower_v));
    int r1 = _mm_movemask_ps(_mm_cmple_ps(b.upper_v,a.upper_v));
    return ((r0 & r1) == 0xF);

    #else
    return (   b.lower.x[0] >= a.lower.x[0] && b.upper.x[0] <= a.upper.x[0]
            && b.lower.x[1] >= a.lower.x[1] && b.upper.x[1] <= a.upper.x[1]
            && b.lower.x[2] >= a.lower.x[2] && b.upper.x[2] <= a.upper.x[2]);

    #endif
    }


//! Merge two AABBs
/*! @param a First AABB
    @param b Second AABB
    \returns A new AABB that encloses *a* and *b*
*/
DEVICE inline AABB merge(const AABB& a, const AABB& b)
    {
    AABB new_aabb;
    #if __AVX__ && !SINGLEPREC && 0
    new_aabb.lower_v = _mm256_min_pd(a.lower_v, b.lower_v);
    new_aabb.upper_v = _mm256_max_pd(a.upper_v, b.upper_v);

    #elif __SSE__ && SINGLEPREC
    new_aabb.lower_v = _mm_min_ps(a.lower_v, b.lower_v);
    new_aabb.upper_v = _mm_max_ps(a.upper_v, b.upper_v);

    #else
    new_aabb.lower.x[0] = std::min(a.lower.x[0], b.lower.x[0]);
    new_aabb.lower.x[1] = std::min(a.lower.x[1], b.lower.x[1]);
    new_aabb.lower.x[2] = std::min(a.lower.x[2], b.lower.x[2]);
    new_aabb.upper.x[0] = std::max(a.upper.x[0], b.upper.x[0]);
    new_aabb.upper.x[1] = std::max(a.upper.x[1], b.upper.x[1]);
    new_aabb.upper.x[2] = std::max(a.upper.x[2], b.upper.x[2]);

    #endif

    return new_aabb;
    }

// end group overlap
/*! @}*/

}; // end namespace detail

}; // end namespace hpmc

#undef DEVICE
#endif //__AABB_H__
