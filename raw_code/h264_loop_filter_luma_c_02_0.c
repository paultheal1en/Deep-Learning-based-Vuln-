static inline void h264_loop_filter_luma_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta, int8_t *tc0)

{

    int i, d;

    for( i = 0; i < 4; i++ ) {

        if( tc0[i] < 0 ) {

            pix += 4*ystride;

            continue;

        }

        for( d = 0; d < 4; d++ ) {

            const int p0 = pix[-1*xstride];

            const int p1 = pix[-2*xstride];

            const int p2 = pix[-3*xstride];

            const int q0 = pix[0];

            const int q1 = pix[1*xstride];

            const int q2 = pix[2*xstride];



            if( FFABS( p0 - q0 ) < alpha &&

                FFABS( p1 - p0 ) < beta &&

                FFABS( q1 - q0 ) < beta ) {



                int tc = tc0[i];

                int i_delta;



                if( FFABS( p2 - p0 ) < beta ) {

                    if(tc0[i])

                    pix[-2*xstride] = p1 + av_clip( (( p2 + ( ( p0 + q0 + 1 ) >> 1 ) ) >> 1) - p1, -tc0[i], tc0[i] );

                    tc++;

                }

                if( FFABS( q2 - q0 ) < beta ) {

                    if(tc0[i])

                    pix[   xstride] = q1 + av_clip( (( q2 + ( ( p0 + q0 + 1 ) >> 1 ) ) >> 1) - q1, -tc0[i], tc0[i] );

                    tc++;

                }



                i_delta = av_clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );

                pix[-xstride] = av_clip_uint8( p0 + i_delta );    /* p0' */

                pix[0]        = av_clip_uint8( q0 - i_delta );    /* q0' */

            }

            pix += ystride;

        }

    }

}
