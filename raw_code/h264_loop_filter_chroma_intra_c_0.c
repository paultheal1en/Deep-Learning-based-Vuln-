static inline void h264_loop_filter_chroma_intra_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta)

{

    int d;

    for( d = 0; d < 8; d++ ) {

        const int p0 = pix[-1*xstride];

        const int p1 = pix[-2*xstride];

        const int q0 = pix[0];

        const int q1 = pix[1*xstride];



        if( FFABS( p0 - q0 ) < alpha &&

            FFABS( p1 - p0 ) < beta &&

            FFABS( q1 - q0 ) < beta ) {



            pix[-xstride] = ( 2*p1 + p0 + q1 + 2 ) >> 2;   /* p0' */

            pix[0]        = ( 2*q1 + q0 + p1 + 2 ) >> 2;   /* q0' */

        }

        pix += ystride;

    }

}
