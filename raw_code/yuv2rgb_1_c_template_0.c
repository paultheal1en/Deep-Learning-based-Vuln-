yuv2rgb_1_c_template(SwsContext *c, const uint16_t *buf0,

                     const uint16_t *ubuf0, const uint16_t *ubuf1,

                     const uint16_t *vbuf0, const uint16_t *vbuf1,

                     const uint16_t *abuf0, uint8_t *dest, int dstW,

                     int uvalpha, enum PixelFormat dstFormat,

                     int flags, int y, enum PixelFormat target,

                     int hasAlpha)

{

    int i;



    if (uvalpha < 2048) {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 = buf0[i * 2]     >> 7;

            int Y2 = buf0[i * 2 + 1] >> 7;

            int U  = ubuf1[i]        >> 7;

            int V  = vbuf1[i]        >> 7;

            int A1, A2;

            const void *r =  c->table_rV[V],

                       *g = (c->table_gU[U] + c->table_gV[V]),

                       *b =  c->table_bU[U];



            if (hasAlpha) {

                A1 = abuf0[i * 2    ] >> 7;

                A2 = abuf0[i * 2 + 1] >> 7;

            }



            yuv2rgb_write(dest, i, Y1, Y2, U, V, hasAlpha ? A1 : 0, hasAlpha ? A2 : 0,

                          r, g, b, y, target, hasAlpha);

        }

    } else {

        for (i = 0; i < (dstW >> 1); i++) {

            int Y1 =  buf0[i * 2]          >> 7;

            int Y2 =  buf0[i * 2 + 1]      >> 7;

            int U  = (ubuf0[i] + ubuf1[i]) >> 8;

            int V  = (vbuf0[i] + vbuf1[i]) >> 8;

            int A1, A2;

            const void *r =  c->table_rV[V],

                       *g = (c->table_gU[U] + c->table_gV[V]),

                       *b =  c->table_bU[U];



            if (hasAlpha) {

                A1 = abuf0[i * 2    ] >> 7;

                A2 = abuf0[i * 2 + 1] >> 7;

            }



            yuv2rgb_write(dest, i, Y1, Y2, U, V, hasAlpha ? A1 : 0, hasAlpha ? A2 : 0,

                          r, g, b, y, target, hasAlpha);

        }

    }

}
