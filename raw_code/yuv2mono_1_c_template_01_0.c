yuv2mono_1_c_template(SwsContext *c, const uint16_t *buf0,

                      const uint16_t *ubuf0, const uint16_t *ubuf1,

                      const uint16_t *vbuf0, const uint16_t *vbuf1,

                      const uint16_t *abuf0, uint8_t *dest, int dstW,

                      int uvalpha, enum PixelFormat dstFormat,

                      int flags, int y, enum PixelFormat target)

{

    const uint8_t * const d128 = dither_8x8_220[y & 7];

    uint8_t *g = c->table_gU[128] + c->table_gV[128];

    int i;



    for (i = 0; i < dstW - 7; i += 8) {

        int acc =    g[(buf0[i    ] >> 7) + d128[0]];

        acc += acc + g[(buf0[i + 1] >> 7) + d128[1]];

        acc += acc + g[(buf0[i + 2] >> 7) + d128[2]];

        acc += acc + g[(buf0[i + 3] >> 7) + d128[3]];

        acc += acc + g[(buf0[i + 4] >> 7) + d128[4]];

        acc += acc + g[(buf0[i + 5] >> 7) + d128[5]];

        acc += acc + g[(buf0[i + 6] >> 7) + d128[6]];

        acc += acc + g[(buf0[i + 7] >> 7) + d128[7]];

        output_pixel(*dest++, acc);

    }

}
