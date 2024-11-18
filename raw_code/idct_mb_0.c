static av_always_inline void idct_mb(VP8Context *s, uint8_t *dst[3], VP8Macroblock *mb)

{

    int x, y, ch;



    if (mb->mode != MODE_I4x4) {

        uint8_t *y_dst = dst[0];

        for (y = 0; y < 4; y++) {

            uint32_t nnz4 = AV_RL32(s->non_zero_count_cache[y]);

            if (nnz4) {

                if (nnz4&~0x01010101) {

                    for (x = 0; x < 4; x++) {

                        if ((uint8_t)nnz4 == 1)

                            s->vp8dsp.vp8_idct_dc_add(y_dst+4*x, s->block[y][x], s->linesize);

                        else if((uint8_t)nnz4 > 1)

                            s->vp8dsp.vp8_idct_add(y_dst+4*x, s->block[y][x], s->linesize);

                        nnz4 >>= 8;

                        if (!nnz4)

                            break;

                    }

                } else {

                    s->vp8dsp.vp8_idct_dc_add4y(y_dst, s->block[y], s->linesize);

                }

            }

            y_dst += 4*s->linesize;

        }

    }



    for (ch = 0; ch < 2; ch++) {

        uint32_t nnz4 = AV_RL32(s->non_zero_count_cache[4+ch]);

        if (nnz4) {

            uint8_t *ch_dst = dst[1+ch];

            if (nnz4&~0x01010101) {

                for (y = 0; y < 2; y++) {

                    for (x = 0; x < 2; x++) {

                        if ((uint8_t)nnz4 == 1)

                            s->vp8dsp.vp8_idct_dc_add(ch_dst+4*x, s->block[4+ch][(y<<1)+x], s->uvlinesize);

                        else if((uint8_t)nnz4 > 1)

                            s->vp8dsp.vp8_idct_add(ch_dst+4*x, s->block[4+ch][(y<<1)+x], s->uvlinesize);

                        nnz4 >>= 8;

                        if (!nnz4)

                            break;

                    }

                    ch_dst += 4*s->uvlinesize;

                }

            } else {

                s->vp8dsp.vp8_idct_dc_add4uv(ch_dst, s->block[4+ch], s->uvlinesize);

            }

        }

    }

}
