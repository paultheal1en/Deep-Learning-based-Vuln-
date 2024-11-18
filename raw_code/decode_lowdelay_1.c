static int decode_lowdelay(DiracContext *s)

{

    AVCodecContext *avctx = s->avctx;

    int slice_x, slice_y, bufsize;

    int64_t coef_buf_size, bytes = 0;

    const uint8_t *buf;

    DiracSlice *slices;

    SliceCoeffs tmp[MAX_DWT_LEVELS];

    int slice_num = 0;



    if (s->slice_params_num_buf != (s->num_x * s->num_y)) {

        s->slice_params_buf = av_realloc_f(s->slice_params_buf, s->num_x * s->num_y, sizeof(DiracSlice));

        if (!s->slice_params_buf) {

            av_log(s->avctx, AV_LOG_ERROR, "slice params buffer allocation failure\n");


            return AVERROR(ENOMEM);

        }

        s->slice_params_num_buf = s->num_x * s->num_y;

    }

    slices = s->slice_params_buf;



    /* 8 becacuse that's how much the golomb reader could overread junk data

     * from another plane/slice at most, and 512 because SIMD */

    coef_buf_size = subband_coeffs(s, s->num_x - 1, s->num_y - 1, 0, tmp) + 8;

    coef_buf_size = (coef_buf_size << (1 + s->pshift)) + 512;



    if (s->threads_num_buf != avctx->thread_count ||

        s->thread_buf_size != coef_buf_size) {

        s->threads_num_buf  = avctx->thread_count;

        s->thread_buf_size  = coef_buf_size;

        s->thread_buf       = av_realloc_f(s->thread_buf, avctx->thread_count, s->thread_buf_size);

        if (!s->thread_buf) {

            av_log(s->avctx, AV_LOG_ERROR, "thread buffer allocation failure\n");

            return AVERROR(ENOMEM);

        }

    }



    align_get_bits(&s->gb);

    /*[DIRAC_STD] 13.5.2 Slices. slice(sx,sy) */

    buf = s->gb.buffer + get_bits_count(&s->gb)/8;

    bufsize = get_bits_left(&s->gb);



    if (s->hq_picture) {

        int i;



        for (slice_y = 0; bufsize > 0 && slice_y < s->num_y; slice_y++) {

            for (slice_x = 0; bufsize > 0 && slice_x < s->num_x; slice_x++) {

                bytes = s->highquality.prefix_bytes + 1;

                for (i = 0; i < 3; i++) {

                    if (bytes <= bufsize/8)

                        bytes += buf[bytes] * s->highquality.size_scaler + 1;

                }

                if (bytes >= INT_MAX || bytes*8 > bufsize) {

                    av_log(s->avctx, AV_LOG_ERROR, "too many bytes\n");

                    return AVERROR_INVALIDDATA;

                }



                slices[slice_num].bytes   = bytes;

                slices[slice_num].slice_x = slice_x;

                slices[slice_num].slice_y = slice_y;

                init_get_bits(&slices[slice_num].gb, buf, bufsize);

                slice_num++;



                buf     += bytes;

                if (bufsize/8 >= bytes)

                    bufsize -= bytes*8;

                else

                    bufsize = 0;

            }

        }



        if (s->num_x*s->num_y != slice_num) {

            av_log(s->avctx, AV_LOG_ERROR, "too few slices\n");

            return AVERROR_INVALIDDATA;

        }



        avctx->execute2(avctx, decode_hq_slice_row, slices, NULL, s->num_y);

    } else {

        for (slice_y = 0; bufsize > 0 && slice_y < s->num_y; slice_y++) {

            for (slice_x = 0; bufsize > 0 && slice_x < s->num_x; slice_x++) {

                bytes = (slice_num+1) * (int64_t)s->lowdelay.bytes.num / s->lowdelay.bytes.den

                       - slice_num    * (int64_t)s->lowdelay.bytes.num / s->lowdelay.bytes.den;

                slices[slice_num].bytes   = bytes;

                slices[slice_num].slice_x = slice_x;

                slices[slice_num].slice_y = slice_y;

                init_get_bits(&slices[slice_num].gb, buf, bufsize);

                slice_num++;



                buf     += bytes;

                if (bufsize/8 >= bytes)

                    bufsize -= bytes*8;

                else

                    bufsize = 0;

            }

        }

        avctx->execute(avctx, decode_lowdelay_slice, slices, NULL, slice_num,

                       sizeof(DiracSlice)); /* [DIRAC_STD] 13.5.2 Slices */

    }



    if (s->dc_prediction) {

        if (s->pshift) {

            intra_dc_prediction_10(&s->plane[0].band[0][0]); /* [DIRAC_STD] 13.3 intra_dc_prediction() */

            intra_dc_prediction_10(&s->plane[1].band[0][0]); /* [DIRAC_STD] 13.3 intra_dc_prediction() */

            intra_dc_prediction_10(&s->plane[2].band[0][0]); /* [DIRAC_STD] 13.3 intra_dc_prediction() */

        } else {

            intra_dc_prediction_8(&s->plane[0].band[0][0]);

            intra_dc_prediction_8(&s->plane[1].band[0][0]);

            intra_dc_prediction_8(&s->plane[2].band[0][0]);

        }

    }



    return 0;

}