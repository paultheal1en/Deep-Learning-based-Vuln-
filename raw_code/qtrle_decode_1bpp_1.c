static void qtrle_decode_1bpp(QtrleContext *s, int stream_ptr, int row_ptr, int lines_to_change)

{

    int rle_code;

    int pixel_ptr = 0;

    int row_inc = s->frame.linesize[0];

    unsigned char pi0, pi1;  /* 2 8-pixel values */

    unsigned char *rgb = s->frame.data[0];

    int pixel_limit = s->frame.linesize[0] * s->avctx->height;

    int skip;



    while (lines_to_change) {

        CHECK_STREAM_PTR(2);

        skip = s->buf[stream_ptr++];

        rle_code = (signed char)s->buf[stream_ptr++];

        if (rle_code == 0)

            break;

        if(skip & 0x80) {

            lines_to_change--;

            row_ptr += row_inc;

            pixel_ptr = row_ptr + 2 * (skip & 0x7f);

        } else

            pixel_ptr += 2 * skip;

        CHECK_PIXEL_PTR(0);  /* make sure pixel_ptr is positive */



        if (rle_code < 0) {

            /* decode the run length code */

            rle_code = -rle_code;

            /* get the next 2 bytes from the stream, treat them as groups

             * of 8 pixels, and output them rle_code times */

            CHECK_STREAM_PTR(2);

            pi0 = s->buf[stream_ptr++];

            pi1 = s->buf[stream_ptr++];

            CHECK_PIXEL_PTR(rle_code * 2);



            while (rle_code--) {

                rgb[pixel_ptr++] = pi0;

                rgb[pixel_ptr++] = pi1;

            }

        } else {

            /* copy the same pixel directly to output 2 times */

            rle_code *= 2;

            CHECK_STREAM_PTR(rle_code);

            CHECK_PIXEL_PTR(rle_code);



            while (rle_code--)

                rgb[pixel_ptr++] = s->buf[stream_ptr++];

        }

    }

}
