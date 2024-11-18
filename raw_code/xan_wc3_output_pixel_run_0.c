static void inline xan_wc3_output_pixel_run(XanContext *s, 

    unsigned char *pixel_buffer, int x, int y, int pixel_count)

{

    int stride;

    int line_inc;

    int index;

    int current_x;

    int width = s->avctx->width;

    unsigned char pix;

    unsigned char *palette_plane;

    unsigned char *y_plane;

    unsigned char *u_plane;

    unsigned char *v_plane;

    unsigned char *rgb_plane;

    unsigned short *rgb16_plane;

    unsigned short *palette16;

    unsigned int *rgb32_plane;

    unsigned int *palette32;



    switch (s->avctx->pix_fmt) {



    case PIX_FMT_PAL8:

        palette_plane = s->current_frame.data[0];

        stride = s->current_frame.linesize[0];

        line_inc = stride - width;

        index = y * stride + x;

        current_x = x;

        while(pixel_count--) {



            /* don't do a memcpy() here; keyframes generally copy an entire

             * frame of data and the stride needs to be accounted for */

            palette_plane[index++] = *pixel_buffer++;



            ADVANCE_CURRENT_X();

        }

        break;



    case PIX_FMT_RGB555:

    case PIX_FMT_RGB565:

        rgb16_plane = (unsigned short *)s->current_frame.data[0];

        palette16 = (unsigned short *)s->palette;

        stride = s->current_frame.linesize[0] / 2;

        line_inc = stride - width;

        index = y * stride + x;

        current_x = x;

        while(pixel_count--) {



            rgb16_plane[index++] = palette16[*pixel_buffer++];



            ADVANCE_CURRENT_X();

        }

        break;



    case PIX_FMT_RGB24:

    case PIX_FMT_BGR24:

        rgb_plane = s->current_frame.data[0];

        stride = s->current_frame.linesize[0];

        line_inc = stride - width * 3;

        index = y * stride + x * 3;

        current_x = x;

        while(pixel_count--) {

            pix = *pixel_buffer++;



            rgb_plane[index++] = s->palette[pix * 4 + 0];

            rgb_plane[index++] = s->palette[pix * 4 + 1];

            rgb_plane[index++] = s->palette[pix * 4 + 2];



            ADVANCE_CURRENT_X();

        }

        break;



    case PIX_FMT_RGBA32:

        rgb32_plane = (unsigned int *)s->current_frame.data[0];

        palette32 = (unsigned int *)s->palette;

        stride = s->current_frame.linesize[0] / 4;

        line_inc = stride - width;

        index = y * stride + x;

        current_x = x;

        while(pixel_count--) {



            rgb32_plane[index++] = palette32[*pixel_buffer++];



            ADVANCE_CURRENT_X();

        }

        break;



    case PIX_FMT_YUV444P:

        y_plane = s->current_frame.data[0];

        u_plane = s->current_frame.data[1];

        v_plane = s->current_frame.data[2];

        stride = s->current_frame.linesize[0];

        line_inc = stride - width;

        index = y * stride + x;

        current_x = x;

        while(pixel_count--) {

            pix = *pixel_buffer++;



            y_plane[index] = s->palette[pix * 4 + 0];

            u_plane[index] = s->palette[pix * 4 + 1];

            v_plane[index] = s->palette[pix * 4 + 2];



            index++;

            ADVANCE_CURRENT_X();

        }

        break;



    default:

        av_log(s->avctx, AV_LOG_ERROR, " Xan WC3: Unhandled colorspace\n");

        break;

    }

}
