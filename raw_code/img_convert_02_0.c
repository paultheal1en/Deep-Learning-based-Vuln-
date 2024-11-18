int img_convert(AVPicture *dst, int dst_pix_fmt,

                const AVPicture *src, int src_pix_fmt,

                int src_width, int src_height)

{

    static int inited;

    int i, ret, dst_width, dst_height, int_pix_fmt;

    const PixFmtInfo *src_pix, *dst_pix;

    const ConvertEntry *ce;

    AVPicture tmp1, *tmp = &tmp1;



    if (src_pix_fmt < 0 || src_pix_fmt >= PIX_FMT_NB ||

        dst_pix_fmt < 0 || dst_pix_fmt >= PIX_FMT_NB)

        return -1;

    if (src_width <= 0 || src_height <= 0)

        return 0;



    if (!inited) {

        inited = 1;

        img_convert_init();

    }



    dst_width = src_width;

    dst_height = src_height;



    dst_pix = &pix_fmt_info[dst_pix_fmt];

    src_pix = &pix_fmt_info[src_pix_fmt];

    if (src_pix_fmt == dst_pix_fmt) {

        /* no conversion needed: just copy */

        av_picture_copy(dst, src, dst_pix_fmt, dst_width, dst_height);

        return 0;

    }



    ce = &convert_table[src_pix_fmt][dst_pix_fmt];

    if (ce->convert) {

        /* specific conversion routine */

        ce->convert(dst, src, dst_width, dst_height);

        return 0;

    }



    /* gray to YUV */

    if (is_yuv_planar(dst_pix) &&

        src_pix_fmt == PIX_FMT_GRAY8) {

        int w, h, y;

        uint8_t *d;



        if (dst_pix->color_type == FF_COLOR_YUV_JPEG) {

            ff_img_copy_plane(dst->data[0], dst->linesize[0],

                     src->data[0], src->linesize[0],

                     dst_width, dst_height);

        } else {

            img_apply_table(dst->data[0], dst->linesize[0],

                            src->data[0], src->linesize[0],

                            dst_width, dst_height,

                            y_jpeg_to_ccir);

        }

        /* fill U and V with 128 */

        w = dst_width;

        h = dst_height;

        w >>= dst_pix->x_chroma_shift;

        h >>= dst_pix->y_chroma_shift;

        for(i = 1; i <= 2; i++) {

            d = dst->data[i];

            for(y = 0; y< h; y++) {

                memset(d, 128, w);

                d += dst->linesize[i];

            }

        }

        return 0;

    }



    /* YUV to gray */

    if (is_yuv_planar(src_pix) &&

        dst_pix_fmt == PIX_FMT_GRAY8) {

        if (src_pix->color_type == FF_COLOR_YUV_JPEG) {

            ff_img_copy_plane(dst->data[0], dst->linesize[0],

                     src->data[0], src->linesize[0],

                     dst_width, dst_height);

        } else {

            img_apply_table(dst->data[0], dst->linesize[0],

                            src->data[0], src->linesize[0],

                            dst_width, dst_height,

                            y_ccir_to_jpeg);

        }

        return 0;

    }



    /* YUV to YUV planar */

    if (is_yuv_planar(dst_pix) && is_yuv_planar(src_pix)) {

        int x_shift, y_shift, w, h, xy_shift;

        void (*resize_func)(uint8_t *dst, int dst_wrap,

                            const uint8_t *src, int src_wrap,

                            int width, int height);



        /* compute chroma size of the smallest dimensions */

        w = dst_width;

        h = dst_height;

        if (dst_pix->x_chroma_shift >= src_pix->x_chroma_shift)

            w >>= dst_pix->x_chroma_shift;

        else

            w >>= src_pix->x_chroma_shift;

        if (dst_pix->y_chroma_shift >= src_pix->y_chroma_shift)

            h >>= dst_pix->y_chroma_shift;

        else

            h >>= src_pix->y_chroma_shift;



        x_shift = (dst_pix->x_chroma_shift - src_pix->x_chroma_shift);

        y_shift = (dst_pix->y_chroma_shift - src_pix->y_chroma_shift);

        xy_shift = ((x_shift & 0xf) << 4) | (y_shift & 0xf);

        /* there must be filters for conversion at least from and to

           YUV444 format */

        switch(xy_shift) {

        case 0x00:

            resize_func = ff_img_copy_plane;

            break;

        case 0x10:

            resize_func = shrink21;

            break;

        case 0x20:

            resize_func = shrink41;

            break;

        case 0x01:

            resize_func = shrink12;

            break;

        case 0x11:

            resize_func = ff_shrink22;

            break;

        case 0x22:

            resize_func = ff_shrink44;

            break;

        case 0xf0:

            resize_func = grow21;

            break;

        case 0x0f:

            resize_func = grow12;

            break;

        case 0xe0:

            resize_func = grow41;

            break;

        case 0xff:

            resize_func = grow22;

            break;

        case 0xee:

            resize_func = grow44;

            break;

        case 0xf1:

            resize_func = conv411;

            break;

        default:

            /* currently not handled */

            goto no_chroma_filter;

        }



        ff_img_copy_plane(dst->data[0], dst->linesize[0],

                       src->data[0], src->linesize[0],

                       dst_width, dst_height);



        for(i = 1;i <= 2; i++)

            resize_func(dst->data[i], dst->linesize[i],

                        src->data[i], src->linesize[i],

                        dst_width>>dst_pix->x_chroma_shift, dst_height>>dst_pix->y_chroma_shift);

        /* if yuv color space conversion is needed, we do it here on

           the destination image */

        if (dst_pix->color_type != src_pix->color_type) {

            const uint8_t *y_table, *c_table;

            if (dst_pix->color_type == FF_COLOR_YUV) {

                y_table = y_jpeg_to_ccir;

                c_table = c_jpeg_to_ccir;

            } else {

                y_table = y_ccir_to_jpeg;

                c_table = c_ccir_to_jpeg;

            }

            img_apply_table(dst->data[0], dst->linesize[0],

                            dst->data[0], dst->linesize[0],

                            dst_width, dst_height,

                            y_table);



            for(i = 1;i <= 2; i++)

                img_apply_table(dst->data[i], dst->linesize[i],

                                dst->data[i], dst->linesize[i],

                                dst_width>>dst_pix->x_chroma_shift,

                                dst_height>>dst_pix->y_chroma_shift,

                                c_table);

        }

        return 0;

    }

 no_chroma_filter:



    /* try to use an intermediate format */

    if (src_pix_fmt == PIX_FMT_YUYV422 ||

        dst_pix_fmt == PIX_FMT_YUYV422) {

        /* specific case: convert to YUV422P first */

        int_pix_fmt = PIX_FMT_YUV422P;

    } else if (src_pix_fmt == PIX_FMT_UYVY422 ||

        dst_pix_fmt == PIX_FMT_UYVY422) {

        /* specific case: convert to YUV422P first */

        int_pix_fmt = PIX_FMT_YUV422P;

    } else if (src_pix_fmt == PIX_FMT_UYYVYY411 ||

        dst_pix_fmt == PIX_FMT_UYYVYY411) {

        /* specific case: convert to YUV411P first */

        int_pix_fmt = PIX_FMT_YUV411P;

    } else if ((src_pix->color_type == FF_COLOR_GRAY &&

                src_pix_fmt != PIX_FMT_GRAY8) ||

               (dst_pix->color_type == FF_COLOR_GRAY &&

                dst_pix_fmt != PIX_FMT_GRAY8)) {

        /* gray8 is the normalized format */

        int_pix_fmt = PIX_FMT_GRAY8;

    } else if ((is_yuv_planar(src_pix) &&

                src_pix_fmt != PIX_FMT_YUV444P &&

                src_pix_fmt != PIX_FMT_YUVJ444P)) {

        /* yuv444 is the normalized format */

        if (src_pix->color_type == FF_COLOR_YUV_JPEG)

            int_pix_fmt = PIX_FMT_YUVJ444P;

        else

            int_pix_fmt = PIX_FMT_YUV444P;

    } else if ((is_yuv_planar(dst_pix) &&

                dst_pix_fmt != PIX_FMT_YUV444P &&

                dst_pix_fmt != PIX_FMT_YUVJ444P)) {

        /* yuv444 is the normalized format */

        if (dst_pix->color_type == FF_COLOR_YUV_JPEG)

            int_pix_fmt = PIX_FMT_YUVJ444P;

        else

            int_pix_fmt = PIX_FMT_YUV444P;

    } else {

        /* the two formats are rgb or gray8 or yuv[j]444p */

        if (src_pix->is_alpha && dst_pix->is_alpha)

            int_pix_fmt = PIX_FMT_RGB32;

        else

            int_pix_fmt = PIX_FMT_RGB24;

    }

    if (src_pix_fmt == int_pix_fmt)

        return -1;

    if (avpicture_alloc(tmp, int_pix_fmt, dst_width, dst_height) < 0)

        return -1;

    ret = -1;

    if (img_convert(tmp, int_pix_fmt,

                    src, src_pix_fmt, src_width, src_height) < 0)

        goto fail1;

    if (img_convert(dst, dst_pix_fmt,

                    tmp, int_pix_fmt, dst_width, dst_height) < 0)

        goto fail1;

    ret = 0;

 fail1:

    avpicture_free(tmp);

    return ret;

}
