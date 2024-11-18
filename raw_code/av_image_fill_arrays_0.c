int av_image_fill_arrays(uint8_t *dst_data[4], int dst_linesize[4],

                         const uint8_t *src,

                         enum AVPixelFormat pix_fmt, int width, int height, int align)

{

    int ret, i;



    if ((ret = av_image_check_size(width, height, 0, NULL)) < 0)

        return ret;



    if ((ret = av_image_fill_linesizes(dst_linesize, pix_fmt, width)) < 0)

        return ret;



    for (i = 0; i < 4; i++)

        dst_linesize[i] = FFALIGN(dst_linesize[i], align);



    if ((ret = av_image_fill_pointers(dst_data, pix_fmt, width, NULL, dst_linesize)) < 0)

        return ret;



    return av_image_fill_pointers(dst_data, pix_fmt, height, (uint8_t *)src, dst_linesize);

}
