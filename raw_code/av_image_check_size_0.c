int av_image_check_size(unsigned int w, unsigned int h, int log_offset, void *log_ctx)

{

    ImgUtils imgutils = { &imgutils_class, log_offset, log_ctx };



    if ((int)w>0 && (int)h>0 && (w+128)*(uint64_t)(h+128) < INT_MAX/8)

        return 0;



    av_log(&imgutils, AV_LOG_ERROR, "Picture size %ux%u is invalid\n", w, h);

    return AVERROR(EINVAL);

}
