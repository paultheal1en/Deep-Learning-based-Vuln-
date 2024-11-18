AVFrame *avcodec_alloc_frame(void)

{

    AVFrame *frame = av_mallocz(sizeof(AVFrame));



    if (frame == NULL)

        return NULL;



FF_DISABLE_DEPRECATION_WARNINGS

    avcodec_get_frame_defaults(frame);

FF_ENABLE_DEPRECATION_WARNINGS



    return frame;

}
