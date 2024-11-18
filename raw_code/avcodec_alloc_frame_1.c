AVFrame *avcodec_alloc_frame(void)

{

    AVFrame *frame = av_malloc(sizeof(AVFrame));



    if (frame == NULL)

        return NULL;



    avcodec_get_frame_defaults(frame);



    return frame;

}
