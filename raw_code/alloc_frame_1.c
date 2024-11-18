static AVFrame *alloc_frame(enum AVPixelFormat pixfmt, int w, int h)

{

    AVFrame *frame = av_frame_alloc();

    if (!frame)

        return NULL;



    frame->format = pixfmt;

    frame->width  = w;

    frame->height = h;



    if (av_frame_get_buffer(frame, 32) < 0)

        return NULL;



    return frame;

}
