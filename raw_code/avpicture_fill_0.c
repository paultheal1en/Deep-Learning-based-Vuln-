int avpicture_fill(AVPicture *picture, uint8_t *ptr,

                   enum AVPixelFormat pix_fmt, int width, int height)

{

    int ret;



    if ((ret = av_image_check_size(width, height, 0, NULL)) < 0)

        return ret;



    if ((ret = av_image_fill_linesizes(picture->linesize, pix_fmt, width)) < 0)

        return ret;



    return av_image_fill_pointers(picture->data, pix_fmt,

                                  height, ptr, picture->linesize);

}
