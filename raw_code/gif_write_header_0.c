static int gif_write_header(AVFormatContext *s)

{

    GIFContext *gif = s->priv_data;

    AVIOContext *pb = s->pb;

    AVCodecContext *enc, *video_enc;

    int i, width, height, loop_count /*, rate*/;



/* XXX: do we reject audio streams or just ignore them ?

    if(s->nb_streams > 1)

        return -1;

*/

    gif->time = 0;

    gif->file_time = 0;



    video_enc = NULL;

    for(i=0;i<s->nb_streams;i++) {

        enc = s->streams[i]->codec;

        if (enc->codec_type != AVMEDIA_TYPE_AUDIO)

            video_enc = enc;

    }



    if (!video_enc) {

        av_free(gif);

        return -1;

    } else {

        width = video_enc->width;

        height = video_enc->height;

        loop_count = s->loop_output;

//        rate = video_enc->time_base.den;

    }



    if (video_enc->pix_fmt != PIX_FMT_RGB24) {

        av_log(s, AV_LOG_ERROR, "ERROR: gif only handles the rgb24 pixel format. Use -pix_fmt rgb24.\n");

        return AVERROR(EIO);

    }



    gif_image_write_header(pb, width, height, loop_count, NULL);



    avio_flush(s->pb);

    return 0;

}
