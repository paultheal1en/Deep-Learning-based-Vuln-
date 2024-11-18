static int video_read_header(AVFormatContext *s,

                             AVFormatParameters *ap)

{

    AVStream *st;



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR_NOMEM;



    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = s->iformat->value;

    st->need_parsing = 1;



    /* for mjpeg, specify frame rate */

    /* for mpeg4 specify it too (most mpeg4 streams dont have the fixed_vop_rate set ...)*/

    if (ap && ap->time_base.num) {

        av_set_pts_info(st, 64, ap->time_base.num, ap->time_base.den);

    } else if ( st->codec->codec_id == CODEC_ID_MJPEG ||

                st->codec->codec_id == CODEC_ID_MPEG4 ||

                st->codec->codec_id == CODEC_ID_H264) {

        av_set_pts_info(st, 64, 1, 25);

    }



    return 0;

}
