static int raw_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    AVStream *st;

    int id;



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR_NOMEM;

    if (ap) {

        id = s->iformat->value;

        if (id == CODEC_ID_RAWVIDEO) {

            st->codec->codec_type = CODEC_TYPE_VIDEO;

        } else {

            st->codec->codec_type = CODEC_TYPE_AUDIO;

        }

        st->codec->codec_id = id;



        switch(st->codec->codec_type) {

        case CODEC_TYPE_AUDIO:

            st->codec->sample_rate = ap->sample_rate;

            st->codec->channels = ap->channels;

            av_set_pts_info(st, 64, 1, st->codec->sample_rate);

            break;

        case CODEC_TYPE_VIDEO:

            av_set_pts_info(st, 64, ap->time_base.num, ap->time_base.den);

            st->codec->width = ap->width;

            st->codec->height = ap->height;

            st->codec->pix_fmt = ap->pix_fmt;

            if(st->codec->pix_fmt == PIX_FMT_NONE)

                st->codec->pix_fmt= PIX_FMT_YUV420P;

            break;

        default:

            return -1;

        }

    } else {

        return -1;

    }

    return 0;

}
