int ff_raw_read_header(AVFormatContext *s, AVFormatParameters *ap)

{

    AVStream *st;

    enum CodecID id;



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);



        id = s->iformat->value;

        if (id == CODEC_ID_RAWVIDEO) {

            st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

        } else {

            st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        }

        st->codec->codec_id = id;



        switch(st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO: {

            RawAudioDemuxerContext *s1 = s->priv_data;



#if FF_API_FORMAT_PARAMETERS

            if (ap->sample_rate)

                st->codec->sample_rate = ap->sample_rate;

            if (ap->channels)

                st->codec->channels    = ap->channels;

            else st->codec->channels   = 1;

#endif



            if (s1->sample_rate)

                st->codec->sample_rate = s1->sample_rate;

            if (s1->channels)

                st->codec->channels    = s1->channels;



            st->codec->bits_per_coded_sample = av_get_bits_per_sample(st->codec->codec_id);

            assert(st->codec->bits_per_coded_sample > 0);

            st->codec->block_align = st->codec->bits_per_coded_sample*st->codec->channels/8;

            av_set_pts_info(st, 64, 1, st->codec->sample_rate);

            break;

            }

        case AVMEDIA_TYPE_VIDEO: {

            FFRawVideoDemuxerContext *s1 = s->priv_data;

            int width = 0, height = 0, ret;

            enum PixelFormat pix_fmt;



            if(ap->time_base.num)

                av_set_pts_info(st, 64, ap->time_base.num, ap->time_base.den);

            else

                av_set_pts_info(st, 64, 1, 25);

            if (s1->video_size && (ret = av_parse_video_size(&width, &height, s1->video_size)) < 0) {

                av_log(s, AV_LOG_ERROR, "Couldn't parse video size.\n");

                goto fail;

            }

            if ((pix_fmt = av_get_pix_fmt(s1->pixel_format)) == PIX_FMT_NONE) {

                av_log(s, AV_LOG_ERROR, "No such pixel format: %s.\n", s1->pixel_format);

                ret = AVERROR(EINVAL);

                goto fail;

            }

#if FF_API_FORMAT_PARAMETERS

            if (ap->width > 0)

                width = ap->width;

            if (ap->height > 0)

                height = ap->height;

            if (ap->pix_fmt)

                pix_fmt = ap->pix_fmt;

#endif

            st->codec->width  = width;

            st->codec->height = height;

            st->codec->pix_fmt = pix_fmt;

fail:

            av_freep(&s1->video_size);

            av_freep(&s1->pixel_format);

            return ret;

            }

        default:

            return -1;

        }

    return 0;

}
