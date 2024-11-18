static int idcin_read_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    IdcinDemuxContext *idcin = s->priv_data;

    AVStream *st;

    unsigned int width, height;

    unsigned int sample_rate, bytes_per_sample, channels;

    int ret;



    /* get the 5 header parameters */

    width = avio_rl32(pb);

    height = avio_rl32(pb);

    sample_rate = avio_rl32(pb);

    bytes_per_sample = avio_rl32(pb);

    channels = avio_rl32(pb);



    if (s->pb->eof_reached) {

        av_log(s, AV_LOG_ERROR, "incomplete header\n");

        return s->pb->error ? s->pb->error : AVERROR_EOF;

    }



    if (av_image_check_size(width, height, 0, s) < 0)

        return AVERROR_INVALIDDATA;

    if (sample_rate > 0) {

        if (sample_rate < 14 || sample_rate > INT_MAX) {

            av_log(s, AV_LOG_ERROR, "invalid sample rate: %u\n", sample_rate);

            return AVERROR_INVALIDDATA;

        }

        if (bytes_per_sample < 1 || bytes_per_sample > 2) {

            av_log(s, AV_LOG_ERROR, "invalid bytes per sample: %u\n",

                   bytes_per_sample);

            return AVERROR_INVALIDDATA;

        }

        if (channels < 1 || channels > 2) {

            av_log(s, AV_LOG_ERROR, "invalid channels: %u\n", channels);

            return AVERROR_INVALIDDATA;

        }

        idcin->audio_present = 1;

    } else {

        /* if sample rate is 0, assume no audio */

        idcin->audio_present = 0;

    }



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    avpriv_set_pts_info(st, 33, 1, IDCIN_FPS);

    st->start_time = 0;

    idcin->video_stream_index = st->index;

    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = AV_CODEC_ID_IDCIN;

    st->codec->codec_tag = 0;  /* no fourcc */

    st->codec->width = width;

    st->codec->height = height;



    /* load up the Huffman tables into extradata */

    st->codec->extradata_size = HUFFMAN_TABLE_SIZE;

    st->codec->extradata = av_malloc(HUFFMAN_TABLE_SIZE);

    ret = avio_read(pb, st->codec->extradata, HUFFMAN_TABLE_SIZE);

    if (ret < 0) {

        return ret;

    } else if (ret != HUFFMAN_TABLE_SIZE) {

        av_log(s, AV_LOG_ERROR, "incomplete header\n");

        return AVERROR(EIO);

    }



    if (idcin->audio_present) {

        idcin->audio_present = 1;

        st = avformat_new_stream(s, NULL);

        if (!st)

            return AVERROR(ENOMEM);

        avpriv_set_pts_info(st, 63, 1, sample_rate);

        st->start_time = 0;

        idcin->audio_stream_index = st->index;

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codec->codec_tag = 1;

        st->codec->channels = channels;

        st->codec->channel_layout = channels > 1 ? AV_CH_LAYOUT_STEREO :

                                                   AV_CH_LAYOUT_MONO;

        st->codec->sample_rate = sample_rate;

        st->codec->bits_per_coded_sample = bytes_per_sample * 8;

        st->codec->bit_rate = sample_rate * bytes_per_sample * 8 * channels;

        st->codec->block_align = idcin->block_align = bytes_per_sample * channels;

        if (bytes_per_sample == 1)

            st->codec->codec_id = AV_CODEC_ID_PCM_U8;

        else

            st->codec->codec_id = AV_CODEC_ID_PCM_S16LE;



        if (sample_rate % 14 != 0) {

            idcin->audio_chunk_size1 = (sample_rate / 14) *

            bytes_per_sample * channels;

            idcin->audio_chunk_size2 = (sample_rate / 14 + 1) *

                bytes_per_sample * channels;

        } else {

            idcin->audio_chunk_size1 = idcin->audio_chunk_size2 =

                (sample_rate / 14) * bytes_per_sample * channels;

        }

        idcin->current_audio_chunk = 0;

    }



    idcin->next_chunk_is_video = 1;

    idcin->first_pkt_pos = avio_tell(s->pb);



    return 0;

}
