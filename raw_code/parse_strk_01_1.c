static int parse_strk(AVFormatContext *s,

                      FourxmDemuxContext *fourxm, uint8_t *buf, int size)

{

    AVStream *st;

    int track;

    /* check that there is enough data */

    if (size != strk_SIZE)

        return AVERROR_INVALIDDATA;



    track = AV_RL32(buf + 8);

    if (track + 1 > fourxm->track_count) {

        if (av_reallocp_array(&fourxm->tracks, track + 1, sizeof(AudioTrack)))

            return AVERROR(ENOMEM);

        memset(&fourxm->tracks[fourxm->track_count], 0,

               sizeof(AudioTrack) * (track + 1 - fourxm->track_count));

        fourxm->track_count = track + 1;

    }

    fourxm->tracks[track].adpcm       = AV_RL32(buf + 12);

    fourxm->tracks[track].channels    = AV_RL32(buf + 36);

    fourxm->tracks[track].sample_rate = AV_RL32(buf + 40);

    fourxm->tracks[track].bits        = AV_RL32(buf + 44);

    fourxm->tracks[track].audio_pts   = 0;



    if (fourxm->tracks[track].channels    <= 0 ||

        fourxm->tracks[track].sample_rate <= 0 ||

        fourxm->tracks[track].bits        < 0) {

        av_log(s, AV_LOG_ERROR, "audio header invalid\n");

        return AVERROR_INVALIDDATA;

    }

    /* allocate a new AVStream */

    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    st->id = track;

    avpriv_set_pts_info(st, 60, 1, fourxm->tracks[track].sample_rate);



    fourxm->tracks[track].stream_index = st->index;



    st->codec->codec_type            = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_tag             = 0;

    st->codec->channels              = fourxm->tracks[track].channels;

    st->codec->sample_rate           = fourxm->tracks[track].sample_rate;

    st->codec->bits_per_coded_sample = fourxm->tracks[track].bits;

    st->codec->bit_rate              = st->codec->channels *

                                       st->codec->sample_rate *

                                       st->codec->bits_per_coded_sample;

    st->codec->block_align           = st->codec->channels *

                                       st->codec->bits_per_coded_sample;



    if (fourxm->tracks[track].adpcm){

        st->codec->codec_id = AV_CODEC_ID_ADPCM_4XM;

    } else if (st->codec->bits_per_coded_sample == 8) {

        st->codec->codec_id = AV_CODEC_ID_PCM_U8;

    } else

        st->codec->codec_id = AV_CODEC_ID_PCM_S16LE;



    return 0;

}
