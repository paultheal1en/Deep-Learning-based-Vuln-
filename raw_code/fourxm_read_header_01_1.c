static int fourxm_read_header(AVFormatContext *s,

                              AVFormatParameters *ap)

{

    AVIOContext *pb = s->pb;

    unsigned int fourcc_tag;

    unsigned int size;

    int header_size;

    FourxmDemuxContext *fourxm = s->priv_data;

    unsigned char *header;

    int i, ret;

    AVStream *st;



    fourxm->track_count = 0;

    fourxm->tracks = NULL;

    fourxm->fps = 1.0;



    /* skip the first 3 32-bit numbers */

    avio_skip(pb, 12);



    /* check for LIST-HEAD */

    GET_LIST_HEADER();

    header_size = size - 4;

    if (fourcc_tag != HEAD_TAG || header_size < 0)

        return AVERROR_INVALIDDATA;



    /* allocate space for the header and load the whole thing */

    header = av_malloc(header_size);

    if (!header)

        return AVERROR(ENOMEM);

    if (avio_read(pb, header, header_size) != header_size){

        av_free(header);

        return AVERROR(EIO);

    }



    /* take the lazy approach and search for any and all vtrk and strk chunks */

    for (i = 0; i < header_size - 8; i++) {

        fourcc_tag = AV_RL32(&header[i]);

        size = AV_RL32(&header[i + 4]);



        if (fourcc_tag == std__TAG) {

            fourxm->fps = av_int2flt(AV_RL32(&header[i + 12]));

        } else if (fourcc_tag == vtrk_TAG) {

            /* check that there is enough data */

            if (size != vtrk_SIZE) {

                ret= AVERROR_INVALIDDATA;

                goto fail;

            }

            fourxm->width  = AV_RL32(&header[i + 36]);

            fourxm->height = AV_RL32(&header[i + 40]);



            /* allocate a new AVStream */

            st = av_new_stream(s, 0);

            if (!st){

                ret= AVERROR(ENOMEM);

                goto fail;

            }

            av_set_pts_info(st, 60, 1, fourxm->fps);



            fourxm->video_stream_index = st->index;



            st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            st->codec->codec_id = CODEC_ID_4XM;

            st->codec->extradata_size = 4;

            st->codec->extradata = av_malloc(4);

            AV_WL32(st->codec->extradata, AV_RL32(&header[i + 16]));

            st->codec->width  = fourxm->width;

            st->codec->height = fourxm->height;



            i += 8 + size;

        } else if (fourcc_tag == strk_TAG) {

            int current_track;

            /* check that there is enough data */

            if (size != strk_SIZE) {

                ret= AVERROR_INVALIDDATA;

                goto fail;

            }

            current_track = AV_RL32(&header[i + 8]);

            if((unsigned)current_track >= UINT_MAX / sizeof(AudioTrack) - 1){

                av_log(s, AV_LOG_ERROR, "current_track too large\n");

                ret= -1;

                goto fail;

            }

            if (current_track + 1 > fourxm->track_count) {

                fourxm->track_count = current_track + 1;

                fourxm->tracks = av_realloc(fourxm->tracks,

                    fourxm->track_count * sizeof(AudioTrack));

                if (!fourxm->tracks) {

                    ret=  AVERROR(ENOMEM);

                    goto fail;

                }

            }

            fourxm->tracks[current_track].adpcm       = AV_RL32(&header[i + 12]);

            fourxm->tracks[current_track].channels    = AV_RL32(&header[i + 36]);

            fourxm->tracks[current_track].sample_rate = AV_RL32(&header[i + 40]);

            fourxm->tracks[current_track].bits        = AV_RL32(&header[i + 44]);

            fourxm->tracks[current_track].audio_pts   = 0;

            if(   fourxm->tracks[current_track].channels    <= 0

               || fourxm->tracks[current_track].sample_rate <= 0

               || fourxm->tracks[current_track].bits        <  0){

                av_log(s, AV_LOG_ERROR, "audio header invalid\n");

                ret= -1;

                goto fail;

            }

            i += 8 + size;



            /* allocate a new AVStream */

            st = av_new_stream(s, current_track);

            if (!st){

                ret= AVERROR(ENOMEM);

                goto fail;

            }



            av_set_pts_info(st, 60, 1, fourxm->tracks[current_track].sample_rate);



            fourxm->tracks[current_track].stream_index = st->index;



            st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            st->codec->codec_tag = 0;

            st->codec->channels              = fourxm->tracks[current_track].channels;

            st->codec->sample_rate           = fourxm->tracks[current_track].sample_rate;

            st->codec->bits_per_coded_sample = fourxm->tracks[current_track].bits;

            st->codec->bit_rate              = st->codec->channels * st->codec->sample_rate *

                st->codec->bits_per_coded_sample;

            st->codec->block_align = st->codec->channels * st->codec->bits_per_coded_sample;

            if (fourxm->tracks[current_track].adpcm){

                st->codec->codec_id = CODEC_ID_ADPCM_4XM;

            }else if (st->codec->bits_per_coded_sample == 8){

                st->codec->codec_id = CODEC_ID_PCM_U8;

            }else

                st->codec->codec_id = CODEC_ID_PCM_S16LE;

        }

    }



    /* skip over the LIST-MOVI chunk (which is where the stream should be */

    GET_LIST_HEADER();

    if (fourcc_tag != MOVI_TAG){

        ret= AVERROR_INVALIDDATA;

        goto fail;

    }



    av_free(header);

    /* initialize context members */

    fourxm->video_pts = -1;  /* first frame will push to 0 */



    return 0;

fail:

    av_freep(&fourxm->tracks);

    av_free(header);

    return ret;

}
