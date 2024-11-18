static av_cold int mpeg_mux_init(AVFormatContext *ctx)

{

    MpegMuxContext *s = ctx->priv_data;

    int bitrate, i, mpa_id, mpv_id, h264_id, mps_id, ac3_id, dts_id, lpcm_id, j;

    AVStream *st;

    StreamInfo *stream;

    int audio_bitrate;

    int video_bitrate;



    s->packet_number = 0;

    s->is_vcd   =  (CONFIG_MPEG1VCD_MUXER  && ctx->oformat == &ff_mpeg1vcd_muxer);

    s->is_svcd  =  (CONFIG_MPEG2SVCD_MUXER && ctx->oformat == &ff_mpeg2svcd_muxer);

    s->is_mpeg2 = ((CONFIG_MPEG2VOB_MUXER  && ctx->oformat == &ff_mpeg2vob_muxer) ||

                   (CONFIG_MPEG2DVD_MUXER  && ctx->oformat == &ff_mpeg2dvd_muxer) ||

                   (CONFIG_MPEG2SVCD_MUXER && ctx->oformat == &ff_mpeg2svcd_muxer));

    s->is_dvd   =  (CONFIG_MPEG2DVD_MUXER  && ctx->oformat == &ff_mpeg2dvd_muxer);



    if (ctx->packet_size) {

        if (ctx->packet_size < 20 || ctx->packet_size > (1 << 23) + 10) {

            av_log(ctx, AV_LOG_ERROR, "Invalid packet size %d\n",

                   ctx->packet_size);

            goto fail;

        }

        s->packet_size = ctx->packet_size;

    } else

        s->packet_size = 2048;

    if (ctx->max_delay < 0)     /* Not set by the caller */

        ctx->max_delay = AV_TIME_BASE*7/10;



    s->vcd_padding_bytes_written = 0;

    s->vcd_padding_bitrate_num   = 0;



    s->audio_bound = 0;

    s->video_bound = 0;



    mpa_id  = AUDIO_ID;

    ac3_id  = AC3_ID;

    dts_id  = DTS_ID;

    mpv_id  = VIDEO_ID;

    h264_id = H264_ID;

    mps_id  = SUB_ID;

    lpcm_id = LPCM_ID;



    for (i = 0; i < ctx->nb_streams; i++) {

        st     = ctx->streams[i];

        stream = av_mallocz(sizeof(StreamInfo));

        if (!stream)

            goto fail;

        st->priv_data = stream;



        avpriv_set_pts_info(st, 64, 1, 90000);



        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if (!s->is_mpeg2 &&

                (st->codec->codec_id == AV_CODEC_ID_AC3 ||

                 st->codec->codec_id == AV_CODEC_ID_DTS ||

                 st->codec->codec_id == AV_CODEC_ID_PCM_S16BE))

                 av_log(ctx, AV_LOG_WARNING,

                        "%s in MPEG-1 system streams is not widely supported, "

                        "consider using the vob or the dvd muxer "

                        "to force a MPEG-2 program stream.\n",

                        avcodec_get_name(st->codec->codec_id));

            if (st->codec->codec_id == AV_CODEC_ID_AC3) {

                stream->id = ac3_id++;

            } else if (st->codec->codec_id == AV_CODEC_ID_DTS) {

                stream->id = dts_id++;

            } else if (st->codec->codec_id == AV_CODEC_ID_PCM_S16BE) {

                stream->id = lpcm_id++;

                for (j = 0; j < 4; j++) {

                    if (lpcm_freq_tab[j] == st->codec->sample_rate)

                        break;

                }

                if (j == 4)

                    goto fail;

                if (st->codec->channels > 8)

                    return -1;

                stream->lpcm_header[0] = 0x0c;

                stream->lpcm_header[1] = (st->codec->channels - 1) | (j << 4);

                stream->lpcm_header[2] = 0x80;

                stream->lpcm_align     = st->codec->channels * 2;

            } else {

                stream->id = mpa_id++;

            }



            /* This value HAS to be used for VCD (see VCD standard, p. IV-7).

             * Right now it is also used for everything else. */

            stream->max_buffer_size = 4 * 1024;

            s->audio_bound++;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (st->codec->codec_id == AV_CODEC_ID_H264)

                stream->id = h264_id++;

            else

                stream->id = mpv_id++;

            if (st->codec->rc_buffer_size)

                stream->max_buffer_size = 6 * 1024 + st->codec->rc_buffer_size / 8;

            else {

                av_log(ctx, AV_LOG_WARNING,

                       "VBV buffer size not set, using default size of 130KB\n"

                       "If you want the mpeg file to be compliant to some specification\n"

                       "Like DVD, VCD or others, make sure you set the correct buffer size\n");

                // FIXME: this is probably too small as default

                stream->max_buffer_size = 230 * 1024;

            }

            if (stream->max_buffer_size > 1024 * 8191) {

                av_log(ctx, AV_LOG_WARNING, "buffer size %d, too large\n", stream->max_buffer_size);

                stream->max_buffer_size = 1024 * 8191;

            }

            s->video_bound++;

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            stream->id              = mps_id++;

            stream->max_buffer_size = 16 * 1024;

            break;

        default:

            return -1;

        }

        stream->fifo = av_fifo_alloc(16);

        if (!stream->fifo)

            goto fail;

    }

    bitrate       = 0;

    audio_bitrate = 0;

    video_bitrate = 0;

    for (i = 0; i < ctx->nb_streams; i++) {

        int codec_rate;

        st     = ctx->streams[i];

        stream = (StreamInfo *)st->priv_data;



        if (st->codec->rc_max_rate ||

            st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            codec_rate = st->codec->rc_max_rate;

        else

            codec_rate = st->codec->bit_rate;



        if (!codec_rate)

            codec_rate = (1 << 21) * 8 * 50 / ctx->nb_streams;



        bitrate += codec_rate;



        if ((stream->id & 0xe0) == AUDIO_ID)

            audio_bitrate += codec_rate;

        else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            video_bitrate += codec_rate;

    }



    if (s->user_mux_rate) {

        s->mux_rate = (s->user_mux_rate + (8 * 50) - 1) / (8 * 50);

    } else {

        /* we increase slightly the bitrate to take into account the

         * headers. XXX: compute it exactly */

        bitrate    += bitrate / 20;

        bitrate    += 10000;

        s->mux_rate = (bitrate + (8 * 50) - 1) / (8 * 50);

        if (s->mux_rate >= (1<<22)) {

            av_log(ctx, AV_LOG_WARNING, "mux rate %d is too large\n", s->mux_rate);

            s->mux_rate = (1<<22) - 1;

        }

    }



    if (s->is_vcd) {

        int64_t overhead_rate;



        /* The VCD standard mandates that the mux_rate field is 3528

         * (see standard p. IV-6).

         * The value is actually "wrong", i.e. if you calculate

         * it using the normal formula and the 75 sectors per second transfer

         * rate you get a different value because the real pack size is 2324,

         * not 2352. But the standard explicitly specifies that the mux_rate

         * field in the header must have this value. */

        // s->mux_rate = 2352 * 75 / 50;    /* = 3528 */



        /* The VCD standard states that the muxed stream must be

         * exactly 75 packs / second (the data rate of a single speed cdrom).

         * Since the video bitrate (probably 1150000 bits/sec) will be below

         * the theoretical maximum we have to add some padding packets

         * to make up for the lower data rate.

         * (cf. VCD standard p. IV-6 ) */



        /* Add the header overhead to the data rate.

         * 2279 data bytes per audio pack, 2294 data bytes per video pack */

        overhead_rate  = audio_bitrate * 2294LL * (2324 - 2279);

        overhead_rate += video_bitrate * 2279LL * (2324 - 2294);



        /* Add padding so that the full bitrate is 2324*75 bytes/sec */

        s->vcd_padding_bitrate_num = (2324LL * 75 * 8 - bitrate) * 2279 * 2294 - overhead_rate;

#define VCD_PADDING_BITRATE_DEN (2279 * 2294)

    }



    if (s->is_vcd || s->is_mpeg2)

        /* every packet */

        s->pack_header_freq = 1;

    else

        /* every 2 seconds */

        s->pack_header_freq = 2 * bitrate / s->packet_size / 8;



    /* the above seems to make pack_header_freq zero sometimes */

    if (s->pack_header_freq == 0)

        s->pack_header_freq = 1;



    if (s->is_mpeg2)

        /* every 200 packets. Need to look at the spec.  */

        s->system_header_freq = s->pack_header_freq * 40;

    else if (s->is_vcd)

        /* the standard mandates that there are only two system headers

         * in the whole file: one in the first packet of each stream.

         * (see standard p. IV-7 and IV-8) */

        s->system_header_freq = 0x7fffffff;

    else

        s->system_header_freq = s->pack_header_freq * 5;



    for (i = 0; i < ctx->nb_streams; i++) {

        stream                = ctx->streams[i]->priv_data;

        stream->packet_number = 0;

    }

    s->system_header_size = get_system_header_size(ctx);

    s->last_scr           = AV_NOPTS_VALUE;

    return 0;



fail:

    for (i = 0; i < ctx->nb_streams; i++)

        av_freep(&ctx->streams[i]->priv_data);

    return AVERROR(ENOMEM);

}
