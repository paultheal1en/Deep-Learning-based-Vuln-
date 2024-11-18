static int gxf_write_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    GXFContext *gxf = s->priv_data;

    GXFStreamContext *vsc = NULL;

    uint8_t tracks[255] = {0};

    int i, media_info = 0;

    AVDictionaryEntry *tcr = av_dict_get(s->metadata, "timecode", NULL, 0);



    if (!pb->seekable) {

        av_log(s, AV_LOG_ERROR, "gxf muxer does not support streamed output, patch welcome\n");

        return -1;

    }



    gxf->flags |= 0x00080000; /* material is simple clip */

    for (i = 0; i < s->nb_streams; ++i) {

        AVStream *st = s->streams[i];

        GXFStreamContext *sc = av_mallocz(sizeof(*sc));

        if (!sc)

            return AVERROR(ENOMEM);

        st->priv_data = sc;



        sc->media_type = ff_codec_get_tag(gxf_media_types, st->codec->codec_id);

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (st->codec->codec_id != AV_CODEC_ID_PCM_S16LE) {

                av_log(s, AV_LOG_ERROR, "only 16 BIT PCM LE allowed for now\n");

                return -1;

            }

            if (st->codec->sample_rate != 48000) {

                av_log(s, AV_LOG_ERROR, "only 48000hz sampling rate is allowed\n");

                return -1;

            }

            if (st->codec->channels != 1) {

                av_log(s, AV_LOG_ERROR, "only mono tracks are allowed\n");

                return -1;

            }

            sc->track_type = 2;

            sc->sample_rate = st->codec->sample_rate;

            avpriv_set_pts_info(st, 64, 1, sc->sample_rate);

            sc->sample_size = 16;

            sc->frame_rate_index = -2;

            sc->lines_index = -2;

            sc->fields = -2;

            gxf->audio_tracks++;

            gxf->flags |= 0x04000000; /* audio is 16 bit pcm */

            media_info = 'A';

        } else if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if (i != 0) {

                av_log(s, AV_LOG_ERROR, "video stream must be the first track\n");

                return -1;

            }

            /* FIXME check from time_base ? */

            if (st->codec->height == 480 || st->codec->height == 512) { /* NTSC or NTSC+VBI */

                sc->frame_rate_index = 5;

                sc->sample_rate = 60;

                gxf->flags |= 0x00000080;

                gxf->time_base = (AVRational){ 1001, 60000 };

            } else if (st->codec->height == 576 || st->codec->height == 608) { /* PAL or PAL+VBI */

                sc->frame_rate_index = 6;

                sc->media_type++;

                sc->sample_rate = 50;

                gxf->flags |= 0x00000040;

                gxf->time_base = (AVRational){ 1, 50 };

            } else {

                av_log(s, AV_LOG_ERROR, "unsupported video resolution, "

                       "gxf muxer only accepts PAL or NTSC resolutions currently\n");

                return -1;

            }

            if (!tcr)

                tcr = av_dict_get(st->metadata, "timecode", NULL, 0);

            avpriv_set_pts_info(st, 64, gxf->time_base.num, gxf->time_base.den);

            if (gxf_find_lines_index(st) < 0)

                sc->lines_index = -1;

            sc->sample_size = st->codec->bit_rate;

            sc->fields = 2; /* interlaced */



            vsc = sc;



            switch (st->codec->codec_id) {

            case AV_CODEC_ID_MJPEG:

                sc->track_type = 1;

                gxf->flags |= 0x00004000;

                media_info = 'J';

                break;

            case AV_CODEC_ID_MPEG1VIDEO:

                sc->track_type = 9;

                gxf->mpeg_tracks++;

                media_info = 'L';

                break;

            case AV_CODEC_ID_MPEG2VIDEO:

                sc->first_gop_closed = -1;

                sc->track_type = 4;

                gxf->mpeg_tracks++;

                gxf->flags |= 0x00008000;

                media_info = 'M';

                break;

            case AV_CODEC_ID_DVVIDEO:

                if (st->codec->pix_fmt == AV_PIX_FMT_YUV422P) {

                    sc->media_type += 2;

                    sc->track_type = 6;

                    gxf->flags |= 0x00002000;

                    media_info = 'E';

                } else {

                    sc->track_type = 5;

                    gxf->flags |= 0x00001000;

                    media_info = 'D';

                }

                break;

            default:

                av_log(s, AV_LOG_ERROR, "video codec not supported\n");

                return -1;

            }

        }

        /* FIXME first 10 audio tracks are 0 to 9 next 22 are A to V */

        sc->media_info = media_info<<8 | ('0'+tracks[media_info]++);

        sc->order = s->nb_streams - st->index;

    }



    if (ff_audio_interleave_init(s, GXF_samples_per_frame, (AVRational){ 1, 48000 }) < 0)

        return -1;



    if (tcr)

        gxf_init_timecode(s, &gxf->tc, tcr->value, vsc->fields);



    gxf_init_timecode_track(&gxf->timecode_track, vsc);

    gxf->flags |= 0x200000; // time code track is non-drop frame



    gxf_write_map_packet(s, 0);

    gxf_write_flt_packet(s);

    gxf_write_umf_packet(s);



    gxf->packet_count = 3;



    avio_flush(pb);

    return 0;

}
