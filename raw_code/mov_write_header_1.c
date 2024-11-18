static int mov_write_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    MOVMuxContext *mov = s->priv_data;

    AVDictionaryEntry *t;

    int i, hint_track = 0;



    /* Default mode == MP4 */

    mov->mode = MODE_MP4;



    if (s->oformat != NULL) {

        if (!strcmp("3gp", s->oformat->name)) mov->mode = MODE_3GP;

        else if (!strcmp("3g2", s->oformat->name)) mov->mode = MODE_3GP|MODE_3G2;

        else if (!strcmp("mov", s->oformat->name)) mov->mode = MODE_MOV;

        else if (!strcmp("psp", s->oformat->name)) mov->mode = MODE_PSP;

        else if (!strcmp("ipod",s->oformat->name)) mov->mode = MODE_IPOD;

        else if (!strcmp("ismv",s->oformat->name)) mov->mode = MODE_ISM;

    }



    /* Set the FRAGMENT flag if any of the fragmentation methods are

     * enabled. */

    if (mov->max_fragment_duration || mov->max_fragment_size ||

        mov->mode == MODE_ISM ||

        mov->flags & (FF_MOV_FLAG_EMPTY_MOOV |

                      FF_MOV_FLAG_FRAG_KEYFRAME |

                      FF_MOV_FLAG_FRAG_CUSTOM))

        mov->flags |= FF_MOV_FLAG_FRAGMENT;



    /* faststart: moov at the beginning of the file, if supported */

    if (mov->flags & FF_MOV_FLAG_FASTSTART) {

        if ((mov->flags & FF_MOV_FLAG_FRAGMENT) ||

            (s->flags & AVFMT_FLAG_CUSTOM_IO)) {

            av_log(s, AV_LOG_WARNING, "The faststart flag is incompatible "

                   "with fragmentation and custom IO, disabling faststart\n");

            mov->flags &= ~FF_MOV_FLAG_FASTSTART;

        }

    }



    /* Non-seekable output is ok if using fragmentation. If ism_lookahead

     * is enabled, we don't support non-seekable output at all. */

    if (!s->pb->seekable &&

        (!(mov->flags & FF_MOV_FLAG_FRAGMENT) || mov->ism_lookahead)) {

        av_log(s, AV_LOG_ERROR, "muxer does not support non seekable output\n");

        return -1;

    }





    mov_write_ftyp_tag(pb,s);

    if (mov->mode == MODE_PSP) {

        if (s->nb_streams != 2) {

            av_log(s, AV_LOG_ERROR, "PSP mode need one video and one audio stream\n");

            return -1;

        }

        mov_write_uuidprof_tag(pb, s);

    }



    mov->nb_streams = s->nb_streams;

    if (mov->mode & (MODE_MP4|MODE_MOV|MODE_IPOD) && s->nb_chapters)

        mov->chapter_track = mov->nb_streams++;



    if (mov->flags & FF_MOV_FLAG_RTP_HINT) {

        /* Add hint tracks for each audio and video stream */

        hint_track = mov->nb_streams;

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO ||

                st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

                mov->nb_streams++;

            }

        }

    }



    // Reserve an extra stream for chapters for the case where chapters

    // are written in the trailer

    mov->tracks = av_mallocz((mov->nb_streams + 1) * sizeof(*mov->tracks));

    if (!mov->tracks)

        return AVERROR(ENOMEM);



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st= s->streams[i];

        MOVTrack *track= &mov->tracks[i];

        AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", NULL,0);



        track->enc = st->codec;

        track->language = ff_mov_iso639_to_lang(lang?lang->value:"und", mov->mode!=MODE_MOV);

        if (track->language < 0)

            track->language = 0;

        track->mode = mov->mode;

        track->tag  = mov_find_codec_tag(s, track);

        if (!track->tag) {

            av_log(s, AV_LOG_ERROR, "track %d: could not find tag, "

                   "codec not currently supported in container\n", i);

            goto error;

        }

        /* If hinting of this track is enabled by a later hint track,

         * this is updated. */

        track->hint_track = -1;

        track->start_dts  = AV_NOPTS_VALUE;

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if (track->tag == MKTAG('m','x','3','p') || track->tag == MKTAG('m','x','3','n') ||

                track->tag == MKTAG('m','x','4','p') || track->tag == MKTAG('m','x','4','n') ||

                track->tag == MKTAG('m','x','5','p') || track->tag == MKTAG('m','x','5','n')) {

                if (st->codec->width != 720 || (st->codec->height != 608 && st->codec->height != 512)) {

                    av_log(s, AV_LOG_ERROR, "D-10/IMX must use 720x608 or 720x512 video resolution\n");

                    goto error;

                }

                track->height = track->tag >> 24 == 'n' ? 486 : 576;

            }

            track->timescale = st->codec->time_base.den;

            if (track->mode == MODE_MOV && track->timescale > 100000)

                av_log(s, AV_LOG_WARNING,

                       "WARNING codec timebase is very high. If duration is too long,\n"

                       "file may not be playable by quicktime. Specify a shorter timebase\n"

                       "or choose different container.\n");

        } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            track->timescale = st->codec->sample_rate;

            /* set sample_size for PCM and ADPCM */

            if (av_get_bits_per_sample(st->codec->codec_id) ||

                st->codec->codec_id == AV_CODEC_ID_ILBC) {

                if (!st->codec->block_align) {

                    av_log(s, AV_LOG_ERROR, "track %d: codec block align is not set\n", i);

                    goto error;

                }

                track->sample_size = st->codec->block_align;

            }

            /* set audio_vbr for compressed audio */

            if (av_get_bits_per_sample(st->codec->codec_id) < 8) {

                track->audio_vbr = 1;

            }

            if (track->mode != MODE_MOV &&

                track->enc->codec_id == AV_CODEC_ID_MP3 && track->timescale < 16000) {

                av_log(s, AV_LOG_ERROR, "track %d: muxing mp3 at %dhz is not supported\n",

                       i, track->enc->sample_rate);

                goto error;

            }

        } else if (st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            track->timescale = st->codec->time_base.den;

        } else if (st->codec->codec_type == AVMEDIA_TYPE_DATA) {

            track->timescale = st->codec->time_base.den;

        }

        if (!track->height)

            track->height = st->codec->height;

        /* The ism specific timescale isn't mandatory, but is assumed by

         * some tools, such as mp4split. */

        if (mov->mode == MODE_ISM)

            track->timescale = 10000000;



        avpriv_set_pts_info(st, 64, 1, track->timescale);



        /* copy extradata if it exists */

        if (st->codec->extradata_size) {

            track->vos_len  = st->codec->extradata_size;

            track->vos_data = av_malloc(track->vos_len);

            memcpy(track->vos_data, st->codec->extradata, track->vos_len);

        }

    }



    enable_tracks(s);



    if (mov->mode == MODE_ISM) {

        /* If no fragmentation options have been set, set a default. */

        if (!(mov->flags & (FF_MOV_FLAG_FRAG_KEYFRAME |

                            FF_MOV_FLAG_FRAG_CUSTOM)) &&

            !mov->max_fragment_duration && !mov->max_fragment_size)

            mov->max_fragment_duration = 5000000;

        mov->flags |= FF_MOV_FLAG_EMPTY_MOOV | FF_MOV_FLAG_SEPARATE_MOOF;

    }



    if (!(mov->flags & FF_MOV_FLAG_FRAGMENT)) {

        if (mov->flags & FF_MOV_FLAG_FASTSTART)

            mov->reserved_moov_pos = avio_tell(pb);

        mov_write_mdat_tag(pb, mov);

    }



    if (t = av_dict_get(s->metadata, "creation_time", NULL, 0))

        mov->time = ff_iso8601_to_unix_time(t->value);

    if (mov->time)

        mov->time += 0x7C25B080; // 1970 based -> 1904 based



    if (mov->chapter_track)

        if (mov_create_chapter_track(s, mov->chapter_track) < 0)

            goto error;



    if (mov->flags & FF_MOV_FLAG_RTP_HINT) {

        /* Initialize the hint tracks for each audio and video stream */

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO ||

                st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

                ff_mov_init_hinting(s, hint_track, i);

                hint_track++;

            }

        }

    }



    avio_flush(pb);



    if (mov->flags & FF_MOV_FLAG_ISML)

        mov_write_isml_manifest(pb, mov);



    if (mov->flags & FF_MOV_FLAG_EMPTY_MOOV) {

        mov_write_moov_tag(pb, mov, s);

        mov->fragments++;

    }



    return 0;

 error:

    mov_free(s);

    return -1;

}
