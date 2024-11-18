static int mov_write_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    MOVMuxContext *mov = s->priv_data;

    AVDictionaryEntry *t, *global_tcr = av_dict_get(s->metadata, "timecode", NULL, 0);

    int i, ret, hint_track = 0, tmcd_track = 0;



    mov->fc = s;



    /* Default mode == MP4 */

    mov->mode = MODE_MP4;



    if (s->oformat) {

        if (!strcmp("3gp", s->oformat->name)) mov->mode = MODE_3GP;

        else if (!strcmp("3g2", s->oformat->name)) mov->mode = MODE_3GP|MODE_3G2;

        else if (!strcmp("mov", s->oformat->name)) mov->mode = MODE_MOV;

        else if (!strcmp("psp", s->oformat->name)) mov->mode = MODE_PSP;

        else if (!strcmp("ipod",s->oformat->name)) mov->mode = MODE_IPOD;

        else if (!strcmp("ismv",s->oformat->name)) mov->mode = MODE_ISM;

        else if (!strcmp("f4v", s->oformat->name)) mov->mode = MODE_F4V;

    }



    if (s->flags & AVFMT_FLAG_BITEXACT)

        mov->exact = 1;



    if (mov->flags & FF_MOV_FLAG_DELAY_MOOV)

        mov->flags |= FF_MOV_FLAG_EMPTY_MOOV;



    /* Set the FRAGMENT flag if any of the fragmentation methods are

     * enabled. */

    if (mov->max_fragment_duration || mov->max_fragment_size ||

        mov->flags & (FF_MOV_FLAG_EMPTY_MOOV |

                      FF_MOV_FLAG_FRAG_KEYFRAME |

                      FF_MOV_FLAG_FRAG_CUSTOM))

        mov->flags |= FF_MOV_FLAG_FRAGMENT;



    /* Set other implicit flags immediately */

    if (mov->mode == MODE_ISM)

        mov->flags |= FF_MOV_FLAG_EMPTY_MOOV | FF_MOV_FLAG_SEPARATE_MOOF |

                      FF_MOV_FLAG_FRAGMENT;

    if (mov->flags & FF_MOV_FLAG_DASH)

        mov->flags |= FF_MOV_FLAG_FRAGMENT | FF_MOV_FLAG_EMPTY_MOOV |

                      FF_MOV_FLAG_DEFAULT_BASE_MOOF;



    if (mov->flags & FF_MOV_FLAG_FASTSTART) {

        mov->reserved_moov_size = -1;

    }



    if (mov->use_editlist < 0) {

        mov->use_editlist = 1;

        if (mov->flags & FF_MOV_FLAG_FRAGMENT &&

            !(mov->flags & FF_MOV_FLAG_DELAY_MOOV)) {

            // If we can avoid needing an edit list by shifting the

            // tracks, prefer that over (trying to) write edit lists

            // in fragmented output.

            if (s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_AUTO ||

                s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_MAKE_ZERO)

                mov->use_editlist = 0;

        }

    }

    if (mov->flags & FF_MOV_FLAG_EMPTY_MOOV &&

        !(mov->flags & FF_MOV_FLAG_DELAY_MOOV) && mov->use_editlist)

        av_log(s, AV_LOG_WARNING, "No meaningful edit list will be written when using empty_moov without delay_moov\n");



    if (!mov->use_editlist && s->avoid_negative_ts == AVFMT_AVOID_NEG_TS_AUTO)

        s->avoid_negative_ts = AVFMT_AVOID_NEG_TS_MAKE_ZERO;



    /* Clear the omit_tfhd_offset flag if default_base_moof is set;

     * if the latter is set that's enough and omit_tfhd_offset doesn't

     * add anything extra on top of that. */

    if (mov->flags & FF_MOV_FLAG_OMIT_TFHD_OFFSET &&

        mov->flags & FF_MOV_FLAG_DEFAULT_BASE_MOOF)

        mov->flags &= ~FF_MOV_FLAG_OMIT_TFHD_OFFSET;



    if (mov->frag_interleave &&

        mov->flags & (FF_MOV_FLAG_OMIT_TFHD_OFFSET | FF_MOV_FLAG_SEPARATE_MOOF)) {

        av_log(s, AV_LOG_ERROR,

               "Sample interleaving in fragments is mutually exclusive with "

               "omit_tfhd_offset and separate_moof\n");

        return AVERROR(EINVAL);

    }



    /* Non-seekable output is ok if using fragmentation. If ism_lookahead

     * is enabled, we don't support non-seekable output at all. */

    if (!s->pb->seekable &&

        (!(mov->flags & FF_MOV_FLAG_FRAGMENT) || mov->ism_lookahead)) {

        av_log(s, AV_LOG_ERROR, "muxer does not support non seekable output\n");

        return AVERROR(EINVAL);

    }



    if (!(mov->flags & FF_MOV_FLAG_DELAY_MOOV)) {

        if ((ret = mov_write_identification(pb, s)) < 0)

            return ret;

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



    if (mov->mode == MODE_MOV) {

        tmcd_track = mov->nb_streams;



        /* +1 tmcd track for each video stream with a timecode */

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO &&

                (global_tcr || av_dict_get(st->metadata, "timecode", NULL, 0)))

                mov->nb_meta_tmcd++;

        }



        /* check if there is already a tmcd track to remux */

        if (mov->nb_meta_tmcd) {

            for (i = 0; i < s->nb_streams; i++) {

                AVStream *st = s->streams[i];

                if (st->codec->codec_tag == MKTAG('t','m','c','d')) {

                    av_log(s, AV_LOG_WARNING, "You requested a copy of the original timecode track "

                           "so timecode metadata are now ignored\n");

                    mov->nb_meta_tmcd = 0;

                }

            }

        }



        mov->nb_streams += mov->nb_meta_tmcd;

    }



    // Reserve an extra stream for chapters for the case where chapters

    // are written in the trailer

    mov->tracks = av_mallocz_array((mov->nb_streams + 1), sizeof(*mov->tracks));

    if (!mov->tracks)

        return AVERROR(ENOMEM);



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st= s->streams[i];

        MOVTrack *track= &mov->tracks[i];

        AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", NULL,0);



        track->st  = st;

        track->enc = st->codec;

        track->language = ff_mov_iso639_to_lang(lang?lang->value:"und", mov->mode!=MODE_MOV);

        if (track->language < 0)

            track->language = 0;

        track->mode = mov->mode;

        track->tag  = mov_find_codec_tag(s, track);

        if (!track->tag) {

            av_log(s, AV_LOG_ERROR, "Could not find tag for codec %s in stream #%d, "

                   "codec not currently supported in container\n",

                   avcodec_get_name(st->codec->codec_id), i);

            ret = AVERROR(EINVAL);

            goto error;

        }

        /* If hinting of this track is enabled by a later hint track,

         * this is updated. */

        track->hint_track = -1;

        track->start_dts  = AV_NOPTS_VALUE;

        track->start_cts  = AV_NOPTS_VALUE;

        track->end_pts    = AV_NOPTS_VALUE;

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if (track->tag == MKTAG('m','x','3','p') || track->tag == MKTAG('m','x','3','n') ||

                track->tag == MKTAG('m','x','4','p') || track->tag == MKTAG('m','x','4','n') ||

                track->tag == MKTAG('m','x','5','p') || track->tag == MKTAG('m','x','5','n')) {

                if (st->codec->width != 720 || (st->codec->height != 608 && st->codec->height != 512)) {

                    av_log(s, AV_LOG_ERROR, "D-10/IMX must use 720x608 or 720x512 video resolution\n");

                    ret = AVERROR(EINVAL);

                    goto error;

                }

                track->height = track->tag >> 24 == 'n' ? 486 : 576;

            }

            if (mov->video_track_timescale) {

                track->timescale = mov->video_track_timescale;

            } else {

                track->timescale = st->time_base.den;

                while(track->timescale < 10000)

                    track->timescale *= 2;

            }

            if (st->codec->width > 65535 || st->codec->height > 65535) {

                av_log(s, AV_LOG_ERROR, "Resolution %dx%d too large for mov/mp4\n", st->codec->width, st->codec->height);

                ret = AVERROR(EINVAL);

                goto error;

            }

            if (track->mode == MODE_MOV && track->timescale > 100000)

                av_log(s, AV_LOG_WARNING,

                       "WARNING codec timebase is very high. If duration is too long,\n"

                       "file may not be playable by quicktime. Specify a shorter timebase\n"

                       "or choose different container.\n");

        } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            track->timescale = st->codec->sample_rate;

            if (!st->codec->frame_size && !av_get_bits_per_sample(st->codec->codec_id)) {

                av_log(s, AV_LOG_WARNING, "track %d: codec frame size is not set\n", i);

                track->audio_vbr = 1;

            }else if (st->codec->codec_id == AV_CODEC_ID_ADPCM_MS ||

                     st->codec->codec_id == AV_CODEC_ID_ADPCM_IMA_WAV ||

                     st->codec->codec_id == AV_CODEC_ID_ILBC){

                if (!st->codec->block_align) {

                    av_log(s, AV_LOG_ERROR, "track %d: codec block align is not set for adpcm\n", i);

                    ret = AVERROR(EINVAL);

                    goto error;

                }

                track->sample_size = st->codec->block_align;

            }else if (st->codec->frame_size > 1){ /* assume compressed audio */

                track->audio_vbr = 1;

            }else{

                track->sample_size = (av_get_bits_per_sample(st->codec->codec_id) >> 3) * st->codec->channels;

            }

            if (st->codec->codec_id == AV_CODEC_ID_ILBC ||

                st->codec->codec_id == AV_CODEC_ID_ADPCM_IMA_QT) {

                track->audio_vbr = 1;

            }

            if (track->mode != MODE_MOV &&

                track->enc->codec_id == AV_CODEC_ID_MP3 && track->timescale < 16000) {

                if (track->enc->strict_std_compliance >= FF_COMPLIANCE_NORMAL) {

                    av_log(s, AV_LOG_ERROR, "track %d: muxing mp3 at %dhz is not standard, to mux anyway set strict to -1\n",

                        i, track->enc->sample_rate);

                    ret = AVERROR(EINVAL);

                    goto error;

                } else {

                    av_log(s, AV_LOG_WARNING, "track %d: muxing mp3 at %dhz is not standard in MP4\n",

                           i, track->enc->sample_rate);

                }

            }

        } else if (st->codec->codec_type == AVMEDIA_TYPE_SUBTITLE) {

            track->timescale = st->time_base.den;

        } else if (st->codec->codec_type == AVMEDIA_TYPE_DATA) {

            track->timescale = st->time_base.den;

        } else {

            track->timescale = MOV_TIMESCALE;

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

            if (st->codec->codec_id == AV_CODEC_ID_DVD_SUBTITLE)

                mov_create_dvd_sub_decoder_specific_info(track, st);

            else if (!TAG_IS_AVCI(track->tag) && st->codec->codec_id != AV_CODEC_ID_DNXHD) {

                track->vos_len  = st->codec->extradata_size;

                track->vos_data = av_malloc(track->vos_len);

                if (!track->vos_data) {

                    ret = AVERROR(ENOMEM);

                    goto error;

                }

                memcpy(track->vos_data, st->codec->extradata, track->vos_len);

            }

        }

    }



    for (i = 0; i < s->nb_streams; i++) {

        int j;

        AVStream *st= s->streams[i];

        MOVTrack *track= &mov->tracks[i];



        if (st->codec->codec_type != AVMEDIA_TYPE_AUDIO ||

            track->enc->channel_layout != AV_CH_LAYOUT_MONO)

            continue;



        for (j = 0; j < s->nb_streams; j++) {

            AVStream *stj= s->streams[j];

            MOVTrack *trackj= &mov->tracks[j];

            if (j == i)

                continue;



            if (stj->codec->codec_type != AVMEDIA_TYPE_AUDIO ||

                trackj->enc->channel_layout != AV_CH_LAYOUT_MONO ||

                trackj->language != track->language ||

                trackj->tag != track->tag

            )

                continue;

            track->multichannel_as_mono++;

        }

    }



    enable_tracks(s);





    if (mov->reserved_moov_size){

        mov->reserved_moov_pos= avio_tell(pb);

        if (mov->reserved_moov_size > 0)

            avio_skip(pb, mov->reserved_moov_size);

    }



    if (mov->flags & FF_MOV_FLAG_FRAGMENT) {

        /* If no fragmentation options have been set, set a default. */

        if (!(mov->flags & (FF_MOV_FLAG_FRAG_KEYFRAME |

                            FF_MOV_FLAG_FRAG_CUSTOM)) &&

            !mov->max_fragment_duration && !mov->max_fragment_size)

            mov->flags |= FF_MOV_FLAG_FRAG_KEYFRAME;

    } else {

        if (mov->flags & FF_MOV_FLAG_FASTSTART)

            mov->reserved_moov_pos = avio_tell(pb);

        mov_write_mdat_tag(pb, mov);

    }



    if (t = av_dict_get(s->metadata, "creation_time", NULL, 0))

        mov->time = ff_iso8601_to_unix_time(t->value);

    if (mov->time)

        mov->time += 0x7C25B080; // 1970 based -> 1904 based



    if (mov->chapter_track)

        if ((ret = mov_create_chapter_track(s, mov->chapter_track)) < 0)

            goto error;



    if (mov->flags & FF_MOV_FLAG_RTP_HINT) {

        /* Initialize the hint tracks for each audio and video stream */

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO ||

                st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

                if ((ret = ff_mov_init_hinting(s, hint_track, i)) < 0)

                    goto error;

                hint_track++;

            }

        }

    }



    if (mov->nb_meta_tmcd) {

        /* Initialize the tmcd tracks */

        for (i = 0; i < s->nb_streams; i++) {

            AVStream *st = s->streams[i];

            t = global_tcr;



            if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

                if (!t)

                    t = av_dict_get(st->metadata, "timecode", NULL, 0);

                if (!t)

                    continue;

                if ((ret = mov_create_timecode_track(s, tmcd_track, i, t->value)) < 0)

                    goto error;

                tmcd_track++;

            }

        }

    }



    avio_flush(pb);



    if (mov->flags & FF_MOV_FLAG_ISML)

        mov_write_isml_manifest(pb, mov);



    if (mov->flags & FF_MOV_FLAG_EMPTY_MOOV &&

        !(mov->flags & FF_MOV_FLAG_DELAY_MOOV)) {

        if ((ret = mov_write_moov_tag(pb, mov, s)) < 0)

            return ret;

        mov->moov_written = 1;

        if (mov->flags & FF_MOV_FLAG_FASTSTART)

            mov->reserved_moov_pos = avio_tell(pb);

    }



    return 0;

 error:

    mov_free(s);

    return ret;

}
