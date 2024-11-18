matroska_add_stream (MatroskaDemuxContext *matroska)

{

    int res = 0;

    uint32_t id;

    MatroskaTrack *track;



    av_log(matroska->ctx, AV_LOG_DEBUG, "parsing track, adding stream..,\n");



    /* Allocate a generic track. As soon as we know its type we'll realloc. */

    track = av_mallocz(MAX_TRACK_SIZE);

    matroska->num_tracks++;

    strcpy(track->language, "eng");



    /* start with the master */

    if ((res = ebml_read_master(matroska, &id)) < 0)

        return res;



    /* try reading the trackentry headers */

    while (res == 0) {

        if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

            res = AVERROR(EIO);

            break;

        } else if (matroska->level_up > 0) {

            matroska->level_up--;

            break;

        }



        switch (id) {

            /* track number (unique stream ID) */

            case MATROSKA_ID_TRACKNUMBER: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                track->num = num;

                break;

            }



            /* track UID (unique identifier) */

            case MATROSKA_ID_TRACKUID: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                track->uid = num;

                break;

            }



            /* track type (video, audio, combined, subtitle, etc.) */

            case MATROSKA_ID_TRACKTYPE: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                if (track->type && track->type != num) {

                    av_log(matroska->ctx, AV_LOG_INFO,

                           "More than one tracktype in an entry - skip\n");

                    break;

                }

                track->type = num;



                switch (track->type) {

                    case MATROSKA_TRACK_TYPE_VIDEO:

                    case MATROSKA_TRACK_TYPE_AUDIO:

                    case MATROSKA_TRACK_TYPE_SUBTITLE:

                        break;

                    case MATROSKA_TRACK_TYPE_COMPLEX:

                    case MATROSKA_TRACK_TYPE_LOGO:

                    case MATROSKA_TRACK_TYPE_CONTROL:

                    default:

                        av_log(matroska->ctx, AV_LOG_INFO,

                               "Unknown or unsupported track type 0x%x\n",

                               track->type);

                        track->type = MATROSKA_TRACK_TYPE_NONE;

                        break;

                }

                matroska->tracks[matroska->num_tracks - 1] = track;

                break;

            }



            /* tracktype specific stuff for video */

            case MATROSKA_ID_TRACKVIDEO: {

                MatroskaVideoTrack *videotrack;

                if (!track->type)

                    track->type = MATROSKA_TRACK_TYPE_VIDEO;

                if (track->type != MATROSKA_TRACK_TYPE_VIDEO) {

                    av_log(matroska->ctx, AV_LOG_INFO,

                           "video data in non-video track - ignoring\n");

                    res = AVERROR_INVALIDDATA;

                    break;

                } else if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;

                videotrack = (MatroskaVideoTrack *)track;



                while (res == 0) {

                    if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

                        res = AVERROR(EIO);

                        break;

                    } else if (matroska->level_up > 0) {

                        matroska->level_up--;

                        break;

                    }



                    switch (id) {

                        /* fixme, this should be one-up, but I get it here */

                        case MATROSKA_ID_TRACKDEFAULTDURATION: {

                            uint64_t num;

                            if ((res = ebml_read_uint (matroska, &id,

                                                       &num)) < 0)

                                break;

                            track->default_duration = num;

                            break;

                        }



                        /* video framerate */

                        case MATROSKA_ID_VIDEOFRAMERATE: {

                            double num;

                            if ((res = ebml_read_float(matroska, &id,

                                                       &num)) < 0)

                                break;

                            if (!track->default_duration)

                                track->default_duration = 1000000000/num;

                            break;

                        }



                        /* width of the size to display the video at */

                        case MATROSKA_ID_VIDEODISPLAYWIDTH: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            videotrack->display_width = num;

                            break;

                        }



                        /* height of the size to display the video at */

                        case MATROSKA_ID_VIDEODISPLAYHEIGHT: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            videotrack->display_height = num;

                            break;

                        }



                        /* width of the video in the file */

                        case MATROSKA_ID_VIDEOPIXELWIDTH: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            videotrack->pixel_width = num;

                            break;

                        }



                        /* height of the video in the file */

                        case MATROSKA_ID_VIDEOPIXELHEIGHT: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            videotrack->pixel_height = num;

                            break;

                        }



                        /* whether the video is interlaced */

                        case MATROSKA_ID_VIDEOFLAGINTERLACED: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            if (num)

                                track->flags |=

                                    MATROSKA_VIDEOTRACK_INTERLACED;

                            else

                                track->flags &=

                                    ~MATROSKA_VIDEOTRACK_INTERLACED;

                            break;

                        }



                        /* stereo mode (whether the video has two streams,

                         * where one is for the left eye and the other for

                         * the right eye, which creates a 3D-like

                         * effect) */

                        case MATROSKA_ID_VIDEOSTEREOMODE: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            if (num != MATROSKA_EYE_MODE_MONO &&

                                num != MATROSKA_EYE_MODE_LEFT &&

                                num != MATROSKA_EYE_MODE_RIGHT &&

                                num != MATROSKA_EYE_MODE_BOTH) {

                                av_log(matroska->ctx, AV_LOG_INFO,

                                       "Ignoring unknown eye mode 0x%x\n",

                                       (uint32_t) num);

                                break;

                            }

                            videotrack->eye_mode = num;

                            break;

                        }



                        /* aspect ratio behaviour */

                        case MATROSKA_ID_VIDEOASPECTRATIO: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            if (num != MATROSKA_ASPECT_RATIO_MODE_FREE &&

                                num != MATROSKA_ASPECT_RATIO_MODE_KEEP &&

                                num != MATROSKA_ASPECT_RATIO_MODE_FIXED) {

                                av_log(matroska->ctx, AV_LOG_INFO,

                                       "Ignoring unknown aspect ratio 0x%x\n",

                                       (uint32_t) num);

                                break;

                            }

                            videotrack->ar_mode = num;

                            break;

                        }



                        /* colorspace (only matters for raw video)

                         * fourcc */

                        case MATROSKA_ID_VIDEOCOLORSPACE: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            videotrack->fourcc = num;

                            break;

                        }



                        default:

                            av_log(matroska->ctx, AV_LOG_INFO,

                                   "Unknown video track header entry "

                                   "0x%x - ignoring\n", id);

                            /* pass-through */



                        case EBML_ID_VOID:

                            res = ebml_read_skip(matroska);

                            break;

                    }



                    if (matroska->level_up) {

                        matroska->level_up--;

                        break;

                    }

                }

                break;

            }



            /* tracktype specific stuff for audio */

            case MATROSKA_ID_TRACKAUDIO: {

                MatroskaAudioTrack *audiotrack;

                if (!track->type)

                    track->type = MATROSKA_TRACK_TYPE_AUDIO;

                if (track->type != MATROSKA_TRACK_TYPE_AUDIO) {

                    av_log(matroska->ctx, AV_LOG_INFO,

                           "audio data in non-audio track - ignoring\n");

                    res = AVERROR_INVALIDDATA;

                    break;

                } else if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;

                audiotrack = (MatroskaAudioTrack *)track;

                audiotrack->channels = 1;

                audiotrack->samplerate = 8000;



                while (res == 0) {

                    if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

                        res = AVERROR(EIO);

                        break;

                    } else if (matroska->level_up > 0) {

                        matroska->level_up--;

                        break;

                    }



                    switch (id) {

                        /* samplerate */

                        case MATROSKA_ID_AUDIOSAMPLINGFREQ: {

                            double num;

                            if ((res = ebml_read_float(matroska, &id,

                                                       &num)) < 0)

                                break;

                            audiotrack->internal_samplerate =

                            audiotrack->samplerate = num;

                            break;

                        }



                        case MATROSKA_ID_AUDIOOUTSAMPLINGFREQ: {

                            double num;

                            if ((res = ebml_read_float(matroska, &id,

                                                       &num)) < 0)

                                break;

                            audiotrack->samplerate = num;

                            break;

                        }



                            /* bitdepth */

                        case MATROSKA_ID_AUDIOBITDEPTH: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            audiotrack->bitdepth = num;

                            break;

                        }



                            /* channels */

                        case MATROSKA_ID_AUDIOCHANNELS: {

                            uint64_t num;

                            if ((res = ebml_read_uint(matroska, &id,

                                                      &num)) < 0)

                                break;

                            audiotrack->channels = num;

                            break;

                        }



                        default:

                            av_log(matroska->ctx, AV_LOG_INFO,

                                   "Unknown audio track header entry "

                                   "0x%x - ignoring\n", id);

                            /* pass-through */



                        case EBML_ID_VOID:

                            res = ebml_read_skip(matroska);

                            break;

                    }



                    if (matroska->level_up) {

                        matroska->level_up--;

                        break;

                    }

                }

                break;

            }



                /* codec identifier */

            case MATROSKA_ID_CODECID: {

                char *text;

                if ((res = ebml_read_ascii(matroska, &id, &text)) < 0)

                    break;

                track->codec_id = text;

                break;

            }



                /* codec private data */

            case MATROSKA_ID_CODECPRIVATE: {

                uint8_t *data;

                int size;

                if ((res = ebml_read_binary(matroska, &id, &data, &size) < 0))

                    break;

                track->codec_priv = data;

                track->codec_priv_size = size;

                break;

            }



                /* name of the codec */

            case MATROSKA_ID_CODECNAME: {

                char *text;

                if ((res = ebml_read_utf8(matroska, &id, &text)) < 0)

                    break;

                track->codec_name = text;

                break;

            }



                /* name of this track */

            case MATROSKA_ID_TRACKNAME: {

                char *text;

                if ((res = ebml_read_utf8(matroska, &id, &text)) < 0)

                    break;

                track->name = text;

                break;

            }



                /* language (matters for audio/subtitles, mostly) */

            case MATROSKA_ID_TRACKLANGUAGE: {

                char *text, *end;

                if ((res = ebml_read_utf8(matroska, &id, &text)) < 0)

                    break;

                if ((end = strchr(text, '-')))

                    *end = '\0';

                if (strlen(text) == 3)

                    strcpy(track->language, text);

                av_free(text);

                break;

            }



                /* whether this is actually used */

            case MATROSKA_ID_TRACKFLAGENABLED: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                if (num)

                    track->flags |= MATROSKA_TRACK_ENABLED;

                else

                    track->flags &= ~MATROSKA_TRACK_ENABLED;

                break;

            }



                /* whether it's the default for this track type */

            case MATROSKA_ID_TRACKFLAGDEFAULT: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                if (num)

                    track->flags |= MATROSKA_TRACK_DEFAULT;

                else

                    track->flags &= ~MATROSKA_TRACK_DEFAULT;

                break;

            }



                /* lacing (like MPEG, where blocks don't end/start on frame

                 * boundaries) */

            case MATROSKA_ID_TRACKFLAGLACING: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                if (num)

                    track->flags |= MATROSKA_TRACK_LACING;

                else

                    track->flags &= ~MATROSKA_TRACK_LACING;

                break;

            }



                /* default length (in time) of one data block in this track */

            case MATROSKA_ID_TRACKDEFAULTDURATION: {

                uint64_t num;

                if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                    break;

                track->default_duration = num;

                break;

            }



            case MATROSKA_ID_TRACKCONTENTENCODINGS: {

                if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;



                while (res == 0) {

                    if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

                        res = AVERROR(EIO);

                        break;

                    } else if (matroska->level_up > 0) {

                        matroska->level_up--;

                        break;

                    }



                    switch (id) {

                        case MATROSKA_ID_TRACKCONTENTENCODING: {

                            int encoding_scope = 1;

                            if ((res = ebml_read_master(matroska, &id)) < 0)

                                break;



                            while (res == 0) {

                                if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

                                    res = AVERROR(EIO);

                                    break;

                                } else if (matroska->level_up > 0) {

                                    matroska->level_up--;

                                    break;

                                }



                                switch (id) {

                                    case MATROSKA_ID_ENCODINGSCOPE: {

                                        uint64_t num;

                                        if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                                            break;

                                        encoding_scope = num;

                                        break;

                                    }



                                    case MATROSKA_ID_ENCODINGTYPE: {

                                        uint64_t num;

                                        if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                                            break;

                                        if (num)

                                            av_log(matroska->ctx, AV_LOG_ERROR,

                                                   "Unsupported encoding type");

                                        break;

                                    }



                                    case MATROSKA_ID_ENCODINGCOMPRESSION: {

                                        if ((res = ebml_read_master(matroska, &id)) < 0)

                                            break;



                                        while (res == 0) {

                                            if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

                                                res = AVERROR(EIO);

                                                break;

                                            } else if (matroska->level_up > 0) {

                                                matroska->level_up--;

                                                break;

                                            }



                                            switch (id) {

                                                case MATROSKA_ID_ENCODINGCOMPALGO: {

                                                    uint64_t num;

                                                    if ((res = ebml_read_uint(matroska, &id, &num)) < 0)

                                                        break;

                                                    if (num != MATROSKA_TRACK_ENCODING_COMP_HEADERSTRIP &&

#ifdef CONFIG_ZLIB

                                                        num != MATROSKA_TRACK_ENCODING_COMP_ZLIB &&

#endif

#ifdef CONFIG_BZLIB

                                                        num != MATROSKA_TRACK_ENCODING_COMP_BZLIB &&

#endif

                                                        num != MATROSKA_TRACK_ENCODING_COMP_LZO)

                                                        av_log(matroska->ctx, AV_LOG_ERROR,

                                                               "Unsupported compression algo\n");

                                                    track->encoding_algo = num;

                                                    break;

                                                }



                                                case MATROSKA_ID_ENCODINGCOMPSETTINGS: {

                                                    uint8_t *data;

                                                    int size;

                                                    if ((res = ebml_read_binary(matroska, &id, &data, &size) < 0))

                                                        break;

                                                    track->encoding_settings = data;

                                                    track->encoding_settings_len = size;

                                                    break;

                                                }



                                                default:

                                                    av_log(matroska->ctx, AV_LOG_INFO,

                                                           "Unknown compression header entry "

                                                           "0x%x - ignoring\n", id);

                                                    /* pass-through */



                                                case EBML_ID_VOID:

                                                    res = ebml_read_skip(matroska);

                                                    break;

                                            }



                                            if (matroska->level_up) {

                                                matroska->level_up--;

                                                break;

                                            }

                                        }

                                        break;

                                    }



                                    default:

                                        av_log(matroska->ctx, AV_LOG_INFO,

                                               "Unknown content encoding header entry "

                                               "0x%x - ignoring\n", id);

                                        /* pass-through */



                                    case EBML_ID_VOID:

                                        res = ebml_read_skip(matroska);

                                        break;

                                }



                                if (matroska->level_up) {

                                    matroska->level_up--;

                                    break;

                                }

                            }



                            track->encoding_scope = encoding_scope;

                            break;

                        }



                        default:

                            av_log(matroska->ctx, AV_LOG_INFO,

                                   "Unknown content encodings header entry "

                                   "0x%x - ignoring\n", id);

                            /* pass-through */



                        case EBML_ID_VOID:

                            res = ebml_read_skip(matroska);

                            break;

                    }



                    if (matroska->level_up) {

                        matroska->level_up--;

                        break;

                    }

                }

                break;

            }



            case MATROSKA_ID_TRACKTIMECODESCALE: {

                double num;

                if ((res = ebml_read_float(matroska, &id, &num)) < 0)

                    break;

                track->time_scale = num;

                break;

            }



            default:

                av_log(matroska->ctx, AV_LOG_INFO,

                       "Unknown track header entry 0x%x - ignoring\n", id);

                /* pass-through */



            case EBML_ID_VOID:

            /* we ignore these because they're nothing useful. */

            case MATROSKA_ID_TRACKFLAGFORCED:

            case MATROSKA_ID_CODECDECODEALL:

            case MATROSKA_ID_CODECINFOURL:

            case MATROSKA_ID_CODECDOWNLOADURL:

            case MATROSKA_ID_TRACKMINCACHE:

            case MATROSKA_ID_TRACKMAXCACHE:

                res = ebml_read_skip(matroska);

                break;

        }



        if (matroska->level_up) {

            matroska->level_up--;

            break;

        }

    }



    return res;

}
