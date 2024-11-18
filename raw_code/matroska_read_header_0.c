matroska_read_header (AVFormatContext    *s,

                      AVFormatParameters *ap)

{

    MatroskaDemuxContext *matroska = s->priv_data;

    char *doctype;

    int version, last_level, res = 0;

    uint32_t id;



    matroska->ctx = s;



    /* First read the EBML header. */

    doctype = NULL;

    if ((res = ebml_read_header(matroska, &doctype, &version)) < 0)

        return res;

    if ((doctype == NULL) || strcmp(doctype, "matroska")) {

        av_log(matroska->ctx, AV_LOG_ERROR,

               "Wrong EBML doctype ('%s' != 'matroska').\n",

               doctype ? doctype : "(none)");

        if (doctype)

            av_free(doctype);

        return AVERROR_NOFMT;

    }

    av_free(doctype);

    if (version > 2) {

        av_log(matroska->ctx, AV_LOG_ERROR,

               "Matroska demuxer version 2 too old for file version %d\n",

               version);

        return AVERROR_NOFMT;

    }



    /* The next thing is a segment. */

    while (1) {

        if (!(id = ebml_peek_id(matroska, &last_level)))

            return AVERROR_IO;

        if (id == MATROSKA_ID_SEGMENT)

            break;



        /* oi! */

        av_log(matroska->ctx, AV_LOG_INFO,

               "Expected a Segment ID (0x%x), but received 0x%x!\n",

               MATROSKA_ID_SEGMENT, id);

        if ((res = ebml_read_skip(matroska)) < 0)

            return res;

    }



    /* We now have a Matroska segment.

     * Seeks are from the beginning of the segment,

     * after the segment ID/length. */

    if ((res = ebml_read_master(matroska, &id)) < 0)

        return res;

    matroska->segment_start = url_ftell(&s->pb);



    matroska->time_scale = 1000000;

    /* we've found our segment, start reading the different contents in here */

    while (res == 0) {

        if (!(id = ebml_peek_id(matroska, &matroska->level_up))) {

            res = AVERROR_IO;

            break;

        } else if (matroska->level_up) {

            matroska->level_up--;

            break;

        }



        switch (id) {

            /* stream info */

            case MATROSKA_ID_INFO: {

                if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;

                res = matroska_parse_info(matroska);

                break;

            }



            /* track info headers */

            case MATROSKA_ID_TRACKS: {

                if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;

                res = matroska_parse_tracks(matroska);

                break;

            }



            /* stream index */

            case MATROSKA_ID_CUES: {

                if (!matroska->index_parsed) {

                    if ((res = ebml_read_master(matroska, &id)) < 0)

                        break;

                    res = matroska_parse_index(matroska);

                } else

                    res = ebml_read_skip(matroska);

                break;

            }



            /* metadata */

            case MATROSKA_ID_TAGS: {

                if (!matroska->metadata_parsed) {

                    if ((res = ebml_read_master(matroska, &id)) < 0)

                        break;

                    res = matroska_parse_metadata(matroska);

                } else

                    res = ebml_read_skip(matroska);

                break;

            }



            /* file index (if seekable, seek to Cues/Tags to parse it) */

            case MATROSKA_ID_SEEKHEAD: {

                if ((res = ebml_read_master(matroska, &id)) < 0)

                    break;

                res = matroska_parse_seekhead(matroska);

                break;

            }



            case MATROSKA_ID_CLUSTER: {

                /* Do not read the master - this will be done in the next

                 * call to matroska_read_packet. */

                res = 1;

                break;

            }



            default:

                av_log(matroska->ctx, AV_LOG_INFO,

                       "Unknown matroska file header ID 0x%x\n", id);

            /* fall-through */



            case EBML_ID_VOID:

                res = ebml_read_skip(matroska);

                break;

        }



        if (matroska->level_up) {

            matroska->level_up--;

            break;

        }

    }



    /* Have we found a cluster? */

    if (ebml_peek_id(matroska, NULL) == MATROSKA_ID_CLUSTER) {

        int i, j;

        MatroskaTrack *track;

        AVStream *st;



        for (i = 0; i < matroska->num_tracks; i++) {

            enum CodecID codec_id = CODEC_ID_NONE;

            uint8_t *extradata = NULL;

            int extradata_size = 0;

            int extradata_offset = 0;

            track = matroska->tracks[i];



            /* libavformat does not really support subtitles.

             * Also apply some sanity checks. */

            if ((track->type == MATROSKA_TRACK_TYPE_SUBTITLE) ||

                (track->codec_id == NULL))

                continue;



            for(j=0; codec_tags[j].str; j++){

                if(!strncmp(codec_tags[j].str, track->codec_id,

                            strlen(codec_tags[j].str))){

                    codec_id= codec_tags[j].id;

                    break;

                }

            }



            /* Set the FourCC from the CodecID. */

            /* This is the MS compatibility mode which stores a

             * BITMAPINFOHEADER in the CodecPrivate. */

            if (!strcmp(track->codec_id,

                        MATROSKA_CODEC_ID_VIDEO_VFW_FOURCC) &&

                (track->codec_priv_size >= 40) &&

                (track->codec_priv != NULL)) {

                unsigned char *p;



                /* Offset of biCompression. Stored in LE. */

                p = (unsigned char *)track->codec_priv + 16;

                ((MatroskaVideoTrack *)track)->fourcc = (p[3] << 24) |

                                 (p[2] << 16) | (p[1] << 8) | p[0];

                codec_id = codec_get_id(codec_bmp_tags, ((MatroskaVideoTrack *)track)->fourcc);



            }



            /* This is the MS compatibility mode which stores a

             * WAVEFORMATEX in the CodecPrivate. */

            else if (!strcmp(track->codec_id,

                             MATROSKA_CODEC_ID_AUDIO_ACM) &&

                (track->codec_priv_size >= 18) &&

                (track->codec_priv != NULL)) {

                unsigned char *p;

                uint16_t tag;



                /* Offset of wFormatTag. Stored in LE. */

                p = (unsigned char *)track->codec_priv;

                tag = (p[1] << 8) | p[0];

                codec_id = codec_get_id(codec_wav_tags, tag);



            }



            else if (codec_id == CODEC_ID_MPEG1VIDEO ||

                     codec_id == CODEC_ID_MPEG2VIDEO ||

                     codec_id == CODEC_ID_MPEG4      ||

                     codec_id == CODEC_ID_MSMPEG4V3  ||

                     codec_id == CODEC_ID_H264) {

                track->flags |= MATROSKA_TRACK_REORDER;

            }



            else if (codec_id == CODEC_ID_AAC && !track->codec_priv_size) {

                MatroskaAudioTrack *audiotrack = (MatroskaAudioTrack *) track;

                int profile = matroska_aac_profile(track->codec_id);

                int sri = matroska_aac_sri(audiotrack->internal_samplerate);

                extradata = av_malloc(5);

                if (extradata == NULL)

                    return AVERROR_NOMEM;

                extradata[0] = (profile << 3) | ((sri&0x0E) >> 1);

                extradata[1] = ((sri&0x01) << 7) | (audiotrack->channels<<3);

                if (strstr(track->codec_id, "SBR")) {

                    sri = matroska_aac_sri(audiotrack->samplerate);

                    extradata[2] = 0x56;

                    extradata[3] = 0xE5;

                    extradata[4] = 0x80 | (sri<<3);

                    extradata_size = 5;

                } else {

                    extradata_size = 2;

                }

                track->default_duration = 1024*1000 / audiotrack->internal_samplerate;

            }



            else if (codec_id == CODEC_ID_TTA) {

                MatroskaAudioTrack *audiotrack = (MatroskaAudioTrack *) track;

                ByteIOContext b;

                extradata_size = 30;

                extradata = av_mallocz(extradata_size);

                if (extradata == NULL)

                    return AVERROR_NOMEM;

                init_put_byte(&b, extradata, extradata_size, 1,

                              NULL, NULL, NULL, NULL);

                put_buffer(&b, (uint8_t *) "TTA1", 4);

                put_le16(&b, 1);

                put_le16(&b, audiotrack->channels);

                put_le16(&b, audiotrack->bitdepth);

                put_le32(&b, audiotrack->samplerate);

                put_le32(&b, matroska->ctx->duration * audiotrack->samplerate);

            }



            else if (codec_id == CODEC_ID_RV10 || codec_id == CODEC_ID_RV20 ||

                     codec_id == CODEC_ID_RV30 || codec_id == CODEC_ID_RV40) {

                extradata_offset = 26;

                track->codec_priv_size -= extradata_offset;

                track->flags |= MATROSKA_TRACK_REAL_V;

            }



            if (codec_id == CODEC_ID_NONE) {

                av_log(matroska->ctx, AV_LOG_INFO,

                       "Unknown/unsupported CodecID %s.\n",

                       track->codec_id);

            }



            track->stream_index = matroska->num_streams;



            matroska->num_streams++;

            st = av_new_stream(s, track->stream_index);

            if (st == NULL)

                return AVERROR_NOMEM;

            av_set_pts_info(st, 64, matroska->time_scale, 1000*1000*1000); /* 64 bit pts in ns */



            st->codec->codec_id = codec_id;



            if (track->default_duration)

                av_reduce(&st->codec->time_base.num, &st->codec->time_base.den,

                          track->default_duration, 1000, 30000);



            if(extradata){

                st->codec->extradata = extradata;

                st->codec->extradata_size = extradata_size;

            } else if(track->codec_priv && track->codec_priv_size > 0){

                st->codec->extradata = av_malloc(track->codec_priv_size);

                if(st->codec->extradata == NULL)

                    return AVERROR_NOMEM;

                st->codec->extradata_size = track->codec_priv_size;

                memcpy(st->codec->extradata,track->codec_priv+extradata_offset,

                       track->codec_priv_size);

            }



            if (track->type == MATROSKA_TRACK_TYPE_VIDEO) {

                MatroskaVideoTrack *videotrack = (MatroskaVideoTrack *)track;



                st->codec->codec_type = CODEC_TYPE_VIDEO;

                st->codec->codec_tag = videotrack->fourcc;

                st->codec->width = videotrack->pixel_width;

                st->codec->height = videotrack->pixel_height;

                if (videotrack->display_width == 0)

                    videotrack->display_width= videotrack->pixel_width;

                if (videotrack->display_height == 0)

                    videotrack->display_height= videotrack->pixel_height;

                av_reduce(&st->codec->sample_aspect_ratio.num,

                          &st->codec->sample_aspect_ratio.den,

                          st->codec->height * videotrack->display_width,

                          st->codec-> width * videotrack->display_height,

                          255);

            } else if (track->type == MATROSKA_TRACK_TYPE_AUDIO) {

                MatroskaAudioTrack *audiotrack = (MatroskaAudioTrack *)track;



                st->codec->codec_type = CODEC_TYPE_AUDIO;

                st->codec->sample_rate = audiotrack->samplerate;

                st->codec->channels = audiotrack->channels;

            } else if (track->type == MATROSKA_TRACK_TYPE_SUBTITLE) {

                st->codec->codec_type = CODEC_TYPE_SUBTITLE;

            }



            /* What do we do with private data? E.g. for Vorbis. */

        }

        res = 0;

    }



    return res;

}
