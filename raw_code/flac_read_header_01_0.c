static int flac_read_header(AVFormatContext *s)

{

    int ret, metadata_last=0, metadata_type, metadata_size, found_streaminfo=0;

    uint8_t header[4];

    uint8_t *buffer=NULL;

    FLACDecContext *flac = s->priv_data;

    AVStream *st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    st->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codecpar->codec_id = AV_CODEC_ID_FLAC;

    st->need_parsing = AVSTREAM_PARSE_FULL_RAW;

    /* the parameters will be extracted from the compressed bitstream */



    /* if fLaC marker is not found, assume there is no header */

    if (avio_rl32(s->pb) != MKTAG('f','L','a','C')) {

        avio_seek(s->pb, -4, SEEK_CUR);

        return 0;

    }



    /* process metadata blocks */

    while (!avio_feof(s->pb) && !metadata_last) {

        avio_read(s->pb, header, 4);

        flac_parse_block_header(header, &metadata_last, &metadata_type,

                                   &metadata_size);

        switch (metadata_type) {

        /* allocate and read metadata block for supported types */

        case FLAC_METADATA_TYPE_STREAMINFO:

        case FLAC_METADATA_TYPE_CUESHEET:

        case FLAC_METADATA_TYPE_PICTURE:

        case FLAC_METADATA_TYPE_VORBIS_COMMENT:

        case FLAC_METADATA_TYPE_SEEKTABLE:

            buffer = av_mallocz(metadata_size + AV_INPUT_BUFFER_PADDING_SIZE);

            if (!buffer) {

                return AVERROR(ENOMEM);

            }

            if (avio_read(s->pb, buffer, metadata_size) != metadata_size) {

                RETURN_ERROR(AVERROR(EIO));

            }

            break;

        /* skip metadata block for unsupported types */

        default:

            ret = avio_skip(s->pb, metadata_size);

            if (ret < 0)

                return ret;

        }



        if (metadata_type == FLAC_METADATA_TYPE_STREAMINFO) {

            uint32_t samplerate;

            uint64_t samples;



            /* STREAMINFO can only occur once */

            if (found_streaminfo) {

                RETURN_ERROR(AVERROR_INVALIDDATA);

            }

            if (metadata_size != FLAC_STREAMINFO_SIZE) {

                RETURN_ERROR(AVERROR_INVALIDDATA);

            }

            found_streaminfo = 1;

            st->codecpar->extradata      = buffer;

            st->codecpar->extradata_size = metadata_size;

            buffer = NULL;



            /* get sample rate and sample count from STREAMINFO header;

             * other parameters will be extracted by the parser */

            samplerate = AV_RB24(st->codecpar->extradata + 10) >> 4;

            samples    = (AV_RB64(st->codecpar->extradata + 13) >> 24) & ((1ULL << 36) - 1);



            /* set time base and duration */

            if (samplerate > 0) {

                avpriv_set_pts_info(st, 64, 1, samplerate);

                if (samples > 0)

                    st->duration = samples;

            }

        } else if (metadata_type == FLAC_METADATA_TYPE_CUESHEET) {

            uint8_t isrc[13];

            uint64_t start;

            const uint8_t *offset;

            int i, chapters, track, ti;

            if (metadata_size < 431)

                RETURN_ERROR(AVERROR_INVALIDDATA);

            offset = buffer + 395;

            chapters = bytestream_get_byte(&offset) - 1;

            if (chapters <= 0)

                RETURN_ERROR(AVERROR_INVALIDDATA);

            for (i = 0; i < chapters; i++) {

                if (offset + 36 - buffer > metadata_size)

                    RETURN_ERROR(AVERROR_INVALIDDATA);

                start = bytestream_get_be64(&offset);

                track = bytestream_get_byte(&offset);

                bytestream_get_buffer(&offset, isrc, 12);

                isrc[12] = 0;

                offset += 14;

                ti = bytestream_get_byte(&offset);

                if (ti <= 0) RETURN_ERROR(AVERROR_INVALIDDATA);

                offset += ti * 12;

                avpriv_new_chapter(s, track, st->time_base, start, AV_NOPTS_VALUE, isrc);

            }

            av_freep(&buffer);

        } else if (metadata_type == FLAC_METADATA_TYPE_PICTURE) {

            ret = ff_flac_parse_picture(s, buffer, metadata_size);

            av_freep(&buffer);

            if (ret < 0) {

                av_log(s, AV_LOG_ERROR, "Error parsing attached picture.\n");

                return ret;

            }

        } else if (metadata_type == FLAC_METADATA_TYPE_SEEKTABLE) {

            const uint8_t *seekpoint = buffer;

            int i, seek_point_count = metadata_size/SEEKPOINT_SIZE;

            flac->found_seektable = 1;

            if ((s->flags&AVFMT_FLAG_FAST_SEEK)) {

                for(i=0; i<seek_point_count; i++) {

                    int64_t timestamp = bytestream_get_be64(&seekpoint);

                    int64_t pos = bytestream_get_be64(&seekpoint);

                    /* skip number of samples */

                    bytestream_get_be16(&seekpoint);

                    av_add_index_entry(st, pos, timestamp, 0, 0, AVINDEX_KEYFRAME);

                }

            }

            av_freep(&buffer);

        }

        else {



            /* STREAMINFO must be the first block */

            if (!found_streaminfo) {

                RETURN_ERROR(AVERROR_INVALIDDATA);

            }

            /* process supported blocks other than STREAMINFO */

            if (metadata_type == FLAC_METADATA_TYPE_VORBIS_COMMENT) {

                AVDictionaryEntry *chmask;



                ret = ff_vorbis_comment(s, &s->metadata, buffer, metadata_size, 1);

                if (ret < 0) {

                    av_log(s, AV_LOG_WARNING, "error parsing VorbisComment metadata\n");

                } else if (ret > 0) {

                    s->event_flags |= AVFMT_EVENT_FLAG_METADATA_UPDATED;

                }



                /* parse the channels mask if present */

                chmask = av_dict_get(s->metadata, "WAVEFORMATEXTENSIBLE_CHANNEL_MASK", NULL, 0);

                if (chmask) {

                    uint64_t mask = strtol(chmask->value, NULL, 0);

                    if (!mask || mask & ~0x3ffffULL) {

                        av_log(s, AV_LOG_WARNING,

                               "Invalid value of WAVEFORMATEXTENSIBLE_CHANNEL_MASK\n");

                    } else {

                        st->codecpar->channel_layout = mask;

                        av_dict_set(&s->metadata, "WAVEFORMATEXTENSIBLE_CHANNEL_MASK", NULL, 0);

                    }

                }

            }

            av_freep(&buffer);

        }

    }



    ret = ff_replaygain_export(st, s->metadata);

    if (ret < 0)

        return ret;



    reset_index_position(avio_tell(s->pb), st);

    return 0;



fail:

    av_free(buffer);

    return ret;

}
