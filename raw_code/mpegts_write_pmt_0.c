static void mpegts_write_pmt(AVFormatContext *s, MpegTSService *service)

{

    MpegTSWrite *ts = s->priv_data;

    uint8_t data[1012], *q, *desc_length_ptr, *program_info_length_ptr;

    int val, stream_type, i;



    q = data;

    put16(&q, 0xe000 | service->pcr_pid);



    program_info_length_ptr = q;

    q += 2; /* patched after */



    /* put program info here */



    val = 0xf000 | (q - program_info_length_ptr - 2);

    program_info_length_ptr[0] = val >> 8;

    program_info_length_ptr[1] = val;



    for(i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        MpegTSWriteStream *ts_st = st->priv_data;

        AVDictionaryEntry *lang = av_dict_get(st->metadata, "language", NULL,0);

        switch(st->codec->codec_id) {

        case AV_CODEC_ID_MPEG1VIDEO:

        case AV_CODEC_ID_MPEG2VIDEO:

            stream_type = STREAM_TYPE_VIDEO_MPEG2;

            break;

        case AV_CODEC_ID_MPEG4:

            stream_type = STREAM_TYPE_VIDEO_MPEG4;

            break;

        case AV_CODEC_ID_H264:

            stream_type = STREAM_TYPE_VIDEO_H264;

            break;

        case AV_CODEC_ID_CAVS:

            stream_type = STREAM_TYPE_VIDEO_CAVS;

            break;

        case AV_CODEC_ID_DIRAC:

            stream_type = STREAM_TYPE_VIDEO_DIRAC;

            break;

        case AV_CODEC_ID_MP2:

        case AV_CODEC_ID_MP3:

            stream_type = STREAM_TYPE_AUDIO_MPEG1;

            break;

        case AV_CODEC_ID_AAC:

            stream_type = (ts->flags & MPEGTS_FLAG_AAC_LATM) ? STREAM_TYPE_AUDIO_AAC_LATM : STREAM_TYPE_AUDIO_AAC;

            break;

        case AV_CODEC_ID_AAC_LATM:

            stream_type = STREAM_TYPE_AUDIO_AAC_LATM;

            break;

        case AV_CODEC_ID_AC3:

            stream_type = STREAM_TYPE_AUDIO_AC3;

            break;

        default:

            stream_type = STREAM_TYPE_PRIVATE_DATA;

            break;

        }

        *q++ = stream_type;

        put16(&q, 0xe000 | ts_st->pid);

        desc_length_ptr = q;

        q += 2; /* patched after */



        /* write optional descriptors here */

        switch(st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if(st->codec->codec_id==AV_CODEC_ID_EAC3){

                *q++=0x7a; // EAC3 descriptor see A038 DVB SI

                *q++=1; // 1 byte, all flags sets to 0

                *q++=0; // omit all fields...

            }

            if(st->codec->codec_id==AV_CODEC_ID_S302M){

                *q++ = 0x05; /* MPEG-2 registration descriptor*/

                *q++ = 4;

                *q++ = 'B';

                *q++ = 'S';

                *q++ = 'S';

                *q++ = 'D';

            }



            if (lang) {

                char *p;

                char *next = lang->value;

                uint8_t *len_ptr;



                *q++ = 0x0a; /* ISO 639 language descriptor */

                len_ptr = q++;

                *len_ptr = 0;



                for (p = lang->value; next && *len_ptr < 255 / 4 * 4; p = next + 1) {

                    next = strchr(p, ',');

                    if (strlen(p) != 3 && (!next || next != p + 3))

                        continue; /* not a 3-letter code */



                    *q++ = *p++;

                    *q++ = *p++;

                    *q++ = *p++;



                if (st->disposition & AV_DISPOSITION_CLEAN_EFFECTS)

                    *q++ = 0x01;

                else if (st->disposition & AV_DISPOSITION_HEARING_IMPAIRED)

                    *q++ = 0x02;

                else if (st->disposition & AV_DISPOSITION_VISUAL_IMPAIRED)

                    *q++ = 0x03;

                else

                    *q++ = 0; /* undefined type */



                    *len_ptr += 4;

                }



                if (*len_ptr == 0)

                    q -= 2; /* no language codes were written */

            }

            break;

        case AVMEDIA_TYPE_SUBTITLE:

            {

                const char default_language[] = "und";

                const char *language = lang && strlen(lang->value) >= 3 ? lang->value : default_language;



                if (st->codec->codec_id == AV_CODEC_ID_DVB_SUBTITLE) {

                    uint8_t *len_ptr;

                    int extradata_copied = 0;



                    *q++ = 0x59; /* subtitling_descriptor */

                    len_ptr = q++;



                    while (strlen(language) >= 3 && (sizeof(data) - (q - data)) >= 8) { /* 8 bytes per DVB subtitle substream data */

                        *q++ = *language++;

                        *q++ = *language++;

                        *q++ = *language++;

                        /* Skip comma */

                        if (*language != '\0')

                            language++;



                        if (st->codec->extradata_size - extradata_copied >= 5) {

                            *q++ = st->codec->extradata[extradata_copied + 4]; /* subtitling_type */

                            memcpy(q, st->codec->extradata + extradata_copied, 4); /* composition_page_id and ancillary_page_id */

                            extradata_copied += 5;

                            q += 4;

                        } else {

                            /* subtitling_type:

                             * 0x10 - normal with no monitor aspect ratio criticality

                             * 0x20 - for the hard of hearing with no monitor aspect ratio criticality */

                            *q++ = (st->disposition & AV_DISPOSITION_HEARING_IMPAIRED) ? 0x20 : 0x10;

                            if ((st->codec->extradata_size == 4) && (extradata_copied == 0)) {

                                /* support of old 4-byte extradata format */

                                memcpy(q, st->codec->extradata, 4); /* composition_page_id and ancillary_page_id */

                                extradata_copied += 4;

                                q += 4;

                            } else {

                                put16(&q, 1); /* composition_page_id */

                                put16(&q, 1); /* ancillary_page_id */

                            }

                        }

                    }



                    *len_ptr = q - len_ptr - 1;

                } else if (st->codec->codec_id == AV_CODEC_ID_DVB_TELETEXT) {

                    uint8_t *len_ptr = NULL;

                    int extradata_copied = 0;



                    /* The descriptor tag. teletext_descriptor */

                    *q++ = 0x56;

                    len_ptr = q++;



                    while (strlen(language) >= 3) {

                        *q++ = *language++;

                        *q++ = *language++;

                        *q++ = *language++;

                        /* Skip comma */

                        if (*language != '\0')

                            language++;



                        if (st->codec->extradata_size - 1 > extradata_copied) {

                            memcpy(q, st->codec->extradata + extradata_copied, 2);

                            extradata_copied += 2;

                            q += 2;

                        } else {

                            /* The Teletext descriptor:

                             * teletext_type: This 5-bit field indicates the type of Teletext page indicated. (0x01 Initial Teletext page)

                             * teletext_magazine_number: This is a 3-bit field which identifies the magazine number.

                             * teletext_page_number: This is an 8-bit field giving two 4-bit hex digits identifying the page number. */

                            *q++ = 0x08;

                            *q++ = 0x00;

                        }

                    }



                    *len_ptr = q - len_ptr - 1;

                 }

            }

            break;

        case AVMEDIA_TYPE_VIDEO:

            if (stream_type == STREAM_TYPE_VIDEO_DIRAC) {

                *q++ = 0x05; /*MPEG-2 registration descriptor*/

                *q++ = 4;

                *q++ = 'd';

                *q++ = 'r';

                *q++ = 'a';

                *q++ = 'c';

            }

            break;

        case AVMEDIA_TYPE_DATA:

            if (st->codec->codec_id == AV_CODEC_ID_SMPTE_KLV) {

                *q++ = 0x05; /* MPEG-2 registration descriptor */

                *q++ = 4;

                *q++ = 'K';

                *q++ = 'L';

                *q++ = 'V';

                *q++ = 'A';

            }

            break;

        }



        val = 0xf000 | (q - desc_length_ptr - 2);

        desc_length_ptr[0] = val >> 8;

        desc_length_ptr[1] = val;

    }

    mpegts_write_section1(&service->pmt, PMT_TID, service->sid, ts->tables_version, 0, 0,

                          data, q - data);

}
