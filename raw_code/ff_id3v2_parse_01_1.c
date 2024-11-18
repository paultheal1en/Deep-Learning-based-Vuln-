static void ff_id3v2_parse(AVFormatContext *s, int len, uint8_t version, uint8_t flags, ID3v2ExtraMeta **extra_meta)

{

    int isv34, tlen, unsync;

    char tag[5];

    int64_t next, end = avio_tell(s->pb) + len;

    int taghdrlen;

    const char *reason = NULL;

    AVIOContext pb;

    AVIOContext *pbx;

    unsigned char *buffer = NULL;

    int buffer_size = 0;

    void (*extra_func)(AVFormatContext*, AVIOContext*, int, char*, ID3v2ExtraMeta**) = NULL;



    switch (version) {

    case 2:

        if (flags & 0x40) {

            reason = "compression";

            goto error;

        }

        isv34 = 0;

        taghdrlen = 6;

        break;



    case 3:

    case 4:

        isv34 = 1;

        taghdrlen = 10;

        break;



    default:

        reason = "version";

        goto error;

    }



    unsync = flags & 0x80;



    if (isv34 && flags & 0x40) /* Extended header present, just skip over it */

        avio_skip(s->pb, get_size(s->pb, 4));



    while (len >= taghdrlen) {

        unsigned int tflags = 0;

        int tunsync = 0;



        if (isv34) {

            avio_read(s->pb, tag, 4);

            tag[4] = 0;

            if(version==3){

                tlen = avio_rb32(s->pb);

            }else

                tlen = get_size(s->pb, 4);

            tflags = avio_rb16(s->pb);

            tunsync = tflags & ID3v2_FLAG_UNSYNCH;

        } else {

            avio_read(s->pb, tag, 3);

            tag[3] = 0;

            tlen = avio_rb24(s->pb);

        }

        if (tlen < 0 || tlen > len - taghdrlen) {

            av_log(s, AV_LOG_WARNING, "Invalid size in frame %s, skipping the rest of tag.\n", tag);

            break;

        }

        len -= taghdrlen + tlen;

        next = avio_tell(s->pb) + tlen;



        if (!tlen) {

            if (tag[0])

                av_log(s, AV_LOG_DEBUG, "Invalid empty frame %s, skipping.\n", tag);

            continue;

        }



        if (tflags & ID3v2_FLAG_DATALEN) {

            avio_rb32(s->pb);

            tlen -= 4;

        }



        if (tflags & (ID3v2_FLAG_ENCRYPTION | ID3v2_FLAG_COMPRESSION)) {

            av_log(s, AV_LOG_WARNING, "Skipping encrypted/compressed ID3v2 frame %s.\n", tag);

            avio_skip(s->pb, tlen);

        /* check for text tag or supported special meta tag */

        } else if (tag[0] == 'T' || (extra_meta && (extra_func = get_extra_meta_func(tag, isv34)->read))) {

            if (unsync || tunsync) {

                int i, j;

                av_fast_malloc(&buffer, &buffer_size, tlen);

                if (!buffer) {

                    av_log(s, AV_LOG_ERROR, "Failed to alloc %d bytes\n", tlen);

                    goto seek;

                }

                for (i = 0, j = 0; i < tlen; i++, j++) {

                    buffer[j] = avio_r8(s->pb);

                    if (j > 0 && !buffer[j] && buffer[j - 1] == 0xff) {

                        /* Unsynchronised byte, skip it */

                        j--;

                    }

                }

                ffio_init_context(&pb, buffer, j, 0, NULL, NULL, NULL, NULL);

                tlen = j;

                pbx = &pb; // read from sync buffer

            } else {

                pbx = s->pb; // read straight from input

            }

            if (tag[0] == 'T')

                /* parse text tag */

                read_ttag(s, pbx, tlen, tag);

            else

                /* parse special meta tag */

                extra_func(s, pbx, tlen, tag, extra_meta);

        }

        else if (!tag[0]) {

            if (tag[1])

                av_log(s, AV_LOG_WARNING, "invalid frame id, assuming padding");

            avio_skip(s->pb, tlen);

            break;

        }

        /* Skip to end of tag */

seek:

        avio_seek(s->pb, next, SEEK_SET);

    }



    if (version == 4 && flags & 0x10) /* Footer preset, always 10 bytes, skip over it */

        end += 10;



  error:

    if (reason)

        av_log(s, AV_LOG_INFO, "ID3v2.%d tag skipped, cannot handle %s\n", version, reason);

    avio_seek(s->pb, end, SEEK_SET);

    av_free(buffer);

    return;

}
