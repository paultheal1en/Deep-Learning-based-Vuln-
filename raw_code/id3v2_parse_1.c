static void id3v2_parse(AVFormatContext *s, int len, uint8_t version,

                        uint8_t flags, ID3v2ExtraMeta **extra_meta)

{

    int isv34, unsync;

    unsigned tlen;

    char tag[5];

    int64_t next, end = avio_tell(s->pb) + len;

    int taghdrlen;

    const char *reason = NULL;

    AVIOContext pb;

    AVIOContext *pbx;

    unsigned char *buffer = NULL;

    int buffer_size       = 0;

    const ID3v2EMFunc *extra_func = NULL;

    unsigned char *uncompressed_buffer = NULL;

    int uncompressed_buffer_size = 0;



    av_log(s, AV_LOG_DEBUG, "id3v2 ver:%d flags:%02X len:%d\n", version, flags, len);



    switch (version) {

    case 2:

        if (flags & 0x40) {

            reason = "compression";

            goto error;

        }

        isv34     = 0;

        taghdrlen = 6;

        break;



    case 3:

    case 4:

        isv34     = 1;

        taghdrlen = 10;

        break;



    default:

        reason = "version";

        goto error;

    }



    unsync = flags & 0x80;



    if (isv34 && flags & 0x40) { /* Extended header present, just skip over it */

        int extlen = get_size(s->pb, 4);

        if (version == 4)

            /* In v2.4 the length includes the length field we just read. */

            extlen -= 4;



        if (extlen < 0) {

            reason = "invalid extended header length";

            goto error;

        }

        avio_skip(s->pb, extlen);

        len -= extlen + 4;

        if (len < 0) {

            reason = "extended header too long.";

            goto error;

        }

    }



    while (len >= taghdrlen) {

        unsigned int tflags = 0;

        int tunsync         = 0;

        int tcomp           = 0;

        int tencr           = 0;

        unsigned long dlen;



        if (isv34) {

            avio_read(s->pb, tag, 4);

            tag[4] = 0;

            if (version == 3) {

                tlen = avio_rb32(s->pb);

            } else

                tlen = get_size(s->pb, 4);

            tflags  = avio_rb16(s->pb);

            tunsync = tflags & ID3v2_FLAG_UNSYNCH;

        } else {

            avio_read(s->pb, tag, 3);

            tag[3] = 0;

            tlen   = avio_rb24(s->pb);

        }

        if (tlen > (1<<28))

            break;

        len -= taghdrlen + tlen;



        if (len < 0)

            break;



        next = avio_tell(s->pb) + tlen;



        if (!tlen) {

            if (tag[0])

                av_log(s, AV_LOG_DEBUG, "Invalid empty frame %s, skipping.\n",

                       tag);

            continue;

        }



        if (tflags & ID3v2_FLAG_DATALEN) {

            if (tlen < 4)

                break;

            dlen = avio_rb32(s->pb);

            tlen -= 4;

        } else

            dlen = tlen;



        tcomp = tflags & ID3v2_FLAG_COMPRESSION;

        tencr = tflags & ID3v2_FLAG_ENCRYPTION;



        /* skip encrypted tags and, if no zlib, compressed tags */

        if (tencr || (!CONFIG_ZLIB && tcomp)) {

            const char *type;

            if (!tcomp)

                type = "encrypted";

            else if (!tencr)

                type = "compressed";

            else

                type = "encrypted and compressed";



            av_log(s, AV_LOG_WARNING, "Skipping %s ID3v2 frame %s.\n", type, tag);

            avio_skip(s->pb, tlen);

        /* check for text tag or supported special meta tag */

        } else if (tag[0] == 'T' ||

                   (extra_meta &&

                    (extra_func = get_extra_meta_func(tag, isv34)))) {

            pbx = s->pb;



            if (unsync || tunsync || tcomp) {

                av_fast_malloc(&buffer, &buffer_size, tlen);

                if (!buffer) {

                    av_log(s, AV_LOG_ERROR, "Failed to alloc %d bytes\n", tlen);

                    goto seek;

                }

            }

            if (unsync || tunsync) {

                int64_t end = avio_tell(s->pb) + tlen;

                uint8_t *b;



                b = buffer;

                while (avio_tell(s->pb) < end && b - buffer < tlen && !s->pb->eof_reached) {

                    *b++ = avio_r8(s->pb);

                    if (*(b - 1) == 0xff && avio_tell(s->pb) < end - 1 &&

                        b - buffer < tlen &&

                        !s->pb->eof_reached ) {

                        uint8_t val = avio_r8(s->pb);

                        *b++ = val ? val : avio_r8(s->pb);

                    }

                }

                ffio_init_context(&pb, buffer, b - buffer, 0, NULL, NULL, NULL,

                                  NULL);

                tlen = b - buffer;

                pbx  = &pb; // read from sync buffer

            }



#if CONFIG_ZLIB

                if (tcomp) {

                    int err;



                    av_log(s, AV_LOG_DEBUG, "Compresssed frame %s tlen=%d dlen=%ld\n", tag, tlen, dlen);



                    av_fast_malloc(&uncompressed_buffer, &uncompressed_buffer_size, dlen);

                    if (!uncompressed_buffer) {

                        av_log(s, AV_LOG_ERROR, "Failed to alloc %ld bytes\n", dlen);

                        goto seek;

                    }



                    if (!(unsync || tunsync)) {

                        err = avio_read(s->pb, buffer, tlen);

                        if (err < 0) {

                            av_log(s, AV_LOG_ERROR, "Failed to read compressed tag\n");

                            goto seek;

                        }

                        tlen = err;

                    }



                    err = uncompress(uncompressed_buffer, &dlen, buffer, tlen);

                    if (err != Z_OK) {

                        av_log(s, AV_LOG_ERROR, "Failed to uncompress tag: %d\n", err);

                        goto seek;

                    }

                    ffio_init_context(&pb, uncompressed_buffer, dlen, 0, NULL, NULL, NULL, NULL);

                    tlen = dlen;

                    pbx = &pb; // read from sync buffer

                }

#endif

            if (tag[0] == 'T')

                /* parse text tag */

                read_ttag(s, pbx, tlen, &s->metadata, tag);

            else

                /* parse special meta tag */

                extra_func->read(s, pbx, tlen, tag, extra_meta);

        } else if (!tag[0]) {

            if (tag[1])

                av_log(s, AV_LOG_WARNING, "invalid frame id, assuming padding\n");

            avio_skip(s->pb, tlen);

            break;

        }

        /* Skip to end of tag */

seek:

        avio_seek(s->pb, next, SEEK_SET);

    }



    /* Footer preset, always 10 bytes, skip over it */

    if (version == 4 && flags & 0x10)

        end += 10;



error:

    if (reason)

        av_log(s, AV_LOG_INFO, "ID3v2.%d tag skipped, cannot handle %s\n",

               version, reason);

    avio_seek(s->pb, end, SEEK_SET);

    av_free(buffer);

    av_free(uncompressed_buffer);

    return;

}
