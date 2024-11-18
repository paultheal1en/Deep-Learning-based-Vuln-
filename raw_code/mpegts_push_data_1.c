static int mpegts_push_data(MpegTSFilter *filter,

                            const uint8_t *buf, int buf_size, int is_start,

                            int64_t pos)

{

    PESContext *pes = filter->u.pes_filter.opaque;

    MpegTSContext *ts = pes->ts;

    const uint8_t *p;

    int len, code;



    if(!ts->pkt)

        return 0;



    if (is_start) {

        if (pes->state == MPEGTS_PAYLOAD && pes->data_index > 0) {

            new_pes_packet(pes, ts->pkt);

            ts->stop_parse = 1;

        }

        pes->state = MPEGTS_HEADER;

        pes->data_index = 0;

        pes->ts_packet_pos = pos;

    }

    p = buf;

    while (buf_size > 0) {

        switch(pes->state) {

        case MPEGTS_HEADER:

            len = PES_START_SIZE - pes->data_index;

            if (len > buf_size)

                len = buf_size;

            memcpy(pes->header + pes->data_index, p, len);

            pes->data_index += len;

            p += len;

            buf_size -= len;

            if (pes->data_index == PES_START_SIZE) {

                /* we got all the PES or section header. We can now

                   decide */

                if (pes->header[0] == 0x00 && pes->header[1] == 0x00 &&

                    pes->header[2] == 0x01) {

                    /* it must be an mpeg2 PES stream */

                    code = pes->header[3] | 0x100;

                    av_dlog(pes->stream, "pid=%x pes_code=%#x\n", pes->pid, code);



                    if ((pes->st && pes->st->discard == AVDISCARD_ALL) ||

                        code == 0x1be) /* padding_stream */

                        goto skip;



                    /* stream not present in PMT */

                    if (!pes->st) {

                        pes->st = av_new_stream(ts->stream, pes->pid);

                        if (!pes->st)

                            return AVERROR(ENOMEM);

                        mpegts_set_stream_info(pes->st, pes, 0, 0);

                    }



                    pes->total_size = AV_RB16(pes->header + 4);

                    /* NOTE: a zero total size means the PES size is

                       unbounded */

                    if (!pes->total_size)

                        pes->total_size = MAX_PES_PAYLOAD;



                    /* allocate pes buffer */

                    pes->buffer = av_malloc(pes->total_size+FF_INPUT_BUFFER_PADDING_SIZE);

                    if (!pes->buffer)

                        return AVERROR(ENOMEM);



                    if (code != 0x1bc && code != 0x1bf && /* program_stream_map, private_stream_2 */

                        code != 0x1f0 && code != 0x1f1 && /* ECM, EMM */

                        code != 0x1ff && code != 0x1f2 && /* program_stream_directory, DSMCC_stream */

                        code != 0x1f8) {                  /* ITU-T Rec. H.222.1 type E stream */

                        pes->state = MPEGTS_PESHEADER;

                        if (pes->st->codec->codec_id == CODEC_ID_NONE) {

                            av_dlog(pes->stream, "pid=%x stream_type=%x probing\n",

                                    pes->pid, pes->stream_type);

                            pes->st->codec->codec_id = CODEC_ID_PROBE;

                        }

                    } else {

                        pes->state = MPEGTS_PAYLOAD;

                        pes->data_index = 0;

                    }

                } else {

                    /* otherwise, it should be a table */

                    /* skip packet */

                skip:

                    pes->state = MPEGTS_SKIP;

                    continue;

                }

            }

            break;

            /**********************************************/

            /* PES packing parsing */

        case MPEGTS_PESHEADER:

            len = PES_HEADER_SIZE - pes->data_index;

            if (len < 0)

                return -1;

            if (len > buf_size)

                len = buf_size;

            memcpy(pes->header + pes->data_index, p, len);

            pes->data_index += len;

            p += len;

            buf_size -= len;

            if (pes->data_index == PES_HEADER_SIZE) {

                pes->pes_header_size = pes->header[8] + 9;

                pes->state = MPEGTS_PESHEADER_FILL;

            }

            break;

        case MPEGTS_PESHEADER_FILL:

            len = pes->pes_header_size - pes->data_index;

            if (len < 0)

                return -1;

            if (len > buf_size)

                len = buf_size;

            memcpy(pes->header + pes->data_index, p, len);

            pes->data_index += len;

            p += len;

            buf_size -= len;

            if (pes->data_index == pes->pes_header_size) {

                const uint8_t *r;

                unsigned int flags, pes_ext, skip;



                flags = pes->header[7];

                r = pes->header + 9;

                pes->pts = AV_NOPTS_VALUE;

                pes->dts = AV_NOPTS_VALUE;

                if ((flags & 0xc0) == 0x80) {

                    pes->dts = pes->pts = ff_parse_pes_pts(r);

                    r += 5;

                } else if ((flags & 0xc0) == 0xc0) {

                    pes->pts = ff_parse_pes_pts(r);

                    r += 5;

                    pes->dts = ff_parse_pes_pts(r);

                    r += 5;

                }

                pes->extended_stream_id = -1;

                if (flags & 0x01) { /* PES extension */

                    pes_ext = *r++;

                    /* Skip PES private data, program packet sequence counter and P-STD buffer */

                    skip = (pes_ext >> 4) & 0xb;

                    skip += skip & 0x9;

                    r += skip;

                    if ((pes_ext & 0x41) == 0x01 &&

                        (r + 2) <= (pes->header + pes->pes_header_size)) {

                        /* PES extension 2 */

                        if ((r[0] & 0x7f) > 0 && (r[1] & 0x80) == 0)

                            pes->extended_stream_id = r[1];

                    }

                }



                /* we got the full header. We parse it and get the payload */

                pes->state = MPEGTS_PAYLOAD;

                pes->data_index = 0;

            }

            break;

        case MPEGTS_PAYLOAD:

            if (buf_size > 0 && pes->buffer) {

                if (pes->data_index > 0 && pes->data_index+buf_size > pes->total_size) {

                    new_pes_packet(pes, ts->pkt);

                    pes->total_size = MAX_PES_PAYLOAD;

                    pes->buffer = av_malloc(pes->total_size+FF_INPUT_BUFFER_PADDING_SIZE);

                    if (!pes->buffer)

                        return AVERROR(ENOMEM);

                    ts->stop_parse = 1;

                } else if (pes->data_index == 0 && buf_size > pes->total_size) {

                    // pes packet size is < ts size packet and pes data is padded with 0xff

                    // not sure if this is legal in ts but see issue #2392

                    buf_size = pes->total_size;

                    pes->flags |= AV_PKT_FLAG_CORRUPT;

                }

                memcpy(pes->buffer+pes->data_index, p, buf_size);

                pes->data_index += buf_size;

            }

            buf_size = 0;

            /* emit complete packets with known packet size

             * decreases demuxer delay for infrequent packets like subtitles from

             * a couple of seconds to milliseconds for properly muxed files.

             * total_size is the number of bytes following pes_packet_length

             * in the pes header, i.e. not counting the first 6 bytes */

            if (!ts->stop_parse && pes->total_size < MAX_PES_PAYLOAD &&

                pes->pes_header_size + pes->data_index == pes->total_size + 6) {

                ts->stop_parse = 1;

                new_pes_packet(pes, ts->pkt);

            }

            break;

        case MPEGTS_SKIP:

            buf_size = 0;

            break;

        }

    }



    return 0;

}
