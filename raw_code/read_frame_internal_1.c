static int read_frame_internal(AVFormatContext *s, AVPacket *pkt)

{

    AVStream *st;

    int len, ret, i;



    av_init_packet(pkt);



    for(;;) {

        /* select current input stream component */

        st = s->cur_st;

        if (st) {

            if (!st->need_parsing || !st->parser) {

                /* no parsing needed: we just output the packet as is */

                /* raw data support */

                *pkt = st->cur_pkt; st->cur_pkt.data= NULL;

                compute_pkt_fields(s, st, NULL, pkt);

                s->cur_st = NULL;

                if ((s->iformat->flags & AVFMT_GENERIC_INDEX) &&

                    (pkt->flags & AV_PKT_FLAG_KEY) && pkt->dts != AV_NOPTS_VALUE) {

                    ff_reduce_index(s, st->index);

                    av_add_index_entry(st, pkt->pos, pkt->dts, 0, 0, AVINDEX_KEYFRAME);

                }

                break;

            } else if (st->cur_len > 0 && st->discard < AVDISCARD_ALL) {

                len = av_parser_parse2(st->parser, st->codec, &pkt->data, &pkt->size,

                                       st->cur_ptr, st->cur_len,

                                       st->cur_pkt.pts, st->cur_pkt.dts,

                                       st->cur_pkt.pos);

                st->cur_pkt.pts = AV_NOPTS_VALUE;

                st->cur_pkt.dts = AV_NOPTS_VALUE;

                /* increment read pointer */

                st->cur_ptr += len;

                st->cur_len -= len;



                /* return packet if any */

                if (pkt->size) {

                got_packet:

                    pkt->duration = 0;

                    pkt->stream_index = st->index;

                    pkt->pts = st->parser->pts;

                    pkt->dts = st->parser->dts;

                    pkt->pos = st->parser->pos;

                    if(pkt->data == st->cur_pkt.data && pkt->size == st->cur_pkt.size){

                        s->cur_st = NULL;

                        pkt->destruct= st->cur_pkt.destruct;

                        st->cur_pkt.destruct= NULL;

                        st->cur_pkt.data    = NULL;

                        assert(st->cur_len == 0);

                    }else{

                        pkt->destruct = NULL;

                    }

                    compute_pkt_fields(s, st, st->parser, pkt);



                    if((s->iformat->flags & AVFMT_GENERIC_INDEX) && pkt->flags & AV_PKT_FLAG_KEY){

                        int64_t pos= (st->parser->flags & PARSER_FLAG_COMPLETE_FRAMES) ? pkt->pos : st->parser->frame_offset;

                        ff_reduce_index(s, st->index);

                        av_add_index_entry(st, pos, pkt->dts,

                                           0, 0, AVINDEX_KEYFRAME);

                    }



                    break;

                }

            } else {

                /* free packet */

                av_free_packet(&st->cur_pkt);

                s->cur_st = NULL;

            }

        } else {

            AVPacket cur_pkt;

            /* read next packet */

            ret = av_read_packet(s, &cur_pkt);

            if (ret < 0) {

                if (ret == AVERROR(EAGAIN))

                    return ret;

                /* return the last frames, if any */

                for(i = 0; i < s->nb_streams; i++) {

                    st = s->streams[i];

                    if (st->parser && st->need_parsing) {

                        av_parser_parse2(st->parser, st->codec,

                                        &pkt->data, &pkt->size,

                                        NULL, 0,

                                        AV_NOPTS_VALUE, AV_NOPTS_VALUE,

                                        AV_NOPTS_VALUE);

                        if (pkt->size)

                            goto got_packet;

                    }

                }

                /* no more packets: really terminate parsing */

                return ret;

            }

            st = s->streams[cur_pkt.stream_index];

            st->cur_pkt= cur_pkt;



            if(st->cur_pkt.pts != AV_NOPTS_VALUE &&

               st->cur_pkt.dts != AV_NOPTS_VALUE &&

               st->cur_pkt.pts < st->cur_pkt.dts){

                av_log(s, AV_LOG_WARNING, "Invalid timestamps stream=%d, pts=%"PRId64", dts=%"PRId64", size=%d\n",

                    st->cur_pkt.stream_index,

                    st->cur_pkt.pts,

                    st->cur_pkt.dts,

                    st->cur_pkt.size);

//                av_free_packet(&st->cur_pkt);

//                return -1;

            }



            if(s->debug & FF_FDEBUG_TS)

                av_log(s, AV_LOG_DEBUG, "av_read_packet stream=%d, pts=%"PRId64", dts=%"PRId64", size=%d, duration=%d, flags=%d\n",

                    st->cur_pkt.stream_index,

                    st->cur_pkt.pts,

                    st->cur_pkt.dts,

                    st->cur_pkt.size,

                    st->cur_pkt.duration,

                    st->cur_pkt.flags);



            s->cur_st = st;

            st->cur_ptr = st->cur_pkt.data;

            st->cur_len = st->cur_pkt.size;

            if (st->need_parsing && !st->parser && !(s->flags & AVFMT_FLAG_NOPARSE)) {

                st->parser = av_parser_init(st->codec->codec_id);

                if (!st->parser) {

                    av_log(s, AV_LOG_WARNING, "parser not found for codec "

                           "%s, packets or times may be invalid.\n",

                           avcodec_get_name(st->codec->codec_id));

                    /* no parser available: just output the raw packets */

                    st->need_parsing = AVSTREAM_PARSE_NONE;

                }else if(st->need_parsing == AVSTREAM_PARSE_HEADERS){

                    st->parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;

                }else if(st->need_parsing == AVSTREAM_PARSE_FULL_ONCE){

                    st->parser->flags |= PARSER_FLAG_ONCE;

                }

            }

        }

    }

    if(s->debug & FF_FDEBUG_TS)

        av_log(s, AV_LOG_DEBUG, "read_frame_internal stream=%d, pts=%"PRId64", dts=%"PRId64", size=%d, duration=%d, flags=%d\n",

            pkt->stream_index,

            pkt->pts,

            pkt->dts,

            pkt->size,

            pkt->duration,

            pkt->flags);



    return 0;

}
