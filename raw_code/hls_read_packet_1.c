static int hls_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    HLSContext *c = s->priv_data;

    int ret, i, minvariant = -1;



    if (c->first_packet) {

        recheck_discard_flags(s, 1);

        c->first_packet = 0;

    }



start:

    c->end_of_segment = 0;

    for (i = 0; i < c->n_variants; i++) {

        struct variant *var = c->variants[i];

        /* Make sure we've got one buffered packet from each open variant

         * stream */

        if (var->needed && !var->pkt.data) {

            while (1) {

                int64_t ts_diff;

                AVStream *st;

                ret = av_read_frame(var->ctx, &var->pkt);

                if (ret < 0) {

                    if (!var->pb.eof_reached)

                        return ret;


                    break;

                } else {

                    if (c->first_timestamp == AV_NOPTS_VALUE &&

                        var->pkt.dts       != AV_NOPTS_VALUE)

                        c->first_timestamp = av_rescale_q(var->pkt.dts,

                            var->ctx->streams[var->pkt.stream_index]->time_base,

                            AV_TIME_BASE_Q);

                }



                if (c->seek_timestamp == AV_NOPTS_VALUE)

                    break;



                if (var->pkt.dts == AV_NOPTS_VALUE) {

                    c->seek_timestamp = AV_NOPTS_VALUE;

                    break;

                }



                st = var->ctx->streams[var->pkt.stream_index];

                ts_diff = av_rescale_rnd(var->pkt.dts, AV_TIME_BASE,

                                         st->time_base.den, AV_ROUND_DOWN) -

                          c->seek_timestamp;

                if (ts_diff >= 0 && (c->seek_flags  & AVSEEK_FLAG_ANY ||

                                     var->pkt.flags & AV_PKT_FLAG_KEY)) {

                    c->seek_timestamp = AV_NOPTS_VALUE;

                    break;

                }



            }

        }

        /* Check if this stream still is on an earlier segment number, or

         * has the packet with the lowest dts */

        if (var->pkt.data) {

            struct variant *minvar = c->variants[minvariant];

            if (minvariant < 0 || var->cur_seq_no < minvar->cur_seq_no) {

                minvariant = i;

            } else if (var->cur_seq_no == minvar->cur_seq_no) {

                int64_t dts     =    var->pkt.dts;

                int64_t mindts  = minvar->pkt.dts;

                AVStream *st    =    var->ctx->streams[var->pkt.stream_index];

                AVStream *minst = minvar->ctx->streams[minvar->pkt.stream_index];



                if (dts == AV_NOPTS_VALUE) {

                    minvariant = i;

                } else if (mindts != AV_NOPTS_VALUE) {

                    if (st->start_time    != AV_NOPTS_VALUE)

                        dts    -= st->start_time;

                    if (minst->start_time != AV_NOPTS_VALUE)

                        mindts -= minst->start_time;



                    if (av_compare_ts(dts, st->time_base,

                                      mindts, minst->time_base) < 0)

                        minvariant = i;

                }

            }

        }

    }

    if (c->end_of_segment) {

        if (recheck_discard_flags(s, 0))

            goto start;

    }

    /* If we got a packet, return it */

    if (minvariant >= 0) {

        *pkt = c->variants[minvariant]->pkt;

        pkt->stream_index += c->variants[minvariant]->stream_offset;

        reset_packet(&c->variants[minvariant]->pkt);

        return 0;

    }

    return AVERROR_EOF;

}