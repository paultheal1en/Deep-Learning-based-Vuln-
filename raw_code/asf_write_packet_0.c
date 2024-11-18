static int asf_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    ASFContext *asf = s->priv_data;

    ASFStream *stream;

    int64_t duration;

    AVCodecContext *codec;

    int64_t packet_st, pts;

    int start_sec, i;

    int flags = pkt->flags;



    codec  = s->streams[pkt->stream_index]->codec;

    stream = &asf->streams[pkt->stream_index];



    if (codec->codec_type == AVMEDIA_TYPE_AUDIO)

        flags &= ~AV_PKT_FLAG_KEY;



    pts = (pkt->pts != AV_NOPTS_VALUE) ? pkt->pts : pkt->dts;



    if (pts < 0) {

        av_log(s, AV_LOG_ERROR,

               "Negative dts not supported stream %d, dts %"PRId64"\n",

               pkt->stream_index, pts);

        return AVERROR(ENOSYS);

    }



    assert(pts != AV_NOPTS_VALUE);

    duration      = pts * 10000;

    asf->duration = FFMAX(asf->duration, duration + pkt->duration * 10000);



    packet_st = asf->nb_packets;

    put_frame(s, stream, s->streams[pkt->stream_index],

              pkt->dts, pkt->data, pkt->size, flags);



    /* check index */

    if ((!asf->is_streamed) && (flags & AV_PKT_FLAG_KEY)) {

        start_sec = (int)(duration / INT64_C(10000000));

        if (start_sec != (int)(asf->last_indexed_pts / INT64_C(10000000))) {

            for (i = asf->nb_index_count; i < start_sec; i++) {

                if (i >= asf->nb_index_memory_alloc) {

                    asf->nb_index_memory_alloc += ASF_INDEX_BLOCK;

                    asf->index_ptr              = (ASFIndex *)av_realloc(asf->index_ptr,

                                                                         sizeof(ASFIndex) *

                                                                         asf->nb_index_memory_alloc);

                }

                // store

                asf->index_ptr[i].packet_number = (uint32_t)packet_st;

                asf->index_ptr[i].packet_count  = (uint16_t)(asf->nb_packets - packet_st);

                asf->maximum_packet             = FFMAX(asf->maximum_packet,

                                                        (uint16_t)(asf->nb_packets - packet_st));

            }

            asf->nb_index_count   = start_sec;

            asf->last_indexed_pts = duration;

        }

    }

    return 0;

}
