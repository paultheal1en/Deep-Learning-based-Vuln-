int ff_interleave_add_packet(AVFormatContext *s, AVPacket *pkt,

                             int (*compare)(AVFormatContext *, AVPacket *, AVPacket *))

{

    int ret;

    AVPacketList **next_point, *this_pktl;

    AVStream *st   = s->streams[pkt->stream_index];

    int chunked    = s->max_chunk_size || s->max_chunk_duration;



    this_pktl      = av_mallocz(sizeof(AVPacketList));

    if (!this_pktl)

        return AVERROR(ENOMEM);

    if ((pkt->flags & AV_PKT_FLAG_UNCODED_FRAME)) {

        av_assert0(pkt->size == UNCODED_FRAME_PACKET_SIZE);

        av_assert0(((AVFrame *)pkt->data)->buf);

    }



    if ((ret = av_packet_ref(&this_pktl->pkt, pkt)) < 0) {

        av_free(this_pktl);

        return ret;

    }



    if (s->streams[pkt->stream_index]->last_in_packet_buffer) {

        next_point = &(st->last_in_packet_buffer->next);

    } else {

        next_point = &s->internal->packet_buffer;

    }



    if (chunked) {

        uint64_t max= av_rescale_q_rnd(s->max_chunk_duration, AV_TIME_BASE_Q, st->time_base, AV_ROUND_UP);

        st->interleaver_chunk_size     += pkt->size;

        st->interleaver_chunk_duration += pkt->duration;

        if (   (s->max_chunk_size && st->interleaver_chunk_size > s->max_chunk_size)

            || (max && st->interleaver_chunk_duration           > max)) {

            st->interleaver_chunk_size      = 0;

            this_pktl->pkt.flags |= CHUNK_START;

            if (max && st->interleaver_chunk_duration > max) {

                int64_t syncoffset = (st->codec->codec_type == AVMEDIA_TYPE_VIDEO)*max/2;

                int64_t syncto = av_rescale(pkt->dts + syncoffset, 1, max)*max - syncoffset;



                st->interleaver_chunk_duration += (pkt->dts - syncto)/8 - max;

            } else

                st->interleaver_chunk_duration = 0;

        }

    }

    if (*next_point) {

        if (chunked && !(this_pktl->pkt.flags & CHUNK_START))

            goto next_non_null;



        if (compare(s, &s->internal->packet_buffer_end->pkt, pkt)) {

            while (   *next_point

                   && ((chunked && !((*next_point)->pkt.flags&CHUNK_START))

                       || !compare(s, &(*next_point)->pkt, pkt)))

                next_point = &(*next_point)->next;

            if (*next_point)

                goto next_non_null;

        } else {

            next_point = &(s->internal->packet_buffer_end->next);

        }

    }

    av_assert1(!*next_point);



    s->internal->packet_buffer_end = this_pktl;

next_non_null:



    this_pktl->next = *next_point;



    s->streams[pkt->stream_index]->last_in_packet_buffer =

        *next_point                                      = this_pktl;



    av_packet_unref(pkt);



    return 0;

}
