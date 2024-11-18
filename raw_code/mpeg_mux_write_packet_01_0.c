static int mpeg_mux_write_packet(AVFormatContext *ctx, AVPacket *pkt)

{

    MpegMuxContext *s = ctx->priv_data;

    int stream_index= pkt->stream_index;

    int size= pkt->size;

    uint8_t *buf= pkt->data;

    AVStream *st = ctx->streams[stream_index];

    StreamInfo *stream = st->priv_data;

    int64_t pts, dts, new_start_pts, new_start_dts;

    int len, avail_size;

    

    //XXX/FIXME this is and always was broken

//    compute_pts_dts(st, &pts, &dts, pkt->pts);



    pts= pkt->pts;

    dts= pkt->dts;



    if(s->is_svcd) {

        /* offset pts and dts slightly into the future to be able

           to do the compatibility fix below.*/

        pts = (pts + 2) & ((1LL << 33) - 1);

        dts = (dts + 2) & ((1LL << 33) - 1);



        if (stream->packet_number == 0 && dts == pts)

            /* For the very first packet we want to force the DTS to be included.

               This increases compatibility with lots of DVD players.

               Since the MPEG-2 standard mandates that DTS is only written when

               it is different from PTS we have to move it slightly into the past.*/

            dts = (dts - 2) & ((1LL << 33) - 1);

    }

    if(s->is_vcd) {

        /* We have to offset the PTS, so that it is consistent with the SCR.

           SCR starts at 36000, but the first two packs contain only padding

           and the first pack from the other stream, respectively, may also have

           been written before.

           So the real data starts at SCR 36000+3*1200. */

        pts = (pts + 36000 + 3600) & ((1LL << 33) - 1);

        dts = (dts + 36000 + 3600) & ((1LL << 33) - 1);

    }

    

#if 0

    update_scr(ctx,stream_index,pts);



    printf("%d: pts=%0.3f dts=%0.3f scr=%0.3f\n", 

           stream_index, 

           pts / 90000.0, 

           dts / 90000.0, 

           s->last_scr / 90000.0);

#endif

    

    /* we assume here that pts != AV_NOPTS_VALUE */

    new_start_pts = stream->start_pts;

    new_start_dts = stream->start_dts;

    

    if (stream->start_pts == AV_NOPTS_VALUE) {

        new_start_pts = pts;

        new_start_dts = dts;

    }

    avail_size = get_packet_payload_size(ctx, stream_index,

                                         new_start_pts, 

                                         new_start_dts);

    if (stream->buffer_ptr >= avail_size) {



        update_scr(ctx,stream_index,stream->start_pts);



        /* unlikely case: outputing the pts or dts increase the packet

           size so that we cannot write the start of the next

           packet. In this case, we must flush the current packet with

           padding.

           Note: this always happens for the first audio and video packet

           in a VCD file, since they do not carry any data.*/

        flush_packet(ctx, stream_index,

                     stream->start_pts, stream->start_dts, s->last_scr);

        stream->buffer_ptr = 0;

    }

    stream->start_pts = new_start_pts;

    stream->start_dts = new_start_dts;

    stream->nb_frames++;

    if (stream->frame_start_offset == 0)

        stream->frame_start_offset = stream->buffer_ptr;

    while (size > 0) {

        avail_size = get_packet_payload_size(ctx, stream_index,

                                             stream->start_pts, 

                                             stream->start_dts);

        len = avail_size - stream->buffer_ptr;

        if (len > size)

            len = size;

        memcpy(stream->buffer + stream->buffer_ptr, buf, len);

        stream->buffer_ptr += len;

        buf += len;

        size -= len;

        if (stream->buffer_ptr >= avail_size) {



            update_scr(ctx,stream_index,stream->start_pts);



            /* if packet full, we send it now */

            flush_packet(ctx, stream_index,

                         stream->start_pts, stream->start_dts, s->last_scr);

            stream->buffer_ptr = 0;



            if (s->is_vcd) {

                /* Write one or more padding sectors, if necessary, to reach

                   the constant overall bitrate.*/

                int vcd_pad_bytes;

            

                while((vcd_pad_bytes = get_vcd_padding_size(ctx,stream->start_pts) ) >= s->packet_size)

                    put_vcd_padding_sector(ctx);

            }



            /* Make sure only the FIRST pes packet for this frame has

               a timestamp */

            stream->start_pts = AV_NOPTS_VALUE;

            stream->start_dts = AV_NOPTS_VALUE;

        }

    }



    return 0;

}
