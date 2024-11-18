static void flush_packet(AVFormatContext *ctx, int stream_index, 

                         int64_t pts, int64_t dts, int64_t scr)

{

    MpegMuxContext *s = ctx->priv_data;

    StreamInfo *stream = ctx->streams[stream_index]->priv_data;

    uint8_t *buf_ptr;

    int size, payload_size, startcode, id, stuffing_size, i, header_len;

    int packet_size;

    uint8_t buffer[128];

    int zero_trail_bytes = 0;

    int pad_packet_bytes = 0;

    

    id = stream->id;

    

#if 0

    printf("packet ID=%2x PTS=%0.3f\n", 

           id, pts / 90000.0);

#endif



    buf_ptr = buffer;



    if (((s->packet_number % s->pack_header_freq) == 0)) {

        /* output pack and systems header if needed */

        size = put_pack_header(ctx, buf_ptr, scr);

        buf_ptr += size;



        if (s->is_vcd) {

            /* there is exactly one system header for each stream in a VCD MPEG,

               One in the very first video packet and one in the very first

               audio packet (see VCD standard p. IV-7 and IV-8).*/

            

            if (stream->packet_number==0) {

                size = put_system_header(ctx, buf_ptr, id);

                buf_ptr += size;

            }

        } else {

            if ((s->packet_number % s->system_header_freq) == 0) {

                size = put_system_header(ctx, buf_ptr, 0);

                buf_ptr += size;

            }

        }

    }

    size = buf_ptr - buffer;

    put_buffer(&ctx->pb, buffer, size);



    packet_size = s->packet_size - size;



    if (s->is_vcd && id == AUDIO_ID)

        /* The VCD standard demands that 20 zero bytes follow

           each audio pack (see standard p. IV-8).*/

        zero_trail_bytes += 20;

            

    if (s->is_vcd && stream->packet_number==0) {

        /* the first pack of each stream contains only the pack header,

           the system header and lots of padding (see VCD standard p. IV-6).

           In the case of an audio pack, 20 zero bytes are also added at

           the end.*/

        pad_packet_bytes = packet_size - zero_trail_bytes;

    }



    packet_size -= pad_packet_bytes + zero_trail_bytes;



    if (packet_size > 0) {



        /* packet header size */

        packet_size -= 6;

        

        /* packet header */

        if (s->is_mpeg2) {

            header_len = 3;

        } else {

            header_len = 0;

        }

        if (pts != AV_NOPTS_VALUE) {

            if (dts != pts)

                header_len += 5 + 5;

            else

                header_len += 5;

        } else {

            if (!s->is_mpeg2)

                header_len++;

        }



        payload_size = packet_size - header_len;

        if (id < 0xc0) {

            startcode = PRIVATE_STREAM_1;

            payload_size -= 4;

            if (id >= 0xa0)

                payload_size -= 3;

        } else {

            startcode = 0x100 + id;

        }



        stuffing_size = payload_size - stream->buffer_ptr;

        if (stuffing_size < 0)

            stuffing_size = 0;

        put_be32(&ctx->pb, startcode);



        put_be16(&ctx->pb, packet_size);

        

        if (!s->is_mpeg2)

            for(i=0;i<stuffing_size;i++)

                put_byte(&ctx->pb, 0xff);



        if (s->is_mpeg2) {

            put_byte(&ctx->pb, 0x80); /* mpeg2 id */



            if (pts != AV_NOPTS_VALUE) {

                if (dts != pts) {

                    put_byte(&ctx->pb, 0xc0); /* flags */

                    put_byte(&ctx->pb, header_len - 3 + stuffing_size);

                    put_timestamp(&ctx->pb, 0x03, pts);

                    put_timestamp(&ctx->pb, 0x01, dts);

                } else {

                    put_byte(&ctx->pb, 0x80); /* flags */

                    put_byte(&ctx->pb, header_len - 3 + stuffing_size);

                    put_timestamp(&ctx->pb, 0x02, pts);

                }

            } else {

                put_byte(&ctx->pb, 0x00); /* flags */

                put_byte(&ctx->pb, header_len - 3 + stuffing_size);

            }

        } else {

            if (pts != AV_NOPTS_VALUE) {

                if (dts != pts) {

                    put_timestamp(&ctx->pb, 0x03, pts);

                    put_timestamp(&ctx->pb, 0x01, dts);

                } else {

                    put_timestamp(&ctx->pb, 0x02, pts);

                }

            } else {

                put_byte(&ctx->pb, 0x0f);

            }

        }



        if (startcode == PRIVATE_STREAM_1) {

            put_byte(&ctx->pb, id);

            if (id >= 0xa0) {

                /* LPCM (XXX: check nb_frames) */

                put_byte(&ctx->pb, 7);

                put_be16(&ctx->pb, 4); /* skip 3 header bytes */

                put_byte(&ctx->pb, stream->lpcm_header[0]);

                put_byte(&ctx->pb, stream->lpcm_header[1]);

                put_byte(&ctx->pb, stream->lpcm_header[2]);

            } else {

                /* AC3 */

                put_byte(&ctx->pb, stream->nb_frames);

                put_be16(&ctx->pb, stream->frame_start_offset);

            }

        }



        if (s->is_mpeg2)

            for(i=0;i<stuffing_size;i++)

                put_byte(&ctx->pb, 0xff);



        /* output data */

        put_buffer(&ctx->pb, stream->buffer, payload_size - stuffing_size);

    }



    if (pad_packet_bytes > 0)

        put_padding_packet(ctx,&ctx->pb, pad_packet_bytes);    



    for(i=0;i<zero_trail_bytes;i++)

        put_byte(&ctx->pb, 0x00);

        

    put_flush_packet(&ctx->pb);

    

    s->packet_number++;

    stream->packet_number++;

    stream->nb_frames = 0;

    stream->frame_start_offset = 0;

}
