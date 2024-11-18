static void nal_send(AVFormatContext *ctx, const uint8_t *buf, int len, int last_packet_of_frame)

{

    RTPMuxContext *rtp_ctx = ctx->priv_data;

    int rtp_payload_size   = rtp_ctx->max_payload_size - RTP_HEVC_HEADERS_SIZE;

    int nal_type           = (buf[0] >> 1) & 0x3F;



    /* send it as one single NAL unit? */

    if (len <= rtp_ctx->max_payload_size) {

        int buffered_size = rtp_ctx->buf_ptr - rtp_ctx->buf;

        /* Flush buffered NAL units if the current unit doesn't fit */

        if (buffered_size + 2 + len > rtp_ctx->max_payload_size) {

            flush_buffered(ctx, 0);

            buffered_size = 0;

        }

        /* If the NAL unit fits including the framing, write the unit

         * to the buffer as an aggregate packet, otherwise flush and

         * send as single NAL. */

        if (buffered_size + 4 + len <= rtp_ctx->max_payload_size) {

            if (buffered_size == 0) {

                *rtp_ctx->buf_ptr++ = 48 << 1;

                *rtp_ctx->buf_ptr++ = 1;

            }

            AV_WB16(rtp_ctx->buf_ptr, len);

            rtp_ctx->buf_ptr += 2;

            memcpy(rtp_ctx->buf_ptr, buf, len);

            rtp_ctx->buf_ptr += len;

            rtp_ctx->buffered_nals++;

        } else {

            flush_buffered(ctx, 0);

            ff_rtp_send_data(ctx, buf, len, last_packet_of_frame);

        }

    } else {

        flush_buffered(ctx, 0);

        /*

          create the HEVC payload header and transmit the buffer as fragmentation units (FU)



             0                   1

             0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5

            +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

            |F|   Type    |  LayerId  | TID |

            +-------------+-----------------+



               F       = 0

               Type    = 49 (fragmentation unit (FU))

               LayerId = 0

               TID     = 1

         */

        rtp_ctx->buf[0] = 49 << 1;

        rtp_ctx->buf[1] = 1;



        /*

              create the FU header



              0 1 2 3 4 5 6 7

             +-+-+-+-+-+-+-+-+

             |S|E|  FuType   |

             +---------------+



                S       = variable

                E       = variable

                FuType  = NAL unit type

         */

        rtp_ctx->buf[2]  = nal_type;

        /* set the S bit: mark as start fragment */

        rtp_ctx->buf[2] |= 1 << 7;



        /* pass the original NAL header */

        buf += 2;

        len -= 2;



        while (len > rtp_payload_size) {

            /* complete and send current RTP packet */

            memcpy(&rtp_ctx->buf[RTP_HEVC_HEADERS_SIZE], buf, rtp_payload_size);

            ff_rtp_send_data(ctx, rtp_ctx->buf, rtp_ctx->max_payload_size, 0);



            buf += rtp_payload_size;

            len -= rtp_payload_size;



            /* reset the S bit */

            rtp_ctx->buf[2] &= ~(1 << 7);

        }



        /* set the E bit: mark as last fragment */

        rtp_ctx->buf[2] |= 1 << 6;



        /* complete and send last RTP packet */

        memcpy(&rtp_ctx->buf[RTP_HEVC_HEADERS_SIZE], buf, len);

        ff_rtp_send_data(ctx, rtp_ctx->buf, len + 2, last_packet_of_frame);

    }

}
