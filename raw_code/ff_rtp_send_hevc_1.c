void ff_rtp_send_hevc(AVFormatContext *ctx, const uint8_t *frame_buf, int frame_size)

{

    const uint8_t *next_NAL_unit;

    const uint8_t *buf_ptr, *buf_end = frame_buf + frame_size;

    RTPMuxContext *rtp_ctx = ctx->priv_data;



    /* use the default 90 KHz time stamp */

    rtp_ctx->timestamp = rtp_ctx->cur_timestamp;

    rtp_ctx->buf_ptr   = rtp_ctx->buf;



    if (rtp_ctx->nal_length_size)

        buf_ptr = ff_avc_mp4_find_startcode(frame_buf, buf_end, rtp_ctx->nal_length_size) ? frame_buf : buf_end;

    else

        buf_ptr = ff_avc_find_startcode(frame_buf, buf_end);



    /* find all NAL units and send them as separate packets */

    while (buf_ptr < buf_end) {

        if (rtp_ctx->nal_length_size) {

            next_NAL_unit = ff_avc_mp4_find_startcode(buf_ptr, buf_end, rtp_ctx->nal_length_size);

            if (!next_NAL_unit)

                next_NAL_unit = buf_end;



            buf_ptr += rtp_ctx->nal_length_size;

        } else {

            while (!*(buf_ptr++))

                ;

            next_NAL_unit = ff_avc_find_startcode(buf_ptr, buf_end);

        }

        /* send the next NAL unit */

        nal_send(ctx, buf_ptr, next_NAL_unit - buf_ptr, next_NAL_unit == buf_end);



        /* jump to the next NAL unit */

        buf_ptr = next_NAL_unit;

    }

    flush_buffered(ctx, 1);

}
