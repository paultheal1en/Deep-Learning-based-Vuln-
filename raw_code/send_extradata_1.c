static int send_extradata(APNGDemuxContext *ctx, AVPacket *pkt)

{

    if (!ctx->extra_data_updated) {

        uint8_t *side_data = av_packet_new_side_data(pkt, AV_PKT_DATA_NEW_EXTRADATA, ctx->extra_data_size);

        if (!side_data)

            return AVERROR(ENOMEM);

        memcpy(side_data, ctx->extra_data, ctx->extra_data_size);

        ctx->extra_data_updated = 1;

    }

    return 0;

}
