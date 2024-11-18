static int append_extradata(APNGDemuxContext *ctx, AVIOContext *pb, int len)

{

    int previous_size = ctx->extra_data_size;

    int new_size, ret;

    uint8_t *new_extradata;



    if (previous_size > INT_MAX - len)

        return AVERROR_INVALIDDATA;



    new_size = previous_size + len;

    new_extradata = av_realloc(ctx->extra_data, new_size + AV_INPUT_BUFFER_PADDING_SIZE);

    if (!new_extradata)

        return AVERROR(ENOMEM);

    ctx->extra_data = new_extradata;

    ctx->extra_data_size = new_size;



    if ((ret = avio_read(pb, ctx->extra_data + previous_size, len)) < 0)

        return ret;



    return previous_size;

}
