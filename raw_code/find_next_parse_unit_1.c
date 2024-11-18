static SchroBuffer *find_next_parse_unit(SchroParseUnitContext *parse_ctx)

{

    SchroBuffer *enc_buf = NULL;

    int next_pu_offset = 0;

    unsigned char *in_buf;



    if (parse_ctx->buf_size < 13 ||

        parse_ctx->buf[0] != 'B' ||

        parse_ctx->buf[1] != 'B' ||

        parse_ctx->buf[2] != 'C' ||

        parse_ctx->buf[3] != 'D')

        return NULL;



    next_pu_offset = (parse_ctx->buf[5] << 24) +

                     (parse_ctx->buf[6] << 16) +

                     (parse_ctx->buf[7] <<  8) +

                      parse_ctx->buf[8];



    if (next_pu_offset == 0 &&

        SCHRO_PARSE_CODE_IS_END_OF_SEQUENCE(parse_ctx->buf[4]))

        next_pu_offset = 13;



    if (next_pu_offset <= 0 || parse_ctx->buf_size < next_pu_offset)

        return NULL;



    in_buf = av_malloc(next_pu_offset);

    if (!in_buf) {

        av_log(parse_ctx, AV_LOG_ERROR, "Unable to allocate input buffer\n");

        return NULL;

    }



    memcpy(in_buf, parse_ctx->buf, next_pu_offset);

    enc_buf       = schro_buffer_new_with_data(in_buf, next_pu_offset);

    enc_buf->free = libschroedinger_decode_buffer_free;

    enc_buf->priv = in_buf;



    parse_ctx->buf      += next_pu_offset;

    parse_ctx->buf_size -= next_pu_offset;



    return enc_buf;

}
