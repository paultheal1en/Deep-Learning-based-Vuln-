static int tm2_read_stream(TM2Context *ctx, const uint8_t *buf, int stream_id, int buf_size)

{

    int i;

    int skip = 0;

    int len, toks, pos;

    TM2Codes codes;

    GetByteContext gb;



    if (buf_size < 4) {

        av_log(ctx->avctx, AV_LOG_ERROR, "not enough space for len left\n");

        return AVERROR_INVALIDDATA;

    }



    /* get stream length in dwords */

    bytestream2_init(&gb, buf, buf_size);

    len  = bytestream2_get_be32(&gb);

    skip = len * 4 + 4;



    if(len == 0)

        return 4;



    if (len >= INT_MAX/4-1 || len < 0 || skip > buf_size) {

        av_log(ctx->avctx, AV_LOG_ERROR, "invalid stream size\n");

        return AVERROR_INVALIDDATA;

    }



    toks = bytestream2_get_be32(&gb);

    if(toks & 1) {

        len = bytestream2_get_be32(&gb);

        if(len == TM2_ESCAPE) {

            len = bytestream2_get_be32(&gb);

        }

        if(len > 0) {

            pos = bytestream2_tell(&gb);

            if (skip <= pos)

                return AVERROR_INVALIDDATA;

            init_get_bits(&ctx->gb, buf + pos, (skip - pos) * 8);

            if(tm2_read_deltas(ctx, stream_id) == -1)

                return AVERROR_INVALIDDATA;

            bytestream2_skip(&gb, ((get_bits_count(&ctx->gb) + 31) >> 5) << 2);

        }

    }

    /* skip unused fields */

    len = bytestream2_get_be32(&gb);

    if(len == TM2_ESCAPE) { /* some unknown length - could be escaped too */

        bytestream2_skip(&gb, 8); /* unused by decoder */

    } else {

        bytestream2_skip(&gb, 4); /* unused by decoder */

    }



    pos = bytestream2_tell(&gb);

    if (skip <= pos)

        return AVERROR_INVALIDDATA;

    init_get_bits(&ctx->gb, buf + pos, (skip - pos) * 8);

    if(tm2_build_huff_table(ctx, &codes) == -1)

        return AVERROR_INVALIDDATA;

    bytestream2_skip(&gb, ((get_bits_count(&ctx->gb) + 31) >> 5) << 2);



    toks >>= 1;

    /* check if we have sane number of tokens */

    if((toks < 0) || (toks > 0xFFFFFF)){

        av_log(ctx->avctx, AV_LOG_ERROR, "Incorrect number of tokens: %i\n", toks);

        tm2_free_codes(&codes);

        return AVERROR_INVALIDDATA;

    }

    ctx->tokens[stream_id] = av_realloc(ctx->tokens[stream_id], toks * sizeof(int));

    ctx->tok_lens[stream_id] = toks;

    len = bytestream2_get_be32(&gb);

    if(len > 0) {

        pos = bytestream2_tell(&gb);

        if (skip <= pos)

            return AVERROR_INVALIDDATA;

        init_get_bits(&ctx->gb, buf + pos, (skip - pos) * 8);

        for(i = 0; i < toks; i++) {

            if (get_bits_left(&ctx->gb) <= 0) {

                av_log(ctx->avctx, AV_LOG_ERROR, "Incorrect number of tokens: %i\n", toks);

                return AVERROR_INVALIDDATA;

            }

            ctx->tokens[stream_id][i] = tm2_get_token(&ctx->gb, &codes);

            if (stream_id <= TM2_MOT && ctx->tokens[stream_id][i] >= TM2_DELTAS) {

                av_log(ctx->avctx, AV_LOG_ERROR, "Invalid delta token index %d for type %d, n=%d\n",

                       ctx->tokens[stream_id][i], stream_id, i);

                return AVERROR_INVALIDDATA;

            }

        }

    } else {

        for(i = 0; i < toks; i++) {

            ctx->tokens[stream_id][i] = codes.recode[0];

            if (stream_id <= TM2_MOT && ctx->tokens[stream_id][i] >= TM2_DELTAS) {

                av_log(ctx->avctx, AV_LOG_ERROR, "Invalid delta token index %d for type %d, n=%d\n",

                       ctx->tokens[stream_id][i], stream_id, i);

                return AVERROR_INVALIDDATA;

            }

        }

    }

    tm2_free_codes(&codes);



    return skip;

}
