static inline void range_dec_normalize(APEContext * ctx)

{

    while (ctx->rc.range <= BOTTOM_VALUE) {

        ctx->rc.buffer = (ctx->rc.buffer << 8) | bytestream_get_byte(&ctx->ptr);

        ctx->rc.low    = (ctx->rc.low << 8)    | ((ctx->rc.buffer >> 1) & 0xFF);

        ctx->rc.range  <<= 8;

    }

}
