static int decode_5(SANMVideoContext *ctx)

{

#if HAVE_BIGENDIAN

    uint16_t *frm;

    int npixels;

#endif

    uint8_t *dst = (uint8_t*)ctx->frm0;



    if (rle_decode(ctx, dst, ctx->buf_size))

        return AVERROR_INVALIDDATA;



#if HAVE_BIGENDIAN

    npixels = ctx->npixels;

    frm = ctx->frm0;

    while (npixels--)

        *frm++ = av_bswap16(*frm);

#endif



    return 0;

}
