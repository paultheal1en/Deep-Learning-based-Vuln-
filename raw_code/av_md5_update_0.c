void av_md5_update(AVMD5 *ctx, const uint8_t *src, int len)

{

    const uint8_t *end;

    int j;



    j = ctx->len & 63;

    ctx->len += len;



    if (j) {

        int cnt = FFMIN(len, 64 - j);

        memcpy(ctx->block + j, src, cnt);

        src += cnt;

        len -= cnt;

        if (j + cnt < 64)

            return;

        body(ctx->ABCD, (uint32_t *)ctx->block);

    }



    end = src + (len & ~63);

    if (HAVE_BIGENDIAN || (!HAVE_FAST_UNALIGNED && ((intptr_t)src & 3))) {

       while (src < end) {

           memcpy(ctx->block, src, 64);

           body(ctx->ABCD, (uint32_t *) ctx->block);

           src += 64;

        }

    } else {

        while (src < end) {

            body(ctx->ABCD, (uint32_t *)src);

            src += 64;

        }

    }

    len &= 63;

    if (len > 0)

        memcpy(ctx->block, src, len);

}
