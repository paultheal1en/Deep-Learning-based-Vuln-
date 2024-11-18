static int fic_decode_slice(AVCodecContext *avctx, void *tdata)

{

    FICContext *ctx        = avctx->priv_data;

    FICThreadContext *tctx = tdata;

    GetBitContext gb;

    uint8_t *src = tctx->src;

    int slice_h  = tctx->slice_h;

    int src_size = tctx->src_size;

    int y_off    = tctx->y_off;

    int x, y, p;



    init_get_bits(&gb, src, src_size * 8);



    for (p = 0; p < 3; p++) {

        int stride   = ctx->frame->linesize[p];

        uint8_t* dst = ctx->frame->data[p] + (y_off >> !!p) * stride;



        for (y = 0; y < (slice_h >> !!p); y += 8) {

            for (x = 0; x < (ctx->aligned_width >> !!p); x += 8) {

                int ret;



                if ((ret = fic_decode_block(ctx, &gb, dst + x, stride, tctx->block)) != 0)

                    return ret;

            }



            dst += 8 * stride;

        }

    }



    return 0;

}
