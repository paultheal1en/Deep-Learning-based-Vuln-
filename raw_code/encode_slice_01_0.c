static int encode_slice(AVCodecContext *avctx, const AVFrame *pic,

                        PutBitContext *pb,

                        int sizes[4], int x, int y, int quant,

                        int mbs_per_slice)

{

    ProresContext *ctx = avctx->priv_data;

    int i, xp, yp;

    int total_size = 0;

    const uint16_t *src;

    int slice_width_factor = av_log2(mbs_per_slice);

    int num_cblocks, pwidth;

    int plane_factor, is_chroma;



    for (i = 0; i < ctx->num_planes; i++) {

        is_chroma    = (i == 1 || i == 2);

        plane_factor = slice_width_factor + 2;

        if (is_chroma)

            plane_factor += ctx->chroma_factor - 3;

        if (!is_chroma || ctx->chroma_factor == CFACTOR_Y444) {

            xp          = x << 4;

            yp          = y << 4;

            num_cblocks = 4;

            pwidth      = avctx->width;

        } else {

            xp          = x << 3;

            yp          = y << 4;

            num_cblocks = 2;

            pwidth      = avctx->width >> 1;

        }

        src = (const uint16_t*)(pic->data[i] + yp * pic->linesize[i]) + xp;



        get_slice_data(ctx, src, pic->linesize[i], xp, yp,

                       pwidth, avctx->height, ctx->blocks[0],

                       mbs_per_slice, num_cblocks);

        sizes[i] = encode_slice_plane(ctx, pb, src, pic->linesize[i],

                                      mbs_per_slice, ctx->blocks[0],

                                      num_cblocks, plane_factor,

                                      ctx->quants[quant]);

        total_size += sizes[i];

    }

    return total_size;

}
