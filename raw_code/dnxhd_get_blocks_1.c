void dnxhd_get_blocks(DNXHDEncContext *ctx, int mb_x, int mb_y)

{

    const int bs = ctx->block_width_l2;

    const int bw = 1 << bs;

    int dct_y_offset = ctx->dct_y_offset;

    int dct_uv_offset = ctx->dct_uv_offset;

    int linesize = ctx->m.linesize;

    int uvlinesize = ctx->m.uvlinesize;

    const uint8_t *ptr_y = ctx->thread[0]->src[0] +

                           ((mb_y << 4) * ctx->m.linesize) + (mb_x << bs + 1);

    const uint8_t *ptr_u = ctx->thread[0]->src[1] +

                           ((mb_y << 4) * ctx->m.uvlinesize) + (mb_x << bs + ctx->is_444);

    const uint8_t *ptr_v = ctx->thread[0]->src[2] +

                           ((mb_y << 4) * ctx->m.uvlinesize) + (mb_x << bs + ctx->is_444);

    PixblockDSPContext *pdsp = &ctx->m.pdsp;

    VideoDSPContext *vdsp = &ctx->m.vdsp;



    if (ctx->bit_depth != 10 && vdsp->emulated_edge_mc && ((mb_x << 4) + 16 > ctx->m.avctx->width ||

                                                           (mb_y << 4) + 16 > ctx->m.avctx->height)) {

        int y_w = ctx->m.avctx->width  - (mb_x << 4);

        int y_h = ctx->m.avctx->height - (mb_y << 4);

        int uv_w = (y_w + 1) / 2;

        int uv_h = y_h;

        linesize = 16;

        uvlinesize = 8;



        vdsp->emulated_edge_mc(&ctx->edge_buf_y[0], ptr_y,

                               linesize, ctx->m.linesize,

                               linesize, 16,

                               0, 0, y_w, y_h);

        vdsp->emulated_edge_mc(&ctx->edge_buf_uv[0][0], ptr_u,

                               uvlinesize, ctx->m.uvlinesize,

                               uvlinesize, 16,

                               0, 0, uv_w, uv_h);

        vdsp->emulated_edge_mc(&ctx->edge_buf_uv[1][0], ptr_v,

                               uvlinesize, ctx->m.uvlinesize,

                               uvlinesize, 16,

                               0, 0, uv_w, uv_h);



        dct_y_offset =  bw * linesize;

        dct_uv_offset = bw * uvlinesize;

        ptr_y = &ctx->edge_buf_y[0];

        ptr_u = &ctx->edge_buf_uv[0][0];

        ptr_v = &ctx->edge_buf_uv[1][0];

    } else if (ctx->bit_depth == 10 && vdsp->emulated_edge_mc && ((mb_x << 3) + 8 > ctx->m.avctx->width ||

                                                                  (mb_y << 3) + 8 > ctx->m.avctx->height)) {

        int y_w = ctx->m.avctx->width  - (mb_x << 3);

        int y_h = ctx->m.avctx->height - (mb_y << 3);

        int uv_w = ctx->is_444 ? y_w : (y_w + 1) / 2;

        int uv_h = y_h;

        linesize = 16;

        uvlinesize = 8 + 8 * ctx->is_444;



        vdsp->emulated_edge_mc(&ctx->edge_buf_y[0], ptr_y,

                               linesize, ctx->m.linesize,

                               linesize / 2, 16,

                               0, 0, y_w, y_h);

        vdsp->emulated_edge_mc(&ctx->edge_buf_uv[0][0], ptr_u,

                               uvlinesize, ctx->m.uvlinesize,

                               uvlinesize / 2, 16,

                               0, 0, uv_w, uv_h);

        vdsp->emulated_edge_mc(&ctx->edge_buf_uv[1][0], ptr_v,

                               uvlinesize, ctx->m.uvlinesize,

                               uvlinesize / 2, 16,

                               0, 0, uv_w, uv_h);



        dct_y_offset =  bw * linesize;

        dct_uv_offset = bw * uvlinesize;

        ptr_y = &ctx->edge_buf_y[0];

        ptr_u = &ctx->edge_buf_uv[0][0];

        ptr_v = &ctx->edge_buf_uv[1][0];

    }



    if (!ctx->is_444) {

        pdsp->get_pixels(ctx->blocks[0], ptr_y,      linesize);

        pdsp->get_pixels(ctx->blocks[1], ptr_y + bw, linesize);

        pdsp->get_pixels(ctx->blocks[2], ptr_u,      uvlinesize);

        pdsp->get_pixels(ctx->blocks[3], ptr_v,      uvlinesize);



        if (mb_y + 1 == ctx->m.mb_height && ctx->m.avctx->height == 1080) {

            if (ctx->interlaced) {

                ctx->get_pixels_8x4_sym(ctx->blocks[4],

                                        ptr_y + dct_y_offset,

                                        linesize);

                ctx->get_pixels_8x4_sym(ctx->blocks[5],

                                        ptr_y + dct_y_offset + bw,

                                        linesize);

                ctx->get_pixels_8x4_sym(ctx->blocks[6],

                                        ptr_u + dct_uv_offset,

                                        uvlinesize);

                ctx->get_pixels_8x4_sym(ctx->blocks[7],

                                        ptr_v + dct_uv_offset,

                                        uvlinesize);

            } else {

                ctx->bdsp.clear_block(ctx->blocks[4]);

                ctx->bdsp.clear_block(ctx->blocks[5]);

                ctx->bdsp.clear_block(ctx->blocks[6]);

                ctx->bdsp.clear_block(ctx->blocks[7]);

            }

        } else {

            pdsp->get_pixels(ctx->blocks[4],

                             ptr_y + dct_y_offset, linesize);

            pdsp->get_pixels(ctx->blocks[5],

                             ptr_y + dct_y_offset + bw, linesize);

            pdsp->get_pixels(ctx->blocks[6],

                             ptr_u + dct_uv_offset, uvlinesize);

            pdsp->get_pixels(ctx->blocks[7],

                             ptr_v + dct_uv_offset, uvlinesize);

        }

    } else {

        pdsp->get_pixels(ctx->blocks[0], ptr_y,      linesize);

        pdsp->get_pixels(ctx->blocks[1], ptr_y + bw, linesize);

        pdsp->get_pixels(ctx->blocks[6], ptr_y + dct_y_offset, linesize);

        pdsp->get_pixels(ctx->blocks[7], ptr_y + dct_y_offset + bw, linesize);



        pdsp->get_pixels(ctx->blocks[2], ptr_u,      uvlinesize);

        pdsp->get_pixels(ctx->blocks[3], ptr_u + bw, uvlinesize);

        pdsp->get_pixels(ctx->blocks[8], ptr_u + dct_uv_offset, uvlinesize);

        pdsp->get_pixels(ctx->blocks[9], ptr_u + dct_uv_offset + bw, uvlinesize);



        pdsp->get_pixels(ctx->blocks[4], ptr_v,      uvlinesize);

        pdsp->get_pixels(ctx->blocks[5], ptr_v + bw, uvlinesize);

        pdsp->get_pixels(ctx->blocks[10], ptr_v + dct_uv_offset, uvlinesize);

        pdsp->get_pixels(ctx->blocks[11], ptr_v + dct_uv_offset + bw, uvlinesize);

    }

}
