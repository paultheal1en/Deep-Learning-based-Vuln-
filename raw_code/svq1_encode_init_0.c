static av_cold int svq1_encode_init(AVCodecContext *avctx)

{

    SVQ1EncContext *const s = avctx->priv_data;

    int ret;



    ff_hpeldsp_init(&s->hdsp, avctx->flags);

    ff_me_cmp_init(&s->mecc, avctx);

    ff_mpegvideoencdsp_init(&s->m.mpvencdsp, avctx);



    avctx->coded_frame = av_frame_alloc();

    s->current_picture = av_frame_alloc();

    s->last_picture    = av_frame_alloc();

    if (!avctx->coded_frame || !s->current_picture || !s->last_picture) {

        svq1_encode_end(avctx);

        return AVERROR(ENOMEM);

    }



    s->frame_width  = avctx->width;

    s->frame_height = avctx->height;



    s->y_block_width  = (s->frame_width  + 15) / 16;

    s->y_block_height = (s->frame_height + 15) / 16;



    s->c_block_width  = (s->frame_width  / 4 + 15) / 16;

    s->c_block_height = (s->frame_height / 4 + 15) / 16;



    s->avctx               = avctx;

    s->m.avctx             = avctx;



    if ((ret = ff_mpv_common_init(&s->m)) < 0) {

        svq1_encode_end(avctx);

        return ret;

    }



    s->m.picture_structure = PICT_FRAME;

    s->m.me.temp           =

    s->m.me.scratchpad     = av_mallocz((avctx->width + 64) *

                                        2 * 16 * 2 * sizeof(uint8_t));

    s->m.me.map            = av_mallocz(ME_MAP_SIZE * sizeof(uint32_t));

    s->m.me.score_map      = av_mallocz(ME_MAP_SIZE * sizeof(uint32_t));

    s->mb_type             = av_mallocz((s->y_block_width + 1) *

                                        s->y_block_height * sizeof(int16_t));

    s->dummy               = av_mallocz((s->y_block_width + 1) *

                                        s->y_block_height * sizeof(int32_t));

    s->ssd_int8_vs_int16   = ssd_int8_vs_int16_c;



    if (!s->m.me.temp || !s->m.me.scratchpad || !s->m.me.map ||

        !s->m.me.score_map || !s->mb_type || !s->dummy) {

        svq1_encode_end(avctx);

        return AVERROR(ENOMEM);

    }



    if (ARCH_PPC)

        ff_svq1enc_init_ppc(s);

    if (ARCH_X86)

        ff_svq1enc_init_x86(s);



    ff_h263_encode_init(&s->m); // mv_penalty



    return 0;

}
