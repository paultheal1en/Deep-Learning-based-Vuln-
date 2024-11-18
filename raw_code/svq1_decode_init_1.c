static int svq1_decode_init(AVCodecContext *avctx)

{

    MpegEncContext *s = avctx->priv_data;

    int i;



    MPV_decode_defaults(s);



    s->avctx = avctx;

    s->width = (avctx->width+3)&~3;

    s->height = (avctx->height+3)&~3;

    s->codec_id= avctx->codec->id;

    avctx->pix_fmt = PIX_FMT_YUV410P;

    avctx->has_b_frames= 1; // not true, but DP frames and these behave like unidirectional b frames

    s->flags= avctx->flags;

    if (MPV_common_init(s) < 0) return -1;



    init_vlc(&svq1_block_type, 2, 4,

        &svq1_block_type_vlc[0][1], 2, 1,

        &svq1_block_type_vlc[0][0], 2, 1);



    init_vlc(&svq1_motion_component, 7, 33,

        &mvtab[0][1], 2, 1,

        &mvtab[0][0], 2, 1);



    for (i = 0; i < 6; i++) {

        init_vlc(&svq1_intra_multistage[i], 3, 8,

            &svq1_intra_multistage_vlc[i][0][1], 2, 1,

            &svq1_intra_multistage_vlc[i][0][0], 2, 1);

        init_vlc(&svq1_inter_multistage[i], 3, 8,

            &svq1_inter_multistage_vlc[i][0][1], 2, 1,

            &svq1_inter_multistage_vlc[i][0][0], 2, 1);

    }



    init_vlc(&svq1_intra_mean, 8, 256,

        &svq1_intra_mean_vlc[0][1], 4, 2,

        &svq1_intra_mean_vlc[0][0], 4, 2);



    init_vlc(&svq1_inter_mean, 9, 512,

        &svq1_inter_mean_vlc[0][1], 4, 2,

        &svq1_inter_mean_vlc[0][0], 4, 2);



    return 0;

}
