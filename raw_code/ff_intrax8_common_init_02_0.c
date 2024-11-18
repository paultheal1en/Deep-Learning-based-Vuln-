av_cold int ff_intrax8_common_init(AVCodecContext *avctx,

                                   IntraX8Context *w, IDCTDSPContext *idsp,

                                   int16_t (*block)[64],

                                   int block_last_index[12],

                                   int mb_width, int mb_height)

{

    int ret = x8_vlc_init();

    if (ret < 0)

        return ret;



    w->avctx = avctx;

    w->idsp = *idsp;

    w->mb_width  = mb_width;

    w->mb_height = mb_height;

    w->block = block;

    w->block_last_index = block_last_index;



    // two rows, 2 blocks per cannon mb

    w->prediction_table = av_mallocz(w->mb_width * 2 * 2);

    if (!w->prediction_table)

        return AVERROR(ENOMEM);



    ff_init_scantable(w->idsp.idct_permutation, &w->scantable[0],

                      ff_wmv1_scantable[0]);

    ff_init_scantable(w->idsp.idct_permutation, &w->scantable[1],

                      ff_wmv1_scantable[2]);

    ff_init_scantable(w->idsp.idct_permutation, &w->scantable[2],

                      ff_wmv1_scantable[3]);



    ff_intrax8dsp_init(&w->dsp);

    ff_blockdsp_init(&w->bdsp, avctx);



    return 0;

}
