av_cold int ff_intrax8_common_init(IntraX8Context *w, IDCTDSPContext *idsp,

                                   MpegEncContext *const s)

{

    int ret = x8_vlc_init();

    if (ret < 0)

        return ret;



    w->idsp = *idsp;

    w->s = s;



    // two rows, 2 blocks per cannon mb

    w->prediction_table = av_mallocz(s->mb_width * 2 * 2);

    if (!w->prediction_table)

        return AVERROR(ENOMEM);



    ff_init_scantable(w->idsp.idct_permutation, &w->scantable[0],

                      ff_wmv1_scantable[0]);

    ff_init_scantable(w->idsp.idct_permutation, &w->scantable[1],

                      ff_wmv1_scantable[2]);

    ff_init_scantable(w->idsp.idct_permutation, &w->scantable[2],

                      ff_wmv1_scantable[3]);



    ff_intrax8dsp_init(&w->dsp);



    return 0;

}
