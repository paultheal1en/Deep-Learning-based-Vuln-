av_cold void ff_wmv2_common_init(Wmv2Context *w)

{

    MpegEncContext *const s = &w->s;



    ff_blockdsp_init(&s->bdsp, s->avctx);

    ff_wmv2dsp_init(&w->wdsp);

    s->idsp.perm_type = w->wdsp.idct_perm;

    ff_init_scantable_permutation(s->idsp.idct_permutation,

                                  w->wdsp.idct_perm);

    ff_init_scantable(s->idsp.idct_permutation, &w->abt_scantable[0],

                      ff_wmv2_scantableA);

    ff_init_scantable(s->idsp.idct_permutation, &w->abt_scantable[1],

                      ff_wmv2_scantableB);

    ff_init_scantable(s->idsp.idct_permutation, &s->intra_scantable,

                      ff_wmv1_scantable[1]);

    ff_init_scantable(s->idsp.idct_permutation, &s->intra_h_scantable,

                      ff_wmv1_scantable[2]);

    ff_init_scantable(s->idsp.idct_permutation, &s->intra_v_scantable,

                      ff_wmv1_scantable[3]);

    ff_init_scantable(s->idsp.idct_permutation, &s->inter_scantable,

                      ff_wmv1_scantable[0]);

    s->idsp.idct_put = w->wdsp.idct_put;

    s->idsp.idct_add = w->wdsp.idct_add;

    s->idsp.idct     = NULL;

}
