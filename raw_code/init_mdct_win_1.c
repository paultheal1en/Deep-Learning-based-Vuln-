static av_cold void init_mdct_win(TwinContext *tctx)

{

    int i,j;

    const ModeTab *mtab = tctx->mtab;

    int size_s = mtab->size / mtab->fmode[FT_SHORT].sub;

    int size_m = mtab->size / mtab->fmode[FT_MEDIUM].sub;

    int channels = tctx->avctx->channels;

    float norm = channels == 1 ? 2. : 1.;



    for (i = 0; i < 3; i++) {

        int bsize = tctx->mtab->size/tctx->mtab->fmode[i].sub;

        ff_mdct_init(&tctx->mdct_ctx[i], av_log2(bsize) + 1, 1,

                     -sqrt(norm/bsize) / (1<<15));

    }



    tctx->tmp_buf  = av_malloc(mtab->size            * sizeof(*tctx->tmp_buf));



    tctx->spectrum  = av_malloc(2*mtab->size*channels*sizeof(float));

    tctx->curr_frame = av_malloc(2*mtab->size*channels*sizeof(float));

    tctx->prev_frame  = av_malloc(2*mtab->size*channels*sizeof(float));



    for (i = 0; i < 3; i++) {

        int m = 4*mtab->size/mtab->fmode[i].sub;

        double freq = 2*M_PI/m;

        tctx->cos_tabs[i] = av_malloc((m/4)*sizeof(*tctx->cos_tabs));



        for (j = 0; j <= m/8; j++)

            tctx->cos_tabs[i][j] = cos((2*j + 1)*freq);

        for (j = 1; j <  m/8; j++)

            tctx->cos_tabs[i][m/4-j] = tctx->cos_tabs[i][j];

    }





    ff_init_ff_sine_windows(av_log2(size_m));

    ff_init_ff_sine_windows(av_log2(size_s/2));

    ff_init_ff_sine_windows(av_log2(mtab->size));

}
