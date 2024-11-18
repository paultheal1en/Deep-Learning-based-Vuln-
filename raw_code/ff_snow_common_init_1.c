av_cold int ff_snow_common_init(AVCodecContext *avctx){

    SnowContext *s = avctx->priv_data;

    int width, height;

    int i, j;



    s->avctx= avctx;

    s->max_ref_frames=1; //just make sure it's not an invalid value in case of no initial keyframe




    ff_me_cmp_init(&s->mecc, avctx);

    ff_hpeldsp_init(&s->hdsp, avctx->flags);

    ff_videodsp_init(&s->vdsp, 8);

    ff_dwt_init(&s->dwt);

    ff_h264qpel_init(&s->h264qpel, 8);



#define mcf(dx,dy)\

    s->qdsp.put_qpel_pixels_tab       [0][dy+dx/4]=\

    s->qdsp.put_no_rnd_qpel_pixels_tab[0][dy+dx/4]=\

        s->h264qpel.put_h264_qpel_pixels_tab[0][dy+dx/4];\

    s->qdsp.put_qpel_pixels_tab       [1][dy+dx/4]=\

    s->qdsp.put_no_rnd_qpel_pixels_tab[1][dy+dx/4]=\

        s->h264qpel.put_h264_qpel_pixels_tab[1][dy+dx/4];



    mcf( 0, 0)

    mcf( 4, 0)

    mcf( 8, 0)

    mcf(12, 0)

    mcf( 0, 4)

    mcf( 4, 4)

    mcf( 8, 4)

    mcf(12, 4)

    mcf( 0, 8)

    mcf( 4, 8)

    mcf( 8, 8)

    mcf(12, 8)

    mcf( 0,12)

    mcf( 4,12)

    mcf( 8,12)

    mcf(12,12)



#define mcfh(dx,dy)\

    s->hdsp.put_pixels_tab       [0][dy/4+dx/8]=\

    s->hdsp.put_no_rnd_pixels_tab[0][dy/4+dx/8]=\

        mc_block_hpel ## dx ## dy ## 16;\

    s->hdsp.put_pixels_tab       [1][dy/4+dx/8]=\

    s->hdsp.put_no_rnd_pixels_tab[1][dy/4+dx/8]=\

        mc_block_hpel ## dx ## dy ## 8;



    mcfh(0, 0)

    mcfh(8, 0)

    mcfh(0, 8)

    mcfh(8, 8)



    init_qexp();



//    dec += FFMAX(s->chroma_h_shift, s->chroma_v_shift);



    width= s->avctx->width;

    height= s->avctx->height;



    FF_ALLOCZ_ARRAY_OR_GOTO(avctx, s->spatial_idwt_buffer, width, height * sizeof(IDWTELEM), fail);

    FF_ALLOCZ_ARRAY_OR_GOTO(avctx, s->spatial_dwt_buffer,  width, height * sizeof(DWTELEM),  fail); //FIXME this does not belong here

    FF_ALLOCZ_ARRAY_OR_GOTO(avctx, s->temp_dwt_buffer,     width, sizeof(DWTELEM),  fail);

    FF_ALLOCZ_ARRAY_OR_GOTO(avctx, s->temp_idwt_buffer,    width, sizeof(IDWTELEM), fail);

    FF_ALLOC_ARRAY_OR_GOTO(avctx,  s->run_buffer,          ((width + 1) >> 1), ((height + 1) >> 1) * sizeof(*s->run_buffer), fail);



    for(i=0; i<MAX_REF_FRAMES; i++) {

        for(j=0; j<MAX_REF_FRAMES; j++)

            ff_scale_mv_ref[i][j] = 256*(i+1)/(j+1);

        s->last_picture[i] = av_frame_alloc();

        if (!s->last_picture[i])

            goto fail;

    }



    s->mconly_picture = av_frame_alloc();

    s->current_picture = av_frame_alloc();

    if (!s->mconly_picture || !s->current_picture)

        goto fail;



    return 0;

fail:

    return AVERROR(ENOMEM);

}