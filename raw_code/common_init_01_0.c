static av_cold void common_init(H264Context *h){

    MpegEncContext * const s = &h->s;



    s->width = s->avctx->width;

    s->height = s->avctx->height;

    s->codec_id= s->avctx->codec->id;



    ff_h264dsp_init(&h->h264dsp, 8, 1);

    ff_h264_pred_init(&h->hpc, s->codec_id, 8, 1);



    h->dequant_coeff_pps= -1;

    s->unrestricted_mv=1;

    s->decode=1; //FIXME



    dsputil_init(&s->dsp, s->avctx); // needed so that idct permutation is known early



    memset(h->pps.scaling_matrix4, 16, 6*16*sizeof(uint8_t));

    memset(h->pps.scaling_matrix8, 16, 2*64*sizeof(uint8_t));

}
