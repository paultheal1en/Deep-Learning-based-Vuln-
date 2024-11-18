static void flush_dpb(AVCodecContext *avctx){

    H264Context *h= avctx->priv_data;

    int i;

    for(i=0; i<16; i++)

        h->delayed_pic[i]= NULL;

    h->delayed_output_pic= NULL;

    idr(h);


}