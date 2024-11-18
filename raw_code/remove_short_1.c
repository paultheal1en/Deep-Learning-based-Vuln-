static Picture * remove_short(H264Context *h, int frame_num){

    MpegEncContext * const s = &h->s;

    int i;



    if(s->avctx->debug&FF_DEBUG_MMCO)

        av_log(h->s.avctx, AV_LOG_DEBUG, "remove short %d count %d\n", frame_num, h->short_ref_count);



    for(i=0; i<h->short_ref_count; i++){

        Picture *pic= h->short_ref[i];

        if(s->avctx->debug&FF_DEBUG_MMCO)

            av_log(h->s.avctx, AV_LOG_DEBUG, "%d %d %p\n", i, pic->frame_num, pic);

        if(pic->frame_num == frame_num){

            h->short_ref[i]= NULL;

            memmove(&h->short_ref[i], &h->short_ref[i+1], (h->short_ref_count - i - 1)*sizeof(Picture*));

            h->short_ref_count--;

            return pic;

        }

    }

    return NULL;

}
