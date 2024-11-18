static int mpegvideo_parse(AVCodecParserContext *s,

                           AVCodecContext *avctx,

                           const uint8_t **poutbuf, int *poutbuf_size,

                           const uint8_t *buf, int buf_size)

{

    ParseContext1 *pc1 = s->priv_data;

    ParseContext *pc= &pc1->pc;

    int next;



    if(s->flags & PARSER_FLAG_COMPLETE_FRAMES){

        next= buf_size;

    }else{

        next= ff_mpeg1_find_frame_end(pc, buf, buf_size);



        if (ff_combine_frame(pc, next, &buf, &buf_size) < 0) {

            *poutbuf = NULL;

            *poutbuf_size = 0;

            return buf_size;

        }



    }

    /* we have a full frame : we just parse the first few MPEG headers

       to have the full timing information. The time take by this

       function should be negligible for uncorrupted streams */

    mpegvideo_extract_headers(s, avctx, buf, buf_size);

#if 0

    printf("pict_type=%d frame_rate=%0.3f repeat_pict=%d\n",

           s->pict_type, (double)avctx->time_base.den / avctx->time_base.num, s->repeat_pict);

#endif



    *poutbuf = buf;

    *poutbuf_size = buf_size;

    return next;

}
