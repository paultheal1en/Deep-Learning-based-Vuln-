int ff_wmv2_decode_picture_header(MpegEncContext * s)

{

    Wmv2Context * const w= (Wmv2Context*)s;

    int code;



#if 0

{

int i;

for(i=0; i<s->gb.size*8; i++)

    printf("%d", get_bits1(&s->gb));

//    get_bits1(&s->gb);

printf("END\n");

return -1;

}

#endif

    if(s->picture_number==0)

        decode_ext_header(w);



    s->pict_type = get_bits1(&s->gb) + 1;

    if(s->pict_type == I_TYPE){

        code = get_bits(&s->gb, 7);

        av_log(s->avctx, AV_LOG_DEBUG, "I7:%X/\n", code);

    }

    s->chroma_qscale= s->qscale = get_bits(&s->gb, 5);

    if(s->qscale < 0)

       return -1;



    return 0;

}
