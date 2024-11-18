static int ff_h261_resync(H261Context *h){

    MpegEncContext * const s = &h->s;

    int left, ret;



    if(show_bits(&s->gb, 15)==0){

        ret= h261_decode_gob_header(h);

        if(ret>=0)

            return 0;

    }

    //ok, its not where its supposed to be ...

    s->gb= s->last_resync_gb;

    align_get_bits(&s->gb);

    left= s->gb.size_in_bits - get_bits_count(&s->gb);



    for(;left>15+1+4+5; left-=8){

        if(show_bits(&s->gb, 15)==0){

            GetBitContext bak= s->gb;



            ret= h261_decode_gob_header(h);

            if(ret>=0)

                return 0;



            s->gb= bak;

        }

        skip_bits(&s->gb, 8);

    }



    return -1;

}
