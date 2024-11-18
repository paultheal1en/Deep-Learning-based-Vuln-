static inline int mpeg4_is_resync(MpegEncContext *s){

    const int bits_count= get_bits_count(&s->gb);

    

    if(s->workaround_bugs&FF_BUG_NO_PADDING){

        return 0;

    }



    if(bits_count + 8 >= s->gb.size*8){

        int v= show_bits(&s->gb, 8);

        v|= 0x7F >> (7-(bits_count&7));

                

        if(v==0x7F)

            return 1;

    }else{

        if(show_bits(&s->gb, 16) == ff_mpeg4_resync_prefix[bits_count&7]){

            int len;

            GetBitContext gb= s->gb;

        

            skip_bits(&s->gb, 1);

            align_get_bits(&s->gb);

        

            for(len=0; len<32; len++){

                if(get_bits1(&s->gb)) break;

            }



            s->gb= gb;



            if(len>=ff_mpeg4_get_video_packet_prefix_length(s))

                return 1;

        }

    }

    return 0;

}
