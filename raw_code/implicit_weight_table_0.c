static void implicit_weight_table(H264Context *h){

    MpegEncContext * const s = &h->s;

    int ref0, ref1, i;

    int cur_poc = s->current_picture_ptr->poc;



    if(   h->ref_count[0] == 1 && h->ref_count[1] == 1

       && h->ref_list[0][0].poc + h->ref_list[1][0].poc == 2*cur_poc){

        h->use_weight= 0;

        h->use_weight_chroma= 0;

        return;

    }



    h->use_weight= 2;

    h->use_weight_chroma= 2;

    h->luma_log2_weight_denom= 5;

    h->chroma_log2_weight_denom= 5;

    for (i = 0; i < 2; i++) {

        h->luma_weight_flag[i]   = 0;

        h->chroma_weight_flag[i] = 0;

    }



    for(ref0=0; ref0 < h->ref_count[0]; ref0++){

        int poc0 = h->ref_list[0][ref0].poc;

        for(ref1=0; ref1 < h->ref_count[1]; ref1++){

            int poc1 = h->ref_list[1][ref1].poc;

            int td = av_clip(poc1 - poc0, -128, 127);

            if(td){

                int tb = av_clip(cur_poc - poc0, -128, 127);

                int tx = (16384 + (FFABS(td) >> 1)) / td;

                int dist_scale_factor = av_clip((tb*tx + 32) >> 6, -1024, 1023) >> 2;

                if(dist_scale_factor < -64 || dist_scale_factor > 128)

                    h->implicit_weight[ref0][ref1] = 32;

                else

                    h->implicit_weight[ref0][ref1] = 64 - dist_scale_factor;

            }else

                h->implicit_weight[ref0][ref1] = 32;

        }

    }

}
