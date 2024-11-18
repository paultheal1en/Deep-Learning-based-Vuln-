static int fill_default_ref_list(H264Context *h){

    MpegEncContext * const s = &h->s;

    int i;

    int smallest_poc_greater_than_current = -1;

    int structure_sel;

    Picture sorted_short_ref[32];

    Picture field_entry_list[2][32];

    Picture *frame_list[2];



    if (FIELD_PICTURE) {

        structure_sel = PICT_FRAME;

        frame_list[0] = field_entry_list[0];

        frame_list[1] = field_entry_list[1];

    } else {

        structure_sel = 0;

        frame_list[0] = h->default_ref_list[0];

        frame_list[1] = h->default_ref_list[1];

    }



    if(h->slice_type_nos==FF_B_TYPE){

        int list;

        int len[2];

        int short_len[2];

        int out_i;

        int limit= INT_MIN;



        /* sort frame according to POC in B slice */

        for(out_i=0; out_i<h->short_ref_count; out_i++){

            int best_i=INT_MIN;

            int best_poc=INT_MAX;



            for(i=0; i<h->short_ref_count; i++){

                const int poc= h->short_ref[i]->poc;

                if(poc > limit && poc < best_poc){

                    best_poc= poc;

                    best_i= i;

                }

            }



            assert(best_i != INT_MIN);



            limit= best_poc;

            sorted_short_ref[out_i]= *h->short_ref[best_i];

            tprintf(h->s.avctx, "sorted poc: %d->%d poc:%d fn:%d\n", best_i, out_i, sorted_short_ref[out_i].poc, sorted_short_ref[out_i].frame_num);

            if (-1 == smallest_poc_greater_than_current) {

                if (h->short_ref[best_i]->poc >= s->current_picture_ptr->poc) {

                    smallest_poc_greater_than_current = out_i;

                }

            }

        }



        tprintf(h->s.avctx, "current poc: %d, smallest_poc_greater_than_current: %d\n", s->current_picture_ptr->poc, smallest_poc_greater_than_current);



        // find the largest POC

        for(list=0; list<2; list++){

            int index = 0;

            int j= -99;

            int step= list ? -1 : 1;



            for(i=0; i<h->short_ref_count && index < h->ref_count[list]; i++, j+=step) {

                int sel;

                while(j<0 || j>= h->short_ref_count){

                    if(j != -99 && step == (list ? -1 : 1))

                        return -1;

                    step = -step;

                    j= smallest_poc_greater_than_current + (step>>1);

                }

                sel = sorted_short_ref[j].reference | structure_sel;

                if(sel != PICT_FRAME) continue;

                frame_list[list][index  ]= sorted_short_ref[j];

                frame_list[list][index++].pic_id= sorted_short_ref[j].frame_num;

            }

            short_len[list] = index;



            for(i = 0; i < 16 && index < h->ref_count[ list ]; i++){

                int sel;

                if(h->long_ref[i] == NULL) continue;

                sel = h->long_ref[i]->reference | structure_sel;

                if(sel != PICT_FRAME) continue;



                frame_list[ list ][index  ]= *h->long_ref[i];

                frame_list[ list ][index++].pic_id= i;

            }

            len[list] = index;

        }



        for(list=0; list<2; list++){

            if (FIELD_PICTURE)

                len[list] = split_field_ref_list(h->default_ref_list[list],

                                                 h->ref_count[list],

                                                 frame_list[list],

                                                 len[list],

                                                 s->picture_structure,

                                                 short_len[list]);



            // swap the two first elements of L1 when L0 and L1 are identical

            if(list && len[0] > 1 && len[0] == len[1])

                for(i=0; h->default_ref_list[0][i].data[0] == h->default_ref_list[1][i].data[0]; i++)

                    if(i == len[0]){

                        FFSWAP(Picture, h->default_ref_list[1][0], h->default_ref_list[1][1]);

                        break;

                    }



            if(len[list] < h->ref_count[ list ])

                memset(&h->default_ref_list[list][len[list]], 0, sizeof(Picture)*(h->ref_count[ list ] - len[list]));

        }





    }else{

        int index=0;

        int short_len;

        for(i=0; i<h->short_ref_count; i++){

            int sel;

            sel = h->short_ref[i]->reference | structure_sel;

            if(sel != PICT_FRAME) continue;

            frame_list[0][index  ]= *h->short_ref[i];

            frame_list[0][index++].pic_id= h->short_ref[i]->frame_num;

        }

        short_len = index;

        for(i = 0; i < 16; i++){

            int sel;

            if(h->long_ref[i] == NULL) continue;

            sel = h->long_ref[i]->reference | structure_sel;

            if(sel != PICT_FRAME) continue;

            frame_list[0][index  ]= *h->long_ref[i];

            frame_list[0][index++].pic_id= i;

        }



        if (FIELD_PICTURE)

            index = split_field_ref_list(h->default_ref_list[0],

                                         h->ref_count[0], frame_list[0],

                                         index, s->picture_structure,

                                         short_len);



        if(index < h->ref_count[0])

            memset(&h->default_ref_list[0][index], 0, sizeof(Picture)*(h->ref_count[0] - index));

    }

#ifdef TRACE

    for (i=0; i<h->ref_count[0]; i++) {

        tprintf(h->s.avctx, "List0: %s fn:%d 0x%p\n", (h->default_ref_list[0][i].long_ref ? "LT" : "ST"), h->default_ref_list[0][i].pic_id, h->default_ref_list[0][i].data[0]);

    }

    if(h->slice_type_nos==FF_B_TYPE){

        for (i=0; i<h->ref_count[1]; i++) {

            tprintf(h->s.avctx, "List1: %s fn:%d 0x%p\n", (h->default_ref_list[1][i].long_ref ? "LT" : "ST"), h->default_ref_list[1][i].pic_id, h->default_ref_list[1][i].data[0]);

        }

    }

#endif

    return 0;

}
