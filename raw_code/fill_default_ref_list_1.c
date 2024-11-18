static int fill_default_ref_list(H264Context *h){

    MpegEncContext * const s = &h->s;

    int i;

    int smallest_poc_greater_than_current = -1;

    Picture sorted_short_ref[32];

    

    if(h->slice_type==B_TYPE){

        int out_i;

        int limit= -1;



        /* sort frame according to poc in B slice */

        for(out_i=0; out_i<h->short_ref_count; out_i++){

            int best_i=-1;

            int best_poc=INT_MAX;



            for(i=0; i<h->short_ref_count; i++){

                const int poc= h->short_ref[i]->poc;

                if(poc > limit && poc < best_poc){

                    best_poc= poc;

                    best_i= i;

                }

            }

            

            assert(best_i != -1);

            

            limit= best_poc;

            sorted_short_ref[out_i]= *h->short_ref[best_i];

            tprintf("sorted poc: %d->%d poc:%d fn:%d\n", best_i, out_i, sorted_short_ref[out_i].poc, sorted_short_ref[out_i].frame_num);

            if (-1 == smallest_poc_greater_than_current) {

                if (h->short_ref[best_i]->poc >= s->current_picture_ptr->poc) {

                    smallest_poc_greater_than_current = out_i;

                }

            }

        }

    }



    if(s->picture_structure == PICT_FRAME){

        if(h->slice_type==B_TYPE){

            int list;

            tprintf("current poc: %d, smallest_poc_greater_than_current: %d\n", s->current_picture_ptr->poc, smallest_poc_greater_than_current);



            // find the largest poc

            for(list=0; list<2; list++){

                int index = 0;

                int j= -99;

                int step= list ? -1 : 1;



                for(i=0; i<h->short_ref_count && index < h->ref_count[list]; i++, j+=step) {

                    while(j<0 || j>= h->short_ref_count){

                        if(j != -99 && step == (list ? -1 : 1))

                            return -1;

                        step = -step;

                        j= smallest_poc_greater_than_current + (step>>1);

                    }

                    if(sorted_short_ref[j].reference != 3) continue;

                    h->default_ref_list[list][index  ]= sorted_short_ref[j];

                    h->default_ref_list[list][index++].pic_id= sorted_short_ref[j].frame_num;

                }



                for(i = 0; i < 16 && index < h->ref_count[ list ]; i++){

                    if(h->long_ref[i] == NULL) continue;

                    if(h->long_ref[i]->reference != 3) continue;



                    h->default_ref_list[ list ][index  ]= *h->long_ref[i];

                    h->default_ref_list[ list ][index++].pic_id= i;;

                }

                

                if(list && (smallest_poc_greater_than_current<=0 || smallest_poc_greater_than_current>=h->short_ref_count) && (1 < index)){

                    // swap the two first elements of L1 when

                    // L0 and L1 are identical

                    Picture temp= h->default_ref_list[1][0];

                    h->default_ref_list[1][0] = h->default_ref_list[1][1];

                    h->default_ref_list[1][1] = temp;

                }



                if(index < h->ref_count[ list ])

                    memset(&h->default_ref_list[list][index], 0, sizeof(Picture)*(h->ref_count[ list ] - index));

            }

        }else{

            int index=0;

            for(i=0; i<h->short_ref_count; i++){

                if(h->short_ref[i]->reference != 3) continue; //FIXME refernce field shit

                h->default_ref_list[0][index  ]= *h->short_ref[i];

                h->default_ref_list[0][index++].pic_id= h->short_ref[i]->frame_num;

            }

            for(i = 0; i < 16; i++){

                if(h->long_ref[i] == NULL) continue;

                if(h->long_ref[i]->reference != 3) continue;

                h->default_ref_list[0][index  ]= *h->long_ref[i];

                h->default_ref_list[0][index++].pic_id= i;;

            }

            if(index < h->ref_count[0])

                memset(&h->default_ref_list[0][index], 0, sizeof(Picture)*(h->ref_count[0] - index));

        }

    }else{ //FIELD

        if(h->slice_type==B_TYPE){

        }else{

            //FIXME second field balh

        }

    }

#ifdef TRACE

    for (i=0; i<h->ref_count[0]; i++) {

        tprintf("List0: %s fn:%d 0x%p\n", (h->default_ref_list[0][i].long_ref ? "LT" : "ST"), h->default_ref_list[0][i].pic_id, h->default_ref_list[0][i].data[0]);

    }

    if(h->slice_type==B_TYPE){

        for (i=0; i<h->ref_count[1]; i++) {

            tprintf("List1: %s fn:%d 0x%p\n", (h->default_ref_list[1][i].long_ref ? "LT" : "ST"), h->default_ref_list[1][i].pic_id, h->default_ref_list[0][i].data[0]);

        }

    }

#endif

    return 0;

}
