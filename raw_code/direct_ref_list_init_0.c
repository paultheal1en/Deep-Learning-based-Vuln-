static inline void direct_ref_list_init(H264Context * const h){

    MpegEncContext * const s = &h->s;

    Picture * const ref1 = &h->ref_list[1][0];

    Picture * const cur = s->current_picture_ptr;

    int list, i, j;

    int sidx= s->picture_structure&1;

    int ref1sidx= ref1->reference&1;

    for(list=0; list<2; list++){

        cur->ref_count[sidx][list] = h->ref_count[list];

        for(j=0; j<h->ref_count[list]; j++)

            cur->ref_poc[sidx][list][j] = h->ref_list[list][j].poc;

    }

    if(s->picture_structure == PICT_FRAME){

        memcpy(cur->ref_count[0], cur->ref_count[1], sizeof(cur->ref_count[0]));

        memcpy(cur->ref_poc  [0], cur->ref_poc  [1], sizeof(cur->ref_poc  [0]));

    }

    if(cur->pict_type != FF_B_TYPE || h->direct_spatial_mv_pred)

        return;

    for(list=0; list<2; list++){

        for(i=0; i<ref1->ref_count[ref1sidx][list]; i++){

            const int poc = ref1->ref_poc[ref1sidx][list][i];

            h->map_col_to_list0[list][i] = 0; /* bogus; fills in for missing frames */

            for(j=0; j<h->ref_count[list]; j++)

                if(h->ref_list[list][j].poc == poc){

                    h->map_col_to_list0[list][i] = j;

                    break;

                }

        }

    }

    if(FRAME_MBAFF){

        for(list=0; list<2; list++){

            for(i=0; i<ref1->ref_count[ref1sidx][list]; i++){

                j = h->map_col_to_list0[list][i];

                h->map_col_to_list0_field[list][2*i] = 2*j;

                h->map_col_to_list0_field[list][2*i+1] = 2*j+1;

            }

        }

    }

}
