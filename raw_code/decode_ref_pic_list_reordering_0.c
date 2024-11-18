static int decode_ref_pic_list_reordering(H264Context *h){

    MpegEncContext * const s = &h->s;

    int list, index, pic_structure;



    print_short_term(h);

    print_long_term(h);

    if(h->slice_type==FF_I_TYPE || h->slice_type==FF_SI_TYPE) return 0; //FIXME move before func



    for(list=0; list<h->list_count; list++){

        memcpy(h->ref_list[list], h->default_ref_list[list], sizeof(Picture)*h->ref_count[list]);



        if(get_bits1(&s->gb)){

            int pred= h->curr_pic_num;



            for(index=0; ; index++){

                unsigned int reordering_of_pic_nums_idc= get_ue_golomb(&s->gb);

                unsigned int pic_id;

                int i;

                Picture *ref = NULL;



                if(reordering_of_pic_nums_idc==3)

                    break;



                if(index >= h->ref_count[list]){

                    av_log(h->s.avctx, AV_LOG_ERROR, "reference count overflow\n");

                    return -1;

                }



                if(reordering_of_pic_nums_idc<3){

                    if(reordering_of_pic_nums_idc<2){

                        const unsigned int abs_diff_pic_num= get_ue_golomb(&s->gb) + 1;

                        int frame_num;



                        if(abs_diff_pic_num > h->max_pic_num){

                            av_log(h->s.avctx, AV_LOG_ERROR, "abs_diff_pic_num overflow\n");

                            return -1;

                        }



                        if(reordering_of_pic_nums_idc == 0) pred-= abs_diff_pic_num;

                        else                                pred+= abs_diff_pic_num;

                        pred &= h->max_pic_num - 1;



                        frame_num = pic_num_extract(h, pred, &pic_structure);



                        for(i= h->short_ref_count-1; i>=0; i--){

                            ref = h->short_ref[i];

                            assert(ref->reference);

                            assert(!ref->long_ref);

                            if(ref->data[0] != NULL &&

                                   ref->frame_num == frame_num &&

                                   (ref->reference & pic_structure) &&

                                   ref->long_ref == 0) // ignore non existing pictures by testing data[0] pointer

                                break;

                        }

                        if(i>=0)

                            ref->pic_id= pred;

                    }else{

                        int long_idx;

                        pic_id= get_ue_golomb(&s->gb); //long_term_pic_idx



                        long_idx= pic_num_extract(h, pic_id, &pic_structure);



                        if(long_idx>31){

                            av_log(h->s.avctx, AV_LOG_ERROR, "long_term_pic_idx overflow\n");

                            return -1;

                        }

                        ref = h->long_ref[long_idx];

                        assert(!(ref && !ref->reference));

                        if(ref && (ref->reference & pic_structure)){

                            ref->pic_id= pic_id;

                            assert(ref->long_ref);

                            i=0;

                        }else{

                            i=-1;

                        }

                    }



                    if (i < 0) {

                        av_log(h->s.avctx, AV_LOG_ERROR, "reference picture missing during reorder\n");

                        memset(&h->ref_list[list][index], 0, sizeof(Picture)); //FIXME

                    } else {

                        for(i=index; i+1<h->ref_count[list]; i++){

                            if(ref->long_ref == h->ref_list[list][i].long_ref && ref->pic_id == h->ref_list[list][i].pic_id)

                                break;

                        }

                        for(; i > index; i--){

                            h->ref_list[list][i]= h->ref_list[list][i-1];

                        }

                        h->ref_list[list][index]= *ref;

                        if (FIELD_PICTURE){

                            pic_as_field(&h->ref_list[list][index], pic_structure);

                        }

                    }

                }else{

                    av_log(h->s.avctx, AV_LOG_ERROR, "illegal reordering_of_pic_nums_idc\n");

                    return -1;

                }

            }

        }

    }

    for(list=0; list<h->list_count; list++){

        for(index= 0; index < h->ref_count[list]; index++){

            if(!h->ref_list[list][index].data[0])

                h->ref_list[list][index]= s->current_picture;

        }

    }



    if(h->slice_type==FF_B_TYPE && !h->direct_spatial_mv_pred)

        direct_dist_scale_factor(h);

    direct_ref_list_init(h);

    return 0;

}
