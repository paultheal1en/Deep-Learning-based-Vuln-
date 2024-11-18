static int decode_ref_pic_marking(H264Context *h, GetBitContext *gb){

    MpegEncContext * const s = &h->s;

    int i;




    if(h->nal_unit_type == NAL_IDR_SLICE){ //FIXME fields

        s->broken_link= get_bits1(gb) -1;

        h->mmco[0].long_arg= get_bits1(gb) - 1; // current_long_term_idx

        if(h->mmco[0].long_arg == -1)


        else{

            h->mmco[0].opcode= MMCO_LONG;

            h->mmco_index= 1;

        }

    }else{

        if(get_bits1(gb)){ // adaptive_ref_pic_marking_mode_flag

            for(i= 0; i<MAX_MMCO_COUNT; i++) {

                MMCOOpcode opcode= get_ue_golomb(gb);



                h->mmco[i].opcode= opcode;

                if(opcode==MMCO_SHORT2UNUSED || opcode==MMCO_SHORT2LONG){

                    h->mmco[i].short_pic_num= (h->curr_pic_num - get_ue_golomb(gb) - 1) & (h->max_pic_num - 1);

/*                    if(h->mmco[i].short_pic_num >= h->short_ref_count || h->short_ref[ h->mmco[i].short_pic_num ] == NULL){

                        av_log(s->avctx, AV_LOG_ERROR, "illegal short ref in memory management control operation %d\n", mmco);

                        return -1;

                    }*/

                }

                if(opcode==MMCO_SHORT2LONG || opcode==MMCO_LONG2UNUSED || opcode==MMCO_LONG || opcode==MMCO_SET_MAX_LONG){

                    unsigned int long_arg= get_ue_golomb(gb);

                    if(long_arg >= 32 || (long_arg >= 16 && !(opcode == MMCO_LONG2UNUSED && FIELD_PICTURE))){

                        av_log(h->s.avctx, AV_LOG_ERROR, "illegal long ref in memory management control operation %d\n", opcode);

                        return -1;

                    }

                    h->mmco[i].long_arg= long_arg;

                }



                if(opcode > (unsigned)MMCO_LONG){

                    av_log(h->s.avctx, AV_LOG_ERROR, "illegal memory management control operation %d\n", opcode);

                    return -1;

                }

                if(opcode == MMCO_END)

                    break;

            }

            h->mmco_index= i;

        }else{

            assert(h->long_ref_count + h->short_ref_count <= h->sps.ref_frame_count);



            if(h->short_ref_count && h->long_ref_count + h->short_ref_count == h->sps.ref_frame_count &&

                    !(FIELD_PICTURE && !s->first_field && s->current_picture_ptr->reference)) {

                h->mmco[0].opcode= MMCO_SHORT2UNUSED;

                h->mmco[0].short_pic_num= h->short_ref[ h->short_ref_count - 1 ]->frame_num;

                h->mmco_index= 1;

                if (FIELD_PICTURE) {

                    h->mmco[0].short_pic_num *= 2;

                    h->mmco[1].opcode= MMCO_SHORT2UNUSED;

                    h->mmco[1].short_pic_num= h->mmco[0].short_pic_num + 1;

                    h->mmco_index= 2;

                }

            }else


        }

    }



    return 0;

}