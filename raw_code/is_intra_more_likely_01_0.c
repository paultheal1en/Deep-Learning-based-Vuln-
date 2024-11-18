static int is_intra_more_likely(MpegEncContext *s){

    int is_intra_likely, i, j, undamaged_count, skip_amount, mb_x, mb_y;



    if (!s->last_picture_ptr || !s->last_picture_ptr->f.data[0]) return 1; //no previous frame available -> use spatial prediction



    undamaged_count=0;

    for(i=0; i<s->mb_num; i++){

        const int mb_xy= s->mb_index2xy[i];

        const int error= s->error_status_table[mb_xy];

        if(!((error&DC_ERROR) && (error&MV_ERROR)))

            undamaged_count++;

    }



    if(s->codec_id == CODEC_ID_H264){

        H264Context *h= (void*)s;

        if (h->ref_count[0] <= 0 || !h->ref_list[0][0].f.data[0])

            return 1;

    }



    if(undamaged_count < 5) return 0; //almost all MBs damaged -> use temporal prediction



    //prevent dsp.sad() check, that requires access to the image

    if(CONFIG_MPEG_XVMC_DECODER && s->avctx->xvmc_acceleration && s->pict_type == AV_PICTURE_TYPE_I)

        return 1;



    skip_amount= FFMAX(undamaged_count/50, 1); //check only upto 50 MBs

    is_intra_likely=0;



    j=0;

    for(mb_y= 0; mb_y<s->mb_height-1; mb_y++){

        for(mb_x= 0; mb_x<s->mb_width; mb_x++){

            int error;

            const int mb_xy= mb_x + mb_y*s->mb_stride;



            error= s->error_status_table[mb_xy];

            if((error&DC_ERROR) && (error&MV_ERROR))

                continue; //skip damaged



            j++;

            if((j%skip_amount) != 0) continue; //skip a few to speed things up



            if(s->pict_type==AV_PICTURE_TYPE_I){

                uint8_t *mb_ptr     = s->current_picture.f.data[0] + mb_x*16 + mb_y*16*s->linesize;

                uint8_t *last_mb_ptr= s->last_picture.f.data   [0] + mb_x*16 + mb_y*16*s->linesize;



                if (s->avctx->codec_id == CODEC_ID_H264) {

                    // FIXME

                } else {

                    ff_thread_await_progress((AVFrame *) s->last_picture_ptr,

                                             mb_y, 0);

                }

                is_intra_likely += s->dsp.sad[0](NULL, last_mb_ptr, mb_ptr                    , s->linesize, 16);

                // FIXME need await_progress() here

                is_intra_likely -= s->dsp.sad[0](NULL, last_mb_ptr, last_mb_ptr+s->linesize*16, s->linesize, 16);

            }else{

                if (IS_INTRA(s->current_picture.f.mb_type[mb_xy]))

                   is_intra_likely++;

                else

                   is_intra_likely--;

            }

        }

    }

//printf("is_intra_likely: %d type:%d\n", is_intra_likely, s->pict_type);

    return is_intra_likely > 0;

}
