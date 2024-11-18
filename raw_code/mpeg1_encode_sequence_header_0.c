static void mpeg1_encode_sequence_header(MpegEncContext *s)

{

        unsigned int vbv_buffer_size;

        unsigned int fps, v;

        int i;

        uint64_t time_code;

        float best_aspect_error= 1E10;

        float aspect_ratio= av_q2d(s->avctx->sample_aspect_ratio);

        int constraint_parameter_flag;

        

        if(aspect_ratio==0.0) aspect_ratio= 1.0; //pixel aspect 1:1 (VGA)

        

        if (s->current_picture.key_frame) {

            AVRational framerate= frame_rate_tab[s->frame_rate_index];



            /* mpeg1 header repeated every gop */

            put_header(s, SEQ_START_CODE);

 

            put_bits(&s->pb, 12, s->width);

            put_bits(&s->pb, 12, s->height);

            

            for(i=1; i<15; i++){

                float error= aspect_ratio;

                if(s->codec_id == CODEC_ID_MPEG1VIDEO || i <=1)

                    error-= 1.0/mpeg1_aspect[i];

                else

                    error-= av_q2d(mpeg2_aspect[i])*s->height/s->width;

             

                error= ABS(error);

                

                if(error < best_aspect_error){

                    best_aspect_error= error;

                    s->aspect_ratio_info= i;

                }

            }

            

            put_bits(&s->pb, 4, s->aspect_ratio_info);

            put_bits(&s->pb, 4, s->frame_rate_index);

            

            if(s->avctx->rc_max_rate){

                v = (s->avctx->rc_max_rate + 399) / 400;

                if (v > 0x3ffff && s->codec_id == CODEC_ID_MPEG1VIDEO)

                    v = 0x3ffff;

            }else{

                v= 0x3FFFF;

            }



            if(s->avctx->rc_buffer_size)

                vbv_buffer_size = s->avctx->rc_buffer_size;

            else

                /* VBV calculation: Scaled so that a VCD has the proper VBV size of 40 kilobytes */

                vbv_buffer_size = (( 20 * s->bit_rate) / (1151929 / 2)) * 8 * 1024;

            vbv_buffer_size= (vbv_buffer_size + 16383) / 16384;



            put_bits(&s->pb, 18, v & 0x3FFFF);

            put_bits(&s->pb, 1, 1); /* marker */

            put_bits(&s->pb, 10, vbv_buffer_size & 0x3FF);



            constraint_parameter_flag= 

                s->width <= 768 && s->height <= 576 && 

                s->mb_width * s->mb_height <= 396 &&

                s->mb_width * s->mb_height * framerate.num <= framerate.den*396*25 &&

                framerate.num <= framerate.den*30 &&

                vbv_buffer_size <= 20 &&

                v <= 1856000/400 &&

                s->codec_id == CODEC_ID_MPEG1VIDEO;

                

            put_bits(&s->pb, 1, constraint_parameter_flag);

            

            ff_write_quant_matrix(&s->pb, s->avctx->intra_matrix);

            ff_write_quant_matrix(&s->pb, s->avctx->inter_matrix);



            if(s->codec_id == CODEC_ID_MPEG2VIDEO){

                put_header(s, EXT_START_CODE);

                put_bits(&s->pb, 4, 1); //seq ext

                put_bits(&s->pb, 1, 0); //esc

                put_bits(&s->pb, 3, 4); //profile

                put_bits(&s->pb, 4, 8); //level

                put_bits(&s->pb, 1, s->progressive_sequence);

                put_bits(&s->pb, 2, 1); //chroma format 4:2:0

                put_bits(&s->pb, 2, 0); //horizontal size ext

                put_bits(&s->pb, 2, 0); //vertical size ext

                put_bits(&s->pb, 12, v>>18); //bitrate ext

                put_bits(&s->pb, 1, 1); //marker

                put_bits(&s->pb, 8, vbv_buffer_size >>10); //vbv buffer ext

                put_bits(&s->pb, 1, s->low_delay);

                put_bits(&s->pb, 2, 0); // frame_rate_ext_n

                put_bits(&s->pb, 5, 0); // frame_rate_ext_d

            }

            

            put_header(s, GOP_START_CODE);

            put_bits(&s->pb, 1, 0); /* do drop frame */

            /* time code : we must convert from the real frame rate to a

               fake mpeg frame rate in case of low frame rate */

            fps = (framerate.num + framerate.den/2)/ framerate.den;

            time_code = s->current_picture_ptr->coded_picture_number;



            s->gop_picture_number = time_code;

            put_bits(&s->pb, 5, (uint32_t)((time_code / (fps * 3600)) % 24));

            put_bits(&s->pb, 6, (uint32_t)((time_code / (fps * 60)) % 60));

            put_bits(&s->pb, 1, 1);

            put_bits(&s->pb, 6, (uint32_t)((time_code / fps) % 60));

            put_bits(&s->pb, 6, (uint32_t)((time_code % fps)));

            put_bits(&s->pb, 1, !!(s->flags & CODEC_FLAG_CLOSED_GOP));

            put_bits(&s->pb, 1, 0); /* broken link */

        }

}
