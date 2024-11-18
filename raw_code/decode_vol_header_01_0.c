static int decode_vol_header(MpegEncContext *s, GetBitContext *gb){

    int width, height, vo_ver_id;



    /* vol header */

    skip_bits(gb, 1); /* random access */

    s->vo_type= get_bits(gb, 8);

    if (get_bits1(gb) != 0) { /* is_ol_id */

        vo_ver_id = get_bits(gb, 4); /* vo_ver_id */

        skip_bits(gb, 3); /* vo_priority */

    } else {

        vo_ver_id = 1;

    }

//printf("vo type:%d\n",s->vo_type);

    s->aspect_ratio_info= get_bits(gb, 4);

    if(s->aspect_ratio_info == FF_ASPECT_EXTENDED){	    

        s->aspected_width = get_bits(gb, 8); // par_width

        s->aspected_height = get_bits(gb, 8); // par_height

    }else{

        s->aspected_width = pixel_aspect[s->aspect_ratio_info][0];

        s->aspected_height= pixel_aspect[s->aspect_ratio_info][1];

    }



    if ((s->vol_control_parameters=get_bits1(gb))) { /* vol control parameter */

        int chroma_format= get_bits(gb, 2);

        if(chroma_format!=1){

            printf("illegal chroma format\n");

        }

        s->low_delay= get_bits1(gb);

        if(get_bits1(gb)){ /* vbv parameters */

            get_bits(gb, 15);	/* first_half_bitrate */

            skip_bits1(gb);	/* marker */

            get_bits(gb, 15);	/* latter_half_bitrate */

            skip_bits1(gb);	/* marker */

            get_bits(gb, 15);	/* first_half_vbv_buffer_size */

            skip_bits1(gb);	/* marker */

            get_bits(gb, 3);	/* latter_half_vbv_buffer_size */

            get_bits(gb, 11);	/* first_half_vbv_occupancy */

            skip_bits1(gb);	/* marker */

            get_bits(gb, 15);	/* latter_half_vbv_occupancy */

            skip_bits1(gb);	/* marker */               

        }

    }else{

        // set low delay flag only once so the smart? low delay detection wont be overriden

        if(s->picture_number==0)

            s->low_delay=0;

    }



    s->shape = get_bits(gb, 2); /* vol shape */

    if(s->shape != RECT_SHAPE) printf("only rectangular vol supported\n");

    if(s->shape == GRAY_SHAPE && vo_ver_id != 1){

        printf("Gray shape not supported\n");

        skip_bits(gb, 4);  //video_object_layer_shape_extension

    }



    skip_bits1(gb);   /* marker */

    

    s->time_increment_resolution = get_bits(gb, 16);

    

    s->time_increment_bits = av_log2(s->time_increment_resolution - 1) + 1;

    if (s->time_increment_bits < 1)

        s->time_increment_bits = 1;

    skip_bits1(gb);   /* marker */



    if (get_bits1(gb) != 0) {   /* fixed_vop_rate  */

        skip_bits(gb, s->time_increment_bits);

    }



    if (s->shape != BIN_ONLY_SHAPE) {

        if (s->shape == RECT_SHAPE) {

            skip_bits1(gb);   /* marker */

            width = get_bits(gb, 13);

            skip_bits1(gb);   /* marker */

            height = get_bits(gb, 13);

            skip_bits1(gb);   /* marker */

            if(width && height){ /* they should be non zero but who knows ... */

                s->width = width;

                s->height = height;

//                printf("width/height: %d %d\n", width, height);

            }

        }

        

        s->progressive_sequence= get_bits1(gb)^1;

        if(!get_bits1(gb)) printf("OBMC not supported (very likely buggy encoder)\n");   /* OBMC Disable */

        if (vo_ver_id == 1) {

            s->vol_sprite_usage = get_bits1(gb); /* vol_sprite_usage */

        } else {

            s->vol_sprite_usage = get_bits(gb, 2); /* vol_sprite_usage */

        }

        if(s->vol_sprite_usage==STATIC_SPRITE) printf("Static Sprites not supported\n");

        if(s->vol_sprite_usage==STATIC_SPRITE || s->vol_sprite_usage==GMC_SPRITE){

            if(s->vol_sprite_usage==STATIC_SPRITE){

                s->sprite_width = get_bits(gb, 13);

                skip_bits1(gb); /* marker */

                s->sprite_height= get_bits(gb, 13);

                skip_bits1(gb); /* marker */

                s->sprite_left  = get_bits(gb, 13);

                skip_bits1(gb); /* marker */

                s->sprite_top   = get_bits(gb, 13);

                skip_bits1(gb); /* marker */

            }

            s->num_sprite_warping_points= get_bits(gb, 6);

            s->sprite_warping_accuracy = get_bits(gb, 2);

            s->sprite_brightness_change= get_bits1(gb);

            if(s->vol_sprite_usage==STATIC_SPRITE)

                s->low_latency_sprite= get_bits1(gb);            

        }

        // FIXME sadct disable bit if verid!=1 && shape not rect

        

        if (get_bits1(gb) == 1) {   /* not_8_bit */

            s->quant_precision = get_bits(gb, 4); /* quant_precision */

            if(get_bits(gb, 4)!=8) printf("N-bit not supported\n"); /* bits_per_pixel */

            if(s->quant_precision!=5) printf("quant precission %d\n", s->quant_precision);

        } else {

            s->quant_precision = 5;

        }

        

        // FIXME a bunch of grayscale shape things



        if((s->mpeg_quant=get_bits1(gb))){ /* vol_quant_type */

            int i, v;

            

            /* load default matrixes */

            for(i=0; i<64; i++){

                int j= s->dsp.idct_permutation[i];

                v= ff_mpeg4_default_intra_matrix[i];

                s->intra_matrix[j]= v;

                s->chroma_intra_matrix[j]= v;

                

                v= ff_mpeg4_default_non_intra_matrix[i];

                s->inter_matrix[j]= v;

                s->chroma_inter_matrix[j]= v;

            }



            /* load custom intra matrix */

            if(get_bits1(gb)){

                int last=0;

		for(i=0; i<64; i++){

                    int j;

                    v= get_bits(gb, 8);

                    if(v==0) break;

                    

                    last= v;

                    j= s->dsp.idct_permutation[ ff_zigzag_direct[i] ];

                    s->intra_matrix[j]= v;

                    s->chroma_intra_matrix[j]= v;

                }



                /* replicate last value */

                for(; i<64; i++){

		    int j= s->dsp.idct_permutation[ ff_zigzag_direct[i] ];

                    s->intra_matrix[j]= v;

                    s->chroma_intra_matrix[j]= v;

                }

            }



            /* load custom non intra matrix */

            if(get_bits1(gb)){

                int last=0;

		for(i=0; i<64; i++){

                    int j;

                    v= get_bits(gb, 8);

                    if(v==0) break;



                    last= v;

                    j= s->dsp.idct_permutation[ ff_zigzag_direct[i] ];

                    s->inter_matrix[j]= v;

                    s->chroma_inter_matrix[j]= v;

                }



                /* replicate last value */

                for(; i<64; i++){

		    int j= s->dsp.idct_permutation[ ff_zigzag_direct[i] ];

                    s->inter_matrix[j]= last;

                    s->chroma_inter_matrix[j]= last;

                }

            }



            // FIXME a bunch of grayscale shape things

        }



        if(vo_ver_id != 1)

             s->quarter_sample= get_bits1(gb);

        else s->quarter_sample=0;



        if(!get_bits1(gb)) printf("Complexity estimation not supported\n");



        s->resync_marker= !get_bits1(gb); /* resync_marker_disabled */



        s->data_partitioning= get_bits1(gb);

        if(s->data_partitioning){

            s->rvlc= get_bits1(gb);

            if(s->rvlc){

                printf("reversible vlc not supported\n");

            }

        }

        

        if(vo_ver_id != 1) {

            s->new_pred= get_bits1(gb);

            if(s->new_pred){

                printf("new pred not supported\n");

                skip_bits(gb, 2); /* requested upstream message type */

                skip_bits1(gb); /* newpred segment type */

            }

            s->reduced_res_vop= get_bits1(gb);

            if(s->reduced_res_vop) printf("reduced resolution VOP not supported\n");

        }

        else{

            s->new_pred=0;

            s->reduced_res_vop= 0;

        }



        s->scalability= get_bits1(gb);



        if (s->scalability) {

            GetBitContext bak= *gb;

            int ref_layer_id;

            int ref_layer_sampling_dir;

            int h_sampling_factor_n;

            int h_sampling_factor_m;

            int v_sampling_factor_n;

            int v_sampling_factor_m;

            

            s->hierachy_type= get_bits1(gb);

            ref_layer_id= get_bits(gb, 4);

            ref_layer_sampling_dir= get_bits1(gb);

            h_sampling_factor_n= get_bits(gb, 5);

            h_sampling_factor_m= get_bits(gb, 5);

            v_sampling_factor_n= get_bits(gb, 5);

            v_sampling_factor_m= get_bits(gb, 5);

            s->enhancement_type= get_bits1(gb);

            

            if(   h_sampling_factor_n==0 || h_sampling_factor_m==0 

               || v_sampling_factor_n==0 || v_sampling_factor_m==0){

               

//                fprintf(stderr, "illegal scalability header (VERY broken encoder), trying to workaround\n");

                s->scalability=0;

               

                *gb= bak;

            }else

                printf("scalability not supported\n");

            

            // bin shape stuff FIXME

        }

    }

    return 0;

}
