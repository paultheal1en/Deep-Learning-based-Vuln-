static void encode_header(SnowContext *s){

    int plane_index, level, orientation;



    put_cabac(&s->c, s->header_state, s->keyframe); // state clearing stuff?

    if(s->keyframe){

        put_symbol(&s->c, s->header_state, s->version, 0);

        put_symbol(&s->c, s->header_state, s->temporal_decomposition_type, 0);

        put_symbol(&s->c, s->header_state, s->temporal_decomposition_count, 0);

        put_symbol(&s->c, s->header_state, s->spatial_decomposition_count, 0);

        put_symbol(&s->c, s->header_state, s->colorspace_type, 0);

        put_symbol(&s->c, s->header_state, s->b_width, 0);

        put_symbol(&s->c, s->header_state, s->b_height, 0);

        put_symbol(&s->c, s->header_state, s->chroma_h_shift, 0);

        put_symbol(&s->c, s->header_state, s->chroma_v_shift, 0);

        put_cabac(&s->c, s->header_state, s->spatial_scalability);

//        put_cabac(&s->c, s->header_state, s->rate_scalability);



        for(plane_index=0; plane_index<2; plane_index++){

            for(level=0; level<s->spatial_decomposition_count; level++){

                for(orientation=level ? 1:0; orientation<4; orientation++){

                    if(orientation==2) continue;

                    put_symbol(&s->c, s->header_state, s->plane[plane_index].band[level][orientation].qlog, 1);

                }

            }

        }

    }

    put_symbol(&s->c, s->header_state, s->spatial_decomposition_type, 0);

    put_symbol(&s->c, s->header_state, s->qlog, 1); 

    put_symbol(&s->c, s->header_state, s->mv_scale, 0); 

    put_symbol(&s->c, s->header_state, s->qbias, 1);

}
