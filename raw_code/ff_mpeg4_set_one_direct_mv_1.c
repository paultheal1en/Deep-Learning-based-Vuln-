static inline void ff_mpeg4_set_one_direct_mv(MpegEncContext *s, int mx, int my, int i){

    static const int tab_size = sizeof(s->direct_scale_mv[0])/sizeof(int16_t);

    static const int tab_bias = (tab_size/2);

    int xy= s->block_index[i];

    uint16_t time_pp= s->pp_time;

    uint16_t time_pb= s->pb_time;

    int p_mx, p_my;



    p_mx= s->next_picture.motion_val[0][xy][0];

    if((unsigned)(p_mx + tab_bias) < tab_size){

        s->mv[0][i][0] = s->direct_scale_mv[0][p_mx + tab_bias] + mx;

        s->mv[1][i][0] = mx ? s->mv[0][i][0] - p_mx

                            : s->direct_scale_mv[1][p_mx + tab_bias];

    }else{

        s->mv[0][i][0] = p_mx*time_pb/time_pp + mx;

        s->mv[1][i][0] = mx ? s->mv[0][i][0] - p_mx

                            : p_mx*(time_pb - time_pp)/time_pp;

    }

    p_my= s->next_picture.motion_val[0][xy][1];

    if((unsigned)(p_my + tab_bias) < tab_size){

        s->mv[0][i][1] = s->direct_scale_mv[0][p_my + tab_bias] + my;

        s->mv[1][i][1] = my ? s->mv[0][i][1] - p_my

                            : s->direct_scale_mv[1][p_my + tab_bias];

    }else{

        s->mv[0][i][1] = p_my*time_pb/time_pp + my;

        s->mv[1][i][1] = my ? s->mv[0][i][1] - p_my

                            : p_my*(time_pb - time_pp)/time_pp;

    }

}
