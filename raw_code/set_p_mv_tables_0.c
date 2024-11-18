static inline void set_p_mv_tables(MpegEncContext * s, int mx, int my)

{

    const int xy= s->mb_x + 1 + (s->mb_y + 1)*(s->mb_width + 2);

    

    s->p_mv_table[xy][0] = mx;

    s->p_mv_table[xy][1] = my;



    /* has allready been set to the 4 MV if 4MV is done */

    if(!(s->flags&CODEC_FLAG_4MV)){

        int mot_xy= s->block_index[0];



        s->motion_val[mot_xy  ][0]= mx;

        s->motion_val[mot_xy  ][1]= my;

        s->motion_val[mot_xy+1][0]= mx;

        s->motion_val[mot_xy+1][1]= my;



        mot_xy += s->block_wrap[0];

        s->motion_val[mot_xy  ][0]= mx;

        s->motion_val[mot_xy  ][1]= my;

        s->motion_val[mot_xy+1][0]= mx;

        s->motion_val[mot_xy+1][1]= my;

    }

}
