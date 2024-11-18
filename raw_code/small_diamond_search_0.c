static inline int small_diamond_search(MpegEncContext * s, int *best, int dmin,

                                       UINT8 *new_pic, UINT8 *old_pic, int pic_stride,

                                       int pred_x, int pred_y, UINT16 *mv_penalty, int quant,

                                       int xmin, int ymin, int xmax, int ymax, int shift)

{

    int next_dir=-1;



    for(;;){

        int d;

        const int dir= next_dir;

        const int x= best[0];

        const int y= best[1];

        next_dir=-1;



//printf("%d", dir);

        if(dir!=2 && x>xmin) CHECK_MV_DIR(x-1, y  , 0)

        if(dir!=3 && y>ymin) CHECK_MV_DIR(x  , y-1, 1)

        if(dir!=0 && x<xmax) CHECK_MV_DIR(x+1, y  , 2)

        if(dir!=1 && y<ymax) CHECK_MV_DIR(x  , y+1, 3)



        if(next_dir==-1){

            return dmin;

        }

    }



/*    for(;;){

        int d;

        const int x= best[0];

        const int y= best[1];

        const int last_min=dmin;

        if(x>xmin) CHECK_MV(x-1, y  )

        if(y>xmin) CHECK_MV(x  , y-1)

        if(x<xmax) CHECK_MV(x+1, y  )

        if(y<xmax) CHECK_MV(x  , y+1)

        if(x>xmin && y>ymin) CHECK_MV(x-1, y-1)

        if(x>xmin && y<ymax) CHECK_MV(x-1, y+1)

        if(x<xmax && y>ymin) CHECK_MV(x+1, y-1)

        if(x<xmax && y<ymax) CHECK_MV(x+1, y+1)

        if(x-1>xmin) CHECK_MV(x-2, y  )

        if(y-1>xmin) CHECK_MV(x  , y-2)

        if(x+1<xmax) CHECK_MV(x+2, y  )

        if(y+1<xmax) CHECK_MV(x  , y+2)

        if(x-1>xmin && y-1>ymin) CHECK_MV(x-2, y-2)

        if(x-1>xmin && y+1<ymax) CHECK_MV(x-2, y+2)

        if(x+1<xmax && y-1>ymin) CHECK_MV(x+2, y-2)

        if(x+1<xmax && y+1<ymax) CHECK_MV(x+2, y+2)

        if(dmin==last_min) return dmin;

    }

    */

}
