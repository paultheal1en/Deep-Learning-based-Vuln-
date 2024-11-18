static inline int small_diamond_search4MV(MpegEncContext * s, int *best, int dmin,

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

        if(dir!=2 && x>xmin) CHECK_MV4_DIR(x-1, y  , 0)

        if(dir!=3 && y>ymin) CHECK_MV4_DIR(x  , y-1, 1)

        if(dir!=0 && x<xmax) CHECK_MV4_DIR(x+1, y  , 2)

        if(dir!=1 && y<ymax) CHECK_MV4_DIR(x  , y+1, 3)



        if(next_dir==-1){

            return dmin;

        }

    }

}
