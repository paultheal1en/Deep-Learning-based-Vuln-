static int qpel_motion_search(MpegEncContext * s,

				  int *mx_ptr, int *my_ptr, int dmin,

                                  int src_index, int ref_index,                                  

                                  int size, int h)

{

    MotionEstContext * const c= &s->me;

    const int mx = *mx_ptr;

    const int my = *my_ptr;   

    const int penalty_factor= c->sub_penalty_factor;

    const int map_generation= c->map_generation;

    const int subpel_quality= c->avctx->me_subpel_quality;

    uint32_t *map= c->map;

    me_cmp_func cmpf, chroma_cmpf;

    me_cmp_func cmp_sub, chroma_cmp_sub;



    LOAD_COMMON

    int flags= c->sub_flags;

    

    cmpf= s->dsp.me_cmp[size];

    chroma_cmpf= s->dsp.me_cmp[size+1]; //factorize FIXME

 //FIXME factorize



    cmp_sub= s->dsp.me_sub_cmp[size];

    chroma_cmp_sub= s->dsp.me_sub_cmp[size+1];



    if(c->skip){ //FIXME somehow move up (benchmark)

        *mx_ptr = 0;

        *my_ptr = 0;

        return dmin;

    }

        

    if(c->avctx->me_cmp != c->avctx->me_sub_cmp){

        dmin= cmp(s, mx, my, 0, 0, size, h, ref_index, src_index, cmp_sub, chroma_cmp_sub, flags);

        if(mx || my || size>0)

            dmin += (mv_penalty[4*mx - pred_x] + mv_penalty[4*my - pred_y])*penalty_factor;

    }

        

    if (mx > xmin && mx < xmax && 

        my > ymin && my < ymax) {

        int bx=4*mx, by=4*my;

        int d= dmin;

        int i, nx, ny;

        const int index= (my<<ME_MAP_SHIFT) + mx;

        const int t= score_map[(index-(1<<ME_MAP_SHIFT)  )&(ME_MAP_SIZE-1)];

        const int l= score_map[(index- 1                 )&(ME_MAP_SIZE-1)];

        const int r= score_map[(index+ 1                 )&(ME_MAP_SIZE-1)];

        const int b= score_map[(index+(1<<ME_MAP_SHIFT)  )&(ME_MAP_SIZE-1)];

        const int c= score_map[(index                    )&(ME_MAP_SIZE-1)];

        int best[8];

        int best_pos[8][2];

        

        memset(best, 64, sizeof(int)*8);

#if 1

        if(s->me.dia_size>=2){        

            const int tl= score_map[(index-(1<<ME_MAP_SHIFT)-1)&(ME_MAP_SIZE-1)];

            const int bl= score_map[(index+(1<<ME_MAP_SHIFT)-1)&(ME_MAP_SIZE-1)];

            const int tr= score_map[(index-(1<<ME_MAP_SHIFT)+1)&(ME_MAP_SIZE-1)];

            const int br= score_map[(index+(1<<ME_MAP_SHIFT)+1)&(ME_MAP_SIZE-1)];



            for(ny= -3; ny <= 3; ny++){

                for(nx= -3; nx <= 3; nx++){

                    const int t2= nx*nx*(tr + tl - 2*t) + 4*nx*(tr-tl) + 32*t;

                    const int c2= nx*nx*( r +  l - 2*c) + 4*nx*( r- l) + 32*c;

                    const int b2= nx*nx*(br + bl - 2*b) + 4*nx*(br-bl) + 32*b;

                    int score= ny*ny*(b2 + t2 - 2*c2) + 4*ny*(b2 - t2) + 32*c2;

                    int i;

                    

                    if((nx&3)==0 && (ny&3)==0) continue;

                    

                    score += 1024*(mv_penalty[4*mx + nx - pred_x] + mv_penalty[4*my + ny - pred_y])*penalty_factor;

                    

//                    if(nx&1) score-=1024*c->penalty_factor;

//                    if(ny&1) score-=1024*c->penalty_factor;

                    

                    for(i=0; i<8; i++){

                        if(score < best[i]){

                            memmove(&best[i+1], &best[i], sizeof(int)*(7-i));

                            memmove(&best_pos[i+1][0], &best_pos[i][0], sizeof(int)*2*(7-i));

                            best[i]= score;

                            best_pos[i][0]= nx + 4*mx;

                            best_pos[i][1]= ny + 4*my;

                            break;

                        }

                    }

                }

            }

        }else{

            int tl;

            const int cx = 4*(r - l);

            const int cx2= r + l - 2*c; 

            const int cy = 4*(b - t);

            const int cy2= b + t - 2*c;

            int cxy;

              

            if(map[(index-(1<<ME_MAP_SHIFT)-1)&(ME_MAP_SIZE-1)] == (my<<ME_MAP_MV_BITS) + mx + map_generation && 0){ //FIXME

                tl= score_map[(index-(1<<ME_MAP_SHIFT)-1)&(ME_MAP_SIZE-1)];

            }else{

                tl= cmp(s, mx-1, my-1, 0, 0, size, h, ref_index, src_index, cmpf, chroma_cmpf, flags);//FIXME wrong if chroma me is different

            }

            

            cxy= 2*tl + (cx + cy)/4 - (cx2 + cy2) - 2*c; 

           

            assert(16*cx2 + 4*cx + 32*c == 32*r);

            assert(16*cx2 - 4*cx + 32*c == 32*l);

            assert(16*cy2 + 4*cy + 32*c == 32*b);

            assert(16*cy2 - 4*cy + 32*c == 32*t);

            assert(16*cxy + 16*cy2 + 16*cx2 - 4*cy - 4*cx + 32*c == 32*tl);

            

            for(ny= -3; ny <= 3; ny++){

                for(nx= -3; nx <= 3; nx++){

                    int score= ny*nx*cxy + nx*nx*cx2 + ny*ny*cy2 + nx*cx + ny*cy + 32*c; //FIXME factor

                    int i;

                    

                    if((nx&3)==0 && (ny&3)==0) continue;

                

                    score += 32*(mv_penalty[4*mx + nx - pred_x] + mv_penalty[4*my + ny - pred_y])*penalty_factor;

//                    if(nx&1) score-=32*c->penalty_factor;

  //                  if(ny&1) score-=32*c->penalty_factor;

                    

                    for(i=0; i<8; i++){

                        if(score < best[i]){

                            memmove(&best[i+1], &best[i], sizeof(int)*(7-i));

                            memmove(&best_pos[i+1][0], &best_pos[i][0], sizeof(int)*2*(7-i));

                            best[i]= score;

                            best_pos[i][0]= nx + 4*mx;

                            best_pos[i][1]= ny + 4*my;

                            break;

                        }

                    }

                }

            }            

        }

        for(i=0; i<subpel_quality; i++){

            nx= best_pos[i][0];

            ny= best_pos[i][1];

            CHECK_QUARTER_MV(nx&3, ny&3, nx>>2, ny>>2)

        }



#if 0

            const int tl= score_map[(index-(1<<ME_MAP_SHIFT)-1)&(ME_MAP_SIZE-1)];

            const int bl= score_map[(index+(1<<ME_MAP_SHIFT)-1)&(ME_MAP_SIZE-1)];

            const int tr= score_map[(index-(1<<ME_MAP_SHIFT)+1)&(ME_MAP_SIZE-1)];

            const int br= score_map[(index+(1<<ME_MAP_SHIFT)+1)&(ME_MAP_SIZE-1)];

//            if(l < r && l < t && l < b && l < tl && l < bl && l < tr && l < br && bl < tl){

            if(tl<br){



//            nx= FFMAX(4*mx - bx, bx - 4*mx);

//            ny= FFMAX(4*my - by, by - 4*my);

            

            static int stats[7][7], count;

            count++;

            stats[4*mx - bx + 3][4*my - by + 3]++;

            if(256*256*256*64 % count ==0){

                for(i=0; i<49; i++){

                    if((i%7)==0) printf("\n");

                    printf("%6d ", stats[0][i]);

                }

                printf("\n");

            }

            }

#endif

#else



        CHECK_QUARTER_MV(2, 2, mx-1, my-1)

        CHECK_QUARTER_MV(0, 2, mx  , my-1)        

        CHECK_QUARTER_MV(2, 2, mx  , my-1)

        CHECK_QUARTER_MV(2, 0, mx  , my  )

        CHECK_QUARTER_MV(2, 2, mx  , my  )

        CHECK_QUARTER_MV(0, 2, mx  , my  )

        CHECK_QUARTER_MV(2, 2, mx-1, my  )

        CHECK_QUARTER_MV(2, 0, mx-1, my  )

        

        nx= bx;

        ny= by;

        

        for(i=0; i<8; i++){

            int ox[8]= {0, 1, 1, 1, 0,-1,-1,-1};

            int oy[8]= {1, 1, 0,-1,-1,-1, 0, 1};

            CHECK_QUARTER_MV((nx + ox[i])&3, (ny + oy[i])&3, (nx + ox[i])>>2, (ny + oy[i])>>2)

        }

#endif

#if 0

        //outer ring

        CHECK_QUARTER_MV(1, 3, mx-1, my-1)

        CHECK_QUARTER_MV(1, 2, mx-1, my-1)

        CHECK_QUARTER_MV(1, 1, mx-1, my-1)

        CHECK_QUARTER_MV(2, 1, mx-1, my-1)

        CHECK_QUARTER_MV(3, 1, mx-1, my-1)

        CHECK_QUARTER_MV(0, 1, mx  , my-1)

        CHECK_QUARTER_MV(1, 1, mx  , my-1)

        CHECK_QUARTER_MV(2, 1, mx  , my-1)

        CHECK_QUARTER_MV(3, 1, mx  , my-1)

        CHECK_QUARTER_MV(3, 2, mx  , my-1)

        CHECK_QUARTER_MV(3, 3, mx  , my-1)

        CHECK_QUARTER_MV(3, 0, mx  , my  )

        CHECK_QUARTER_MV(3, 1, mx  , my  )

        CHECK_QUARTER_MV(3, 2, mx  , my  )

        CHECK_QUARTER_MV(3, 3, mx  , my  )

        CHECK_QUARTER_MV(2, 3, mx  , my  )

        CHECK_QUARTER_MV(1, 3, mx  , my  )

        CHECK_QUARTER_MV(0, 3, mx  , my  )

        CHECK_QUARTER_MV(3, 3, mx-1, my  )

        CHECK_QUARTER_MV(2, 3, mx-1, my  )

        CHECK_QUARTER_MV(1, 3, mx-1, my  )

        CHECK_QUARTER_MV(1, 2, mx-1, my  )

        CHECK_QUARTER_MV(1, 1, mx-1, my  )

        CHECK_QUARTER_MV(1, 0, mx-1, my  )

#endif

        assert(bx >= xmin*4 && bx <= xmax*4 && by >= ymin*4 && by <= ymax*4);



        *mx_ptr = bx;

        *my_ptr = by;

    }else{

        *mx_ptr =4*mx;

        *my_ptr =4*my;

    }



    return dmin;

}
