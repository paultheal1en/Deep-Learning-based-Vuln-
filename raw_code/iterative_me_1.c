static void iterative_me(SnowContext *s){

    int pass, mb_x, mb_y;

    const int b_width = s->b_width  << s->block_max_depth;

    const int b_height= s->b_height << s->block_max_depth;

    const int b_stride= b_width;

    int color[3];



    for(pass=0; pass<50; pass++){

        int change= 0;



        for(mb_y= 0; mb_y<b_height; mb_y++){

            for(mb_x= 0; mb_x<b_width; mb_x++){

                int dia_change, i, j;

                int best_rd= INT_MAX;

                BlockNode backup;

                const int index= mb_x + mb_y * b_stride;

                BlockNode *block= &s->block[index];

                BlockNode *tb =                 mb_y          ? &s->block[index-b_stride  ] : &null_block;

                BlockNode *lb = mb_x                          ? &s->block[index         -1] : &null_block;

                BlockNode *rb = mb_x<b_width                  ? &s->block[index         +1] : &null_block;

                BlockNode *bb =                 mb_y<b_height ? &s->block[index+b_stride  ] : &null_block;

                BlockNode *tlb= mb_x         && mb_y          ? &s->block[index-b_stride-1] : &null_block;

                BlockNode *trb= mb_x<b_width && mb_y          ? &s->block[index-b_stride+1] : &null_block;

                BlockNode *blb= mb_x         && mb_y<b_height ? &s->block[index+b_stride-1] : &null_block;

                BlockNode *brb= mb_x<b_width && mb_y<b_height ? &s->block[index+b_stride+1] : &null_block;



                if(pass && (block->type & BLOCK_OPT))

                    continue;

                block->type |= BLOCK_OPT;



                backup= *block;



                if(!s->me_cache_generation)

                    memset(s->me_cache, 0, sizeof(s->me_cache));

                s->me_cache_generation += 1<<22;



                // get previous score (cant be cached due to OBMC)

                check_block(s, mb_x, mb_y, (int[2]){block->mx, block->my}, 0, &best_rd);

                check_block(s, mb_x, mb_y, (int[2]){0, 0}, 0, &best_rd);

                check_block(s, mb_x, mb_y, (int[2]){tb->mx, tb->my}, 0, &best_rd);

                check_block(s, mb_x, mb_y, (int[2]){lb->mx, lb->my}, 0, &best_rd);

                check_block(s, mb_x, mb_y, (int[2]){rb->mx, rb->my}, 0, &best_rd);

                check_block(s, mb_x, mb_y, (int[2]){bb->mx, bb->my}, 0, &best_rd);



                /* fullpel ME */

                //FIXME avoid subpel interpol / round to nearest integer

                do{

                    dia_change=0;

                    for(i=0; i<FFMAX(s->avctx->dia_size, 1); i++){

                        for(j=0; j<i; j++){

                            dia_change |= check_block(s, mb_x, mb_y, (int[2]){block->mx+4*(i-j), block->my+(4*j)}, 0, &best_rd);

                            dia_change |= check_block(s, mb_x, mb_y, (int[2]){block->mx-4*(i-j), block->my-(4*j)}, 0, &best_rd);

                            dia_change |= check_block(s, mb_x, mb_y, (int[2]){block->mx+4*(i-j), block->my-(4*j)}, 0, &best_rd);

                            dia_change |= check_block(s, mb_x, mb_y, (int[2]){block->mx-4*(i-j), block->my+(4*j)}, 0, &best_rd);

                        }

                    }

                }while(dia_change);

                /* subpel ME */

                do{

                    static const int square[8][2]= {{+1, 0},{-1, 0},{ 0,+1},{ 0,-1},{+1,+1},{-1,-1},{+1,-1},{-1,+1},};

                    dia_change=0;

                    for(i=0; i<8; i++)

                        dia_change |= check_block(s, mb_x, mb_y, (int[2]){block->mx+square[i][0], block->my+square[i][1]}, 0, &best_rd);

                }while(dia_change);

                //FIXME or try the standard 2 pass qpel or similar



                for(i=0; i<3; i++){

                    color[i]= get_dc(s, mb_x, mb_y, i);

                }

                check_block(s, mb_x, mb_y, color, 1, &best_rd);

                //FIXME RD style color selection



                if(!same_block(block, &backup)){

                    if(tb != &null_block) tb ->type &= ~BLOCK_OPT;

                    if(lb != &null_block) lb ->type &= ~BLOCK_OPT;

                    if(rb != &null_block) rb ->type &= ~BLOCK_OPT;

                    if(bb != &null_block) bb ->type &= ~BLOCK_OPT;

                    if(tlb!= &null_block) tlb->type &= ~BLOCK_OPT;

                    if(trb!= &null_block) trb->type &= ~BLOCK_OPT;

                    if(blb!= &null_block) blb->type &= ~BLOCK_OPT;

                    if(brb!= &null_block) brb->type &= ~BLOCK_OPT;

                    change ++;

                }

            }

        }

        av_log(NULL, AV_LOG_ERROR, "pass:%d changed:%d\n", pass, change);

        if(!change)

            break;

    }

}
