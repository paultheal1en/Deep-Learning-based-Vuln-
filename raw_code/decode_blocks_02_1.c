static void decode_blocks(SnowContext *s){

    int x, y;

    int w= s->b_width;

    int h= s->b_height;



    for(y=0; y<h; y++){

        for(x=0; x<w; x++){

            decode_q_branch(s, 0, x, y);

        }

    }

}
