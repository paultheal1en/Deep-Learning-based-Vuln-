static void halfpel_interpol(SnowContext *s, uint8_t *halfpel[4][4], AVFrame *frame){

    int p,x,y;



    for(p=0; p<3; p++){

        int is_chroma= !!p;

        int w= is_chroma ? s->avctx->width >>s->chroma_h_shift : s->avctx->width;

        int h= is_chroma ? s->avctx->height>>s->chroma_v_shift : s->avctx->height;

        int ls= frame->linesize[p];

        uint8_t *src= frame->data[p];



        halfpel[1][p] = (uint8_t*) av_malloc(ls * (h + 2 * EDGE_WIDTH)) + EDGE_WIDTH * (1 + ls);

        halfpel[2][p] = (uint8_t*) av_malloc(ls * (h + 2 * EDGE_WIDTH)) + EDGE_WIDTH * (1 + ls);

        halfpel[3][p] = (uint8_t*) av_malloc(ls * (h + 2 * EDGE_WIDTH)) + EDGE_WIDTH * (1 + ls);



        halfpel[0][p]= src;

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int i= y*ls + x;



                halfpel[1][p][i]= (20*(src[i] + src[i+1]) - 5*(src[i-1] + src[i+2]) + (src[i-2] + src[i+3]) + 16 )>>5;

            }

        }

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int i= y*ls + x;



                halfpel[2][p][i]= (20*(src[i] + src[i+ls]) - 5*(src[i-ls] + src[i+2*ls]) + (src[i-2*ls] + src[i+3*ls]) + 16 )>>5;

            }

        }

        src= halfpel[1][p];

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int i= y*ls + x;



                halfpel[3][p][i]= (20*(src[i] + src[i+ls]) - 5*(src[i-ls] + src[i+2*ls]) + (src[i-2*ls] + src[i+3*ls]) + 16 )>>5;

            }

        }



//FIXME border!

    }

}
