static void filter(AVFilterContext *ctx)

{

    IDETContext *idet = ctx->priv;

    int y, i;

    int64_t alpha[2]={0};

    int64_t delta=0;

    Type type, best_type;

    int match = 0;



    for (i = 0; i < idet->csp->nb_components; i++) {

        int w = idet->cur->video->w;

        int h = idet->cur->video->h;

        int refs = idet->cur->linesize[i];



        if (i && i<3) {

            w >>= idet->csp->log2_chroma_w;

            h >>= idet->csp->log2_chroma_h;

        }



        for (y = 2; y < h - 2; y++) {

            uint8_t *prev = &idet->prev->data[i][y*refs];

            uint8_t *cur  = &idet->cur ->data[i][y*refs];

            uint8_t *next = &idet->next->data[i][y*refs];

            alpha[ y   &1] += idet->filter_line(cur-refs, prev, cur+refs, w);

            alpha[(y^1)&1] += idet->filter_line(cur-refs, next, cur+refs, w);

            delta          += idet->filter_line(cur-refs,  cur, cur+refs, w);

        }

    }



    if      (alpha[0] / (float)alpha[1] > idet->interlace_threshold){

        type = TFF;

    }else if(alpha[1] / (float)alpha[0] > idet->interlace_threshold){

        type = BFF;

    }else if(alpha[1] / (float)delta    > idet->progressive_threshold){

        type = PROGRSSIVE;

    }else{

        type = UNDETERMINED;

    }



    memmove(idet->history+1, idet->history, HIST_SIZE-1);

    idet->history[0] = type;

    best_type = UNDETERMINED;

    for(i=0; i<HIST_SIZE; i++){

        if(idet->history[i] != UNDETERMINED){

            if(best_type == UNDETERMINED)

                best_type = idet->history[i];



            if(idet->history[i] == best_type) {

                match++;

            }else{

                match=0;

                break;

            }

        }

    }

    if(idet->last_type == UNDETERMINED){

        if(match  ) idet->last_type = best_type;

    }else{

        if(match>2) idet->last_type = best_type;

    }



    if      (idet->last_type == TFF){

        idet->cur->video->top_field_first = 1;

        idet->cur->video->interlaced = 1;

    }else if(idet->last_type == BFF){

        idet->cur->video->top_field_first = 0;

        idet->cur->video->interlaced = 1;

    }else if(idet->last_type == PROGRSSIVE){

        idet->cur->video->interlaced = 0;

    }



    idet->prestat [           type] ++;

    idet->poststat[idet->last_type] ++;

    av_log(ctx, AV_LOG_DEBUG, "Single frame:%s, Multi frame:%s\n", type2str(type), type2str(idet->last_type));

}
