static void guess_dc(MpegEncContext *s, int16_t *dc, int w,
                     int h, int stride, int is_luma)
{
    int b_x, b_y;
    int16_t  (*col )[4] = av_malloc(stride*h*sizeof( int16_t)*4);
    uint32_t (*dist)[4] = av_malloc(stride*h*sizeof(uint32_t)*4);
    if(!col || !dist) {
        av_log(s->avctx, AV_LOG_ERROR, "guess_dc() is out of memory\n");
        goto fail;
    }
    for(b_y=0; b_y<h; b_y++){
        int color= 1024;
        int distance= -1;
        for(b_x=0; b_x<w; b_x++){
            int mb_index_j= (b_x>>is_luma) + (b_y>>is_luma)*s->mb_stride;
            int error_j= s->error_status_table[mb_index_j];
            int intra_j = IS_INTRA(s->current_picture.f.mb_type[mb_index_j]);
            if(intra_j==0 || !(error_j&ER_DC_ERROR)){
                color= dc[b_x + b_y*stride];
                distance= b_x;
            }
            col [b_x + b_y*stride][1]= color;
            dist[b_x + b_y*stride][1]= distance >= 0 ? b_x-distance : 9999;
        }
        color= 1024;
        distance= -1;
        for(b_x=w-1; b_x>=0; b_x--){
            int mb_index_j= (b_x>>is_luma) + (b_y>>is_luma)*s->mb_stride;
            int error_j= s->error_status_table[mb_index_j];
            int intra_j = IS_INTRA(s->current_picture.f.mb_type[mb_index_j]);
            if(intra_j==0 || !(error_j&ER_DC_ERROR)){
                color= dc[b_x + b_y*stride];
                distance= b_x;
            }
            col [b_x + b_y*stride][0]= color;
            dist[b_x + b_y*stride][0]= distance >= 0 ? distance-b_x : 9999;
        }
    }
    for(b_x=0; b_x<w; b_x++){
        int color= 1024;
        int distance= -1;
        for(b_y=0; b_y<h; b_y++){
            int mb_index_j= (b_x>>is_luma) + (b_y>>is_luma)*s->mb_stride;
            int error_j= s->error_status_table[mb_index_j];
            int intra_j = IS_INTRA(s->current_picture.f.mb_type[mb_index_j]);
            if(intra_j==0 || !(error_j&ER_DC_ERROR)){
                color= dc[b_x + b_y*stride];
                distance= b_y;
            }
            col [b_x + b_y*stride][3]= color;
            dist[b_x + b_y*stride][3]= distance >= 0 ? b_y-distance : 9999;
        }
        color= 1024;
        distance= -1;
        for(b_y=h-1; b_y>=0; b_y--){
            int mb_index_j= (b_x>>is_luma) + (b_y>>is_luma)*s->mb_stride;
            int error_j= s->error_status_table[mb_index_j];
            int intra_j = IS_INTRA(s->current_picture.f.mb_type[mb_index_j]);
            if(intra_j==0 || !(error_j&ER_DC_ERROR)){
                color= dc[b_x + b_y*stride];
                distance= b_y;
            }
            col [b_x + b_y*stride][2]= color;
            dist[b_x + b_y*stride][2]= distance >= 0 ? distance-b_y : 9999;
        }
    }
    for (b_y = 0; b_y < h; b_y++) {
        for (b_x = 0; b_x < w; b_x++) {
            int mb_index, error, j;
            int64_t guess, weight_sum;
            mb_index = (b_x >> is_luma) + (b_y >> is_luma) * s->mb_stride;
            error    = s->error_status_table[mb_index];
            if (IS_INTER(s->current_picture.f.mb_type[mb_index]))
                continue; // inter
            if (!(error & ER_DC_ERROR))
                continue; // dc-ok
            weight_sum = 0;
            guess      = 0;
            for (j = 0; j < 4; j++) {
                int64_t weight  = 256 * 256 * 256 * 16 / FFMAX(dist[b_x + b_y*stride][j], 1);
                guess          += weight*(int64_t)col[b_x + b_y*stride][j];
                weight_sum     += weight;
            }
            guess = (guess + weight_sum / 2) / weight_sum;
            dc[b_x + b_y * stride] = guess;
        }
    }
    av_freep(&col);
    av_freep(&dist);
}