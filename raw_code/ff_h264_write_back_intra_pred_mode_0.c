void ff_h264_write_back_intra_pred_mode(H264Context *h){

    int8_t *mode= h->intra4x4_pred_mode + h->mb2br_xy[h->mb_xy];



    AV_COPY32(mode, h->intra4x4_pred_mode_cache + 4 + 8*4);

    mode[4]= h->intra4x4_pred_mode_cache[7+8*3];

    mode[5]= h->intra4x4_pred_mode_cache[7+8*2];

    mode[6]= h->intra4x4_pred_mode_cache[7+8*1];

}
