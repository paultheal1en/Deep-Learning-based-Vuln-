void ff_er_add_slice(MpegEncContext *s, int startx, int starty, int endx, int endy, int status){
    const int start_i= clip(startx + starty * s->mb_width    , 0, s->mb_num-1);
    const int end_i  = clip(endx   + endy   * s->mb_width    , 0, s->mb_num);
    const int start_xy= s->mb_index2xy[start_i];
    const int end_xy  = s->mb_index2xy[end_i];
    int mask= -1;
    if(!s->error_resilience) return;
    mask &= ~VP_START;
    if(status & (AC_ERROR|AC_END)){
        mask &= ~(AC_ERROR|AC_END);
        s->error_count -= end_i - start_i + 1;
    if(status & (DC_ERROR|DC_END)){
        mask &= ~(DC_ERROR|DC_END);
        s->error_count -= end_i - start_i + 1;
    if(status & (MV_ERROR|MV_END)){
        mask &= ~(MV_ERROR|MV_END);
        s->error_count -= end_i - start_i + 1;
    if(status & (AC_ERROR|DC_ERROR|MV_ERROR)) s->error_count= INT_MAX;
    if(mask == ~0x7F){
        memset(&s->error_status_table[start_xy], 0, (end_xy - start_xy) * sizeof(uint8_t));
    }else{
        int i;
        for(i=start_xy; i<end_xy; i++){
            s->error_status_table[ i ] &= mask;
    if(end_i == s->mb_num)
        s->error_count= INT_MAX;
    else{
        s->error_status_table[end_xy] &= mask;
        s->error_status_table[end_xy] |= status;
    s->error_status_table[start_xy] |= VP_START;
    if(start_xy > 0 && s->avctx->thread_count <= 1 && s->avctx->skip_top*s->mb_width < start_i){
        int prev_status= s->error_status_table[ s->mb_index2xy[start_i - 1] ];
        prev_status &= ~ VP_START;
        if(prev_status != (MV_END|DC_END|AC_END)) s->error_count= INT_MAX;