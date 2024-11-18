static int encode_picture_lossless(AVCodecContext *avctx, AVPacket *pkt,
                                   const AVFrame *pict, int *got_packet)
{
    MpegEncContext * const s = avctx->priv_data;
    MJpegContext * const m = s->mjpeg_ctx;
    const int width= s->width;
    const int height= s->height;
    AVFrame * const p = &s->current_picture.f;
    const int predictor= avctx->prediction_method+1;
    const int mb_width  = (width  + s->mjpeg_hsample[0] - 1) / s->mjpeg_hsample[0];
    const int mb_height = (height + s->mjpeg_vsample[0] - 1) / s->mjpeg_vsample[0];
    int ret, max_pkt_size = FF_MIN_BUFFER_SIZE;
    if (avctx->pix_fmt == AV_PIX_FMT_BGRA)
        max_pkt_size += width * height * 3 * 4;
    else {
        max_pkt_size += mb_width * mb_height * 3 * 4
                        * s->mjpeg_hsample[0] * s->mjpeg_vsample[0];
    if ((ret = ff_alloc_packet2(avctx, pkt, max_pkt_size)) < 0)
    init_put_bits(&s->pb, pkt->data, pkt->size);
    *p = *pict;
    p->pict_type= AV_PICTURE_TYPE_I;
    p->key_frame= 1;
    ff_mjpeg_encode_picture_header(s);
    s->header_bits= put_bits_count(&s->pb);
    if(avctx->pix_fmt == AV_PIX_FMT_BGR0
        || avctx->pix_fmt == AV_PIX_FMT_BGRA
        || avctx->pix_fmt == AV_PIX_FMT_BGR24){
        int x, y, i;
        const int linesize= p->linesize[0];
        uint16_t (*buffer)[4]= (void *) s->rd_scratchpad;
        int left[3], top[3], topleft[3];
        for(i=0; i<3; i++){
            buffer[0][i]= 1 << (9 - 1);
        for(y = 0; y < height; y++) {
            const int modified_predictor= y ? predictor : 1;
            uint8_t *ptr = p->data[0] + (linesize * y);
            if(s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < width*3*4){
                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");
                return -1;
            for(i=0; i<3; i++){
                top[i]= left[i]= topleft[i]= buffer[0][i];
            for(x = 0; x < width; x++) {
                if(avctx->pix_fmt == AV_PIX_FMT_BGR24){
                    buffer[x][1] = ptr[3*x+0] - ptr[3*x+1] + 0x100;
                    buffer[x][2] = ptr[3*x+2] - ptr[3*x+1] + 0x100;
                    buffer[x][0] = (ptr[3*x+0] + 2*ptr[3*x+1] + ptr[3*x+2])>>2;
                }else{
                buffer[x][1] = ptr[4*x+0] - ptr[4*x+1] + 0x100;
                buffer[x][2] = ptr[4*x+2] - ptr[4*x+1] + 0x100;
                buffer[x][0] = (ptr[4*x+0] + 2*ptr[4*x+1] + ptr[4*x+2])>>2;
                for(i=0;i<3;i++) {
                    int pred, diff;
                    PREDICT(pred, topleft[i], top[i], left[i], modified_predictor);
                    topleft[i]= top[i];
                    top[i]= buffer[x+1][i];
                    left[i]= buffer[x][i];
                    diff= ((left[i] - pred + 0x100)&0x1FF) - 0x100;
                    if(i==0)
                        ff_mjpeg_encode_dc(s, diff, m->huff_size_dc_luminance, m->huff_code_dc_luminance); //FIXME ugly
                    else
                        ff_mjpeg_encode_dc(s, diff, m->huff_size_dc_chrominance, m->huff_code_dc_chrominance);
    }else{
        int mb_x, mb_y, i;
        for(mb_y = 0; mb_y < mb_height; mb_y++) {
            if(s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < mb_width * 4 * 3 * s->mjpeg_hsample[0] * s->mjpeg_vsample[0]){
                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");
                return -1;
            for(mb_x = 0; mb_x < mb_width; mb_x++) {
                if(mb_x==0 || mb_y==0){
                    for(i=0;i<3;i++) {
                        uint8_t *ptr;
                        int x, y, h, v, linesize;
                        h = s->mjpeg_hsample[i];
                        v = s->mjpeg_vsample[i];
                        linesize= p->linesize[i];
                        for(y=0; y<v; y++){
                            for(x=0; x<h; x++){
                                int pred;
                                ptr = p->data[i] + (linesize * (v * mb_y + y)) + (h * mb_x + x); //FIXME optimize this crap
                                if(y==0 && mb_y==0){
                                    if(x==0 && mb_x==0){
                                        pred= 128;
                                    }else{
                                        pred= ptr[-1];
                                }else{
                                    if(x==0 && mb_x==0){
                                        pred= ptr[-linesize];
                                    }else{
                                        PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);
                                if(i==0)
                                    ff_mjpeg_encode_dc(s, *ptr - pred, m->huff_size_dc_luminance, m->huff_code_dc_luminance); //FIXME ugly
                                else
                                    ff_mjpeg_encode_dc(s, *ptr - pred, m->huff_size_dc_chrominance, m->huff_code_dc_chrominance);
                }else{
                    for(i=0;i<3;i++) {
                        uint8_t *ptr;
                        int x, y, h, v, linesize;
                        h = s->mjpeg_hsample[i];
                        v = s->mjpeg_vsample[i];
                        linesize= p->linesize[i];
                        for(y=0; y<v; y++){
                            for(x=0; x<h; x++){
                                int pred;
                                ptr = p->data[i] + (linesize * (v * mb_y + y)) + (h * mb_x + x); //FIXME optimize this crap
                                PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);
                                if(i==0)
                                    ff_mjpeg_encode_dc(s, *ptr - pred, m->huff_size_dc_luminance, m->huff_code_dc_luminance); //FIXME ugly
                                else
                                    ff_mjpeg_encode_dc(s, *ptr - pred, m->huff_size_dc_chrominance, m->huff_code_dc_chrominance);
    emms_c();
    av_assert0(s->esc_pos == s->header_bits >> 3);
    ff_mjpeg_encode_stuffing(s);
    ff_mjpeg_encode_picture_trailer(s);
    s->picture_number++;
    flush_put_bits(&s->pb);
    pkt->size   = put_bits_ptr(&s->pb) - s->pb.buf;
    pkt->flags |= AV_PKT_FLAG_KEY;
    *got_packet = 1;
    return 0;
//    return (put_bits_count(&f->pb)+7)/8;