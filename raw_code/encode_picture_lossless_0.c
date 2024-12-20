static int encode_picture_lossless(AVCodecContext *avctx, unsigned char *buf, int buf_size, void *data){

    MpegEncContext * const s = avctx->priv_data;

    MJpegContext * const m = s->mjpeg_ctx;

    AVFrame *pict = data;

    const int width= s->width;

    const int height= s->height;

    AVFrame * const p= (AVFrame*)&s->current_picture;

    const int predictor= avctx->prediction_method+1;



    init_put_bits(&s->pb, buf, buf_size);



    *p = *pict;

    p->pict_type= FF_I_TYPE;

    p->key_frame= 1;



    ff_mjpeg_encode_picture_header(s);



    s->header_bits= put_bits_count(&s->pb);



    if(avctx->pix_fmt == PIX_FMT_RGB32){

        int x, y, i;

        const int linesize= p->linesize[0];

        uint16_t (*buffer)[4]= (void *) s->rd_scratchpad;

        int left[3], top[3], topleft[3];



        for(i=0; i<3; i++){

            buffer[0][i]= 1 << (9 - 1);

        }



        for(y = 0; y < height; y++) {

            const int modified_predictor= y ? predictor : 1;

            uint8_t *ptr = p->data[0] + (linesize * y);



            if(s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < width*3*4){

                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

                return -1;

            }



            for(i=0; i<3; i++){

                top[i]= left[i]= topleft[i]= buffer[0][i];

            }

            for(x = 0; x < width; x++) {

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

                }

            }

        }

    }else{

        int mb_x, mb_y, i;

        const int mb_width  = (width  + s->mjpeg_hsample[0] - 1) / s->mjpeg_hsample[0];

        const int mb_height = (height + s->mjpeg_vsample[0] - 1) / s->mjpeg_vsample[0];



        for(mb_y = 0; mb_y < mb_height; mb_y++) {

            if(s->pb.buf_end - s->pb.buf - (put_bits_count(&s->pb)>>3) < mb_width * 4 * 3 * s->mjpeg_hsample[0] * s->mjpeg_vsample[0]){

                av_log(s->avctx, AV_LOG_ERROR, "encoded frame too large\n");

                return -1;

            }

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

                                    }

                                }else{

                                    if(x==0 && mb_x==0){

                                        pred= ptr[-linesize];

                                    }else{

                                        PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);

                                    }

                                }



                                if(i==0)

                                    ff_mjpeg_encode_dc(s, (int8_t)(*ptr - pred), m->huff_size_dc_luminance, m->huff_code_dc_luminance); //FIXME ugly

                                else

                                    ff_mjpeg_encode_dc(s, (int8_t)(*ptr - pred), m->huff_size_dc_chrominance, m->huff_code_dc_chrominance);

                            }

                        }

                    }

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

//printf("%d %d %d %d %8X\n", mb_x, mb_y, x, y, ptr);

                                PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);



                                if(i==0)

                                    ff_mjpeg_encode_dc(s, (int8_t)(*ptr - pred), m->huff_size_dc_luminance, m->huff_code_dc_luminance); //FIXME ugly

                                else

                                    ff_mjpeg_encode_dc(s, (int8_t)(*ptr - pred), m->huff_size_dc_chrominance, m->huff_code_dc_chrominance);

                            }

                        }

                    }

                }

            }

        }

    }



    emms_c();



    ff_mjpeg_encode_picture_trailer(s);

    s->picture_number++;



    flush_put_bits(&s->pb);

    return pbBufPtr(&s->pb) - s->pb.buf;

//    return (put_bits_count(&f->pb)+7)/8;

}
