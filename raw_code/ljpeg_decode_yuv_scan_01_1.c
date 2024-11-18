static int ljpeg_decode_yuv_scan(MJpegDecodeContext *s, int predictor,

                                 int point_transform)

{

    int i, mb_x, mb_y;

    const int nb_components=s->nb_components;

    int bits= (s->bits+7)&~7;

    int resync_mb_y = 0;

    int resync_mb_x = 0;



    point_transform += bits - s->bits;



    av_assert0(nb_components>=1 && nb_components<=3);



    for (mb_y = 0; mb_y < s->mb_height; mb_y++) {

        for (mb_x = 0; mb_x < s->mb_width; mb_x++) {

            if (s->restart_interval && !s->restart_count){

                s->restart_count = s->restart_interval;

                resync_mb_x = mb_x;

                resync_mb_y = mb_y;

            }



            if(!mb_x || mb_y == resync_mb_y || mb_y == resync_mb_y+1 && mb_x < resync_mb_x || s->interlaced){

                int toprow  = mb_y == resync_mb_y || mb_y == resync_mb_y+1 && mb_x < resync_mb_x;

                int leftcol = !mb_x || mb_y == resync_mb_y && mb_x == resync_mb_x;

                for (i = 0; i < nb_components; i++) {

                    uint8_t *ptr;

                    uint16_t *ptr16;

                    int n, h, v, x, y, c, j, linesize;

                    n = s->nb_blocks[i];

                    c = s->comp_index[i];

                    h = s->h_scount[i];

                    v = s->v_scount[i];

                    x = 0;

                    y = 0;

                    linesize= s->linesize[c];



                    if(bits>8) linesize /= 2;



                    for(j=0; j<n; j++) {

                        int pred, dc;



                        dc = mjpeg_decode_dc(s, s->dc_index[i]);

                        if(dc == 0xFFFF)

                            return -1;

                        if(bits<=8){

                        ptr = s->picture.data[c] + (linesize * (v * mb_y + y)) + (h * mb_x + x); //FIXME optimize this crap

                        if(y==0 && toprow){

                            if(x==0 && leftcol){

                                pred= 1 << (bits - 1);

                            }else{

                                pred= ptr[-1];

                            }

                        }else{

                            if(x==0 && leftcol){

                                pred= ptr[-linesize];

                            }else{

                                PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);

                            }

                        }



                        if (s->interlaced && s->bottom_field)

                            ptr += linesize >> 1;

                        pred &= (-1)<<(8-s->bits);

                        *ptr= pred + (dc << point_transform);

                        }else{

                            ptr16 = (uint16_t*)(s->picture.data[c] + 2*(linesize * (v * mb_y + y)) + 2*(h * mb_x + x)); //FIXME optimize this crap

                            if(y==0 && toprow){

                                if(x==0 && leftcol){

                                    pred= 1 << (bits - 1);

                                }else{

                                    pred= ptr16[-1];

                                }

                            }else{

                                if(x==0 && leftcol){

                                    pred= ptr16[-linesize];

                                }else{

                                    PREDICT(pred, ptr16[-linesize-1], ptr16[-linesize], ptr16[-1], predictor);

                                }

                            }



                            if (s->interlaced && s->bottom_field)

                                ptr16 += linesize >> 1;

                            pred &= (-1)<<(16-s->bits);

                            *ptr16= pred + (dc << point_transform);

                        }

                        if (++x == h) {

                            x = 0;

                            y++;

                        }

                    }

                }

            } else {

                for (i = 0; i < nb_components; i++) {

                    uint8_t *ptr;

                    uint16_t *ptr16;

                    int n, h, v, x, y, c, j, linesize, dc;

                    n        = s->nb_blocks[i];

                    c        = s->comp_index[i];

                    h        = s->h_scount[i];

                    v        = s->v_scount[i];

                    x        = 0;

                    y        = 0;

                    linesize = s->linesize[c];



                    if(bits>8) linesize /= 2;



                    for (j = 0; j < n; j++) {

                        int pred;



                        dc = mjpeg_decode_dc(s, s->dc_index[i]);

                        if(dc == 0xFFFF)

                            return -1;

                        if(bits<=8){

                            ptr = s->picture.data[c] +

                              (linesize * (v * mb_y + y)) +

                              (h * mb_x + x); //FIXME optimize this crap

                            PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);



                            pred &= (-1)<<(8-s->bits);

                            *ptr = pred + (dc << point_transform);

                        }else{

                            ptr16 = (uint16_t*)(s->picture.data[c] + 2*(linesize * (v * mb_y + y)) + 2*(h * mb_x + x)); //FIXME optimize this crap

                            PREDICT(pred, ptr16[-linesize-1], ptr16[-linesize], ptr16[-1], predictor);



                            pred &= (-1)<<(16-s->bits);

                            *ptr16= pred + (dc << point_transform);

                        }



                        if (++x == h) {

                            x = 0;

                            y++;

                        }

                    }

                }

            }

            if (s->restart_interval && !--s->restart_count) {

                align_get_bits(&s->gb);

                skip_bits(&s->gb, 16); /* skip RSTn */

            }

        }

    }

    return 0;

}