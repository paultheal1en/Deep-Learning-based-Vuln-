static int ljpeg_decode_yuv_scan(MJpegDecodeContext *s, int predictor, int point_transform){

    int i, mb_x, mb_y;

    const int nb_components=3;



    for(mb_y = 0; mb_y < s->mb_height; mb_y++) {

        for(mb_x = 0; mb_x < s->mb_width; mb_x++) {

            if (s->restart_interval && !s->restart_count)

                s->restart_count = s->restart_interval;



            if(mb_x==0 || mb_y==0 || s->interlaced){

                for(i=0;i<nb_components;i++) {

                    uint8_t *ptr;

                    int n, h, v, x, y, c, j, linesize;

                    n = s->nb_blocks[i];

                    c = s->comp_index[i];

                    h = s->h_scount[i];

                    v = s->v_scount[i];

                    x = 0;

                    y = 0;

                    linesize= s->linesize[c];



                    for(j=0; j<n; j++) {

                        int pred;



                        ptr = s->picture.data[c] + (linesize * (v * mb_y + y)) + (h * mb_x + x); //FIXME optimize this crap

                        if(y==0 && mb_y==0){

                            if(x==0 && mb_x==0){

                                pred= 128 << point_transform;

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



                        if (s->interlaced && s->bottom_field)

                            ptr += linesize >> 1;

                        *ptr= pred + (mjpeg_decode_dc(s, s->dc_index[i]) << point_transform);



                        if (++x == h) {

                            x = 0;

                            y++;

                        }

                    }

                }

            }else{

                for(i=0;i<nb_components;i++) {

                    uint8_t *ptr;

                    int n, h, v, x, y, c, j, linesize;

                    n = s->nb_blocks[i];

                    c = s->comp_index[i];

                    h = s->h_scount[i];

                    v = s->v_scount[i];

                    x = 0;

                    y = 0;

                    linesize= s->linesize[c];



                    for(j=0; j<n; j++) {

                        int pred;



                        ptr = s->picture.data[c] + (linesize * (v * mb_y + y)) + (h * mb_x + x); //FIXME optimize this crap

                        PREDICT(pred, ptr[-linesize-1], ptr[-linesize], ptr[-1], predictor);

                        *ptr= pred + (mjpeg_decode_dc(s, s->dc_index[i]) << point_transform);

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
