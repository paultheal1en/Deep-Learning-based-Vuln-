static int decode_frame(AVCodecContext *avctx, void *data, int *data_size, uint8_t *buf, int buf_size){

    HYuvContext *s = avctx->priv_data;

    const int width= s->width;

    const int width2= s->width>>1;

    const int height= s->height;

    int fake_ystride, fake_ustride, fake_vstride;

    AVFrame * const p= &s->picture;



    AVFrame *picture = data;



    *data_size = 0;



    /* no supplementary picture */

    if (buf_size == 0)

        return 0;



    bswap_buf((uint32_t*)s->bitstream_buffer, (uint32_t*)buf, buf_size/4);

    

    init_get_bits(&s->gb, s->bitstream_buffer, buf_size);



    p->reference= 0;

    if(avctx->get_buffer(avctx, p) < 0){

        fprintf(stderr, "get_buffer() failed\n");

        return -1;

    }



    fake_ystride= s->interlaced ? p->linesize[0]*2  : p->linesize[0];

    fake_ustride= s->interlaced ? p->linesize[1]*2  : p->linesize[1];

    fake_vstride= s->interlaced ? p->linesize[2]*2  : p->linesize[2];

    

    s->last_slice_end= 0;

        

    if(s->bitstream_bpp<24){

        int y, cy;

        int lefty, leftu, leftv;

        int lefttopy, lefttopu, lefttopv;

        

        if(s->yuy2){

            p->data[0][3]= get_bits(&s->gb, 8);

            p->data[0][2]= get_bits(&s->gb, 8);

            p->data[0][1]= get_bits(&s->gb, 8);

            p->data[0][0]= get_bits(&s->gb, 8);

            

            fprintf(stderr, "YUY2 output isnt implemenetd yet\n");

            return -1;

        }else{

        

            leftv= p->data[2][0]= get_bits(&s->gb, 8);

            lefty= p->data[0][1]= get_bits(&s->gb, 8);

            leftu= p->data[1][0]= get_bits(&s->gb, 8);

                   p->data[0][0]= get_bits(&s->gb, 8);

        

            switch(s->predictor){

            case LEFT:

            case PLANE:

                decode_422_bitstream(s, width-2);

                lefty= add_left_prediction(p->data[0] + 2, s->temp[0], width-2, lefty);

                if(!(s->flags&CODEC_FLAG_GRAY)){

                    leftu= add_left_prediction(p->data[1] + 1, s->temp[1], width2-1, leftu);

                    leftv= add_left_prediction(p->data[2] + 1, s->temp[2], width2-1, leftv);

                }



                for(cy=y=1; y<s->height; y++,cy++){

                    uint8_t *ydst, *udst, *vdst;

                    

                    if(s->bitstream_bpp==12){

                        decode_gray_bitstream(s, width);

                    

                        ydst= p->data[0] + p->linesize[0]*y;



                        lefty= add_left_prediction(ydst, s->temp[0], width, lefty);

                        if(s->predictor == PLANE){

                            if(y>s->interlaced)

                                s->dsp.add_bytes(ydst, ydst - fake_ystride, width);

                        }

                        y++;

                        if(y>=s->height) break;

                    }

                    

                    draw_slice(s, y);

                    

                    ydst= p->data[0] + p->linesize[0]*y;

                    udst= p->data[1] + p->linesize[1]*cy;

                    vdst= p->data[2] + p->linesize[2]*cy;

                    

                    decode_422_bitstream(s, width);

                    lefty= add_left_prediction(ydst, s->temp[0], width, lefty);

                    if(!(s->flags&CODEC_FLAG_GRAY)){

                        leftu= add_left_prediction(udst, s->temp[1], width2, leftu);

                        leftv= add_left_prediction(vdst, s->temp[2], width2, leftv);

                    }

                    if(s->predictor == PLANE){

                        if(cy>s->interlaced){

                            s->dsp.add_bytes(ydst, ydst - fake_ystride, width);

                            if(!(s->flags&CODEC_FLAG_GRAY)){

                                s->dsp.add_bytes(udst, udst - fake_ustride, width2);

                                s->dsp.add_bytes(vdst, vdst - fake_vstride, width2);

                            }

                        }

                    }

                }

                draw_slice(s, height);

                

                break;

            case MEDIAN:

                /* first line except first 2 pixels is left predicted */

                decode_422_bitstream(s, width-2);

                lefty= add_left_prediction(p->data[0] + 2, s->temp[0], width-2, lefty);

                if(!(s->flags&CODEC_FLAG_GRAY)){

                    leftu= add_left_prediction(p->data[1] + 1, s->temp[1], width2-1, leftu);

                    leftv= add_left_prediction(p->data[2] + 1, s->temp[2], width2-1, leftv);

                }

                

                cy=y=1;

                

                /* second line is left predicted for interlaced case */

                if(s->interlaced){

                    decode_422_bitstream(s, width);

                    lefty= add_left_prediction(p->data[0] + p->linesize[0], s->temp[0], width, lefty);

                    if(!(s->flags&CODEC_FLAG_GRAY)){

                        leftu= add_left_prediction(p->data[1] + p->linesize[2], s->temp[1], width2, leftu);

                        leftv= add_left_prediction(p->data[2] + p->linesize[1], s->temp[2], width2, leftv);

                    }

                    y++; cy++;

                }



                /* next 4 pixels are left predicted too */

                decode_422_bitstream(s, 4);

                lefty= add_left_prediction(p->data[0] + fake_ystride, s->temp[0], 4, lefty);

                if(!(s->flags&CODEC_FLAG_GRAY)){

                    leftu= add_left_prediction(p->data[1] + fake_ustride, s->temp[1], 2, leftu);

                    leftv= add_left_prediction(p->data[2] + fake_vstride, s->temp[2], 2, leftv);

                }



                /* next line except the first 4 pixels is median predicted */

                lefttopy= p->data[0][3];

                decode_422_bitstream(s, width-4);

                add_median_prediction(p->data[0] + fake_ystride+4, p->data[0]+4, s->temp[0], width-4, &lefty, &lefttopy);

                if(!(s->flags&CODEC_FLAG_GRAY)){

                    lefttopu= p->data[1][1];

                    lefttopv= p->data[2][1];

                    add_median_prediction(p->data[1] + fake_ustride+2, p->data[1]+2, s->temp[1], width2-2, &leftu, &lefttopu);

                    add_median_prediction(p->data[2] + fake_vstride+2, p->data[2]+2, s->temp[2], width2-2, &leftv, &lefttopv);

                }

                y++; cy++;

                

                for(; y<height; y++,cy++){

                    uint8_t *ydst, *udst, *vdst;



                    if(s->bitstream_bpp==12){

                        while(2*cy > y){

                            decode_gray_bitstream(s, width);

                            ydst= p->data[0] + p->linesize[0]*y;

                            add_median_prediction(ydst, ydst - fake_ystride, s->temp[0], width, &lefty, &lefttopy);

                            y++;

                        }

                        if(y>=height) break;

                    }

                    draw_slice(s, y);



                    decode_422_bitstream(s, width);



                    ydst= p->data[0] + p->linesize[0]*y;

                    udst= p->data[1] + p->linesize[1]*cy;

                    vdst= p->data[2] + p->linesize[2]*cy;



                    add_median_prediction(ydst, ydst - fake_ystride, s->temp[0], width, &lefty, &lefttopy);

                    if(!(s->flags&CODEC_FLAG_GRAY)){

                        add_median_prediction(udst, udst - fake_ustride, s->temp[1], width2, &leftu, &lefttopu);

                        add_median_prediction(vdst, vdst - fake_vstride, s->temp[2], width2, &leftv, &lefttopv);

                    }

                }



                draw_slice(s, height);

                break;

            }

        }

    }else{

        int y;

        int leftr, leftg, leftb;

        const int last_line= (height-1)*p->linesize[0];

        

        if(s->bitstream_bpp==32){

                   p->data[0][last_line+3]= get_bits(&s->gb, 8);

            leftr= p->data[0][last_line+2]= get_bits(&s->gb, 8);

            leftg= p->data[0][last_line+1]= get_bits(&s->gb, 8);

            leftb= p->data[0][last_line+0]= get_bits(&s->gb, 8);

        }else{

            leftr= p->data[0][last_line+2]= get_bits(&s->gb, 8);

            leftg= p->data[0][last_line+1]= get_bits(&s->gb, 8);

            leftb= p->data[0][last_line+0]= get_bits(&s->gb, 8);

            skip_bits(&s->gb, 8);

        }

        

        if(s->bgr32){

            switch(s->predictor){

            case LEFT:

            case PLANE:

                decode_bgr_bitstream(s, width-1);

                add_left_prediction_bgr32(p->data[0] + last_line+4, s->temp[0], width-1, &leftr, &leftg, &leftb);



                for(y=s->height-2; y>=0; y--){ //yes its stored upside down

                    decode_bgr_bitstream(s, width);

                    

                    add_left_prediction_bgr32(p->data[0] + p->linesize[0]*y, s->temp[0], width, &leftr, &leftg, &leftb);

                    if(s->predictor == PLANE){

                        if((y&s->interlaced)==0){

                            s->dsp.add_bytes(p->data[0] + p->linesize[0]*y, 

                                             p->data[0] + p->linesize[0]*y + fake_ystride, fake_ystride);

                        }

                    }

                }

                draw_slice(s, height); // just 1 large slice as this isnt possible in reverse order

                break;

            default:

                fprintf(stderr, "prediction type not supported!\n");

            }

        }else{



            fprintf(stderr, "BGR24 output isnt implemenetd yet\n");

            return -1;

        }

    }

    emms_c();

    

    *picture= *p;

    

    avctx->release_buffer(avctx, p);



    *data_size = sizeof(AVFrame);

    

    return (get_bits_count(&s->gb)+7)>>3;

}
