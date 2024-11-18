static void add_xblock(DWTELEM *dst, uint8_t *src, uint8_t *obmc, int s_x, int s_y, int b_w, int b_h, int mv_x, int mv_y, int w, int h, int dst_stride, int src_stride, int obmc_stride, int mb_type, int add){

    uint8_t tmp[src_stride*(b_h+5)]; //FIXME move to context to gurantee alignment

    int x,y;



    if(s_x<0){

        obmc -= s_x;

        b_w += s_x;

        s_x=0;

    }else if(s_x + b_w > w){

        b_w = w - s_x;

    }

    if(s_y<0){

        obmc -= s_y*obmc_stride;

        b_h += s_y;

        s_y=0;

    }else if(s_y + b_h> h){

        b_h = h - s_y;

    }



    if(b_w<=0 || b_h<=0) return;

    

    dst += s_x + s_y*dst_stride;

    

    if(mb_type==1){

        src += s_x + s_y*src_stride;

        for(y=0; y < b_h; y++){

            for(x=0; x < b_w; x++){

                if(add) dst[x + y*dst_stride] += obmc[x + y*obmc_stride] * 128 * (256/OBMC_MAX);

                else    dst[x + y*dst_stride] -= obmc[x + y*obmc_stride] * 128 * (256/OBMC_MAX);

            }

        }

    }else{

        int dx= mv_x&15;

        int dy= mv_y&15;

//        int dxy= (mv_x&1) + 2*(mv_y&1);



        s_x += (mv_x>>4) - 2;

        s_y += (mv_y>>4) - 2;

        src += s_x + s_y*src_stride;

        //use dsputil

    

        if(   (unsigned)s_x >= w - b_w - 4

           || (unsigned)s_y >= h - b_h - 4){

            ff_emulated_edge_mc(tmp + 32, src, src_stride, b_w+5, b_h+5, s_x, s_y, w, h);

            src= tmp + 32;

        }



        if(mb_type==0){

            mc_block(tmp, src, tmp + 64+8, src_stride, b_w, b_h, dx, dy);

        }else{

            int sum=0;

            for(y=0; y < b_h; y++){

                for(x=0; x < b_w; x++){

                    sum += src[x+  y*src_stride];

                }

            }

            sum= (sum + b_h*b_w/2) / (b_h*b_w);

            for(y=0; y < b_h; y++){

                for(x=0; x < b_w; x++){

                    tmp[x + y*src_stride]= sum; 

                }

            }

        }



        for(y=0; y < b_h; y++){

            for(x=0; x < b_w; x++){

                if(add) dst[x + y*dst_stride] += obmc[x + y*obmc_stride] * tmp[x + y*src_stride] * (256/OBMC_MAX);

                else    dst[x + y*dst_stride] -= obmc[x + y*obmc_stride] * tmp[x + y*src_stride] * (256/OBMC_MAX);

            }

        }

    }

}
