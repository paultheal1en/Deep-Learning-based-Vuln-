void ff_h264_filter_mb_fast(const H264Context *h, H264SliceContext *sl,

                            int mb_x, int mb_y, uint8_t *img_y,

                            uint8_t *img_cb, uint8_t *img_cr,

                            unsigned int linesize, unsigned int uvlinesize)

{

    assert(!FRAME_MBAFF(h));

    if(!h->h264dsp.h264_loop_filter_strength || h->pps.chroma_qp_diff) {

        ff_h264_filter_mb(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize);

        return;

    }



#if CONFIG_SMALL

    h264_filter_mb_fast_internal(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize, h->pixel_shift);

#else

    if(h->pixel_shift){

        h264_filter_mb_fast_internal(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize, 1);

    }else{

        h264_filter_mb_fast_internal(h, sl, mb_x, mb_y, img_y, img_cb, img_cr, linesize, uvlinesize, 0);

    }

#endif

}
