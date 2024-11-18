static int decode_update_thread_context(AVCodecContext *dst, const AVCodecContext *src){

    H264Context *h= dst->priv_data, *h1= src->priv_data;

    MpegEncContext * const s = &h->s, * const s1 = &h1->s;

    int inited = s->context_initialized, err;

    int i;



    if(dst == src || !s1->context_initialized) return 0;



    err = ff_mpeg_update_thread_context(dst, src);

    if(err) return err;



    //FIXME handle width/height changing

    if(!inited){

        for(i = 0; i < MAX_SPS_COUNT; i++)

            av_freep(h->sps_buffers + i);



        for(i = 0; i < MAX_PPS_COUNT; i++)

            av_freep(h->pps_buffers + i);



        memcpy(&h->s + 1, &h1->s + 1, sizeof(H264Context) - sizeof(MpegEncContext)); //copy all fields after MpegEnc

        memset(h->sps_buffers, 0, sizeof(h->sps_buffers));

        memset(h->pps_buffers, 0, sizeof(h->pps_buffers));

        if (ff_h264_alloc_tables(h) < 0) {

            av_log(dst, AV_LOG_ERROR, "Could not allocate memory for h264\n");

            return AVERROR(ENOMEM);

        }

        context_init(h);



        for(i=0; i<2; i++){

            h->rbsp_buffer[i] = NULL;

            h->rbsp_buffer_size[i] = 0;

        }



        h->thread_context[0] = h;



        // frame_start may not be called for the next thread (if it's decoding a bottom field)

        // so this has to be allocated here

        h->s.obmc_scratchpad = av_malloc(16*6*s->linesize);



        s->dsp.clear_blocks(h->mb);

        s->dsp.clear_blocks(h->mb+(24*16<<h->pixel_shift));

    }



    //extradata/NAL handling

    h->is_avc          = h1->is_avc;



    //SPS/PPS

    copy_parameter_set((void**)h->sps_buffers, (void**)h1->sps_buffers, MAX_SPS_COUNT, sizeof(SPS));

    h->sps             = h1->sps;

    copy_parameter_set((void**)h->pps_buffers, (void**)h1->pps_buffers, MAX_PPS_COUNT, sizeof(PPS));

    h->pps             = h1->pps;



    //Dequantization matrices

    //FIXME these are big - can they be only copied when PPS changes?

    copy_fields(h, h1, dequant4_buffer, dequant4_coeff);



    for(i=0; i<6; i++)

        h->dequant4_coeff[i] = h->dequant4_buffer[0] + (h1->dequant4_coeff[i] - h1->dequant4_buffer[0]);



    for(i=0; i<6; i++)

        h->dequant8_coeff[i] = h->dequant8_buffer[0] + (h1->dequant8_coeff[i] - h1->dequant8_buffer[0]);



    h->dequant_coeff_pps = h1->dequant_coeff_pps;



    //POC timing

    copy_fields(h, h1, poc_lsb, redundant_pic_count);



    //reference lists

    copy_fields(h, h1, ref_count, list_count);

    copy_fields(h, h1, ref_list,  intra_gb);

    copy_fields(h, h1, short_ref, cabac_init_idc);



    copy_picture_range(h->short_ref,   h1->short_ref,   32, s, s1);

    copy_picture_range(h->long_ref,    h1->long_ref,    32, s, s1);

    copy_picture_range(h->delayed_pic, h1->delayed_pic, MAX_DELAYED_PIC_COUNT+2, s, s1);



    h->last_slice_type = h1->last_slice_type;

    h->sync            = h1->sync;



    if(!s->current_picture_ptr) return 0;



    if(!s->dropable) {

        err = ff_h264_execute_ref_pic_marking(h, h->mmco, h->mmco_index);

        h->prev_poc_msb     = h->poc_msb;

        h->prev_poc_lsb     = h->poc_lsb;

    }

    h->prev_frame_num_offset= h->frame_num_offset;

    h->prev_frame_num       = h->frame_num;

    h->outputed_poc         = h->next_outputed_poc;



    return err;

}
