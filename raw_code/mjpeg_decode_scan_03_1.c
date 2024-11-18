static int mjpeg_decode_scan(MJpegDecodeContext *s, int nb_components, int Ah, int Al,

                             const uint8_t *mb_bitmask, const AVFrame *reference){

    int i, mb_x, mb_y;

    uint8_t* data[MAX_COMPONENTS];

    const uint8_t *reference_data[MAX_COMPONENTS];

    int linesize[MAX_COMPONENTS];

    GetBitContext mb_bitmask_gb;



    if (mb_bitmask) {

        init_get_bits(&mb_bitmask_gb, mb_bitmask, s->mb_width*s->mb_height);




    if(s->flipped && s->avctx->flags & CODEC_FLAG_EMU_EDGE) {

        av_log(s->avctx, AV_LOG_ERROR, "Can not flip image with CODEC_FLAG_EMU_EDGE set!\n");

        s->flipped = 0;


    for(i=0; i < nb_components; i++) {

        int c = s->comp_index[i];

        data[c] = s->picture_ptr->data[c];

        reference_data[c] = reference ? reference->data[c] : NULL;

        linesize[c]=s->linesize[c];

        s->coefs_finished[c] |= 1;

        if(s->flipped) {

            //picture should be flipped upside-down for this codec

            int offset = (linesize[c] * (s->v_scount[i] * (8 * s->mb_height -((s->height/s->v_max)&7)) - 1 ));

            data[c] += offset;

            reference_data[c] += offset;

            linesize[c] *= -1;





    for(mb_y = 0; mb_y < s->mb_height; mb_y++) {

        for(mb_x = 0; mb_x < s->mb_width; mb_x++) {

            const int copy_mb = mb_bitmask && !get_bits1(&mb_bitmask_gb);



            if (s->restart_interval && !s->restart_count)

                s->restart_count = s->restart_interval;







            for(i=0;i<nb_components;i++) {

                uint8_t *ptr;

                int n, h, v, x, y, c, j;

                int block_offset;

                n = s->nb_blocks[i];

                c = s->comp_index[i];

                h = s->h_scount[i];

                v = s->v_scount[i];

                x = 0;

                y = 0;

                for(j=0;j<n;j++) {

                    block_offset = (((linesize[c] * (v * mb_y + y) * 8) +

                                     (h * mb_x + x) * 8) >> s->avctx->lowres);



                    if(s->interlaced && s->bottom_field)

                        block_offset += linesize[c] >> 1;

                    ptr = data[c] + block_offset;

                    if(!s->progressive) {

                        if (copy_mb) {

                            mjpeg_copy_block(ptr, reference_data[c] + block_offset, linesize[c], s->avctx->lowres);

                        } else {

                        s->dsp.clear_block(s->block);

                        if(decode_block(s, s->block, i,

                                     s->dc_index[i], s->ac_index[i],

                                     s->quant_matrixes[ s->quant_index[c] ]) < 0) {

                            av_log(s->avctx, AV_LOG_ERROR, "error y=%d x=%d\n", mb_y, mb_x);



                        s->dsp.idct_put(ptr, linesize[c], s->block);


                    } else {

                        int block_idx = s->block_stride[c] * (v * mb_y + y) + (h * mb_x + x);

                        DCTELEM *block = s->blocks[c][block_idx];

                        if(Ah)

                            block[0] += get_bits1(&s->gb) * s->quant_matrixes[ s->quant_index[c] ][0] << Al;

                        else if(decode_dc_progressive(s, block, i, s->dc_index[i], s->quant_matrixes[ s->quant_index[c] ], Al) < 0) {

                            av_log(s->avctx, AV_LOG_ERROR, "error y=%d x=%d\n", mb_y, mb_x);




//                    av_log(s->avctx, AV_LOG_DEBUG, "mb: %d %d processed\n", mb_y, mb_x);

//av_log(NULL, AV_LOG_DEBUG, "%d %d %d %d %d %d %d %d \n", mb_x, mb_y, x, y, c, s->bottom_field, (v * mb_y + y) * 8, (h * mb_x + x) * 8);

                    if (++x == h) {

                        x = 0;

                        y++;






            if (s->restart_interval && !--s->restart_count) {

                align_get_bits(&s->gb);

                skip_bits(&s->gb, 16); /* skip RSTn */

                for (i=0; i<nb_components; i++) /* reset dc */

                    s->last_dc[i] = 1024;




    return 0;
