static void render_slice(Vp3DecodeContext *s, int slice)

{

    int x, y, i, j, fragment;

    LOCAL_ALIGNED_16(DCTELEM, block, [64]);

    int motion_x = 0xdeadbeef, motion_y = 0xdeadbeef;

    int motion_halfpel_index;

    uint8_t *motion_source;

    int plane, first_pixel;



    if (slice >= s->c_superblock_height)

        return;



    for (plane = 0; plane < 3; plane++) {

        uint8_t *output_plane = s->current_frame.data    [plane] + s->data_offset[plane];

        uint8_t *  last_plane = s->   last_frame.data    [plane] + s->data_offset[plane];

        uint8_t *golden_plane = s-> golden_frame.data    [plane] + s->data_offset[plane];

        int stride            = s->current_frame.linesize[plane];

        int plane_width       = s->width  >> (plane && s->chroma_x_shift);

        int plane_height      = s->height >> (plane && s->chroma_y_shift);

        int8_t (*motion_val)[2] = s->motion_val[!!plane];



        int sb_x, sb_y        = slice << (!plane && s->chroma_y_shift);

        int slice_height      = sb_y + 1 + (!plane && s->chroma_y_shift);

        int slice_width       = plane ? s->c_superblock_width : s->y_superblock_width;



        int fragment_width    = s->fragment_width[!!plane];

        int fragment_height   = s->fragment_height[!!plane];

        int fragment_start    = s->fragment_start[plane];

        int do_await          = !plane && HAVE_THREADS && (s->avctx->active_thread_type&FF_THREAD_FRAME);



        if (!s->flipped_image) stride = -stride;

        if (CONFIG_GRAY && plane && (s->avctx->flags & CODEC_FLAG_GRAY))

            continue;



        /* for each superblock row in the slice (both of them)... */

        for (; sb_y < slice_height; sb_y++) {



            /* for each superblock in a row... */

            for (sb_x = 0; sb_x < slice_width; sb_x++) {



                /* for each block in a superblock... */

                for (j = 0; j < 16; j++) {

                    x = 4*sb_x + hilbert_offset[j][0];

                    y = 4*sb_y + hilbert_offset[j][1];

                    fragment = y*fragment_width + x;



                    i = fragment_start + fragment;



                    // bounds check

                    if (x >= fragment_width || y >= fragment_height)

                        continue;



                first_pixel = 8*y*stride + 8*x;



                if (do_await && s->all_fragments[i].coding_method != MODE_INTRA)

                    await_reference_row(s, &s->all_fragments[i], motion_val[fragment][1], (16*y) >> s->chroma_y_shift);



                /* transform if this block was coded */

                if (s->all_fragments[i].coding_method != MODE_COPY) {

                    if ((s->all_fragments[i].coding_method == MODE_USING_GOLDEN) ||

                        (s->all_fragments[i].coding_method == MODE_GOLDEN_MV))

                        motion_source= golden_plane;

                    else

                        motion_source= last_plane;



                    motion_source += first_pixel;

                    motion_halfpel_index = 0;



                    /* sort out the motion vector if this fragment is coded

                     * using a motion vector method */

                    if ((s->all_fragments[i].coding_method > MODE_INTRA) &&

                        (s->all_fragments[i].coding_method != MODE_USING_GOLDEN)) {

                        int src_x, src_y;

                        motion_x = motion_val[fragment][0];

                        motion_y = motion_val[fragment][1];



                        src_x= (motion_x>>1) + 8*x;

                        src_y= (motion_y>>1) + 8*y;



                        motion_halfpel_index = motion_x & 0x01;

                        motion_source += (motion_x >> 1);



                        motion_halfpel_index |= (motion_y & 0x01) << 1;

                        motion_source += ((motion_y >> 1) * stride);



                        if(src_x<0 || src_y<0 || src_x + 9 >= plane_width || src_y + 9 >= plane_height){

                            uint8_t *temp= s->edge_emu_buffer;

                            if(stride<0) temp -= 8*stride;



                            s->dsp.emulated_edge_mc(temp, motion_source, stride, 9, 9, src_x, src_y, plane_width, plane_height);

                            motion_source= temp;

                        }

                    }





                    /* first, take care of copying a block from either the

                     * previous or the golden frame */

                    if (s->all_fragments[i].coding_method != MODE_INTRA) {

                        /* Note, it is possible to implement all MC cases with

                           put_no_rnd_pixels_l2 which would look more like the

                           VP3 source but this would be slower as

                           put_no_rnd_pixels_tab is better optimzed */

                        if(motion_halfpel_index != 3){

                            s->dsp.put_no_rnd_pixels_tab[1][motion_halfpel_index](

                                output_plane + first_pixel,

                                motion_source, stride, 8);

                        }else{

                            int d= (motion_x ^ motion_y)>>31; // d is 0 if motion_x and _y have the same sign, else -1

                            s->dsp.put_no_rnd_pixels_l2[1](

                                output_plane + first_pixel,

                                motion_source - d,

                                motion_source + stride + 1 + d,

                                stride, 8);

                        }

                    }



                        s->dsp.clear_block(block);



                    /* invert DCT and place (or add) in final output */



                    if (s->all_fragments[i].coding_method == MODE_INTRA) {

                        int index;

                        index = vp3_dequant(s, s->all_fragments + i, plane, 0, block);

                        if (index > 63)

                            continue;

                        if(s->avctx->idct_algo!=FF_IDCT_VP3)

                            block[0] += 128<<3;

                        s->dsp.idct_put(

                            output_plane + first_pixel,

                            stride,

                            block);

                    } else {

                        int index = vp3_dequant(s, s->all_fragments + i, plane, 1, block);

                        if (index > 63)

                            continue;

                        if (index > 0) {

                        s->dsp.idct_add(

                            output_plane + first_pixel,

                            stride,

                            block);

                        } else {

                            s->dsp.vp3_idct_dc_add(output_plane + first_pixel, stride, block);

                        }

                    }

                } else {



                    /* copy directly from the previous frame */

                    s->dsp.put_pixels_tab[1][0](

                        output_plane + first_pixel,

                        last_plane + first_pixel,

                        stride, 8);



                }

                }

            }



            // Filter up to the last row in the superblock row

            if (!s->skip_loop_filter)

                apply_loop_filter(s, plane, 4*sb_y - !!sb_y, FFMIN(4*sb_y+3, fragment_height-1));

        }

    }



     /* this looks like a good place for slice dispatch... */

     /* algorithm:

      *   if (slice == s->macroblock_height - 1)

      *     dispatch (both last slice & 2nd-to-last slice);

      *   else if (slice > 0)

      *     dispatch (slice - 1);

      */



    vp3_draw_horiz_band(s, FFMIN((32 << s->chroma_y_shift) * (slice + 1) -16, s->height-16));

}
