int ff_mjpeg_encode_stuffing(MpegEncContext *s)

{

    int i;

    PutBitContext *pbc = &s->pb;

    int mb_y = s->mb_y - !s->mb_x;

    int ret;

    MJpegContext *m;



    m = s->mjpeg_ctx;



    if (s->huffman == HUFFMAN_TABLE_OPTIMAL) {

        ff_mjpeg_build_optimal_huffman(m);



        // Replace the VLCs with the optimal ones.

        // The default ones may be used for trellis during quantization.

        ff_init_uni_ac_vlc(m->huff_size_ac_luminance,   m->uni_ac_vlc_len);

        ff_init_uni_ac_vlc(m->huff_size_ac_chrominance, m->uni_chroma_ac_vlc_len);

        s->intra_ac_vlc_length      =

        s->intra_ac_vlc_last_length = m->uni_ac_vlc_len;

        s->intra_chroma_ac_vlc_length      =

        s->intra_chroma_ac_vlc_last_length = m->uni_chroma_ac_vlc_len;

    }



    ff_mjpeg_encode_picture_header(s->avctx, &s->pb, &s->intra_scantable,

                                   s->pred, s->intra_matrix, s->chroma_intra_matrix);

    ff_mjpeg_encode_picture_frame(s);



    ret = ff_mpv_reallocate_putbitbuffer(s, put_bits_count(&s->pb) / 8 + 100,

                                            put_bits_count(&s->pb) / 4 + 1000);



    if (ret < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "Buffer reallocation failed\n");

        goto fail;

    }



    ff_mjpeg_escape_FF(pbc, s->esc_pos);



    if((s->avctx->active_thread_type & FF_THREAD_SLICE) && mb_y < s->mb_height)

        put_marker(pbc, RST0 + (mb_y&7));

    s->esc_pos = put_bits_count(pbc) >> 3;

fail:



    for(i=0; i<3; i++)

        s->last_dc[i] = 128 << s->intra_dc_precision;



    return ret;

}
