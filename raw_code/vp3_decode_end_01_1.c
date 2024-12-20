static av_cold int vp3_decode_end(AVCodecContext *avctx)

{

    Vp3DecodeContext *s = avctx->priv_data;

    int i;



    if (avctx->is_copy && !s->current_frame.data[0])

        return 0;



    av_free(s->superblock_coding);

    av_free(s->all_fragments);

    av_free(s->coded_fragment_list[0]);

    av_free(s->dct_tokens_base);

    av_free(s->superblock_fragments);

    av_free(s->macroblock_coding);

    av_free(s->motion_val[0]);

    av_free(s->motion_val[1]);

    av_free(s->edge_emu_buffer);



    if (avctx->is_copy) return 0;



    for (i = 0; i < 16; i++) {

        free_vlc(&s->dc_vlc[i]);

        free_vlc(&s->ac_vlc_1[i]);

        free_vlc(&s->ac_vlc_2[i]);

        free_vlc(&s->ac_vlc_3[i]);

        free_vlc(&s->ac_vlc_4[i]);

    }



    free_vlc(&s->superblock_run_length_vlc);

    free_vlc(&s->fragment_run_length_vlc);

    free_vlc(&s->mode_code_vlc);

    free_vlc(&s->motion_vector_vlc);



    /* release all frames */

    if (s->golden_frame.data[0])

        ff_thread_release_buffer(avctx, &s->golden_frame);

    if (s->last_frame.data[0] && s->last_frame.type != FF_BUFFER_TYPE_COPY)

        ff_thread_release_buffer(avctx, &s->last_frame);

    /* no need to release the current_frame since it will always be pointing

     * to the same frame as either the golden or last frame */



    return 0;

}
