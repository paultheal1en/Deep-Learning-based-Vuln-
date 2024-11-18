static av_cold int decode_init_thread_copy(AVCodecContext *avctx)
{
    HYuvContext *s = avctx->priv_data;
    int i, ret;
    if ((ret = ff_huffyuv_alloc_temp(s)) < 0) {
        ff_huffyuv_common_end(s);
        return ret;
    }
    for (i = 0; i < 8; i++)
        s->vlc[i].table = NULL;
    if (s->version >= 2) {
        if ((ret = read_huffman_tables(s, avctx->extradata + 4,
                                       avctx->extradata_size)) < 0)
            return ret;
    } else {
        if ((ret = read_old_huffman_tables(s)) < 0)
            return ret;
    }
    return 0;
}