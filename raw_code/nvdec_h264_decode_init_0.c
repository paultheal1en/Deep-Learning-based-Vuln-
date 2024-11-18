static int nvdec_h264_decode_init(AVCodecContext *avctx)

{

    const H264Context *h = avctx->priv_data;

    const SPS       *sps = h->ps.sps;

    return ff_nvdec_decode_init(avctx, sps->ref_frame_count + sps->num_reorder_frames);

}
