static int output_frame(H264Context *h, AVFrame *dst, H264Picture *srcp)
{
    AVFrame *src = srcp->f;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(src->format);
    int i;
    int ret = av_frame_ref(dst, src);
    if (ret < 0)
        return ret;
    av_dict_set(&dst->metadata, "stereo_mode", ff_h264_sei_stereo_mode(h), 0);
    if (srcp->sei_recovery_frame_cnt == 0)
        dst->key_frame = 1;
    if (!srcp->crop)
        return 0;
    for (i = 0; i < desc->nb_components; i++) {
        int hshift = (i > 0) ? desc->log2_chroma_w : 0;
        int vshift = (i > 0) ? desc->log2_chroma_h : 0;
        int off    = ((srcp->crop_left >> hshift) << h->pixel_shift) +
                      (srcp->crop_top  >> vshift) * dst->linesize[i];
        dst->data[i] += off;
    }
    return 0;
}