static void dsound_write_sample (HWVoiceOut *hw, uint8_t *dst, int dst_len)

{

    int src_len1 = dst_len;

    int src_len2 = 0;

    int pos = hw->rpos + dst_len;

    st_sample_t *src1 = hw->mix_buf + hw->rpos;

    st_sample_t *src2 = NULL;



    if (pos > hw->samples) {

        src_len1 = hw->samples - hw->rpos;

        src2 = hw->mix_buf;

        src_len2 = dst_len - src_len1;

        pos = src_len2;

    }



    if (src_len1) {

        hw->clip (dst, src1, src_len1);

    }



    if (src_len2) {

        dst = advance (dst, src_len1 << hw->info.shift);

        hw->clip (dst, src2, src_len2);

    }



    hw->rpos = pos % hw->samples;

}
