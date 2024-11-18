static int rprobe(AVFormatContext *s, uint8_t *enc_header, const uint8_t *r_val)

{

    OMAContext *oc = s->priv_data;

    unsigned int pos;

    struct AVDES av_des;



    if (!enc_header || !r_val)

        return -1;



    /* m_val */

    av_des_init(&av_des, r_val, 192, 1);

    av_des_crypt(&av_des, oc->m_val, &enc_header[48], 1, NULL, 1);



    /* s_val */

    av_des_init(&av_des, oc->m_val, 64, 0);

    av_des_crypt(&av_des, oc->s_val, NULL, 1, NULL, 0);



    /* sm_val */

    pos = OMA_ENC_HEADER_SIZE + oc->k_size + oc->e_size;

    av_des_init(&av_des, oc->s_val, 64, 0);

    av_des_mac(&av_des, oc->sm_val, &enc_header[pos], (oc->i_size >> 3));



    pos += oc->i_size;



    return memcmp(&enc_header[pos], oc->sm_val, 8) ? -1 : 0;

}
