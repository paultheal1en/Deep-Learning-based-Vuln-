static void dca_exss_parse_header(DCAContext *s)
{
    int asset_size[8];
    int ss_index;
    int blownup;
    int num_audiop = 1;
    int num_assets = 1;
    int active_ss_mask[8];
    int i, j;
    int start_posn;
    int hdrsize;
    uint32_t mkr;
    if (get_bits_left(&s->gb) < 52)
        return;
    start_posn = get_bits_count(&s->gb) - 32;
    skip_bits(&s->gb, 8); // user data
    ss_index = get_bits(&s->gb, 2);
    blownup = get_bits1(&s->gb);
    hdrsize = get_bits(&s->gb,  8 + 4 * blownup) + 1; // header_size
    skip_bits(&s->gb, 16 + 4 * blownup); // hd_size
    s->static_fields = get_bits1(&s->gb);
    if (s->static_fields) {
        skip_bits(&s->gb, 2); // reference clock code
        skip_bits(&s->gb, 3); // frame duration code
        if (get_bits1(&s->gb))
            skip_bits_long(&s->gb, 36); // timestamp
        /* a single stream can contain multiple audio assets that can be
         * combined to form multiple audio presentations */
        num_audiop = get_bits(&s->gb, 3) + 1;
        if (num_audiop > 1) {
            avpriv_request_sample(s->avctx,
                                  "Multiple DTS-HD audio presentations");
            /* ignore such streams for now */
            return;
        }
        num_assets = get_bits(&s->gb, 3) + 1;
        if (num_assets > 1) {
            avpriv_request_sample(s->avctx, "Multiple DTS-HD audio assets");
            /* ignore such streams for now */
            return;
        }
        for (i = 0; i < num_audiop; i++)
            active_ss_mask[i] = get_bits(&s->gb, ss_index + 1);
        for (i = 0; i < num_audiop; i++)
            for (j = 0; j <= ss_index; j++)
                if (active_ss_mask[i] & (1 << j))
                    skip_bits(&s->gb, 8); // active asset mask
        s->mix_metadata = get_bits1(&s->gb);
        if (s->mix_metadata) {
            int mix_out_mask_size;
            skip_bits(&s->gb, 2); // adjustment level
            mix_out_mask_size  = (get_bits(&s->gb, 2) + 1) << 2;
            s->num_mix_configs =  get_bits(&s->gb, 2) + 1;
            for (i = 0; i < s->num_mix_configs; i++) {
                int mix_out_mask        = get_bits(&s->gb, mix_out_mask_size);
                s->mix_config_num_ch[i] = dca_exss_mask2count(mix_out_mask);
            }
        }
    }
    for (i = 0; i < num_assets; i++)
        asset_size[i] = get_bits_long(&s->gb, 16 + 4 * blownup);
    for (i = 0; i < num_assets; i++) {
        if (dca_exss_parse_asset_header(s))
            return;
    }
    /* not parsed further, we were only interested in the extensions mask
     * from the asset header */
        j = get_bits_count(&s->gb);
        if (start_posn + hdrsize * 8 > j)
            skip_bits_long(&s->gb, start_posn + hdrsize * 8 - j);
        for (i = 0; i < num_assets; i++) {
            start_posn = get_bits_count(&s->gb);
            mkr        = get_bits_long(&s->gb, 32);
            /* parse extensions that we know about */
            if (mkr == 0x655e315e) {
                dca_xbr_parse_frame(s);
            } else if (mkr == 0x47004a03) {
                dca_xxch_decode_frame(s);
                s->core_ext_mask |= DCA_EXT_XXCH; /* xxx use for chan reordering */
            } else {
                av_log(s->avctx, AV_LOG_DEBUG,
                       "DTS-ExSS: unknown marker = 0x%08x\n", mkr);
            }
            /* skip to end of block */
            j = get_bits_count(&s->gb);
            if (start_posn + asset_size[i] * 8 > j)
                skip_bits_long(&s->gb, start_posn + asset_size[i] * 8 - j);
        }
}