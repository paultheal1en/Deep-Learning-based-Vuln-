static int gif_video_probe(AVProbeData * pd)

{

    const uint8_t *p, *p_end;

    int bits_per_pixel, has_global_palette, ext_code, ext_len;

    int gce_flags, gce_disposal;



    if (pd->buf_size < 24 ||

        memcmp(pd->buf, gif89a_sig, 6) != 0)

        return 0;

    p_end = pd->buf + pd->buf_size;

    p = pd->buf + 6;

    bits_per_pixel = (p[4] & 0x07) + 1;

    has_global_palette = (p[4] & 0x80);

    p += 7;

    if (has_global_palette)

        p += (1 << bits_per_pixel) * 3;

    for(;;) {

        if (p >= p_end)

            return 0;

        if (*p != '!')

            break;

        p++;

        if (p >= p_end)

            return 0;

        ext_code = *p++;

        if (p >= p_end)

            return 0;

        ext_len = *p++;

        if (ext_code == 0xf9) {

            if (p >= p_end)

                return 0;

            /* if GCE extension found with gce_disposal != 0: it is

               likely to be an animation */

            gce_flags = *p++;

            gce_disposal = (gce_flags >> 2) & 0x7;

            if (gce_disposal != 0)

                return AVPROBE_SCORE_MAX;

            else

                return 0;

        }

        for(;;) {

            if (ext_len == 0)

                break;

            p += ext_len;

            if (p >= p_end)

                return 0;

            ext_len = *p++;

        }

    }

    return 0;

}
