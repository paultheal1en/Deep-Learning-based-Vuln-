int init_vlc(VLC *vlc, int nb_bits, int nb_codes,

             const void *bits, int bits_wrap, int bits_size,

             const void *codes, int codes_wrap, int codes_size)

{

    vlc->bits = nb_bits;

    vlc->table = NULL;

    vlc->table_allocated = 0;

    vlc->table_size = 0;

#ifdef DEBUG_VLC

    printf("build table nb_codes=%d\n", nb_codes);

#endif



    if (build_table(vlc, nb_bits, nb_codes,

                    bits, bits_wrap, bits_size,

                    codes, codes_wrap, codes_size,

                    0, 0) < 0) {

        av_free(vlc->table);

        return -1;

    }

    return 0;

}
