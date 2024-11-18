static inline int wnv1_get_code(WNV1Context *w, int base_value)

{

    int v = get_vlc2(&w->gb, code_vlc.table, CODE_VLC_BITS, 1);



    if (v == 15)

        return ff_reverse[get_bits(&w->gb, 8 - w->shift)];

    else

        return base_value + ((v - 7) << w->shift);

}
