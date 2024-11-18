int ff_mov_lang_to_iso639(int code, char *to)

{

    int i;

    /* is it the mangled iso code? */

    /* see http://www.geocities.com/xhelmboyx/quicktime/formats/mp4-layout.txt */

    if (code > 138) {

        for (i = 2; i >= 0; i--) {

            to[i] = 0x60 + (code & 0x1f);

            code >>= 5;

        }

        return 1;

    }

    /* old fashion apple lang code */

    if (code >= FF_ARRAY_ELEMS(mov_mdhd_language_map))

        return 0;

    if (!mov_mdhd_language_map[code])

        return 0;

    strncpy(to, mov_mdhd_language_map[code], 4);

    return 1;

}
