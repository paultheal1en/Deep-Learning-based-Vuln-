static int unpack_modes(Vp3DecodeContext *s, GetBitContext *gb)

{

    int i, j, k, sb_x, sb_y;

    int scheme;

    int current_macroblock;

    int current_fragment;

    int coding_mode;

    int custom_mode_alphabet[CODING_MODE_COUNT];



    if (s->keyframe) {

        for (i = 0; i < s->fragment_count; i++)

            s->all_fragments[i].coding_method = MODE_INTRA;



    } else {



        /* fetch the mode coding scheme for this frame */

        scheme = get_bits(gb, 3);



        /* is it a custom coding scheme? */

        if (scheme == 0) {

            for (i = 0; i < 8; i++)

                custom_mode_alphabet[i] = MODE_INTER_NO_MV;

            for (i = 0; i < 8; i++)

                custom_mode_alphabet[get_bits(gb, 3)] = i;

        }



        /* iterate through all of the macroblocks that contain 1 or more

         * coded fragments */

        for (sb_y = 0; sb_y < s->y_superblock_height; sb_y++) {

            for (sb_x = 0; sb_x < s->y_superblock_width; sb_x++) {



            for (j = 0; j < 4; j++) {

                int mb_x = 2*sb_x +   (j>>1);

                int mb_y = 2*sb_y + (((j>>1)+j)&1);

                int frags_coded = 0;

                current_macroblock = mb_y * s->macroblock_width + mb_x;



                if (mb_x >= s->macroblock_width || mb_y >= s->macroblock_height)

                    continue;



#define BLOCK_X (2*mb_x + (k&1))

#define BLOCK_Y (2*mb_y + (k>>1))

                /* coding modes are only stored if the macroblock has at least one

                 * luma block coded, otherwise it must be INTER_NO_MV */

                for (k = 0; k < 4; k++) {

                    current_fragment = BLOCK_Y*s->fragment_width + BLOCK_X;

                    if (s->all_fragments[current_fragment].coding_method != MODE_COPY)

                        break;

                }

                if (k == 4) {

                    s->macroblock_coding[current_macroblock] = MODE_INTER_NO_MV;

                    continue;

                }



                /* mode 7 means get 3 bits for each coding mode */

                if (scheme == 7)

                    coding_mode = get_bits(gb, 3);

                else if(scheme == 0)

                    coding_mode = custom_mode_alphabet

                        [get_vlc2(gb, s->mode_code_vlc.table, 3, 3)];

                else

                    coding_mode = ModeAlphabet[scheme-1]

                        [get_vlc2(gb, s->mode_code_vlc.table, 3, 3)];



                s->macroblock_coding[current_macroblock] = coding_mode;

                for (k = 0; k < 4; k++) {

                    current_fragment =

                        BLOCK_Y*s->fragment_width + BLOCK_X;

                    if (s->all_fragments[current_fragment].coding_method !=

                        MODE_COPY)

                        s->all_fragments[current_fragment].coding_method =

                            coding_mode;

                }

                for (k = 0; k < 2; k++) {

                    current_fragment = s->fragment_start[k+1] +

                        mb_y*(s->fragment_width>>1) + mb_x;

                    if (s->all_fragments[current_fragment].coding_method !=

                        MODE_COPY)

                        s->all_fragments[current_fragment].coding_method =

                            coding_mode;

                }

            }

            }

        }

    }



    return 0;

}
