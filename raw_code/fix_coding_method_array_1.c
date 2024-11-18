static void fix_coding_method_array (int sb, int channels, sb_int8_array coding_method)

{

    int j,k;

    int ch;

    int run, case_val;

    int switchtable[23] = {0,5,1,5,5,5,5,5,2,5,5,5,5,5,5,5,3,5,5,5,5,5,4};



    for (ch = 0; ch < channels; ch++) {

        for (j = 0; j < 64; ) {

            if((coding_method[ch][sb][j] - 8) > 22) {

                run = 1;

                case_val = 8;

            } else {

                switch (switchtable[coding_method[ch][sb][j]]) {

                    case 0: run = 10; case_val = 10; break;

                    case 1: run = 1; case_val = 16; break;

                    case 2: run = 5; case_val = 24; break;

                    case 3: run = 3; case_val = 30; break;

                    case 4: run = 1; case_val = 30; break;

                    case 5: run = 1; case_val = 8; break;

                    default: run = 1; case_val = 8; break;

                }

            }

            for (k = 0; k < run; k++)

                if (j + k < 128)

                    if (coding_method[ch][sb + (j + k) / 64][(j + k) % 64] > coding_method[ch][sb][j])

                        if (k > 0) {

                           SAMPLES_NEEDED

                            //not debugged, almost never used

                            memset(&coding_method[ch][sb][j + k], case_val, k * sizeof(int8_t));

                            memset(&coding_method[ch][sb][j + k], case_val, 3 * sizeof(int8_t));

                        }

            j += run;

        }

    }

}
