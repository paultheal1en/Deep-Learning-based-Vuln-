static void get_downmix_coeffs(AC3DecodeContext *ctx)

{

    int from = ctx->bsi.acmod;

    int to = ctx->output;

    float clev = clevs[ctx->bsi.cmixlev];

    float slev = slevs[ctx->bsi.surmixlev];

    ac3_audio_block *ab = &ctx->audio_block;



    if (to == AC3_OUTPUT_UNMODIFIED)

        return 0;



    switch (from) {

        case AC3_INPUT_DUALMONO:

            switch (to) {

                case AC3_OUTPUT_MONO:

                case AC3_OUTPUT_STEREO: /* We Assume that sum of both mono channels is requested */

                    ab->chcoeffs[0] *= LEVEL_MINUS_6DB;

                    ab->chcoeffs[1] *= LEVEL_MINUS_6DB;

                    break;

            }

            break;

        case AC3_INPUT_MONO:

            switch (to) {

                case AC3_OUTPUT_STEREO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    break;

            }

            break;

        case AC3_INPUT_STEREO:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[1] *= LEVEL_MINUS_3DB;

                    break;

            }

            break;

        case AC3_INPUT_3F:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[2] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[1] *= clev * LEVEL_PLUS_3DB;

                    break;

                case AC3_OUTPUT_STEREO:

                    ab->chcoeffs[1] *= clev;

                    break;

            }

            break;

        case AC3_INPUT_2F_1R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[1] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[2] *= slev * LEVEL_MINUS_3DB;

                    break;

                case AC3_OUTPUT_STEREO:

                    ab->chcoeffs[2] *= slev * LEVEL_MINUS_3DB;

                    break;

                case AC3_OUTPUT_DOLBY:

                    ab->chcoeffs[2] *= LEVEL_MINUS_3DB;

                    break;

            }

            break;

        case AC3_INPUT_3F_1R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[2] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[1] *= clev * LEVEL_PLUS_3DB;

                    ab->chcoeffs[3] *= slev * LEVEL_MINUS_3DB;

                    break;

                case AC3_OUTPUT_STEREO:

                    ab->chcoeffs[1] *= clev;

                    ab->chcoeffs[3] *= slev * LEVEL_MINUS_3DB;

                    break;

                case AC3_OUTPUT_DOLBY:

                    ab->chcoeffs[1] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[3] *= LEVEL_MINUS_3DB;

                    break;

            }

            break;

        case AC3_INPUT_2F_2R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[1] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[2] *= slev * LEVEL_MINUS_3DB;

                    ab->chcoeffs[3] *= slev * LEVEL_MINUS_3DB;

                    break;

                case AC3_OUTPUT_STEREO:

                    ab->chcoeffs[2] *= slev;

                    ab->chcoeffs[3] *= slev;

                    break;

                case AC3_OUTPUT_DOLBY:

                    ab->chcoeffs[2] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[3] *= LEVEL_MINUS_3DB;

                    break;

            }

            break;

        case AC3_INPUT_3F_2R:

            switch (to) {

                case AC3_OUTPUT_MONO:

                    ab->chcoeffs[0] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[2] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[1] *= clev * LEVEL_PLUS_3DB;

                    ab->chcoeffs[3] *= slev * LEVEL_MINUS_3DB;

                    ab->chcoeffs[4] *= slev * LEVEL_MINUS_3DB;

                    break;

                case AC3_OUTPUT_STEREO:

                    ab->chcoeffs[1] *= clev;

                    ab->chcoeffs[3] *= slev;

                    ab->chcoeffs[4] *= slev;

                    break;

                case AC3_OUTPUT_DOLBY:

                    ab->chcoeffs[1] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[3] *= LEVEL_MINUS_3DB;

                    ab->chcoeffs[4] *= LEVEL_MINUS_3DB;

                    break;

            }

            break;

    }

}
