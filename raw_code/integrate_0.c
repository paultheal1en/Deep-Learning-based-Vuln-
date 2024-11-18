static int integrate(hdcd_state_t *state, int *flag, const int32_t *samples, int count, int stride)

{

    uint32_t bits = 0;

    int result = FFMIN(state->readahead, count);

    int i;

    *flag = 0;



    for (i = result - 1; i >= 0; i--) {

        bits |= (*samples & 1) << i; /* might be better as a conditional? */

        samples += stride;

    }



    state->window = (state->window << result) | bits;

    state->readahead -= result;

    if (state->readahead > 0)

        return result;



    bits = (state->window ^ state->window >> 5 ^ state->window >> 23);



    if (state->arg) {

        if ((bits & 0xffffffc8) == 0x0fa00500) {

            state->control = (bits & 255) + (bits & 7);

            *flag = 1;

            state->code_counterA++;

        }

        if (((bits ^ (~bits >> 8 & 255)) & 0xffff00ff) == 0xa0060000) {

            state->control = bits >> 8 & 255;

            *flag = 1;

            state->code_counterB++;

        }

        state->arg = 0;

    }

    if (bits == 0x7e0fa005 || bits == 0x7e0fa006) {

        state->readahead = (bits & 3) * 8;

        state->arg = 1;

        state->code_counterC++;

    } else {

        if (bits)

            state->readahead = readaheadtab[bits & ~(-1 << 8)];

        else

            state->readahead = 31; /* ffwd over digisilence */

    }

    return result;

}
