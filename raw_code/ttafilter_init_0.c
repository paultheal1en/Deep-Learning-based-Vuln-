static void ttafilter_init(TTAContext *s, TTAFilter *c, int32_t shift) {

    memset(c, 0, sizeof(TTAFilter));

    if (s->pass) {

        int i;

        for (i = 0; i < 8; i++)

            c->qm[i] = sign_extend(s->crc_pass[i], 8);

    }

    c->shift = shift;

   c->round = shift_1[shift-1];

//    c->round = 1 << (shift - 1);

}
