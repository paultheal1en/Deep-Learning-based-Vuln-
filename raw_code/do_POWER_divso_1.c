void do_POWER_divso (void)

{

    if (((int32_t)T0 == INT32_MIN && (int32_t)T1 == -1) || (int32_t)T1 == 0) {

        T0 = (long)((-1) * (T0 >> 31));

        env->spr[SPR_MQ] = 0;

        xer_ov = 1;

        xer_so = 1;

    } else {

        T0 = (int32_t)T0 / (int32_t)T1;

        env->spr[SPR_MQ] = (int32_t)T0 % (int32_t)T1;

        xer_ov = 0;

    }

}
