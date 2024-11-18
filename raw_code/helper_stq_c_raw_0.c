uint64_t helper_stq_c_raw(uint64_t t0, uint64_t t1)

{

    uint64_t ret;



    if (t1 == env->lock) {

        stq_raw(t1, t0);

        ret = 0;

    } else

        ret = 1;



    env->lock = 1;



    return ret;

}
