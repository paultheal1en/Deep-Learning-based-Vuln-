void helper_fcmp_gt_FT(CPUSH4State *env, float32 t0, float32 t1)

{

    int relation;



    set_float_exception_flags(0, &env->fp_status);

    relation = float32_compare(t0, t1, &env->fp_status);

    if (unlikely(relation == float_relation_unordered)) {

        update_fpscr(env, GETPC());

    } else {

        env->sr_t = (relation == float_relation_greater);

    }

}
