IEEE_ARITH2(sqrtt)

IEEE_ARITH2(cvtst)

IEEE_ARITH2(cvtts)



static void gen_cvttq(DisasContext *ctx, int rb, int rc, int fn11)

{

    TCGv vb, vc;



    /* No need to set flushzero, since we have an integer output.  */

    vb = gen_ieee_input(ctx, rb, fn11, 0);

    vc = dest_fpr(ctx, rc);



    /* Almost all integer conversions use cropped rounding, and most

       also do not have integer overflow enabled.  Special case that.  */

    switch (fn11) {

    case QUAL_RM_C:

        gen_helper_cvttq_c(vc, cpu_env, vb);

        break;

    case QUAL_V | QUAL_RM_C:

    case QUAL_S | QUAL_V | QUAL_RM_C:

    case QUAL_S | QUAL_V | QUAL_I | QUAL_RM_C:

        gen_helper_cvttq_svic(vc, cpu_env, vb);

        break;

    default:

        gen_qual_roundmode(ctx, fn11);

        gen_helper_cvttq(vc, cpu_env, vb);

        break;

    }



    gen_fp_exc_raise(rc, fn11);

}
