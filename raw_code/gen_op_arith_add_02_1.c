static inline void gen_op_arith_add(DisasContext *ctx, TCGv ret, TCGv arg1,

                                    TCGv arg2, bool add_ca, bool compute_ca,

                                    bool compute_ov, bool compute_rc0)

{

    TCGv t0 = ret;



    if (((compute_ca && add_ca) || compute_ov)

        && (TCGV_EQUAL(ret, arg1) || TCGV_EQUAL(ret, arg2)))  {

        t0 = tcg_temp_new();

    }



    if (compute_ca) {

        TCGv zero = tcg_const_tl(0);

        if (add_ca) {

            tcg_gen_add2_tl(t0, cpu_ca, arg1, zero, cpu_ca, zero);

            tcg_gen_add2_tl(t0, cpu_ca, t0, cpu_ca, arg2, zero);

        } else {

            tcg_gen_add2_tl(t0, cpu_ca, arg1, zero, arg2, zero);

        }

        tcg_temp_free(zero);

    } else {

        tcg_gen_add_tl(t0, arg1, arg2);

        if (add_ca) {

            tcg_gen_add_tl(t0, t0, cpu_ca);

        }

    }



    if (compute_ov) {

        gen_op_arith_compute_ov(ctx, t0, arg1, arg2, 0);

    }

    if (unlikely(compute_rc0)) {

        gen_set_Rc0(ctx, t0);

    }



    if (!TCGV_EQUAL(t0, ret)) {

        tcg_gen_mov_tl(ret, t0);

        tcg_temp_free(t0);

    }

}
