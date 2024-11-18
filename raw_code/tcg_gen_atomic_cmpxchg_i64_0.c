void tcg_gen_atomic_cmpxchg_i64(TCGv_i64 retv, TCGv addr, TCGv_i64 cmpv,

                                TCGv_i64 newv, TCGArg idx, TCGMemOp memop)

{

    memop = tcg_canonicalize_memop(memop, 1, 0);



    if (!parallel_cpus) {

        TCGv_i64 t1 = tcg_temp_new_i64();

        TCGv_i64 t2 = tcg_temp_new_i64();



        tcg_gen_ext_i64(t2, cmpv, memop & MO_SIZE);



        tcg_gen_qemu_ld_i64(t1, addr, idx, memop & ~MO_SIGN);

        tcg_gen_movcond_i64(TCG_COND_EQ, t2, t1, t2, newv, t1);

        tcg_gen_qemu_st_i64(t2, addr, idx, memop);

        tcg_temp_free_i64(t2);



        if (memop & MO_SIGN) {

            tcg_gen_ext_i64(retv, t1, memop);

        } else {

            tcg_gen_mov_i64(retv, t1);

        }

        tcg_temp_free_i64(t1);

    } else if ((memop & MO_SIZE) == MO_64) {

#ifdef CONFIG_ATOMIC64

        gen_atomic_cx_i64 gen;



        gen = table_cmpxchg[memop & (MO_SIZE | MO_BSWAP)];

        tcg_debug_assert(gen != NULL);



#ifdef CONFIG_SOFTMMU

        {

            TCGv_i32 oi = tcg_const_i32(make_memop_idx(memop, idx));

            gen(retv, tcg_ctx.tcg_env, addr, cmpv, newv, oi);

            tcg_temp_free_i32(oi);

        }

#else

        gen(retv, tcg_ctx.tcg_env, addr, cmpv, newv);

#endif

#else

        gen_helper_exit_atomic(tcg_ctx.tcg_env);

        /* Produce a result, so that we have a well-formed opcode stream

           with respect to uses of the result in the (dead) code following.  */

        tcg_gen_movi_i64(retv, 0);

#endif /* CONFIG_ATOMIC64 */

    } else {

        TCGv_i32 c32 = tcg_temp_new_i32();

        TCGv_i32 n32 = tcg_temp_new_i32();

        TCGv_i32 r32 = tcg_temp_new_i32();



        tcg_gen_extrl_i64_i32(c32, cmpv);

        tcg_gen_extrl_i64_i32(n32, newv);

        tcg_gen_atomic_cmpxchg_i32(r32, addr, c32, n32, idx, memop & ~MO_SIGN);

        tcg_temp_free_i32(c32);

        tcg_temp_free_i32(n32);



        tcg_gen_extu_i32_i64(retv, r32);

        tcg_temp_free_i32(r32);



        if (memop & MO_SIGN) {

            tcg_gen_ext_i64(retv, retv, memop);

        }

    }

}
