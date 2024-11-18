static void gen_lea_v_seg(DisasContext *s, TCGMemOp aflag, TCGv a0,

                          int def_seg, int ovr_seg)

{

    switch (aflag) {

#ifdef TARGET_X86_64

    case MO_64:

        if (ovr_seg < 0) {

            tcg_gen_mov_tl(cpu_A0, a0);

            return;

        }

        break;

#endif

    case MO_32:

        /* 32 bit address */

        if (ovr_seg < 0) {

            if (s->addseg) {

                ovr_seg = def_seg;

            } else {

                tcg_gen_ext32u_tl(cpu_A0, a0);

                return;

            }

        }

        break;

    case MO_16:

        /* 16 bit address */

        if (ovr_seg < 0) {

            ovr_seg = def_seg;

        }

        tcg_gen_ext16u_tl(cpu_A0, a0);

        /* ADDSEG will only be false in 16-bit mode for LEA.  */

        if (!s->addseg) {

            return;

        }

        a0 = cpu_A0;

        break;

    default:

        tcg_abort();

    }



    if (ovr_seg >= 0) {

        TCGv seg = tcg_temp_new();



        tcg_gen_ld_tl(seg, cpu_env, offsetof(CPUX86State, segs[ovr_seg].base));



        if (aflag == MO_64) {

            tcg_gen_add_tl(cpu_A0, a0, seg);

        } else if (CODE64(s)) {

            tcg_gen_ext32u_tl(cpu_A0, a0);

            tcg_gen_add_tl(cpu_A0, cpu_A0, seg);

        } else {

            tcg_gen_add_tl(cpu_A0, a0, seg);

            tcg_gen_ext32u_tl(cpu_A0, cpu_A0);

        }



        tcg_temp_free(seg);

    }

}
