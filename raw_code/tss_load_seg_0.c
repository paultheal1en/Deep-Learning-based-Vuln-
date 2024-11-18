static void tss_load_seg(CPUX86State *env, int seg_reg, int selector)

{

    uint32_t e1, e2;

    int rpl, dpl, cpl;



    if ((selector & 0xfffc) != 0) {

        if (load_segment(env, &e1, &e2, selector) != 0) {

            raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

        }

        if (!(e2 & DESC_S_MASK)) {

            raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

        }

        rpl = selector & 3;

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        cpl = env->hflags & HF_CPL_MASK;

        if (seg_reg == R_CS) {

            if (!(e2 & DESC_CS_MASK)) {

                raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

            }

            /* XXX: is it correct? */

            if (dpl != rpl) {

                raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

            }

            if ((e2 & DESC_C_MASK) && dpl > rpl) {

                raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

            }

        } else if (seg_reg == R_SS) {

            /* SS must be writable data */

            if ((e2 & DESC_CS_MASK) || !(e2 & DESC_W_MASK)) {

                raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

            }

            if (dpl != cpl || dpl != rpl) {

                raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

            }

        } else {

            /* not readable code */

            if ((e2 & DESC_CS_MASK) && !(e2 & DESC_R_MASK)) {

                raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

            }

            /* if data or non conforming code, checks the rights */

            if (((e2 >> DESC_TYPE_SHIFT) & 0xf) < 12) {

                if (dpl < cpl || dpl < rpl) {

                    raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

                }

            }

        }

        if (!(e2 & DESC_P_MASK)) {

            raise_exception_err(env, EXCP0B_NOSEG, selector & 0xfffc);

        }

        cpu_x86_load_seg_cache(env, seg_reg, selector,

                               get_seg_base(e1, e2),

                               get_seg_limit(e1, e2),

                               e2);

    } else {

        if (seg_reg == R_SS || seg_reg == R_CS) {

            raise_exception_err(env, EXCP0A_TSS, selector & 0xfffc);

        }

    }

}
