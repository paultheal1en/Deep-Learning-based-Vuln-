void helper_ljmp_protected_T0_T1(void)

{

    int new_cs, new_eip;

    uint32_t e1, e2, cpl, dpl, rpl, limit;



    new_cs = T0;

    new_eip = T1;

    if ((new_cs & 0xfffc) == 0)

        raise_exception_err(EXCP0D_GPF, 0);

    if (load_segment(&e1, &e2, new_cs) != 0)

        raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

    cpl = env->hflags & HF_CPL_MASK;

    if (e2 & DESC_S_MASK) {

        if (!(e2 & DESC_CS_MASK))

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        dpl = (e2 >> DESC_DPL_SHIFT) & 3;

        if (e2 & DESC_CS_MASK) {

            /* conforming code segment */

            if (dpl > cpl)

                raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        } else {

            /* non conforming code segment */

            rpl = new_cs & 3;

            if (rpl > cpl)

                raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

            if (dpl != cpl)

                raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        }

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, new_cs & 0xfffc);

        limit = get_seg_limit(e1, e2);

        if (new_eip > limit)

            raise_exception_err(EXCP0D_GPF, new_cs & 0xfffc);

        cpu_x86_load_seg_cache(env, R_CS, (new_cs & 0xfffc) | cpl,

                       get_seg_base(e1, e2), limit, e2);

        EIP = new_eip;

    } else {

        cpu_abort(env, "jmp to call/task gate not supported 0x%04x:0x%08x", 

                  new_cs, new_eip);

    }

}
