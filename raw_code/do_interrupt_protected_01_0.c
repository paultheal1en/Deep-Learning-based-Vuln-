static void do_interrupt_protected(int intno, int is_int, int error_code,

                                   unsigned int next_eip, int is_hw)

{

    SegmentCache *dt;

    uint8_t *ptr, *ssp;

    int type, dpl, selector, ss_dpl, cpl, sp_mask;

    int has_error_code, new_stack, shift;

    uint32_t e1, e2, offset, ss, esp, ss_e1, ss_e2;

    uint32_t old_eip;



    has_error_code = 0;

    if (!is_int && !is_hw) {

        switch(intno) {

        case 8:

        case 10:

        case 11:

        case 12:

        case 13:

        case 14:

        case 17:

            has_error_code = 1;

            break;

        }

    }



    dt = &env->idt;

    if (intno * 8 + 7 > dt->limit)

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

    ptr = dt->base + intno * 8;

    e1 = ldl_kernel(ptr);

    e2 = ldl_kernel(ptr + 4);

    /* check gate type */

    type = (e2 >> DESC_TYPE_SHIFT) & 0x1f;

    switch(type) {

    case 5: /* task gate */

        /* must do that check here to return the correct error code */

        if (!(e2 & DESC_P_MASK))

            raise_exception_err(EXCP0B_NOSEG, intno * 8 + 2);

        switch_tss(intno * 8, e1, e2, SWITCH_TSS_CALL);

        if (has_error_code) {

            int mask;

            /* push the error code */

            shift = (env->segs[R_CS].flags >> DESC_B_SHIFT) & 1;

            if (env->segs[R_SS].flags & DESC_B_MASK)

                mask = 0xffffffff;

            else

                mask = 0xffff;

            esp = (env->regs[R_ESP] - (2 << shift)) & mask;

            ssp = env->segs[R_SS].base + esp;

            if (shift)

                stl_kernel(ssp, error_code);

            else

                stw_kernel(ssp, error_code);

            env->regs[R_ESP] = (esp & mask) | (env->regs[R_ESP] & ~mask);

        }

        return;

    case 6: /* 286 interrupt gate */

    case 7: /* 286 trap gate */

    case 14: /* 386 interrupt gate */

    case 15: /* 386 trap gate */

        break;

    default:

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

        break;

    }

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    cpl = env->hflags & HF_CPL_MASK;

    /* check privledge if software int */

    if (is_int && dpl < cpl)

        raise_exception_err(EXCP0D_GPF, intno * 8 + 2);

    /* check valid bit */

    if (!(e2 & DESC_P_MASK))

        raise_exception_err(EXCP0B_NOSEG, intno * 8 + 2);

    selector = e1 >> 16;

    offset = (e2 & 0xffff0000) | (e1 & 0x0000ffff);

    if ((selector & 0xfffc) == 0)

        raise_exception_err(EXCP0D_GPF, 0);



    if (load_segment(&e1, &e2, selector) != 0)

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

    if (!(e2 & DESC_S_MASK) || !(e2 & (DESC_CS_MASK)))

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

    dpl = (e2 >> DESC_DPL_SHIFT) & 3;

    if (dpl > cpl)

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

    if (!(e2 & DESC_P_MASK))

        raise_exception_err(EXCP0B_NOSEG, selector & 0xfffc);

    if (!(e2 & DESC_C_MASK) && dpl < cpl) {

        /* to inner priviledge */

        get_ss_esp_from_tss(&ss, &esp, dpl);

        if ((ss & 0xfffc) == 0)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if ((ss & 3) != dpl)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if (load_segment(&ss_e1, &ss_e2, ss) != 0)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        ss_dpl = (ss_e2 >> DESC_DPL_SHIFT) & 3;

        if (ss_dpl != dpl)

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if (!(ss_e2 & DESC_S_MASK) ||

            (ss_e2 & DESC_CS_MASK) ||

            !(ss_e2 & DESC_W_MASK))

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        if (!(ss_e2 & DESC_P_MASK))

            raise_exception_err(EXCP0A_TSS, ss & 0xfffc);

        new_stack = 1;

        sp_mask = get_sp_mask(ss_e2);

        ssp = get_seg_base(ss_e1, ss_e2);

    } else if ((e2 & DESC_C_MASK) || dpl == cpl) {

        /* to same priviledge */

        new_stack = 0;

        sp_mask = get_sp_mask(env->segs[R_SS].flags);

        ssp = env->segs[R_SS].base;

        esp = ESP;

    } else {

        raise_exception_err(EXCP0D_GPF, selector & 0xfffc);

        new_stack = 0; /* avoid warning */

        sp_mask = 0; /* avoid warning */

        ssp = NULL; /* avoid warning */

        esp = 0; /* avoid warning */

    }



    shift = type >> 3;



#if 0

    /* XXX: check that enough room is available */

    push_size = 6 + (new_stack << 2) + (has_error_code << 1);

    if (env->eflags & VM_MASK)

        push_size += 8;

    push_size <<= shift;

#endif

    if (is_int)

        old_eip = next_eip;

    else

        old_eip = env->eip;

    if (shift == 1) {

        if (env->eflags & VM_MASK) {

            PUSHL(ssp, esp, sp_mask, env->segs[R_GS].selector);

            PUSHL(ssp, esp, sp_mask, env->segs[R_FS].selector);

            PUSHL(ssp, esp, sp_mask, env->segs[R_DS].selector);

            PUSHL(ssp, esp, sp_mask, env->segs[R_ES].selector);

        }

        if (new_stack) {

            PUSHL(ssp, esp, sp_mask, env->segs[R_SS].selector);

            PUSHL(ssp, esp, sp_mask, ESP);

        }

        PUSHL(ssp, esp, sp_mask, compute_eflags());

        PUSHL(ssp, esp, sp_mask, env->segs[R_CS].selector);

        PUSHL(ssp, esp, sp_mask, old_eip);

        if (has_error_code) {

            PUSHL(ssp, esp, sp_mask, error_code);

        }

    } else {

        if (new_stack) {

            PUSHW(ssp, esp, sp_mask, env->segs[R_SS].selector);

            PUSHW(ssp, esp, sp_mask, ESP);

        }

        PUSHW(ssp, esp, sp_mask, compute_eflags());

        PUSHW(ssp, esp, sp_mask, env->segs[R_CS].selector);

        PUSHW(ssp, esp, sp_mask, old_eip);

        if (has_error_code) {

            PUSHW(ssp, esp, sp_mask, error_code);

        }

    }

    

    if (new_stack) {

        ss = (ss & ~3) | dpl;

        cpu_x86_load_seg_cache(env, R_SS, ss, 

                               ssp, get_seg_limit(ss_e1, ss_e2), ss_e2);

    }

    ESP = (ESP & ~sp_mask) | (esp & sp_mask);



    selector = (selector & ~3) | dpl;

    cpu_x86_load_seg_cache(env, R_CS, selector, 

                   get_seg_base(e1, e2),

                   get_seg_limit(e1, e2),

                   e2);

    cpu_x86_set_cpl(env, dpl);

    env->eip = offset;



    /* interrupt gate clear IF mask */

    if ((type & 1) == 0) {

        env->eflags &= ~IF_MASK;

    }

    env->eflags &= ~(TF_MASK | VM_MASK | RF_MASK | NT_MASK);

}
