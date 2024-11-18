void handle_vm86_fault(CPUX86State *env)

{

    TaskState *ts = env->opaque;

    uint8_t *csp, *pc, *ssp;

    unsigned int ip, sp, newflags, newip, newcs, opcode, intno;

    int data32, pref_done;



    csp = (uint8_t *)(env->segs[R_CS] << 4);

    ip = env->eip & 0xffff;

    pc = csp + ip;

    

    ssp = (uint8_t *)(env->segs[R_SS] << 4);

    sp = env->regs[R_ESP] & 0xffff;



#if defined(DEBUG_VM86)

    fprintf(logfile, "VM86 exception %04x:%08x %02x %02x\n",

            env->segs[R_CS], env->eip, pc[0], pc[1]);

#endif



    data32 = 0;

    pref_done = 0;

    do {

        opcode = csp[ip];

        ADD16(ip, 1);

        switch (opcode) {

        case 0x66:      /* 32-bit data */     data32=1; break;

        case 0x67:      /* 32-bit address */  break;

        case 0x2e:      /* CS */              break;

        case 0x3e:      /* DS */              break;

        case 0x26:      /* ES */              break;

        case 0x36:      /* SS */              break;

        case 0x65:      /* GS */              break;

        case 0x64:      /* FS */              break;

        case 0xf2:      /* repnz */	      break;

        case 0xf3:      /* rep */             break;

        default: pref_done = 1;

        }

    } while (!pref_done);



    /* VM86 mode */

    switch(opcode) {

    case 0x9c: /* pushf */

        ADD16(env->eip, 2);

        if (data32) {

            vm_putl(ssp, sp - 4, get_vflags(env));

            ADD16(env->regs[R_ESP], -4);

        } else {

            vm_putw(ssp, sp - 2, get_vflags(env));

            ADD16(env->regs[R_ESP], -2);

        }

        env->eip = ip;

        VM86_FAULT_RETURN;



    case 0x9d: /* popf */

        if (data32) {

            newflags = vm_getl(ssp, sp);

            ADD16(env->regs[R_ESP], 4);

        } else {

            newflags = vm_getw(ssp, sp);

            ADD16(env->regs[R_ESP], 2);

        }

        env->eip = ip;

        CHECK_IF_IN_TRAP();

        if (data32) {

            if (set_vflags_long(newflags, env))

                return;

        } else {

            if (set_vflags_short(newflags, env))

                return;

        }

        VM86_FAULT_RETURN;



    case 0xcd: /* int */

        intno = csp[ip];

        ADD16(ip, 1);

        env->eip = ip;

        if (ts->vm86plus.vm86plus.flags & TARGET_vm86dbg_active) {

            if ( (ts->vm86plus.vm86plus.vm86dbg_intxxtab[intno >> 3] >> 

                  (intno &7)) & 1) {

                return_to_32bit(env, TARGET_VM86_INTx + (intno << 8));

                return;

            }

        }

        do_int(env, intno);

        break;



    case 0xcf: /* iret */

        if (data32) {

            newip = vm_getl(ssp, sp) & 0xffff;

            newcs = vm_getl(ssp, sp + 4) & 0xffff;

            newflags = vm_getl(ssp, sp + 8);

            ADD16(env->regs[R_ESP], 12);

        } else {

            newip = vm_getw(ssp, sp);

            newcs = vm_getw(ssp, sp + 2);

            newflags = vm_getw(ssp, sp + 4);

            ADD16(env->regs[R_ESP], 6);

        }

        env->eip = newip;

        cpu_x86_load_seg(env, R_CS, newcs);

        CHECK_IF_IN_TRAP();

        if (data32) {

            if (set_vflags_long(newflags, env))

                return;

        } else {

            if (set_vflags_short(newflags, env))

                return;

        }

        VM86_FAULT_RETURN;

        

    case 0xfa: /* cli */

        env->eip = ip;

        clear_IF(env);

        VM86_FAULT_RETURN;

        

    case 0xfb: /* sti */

        env->eip = ip;

        if (set_IF(env))

            return;

        VM86_FAULT_RETURN;



    default:

        /* real VM86 GPF exception */

        return_to_32bit(env, TARGET_VM86_UNKNOWN);

        break;

    }

}
