void superh_cpu_do_interrupt(CPUState *cs)

{

    SuperHCPU *cpu = SUPERH_CPU(cs);

    CPUSH4State *env = &cpu->env;

    int do_irq = cs->interrupt_request & CPU_INTERRUPT_HARD;

    int do_exp, irq_vector = cs->exception_index;



    /* prioritize exceptions over interrupts */



    do_exp = cs->exception_index != -1;

    do_irq = do_irq && (cs->exception_index == -1);



    if (env->sr & (1u << SR_BL)) {

        if (do_exp && cs->exception_index != 0x1e0) {

            cs->exception_index = 0x000; /* masked exception -> reset */

        }

        if (do_irq && !env->in_sleep) {

            return; /* masked */

        }

    }

    env->in_sleep = 0;



    if (do_irq) {

        irq_vector = sh_intc_get_pending_vector(env->intc_handle,

						(env->sr >> 4) & 0xf);

        if (irq_vector == -1) {

            return; /* masked */

	}

    }



    if (qemu_loglevel_mask(CPU_LOG_INT)) {

	const char *expname;

        switch (cs->exception_index) {

	case 0x0e0:

	    expname = "addr_error";

	    break;

	case 0x040:

	    expname = "tlb_miss";

	    break;

	case 0x0a0:

	    expname = "tlb_violation";

	    break;

	case 0x180:

	    expname = "illegal_instruction";

	    break;

	case 0x1a0:

	    expname = "slot_illegal_instruction";

	    break;

	case 0x800:

	    expname = "fpu_disable";

	    break;

	case 0x820:

	    expname = "slot_fpu";

	    break;

	case 0x100:

	    expname = "data_write";

	    break;

	case 0x060:

	    expname = "dtlb_miss_write";

	    break;

	case 0x0c0:

	    expname = "dtlb_violation_write";

	    break;

	case 0x120:

	    expname = "fpu_exception";

	    break;

	case 0x080:

	    expname = "initial_page_write";

	    break;

	case 0x160:

	    expname = "trapa";

	    break;

	default:

            expname = do_irq ? "interrupt" : "???";

            break;

	}

	qemu_log("exception 0x%03x [%s] raised\n",

		  irq_vector, expname);

        log_cpu_state(cs, 0);

    }



    env->ssr = cpu_read_sr(env);

    env->spc = env->pc;

    env->sgr = env->gregs[15];

    env->sr |= (1u << SR_BL) | (1u << SR_MD) | (1u << SR_RB);



    if (env->flags & (DELAY_SLOT | DELAY_SLOT_CONDITIONAL)) {

        /* Branch instruction should be executed again before delay slot. */

	env->spc -= 2;

	/* Clear flags for exception/interrupt routine. */

        env->flags &= ~(DELAY_SLOT | DELAY_SLOT_CONDITIONAL);

    }



    if (do_exp) {

        env->expevt = cs->exception_index;

        switch (cs->exception_index) {

        case 0x000:

        case 0x020:

        case 0x140:

            env->sr &= ~(1u << SR_FD);

            env->sr |= 0xf << 4; /* IMASK */

            env->pc = 0xa0000000;

            break;

        case 0x040:

        case 0x060:

            env->pc = env->vbr + 0x400;

            break;

        case 0x160:

            env->spc += 2; /* special case for TRAPA */

            /* fall through */

        default:

            env->pc = env->vbr + 0x100;

            break;

        }

        return;

    }



    if (do_irq) {

        env->intevt = irq_vector;

        env->pc = env->vbr + 0x600;

        return;

    }

}
