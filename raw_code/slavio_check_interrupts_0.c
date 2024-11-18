static void slavio_check_interrupts(void *opaque)

{

    CPUState *env;

    SLAVIO_INTCTLState *s = opaque;

    uint32_t pending = s->intregm_pending;

    unsigned int i, j, max = 0;



    pending &= ~s->intregm_disabled;



    if (pending && !(s->intregm_disabled & 0x80000000)) {

	for (i = 0; i < 32; i++) {

	    if (pending & (1 << i)) {

		if (max < s->intbit_to_level[i])

		    max = s->intbit_to_level[i];

	    }

	}

        env = s->cpu_envs[s->target_cpu];

        if (!env) {

	    DPRINTF("No CPU %d, not triggered (pending %x)\n", s->target_cpu, pending);

        }

	else {

            if (env->halted)

                env->halted = 0;

            if (env->interrupt_index == 0) {

                DPRINTF("Triggered CPU %d pil %d\n", s->target_cpu, max);

#ifdef DEBUG_IRQ_COUNT

                s->irq_count[max]++;

#endif

                env->interrupt_index = TT_EXTINT | max;

                cpu_interrupt(env, CPU_INTERRUPT_HARD);

            }

            else

                DPRINTF("Not triggered (pending %x), pending exception %x\n", pending, env->interrupt_index);

	}

    }

    else

	DPRINTF("Not triggered (pending %x), disabled %x\n", pending, s->intregm_disabled);

    

    for (i = 0; i < MAX_CPUS; i++) {

        max = 0;

        env = s->cpu_envs[i];

        if (!env)

            continue;

        for (j = 17; j < 32; j++) {

            if (s->intreg_pending[i] & (1 << j)) {

                if (max < j - 16)

                    max = j - 16;

            }

        }

	if (max > 0) {

            if (env->halted)

                env->halted = 0;

            if (env->interrupt_index == 0) {

                DPRINTF("Triggered softint %d for cpu %d (pending %x)\n", max, i, pending);

#ifdef DEBUG_IRQ_COUNT

                s->irq_count[max]++;

#endif

                env->interrupt_index = TT_EXTINT | max;

                cpu_interrupt(env, CPU_INTERRUPT_HARD);

            }

        }

    }

}
