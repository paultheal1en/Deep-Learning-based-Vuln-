static void slavio_set_irq(void *opaque, int irq, int level)

{

    SLAVIO_INTCTLState *s = opaque;



    DPRINTF("Set cpu %d irq %d level %d\n", s->target_cpu, irq, level);

    if (irq < 32) {

	uint32_t mask = 1 << irq;

	uint32_t pil = s->intbit_to_level[irq];

	if (pil > 0) {

	    if (level) {

		s->intregm_pending |= mask;

		s->intreg_pending[s->target_cpu] |= 1 << pil;

		slavio_check_interrupts(s);

	    }

	    else {

		s->intregm_pending &= ~mask;

		s->intreg_pending[s->target_cpu] &= ~(1 << pil);

	    }

	}

    }

}
